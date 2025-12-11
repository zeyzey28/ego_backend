"""
FastAPI Backend for Accessibility Analysis
Serves two panels: User Panel and Municipality Panel
With JWT-based authentication and role-based access control
"""

import base64
import os
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from geo import (
    calculate_walking_duration,
    find_grid_for_point,
    find_nearest_grid,
    load_json,
    save_json,
)
from auth import (
    UserRegister,
    UserLogin,
    StaffLogin,
    StaffCreate,
    StaffRole,
    Token,
    TokenData,
    UserResponse,
    StaffResponse,
    register_user,
    login_user,
    login_staff,
    add_staff,
    get_current_user,
    get_current_staff,
    get_current_yonetici,
    get_users,
    get_staff,
    get_staff_roles,
)

# Initialize FastAPI app
app = FastAPI(
    title="Accessibility Analysis API",
    description="Backend for User Panel and Municipality Panel",
    version="1.0.0"
)

# CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data paths
DATA_DIR = Path(__file__).parent / "data"
GRID_ACCESS_PATH = DATA_DIR / "grid_access_only.geojson"
GRID_NEAREST_STOPS_PATH = DATA_DIR / "grid_nearest_3stops.json"
BUS_STOPS_PATH = DATA_DIR / "bus_stops_list.json"
GRID_SLOPE_PATH = DATA_DIR / "grid_slope_score.json"
COMPLAINTS_PATH = Path(__file__).parent / "complaints.json"
PHOTOS_DIR = Path(__file__).parent / "photos"

# Ensure directories exist
PHOTOS_DIR.mkdir(parents=True, exist_ok=True)

# Mount photos directory for static file serving
app.mount("/photos", StaticFiles(directory=str(PHOTOS_DIR)), name="photos")

# Category to urgency mapping
URGENCY_MAPPING = {
    "boru_patlamasi": "red",
    "boru patlaması": "red",
    "su_baskini": "red",
    "su baskını": "red",
    "yangin": "red",
    "yangın": "red",
    "merdiven_kirik": "yellow",
    "merdiven kırık": "yellow",
    "kaldirim_bozuk": "yellow",
    "kaldırım bozuk": "yellow",
    "rampa_eksik": "yellow",
    "rampa eksik": "yellow",
    "isik_yanmiyor": "green",
    "ışık yanmıyor": "green",
    "cop_toplama": "green",
    "çöp toplama": "green",
    "diger": "green",
    "diğer": "green",
}

# Cache for loaded data
_cache = {}


def get_grid_features():
    """Load and cache grid features."""
    if "grid_features" not in _cache:
        data = load_json(GRID_ACCESS_PATH)
        _cache["grid_features"] = data["features"]
    return _cache["grid_features"]


def get_nearest_stops_data():
    """Load and cache nearest stops data."""
    if "nearest_stops" not in _cache:
        data = load_json(GRID_NEAREST_STOPS_PATH)
        # Convert to dict for faster lookup
        _cache["nearest_stops"] = {item["grid_id"]: item["nearest_stops"] for item in data}
    return _cache["nearest_stops"]


def get_bus_stops():
    """Load and cache bus stops data."""
    if "bus_stops" not in _cache:
        data = load_json(BUS_STOPS_PATH)
        # Convert to dict for faster lookup
        _cache["bus_stops"] = {stop["stop_id"]: stop for stop in data}
    return _cache["bus_stops"]


def get_slope_scores():
    """Load and cache slope scores."""
    if "slope_scores" not in _cache:
        data = load_json(GRID_SLOPE_PATH)
        # Convert to dict for faster lookup
        _cache["slope_scores"] = {item["grid_id"]: item["slope_score"] for item in data}
    return _cache["slope_scores"]


def get_complaints():
    """Load complaints from file."""
    if not COMPLAINTS_PATH.exists():
        return []
    return load_json(COMPLAINTS_PATH)


def save_complaints(complaints: list):
    """Save complaints to file."""
    save_json(COMPLAINTS_PATH, complaints)


def get_urgency(category: str) -> str:
    """Get urgency level based on category."""
    category_lower = category.lower().strip()
    return URGENCY_MAPPING.get(category_lower, "green")


# ============================================
# Response Models
# ============================================

class StopInfo(BaseModel):
    stop_id: int
    stop_name: str
    lat: float
    lon: float
    distance_m: float
    duration_min: float


class NearestStopsResponse(BaseModel):
    grid_id: int
    slope_score: float
    nearest_stops: list[StopInfo]


class ComplaintResponse(BaseModel):
    success: bool
    message: str
    complaint_id: int
    category: str
    urgency: str
    created_at: str


class AuthResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Token] = None


class RegisterResponse(BaseModel):
    success: bool
    message: str
    user_id: int
    username: str


class Complaint(BaseModel):
    id: int
    category: str
    description: str
    lat: float
    lon: float
    urgency: str
    photo: Optional[str] = None
    photo_url: Optional[str] = None  # Fotoğraf URL'i
    created_at: str
    # Geri bildirim alanları
    status: str = "beklemede"  # beklemede, inceleniyor, cozuldu, reddedildi
    feedback: Optional[str] = None  # Belediye geri bildirimi
    feedback_at: Optional[str] = None  # Geri bildirim tarihi
    feedback_by: Optional[str] = None  # Geri bildirim veren personel
    user_id: Optional[int] = None  # Şikayeti oluşturan kullanıcı


class ComplaintCreate(BaseModel):
    """JSON body ile şikayet oluşturma modeli"""
    category: str
    description: str
    lat: float
    lon: float
    photo_base64: Optional[str] = None


class FeedbackCreate(BaseModel):
    """Belediye geri bildirim modeli"""
    status: str  # beklemede, inceleniyor, cozuldu, reddedildi
    feedback: Optional[str] = None  # Opsiyonel - yazılmazsa otomatik mesaj atanır


# Otomatik geri bildirim mesajları
AUTO_FEEDBACK_MESSAGES = {
    "beklemede": "Şikayetiniz alınmıştır. En kısa sürede değerlendirilecektir.",
    "inceleniyor": "Şikayetiniz incelemeye alınmıştır. Ekiplerimiz konuyla ilgilenmektedir.",
    "cozuldu": "Şikayetiniz çözümlenmiştir. İlginiz için teşekkür ederiz.",
    "reddedildi": "Şikayetiniz değerlendirilmiş ancak işlem yapılamamıştır. Detaylı bilgi için belediye ile iletişime geçebilirsiniz."
}


class UserComplaint(BaseModel):
    """Kullanıcının şikayetleri için model"""
    id: int
    category: str
    description: str
    urgency: str
    status: str
    created_at: str
    feedback: Optional[str] = None
    feedback_at: Optional[str] = None


# ============================================
# AUTH ENDPOINTS - KULLANICI (Vatandaş)
# ============================================

@app.post("/auth/user/register")
async def register_user_endpoint(user_data: UserRegister):
    """
    Yeni vatandaş kullanıcı kaydı.
    Kayıt başarılı olursa otomatik olarak giriş yapılır ve token döner.
    
    - **username**: Kullanıcı adı (benzersiz olmalı)
    - **password**: Şifre
    - **email**: E-posta (opsiyonel)
    - **full_name**: Ad Soyad (opsiyonel)
    """
    try:
        new_user = register_user(user_data)
        
        # Otomatik giriş yap
        login_data = UserLogin(username=user_data.username, password=user_data.password)
        token = login_user(login_data)
        
        return {
            "success": True,
            "message": "Kayıt işleminiz başarıyla tamamlandı! Hoş geldiniz.",
            "user_id": new_user["id"],
            "username": new_user["username"],
            "token": token
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Kayıt sırasında bir hata oluştu: {str(e)}"
        )


@app.post("/auth/user/login")
async def login_user_endpoint(login_data: UserLogin):
    """
    Vatandaş kullanıcı girişi.
    
    Başarılı girişte:
    - role: "user" → Kullanıcı Paneline yönlendir
    
    - **username**: Kullanıcı adı
    - **password**: Şifre
    """
    try:
        token = login_user(login_data)
        return {
            "success": True,
            "message": "Giriş başarılı! Hoş geldiniz.",
            "token": token
        }
    except HTTPException as e:
        # Daha anlaşılır hata mesajı
        raise HTTPException(
            status_code=e.status_code,
            detail="Kullanıcı adı veya şifre hatalı. Lütfen bilgilerinizi kontrol edin."
        )


# ============================================
# AUTH ENDPOINTS - BELEDİYE PERSONELİ
# ============================================

@app.post("/auth/staff/login")
async def login_staff_endpoint(login_data: StaffLogin):
    """
    Belediye personeli girişi.
    
    Başarılı girişte:
    - role: "staff" → Belediye Paneline yönlendir
    - staff_role: "yonetici" | "operasyon" | "analiz"
    
    - **username**: Kullanıcı adı
    - **password**: Şifre
    """
    try:
        token = login_staff(login_data)
        return {
            "success": True,
            "message": "Giriş başarılı! Belediye paneline yönlendiriliyorsunuz.",
            "token": token
        }
    except HTTPException as e:
        raise HTTPException(
            status_code=e.status_code,
            detail="Personel kullanıcı adı veya şifre hatalı. Lütfen bilgilerinizi kontrol edin."
        )


@app.get("/auth/me")
async def get_me(current_user: TokenData = Depends(get_current_user)):
    """
    Mevcut kullanıcı bilgilerini döner.
    Token gerektirir.
    """
    if current_user.role == "staff":
        staff_list = get_staff()
        for member in staff_list:
            if member["username"] == current_user.username:
                return {
                    "id": member["id"],
                    "username": member["username"],
                    "full_name": member.get("full_name"),
                    "department": member.get("department"),
                    "staff_role": member.get("staff_role"),
                    "role": "staff"
                }
    else:
        users = get_users()
        for user in users:
            if user["username"] == current_user.username:
                return {
                    "id": user["id"],
                    "username": user["username"],
                    "email": user.get("email"),
                    "full_name": user.get("full_name"),
                    "role": "user"
                }
    
    raise HTTPException(status_code=404, detail="Kullanıcı bulunamadı")


# ============================================
# STAFF MANAGEMENT ENDPOINTS (Belediye Paneli)
# ============================================

@app.get("/staff/roles")
async def get_available_roles():
    """
    Mevcut personel rollerini listele.
    Personel oluştururken kullanılacak roller.
    """
    return get_staff_roles()


@app.post("/staff/add", response_model=StaffResponse)
async def create_staff(
    staff_data: StaffCreate,
    current_user: TokenData = Depends(get_current_yonetici)
):
    """
    Yeni belediye personeli ekle.
    **Sadece yönetici rolündeki personel bu işlemi yapabilir.**
    
    - **username**: Kullanıcı adı (benzersiz olmalı)
    - **password**: Şifre
    - **full_name**: Ad Soyad
    - **department**: Departman (opsiyonel)
    - **staff_role**: Rol (yonetici, operasyon, analiz)
    """
    new_staff = add_staff(staff_data, current_user.username)
    return StaffResponse(
        id=new_staff["id"],
        username=new_staff["username"],
        full_name=new_staff["full_name"],
        department=new_staff.get("department"),
        staff_role=new_staff.get("staff_role", "operasyon"),
        created_at=new_staff["created_at"],
        created_by=new_staff.get("created_by")
    )


@app.get("/staff/list", response_model=list[StaffResponse])
async def list_staff(current_user: TokenData = Depends(get_current_yonetici)):
    """
    Tüm belediye personelini listele.
    **Sadece yönetici rolündeki personel görebilir.**
    """
    staff_list = get_staff()
    return [
        StaffResponse(
            id=s["id"],
            username=s["username"],
            full_name=s["full_name"],
            department=s.get("department"),
            staff_role=s.get("staff_role", "operasyon"),
            created_at=s["created_at"],
            created_by=s.get("created_by")
        )
        for s in staff_list
    ]


# ============================================
# PUBLIC ENDPOINTS
# ============================================

@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "ok", "message": "Accessibility API is running"}


@app.get("/nearest-stops", response_model=NearestStopsResponse)
async def get_nearest_stops(
    lat: float = Query(..., description="Latitude of the point"),
    lon: float = Query(..., description="Longitude of the point")
):
    """
    Get nearest 3 bus stops for a given location.
    
    Returns:
        - grid_id: The grid cell containing the point
        - slope_score: Topographic slope score for accessibility
        - nearest_stops: List of 3 nearest bus stops with details
    """
    # Load data
    grid_features = get_grid_features()
    nearest_stops_data = get_nearest_stops_data()
    bus_stops = get_bus_stops()
    slope_scores = get_slope_scores()
    
    # Find grid for the point
    grid_id = find_grid_for_point(lat, lon, grid_features)
    
    # Fallback to nearest grid if point-in-polygon fails
    if grid_id is None:
        grid_id = find_nearest_grid(lat, lon, grid_features)
    
    if grid_id is None:
        raise HTTPException(
            status_code=404,
            detail="Could not find a grid for the given coordinates"
        )
    
    # Get slope score
    slope_score = slope_scores.get(grid_id, 0)
    
    # Get nearest stops for this grid
    stops_info = nearest_stops_data.get(grid_id, [])
    
    if not stops_info:
        raise HTTPException(
            status_code=404,
            detail=f"No stop data found for grid_id {grid_id}"
        )
    
    # Build response with stop details
    result_stops = []
    for stop in stops_info:
        stop_id = stop["stop_id"]
        distance_m = stop["distance"]
        
        # Get stop details from bus_stops
        stop_details = bus_stops.get(stop_id)
        if stop_details:
            result_stops.append(StopInfo(
                stop_id=stop_id,
                stop_name=stop_details["stop_name"],
                lat=stop_details["lat"],
                lon=stop_details["lon"],
                distance_m=round(distance_m, 2),
                duration_min=round(calculate_walking_duration(distance_m), 2)
            ))
    
    return NearestStopsResponse(
        grid_id=grid_id,
        slope_score=round(slope_score, 2),
        nearest_stops=result_stops
    )


@app.post("/complaints", response_model=ComplaintResponse)
async def create_complaint(
    category: str = Form(..., description="Complaint category"),
    description: str = Form(..., description="Complaint description"),
    lat: float = Form(..., description="Latitude"),
    lon: float = Form(..., description="Longitude"),
    photo: Optional[UploadFile] = File(None, description="Photo file")
):
    """
    Yeni şikayet oluştur.
    
    Kabul edilen parametreler:
        - category: Şikayet kategorisi (örn: "boru_patlamasi", "merdiven_kirik")
        - description: Detaylı açıklama
        - lat, lon: Konum koordinatları
        - photo: Fotoğraf (opsiyonel)
    
    Dönen değerler:
        - success: İşlem başarılı mı
        - message: Kullanıcıya gösterilecek mesaj
        - complaint_id: Şikayet numarası
    """
    try:
    # Load existing complaints
    complaints = get_complaints()
    
    # Generate new ID
    new_id = max([c["id"] for c in complaints], default=0) + 1
    
    # Determine urgency
    urgency = get_urgency(category)
        created_at = datetime.now().isoformat()
    
    # Handle photo
    photo_path = None
    if photo:
        # Save photo to disk
        photo_filename = f"{new_id}_{photo.filename}"
        photo_full_path = PHOTOS_DIR / photo_filename
        
        content = await photo.read()
        with open(photo_full_path, "wb") as f:
            f.write(content)
        
        photo_path = str(photo_filename)
    
    # Create complaint object
    new_complaint = {
        "id": new_id,
        "category": category,
        "description": description,
        "lat": lat,
        "lon": lon,
        "urgency": urgency,
        "photo": photo_path,
            "created_at": created_at
    }
    
    # Append and save
    complaints.append(new_complaint)
    save_complaints(complaints)
    
        # Aciliyet mesajı
        urgency_messages = {
            "red": "Acil durum olarak kaydedildi. En kısa sürede müdahale edilecektir.",
            "yellow": "Orta öncelikli olarak kaydedildi. En kısa sürede değerlendirilecektir.",
            "green": "Normal öncelikli olarak kaydedildi. Sırasıyla değerlendirilecektir."
        }
        
        return ComplaintResponse(
            success=True,
            message=f"Şikayetiniz başarıyla alınmıştır! Şikayet numaranız: #{new_id}. {urgency_messages.get(urgency, '')}",
            complaint_id=new_id,
            category=category,
            urgency=urgency,
            created_at=created_at
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Şikayet oluşturulurken bir hata oluştu: {str(e)}"
        )


@app.post("/complaints/base64", response_model=ComplaintResponse)
async def create_complaint_base64(
    category: str = Form(...),
    description: str = Form(...),
    lat: float = Form(...),
    lon: float = Form(...),
    photo_base64: Optional[str] = Form(None, description="Base64 encoded photo")
):
    """
    Base64 formatında fotoğraf ile şikayet oluştur.
    Mobil uygulamalar için alternatif endpoint.
    """
    try:
    # Load existing complaints
    complaints = get_complaints()
    
    # Generate new ID
    new_id = max([c["id"] for c in complaints], default=0) + 1
    
    # Determine urgency
    urgency = get_urgency(category)
        created_at = datetime.now().isoformat()
    
    # Handle base64 photo
    photo_path = None
    if photo_base64:
        try:
            # Decode and save
            photo_data = base64.b64decode(photo_base64)
            photo_filename = f"{new_id}_photo.jpg"
            photo_full_path = PHOTOS_DIR / photo_filename
            
            with open(photo_full_path, "wb") as f:
                f.write(photo_data)
            
            photo_path = str(photo_filename)
        except Exception as e:
                raise HTTPException(status_code=400, detail=f"Fotoğraf yüklenirken hata oluştu: {str(e)}")
    
    # Create complaint object
    new_complaint = {
        "id": new_id,
        "category": category,
        "description": description,
        "lat": lat,
        "lon": lon,
        "urgency": urgency,
        "photo": photo_path,
            "created_at": created_at
    }
    
    # Append and save
    complaints.append(new_complaint)
    save_complaints(complaints)
    
        # Aciliyet mesajı
        urgency_messages = {
            "red": "Acil durum olarak kaydedildi. En kısa sürede müdahale edilecektir.",
            "yellow": "Orta öncelikli olarak kaydedildi. En kısa sürede değerlendirilecektir.",
            "green": "Normal öncelikli olarak kaydedildi. Sırasıyla değerlendirilecektir."
        }
        
        return ComplaintResponse(
            success=True,
            message=f"Şikayetiniz başarıyla alınmıştır! Şikayet numaranız: #{new_id}. {urgency_messages.get(urgency, '')}",
            complaint_id=new_id,
            category=category,
            urgency=urgency,
            created_at=created_at
        )
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Şikayet oluşturulurken bir hata oluştu: {str(e)}"
        )


@app.post("/complaints/json", response_model=ComplaintResponse)
async def create_complaint_json(complaint_data: ComplaintCreate):
    """
    JSON body ile şikayet oluştur.
    Frontend uygulamaları için önerilen endpoint.
    
    Body:
    {
        "category": "boru_patlamasi",
        "description": "Açıklama...",
        "lat": 39.9208,
        "lon": 32.8541,
        "photo_base64": "..." (opsiyonel)
    }
    """
    try:
        # Load existing complaints
    complaints = get_complaints()
        
        # Generate new ID
        new_id = max([c["id"] for c in complaints], default=0) + 1
        
        # Determine urgency
        urgency = get_urgency(complaint_data.category)
        created_at = datetime.now().isoformat()
        
        # Handle base64 photo
        photo_path = None
        if complaint_data.photo_base64:
            try:
                photo_data = base64.b64decode(complaint_data.photo_base64)
                photo_filename = f"{new_id}_photo.jpg"
                photo_full_path = PHOTOS_DIR / photo_filename
                
                with open(photo_full_path, "wb") as f:
                    f.write(photo_data)
                
                photo_path = str(photo_filename)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Fotoğraf yüklenirken hata oluştu: {str(e)}")
        
        # Create complaint object
        new_complaint = {
            "id": new_id,
            "category": complaint_data.category,
            "description": complaint_data.description,
            "lat": complaint_data.lat,
            "lon": complaint_data.lon,
            "urgency": urgency,
            "photo": photo_path,
            "created_at": created_at,
            "status": "beklemede",
            "feedback": None,
            "feedback_at": None,
            "feedback_by": None,
            "user_id": None  # Anonim şikayet
        }
        
        # Append and save
        complaints.append(new_complaint)
        save_complaints(complaints)
        
        # Aciliyet mesajı
        urgency_messages = {
            "red": "Acil durum olarak kaydedildi. En kısa sürede müdahale edilecektir.",
            "yellow": "Orta öncelikli olarak kaydedildi. En kısa sürede değerlendirilecektir.",
            "green": "Normal öncelikli olarak kaydedildi. Sırasıyla değerlendirilecektir."
        }
        
        return ComplaintResponse(
            success=True,
            message=f"Şikayetiniz başarıyla alınmıştır! Şikayet numaranız: #{new_id}. {urgency_messages.get(urgency, '')}",
            complaint_id=new_id,
            category=complaint_data.category,
            urgency=urgency,
            created_at=created_at
        )
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Şikayet oluşturulurken bir hata oluştu: {str(e)}"
        )


@app.get("/complaints")
async def list_complaints():
    """
    Tüm şikayetleri listele.
    Belediye paneli için şikayet listesi.
    Fotoğraf URL'leri ile birlikte döner.
    """
    complaints = get_complaints()
    
    # Fotoğraf URL'lerini ekle
    for complaint in complaints:
        if complaint.get("photo"):
            complaint["photo_url"] = f"/photos/{complaint['photo']}"
        else:
            complaint["photo_url"] = None
        
        # Eksik alanları varsayılan değerlerle doldur
        if "status" not in complaint:
            complaint["status"] = "beklemede"
        if "feedback" not in complaint:
            complaint["feedback"] = None
        if "feedback_at" not in complaint:
            complaint["feedback_at"] = None
        if "feedback_by" not in complaint:
            complaint["feedback_by"] = None
        if "user_id" not in complaint:
            complaint["user_id"] = None
    
    return complaints


@app.get("/complaints/{complaint_id}")
async def get_complaint(complaint_id: int):
    """Belirli bir şikayeti getir."""
    complaints = get_complaints()
    
    for complaint in complaints:
        if complaint["id"] == complaint_id:
            # Fotoğraf URL'i ekle
            if complaint.get("photo"):
                complaint["photo_url"] = f"/photos/{complaint['photo']}"
            else:
                complaint["photo_url"] = None
            
            # Eksik alanları doldur
            if "status" not in complaint:
                complaint["status"] = "beklemede"
            if "feedback" not in complaint:
                complaint["feedback"] = None
            
            return complaint
    
    raise HTTPException(status_code=404, detail="Şikayet bulunamadı")


# ============================================
# GERİ BİLDİRİM ENDPOINTLERİ (Belediye Paneli)
# ============================================

@app.put("/complaints/{complaint_id}/feedback")
async def add_feedback(
    complaint_id: int,
    feedback_data: FeedbackCreate,
    current_user: TokenData = Depends(get_current_staff)
):
    """
    Şikayete geri bildirim ekle.
    Sadece belediye personeli yapabilir.
    
    Status değerleri:
    - beklemede: Henüz incelenmedi
    - inceleniyor: İnceleme sürecinde
    - cozuldu: Çözüldü
    - reddedildi: Reddedildi
    
    Feedback opsiyoneldir:
    - Yazılmazsa status'a göre otomatik mesaj atanır
    - Yazılırsa özel mesaj kullanılır
    """
    # Status kontrolü
    valid_statuses = ["beklemede", "inceleniyor", "cozuldu", "reddedildi"]
    if feedback_data.status not in valid_statuses:
        raise HTTPException(
            status_code=400,
            detail=f"Geçersiz durum. Geçerli değerler: {', '.join(valid_statuses)}"
        )
    
    complaints = get_complaints()
    
    for complaint in complaints:
        if complaint["id"] == complaint_id:
            complaint["status"] = feedback_data.status
            
            # Feedback yazılmadıysa otomatik mesaj ata
            if feedback_data.feedback:
                complaint["feedback"] = feedback_data.feedback
            else:
                complaint["feedback"] = AUTO_FEEDBACK_MESSAGES.get(
                    feedback_data.status, 
                    "Şikayetiniz güncellendi."
                )
            
            complaint["feedback_at"] = datetime.now().isoformat()
            complaint["feedback_by"] = current_user.username
            
            save_complaints(complaints)
            
            return {
                "success": True,
                "message": "Geri bildirim başarıyla eklendi.",
                "complaint_id": complaint_id,
                "status": feedback_data.status,
                "feedback": complaint["feedback"]
            }
    
    raise HTTPException(status_code=404, detail="Şikayet bulunamadı")


@app.put("/complaints/{complaint_id}/status")
async def update_status_only(
    complaint_id: int,
    status: str = Query(..., description="Yeni durum"),
    current_user: TokenData = Depends(get_current_staff)
):
    """
    Sadece durum güncelle (otomatik mesaj ile).
    Tek tıkla durum değiştirmek için kullanılır.
    
    Örnek: PUT /complaints/1/status?status=cozuldu
    """
    valid_statuses = ["beklemede", "inceleniyor", "cozuldu", "reddedildi"]
    if status not in valid_statuses:
        raise HTTPException(
            status_code=400,
            detail=f"Geçersiz durum. Geçerli değerler: {', '.join(valid_statuses)}"
        )
    
    complaints = get_complaints()
    
    for complaint in complaints:
        if complaint["id"] == complaint_id:
            complaint["status"] = status
            complaint["feedback"] = AUTO_FEEDBACK_MESSAGES.get(status, "Şikayetiniz güncellendi.")
            complaint["feedback_at"] = datetime.now().isoformat()
            complaint["feedback_by"] = current_user.username
            
            save_complaints(complaints)
            
            return {
                "success": True,
                "message": f"Durum '{status}' olarak güncellendi.",
                "complaint_id": complaint_id,
                "status": status,
                "feedback": complaint["feedback"]
            }
    
    raise HTTPException(status_code=404, detail="Şikayet bulunamadı")


@app.get("/complaints/status/{status}")
async def get_complaints_by_status(status: str):
    """
    Duruma göre şikayetleri filtrele.
    
    Status: beklemede, inceleniyor, cozuldu, reddedildi
    """
    valid_statuses = ["beklemede", "inceleniyor", "cozuldu", "reddedildi"]
    if status not in valid_statuses:
        raise HTTPException(
            status_code=400, 
            detail=f"Geçersiz durum. Geçerli değerler: {', '.join(valid_statuses)}"
        )
    
    complaints = get_complaints()
    filtered = [c for c in complaints if c.get("status", "beklemede") == status]
    
    # Fotoğraf URL'lerini ekle
    for complaint in filtered:
        if complaint.get("photo"):
            complaint["photo_url"] = f"/photos/{complaint['photo']}"
    
    return filtered


# ============================================
# KULLANICI ŞİKAYETLERİM ENDPOINTİ
# ============================================

@app.get("/my-complaints")
async def get_my_complaints(current_user: TokenData = Depends(get_current_user)):
    """
    Kullanıcının kendi şikayetlerini listele.
    Geri bildirimlerle birlikte döner.
    """
    complaints = get_complaints()
    users = get_users()
    
    # Kullanıcı ID'sini bul
    user_id = None
    for user in users:
        if user["username"] == current_user.username:
            user_id = user["id"]
            break
    
    if user_id is None:
        return []
    
    # Kullanıcının şikayetlerini filtrele
    my_complaints = [c for c in complaints if c.get("user_id") == user_id]
    
    # Fotoğraf URL'lerini ekle
    for complaint in my_complaints:
        if complaint.get("photo"):
            complaint["photo_url"] = f"/photos/{complaint['photo']}"
        if "status" not in complaint:
            complaint["status"] = "beklemede"
    
    return my_complaints


@app.post("/complaints/json/auth")
async def create_complaint_with_auth(
    complaint_data: ComplaintCreate,
    current_user: TokenData = Depends(get_current_user)
):
    """
    Giriş yapmış kullanıcı için şikayet oluştur.
    Kullanıcı ID'si otomatik eklenir.
    """
    try:
        complaints = get_complaints()
        users = get_users()
        
        # Kullanıcı ID'sini bul
        user_id = None
        for user in users:
            if user["username"] == current_user.username:
                user_id = user["id"]
                break
        
        new_id = max([c["id"] for c in complaints], default=0) + 1
        urgency = get_urgency(complaint_data.category)
        created_at = datetime.now().isoformat()
        
        # Fotoğraf işleme
        photo_path = None
        if complaint_data.photo_base64:
            try:
                photo_data = base64.b64decode(complaint_data.photo_base64)
                photo_filename = f"{new_id}_photo.jpg"
                photo_full_path = PHOTOS_DIR / photo_filename
                
                with open(photo_full_path, "wb") as f:
                    f.write(photo_data)
                
                photo_path = str(photo_filename)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Fotoğraf yüklenirken hata: {str(e)}")
        
        new_complaint = {
            "id": new_id,
            "category": complaint_data.category,
            "description": complaint_data.description,
            "lat": complaint_data.lat,
            "lon": complaint_data.lon,
            "urgency": urgency,
            "photo": photo_path,
            "created_at": created_at,
            "status": "beklemede",
            "feedback": None,
            "feedback_at": None,
            "feedback_by": None,
            "user_id": user_id
        }
        
        complaints.append(new_complaint)
        save_complaints(complaints)
        
        urgency_messages = {
            "red": "Acil durum olarak kaydedildi. En kısa sürede müdahale edilecektir.",
            "yellow": "Orta öncelikli olarak kaydedildi. En kısa sürede değerlendirilecektir.",
            "green": "Normal öncelikli olarak kaydedildi. Sırasıyla değerlendirilecektir."
        }
        
        return {
            "success": True,
            "message": f"Şikayetiniz başarıyla alınmıştır! Şikayet numaranız: #{new_id}. {urgency_messages.get(urgency, '')}",
            "complaint_id": new_id,
            "category": complaint_data.category,
            "urgency": urgency,
            "created_at": created_at
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Şikayet oluşturulurken hata: {str(e)}")


# ============================================
# Additional utility endpoints
# ============================================

@app.get("/categories")
async def get_categories():
    """Get available complaint categories with their urgency levels."""
    categories = [
        {"name": "boru_patlamasi", "label": "Boru Patlaması", "urgency": "red"},
        {"name": "su_baskini", "label": "Su Baskını", "urgency": "red"},
        {"name": "yangin", "label": "Yangın", "urgency": "red"},
        {"name": "merdiven_kirik", "label": "Merdiven Kırık", "urgency": "yellow"},
        {"name": "kaldirim_bozuk", "label": "Kaldırım Bozuk", "urgency": "yellow"},
        {"name": "rampa_eksik", "label": "Rampa Eksik", "urgency": "yellow"},
        {"name": "isik_yanmiyor", "label": "Işık Yanmıyor", "urgency": "green"},
        {"name": "cop_toplama", "label": "Çöp Toplama", "urgency": "green"},
        {"name": "diger", "label": "Diğer", "urgency": "green"},
    ]
    return categories


@app.get("/grid/{grid_id}")
async def get_grid_info(grid_id: int):
    """Get detailed info for a specific grid."""
    nearest_stops_data = get_nearest_stops_data()
    slope_scores = get_slope_scores()
    bus_stops = get_bus_stops()
    
    if grid_id not in slope_scores:
        raise HTTPException(status_code=404, detail="Grid not found")
    
    slope_score = slope_scores.get(grid_id, 0)
    stops_info = nearest_stops_data.get(grid_id, [])
    
    result_stops = []
    for stop in stops_info:
        stop_id = stop["stop_id"]
        distance_m = stop["distance"]
        stop_details = bus_stops.get(stop_id)
        
        if stop_details:
            result_stops.append({
                "stop_id": stop_id,
                "stop_name": stop_details["stop_name"],
                "lat": stop_details["lat"],
                "lon": stop_details["lon"],
                "distance_m": round(distance_m, 2),
                "duration_min": round(calculate_walking_duration(distance_m), 2)
            })
    
    return {
        "grid_id": grid_id,
        "slope_score": round(slope_score, 2),
        "nearest_stops": result_stops
    }


# ============================================
# Municipality Panel - Analytics Endpoints
# ============================================

@app.get("/analytics/summary")
async def get_analytics_summary():
    """
    Belediye paneli için genel özet istatistikler.
    Günlük, haftalık ve aylık şikayet sayıları.
    """
    complaints = get_complaints()
    now = datetime.now()
    
    # Tarih filtreleri
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    week_start = today_start - timedelta(days=now.weekday())  # Pazartesi
    month_start = today_start.replace(day=1)
    
    # Şikayetleri filtrele
    daily_complaints = []
    weekly_complaints = []
    monthly_complaints = []
    
    for complaint in complaints:
        created = datetime.fromisoformat(complaint["created_at"])
        if created >= today_start:
            daily_complaints.append(complaint)
        if created >= week_start:
            weekly_complaints.append(complaint)
        if created >= month_start:
            monthly_complaints.append(complaint)
    
    # Aciliyet dağılımı
    def get_urgency_counts(complaint_list):
        urgency_counter = Counter(c["urgency"] for c in complaint_list)
        return {
            "red": urgency_counter.get("red", 0),
            "yellow": urgency_counter.get("yellow", 0),
            "green": urgency_counter.get("green", 0)
        }
    
    # Kategori dağılımı
    def get_category_counts(complaint_list):
        return dict(Counter(c["category"] for c in complaint_list))
    
    return {
        "total_complaints": len(complaints),
        "daily": {
            "count": len(daily_complaints),
            "by_urgency": get_urgency_counts(daily_complaints),
            "by_category": get_category_counts(daily_complaints)
        },
        "weekly": {
            "count": len(weekly_complaints),
            "by_urgency": get_urgency_counts(weekly_complaints),
            "by_category": get_category_counts(weekly_complaints)
        },
        "monthly": {
            "count": len(monthly_complaints),
            "by_urgency": get_urgency_counts(monthly_complaints),
            "by_category": get_category_counts(monthly_complaints)
        }
    }


@app.get("/analytics/trend")
async def get_trend_analytics(
    days: int = Query(30, description="Kaç günlük trend", ge=1, le=365)
):
    """
    Son X gün için günlük şikayet trendi.
    Grafik çizmek için kullanılabilir.
    """
    complaints = get_complaints()
    now = datetime.now()
    start_date = now - timedelta(days=days)
    
    # Günlük sayıları hesapla
    daily_counts = {}
    for i in range(days):
        date = (start_date + timedelta(days=i+1)).strftime("%Y-%m-%d")
        daily_counts[date] = {"total": 0, "red": 0, "yellow": 0, "green": 0}
    
    for complaint in complaints:
        created = datetime.fromisoformat(complaint["created_at"])
        if created >= start_date:
            date_key = created.strftime("%Y-%m-%d")
            if date_key in daily_counts:
                daily_counts[date_key]["total"] += 1
                daily_counts[date_key][complaint["urgency"]] += 1
    
    # Liste formatına çevir
    trend_data = [
        {"date": date, **counts}
        for date, counts in sorted(daily_counts.items())
    ]
    
    return {
        "period_days": days,
        "trend": trend_data
    }


@app.get("/analytics/hotspots")
async def get_hotspot_analytics():
    """
    Şikayet yoğunluğu yüksek bölgeler (hotspots).
    Haritada göstermek için grid bazlı analiz.
    """
    complaints = get_complaints()
    grid_features = get_grid_features()
    
    # Her şikayetin hangi grid'e düştüğünü bul
    grid_complaints = {}
    
    for complaint in complaints:
        grid_id = find_grid_for_point(
            complaint["lat"], 
            complaint["lon"], 
            grid_features
        )
        if grid_id is None:
            grid_id = find_nearest_grid(
                complaint["lat"], 
                complaint["lon"], 
                grid_features
            )
        
        if grid_id:
            if grid_id not in grid_complaints:
                grid_complaints[grid_id] = {
                    "grid_id": grid_id,
                    "total": 0,
                    "red": 0,
                    "yellow": 0,
                    "green": 0,
                    "categories": []
                }
            grid_complaints[grid_id]["total"] += 1
            grid_complaints[grid_id][complaint["urgency"]] += 1
            grid_complaints[grid_id]["categories"].append(complaint["category"])
    
    # Kategori sayılarını hesapla ve sırala
    hotspots = []
    for grid_id, data in grid_complaints.items():
        data["top_categories"] = dict(Counter(data["categories"]).most_common(3))
        del data["categories"]
        hotspots.append(data)
    
    # En çok şikayet alanlara göre sırala
    hotspots.sort(key=lambda x: x["total"], reverse=True)
    
    return {
        "total_grids_with_complaints": len(hotspots),
        "hotspots": hotspots[:20]  # En yoğun 20 bölge
    }


@app.get("/analytics/urgent")
async def get_urgent_complaints():
    """
    Acil müdahale gerektiren şikayetler (kırmızı aciliyet).
    Belediye panelinde öncelikli gösterilmeli.
    """
    complaints = get_complaints()
    
    # Sadece kırmızı aciliyetli olanları filtrele
    urgent = [c for c in complaints if c["urgency"] == "red"]
    
    # En yeniden eskiye sırala
    urgent.sort(key=lambda x: x["created_at"], reverse=True)
    
    return {
        "count": len(urgent),
        "complaints": urgent
    }


# ============================================
# DURAK ENDPOINTLERİ
# ============================================

# Varsayılan bölge sınırları (Ankara merkez - küçük bir alan)
DEFAULT_BOUNDS = {
    "min_lat": 39.90,
    "max_lat": 39.95,
    "min_lon": 32.82,
    "max_lon": 32.90
}


@app.get("/bus-stops")
async def get_all_bus_stops(
    min_lat: Optional[float] = Query(None, description="Minimum enlem"),
    max_lat: Optional[float] = Query(None, description="Maksimum enlem"),
    min_lon: Optional[float] = Query(None, description="Minimum boylam"),
    max_lon: Optional[float] = Query(None, description="Maksimum boylam"),
    limit: int = Query(100, description="Maksimum durak sayısı", ge=1, le=500)
):
    """
    Tüm durakları veya belirli bir bölgedeki durakları listele.
    
    Parametresiz çağrılırsa varsayılan bölgedeki durakları döner.
    Bölge parametreleri verilirse o bölgedeki durakları döner.
    
    Örnek: /bus-stops?min_lat=39.90&max_lat=39.95&min_lon=32.82&max_lon=32.90
    """
    bus_stops = get_bus_stops()
    stops_list = list(bus_stops.values())
    
    # Bölge filtresi uygula
    if min_lat is None:
        min_lat = DEFAULT_BOUNDS["min_lat"]
    if max_lat is None:
        max_lat = DEFAULT_BOUNDS["max_lat"]
    if min_lon is None:
        min_lon = DEFAULT_BOUNDS["min_lon"]
    if max_lon is None:
        max_lon = DEFAULT_BOUNDS["max_lon"]
    
    filtered_stops = [
        stop for stop in stops_list
        if min_lat <= stop["lat"] <= max_lat and min_lon <= stop["lon"] <= max_lon
    ]
    
    # Limit uygula
    filtered_stops = filtered_stops[:limit]
    
    return {
        "total": len(filtered_stops),
        "bounds": {
            "min_lat": min_lat,
            "max_lat": max_lat,
            "min_lon": min_lon,
            "max_lon": max_lon
        },
        "stops": filtered_stops
    }


@app.get("/bus-stops/nearby")
async def get_nearby_bus_stops(
    lat: float = Query(..., description="Merkez enlem"),
    lon: float = Query(..., description="Merkez boylam"),
    radius_km: float = Query(1.0, description="Yarıçap (km)", ge=0.1, le=5.0),
    limit: int = Query(20, description="Maksimum durak sayısı", ge=1, le=100)
):
    """
    Belirli bir noktanın çevresindeki durakları getir.
    
    Örnek: /bus-stops/nearby?lat=39.92&lon=32.85&radius_km=0.5
    """
    import math
    
    bus_stops = get_bus_stops()
    stops_list = list(bus_stops.values())
    
    # Haversine mesafe hesaplama
    def haversine(lat1, lon1, lat2, lon2):
        R = 6371  # Dünya yarıçapı (km)
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        return R * c
    
    # Mesafeyi hesapla ve filtrele
    nearby_stops = []
    for stop in stops_list:
        distance = haversine(lat, lon, stop["lat"], stop["lon"])
        if distance <= radius_km:
            stop_with_distance = stop.copy()
            stop_with_distance["distance_km"] = round(distance, 3)
            stop_with_distance["distance_m"] = round(distance * 1000, 1)
            nearby_stops.append(stop_with_distance)
    
    # Mesafeye göre sırala
    nearby_stops.sort(key=lambda x: x["distance_km"])
    
    # Limit uygula
    nearby_stops = nearby_stops[:limit]
    
    return {
        "center": {"lat": lat, "lon": lon},
        "radius_km": radius_km,
        "total": len(nearby_stops),
        "stops": nearby_stops
    }


@app.get("/bus-stops/bounds")
async def get_bus_stops_bounds():
    """
    Tüm durakların kapsadığı alanın sınırlarını döner.
    Frontend harita başlangıç konumu için kullanılabilir.
    """
    bus_stops = get_bus_stops()
    stops_list = list(bus_stops.values())
    
    if not stops_list:
        return {"error": "Durak verisi bulunamadı"}
    
    lats = [s["lat"] for s in stops_list]
    lons = [s["lon"] for s in stops_list]
    
    return {
        "total_stops": len(stops_list),
        "bounds": {
            "min_lat": min(lats),
            "max_lat": max(lats),
            "min_lon": min(lons),
            "max_lon": max(lons)
        },
        "center": {
            "lat": (min(lats) + max(lats)) / 2,
            "lon": (min(lons) + max(lons)) / 2
        },
        "default_bounds": DEFAULT_BOUNDS
    }


# ============================================
# FOTOĞRAF ENDPOINTİ
# ============================================

@app.get("/photo/{filename}")
async def get_photo(filename: str):
    """
    Şikayet fotoğrafını getir.
    
    Örnek: /photo/51_photo.jpg
    """
    photo_path = PHOTOS_DIR / filename
    
    if not photo_path.exists():
        raise HTTPException(status_code=404, detail="Fotoğraf bulunamadı")
    
    return FileResponse(
        path=str(photo_path),
        media_type="image/jpeg",
        filename=filename
    )


# ============================================
# DURUM SABİTLERİ
# ============================================

@app.get("/complaint-statuses")
async def get_complaint_statuses():
    """
    Şikayet durumlarını listele.
    Frontend'de dropdown için kullanılabilir.
    """
    return [
        {"value": "beklemede", "label": "Beklemede", "color": "gray"},
        {"value": "inceleniyor", "label": "İnceleniyor", "color": "blue"},
        {"value": "cozuldu", "label": "Çözüldü", "color": "green"},
        {"value": "reddedildi", "label": "Reddedildi", "color": "red"}
    ]


# ============================================
# Run server
# ============================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)

