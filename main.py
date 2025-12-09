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
from pydantic import BaseModel

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
    StaffCreate,
    Token,
    TokenData,
    UserResponse,
    StaffResponse,
    register_user,
    login_user,
    add_staff,
    get_current_user,
    get_current_staff,
    get_users,
    get_staff,
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
    status: str
    id: int


class Complaint(BaseModel):
    id: int
    category: str
    description: str
    lat: float
    lon: float
    urgency: str
    photo: Optional[str] = None
    created_at: str


# ============================================
# AUTH ENDPOINTS
# ============================================

@app.post("/auth/register", response_model=Token)
async def register(user_data: UserRegister):
    """
    Yeni kullanıcı kaydı.
    Kayıt başarılı olursa otomatik olarak giriş yapılır ve token döner.
    
    - **username**: Kullanıcı adı (benzersiz olmalı)
    - **password**: Şifre
    - **email**: E-posta (opsiyonel)
    - **full_name**: Ad Soyad (opsiyonel)
    """
    new_user = register_user(user_data)
    
    # Otomatik giriş yap
    login_data = UserLogin(username=user_data.username, password=user_data.password)
    return login_user(login_data)


@app.post("/auth/login", response_model=Token)
async def login(login_data: UserLogin):
    """
    Kullanıcı veya personel girişi.
    
    Dönen token'daki **role** alanına göre yönlendirme yapılmalı:
    - role: "user" → Kullanıcı Paneli
    - role: "staff" → Belediye Paneli
    
    - **username**: Kullanıcı adı
    - **password**: Şifre
    """
    return login_user(login_data)


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

@app.post("/staff/add", response_model=StaffResponse)
async def create_staff(
    staff_data: StaffCreate,
    current_user: TokenData = Depends(get_current_staff)
):
    """
    Yeni belediye personeli ekle.
    Sadece mevcut belediye personeli bu işlemi yapabilir.
    
    - **username**: Kullanıcı adı (benzersiz olmalı)
    - **password**: Şifre
    - **full_name**: Ad Soyad
    - **department**: Departman (opsiyonel)
    """
    new_staff = add_staff(staff_data, current_user.username)
    return StaffResponse(
        id=new_staff["id"],
        username=new_staff["username"],
        full_name=new_staff["full_name"],
        department=new_staff.get("department"),
        created_at=new_staff["created_at"],
        created_by=new_staff.get("created_by")
    )


@app.get("/staff/list", response_model=list[StaffResponse])
async def list_staff(current_user: TokenData = Depends(get_current_staff)):
    """
    Tüm belediye personelini listele.
    Sadece belediye personeli görebilir.
    """
    staff_list = get_staff()
    return [
        StaffResponse(
            id=s["id"],
            username=s["username"],
            full_name=s["full_name"],
            department=s.get("department"),
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
    Create a new complaint.
    
    Accepts:
        - category: Type of complaint (e.g., "boru_patlamasi", "merdiven_kirik")
        - description: Detailed description
        - lat, lon: Location coordinates
        - photo: Optional photo upload
    
    Returns:
        - status: "ok" if successful
        - id: Generated complaint ID
    """
    # Load existing complaints
    complaints = get_complaints()
    
    # Generate new ID
    new_id = max([c["id"] for c in complaints], default=0) + 1
    
    # Determine urgency
    urgency = get_urgency(category)
    
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
        "created_at": datetime.now().isoformat()
    }
    
    # Append and save
    complaints.append(new_complaint)
    save_complaints(complaints)
    
    return ComplaintResponse(status="ok", id=new_id)


@app.post("/complaints/base64", response_model=ComplaintResponse)
async def create_complaint_base64(
    category: str = Form(...),
    description: str = Form(...),
    lat: float = Form(...),
    lon: float = Form(...),
    photo_base64: Optional[str] = Form(None, description="Base64 encoded photo")
):
    """
    Create a new complaint with base64 encoded photo.
    Alternative endpoint for clients that prefer base64.
    """
    # Load existing complaints
    complaints = get_complaints()
    
    # Generate new ID
    new_id = max([c["id"] for c in complaints], default=0) + 1
    
    # Determine urgency
    urgency = get_urgency(category)
    
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
            raise HTTPException(status_code=400, detail=f"Invalid base64 photo: {str(e)}")
    
    # Create complaint object
    new_complaint = {
        "id": new_id,
        "category": category,
        "description": description,
        "lat": lat,
        "lon": lon,
        "urgency": urgency,
        "photo": photo_path,
        "created_at": datetime.now().isoformat()
    }
    
    # Append and save
    complaints.append(new_complaint)
    save_complaints(complaints)
    
    return ComplaintResponse(status="ok", id=new_id)


@app.get("/complaints", response_model=list[Complaint])
async def list_complaints():
    """
    Get all complaints.
    
    Returns list of all complaints with their details.
    Used by Municipality Panel to view and manage complaints.
    """
    complaints = get_complaints()
    return complaints


@app.get("/complaints/{complaint_id}", response_model=Complaint)
async def get_complaint(complaint_id: int):
    """Get a specific complaint by ID."""
    complaints = get_complaints()
    
    for complaint in complaints:
        if complaint["id"] == complaint_id:
            return complaint
    
    raise HTTPException(status_code=404, detail="Complaint not found")


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
# Run server
# ============================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)

