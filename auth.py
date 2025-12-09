"""
Authentication module for User and Staff management
JWT token based authentication with role-based access control

Roller:
- user: Normal vatandaş kullanıcı
- staff: Belediye personeli
  - yonetici: Tam yetki (personel ekleme dahil)
  - operasyon: Şikayet yönetimi
  - analiz: Sadece görüntüleme ve analiz
"""

from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel

from geo import load_json, save_json

# ============================================
# Configuration
# ============================================

SECRET_KEY = "belediye-accessibility-secret-key-2024"  # Production'da env variable kullanılmalı
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 saat

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Security
security = HTTPBearer()

# Data paths
DATA_DIR = Path(__file__).parent
USERS_PATH = DATA_DIR / "users.json"
STAFF_PATH = DATA_DIR / "staff.json"


# ============================================
# Enums
# ============================================

class StaffRole(str, Enum):
    YONETICI = "yonetici"
    OPERASYON = "operasyon"
    ANALIZ = "analiz"


# ============================================
# Models
# ============================================

class UserRegister(BaseModel):
    username: str
    password: str
    email: Optional[str] = None
    full_name: Optional[str] = None


class UserLogin(BaseModel):
    username: str
    password: str


class StaffLogin(BaseModel):
    """Belediye personeli girişi için ayrı model"""
    username: str
    password: str


class StaffCreate(BaseModel):
    username: str
    password: str
    full_name: str
    department: Optional[str] = None
    staff_role: StaffRole = StaffRole.OPERASYON  # Varsayılan rol


class Token(BaseModel):
    access_token: str
    token_type: str
    role: str  # "user" veya "staff"
    staff_role: Optional[str] = None  # "yonetici", "operasyon", "analiz"
    username: str
    full_name: Optional[str] = None


class TokenData(BaseModel):
    username: str
    role: str
    staff_role: Optional[str] = None


class UserResponse(BaseModel):
    id: int
    username: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    created_at: str


class StaffResponse(BaseModel):
    id: int
    username: str
    full_name: str
    department: Optional[str] = None
    staff_role: str
    created_at: str
    created_by: Optional[str] = None


# ============================================
# Helper Functions
# ============================================

def get_users() -> list:
    """Load users from file."""
    if not USERS_PATH.exists():
        # Varsayılan kullanıcı admin hesabı
        default_users = [{
            "id": 1,
            "username": "kullanici_admin",
            "password_hash": pwd_context.hash("kullanici123"),
            "email": "admin@kullanici.com",
            "full_name": "Kullanıcı Admin",
            "created_at": datetime.now().isoformat()
        }]
        save_json(USERS_PATH, default_users)
        return default_users
    return load_json(USERS_PATH)


def save_users(users: list):
    """Save users to file."""
    save_json(USERS_PATH, users)


def get_staff() -> list:
    """Load staff from file."""
    if not STAFF_PATH.exists():
        # Varsayılan belediye admin kullanıcısı oluştur
        default_staff = [{
            "id": 1,
            "username": "belediye_admin",
            "password_hash": pwd_context.hash("belediye123"),
            "full_name": "Belediye Sistem Yöneticisi",
            "department": "IT",
            "staff_role": "yonetici",
            "created_at": datetime.now().isoformat(),
            "created_by": "system"
        }]
        save_json(STAFF_PATH, default_staff)
        return default_staff
    return load_json(STAFF_PATH)


def save_staff(staff: list):
    """Save staff to file."""
    save_json(STAFF_PATH, staff)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Hash a password."""
    return pwd_context.hash(password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create a JWT access token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def decode_token(token: str) -> Optional[TokenData]:
    """Decode and validate a JWT token."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        role: str = payload.get("role")
        staff_role: str = payload.get("staff_role")
        if username is None or role is None:
            return None
        return TokenData(username=username, role=role, staff_role=staff_role)
    except JWTError:
        return None


# ============================================
# Authentication Functions
# ============================================

def authenticate_user(username: str, password: str) -> Optional[dict]:
    """Authenticate a regular user."""
    users = get_users()
    for user in users:
        if user["username"] == username:
            if verify_password(password, user["password_hash"]):
                return user
    return None


def authenticate_staff(username: str, password: str) -> Optional[dict]:
    """Authenticate a staff member."""
    staff = get_staff()
    for member in staff:
        if member["username"] == username:
            if verify_password(password, member["password_hash"]):
                return member
    return None


def get_user_by_username(username: str) -> Optional[dict]:
    """Get user by username."""
    users = get_users()
    for user in users:
        if user["username"] == username:
            return user
    return None


def get_staff_by_username(username: str) -> Optional[dict]:
    """Get staff by username."""
    staff = get_staff()
    for member in staff:
        if member["username"] == username:
            return member
    return None


# ============================================
# Dependency Functions
# ============================================

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> TokenData:
    """Get current user from JWT token."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Geçersiz kimlik bilgileri",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    token = credentials.credentials
    token_data = decode_token(token)
    
    if token_data is None:
        raise credentials_exception
    
    return token_data


async def get_current_staff(current_user: TokenData = Depends(get_current_user)) -> TokenData:
    """Verify that current user is staff."""
    if current_user.role != "staff":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Bu işlem için belediye personeli yetkisi gerekiyor"
        )
    return current_user


async def get_current_yonetici(current_user: TokenData = Depends(get_current_staff)) -> TokenData:
    """Verify that current user is a yonetici (manager)."""
    if current_user.staff_role != "yonetici":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Bu işlem için yönetici yetkisi gerekiyor"
        )
    return current_user


async def get_current_operasyon_or_yonetici(current_user: TokenData = Depends(get_current_staff)) -> TokenData:
    """Verify that current user is operasyon or yonetici."""
    if current_user.staff_role not in ["yonetici", "operasyon"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Bu işlem için operasyon veya yönetici yetkisi gerekiyor"
        )
    return current_user


# ============================================
# Registration and Login Functions
# ============================================

def register_user(user_data: UserRegister) -> dict:
    """Register a new user."""
    users = get_users()
    
    # Check if username exists
    for user in users:
        if user["username"] == user_data.username:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Bu kullanıcı adı zaten kullanılıyor"
            )
    
    # Check in staff too
    staff = get_staff()
    for member in staff:
        if member["username"] == user_data.username:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Bu kullanıcı adı zaten kullanılıyor"
            )
    
    # Create new user
    new_id = max([u["id"] for u in users], default=0) + 1
    new_user = {
        "id": new_id,
        "username": user_data.username,
        "password_hash": get_password_hash(user_data.password),
        "email": user_data.email,
        "full_name": user_data.full_name,
        "created_at": datetime.now().isoformat()
    }
    
    users.append(new_user)
    save_users(users)
    
    return new_user


def login_user(login_data: UserLogin) -> Token:
    """Login regular user only."""
    user = authenticate_user(login_data.username, login_data.password)
    if user:
        access_token = create_access_token(
            data={"sub": user["username"], "role": "user", "staff_role": None}
        )
        return Token(
            access_token=access_token,
            token_type="bearer",
            role="user",
            staff_role=None,
            username=user["username"],
            full_name=user.get("full_name")
        )
    
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Kullanıcı adı veya şifre hatalı",
        headers={"WWW-Authenticate": "Bearer"},
    )


def login_staff(login_data: StaffLogin) -> Token:
    """Login staff member only."""
    staff_member = authenticate_staff(login_data.username, login_data.password)
    if staff_member:
        staff_role = staff_member.get("staff_role", "operasyon")
        access_token = create_access_token(
            data={
                "sub": staff_member["username"], 
                "role": "staff",
                "staff_role": staff_role
            }
        )
        return Token(
            access_token=access_token,
            token_type="bearer",
            role="staff",
            staff_role=staff_role,
            username=staff_member["username"],
            full_name=staff_member.get("full_name")
        )
    
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Kullanıcı adı veya şifre hatalı",
        headers={"WWW-Authenticate": "Bearer"},
    )


def add_staff(staff_data: StaffCreate, created_by: str) -> dict:
    """Add a new staff member (only by yonetici)."""
    staff = get_staff()
    users = get_users()
    
    # Check if username exists in staff
    for member in staff:
        if member["username"] == staff_data.username:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Bu kullanıcı adı zaten kullanılıyor"
            )
    
    # Check if username exists in users
    for user in users:
        if user["username"] == staff_data.username:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Bu kullanıcı adı zaten kullanılıyor"
            )
    
    # Create new staff member
    new_id = max([s["id"] for s in staff], default=0) + 1
    new_staff = {
        "id": new_id,
        "username": staff_data.username,
        "password_hash": get_password_hash(staff_data.password),
        "full_name": staff_data.full_name,
        "department": staff_data.department,
        "staff_role": staff_data.staff_role.value,
        "created_at": datetime.now().isoformat(),
        "created_by": created_by
    }
    
    staff.append(new_staff)
    save_staff(staff)
    
    return new_staff


def get_staff_roles():
    """Get available staff roles."""
    return [
        {"value": "yonetici", "label": "Yönetici", "description": "Tam yetki - personel ekleme dahil"},
        {"value": "operasyon", "label": "Operasyon Ekibi", "description": "Şikayet yönetimi ve takibi"},
        {"value": "analiz", "label": "Analiz Ekibi", "description": "Raporlama ve analiz görüntüleme"},
    ]
