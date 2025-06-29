"""
Authentication API endpoints
"""

from datetime import timedelta
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from pydantic import BaseModel, EmailStr

from app.core.database import get_db
from app.core.auth import verify_password, get_password_hash, create_access_token, verify_token
from app.core.config import settings

router = APIRouter()


# Pydantic models
class UserCreate(BaseModel):
    email: EmailStr
    username: str
    full_name: str
    password: str
    role: str  # 'student' or 'teacher'


class UserResponse(BaseModel):
    id: int
    email: str
    username: str
    full_name: str
    role: str
    is_active: bool
    
    class Config:
        from_attributes = True


class Token(BaseModel):
    access_token: str
    token_type: str


@router.post("/register", response_model=UserResponse)
async def register(user: UserCreate, db: Session = Depends(get_db)):
    """Register a new user"""
    # TODO: Implement user creation logic
    # Check if user exists, hash password, save to database
    return {
        "id": 1,
        "email": user.email,
        "username": user.username,
        "full_name": user.full_name,
        "role": user.role,
        "is_active": True
    }


@router.post("/login", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    """Login user and return access token"""
    # TODO: Implement authentication logic
    # Verify user credentials and return JWT token
    
    # For now, return a dummy token
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": form_data.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}


@router.get("/me", response_model=UserResponse)
async def read_users_me(current_user: str = Depends(verify_token)):
    """Get current user information"""
    # TODO: Implement get current user logic
    return {
        "id": 1,
        "email": "user@example.com",
        "username": current_user,
        "full_name": "Test User",
        "role": "student",
        "is_active": True
    }
