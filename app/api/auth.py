"""
Authentication API endpoints
"""
from datetime import timedelta

from app.core.auth import (create_access_token, get_password_hash,
                           verify_password, verify_token)
from app.core.config import settings
from app.core.database import get_db
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr
from sqlalchemy.orm import Session

router = APIRouter()
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
    # Import your User model
    from app.models.user import User

    user = db.query(User).filter(User.username == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Incorrect username or password")
    access_token = create_access_token(data={"sub": user.username})
    return {"access_token": access_token, "token_type": "bearer"}
@router.get("/me", response_model=UserResponse)
async def read_users_me(current_user: dict = Depends(verify_token)):
    return {
        "id": current_user.get("id"),
        "email": current_user.get("email"),
        "username": current_user.get("username"),
        "full_name": current_user.get("full_name"),
        "role": current_user.get("role"),
        "is_active": current_user.get("is_active", True)
    }