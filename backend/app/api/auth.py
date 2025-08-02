"""
Authentication API endpoints
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/auth", tags=["authentication"])

class LoginRequest(BaseModel):
    username: str
    password: str

class LoginResponse(BaseModel):
    access_token: str
    token_type: str
    user_id: str

@router.post("/login", response_model=LoginResponse)
async def login(request: LoginRequest):
    """
    User login endpoint
    """
    # Placeholder implementation
    if request.username == "test" and request.password == "test":
        return LoginResponse(
            access_token="fake-jwt-token",
            token_type="bearer",
            user_id="test-user"
        )
    raise HTTPException(status_code=401, detail="Invalid credentials")

@router.post("/logout")
async def logout():
    """
    User logout endpoint
    """
    return {"message": "Logged out successfully"}
