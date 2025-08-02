"""
Student API endpoints
"""
from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Dict, Any

router = APIRouter(prefix="/student", tags=["student"])

class StudentProfile(BaseModel):
    id: str
    name: str
    email: str
    class_id: str

class AttentionData(BaseModel):
    timestamp: int
    attention_score: float
    emotion: str
    engagement_level: float

@router.get("/profile", response_model=StudentProfile)
async def get_student_profile():
    """
    Get student profile information
    """
    return StudentProfile(
        id="student-123",
        name="Test Student",
        email="student@test.com",
        class_id="class-456"
    )

@router.get("/attention", response_model=List[AttentionData])
async def get_attention_data():
    """
    Get student attention data
    """
    return [
        AttentionData(
            timestamp=1643723400,
            attention_score=0.8,
            emotion="happy",
            engagement_level=0.75
        )
    ]

@router.get("/dashboard")
async def get_dashboard_data():
    """
    Get student dashboard data
    """
    return {
        "total_classes": 12,
        "average_attention": 0.78,
        "improvement_rate": 0.15,
        "recent_activities": []
    }
