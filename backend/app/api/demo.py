"""
Demo API endpoints
"""
from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Dict, Any
import random
import time

router = APIRouter(prefix="/demo", tags=["demo"])

class DemoSessionData(BaseModel):
    session_id: str
    start_time: int
    duration: int
    participants: int

@router.get("/session", response_model=DemoSessionData)
async def get_demo_session():
    """
    Get demo session information
    """
    return DemoSessionData(
        session_id="demo-session-123",
        start_time=int(time.time()),
        duration=3600,
        participants=15
    )

@router.get("/live-data")
async def get_live_demo_data():
    """
    Get live demo data for demonstration
    """
    return {
        "timestamp": int(time.time()),
        "active_students": random.randint(10, 20),
        "average_attention": round(random.uniform(0.6, 0.9), 2),
        "emotions": {
            "happy": random.randint(5, 15),
            "neutral": random.randint(3, 8),
            "surprised": random.randint(1, 5),
            "confused": random.randint(0, 3)
        },
        "engagement_levels": [
            round(random.uniform(0.5, 1.0), 2) for _ in range(15)
        ]
    }

@router.post("/start-analysis")
async def start_demo_analysis():
    """
    Start demo analysis session
    """
    return {
        "message": "Demo analysis started",
        "session_id": f"demo-{int(time.time())}",
        "status": "active"
    }

@router.post("/stop-analysis")
async def stop_demo_analysis():
    """
    Stop demo analysis session
    """
    return {
        "message": "Demo analysis stopped",
        "status": "inactive"
    }
