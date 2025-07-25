"""
Student API endpoints
"""

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.orm import Session as dbSession
from pydantic import BaseModel
from datetime import datetime, timezone

from app.core.database import get_db, Session, User, AttentionScore, Feedback
from app.core.auth import verify_token
from app.core.config import settings

router = APIRouter()


# Pydantic models
class SessionCreate(BaseModel):
    session_name: str
    teacher_id: Optional[int] = None


class SessionResponse(BaseModel):
    id: int
    session_name: str
    start_time: datetime
    is_active: bool
    
    class Config:
        from_attributes = True


class AttentionScoreResponse(BaseModel):
    attention_level: float
    engagement_level: Optional[float]
    confidence: float
    timestamp: datetime
    distraction_type: Optional[str]


class FeedbackResponse(BaseModel):
    feedback_type: str
    message: str
    suggestions: Optional[str]
    created_at: datetime


@router.post("/session/start", response_model=SessionResponse)
async def start_session(
    session_data: SessionCreate,
    current_user: str = Depends(verify_token),
    db: dbSession = Depends(get_db)
):
    """Start a new learning session"""
    # TODO: Implement session creation logic

    user = db.query(User).filter(User.username == current_user).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")


    new_session = Session(
        student_id=user.id,
        teacher_id= session_data.teacher_id,
        session_name=session_data.session_name,
        )

    db.add(new_session)
    db.commit()

    return SessionResponse(
        id=new_session.id,
        session_name=new_session.session_name,
        start_time=new_session.start_time,
        is_active=new_session.is_active
    )



@router.post("/session/{session_id}/end")
async def end_session(
    session_id: int,
    current_user: str = Depends(verify_token),
    db: dbSession = Depends(get_db)
):
    """End a learning session"""
    # TODO: Implement session ending logic
    user = db.query(User).filter(User.username == current_user).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    session = db.query(Session).filter(Session.id == session_id).filter(Session.student_id == user.id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    if not session.is_active:
        raise HTTPException(status_code=400, detail="Session is already ended")

    session.is_active = False
    session.end_time = datetime.now()
    db.commit()

    return {"message": "Session ended successfully"}


@router.post("/video/frame")
async def process_video_frame(
    db: dbSession = Depends(get_db),
    file: UploadFile = File(...),
    session_id: int = None,
    current_user: str = Depends(verify_token,
    )
):
    """Process a video frame for attention analysis"""
    # TODO: Implement video frame processing
    # 1. Validate file type and size
    accepted_file_types = ["image/jpeg", "image/png"]
    if file.content_type not in accepted_file_types:
        raise HTTPException(status_code=400, detail="Invalid file type. Only JPEG and PNG are accepted.")

    file_bytes = await file.read()
    if len(file_bytes) > settings.MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File size exceeds the maximum limit of 10MB.")

    # 2. Process with OpenCV
    # 3. Run AI model for attention detection
    # 4. Save results to database

    #Temporary placeholder for processing logic
    attention_score = AttentionScore(
        attention_level = 0.85,
        engagement_level= 0.78,
        confidence = 0.92,
        timestamp = datetime.now(timezone.utc),
        distraction_type = "Phone"
    )

    db.add(attention_score)
    db.commit()

    # 5. Return attention score
    return attention_score


@router.get("/attention/score", response_model=AttentionScoreResponse)
async def get_attention_score(
    session_id: int,
    current_user: str = Depends(verify_token),
    db: dbSession = Depends(get_db)
):
    """Get latest attention score for a session"""
    # TODO: Implement get attention score logic
    user = db.query(User).filter(User.username == current_user).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    session = db.query(Session).filter(Session.id == session_id).filter(Session.student_id == user.id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    attention_score = (db.query(AttentionScore).filter(AttentionScore.session_id == session_id)
                       .order_by(AttentionScore.timestamp.desc()).first())

    attention_score_response = AttentionScoreResponse(
        attention_level=attention_score.attention_level,
        engagement_level=attention_score.engagement_level,
        confidence=attention_score.confidence,
        timestamp=attention_score.timestamp,
        distraction_type=attention_score.distraction_type
    )

    return attention_score_response


@router.get("/attention/history")
async def get_attention_history(
    session_id: int,
    limit: int = 100,
    current_user: str = Depends(verify_token),
    db: dbSession = Depends(get_db)
):
    """Get attention score history for a session"""
    # TODO: Implement attention history logic
    attention_scores = []

    user = db.query(User).filter(User.username == current_user).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    session = db.query(Session).filter(Session.id == session_id).filter(Session.student_id == user.id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    attention_scores = (db.query(AttentionScore).filter(AttentionScore.session_id == session_id)
                       .order_by(AttentionScore.timestamp.desc()).limit(limit).all())

    attention_score_responses = [
        AttentionScoreResponse(
            attention_level=score.attention_level,
            engagement_level=score.engagement_level,
            confidence=score.confidence,
            timestamp=score.timestamp,
            distraction_type=score.distraction_type
        )
        for score in attention_scores
    ]

    return {
        "session_id": session_id,
        "scores": attention_score_responses,
    }


@router.get("/feedback", response_model=List[FeedbackResponse])
async def get_feedback(
    session_id: int,
    current_user: str = Depends(verify_token),
    db: dbSession = Depends(get_db)
):
    """Get personalized feedback for a session"""
    # TODO: Implement feedback generation logic

    user = db.query(User).filter(User.username == current_user).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    session = db.query(Session).filter(Session.id == session_id).filter(Session.student_id == user.id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    feedback = db.query(Feedback).filter(Feedback.session_id == session_id).first()
    feedback_response = FeedbackResponse(
        feedback_type= feedback.feedback_type,
        message=feedback.message,
        suggestions=feedback.suggestions if feedback.suggestions else None,
        created_at=feedback.created_at
    )

    return feedback_response


@router.get("/sessions", response_model=List[SessionResponse])
async def get_user_sessions(
    current_user: str = Depends(verify_token),
    db: dbSession = Depends(get_db)
):
    """Get all sessions for current user"""
    # TODO: Implement get user sessions logic
    sessions = []
    user = db.query(User).filter(User.username == current_user).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    sessions = db.query(Session).filter(Session.student_id == user.id).all()
    if len(sessions) == 0:
        raise HTTPException(status_code=404, detail="No sessions found for user")

    return sessions
