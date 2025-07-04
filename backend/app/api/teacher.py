"""
Teacher API endpoints
"""

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session as dbSession
from pydantic import BaseModel
from datetime import datetime, timezone

from app.core.database import get_db, User, Class, Session, AttentionScore
from app.core.auth import verify_token

router = APIRouter()


# Pydantic models
class StudentStatus(BaseModel):
    student_id: int
    student_name: str
    current_attention: float
    current_engagement: float
    last_update: datetime
    status: str  # 'active', 'distracted', 'offline'
    distraction_type: Optional[str]


class ClassOverview(BaseModel):
    class_id: int
    total_students: int
    active_students: int
    average_attention: float
    alerts_count: int


class SessionAnalytics(BaseModel):
    session_id: int
    student_name: str
    duration_minutes: int
    average_attention: float
    attention_peaks: List[datetime]
    attention_drops: List[datetime]
    total_distractions: int
    engagement_score: float


class Alert(BaseModel):
    alert_id: int
    student_id: int
    student_name: str
    alert_type: str  # 'low_attention', 'distraction', 'offline'
    message: str
    timestamp: datetime
    severity: str  # 'low', 'medium', 'high'


@router.get("/students", response_model=List[StudentStatus])
async def get_students_status(
    class_id: Optional[int] = None,
    current_user: str = Depends(verify_token),
    db: dbSession = Depends(get_db)
):
    """Get current status of all students"""
    # TODO: Implement get students status logic
    students = []
    result = []

    user = db.query(User).filter(User.username == current_user).first()

    if user.role != 'teacher':
        raise HTTPException(status_code=403, detail="Access denied")

    #if class_id is not provided, get all students from all classes the teacher is assigned to
    if class_id is None:
        teachers_classes = user.teaching_classes
        if len(teachers_classes) == 0:
           raise HTTPException(status_code=404, detail="No classes found for user")
        for cls in teachers_classes:
            students.extend(cls.students)
        students = list({student.id: student for student in students}.values()) # Remove duplicates

    #if class_id is provided, get students from that specific class
    else:
        teachers_class = (db.query(Class).filter(Class.id == class_id)
                          .filter(Class.teacher_id == user.id).first())
        if not teachers_class:
            raise HTTPException(status_code=404, detail="Class not found or access denied")
        students = list({student.id: student for student in students}.values())


    # We have students, now we need to get their last session data to retrieve their last attention_score
    for student in students:
    #Set default values for students without sessions, if they have sessions, they will be updated.
        last_session = db.query(Session).filter(Session.student_id == student.id).order_by(Session.start_time.desc()).first()
        attention = 0.0
        engagement = 0.0
        last_update = datetime.now(timezone.utc)
        status = "offline"
        distraction_type = None

        if last_session:
            attention_score = db.query(AttentionScore).filter(AttentionScore.session_id == last_session.id).order_by(AttentionScore.timestamp.desc()).first()
            if attention_score:
                attention =attention_score.attention_level,
                engagement = attention_score.engagement_level,
                last_update= attention_score.timestamp,
                status= "active" if attention_score.attention_level >= 0.5 else "distracted",
                distraction_type= attention_score.distraction_type if attention_score.attention_level < 0.5 else None

        result.append(
            StudentStatus(
                student_id=student.id,
                student_name=student.name,
                current_attention=attention,
                current_engagement=engagement,
                last_update=last_update,
                status=status,
                distraction_type=distraction_type
            )
        )

    return result


@router.get("/overview", response_model=ClassOverview)
async def get_class_overview(
    class_id: Optional[int] = None,
    current_user: str = Depends(verify_token),
    db: dbSession = Depends(get_db)
):
    """Get class overview statistics"""
    # TODO: Implement class overview logic
    students = []

    user = db.query(User).filter(User.username == current_user).first()
    if user.role != 'teacher':
        raise HTTPException(status_code=403, detail="Access denied")

    if class_id is None:
        teachers_classes = user.teaching_classes
        if len(teachers_classes) == 0:
            raise HTTPException(status_code=404, detail="No classes found for user")
        for cls in teachers_classes:
            students.extend(cls.students)

    else:
        teachers_class = (db.query(Class).filter(Class.id == class_id)
                          .filter(Class.teacher_id == user.id).first())
        if not teachers_class:
            raise HTTPException(status_code=404, detail="Class not found or access denied")
        students = teachers_class.students

    students = list({student.id: student for student in students}.values())
    if not students:
        raise HTTPException(status_code=404, detail="No students found in class")

    student_attentions = []
    for student in students:
        last_session = db.query(Session).filter(Session.student_id == student.id).order_by(Session.start_time.desc()).first()
        if last_session:
            attention_score = db.query(AttentionScore).filter(AttentionScore.session_id == last_session.id).order_by(AttentionScore.timestamp.desc()).first()
            if attention_score:
                student_attentions.append(attention_score.attention_level)

    if not student_attentions:
        raise HTTPException(status_code=404, detail="No attention scores found for students")

    average_attention = sum(student_attentions) / len(student_attentions)
    active_students = sum(1 for student in students if student.is_active)



    return {
        "class_id": 1,
        "total_students": 25,
        "active_students": 23,
        "average_attention": 0.78,
        "alerts_count": 3
    }


@router.get("/session/{session_id}/analytics", response_model=SessionAnalytics)
async def get_session_analytics(
    session_id: int,
    current_user: str = Depends(verify_token),
    db: dbSession = Depends(get_db)
):
    """Get detailed analytics for a specific session"""
    # TODO: Implement session analytics logic
    return {
        "session_id": session_id,
        "student_name": "Ahmet Yılmaz",
        "duration_minutes": 45,
        "average_attention": 0.82,
        "attention_peaks": [datetime.now()],
        "attention_drops": [datetime.now()],
        "total_distractions": 2,
        "engagement_score": 0.85
    }


@router.get("/alerts", response_model=List[Alert])
async def get_alerts(
    limit: int = 50,
    severity: Optional[str] = None,
    current_user: str = Depends(verify_token),
    db: dbSession = Depends(get_db)
):
    """Get recent alerts for teacher"""
    # TODO: Implement get alerts logic
    return [
        {
            "alert_id": 1,
            "student_id": 2,
            "student_name": "Ayşe Kaya",
            "alert_type": "low_attention",
            "message": "Dikkat seviyesi düşük - son 5 dakikada %45",
            "timestamp": datetime.now(),
            "severity": "medium"
        }
    ]


@router.post("/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(
    alert_id: int,
    current_user: str = Depends(verify_token),
    db: dbSession = Depends(get_db)
):
    """Mark an alert as acknowledged"""
    # TODO: Implement alert acknowledgment logic
    return {"message": "Alert acknowledged"}


@router.get("/analytics/summary")
async def get_analytics_summary(
    date_from: Optional[datetime] = None,
    date_to: Optional[datetime] = None,
    current_user: str = Depends(verify_token),
    db: dbSession = Depends(get_db)
):
    """Get summary analytics for teacher dashboard"""
    # TODO: Implement analytics summary logic
    return {
        "total_sessions": 45,
        "total_students": 25,
        "average_class_attention": 0.78,
        "most_attentive_time": "10:00-11:00",
        "common_distractions": ["phone", "looking_away"],
        "improvement_suggestions": [
            "En düşük dikkat saatleri 14:00-15:00 arasında",
            "Interaktif aktiviteler dikkat seviyesini %20 artırıyor"
        ]
    }


@router.get("/students/{student_id}/profile")
async def get_student_profile(
    student_id: int,
    current_user: str = Depends(verify_token),
    db: dbSession = Depends(get_db)
):
    """Get detailed profile and statistics for a specific student"""
    # TODO: Implement student profile logic
    return {
        "student_id": student_id,
        "name": "Ahmet Yılmaz",
        "total_sessions": 12,
        "average_attention": 0.82,
        "attention_trend": "improving",
        "best_performance_time": "09:00-10:00",
        "common_distractions": ["phone"],
        "recommendations": [
            "Sabah saatlerinde daha produktif",
            "Telefon kullanımını azaltmaya odaklanmalı"
        ]
    }
