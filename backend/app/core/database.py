"""
Database configuration and models
"""

from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float, Boolean, Text, ForeignKey, Table
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.sql import func

from app.core.config import settings

# Database engine
engine = create_engine(settings.DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    """Database dependency"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

class_students = Table(
        "class_students",
        Base.metadata,
        Column("class_id", Integer, ForeignKey("classes.id")),
        Column("student_id", Integer, ForeignKey("users.id"))
    )


# Database Models
class User(Base):
    """User model for both students and teachers"""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    username = Column(String, unique=True, index=True, nullable=False)
    full_name = Column(String, nullable=False)
    hashed_password = Column(String, nullable=False)
    role = Column(String, nullable=False)  # 'student' or 'teacher'
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    classes = relationship("Class", secondary=class_students, back_populates="students")
    teaching_classes = relationship(
        "Class",
        back_populates="teacher",
        foreign_keys="Class.teacher_id"
    )

class Class(Base):
    """Class model for grouping students and teachers"""
    __tablename__ = "classes"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    teacher_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    teacher = relationship(
        "User",
        back_populates="teaching_classes",
        foreign_keys=[teacher_id]
    )
    students = relationship("User", secondary="class_students", back_populates="classes")


class Session(Base):
    """Learning session model"""
    __tablename__ = "sessions"
    
    id = Column(Integer, primary_key=True, index=True)
    student_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    teacher_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    session_name = Column(String, nullable=False)
    start_time = Column(DateTime(timezone=True), server_default=func.now())
    end_time = Column(DateTime(timezone=True), nullable=True)
    is_active = Column(Boolean, default=True)
    
    # Relationships
    student = relationship("User", foreign_keys=[student_id])
    teacher = relationship("User", foreign_keys=[teacher_id])


class AttentionScore(Base):
    """Attention score tracking"""
    __tablename__ = "attention_scores"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("sessions.id"), nullable=False)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    attention_level = Column(Float, nullable=False)  # 0.0 to 1.0
    engagement_level = Column(Float, nullable=True)
    distraction_type = Column(String, nullable=True)  # 'phone', 'looking_away', etc.
    confidence = Column(Float, nullable=False)  # Model confidence
    
    # Raw data
    face_detected = Column(Boolean, default=False)
    head_pose_x = Column(Float, nullable=True)
    head_pose_y = Column(Float, nullable=True)
    head_pose_z = Column(Float, nullable=True)
    
    # Relationships
    session = relationship("Session")


class Feedback(Base):
    """Personalized feedback for students"""
    __tablename__ = "feedback"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("sessions.id"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    feedback_type = Column(String, nullable=False)  # 'attention', 'engagement', 'general'
    message = Column(Text, nullable=False)
    suggestions = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    session = relationship("Session")
    user = relationship("User")
