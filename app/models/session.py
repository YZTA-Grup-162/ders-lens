"""
Database models for session management
"""
from datetime import datetime
from sqlalchemy import (Boolean, Column, DateTime, Float, ForeignKey, Integer,
                        String, Text)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
Base = declarative_base()
class SessionData(Base):
    __tablename__ = "sessions"
    id = Column(Integer, primary_key=True, index=True)
    session_name = Column(String, nullable=False)
    teacher_id = Column(Integer, ForeignKey("users.id"))
    start_time = Column(DateTime, default=datetime.utcnow)
    end_time = Column(DateTime, nullable=True)
    is_active = Column(Boolean, default=True)
    teacher = relationship("User", back_populates="sessions")
    attention_records = relationship("AttentionRecord", back_populates="session")
class AttentionRecord(Base):
    __tablename__ = "attention_records"
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("sessions.id"))
    student_id = Column(Integer, ForeignKey("users.id"))
    timestamp = Column(DateTime, default=datetime.utcnow)
    attention_level = Column(Integer)
    confidence = Column(Float)
    distraction_type = Column(String)
    face_detected = Column(Boolean)
    head_pose_pitch = Column(Float, nullable=True)
    head_pose_yaw = Column(Float, nullable=True)
    head_pose_roll = Column(Float, nullable=True)
    face_area = Column(Float, nullable=True)
    face_center_x = Column(Float, nullable=True)
    face_center_y = Column(Float, nullable=True)
    num_faces = Column(Integer, default=0)
    num_hands = Column(Integer, default=0)
    phone_detected = Column(Boolean, default=False)
    session = relationship("SessionData", back_populates="attention_records")
    student = relationship("User", back_populates="attention_records")