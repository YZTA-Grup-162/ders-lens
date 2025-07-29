"""
Core configuration for AttentionPulse API
"""

import os
from typing import List

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings"""
    
    # Basic app settings
    APP_NAME: str = "AttentionPulse"
    DEBUG: bool = True
    VERSION: str = "1.0.0"
    ENVIRONMENT: str = "development"
    
    # Security
    SECRET_KEY: str = "PRODUCTION_SECRET_KEY" 
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # Database
    DATABASE_URL: str = "sqlite:///./attention_pulse.db"
    
    # CORS
    ALLOWED_HOSTS: List[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]
    CORS_ORIGINS: str = "[\"http://localhost:3000\", \"http://127.0.0.1:3000\", \"http://localhost:5432\"]"
    
    # API Configuration
    API_HOST: str = "0.0.0.0"
    API_PORT: str = "8000"
    
    # File uploads
    UPLOAD_DIR: str = "./uploads"
    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10MB
    MAX_FILE_SIZE_MB: str = "10"
    UPLOAD_FOLDER: str = "./uploads"
    
    # AI Model settings
    MODEL_PATH: str = "./models"
    OPENCV_CONFIDENCE_THRESHOLD: float = 0.5
    ATTENTION_THRESHOLD: float = 0.7
    FACE_DETECTION_CONFIDENCE: str = "0.5"
    
    # Video processing
    VIDEO_FRAME_RATE: int = 1  # Process 1 frame per second
    MAX_VIDEO_SIZE: int = 5 * 1024 * 1024  # 5MB
    
    # Frontend URL
    FRONTEND_URL: str = "http://localhost:3000"
    
    # Logging
    LOG_LEVEL: str = "INFO"
    
    class Config:
        env_file = ".env"
        extra = "allow"  # Allow extra fields from environment


# Global settings instance
settings = Settings()
