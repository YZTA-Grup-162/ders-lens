"""
Configuration settings for DersLens Backend
"""

from typing import List, Optional, Union

from pydantic import AnyUrl, BaseSettings


class Settings(BaseSettings):
    """Application settings"""
    
    # Basic app settings
    APP_NAME: str = "DersLens"
    DEBUG: bool = True
    VERSION: str = "1.0.0"
    ENVIRONMENT: str = "development"
    
    # Security
    SECRET_KEY: str = "ders-lens-secret-key-change-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # Database
    DATABASE_URL: AnyUrl = "postgresql://postgres:password@localhost:5432/ders_lens"
    REDIS_URL: AnyUrl
    
    # CORS
    CORS_ORIGINS: Union[str, List[str]] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3002",
        "http://127.0.0.1:3002",
        "http://localhost:5173",
        "http://127.0.0.1:5173"
    ]
    
    # API Configuration
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    
    # AI Service
    AI_SERVICE_URL: str = "http://localhost:5000"
    
    # File uploads
    UPLOAD_DIR: str = "./uploads"
    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10MB
    
    # Frontend
    FRONTEND_URL: str = "http://localhost:3000"
    
    # Sentry
    SENTRY_DSN: Optional[str] = None

    class Config:
        env_file = ".env"


# Global settings instance
settings = Settings()
