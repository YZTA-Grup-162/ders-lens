
import json
from functools import lru_cache
from typing import List, Optional

from pydantic import validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # DersLens Application Settings
    app_name: str = "DersLens"
    app_version: str = "1.0.0"
    debug: bool = True
    log_level: str = "INFO"
    api_v1_str: str = "/api/v1"
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: List[str] = ["http://localhost:3001", "http://localhost:3000", "http://localhost:5173"]
    
    # Database Configuration
    database_url: str = "sqlite:///./derslens.db"
    database_pool_size: int = 10
    database_max_overflow: int = 20
    
    # Redis Configuration
    redis_url: str = "redis://localhost:6379/0"
    redis_stream_key: str = "predictions"
    redis_ttl: int = 3600
    
    # Security Settings
    secret_key: str = "derslens-super-secret-key-change-in-production"
    access_token_expire_minutes: int = 30
    algorithm: str = "HS256"
    
    # Model Configuration
    model_path: str = "./models"
    onnx_model_path: str = "./models/onnx"
    max_batch_size: int = 8
    inference_timeout: int = 25
    face_detection_confidence: float = 0.7
    
    # Video Processing
    max_frame_width: int = 320
    max_frame_height: int = 240
    video_fps: int = 15
    sequence_length: int = 32
    
    # Data Management
    data_retention_seconds: int = 86400
    store_frames: bool = False
    gdpr_compliance: bool = True
    anonymize_data: bool = True
    
    # Performance Settings
    max_concurrent_sessions: int = 100
    worker_processes: int = 4
    enable_model_caching: bool = True
    
    # Dataset Configuration
    dataset_root: str = "./datasets"
    daisee_path: str = "./datasets/daisee"
    mendeley_path: str = "./datasets/mendeley"
    download_datasets: bool = False
    
    # Monitoring
    enable_metrics: bool = True
    metrics_port: int = 9090
    health_check_interval: int = 30
    
    # External Services
    kaggle_username: Optional[str] = None
    kaggle_key: Optional[str] = None
    frontend_url: str = "http://localhost:3001"
    websocket_path: str = "/ws"
    
    # Upload Settings
    upload_dir: str = "./uploads"
    max_file_size: int = 10 * 1024 * 1024
    allowed_hosts: List[str] = ["*"]
    
    # Detection Thresholds
    opencv_confidence_threshold: float = 0.5
    attention_threshold: float = 0.7
    video_frame_rate: int = 1
    max_video_size: int = 5 * 1024 * 1024
    @validator("cors_origins", pre=True)
    def assemble_cors_origins(cls, v):
        if isinstance(v, str):
            if not v or v.strip() == "":
                return ["http://localhost:3001", "http://localhost:3000", "http://localhost:5173"]
            if v.startswith("["):
                try:
                    return json.loads(v)
                except json.JSONDecodeError:
                    return [i.strip() for i in v.split(",")]
            return [i.strip() for i in v.split(",") if i.strip()]
        elif isinstance(v, list):
            return v
        else:
            return ["http://localhost:3001", "http://localhost:3000", "http://localhost:5173"]

    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "forbid"  # Prevent extra fields that cause Pydantic errors

@lru_cache()
def get_settings() -> Settings:
    return Settings()

settings = get_settings()