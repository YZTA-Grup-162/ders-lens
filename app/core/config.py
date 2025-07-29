
import json
from functools import lru_cache
from typing import List, Optional
from pydantic import validator
from pydantic_settings import BaseSettings
class Settings(BaseSettings):
    app_name: str = "AttentionPulse"
    app_version: str = "1.0.0"
    debug: bool = False
    log_level: str = "INFO"
    api_v1_str: str = "/api/v1"
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: List[str] = ["http://localhost:3000", "http://localhost:5173"]
    database_url: str = "postgresql+asyncpg://user:password@localhost:5432/attentionpulse"
    database_pool_size: int = 10
    database_max_overflow: int = 20
    redis_url: str = "redis://localhost:6379/0"
    redis_stream_key: str = "predictions"
    redis_ttl: int = 3600
    secret_key: str = "change-this-super-secret-key-in-production"
    access_token_expire_minutes: int = 30
    algorithm: str = "HS256"
    model_path: str = "./models"
    onnx_model_path: str = "./models/onnx"
    max_batch_size: int = 8
    inference_timeout: int = 25
    face_detection_confidence: float = 0.7
    max_frame_width: int = 320
    max_frame_height: int = 240
    video_fps: int = 15
    sequence_length: int = 32
    data_retention_seconds: int = 86400
    store_frames: bool = False
    gdpr_compliance: bool = True
    anonymize_data: bool = True
    max_concurrent_sessions: int = 100
    worker_processes: int = 4
    enable_model_caching: bool = True
    dataset_root: str = "./datasets"
    daisee_path: str = "./datasets/daisee"
    mendeley_path: str = "./datasets/mendeley"
    download_datasets: bool = False
    enable_metrics: bool = True
    metrics_port: int = 9090
    health_check_interval: int = 30
    kaggle_username: Optional[str] = None
    kaggle_key: Optional[str] = None
    frontend_url: str = "http://localhost:3000"
    websocket_path: str = "/ws"
    APP_NAME: str = "AttentionPulse"
    DEBUG: bool = True
    VERSION: str = "1.0.0"
    SECRET_KEY: str = "PRODUCTION_SECRET_KEY"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    DATABASE_URL: str = "sqlite:///./attention_pulse.db"
    ALLOWED_HOSTS: List[str] = ["*"]
    MODEL_PATH: str = "./models"
    OPENCV_CONFIDENCE_THRESHOLD: float = 0.5
    ATTENTION_THRESHOLD: float = 0.7
    VIDEO_FRAME_RATE: int = 1
    MAX_VIDEO_SIZE: int = 5 * 1024 * 1024
    UPLOAD_DIR: str = "./uploads"
    MAX_FILE_SIZE: int = 10 * 1024 * 1024
    @validator("cors_origins", pre=True)
    def assemble_cors_origins(cls, v):
        if isinstance(v, str):
            if not v or v.strip() == "":
                return ["http://localhost:3000", "http://localhost:5173"]
            if v.startswith("["):
                try:
                    return json.loads(v)
                except json.JSONDecodeError:
                    return [i.strip() for i in v.split(",")]
            return [i.strip() for i in v.split(",") if i.strip()]
        elif isinstance(v, list):
            return v
        else:
            return ["http://localhost:3000", "http://localhost:5173"]
    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "allow"
@lru_cache()
def get_settings() -> Settings:
    return Settings()
settings = get_settings()