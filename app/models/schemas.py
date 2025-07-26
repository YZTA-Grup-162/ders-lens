
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Union
from pydantic import BaseModel, Field, validator
class EngagementLevel(int, Enum):
    VERY_LOW = 0
    LOW = 1
    HIGH = 2
    VERY_HIGH = 3
class EmotionClass(int, Enum):
    BOREDOM = 0
    CONFUSION = 1
    ENGAGEMENT = 2
    FRUSTRATION = 3
    HAPPY = 4
    SADNESS = 5
    SURPRISE = 6
    DISGUST = 7
    FEAR = 8
    ANGER = 9
    NEUTRAL = 10
class AttentionState(str, Enum):
    ATTENTIVE = "attentive"
    INATTENTIVE = "inattentive"
class VideoFrameRequest(BaseModel):
    frame_data: str = Field(..., description="Base64 encoded image data")
    timestamp: datetime = Field(default_factory=datetime.now)
    student_id: Optional[str] = Field(None, description="Student identifier")
    session_id: Optional[str] = Field(None, description="Session identifier")
class BatchFrameRequest(BaseModel):
    frames: List[VideoFrameRequest] = Field(..., max_items=32)
    sequence_id: str = Field(..., description="Sequence identifier for temporal analysis")
class WebRTCFrameData(BaseModel):
    data: str = Field(..., description="Base64 encoded frame")
    width: int = Field(..., ge=160, le=1920)
    height: int = Field(..., ge=120, le=1080)
    timestamp: float = Field(..., description="Client timestamp")
class AttentionPrediction(BaseModel):
    attention_score: float = Field(..., ge=0.0, le=1.0, description="Attention probability [0,1]")
    attention_state: AttentionState = Field(..., description="Binary attention classification")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Model confidence")
class EngagementPrediction(BaseModel):
    engagement_level: EngagementLevel = Field(..., description="4-level engagement classification")
    engagement_probabilities: Dict[str, float] = Field(..., description="Probability distribution")
class EmotionPrediction(BaseModel):
    emotion_class: EmotionClass = Field(..., description="Primary emotion")
    emotion_probabilities: Dict[str, float] = Field(..., description="Emotion probability distribution")
class FaceFeatures(BaseModel):
    face_detected: bool = Field(..., description="Whether face was detected")
    face_confidence: float = Field(0.0, ge=0.0, le=1.0, description="Face detection confidence")
    face_bbox: Optional[List[int]] = Field(None, description="Bounding box [x, y, w, h]")
    landmarks: Optional[List[float]] = Field(None, description="Facial landmarks")
    pose_features: Optional[Dict[str, float]] = Field(None, description="Head pose features")
    phone_usage_detected: bool = Field(False, description="Phone usage detection")
class PredictionResult(BaseModel):
    timestamp: datetime = Field(default_factory=datetime.now)
    frame_id: str = Field(..., description="Unique frame identifier")
    student_id: Optional[str] = None
    session_id: Optional[str] = None
    attention: AttentionPrediction
    engagement: EngagementPrediction
    emotion: EmotionPrediction
    face_features: FaceFeatures
    processing_time_ms: float = Field(..., description="Inference time in milliseconds")
    model_version: str = Field(..., description="Model version used")
class SequencePrediction(BaseModel):
    sequence_id: str = Field(..., description="Sequence identifier")
    frame_predictions: List[PredictionResult] = Field(..., description="Individual frame predictions")
    temporal_attention_trend: List[float] = Field(..., description="Attention trend over sequence")
    average_attention: float = Field(..., ge=0.0, le=1.0)
    attention_variance: float = Field(..., ge=0.0)
    engagement_stability: float = Field(..., ge=0.0, le=1.0)
class StudentCard(BaseModel):
    student_id: str
    name: Optional[str] = None
    current_attention: float = Field(..., ge=0.0, le=1.0)
    attention_trend: List[float] = Field(..., description="Last 60 seconds")
    current_engagement: EngagementLevel
    current_emotion: EmotionClass
    last_updated: datetime
    session_duration: int = Field(..., description="Session duration in seconds")
class ClassroomDashboard(BaseModel):
    class_id: str
    timestamp: datetime = Field(default_factory=datetime.now)
    active_students: int
    students: List[StudentCard]
    class_average_attention: float = Field(..., ge=0.0, le=1.0)
    alerts: List[str] = Field(default_factory=list, description="Low attention alerts")
class SessionSummary(BaseModel):
    student_id: str
    session_id: str
    start_time: datetime
    end_time: datetime
    duration_minutes: int
    average_attention: float = Field(..., ge=0.0, le=1.0)
    attention_distribution: Dict[str, float]
    engagement_distribution: Dict[str, int]
    emotion_distribution: Dict[str, int]
    most_attentive_period: Optional[str] = None
    least_attentive_period: Optional[str] = None
    attention_drops: int = Field(0, description="Number of significant attention drops")
    phone_usage_incidents: int = Field(0, description="Phone usage detection count")
    recommendations: List[str] = Field(default_factory=list)
class ClassSessionReport(BaseModel):
    class_id: str
    session_id: str
    teacher_id: Optional[str] = None
    start_time: datetime
    end_time: datetime
    total_students: int
    average_class_attention: float
    engagement_trends: Dict[str, List[float]]
    student_summaries: List[SessionSummary]
    peak_engagement_time: Optional[str] = None
    lowest_engagement_time: Optional[str] = None
    class_recommendations: List[str] = Field(default_factory=list)
class PredictionError(BaseModel):
    error_code: str
    error_message: str
    timestamp: datetime = Field(default_factory=datetime.now)
    frame_id: Optional[str] = None
    details: Optional[Dict] = None
class ModelHealth(BaseModel):
    model_name: str
    status: str = Field(..., pattern="^(healthy|degraded|unhealthy)$")
    last_inference_time: Optional[datetime] = None
    average_inference_time_ms: float
    error_rate: float = Field(..., ge=0.0, le=1.0)
    memory_usage_mb: float
class SystemHealth(BaseModel):
    timestamp: datetime = Field(default_factory=datetime.now)
    status: str = Field(..., pattern="^(healthy|degraded|unhealthy)$")
    active_sessions: int
    cpu_usage: float = Field(..., ge=0.0, le=100.0)
    memory_usage: float = Field(..., ge=0.0, le=100.0)
    model_health: List[ModelHealth]
class ModelConfig(BaseModel):
    model_type: str = Field(..., pattern="^(mobilevit|efficientnet|fusion)$")
    version: str
    input_resolution: tuple = Field((224, 224))
    sequence_length: int = Field(32, ge=1, le=128)
    batch_size: int = Field(8, ge=1, le=64)
    confidence_threshold: float = Field(0.7, ge=0.0, le=1.0)
    onnx_optimized: bool = True
class ConsentRecord(BaseModel):
    user_id: str
    consent_given: bool
    consent_timestamp: datetime
    consent_version: str
    data_retention_days: int = Field(30, ge=1, le=365)
    anonymization_enabled: bool = True
class DataRetentionPolicy(BaseModel):
    retention_period_days: int = Field(30, ge=1, le=365)
    auto_deletion_enabled: bool = True
    anonymization_after_days: int = Field(7, ge=1, le=30)
    backup_retention_days: int = Field(90, ge=30, le=365)
@validator("attention_score", "confidence", pre=True)
def validate_probability(cls, v):
    if not isinstance(v, (int, float)):
        raise ValueError("Must be a number")
    if not 0.0 <= v <= 1.0:
        raise ValueError("Must be between 0.0 and 1.0")
    return float(v)
class FrameMetadata(BaseModel):
    width: int
    height: int
    timestamp: float
    format: str
    channels: int
#from pydantic import BaseModel
#from typing import List
class FrameData(BaseModel):
    data: str
class PredictionRequest(BaseModel):
    frames: List[FrameData]
class PredictionResponse(BaseModel):
    predictions: List[PredictionResult]