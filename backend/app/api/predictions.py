"""
AI Prediction API endpoints
"""
import io
import json
import time
import cv2
import numpy as np
from app.ai.working_inference import HighFidelityAttentionEngine
from app.core.database import AttentionScore
from app.core.database import Session as DBSession
from app.core.database import get_db
from app.models.schemas import (AttentionPrediction, EmotionPrediction,
                                FrameMetadata, PredictionRequest,
                                PredictionResponse)
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from PIL import Image
from sqlalchemy.orm import Session
from typing import Optional, Dict
router = APIRouter()
_inference_engine: Optional[HighFidelityAttentionEngine] = None
def get_inference_engine() -> HighFidelityAttentionEngine:
    global _inference_engine
    if _inference_engine is None:
        _inference_engine = HighFidelityAttentionEngine()
    return _inference_engine
@router.post("/predict/comprehensive", response_model=Dict)
async def predict_comprehensive_analysis(
    frame: UploadFile = File(...),
    include_gaze: bool = True,
    include_emotions: bool = True,
    include_engagement: bool = True,
    include_attention: bool = True,
    session_id: Optional[int] = None,
    db: Session = Depends(get_db),
    inference_engine: HighFidelityAttentionEngine = Depends(get_inference_engine)
):
    try:
        if not frame.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        contents = await frame.read()
        image = Image.open(io.BytesIO(contents))
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        start_time = time.time()
        prediction_result = await inference_engine.process_frame(image_cv)
        response = {
            "timestamp": time.time(),
            "processingTime": (time.time() - start_time) * 1000,
            "modelVersion": inference_engine.model_version
        }
        if include_gaze:
            response["gaze"] = {
                "x": prediction_result.get("gaze_x", 0.5),
                "y": prediction_result.get("gaze_y", 0.5),
                "confidence": prediction_result.get("gaze_confidence", 0.0),
                "direction": prediction_result.get("gaze_direction", "center"),
                "onScreen": prediction_result.get("gaze_on_screen", True)
            }
        if include_emotions:
            response["emotion"] = {
                "dominant": prediction_result["dominant_emotion"],
                "scores": prediction_result["emotion_scores"],
                "valence": prediction_result.get("valence", 0.0),
                "arousal": prediction_result.get("arousal", 0.0),
                "confidence": prediction_result["emotion_confidence"]
            }
        if include_attention:
            response["attention"] = {
                "score": prediction_result["attention_score"],
                "state": prediction_result.get("attention_state", "unknown"),
                "confidence": prediction_result["attention_confidence"],
                "focusRegions": prediction_result.get("focus_regions", [])
            }
        if include_engagement:
            response["engagement"] = {
                "level": prediction_result["engagement_score"],
                "category": prediction_result.get("engagement_category", "moderate"),
                "indicators": prediction_result.get("engagement_indicators", {
                    "headMovement": 0.5,
                    "eyeContact": 0.5,
                    "facialExpression": 0.5,
                    "posture": 0.5
                    })
            }
        response["face"] = {
            "detected": prediction_result.get("face_detected", False),
            "confidence": prediction_result.get("face_confidence", 0.0),
            "landmarks": prediction_result.get("face_landmarks", []),
            "headPose": prediction_result.get("head_pose", {
                "pitch": 0.0,
                "yaw": 0.0,
                "roll": 0.0
            }),
            "eyeAspectRatio": prediction_result.get("eye_aspect_ratio", 0.3),
            "mouthAspectRatio": prediction_result.get("mouth_aspect_ratio", 0.1)
        }
        if session_id and db:
            attention_record = AttentionScore(
                session_id=session_id,
                attention_level=prediction_result["attention_score"],
                engagement_level=prediction_result["engagement_score"],
                confidence=prediction_result["attention_confidence"],
                face_detected=prediction_result.get("face_detected", False),
                head_pose_x=prediction_result.get("head_pose", {}).get("pitch"),
                head_pose_y=prediction_result.get("head_pose", {}).get("yaw"),
                head_pose_z=prediction_result.get("head_pose", {}).get("roll"),
            )
            db.add(attention_record)
            db.commit()
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Comprehensive analysis failed: {str(e)}")
@router.post("/predict/frame", response_model=PredictionResponse)
async def predict_from_frame(
    frame: UploadFile = File(...),
    session_id: Optional[int] = None,
    db: Session = Depends(get_db),
    inference_engine: HighFidelityAttentionEngine = Depends(get_inference_engine)
):
    """
    Predict attention and emotion from uploaded frame.

    This endpoint accepts an image file and returns a prediction response
    containing attention and emotion scores.
    """
    try:
        if not frame.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        contents = await frame.read()
        image = Image.open(io.BytesIO(contents))
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        height, width = image_cv.shape[:2]
        metadata = FrameMetadata(
            width=width,
            height=height,
            timestamp=time.time(),
            format="BGR",
            channels=3
        )
        start_time = time.time()
        prediction_result = await inference_engine.process_frame(image_cv)
        inference_time = time.time() - start_time
        attention_pred = AttentionPrediction(
            attention_score=prediction_result["attention_score"],
            engagement_score=prediction_result["engagement_score"],
            distraction_level=prediction_result["distraction_level"],
            focus_regions=prediction_result.get("focus_regions", []),
            confidence=prediction_result["attention_confidence"]
        )
        emotion_pred = EmotionPrediction(
            dominant_emotion=prediction_result["dominant_emotion"],
            emotion_scores=prediction_result["emotion_scores"],
            valence=prediction_result.get("valence", 0.0),
            arousal=prediction_result.get("arousal", 0.0),
            confidence=prediction_result["emotion_confidence"]
        )
        response = PredictionResponse(
            attention=attention_pred,
            emotion=emotion_pred,
            metadata=metadata,
            processing_time_ms=inference_time * 1000,
            model_version=inference_engine.model_version,
            face_detected=prediction_result.get("face_detected", False)
        )
        if session_id and db:
            attention_record = AttentionScore(
                session_id=session_id,
                attention_level=attention_pred.attention_score,
                engagement_level=attention_pred.engagement_score,
                confidence=attention_pred.confidence,
                face_detected=response.face_detected,
                head_pose_x=prediction_result.get("head_pose", {}).get("x"),
                head_pose_y=prediction_result.get("head_pose", {}).get("y"),
                head_pose_z=prediction_result.get("head_pose", {}).get("z"),
            )
            db.add(attention_record)
            db.commit()
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
@router.post("/predict/batch")
async def predict_batch(
    request: PredictionRequest,
    db: Session = Depends(get_db),
    inference_engine: HighFidelityAttentionEngine = Depends(get_inference_engine)
):
    try:
        results = []
        for frame_data in request.frames:
            image_data = frame_data.data.split(',')[1] if ',' in frame_data.data else frame_data.data
            image_bytes = io.BytesIO(image_data.encode())
            image = Image.open(image_bytes)
            image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            prediction_result = await inference_engine.process_frame(image_cv)
            attention_pred = AttentionPrediction(
                attention_score=prediction_result["attention_score"],
                engagement_score=prediction_result["engagement_score"],
                distraction_level=prediction_result["distraction_level"],
                focus_regions=prediction_result.get("focus_regions", []),
                confidence=prediction_result["attention_confidence"]
            )
            emotion_pred = EmotionPrediction(
                dominant_emotion=prediction_result["dominant_emotion"],
                emotion_scores=prediction_result["emotion_scores"],
                valence=prediction_result.get("valence", 0.0),
                arousal=prediction_result.get("arousal", 0.0),
                confidence=prediction_result["emotion_confidence"]
            )
            metadata = FrameMetadata(
                width=image_cv.shape[1],
                height=image_cv.shape[0],
                timestamp=time.time(),
                format="BGR",
                channels=3
            )
            response = PredictionResponse(
                attention=attention_pred,
                emotion=emotion_pred,
                metadata=metadata,
                processing_time_ms=0,
                model_version=inference_engine.model_version,
                face_detected=prediction_result.get("face_detected", False)
            )
            results.append(response)
        return {"predictions": results, "batch_size": len(results)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")
@router.get("/model/info")
async def get_model_info(
    inference_engine: HighFidelityAttentionEngine = Depends(get_inference_engine)
):
    return {
        "model_version": inference_engine.model_version,
        "model_path": inference_engine.model_path,
        "input_shape": inference_engine.input_shape,
        "supported_emotions": ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"],
        "attention_classes": ["low", "medium", "high"],
        "privacy_compliant": True,
        "data_processing": "local_only"
    }
@router.post("/model/warm-up")
async def warm_up_model(
    inference_engine: HighFidelityAttentionEngine = Depends(get_inference_engine)
):
    try:
        dummy_frame = np.zeros((224, 224, 3), dtype=np.uint8)
        start_time = time.time()
        result = await inference_engine.process_frame(dummy_frame)
        warm_up_time = time.time() - start_time
        return {
            "status": "success",
            "warm_up_time_ms": warm_up_time * 1000,
            "model_ready": True
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model warm-up failed: {str(e)}")
@router.get("/health")
async def health_check(
    inference_engine: HighFidelityAttentionEngine = Depends(get_inference_engine)
):
    try:
        dummy_frame = np.zeros((224, 224, 3), dtype=np.uint8)
        await inference_engine.process_frame(dummy_frame)
        return {
            "status": "healthy",
            "model_loaded": True,
            "inference_ready": True,
            "timestamp": time.time()
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "model_loaded": False,
            "inference_ready": False,
            "error": str(e),
            "timestamp": time.time()
        }
@router.post("/emotional")
async def predict_emotional(image: UploadFile = File(...)):
    try:
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        image_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image_cv is None:
            raise HTTPException(status_code=400, detail="Invalid image format")
        inference_engine = get_inference_engine()
        prediction_result = await inference_engine.process_frame(image_cv)
        response = {
            "emotion": prediction_result["dominant_emotion"],
            "scores": prediction_result["emotion_scores"],
            "confidence": prediction_result["emotion_confidence"],
            "face_detected": prediction_result.get("face_detected", False),
            "timestamp": time.time()
        }
        return response
    except Exception as e:
        print(f"Error in emotional prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))