"""
AI Service Proxy - Routes requests to the AI service on port 8002
"""

import httpx
import base64
from typing import Dict, Any
from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

AI_SERVICE_URL = "http://localhost:8002"

@router.post("/analyze")
async def analyze_frame_proxy(frame: UploadFile = File(...)):
    """
    Proxy analyze requests to the AI service
    """
    try:
        # Read the uploaded image
        image_data = await frame.read()
        
        # Convert to base64 for AI service
        image_base64 = base64.b64encode(image_data).decode('utf-8')
        image_data_url = f"data:image/jpeg;base64,{image_base64}"
        
        # Prepare request for AI service
        request_data = {
            "image": image_data_url
        }
        
        # Make request to AI service
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{AI_SERVICE_URL}/analyze",
                json=request_data,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code != 200:
                logger.error(f"AI service error: {response.status_code} - {response.text}")
                raise HTTPException(
                    status_code=response.status_code, 
                    detail=f"AI service error: {response.text}"
                )
            
            ai_result = response.json()
            
            # Transform AI service response to frontend format
            transformed_result = transform_ai_response(ai_result)
            
            return JSONResponse(content=transformed_result)
            
    except httpx.TimeoutException:
        logger.error("AI service timeout")
        raise HTTPException(status_code=504, detail="AI service timeout")
    except httpx.ConnectError:
        logger.error("Cannot connect to AI service")
        raise HTTPException(status_code=503, detail="AI service unavailable")
    except Exception as e:
        logger.error(f"Analysis proxy error: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


def transform_ai_response(ai_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Transform AI service response to frontend-expected format
    """
    try:
        # Extract emotion data
        emotion_data = ai_result.get("emotion", {})
        emotion_name = emotion_data.get("emotion", "neutral")
        emotion_confidence = emotion_data.get("confidence", 0.0)
        
        # Map Turkish emotion names to English for consistency
        emotion_map = {
            "mutlu": "happy",
            "üzgün": "sad", 
            "sinirli": "angry",
            "korkmuş": "fear",
            "şaşırmış": "surprise",
            "iğrenmiş": "disgust",
            "nötr": "neutral"
        }
        
        # Convert emotion to English if it's in Turkish
        mapped_emotion = emotion_map.get(emotion_name.lower(), emotion_name)
        
        # Extract attention data
        attention_data = ai_result.get("attention", {})
        attention_level = attention_data.get("score", 0.0)
        
        # Extract engagement data
        engagement_data = ai_result.get("engagement", {})
        engagement_level = engagement_data.get("score", 0.0)
        
        # Extract gaze data
        gaze_data = ai_result.get("gaze", {})
        gaze_direction = gaze_data.get("gaze_direction", "merkez")
        
        # Check if face was detected
        face_detected = ai_result.get("face_count", 0) > 0
        
        return {
            "attention": attention_level,
            "engagement": engagement_level,
            "emotion": mapped_emotion,
            "emotionConfidence": emotion_confidence,
            "gazeDirection": gaze_direction,
            "faceDetected": face_detected,
            "timestamp": int(ai_result.get("processing_time", 0) * 1000),  # Convert to ms
            "debug": {
                "original_emotion": emotion_name,
                "face_count": ai_result.get("face_count", 0),
                "processing_time": ai_result.get("processing_time", 0)
            }
        }
        
    except Exception as e:
        logger.error(f"Response transformation error: {e}")
        return {
            "attention": 0.0,
            "engagement": 0.0,
            "emotion": "neutral",
            "emotionConfidence": 0.0,
            "gazeDirection": "merkez",
            "faceDetected": False,
            "timestamp": 0,
            "error": str(e)
        }
