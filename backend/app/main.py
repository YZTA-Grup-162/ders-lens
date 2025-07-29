"""
AttentionPulse Backend - FastAPI Application
"""

import base64
import time
from contextlib import asynccontextmanager
from typing import Any, Dict

import httpx
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api import auth, gaze, student, teacher
from app.core.config import settings
from app.core.database import Base, engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    print("**AttentionPulse API başlatılıyor...**")

    # Create tables
    # Base.metadata.create_all(bind=engine)
    
    yield
    
    # Shutdown
    print("📴 AttentionPulse API kapatılıyor...")


app = FastAPI(
    title="AttentionPulse API",
    description="AI Destekli Sınıf Etkileşim Analizi Sistemi",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_HOSTS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Routes
app.include_router(auth.router, prefix="/api/auth", tags=["authentication"])
app.include_router(student.router, prefix="/api/student", tags=["student"])
app.include_router(teacher.router, prefix="/api/teacher", tags=["teacher"])
app.include_router(gaze.router, prefix="/api/gaze", tags=["gaze_detection"])


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "AttentionPulse API",
        "version": "1.0.0",
        "status": "healthy",
        "docs": "/docs"
    }


@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "database": "connected",  # TODO: Add actual DB check
        "ai_model": "loaded"      # TODO: Add actual model check
    }


@app.get("/api/enhanced-status")
async def enhanced_system_status():
    """Check enhanced system status and model availability"""
    try:
        # Import the enhanced components
        from app.ai.enhanced_inference import EnhancedAttentionEngine
        from app.ai.unified_model_manager import UnifiedModelManager

        # Test model manager
        model_manager = UnifiedModelManager()
        loaded_models = []
        
        for model_name, model_info in model_manager.models.items():
            if model_info.get('loaded', False):
                accuracy = model_info.get('accuracy', {})
                loaded_models.append({
                    "name": model_name,
                    "type": model_info.get('type', 'unknown'),
                    "accuracy": accuracy,
                    "status": "loaded"
                })
        
        # Test enhanced engine
        try:
            engine = EnhancedAttentionEngine(models_dir="models")
            engine_status = "operational"
        except Exception as e:
            engine_status = f"error: {str(e)}"
        
        return {
            "status": "enhanced_system_ready",
            "timestamp": time.time(),
            "models": {
                "total_loaded": len(loaded_models),
                "available_models": loaded_models
            },
            "enhanced_engine": engine_status,
            "features": {
                "high_accuracy_gaze": any("gaze" in m["name"] for m in loaded_models),
                "emotion_recognition": any("emotion" in m["name"] for m in loaded_models),
                "attention_detection": len(loaded_models) > 0,
                "real_time_processing": engine_status == "operational"
            },
            "ready_for_testing": len(loaded_models) >= 1 and engine_status == "operational"
        }
        
    except Exception as e:
        return {
            "status": "enhanced_system_error",
            "error": str(e),
            "message": "Enhanced system components not available",
            "fallback_available": True
        }


@app.post("/api/analyze")
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
                "http://localhost:8003/analyze/frame",
                json=request_data,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code, 
                    detail=f"AI service error: {response.text}"
                )
            
            ai_result = response.json()
            
            # Transform AI service response to frontend format
            transformed_result = transform_ai_response(ai_result)
            
            return JSONResponse(content=transformed_result)
            
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="AI service timeout")
    except httpx.ConnectError:
        raise HTTPException(status_code=503, detail="AI service unavailable")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


def transform_ai_response(ai_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Transform AI service response to frontend-expected format
    """
    try:
        # Extract the nested data from AI service response
        if "data" in ai_result:
            data = ai_result["data"]
        else:
            data = ai_result
        
        # Extract emotion data
        emotion_data = data.get("emotion", {})
        if isinstance(emotion_data, dict):
            emotion_name = emotion_data.get("dominant", "neutral")
            emotion_confidence = emotion_data.get("confidence", 0.0)
        else:
            emotion_name = "neutral"
            emotion_confidence = 0.0
        
        # Map Turkish emotion names to English for consistency
        emotion_map = {
            "mutlu": "happy",
            "üzgün": "sad", 
            "sinirli": "angry",
            "korkmuş": "fear",
            "şaşırmış": "surprise",
            "iğrenmiş": "disgust",
            "nötr": "neutral",
            "happiness": "happy",
            "sadness": "sad",
            "anger": "angry",
            "fear": "fear",
            "surprise": "surprise",
            "disgust": "disgust",
            "neutral": "neutral"
        }
        
        # Convert emotion to English if it's in Turkish
        mapped_emotion = emotion_map.get(emotion_name.lower(), emotion_name)
        
        # Extract attention data
        attention_data = data.get("attention", {})
        if isinstance(attention_data, dict):
            attention_level = attention_data.get("score", 0.0)
        else:
            attention_level = 0.0
        
        # Extract engagement data
        engagement_data = data.get("engagement", {})
        if isinstance(engagement_data, dict):
            engagement_level = engagement_data.get("score", 0.0)
        else:
            engagement_level = 0.0
        
        # Extract gaze data
        gaze_data = data.get("gaze", {})
        if isinstance(gaze_data, dict):
            gaze_direction = gaze_data.get("direction", {})
            if isinstance(gaze_direction, dict):
                gaze_x = gaze_direction.get("x", 0.0)
                gaze_y = gaze_direction.get("y", 0.0)
                # Convert gaze coordinates to direction
                if abs(gaze_x) < 0.2 and abs(gaze_y) < 0.2:
                    gaze_direction_str = "merkez"
                elif gaze_x > 0.2:
                    gaze_direction_str = "sağ"
                elif gaze_x < -0.2:
                    gaze_direction_str = "sol"
                elif gaze_y > 0.2:
                    gaze_direction_str = "aşağı"
                else:
                    gaze_direction_str = "yukarı"
            else:
                gaze_direction_str = "merkez"
        else:
            gaze_direction_str = "merkez"
        
        # Check if face was detected
        metadata = data.get("metadata", {})
        face_detected = metadata.get("faceDetected", False)
        face_count = metadata.get("face_count", 0)
        
        return {
            "attention": attention_level,
            "engagement": engagement_level,
            "emotion": mapped_emotion,
            "emotionConfidence": emotion_confidence,
            "gazeDirection": gaze_direction_str,
            "faceDetected": face_detected,
            "timestamp": int(metadata.get("processingTime", 0)),
            "debug": {
                "original_emotion": emotion_name,
                "face_count": face_count,
                "processing_time": metadata.get("processingTime", 0),
                "success": ai_result.get("success", False)
            }
        }
        
    except Exception as e:
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


# WebSocket for real-time communication
@app.websocket("/ws/student/{session_id}")
async def websocket_student_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for student real-time updates"""
    await websocket.accept()
    try:
        while True:
            # TODO: Implement real-time student data processing
            data = await websocket.receive_text()
            await websocket.send_text(f"Echo: {data}")
    except Exception as e:
        print(f"WebSocket connection error: {e}")
    finally:
        await websocket.close()


@app.websocket("/ws/teacher/{class_id}")
async def websocket_teacher_endpoint(websocket: WebSocket, class_id: str):
    """WebSocket endpoint for teacher real-time monitoring"""
    await websocket.accept()
    try:
        while True:
            # TODO: Implement real-time teacher dashboard updates
            data = await websocket.receive_text()
            await websocket.send_text(f"Teacher update: {data}")
    except Exception as e:
        print(f"WebSocket connection error: {e}")
    finally:
        await websocket.close()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
