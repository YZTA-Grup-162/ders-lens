"""
Gaze Detection API for DersLens
Real-time gaze estimation and attention monitoring endpoints
"""

import base64
import io
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

from ..ai.gaze_detector import (DersLensGazeDetector,
                                create_derslens_gaze_detector)
from ..ai.gaze_trainer import train_derslens_gaze_model

logger = logging.getLogger(__name__)

router = APIRouter()

# Global detector instance (initialized lazily)
_gaze_detector: Optional[DersLensGazeDetector] = None

def get_gaze_detector() -> Optional[DersLensGazeDetector]:
    """Get or create the global gaze detector instance"""
    global _gaze_detector
    
    if _gaze_detector is None:
        try:
            _gaze_detector = create_derslens_gaze_detector()
            if _gaze_detector:
                logger.info("✅ Gaze detector initialized")
            else:
                logger.warning("⚠️ Gaze detector created without model")
        except Exception as e:
            logger.error(f"Failed to create gaze detector: {e}")
            return None
    
    return _gaze_detector

@router.get("/status")
async def get_gaze_detection_status():
    """Get the status of gaze detection system"""
    detector = get_gaze_detector()
    
    if detector is None:
        return {
            "status": "error",
            "message": "Gaze detector not available",
            "model_loaded": False
        }
    
    stats = detector.get_performance_stats()
    
    return {
        "status": "ready",
        "model_loaded": detector.model is not None,
        "model_info": detector.model_info,
        "performance_stats": stats,
        "device": str(detector.device)
    }

@router.post("/detect")
async def detect_gaze_from_image(file: UploadFile = File(...)):
    """
    Detect gaze direction from uploaded image
    
    Args:
        file: Image file (JPEG, PNG, etc.)
        
    Returns:
        Gaze detection results
    """
    detector = get_gaze_detector()
    if detector is None:
        raise HTTPException(status_code=503, detail="Gaze detector not available")
    
    if detector.model is None:
        raise HTTPException(status_code=503, detail="Gaze model not loaded")
    
    try:
        # Read uploaded image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Convert to OpenCV format
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Process frame
        results = detector.process_frame(frame)
        
        # Add analysis summary
        face_count = len(results['faces'])
        attention_zones = [face['attention_zone'] for face in results['faces']]
        attention_summary = {
            'total_faces': face_count,
            'focused_count': attention_zones.count('focused'),
            'attentive_count': attention_zones.count('attentive'),
            'distracted_count': attention_zones.count('distracted'),
            'off_task_count': attention_zones.count('off_task')
        }
        
        results['attention_summary'] = attention_summary
        
        return results
        
    except Exception as e:
        logger.error(f"Gaze detection failed: {e}")
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")

@router.post("/detect_base64")
async def detect_gaze_from_base64(data: Dict[str, str]):
    """
    Detect gaze direction from base64 encoded image
    
    Args:
        data: Dictionary with 'image' key containing base64 encoded image
        
    Returns:
        Gaze detection results
    """
    detector = get_gaze_detector()
    if detector is None:
        raise HTTPException(status_code=503, detail="Gaze detector not available")
    
    if detector.model is None:
        raise HTTPException(status_code=503, detail="Gaze model not loaded")
    
    try:
        # Decode base64 image
        image_data = data.get('image', '')
        if not image_data:
            raise HTTPException(status_code=400, detail="No image data provided")
        
        # Remove data URL prefix if present
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        # Decode base64
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to OpenCV format
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Process frame
        results = detector.process_frame(frame)
        
        # Add analysis summary
        face_count = len(results['faces'])
        attention_zones = [face['attention_zone'] for face in results['faces']]
        attention_summary = {
            'total_faces': face_count,
            'focused_count': attention_zones.count('focused'),
            'attentive_count': attention_zones.count('attentive'),
            'distracted_count': attention_zones.count('distracted'),
            'off_task_count': attention_zones.count('off_task')
        }
        
        results['attention_summary'] = attention_summary
        
        return results
        
    except Exception as e:
        logger.error(f"Gaze detection failed: {e}")
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")

@router.post("/analyze_classroom")
async def analyze_classroom_attention(file: UploadFile = File(...)):
    """
    Analyze classroom attention levels from image
    
    Args:
        file: Classroom image file
        
    Returns:
        Detailed classroom attention analysis
    """
    detector = get_gaze_detector()
    if detector is None:
        raise HTTPException(status_code=503, detail="Gaze detector not available")
    
    if detector.model is None:
        raise HTTPException(status_code=503, detail="Gaze model not loaded")
    
    try:
        # Read uploaded image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Process frame
        results = detector.process_frame(frame)
        
        # Detailed analysis
        faces = results['faces']
        total_faces = len(faces)
        
        if total_faces == 0:
            return {
                "status": "no_faces_detected",
                "total_students": 0,
                "attention_analysis": {},
                "recommendations": ["No students detected in the image"]
            }
        
        # Calculate attention metrics
        attention_zones = [face['attention_zone'] for face in faces]
        focused_count = attention_zones.count('focused')
        attentive_count = attention_zones.count('attentive')
        distracted_count = attention_zones.count('distracted')
        off_task_count = attention_zones.count('off_task')
        
        # Calculate percentages
        focused_pct = (focused_count / total_faces) * 100
        attentive_pct = (attentive_count / total_faces) * 100
        distracted_pct = (distracted_count / total_faces) * 100
        off_task_pct = (off_task_count / total_faces) * 100
        
        # Overall attention score (focused = 100%, attentive = 75%, distracted = 25%, off_task = 0%)
        attention_score = (focused_count * 100 + attentive_count * 75 + distracted_count * 25) / total_faces
        
        # Classify overall classroom attention
        if attention_score >= 85:
            attention_level = "excellent"
            attention_emoji = "🟢"
        elif attention_score >= 70:
            attention_level = "good"
            attention_emoji = "🟡"
        elif attention_score >= 50:
            attention_level = "moderate"
            attention_emoji = "🟠"
        else:
            attention_level = "poor"
            attention_emoji = "🔴"
        
        # Generate recommendations
        recommendations = []
        if focused_pct < 50:
            recommendations.append("Consider using more engaging teaching methods")
        if distracted_pct > 30:
            recommendations.append("Check for external distractions in the classroom")
        if off_task_pct > 20:
            recommendations.append("Students may need a break or change of activity")
        if focused_pct > 80:
            recommendations.append("Excellent engagement! Continue current approach")
        
        # Detailed gaze analysis
        gaze_directions = []
        for face in faces:
            if face['gaze']:
                yaw = face['gaze']['yaw']
                pitch = face['gaze']['pitch']
                gaze_directions.append({'yaw': yaw, 'pitch': pitch})
        
        # Calculate average gaze direction
        avg_yaw = np.mean([g['yaw'] for g in gaze_directions]) if gaze_directions else 0
        avg_pitch = np.mean([g['pitch'] for g in gaze_directions]) if gaze_directions else 0
        
        return {
            "status": "success",
            "total_students": total_faces,
            "attention_analysis": {
                "overall_score": round(attention_score, 1),
                "attention_level": attention_level,
                "attention_emoji": attention_emoji,
                "distribution": {
                    "focused": {"count": focused_count, "percentage": round(focused_pct, 1)},
                    "attentive": {"count": attentive_count, "percentage": round(attentive_pct, 1)},
                    "distracted": {"count": distracted_count, "percentage": round(distracted_pct, 1)},
                    "off_task": {"count": off_task_count, "percentage": round(off_task_pct, 1)}
                },
                "average_gaze": {
                    "yaw": round(avg_yaw, 2),
                    "pitch": round(avg_pitch, 2)
                }
            },
            "individual_students": [
                {
                    "student_id": i + 1,
                    "attention_zone": face['attention_zone'],
                    "gaze": face['gaze'],
                    "bbox": face['bbox']
                }
                for i, face in enumerate(faces)
            ],
            "recommendations": recommendations,
            "processing_info": {
                "processing_time_ms": round(results['processing_time'] * 1000, 2),
                "model_info": detector.model_info
            }
        }
        
    except Exception as e:
        logger.error(f"Classroom analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@router.get("/model/info")
async def get_model_info():
    """Get information about the loaded gaze model"""
    detector = get_gaze_detector()
    if detector is None:
        raise HTTPException(status_code=503, detail="Gaze detector not available")
    
    if detector.model is None:
        return {
            "model_loaded": False,
            "message": "No model loaded"
        }
    
    return {
        "model_loaded": True,
        "model_info": detector.model_info,
        "performance_stats": detector.get_performance_stats()
    }

@router.post("/train")
async def start_model_training(background_tasks: BackgroundTasks, data: Dict[str, str]):
    """
    Start gaze model training in the background
    
    Args:
        data: Dictionary with training parameters
        
    Returns:
        Training start confirmation
    """
    data_dir = data.get('data_dir', 'backend/datasets/MPIIGaze')
    model_dir = data.get('model_dir', 'models_mpiigaze_derslens')
    
    # Check if dataset exists
    if not Path(data_dir).exists():
        raise HTTPException(status_code=400, detail=f"Dataset not found at {data_dir}")
    
    def train_model_task():
        """Background training task"""
        try:
            logger.info("🎯 Starting background gaze model training...")
            results = train_derslens_gaze_model(data_dir, model_dir)
            
            if results["success"]:
                logger.info("🎉 Background training completed successfully!")
                # Reload the detector with new model
                global _gaze_detector
                _gaze_detector = None  # Force reload
            else:
                logger.error(f"Background training failed: {results.get('error')}")
                
        except Exception as e:
            logger.error(f"Training task failed: {e}")
    
    background_tasks.add_task(train_model_task)
    
    return {
        "status": "training_started",
        "message": "Model training started in background",
        "data_dir": data_dir,
        "model_dir": model_dir
    }

@router.get("/attention_zones/info")
async def get_attention_zones_info():
    """Get information about attention zone classifications"""
    return {
        "attention_zones": {
            "focused": {
                "description": "Student is looking directly at teacher/board",
                "criteria": "Gaze angle within ±15° horizontally and ±10° vertically",
                "score_weight": 100,
                "color": "green"
            },
            "attentive": {
                "description": "Student is generally looking forward",
                "criteria": "Gaze angle within ±30° horizontally and ±20° vertically",
                "score_weight": 75,
                "color": "yellow"
            },
            "distracted": {
                "description": "Student is looking to the sides",
                "criteria": "Gaze angle within ±45° horizontally",
                "score_weight": 25,
                "color": "orange"
            },
            "off_task": {
                "description": "Student is looking away from instruction area",
                "criteria": "Gaze angle beyond ±45° horizontally",
                "score_weight": 0,
                "color": "red"
            }
        },
        "scoring": {
            "excellent": "85-100% attention score",
            "good": "70-84% attention score",
            "moderate": "50-69% attention score",
            "poor": "0-49% attention score"
        }
    }

# Health check endpoint
@router.get("/health")
async def health_check():
    """Health check for gaze detection service"""
    try:
        detector = get_gaze_detector()
        
        return {
            "status": "healthy",
            "service": "gaze_detection",
            "detector_available": detector is not None,
            "model_loaded": detector.model is not None if detector else False,
            "device": str(detector.device) if detector else "unknown"
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "service": "gaze_detection",
            "error": str(e)
        }
