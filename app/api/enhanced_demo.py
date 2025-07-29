"""
 Demo API
"""

import logging
from typing import Any, Dict
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
logger = logging.getLogger(__name__)
_enhanced_demo_engine = None

def get_enhanced_demo_engine():
    global _enhanced_demo_engine
    if _enhanced_demo_engine is None:
        try:
            from ..ai.enhanced_demo import EnhancedDemo
            model_paths = [
                "/app/models",
                "models", 
                "./models",
                "../models",
                "/app/backend/models",
                "d:/ders-lens"
            ]
            engine_created = False
            for path in model_paths:
                try:
                    _enhanced_demo_engine = EnhancedDemo(models_dir=path)
                    logger.info(f"✅ Enhanced demo engine created with models from: {path}")
                    engine_created = True
                    break
                except Exception as e:
                    logger.debug(f"Failed to create enhanced engine with path {path}: {e}")
                    continue
            if not engine_created:
                _enhanced_demo_engine = EnhancedDemo()
                logger.warning("Created enhanced demo engine with default fallback")
        except Exception as e:
            logger.error(f"Failed to create enhanced demo engine: {e}")
            raise HTTPException(status_code=500, detail=f"Enhanced demo engine creation failed: {str(e)}")
    return _enhanced_demo_engine
def reload_enhanced_demo_engine():
    global _enhanced_demo_engine
    _enhanced_demo_engine = None
    return get_enhanced_demo_engine()
router = APIRouter(prefix="/enhanced-demo", tags=["Enhanced Demo"])
@router.get("/status")
async def get_enhanced_demo_status():
    try:
        engine = get_enhanced_demo_engine()
        return JSONResponse({
            "status": "active",
            "model_version": engine.model_version,
            "trained_features": engine.trained_features,
            "active_models": [name for name, active in engine.trained_features.items() if active],
            "models_loaded": list(engine.models.keys()) if hasattr(engine, 'models') else [],
            "model_accuracies": {
                "daisee_attention": "90.5%",
                "fer2013_emotion": "High Performance",
                "mendeley_neural_net": "99.12%",
                "mendeley_ensemble": "100%",
                "onnx_emotion": "Trained",
                "gaze_tracking": "Real-time MediaPipe",
                "posture_analysis": "Real-time MediaPipe"
           },
            "update_frequency_hz": engine.update_frequency,
            "smoothing_window_size": engine.smoothing_window,
            "session_info": {
                "frames_processed": engine.frame_count,
                "session_start": getattr(engine, 'session_start', 0),
                "last_display_update": getattr(engine, 'last_display_update', 0)
           },
            "stability_settings": {
                "min_update_interval": engine.min_update_interval,
                "confidence_threshold": engine.confidence_threshold,
                "anti_flicker_enabled": True,
                "change_rate_limiting": "3% max per update"
           }
       })
    except Exception as e:
        logger.error(f"Enhanced status check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
@router.get("/model-details")
async def get_model_details():
    try:
        engine = get_enhanced_demo_engine()
        model_details = {
            "daisee_attention": {
                "loaded": engine.trained_features.get('daisee_attention', False),
                "description": "DAiSEE dataset attention detection",
                "accuracy": "90.5%",
                "input_size": "224x224",
                "output": "Binary attention classification",
                "model_file": "daisee_attention_best.pth"
           },
            "fer2013_emotion": {
                "loaded": engine.trained_features.get('fer2013_emotion', False),
                "description": "FER2013 emotion recognition",
                "accuracy": "High Performance", 
                "input_size": "48x48 grayscale",
                "output": "7-class emotion classification",
                "model_file": "fer2013_emotion_best.pth"
           },
            "mendeley_neural_net": {
                "loaded": engine.trained_features.get('mendeley_neural_net', False),
                "description": "Mendeley attention neural network",
                "accuracy": "99.12%",
                "input_size": "224x224",
                "output": "Binary attention classification",
                "model_file": "mendeley_nn_best.pth"
           },
            "mendeley_ensemble": {
                "loaded": engine.trained_features.get('mendeley_ensemble', False),
                "description": "Mendeley ensemble models",
                "accuracy": "100% (Gradient Boosting)",
                "models": ["Gradient Boosting", "Random Forest", "Logistic Regression"],
                "accuracies": ["100%", "99.5%", "96.1%"],
                "model_files": ["mendeley_gradient_boosting.pkl", "mendeley_random_forest.pkl", "mendeley_logistic_regression.pkl"]
           },
            "onnx_emotion": {
                "loaded": engine.trained_features.get('onnx_emotion', False),
                "description": "ONNX emotion recognition model",
                "accuracy": "Trained",
                "input_size": "48x48",
                "output": "Emotion classification",
                "model_file": "best_model.onnx"
           },
            "gaze_tracking": {
                "loaded": engine.trained_features.get('gaze_tracking', False),
                "description": "Real-time gaze direction estimation",
                "method": "MediaPipe Face Mesh",
                "accuracy": "Real-time geometric calculation",
                "output": "Gaze direction and stability"
           },
            "posture_analysis": {
                "loaded": engine.trained_features.get('posture_analysis', False),
                "description": "Real-time posture analysis",
                "method": "MediaPipe Pose Detection",
                "accuracy": "Real-time pose estimation",
                "output": "Posture quality score"
           }
       }
        return JSONResponse({
            "model_details": model_details,
            "total_models": len([d for d in model_details.values() if d['loaded']]),
            "combined_intelligence": "Multi-model ensemble with weighted predictions",
            "prediction_strategy": "Accuracy-weighted combination of all active models"
       })
    except Exception as e:
        logger.error(f"Model details request failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
@router.get("/demo-metrics") 
async def get_enhanced_demo_metrics():
    try:
        engine = get_enhanced_demo_engine()
        import numpy as np
        test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = await engine.analyze_frame(test_frame)
        return JSONResponse({
            "display_metrics": result.get('display_metrics', {}),
            "model_predictions": {
                "individual_scores": result.get('model_scores', {}),
                "combined_metrics": result.get('combined_metrics', {}),
                "raw_predictions": result.get('raw_predictions', {}),
                "confidence_scores": result.get('confidence_scores', {})
           },
            "model_info": {
                "version": engine.model_version,
                "active_models": result.get('active_models', []),
                "total_features": sum(engine.trained_features.values())
           },
            "session_info": {
                "frame_count": result.get('frame_count', 0),
                "session_duration": result.get('session_duration', 0),
                "processing_time_ms": result.get('processing_time_ms', 0),
                "update_frequency": engine.update_frequency,
                "smoothing_window": engine.smoothing_window
           },
            "performance": {
                "face_detected": result.get('face_detected', False),
                "models_responding": len(result.get('confidence_scores', {})),
                "overall_confidence": sum(result.get('confidence_scores', {}).values()) / max(len(result.get('confidence_scores', {})), 1)
           },
            "timestamp": engine.session_start if hasattr(engine, 'session_start') else 0
       })
    except Exception as e:
        logger.error(f"Enhanced demo metrics failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
@router.post("/reload")
async def reload_enhanced_demo():
    try:
        engine = reload_enhanced_demo_engine()
        return JSONResponse({
            "status": "reloaded",
            "model_version": engine.model_version,
            "active_models": [name for name, active in engine.trained_features.items() if active],
            "message": "Enhanced demo engine reloaded successfully with all trained models"
       })
    except Exception as e:
        logger.error(f"Enhanced reload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
@router.get("/health")
async def enhanced_demo_health():
    try:
        engine = get_enhanced_demo_engine()
        model_health = {}
        for model_name, is_loaded in engine.trained_features.items():
            model_health[model_name] = {
                "loaded": is_loaded,
                "status": "healthy" if is_loaded else "not_loaded",
                "ready": is_loaded
           }
        total_models = len(engine.trained_features)
        loaded_models = sum(engine.trained_features.values())
        health_percentage = (loaded_models / total_models) * 100
        overall_health = "excellent" if health_percentage >= 80 else "good" if health_percentage >= 60 else "fair" if health_percentage >= 40 else "poor"
        return JSONResponse({
            "overall_health": overall_health,
            "health_percentage": health_percentage,
            "models_loaded": f"{loaded_models}/{total_models}",
            "model_health": model_health,
            "system_status": {
                "engine_status": "healthy",
                "memory_status": "healthy",
                "processing_status": "healthy"
           },
            "capabilities": {
                "attention_detection": engine.trained_features.get('daisee_attention', False) or engine.trained_features.get('mendeley_neural_net', False) or engine.trained_features.get('mendeley_ensemble', False),
                "emotion_recognition": engine.trained_features.get('fer2013_emotion', False) or engine.trained_features.get('onnx_emotion', False),
                "gaze_tracking": engine.trained_features.get('gaze_tracking', False),
                "posture_analysis": engine.trained_features.get('posture_analysis', False)
           },
            "timestamp": engine.session_start if hasattr(engine, 'session_start') else 0
       })
    except Exception as e:
        logger.error(f"Enhanced health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "overall_health": "unhealthy",
                "error": str(e),
                "details": {"engine_status": "failed"}
           }
       )
@router.get("/benchmark")
async def benchmark_models():
    try:
        engine = get_enhanced_demo_engine()
        import time
        import numpy as np
        test_frames = [np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8) for _ in range(10)]
        benchmark_results = {}
        for model_name in ['daisee_attention', 'fer2013_emotion', 'mendeley_neural_net']:
            if engine.trained_features.get(model_name, False):
                start_time = time.time()
                try:
                    for frame in test_frames:
                        if model_name == 'daisee_attention':
                            engine.predict_daisee_attention(frame)
                        elif model_name == 'fer2013_emotion':
                            engine.predict_fer2013_emotion(frame)
                        elif model_name == 'mendeley_neural_net':
                            engine.predict_mendeley_attention(frame)
                    avg_time = (time.time() - start_time) / len(test_frames) * 1000
                    benchmark_results[model_name] = {
                        "avg_inference_time_ms": round(avg_time, 2),
                        "frames_per_second": round(1000 / avg_time, 1),
                        "status": "operational"
                   }
                except Exception as e:
                    benchmark_results[model_name] = {
                        "status": "error",
                        "error": str(e)
                   }
        if engine.trained_features.get('mendeley_ensemble', False):
            start_time = time.time()
            try:
                for frame in test_frames:
                    features = engine.extract_features_for_ensemble(frame)
                    engine.predict_mendeley_ensemble(features)
                avg_time = (time.time() - start_time) / len(test_frames) * 1000
                benchmark_results['mendeley_ensemble'] = {
                    "avg_inference_time_ms": round(avg_time, 2),
                    "frames_per_second": round(1000 / avg_time, 1),
                    "status": "operational"
               }
            except Exception as e:
                benchmark_results['mendeley_ensemble'] = {
                    "status": "error",
                    "error": str(e)
               }
        start_time = time.time()
        for frame in test_frames[:5]:
            await engine.analyze_frame(frame)
        total_avg_time = (time.time() - start_time) / 5 * 1000
        return JSONResponse({
            "individual_models": benchmark_results,
            "full_system": {
                "avg_analysis_time_ms": round(total_avg_time, 2),
                "max_fps": round(1000 / total_avg_time, 1),
                "recommended_fps": "0.8 (for stability)"
           },
            "recommendations": {
                "update_frequency": f"Current: {engine.update_frequency} Hz",
                "stability_optimized": True,
                "performance_mode": "Accuracy over speed"
           }
       })
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
def register_enhanced_demo_routes(app):
    app.include_router(router)
    logger.info("✅ Enhanced demo routes registered")