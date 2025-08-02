
import asyncio
import base64
import json
import logging
import os
from datetime import datetime
from io import BytesIO
from typing import Dict, List, Optional

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from enhanced_cv_pipeline import DersLensEnhancedPipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="DersLens API",
    description="AI-powered student engagement monitoring with Gemini integration",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline instance
pipeline: Optional[DersLensEnhancedPipeline] = None
active_connections: List[WebSocket] = []

# Data models
class EngagementQuery(BaseModel):
    question: str
    context: Optional[Dict] = None

class AlertConfig(BaseModel):
    low_attention_threshold: float = 40.0
    sustained_distraction_seconds: int = 30
    emotion_confidence_threshold: float = 0.8

class SessionConfig(BaseModel):
    gemini_api_key: str
    fer_model_path: Optional[str] = "models_fer2013/fer2013_pytorch.onnx"
    camera_index: int = 0
    analysis_interval: int = 5

# API Routes

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "DersLens API",
        "version": "1.0.0",
        "description": "AI-powered student engagement monitoring",
        "features": [
            "Real-time emotion recognition",
            "Gaze tracking and attention monitoring", 
            "Gemini AI contextual analysis",
            "Smart alerts and recommendations",
            "Conversational teaching assistant"
        ],
        "endpoints": {
            "health": "/health",
            "initialize": "/api/v1/session/initialize",
            "start": "/api/v1/session/start",
            "stop": "/api/v1/session/stop",
            "query": "/api/v1/ai/query",
            "metrics": "/api/v1/metrics/current",
            "summary": "/api/v1/session/summary",
            "websocket": "/ws"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "pipeline_initialized": pipeline is not None,
        "active_connections": len(active_connections)
    }

@app.post("/api/v1/session/initialize")
async def initialize_session(config: SessionConfig):
    """Initialize DersLens pipeline with configuration"""
    global pipeline
    
    try:
        if not config.gemini_api_key:
            raise HTTPException(status_code=400, detail="Gemini API key is required")
        
        pipeline = DersLensEnhancedPipeline(
            gemini_api_key=config.gemini_api_key,
            fer_model_path=config.fer_model_path
        )
        
        logger.info("DersLens pipeline initialized successfully")
        
        return {
            "status": "success",
            "message": "Pipeline initialized successfully",
            "timestamp": datetime.now().isoformat(),
            "config": {
                "fer_model_path": config.fer_model_path,
                "camera_index": config.camera_index,
                "analysis_interval": config.analysis_interval
            }
        }
        
    except Exception as e:
        logger.error(f"Pipeline initialization failed: {e}")
        raise HTTPException(status_code=500, detail=f"Initialization failed: {str(e)}")

@app.post("/api/v1/session/start")
async def start_session():
    """Start monitoring session"""
    if pipeline is None:
        raise HTTPException(status_code=400, detail="Pipeline not initialized")
    
  
    
    return {
        "status": "success",
        "message": "Monitoring session started",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/v1/session/stop")
async def stop_session():
    """Stop monitoring session"""
    if pipeline is None:
        raise HTTPException(status_code=400, detail="Pipeline not initialized")
    
    return {
        "status": "success",
        "message": "Monitoring session stopped",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/v1/frame/analyze")
async def analyze_frame(file: UploadFile = File(...)):
    """Analyze uploaded frame for engagement metrics"""
    if pipeline is None:
        raise HTTPException(status_code=400, detail="Pipeline not initialized")
    
    try:
        # Read uploaded image
        contents = await file.read()
        
        # Convert to OpenCV format
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image format")
        
        # Process frame
        result = pipeline.process_video_frame(frame, analyze_with_gemini=True)
        
        # Convert debug frame to base64 for response
        if "debug_frame" in result:
            _, buffer = cv2.imencode('.jpg', result["debug_frame"])
            debug_frame_b64 = base64.b64encode(buffer).decode('utf-8')
            result["debug_frame_b64"] = debug_frame_b64
            del result["debug_frame"]  # Remove numpy array
        
        return {
            "status": "success",
            "analysis": result,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Frame analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/api/v1/ai/query")
async def query_ai_assistant(query: EngagementQuery):
    """Query the AI teaching assistant"""
    if pipeline is None:
        raise HTTPException(status_code=400, detail="Pipeline not initialized")
    
    try:
        response = await pipeline.handle_teacher_query(query.question)
        
        return {
            "status": "success",
            "query": query.question,
            "response": response,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"AI query failed: {e}")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

@app.get("/api/v1/metrics/current")
async def get_current_metrics():
    """Get current engagement metrics"""
    if pipeline is None:
        raise HTTPException(status_code=400, detail="Pipeline not initialized")
    
    try:
        # Get recent attention history
        cv_pipeline = pipeline.cv_pipeline
        
        if not cv_pipeline.attention_history:
            return {
                "status": "no_data",
                "message": "No engagement data available",
                "timestamp": datetime.now().isoformat()
            }
        
        latest_metrics = cv_pipeline.attention_history[-1]
        smoothed_score = cv_pipeline.get_smoothed_attention_score()
        
        return {
            "status": "success",
            "metrics": {
                "attention_score": latest_metrics.attention_score,
                "smoothed_attention_score": smoothed_score,
                "attention_state": latest_metrics.attention_state.value,
                "primary_emotion": latest_metrics.emotion_data.primary_emotion,
                "emotion_confidence": latest_metrics.emotion_data.confidence,
                "gaze_on_screen": latest_metrics.gaze_data.on_screen,
                "gaze_confidence": latest_metrics.gaze_data.confidence,
                "head_pose_angles": latest_metrics.head_pose.euler_angles,
                "timestamp": latest_metrics.timestamp
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get metrics: {str(e)}")

@app.get("/api/v1/metrics/history")
async def get_metrics_history(limit: int = 50):
    """Get historical engagement metrics"""
    if pipeline is None:
        raise HTTPException(status_code=400, detail="Pipeline not initialized")
    
    try:
        cv_pipeline = pipeline.cv_pipeline
        
        if not cv_pipeline.attention_history:
            return {
                "status": "no_data",
                "message": "No engagement data available",
                "timestamp": datetime.now().isoformat()
            }
        
        # Get recent history
        recent_history = cv_pipeline.attention_history[-limit:]
        
        history_data = []
        for metrics in recent_history:
            history_data.append({
                "attention_score": metrics.attention_score,
                "attention_state": metrics.attention_state.value,
                "primary_emotion": metrics.emotion_data.primary_emotion,
                "emotion_confidence": metrics.emotion_data.confidence,
                "gaze_on_screen": metrics.gaze_data.on_screen,
                "timestamp": metrics.timestamp
            })
        
        return {
            "status": "success",
            "history": history_data,
            "count": len(history_data),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get history: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get history: {str(e)}")

@app.get("/api/v1/session/summary")
async def get_session_summary():
    """Get comprehensive session summary"""
    if pipeline is None:
        raise HTTPException(status_code=400, detail="Pipeline not initialized")
    
    try:
        summary = pipeline.get_session_summary()
        
        return {
            "status": "success",
            "summary": summary,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get summary: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get summary: {str(e)}")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await websocket.accept()
    active_connections.append(websocket)
    
    try:
        while True:
            # Wait for client message or send periodic updates
            data = await websocket.receive_text()
            
            if data == "get_metrics":
                if pipeline and pipeline.cv_pipeline.attention_history:
                    latest_metrics = pipeline.cv_pipeline.attention_history[-1]
                    
                    response = {
                        "type": "metrics_update",
                        "data": {
                            "attention_score": latest_metrics.attention_score,
                            "attention_state": latest_metrics.attention_state.value,
                            "primary_emotion": latest_metrics.emotion_data.primary_emotion,
                            "gaze_on_screen": latest_metrics.gaze_data.on_screen,
                            "timestamp": latest_metrics.timestamp
                        }
                    }
                    
                    await websocket.send_text(json.dumps(response))
                else:
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "message": "No data available"
                    }))
            
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        active_connections.remove(websocket)

async def broadcast_update(message: Dict):
    """Broadcast update to all connected clients"""
    if active_connections:
        disconnected = []
        for connection in active_connections:
            try:
                await connection.send_text(json.dumps(message))
            except:
                disconnected.append(connection)
        
        # Remove disconnected clients
        for connection in disconnected:
            active_connections.remove(connection)

# Static files for demo frontend
@app.get("/demo", response_class=HTMLResponse)
async def demo_page():
    """Simple demo page"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>DersLens Demo</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .container { max-width: 800px; margin: 0 auto; }
            .metric { background: #f5f5f5; padding: 20px; margin: 10px 0; border-radius: 5px; }
            .button { background: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; }
            #status { margin: 20px 0; padding: 10px; border-radius: 5px; }
            .success { background: #d4edda; color: #155724; }
            .error { background: #f8d7da; color: #721c24; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎓 DersLens Demo</h1>
            <p>AI-powered student engagement monitoring with Gemini integration</p>
            
            <div id="status" class="success">Ready to connect</div>
            
            <button class="button" onclick="connectWebSocket()">Connect WebSocket</button>
            <button class="button" onclick="getMetrics()">Get Current Metrics</button>
            
            <div id="metrics">
                <div class="metric">
                    <h3>Attention Score</h3>
                    <div id="attention-score">--</div>
                </div>
                <div class="metric">
                    <h3>Attention State</h3>
                    <div id="attention-state">--</div>
                </div>
                <div class="metric">
                    <h3>Primary Emotion</h3>
                    <div id="primary-emotion">--</div>
                </div>
                <div class="metric">
                    <h3>Gaze Status</h3>
                    <div id="gaze-status">--</div>
                </div>
            </div>
        </div>
        
        <script>
            let ws = null;
            
            function connectWebSocket() {
                ws = new WebSocket('ws://localhost:8000/ws');
                
                ws.onopen = function(event) {
                    document.getElementById('status').innerHTML = 'Connected to WebSocket';
                    document.getElementById('status').className = 'success';
                };
                
                ws.onmessage = function(event) {
                    const data = JSON.parse(event.data);
                    if (data.type === 'metrics_update') {
                        updateMetrics(data.data);
                    }
                };
                
                ws.onclose = function(event) {
                    document.getElementById('status').innerHTML = 'WebSocket connection closed';
                    document.getElementById('status').className = 'error';
                };
            }
            
            function getMetrics() {
                if (ws && ws.readyState === WebSocket.OPEN) {
                    ws.send('get_metrics');
                } else {
                    fetch('/api/v1/metrics/current')
                        .then(response => response.json())
                        .then(data => {
                            if (data.status === 'success') {
                                updateMetrics(data.metrics);
                            }
                        });
                }
            }
            
            function updateMetrics(metrics) {
                document.getElementById('attention-score').innerHTML = metrics.attention_score.toFixed(1) + '%';
                document.getElementById('attention-state').innerHTML = metrics.attention_state.replace('_', ' ');
                document.getElementById('primary-emotion').innerHTML = metrics.primary_emotion;
                document.getElementById('gaze-status').innerHTML = metrics.gaze_on_screen ? 'On Screen' : 'Off Screen';
            }
            
            // Auto-refresh every 2 seconds
            setInterval(getMetrics, 2000);
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

if __name__ == "__main__":
    # Run the FastAPI server
    uvicorn.run(
        "fastapi_backend:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
