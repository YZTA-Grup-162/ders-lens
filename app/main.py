"""
Ders Lens Backend - FastAPI Application
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api import auth, demo, enhanced_demo, predictions, student, teacher
from app.api.websocket import (websocket_endpoint_student,
                               websocket_endpoint_teacher)
from app.core.config import settings
from app.core.database import Base, engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("**Ders Lens API başlatılıyor...**")
    Base.metadata.create_all(bind=engine)
    yield
    print("📴 Ders Lens API kapatılıyor...")
app = FastAPI(
    title="Ders Lens API",
    description="AI Destekli Sınıf Etkileşim Analizi Sistemi",
    version="1.0.0",
    lifespan=lifespan
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_HOSTS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(auth.router, prefix="/api/auth", tags=["authentication"])
app.include_router(predictions.router, prefix="/api/ai", tags=["predictions"])
app.include_router(demo.router, prefix="/api", tags=["demo"])
app.include_router(enhanced_demo.router, prefix="/api", tags=["enhanced-demo"])
app.include_router(student.router, prefix="/api/student", tags=["student"])
app.include_router(teacher.router, prefix="/api/teacher", tags=["teacher"])
app.mount("/static", StaticFiles(directory="../"), name="static")
@app.get("/")
async def root():
    return {
        "message": "Ders Lens API",
        "version": "1.0.0",
        "status": "healthy",
        "docs": "/docs"
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "database": "connected",  
        "ai_model": "loaded"       
    }
@app.post("/api/v1/analyze/frame")
async def analyze_frame_endpoint(request: dict):
    import httpx
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "http://localhost:8001/api/v1/analyze/frame",
                json=request,
                timeout=30.0
            )
            ai_result = response.json()
            if 'data' in ai_result and 'metadata' in ai_result['data']:
                metadata = ai_result['data']['metadata']
                if 'face_count' not in metadata and 'faceDetected' in metadata:
                    metadata['face_count'] = 1 if metadata['faceDetected'] else 0
            return ai_result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI service error: {str(e)}")
@app.websocket("/ws/test")
async def websocket_test_endpoint(websocket: WebSocket):
    await websocket.accept()
    await websocket.send_text("Hello from test WebSocket!")
    try:
        while True:
            data = await websocket.receive_text()
            await websocket.send_text(f"Echo: {data}")
    except Exception as e:
        print(f"WebSocket error: {e}")
@app.websocket("/ws/student/{user_id}")
async def websocket_student_endpoint_route(websocket: WebSocket, user_id: str):
    await websocket_endpoint_student(websocket, user_id)
@app.websocket("/ws/teacher/{user_id}")
async def websocket_teacher_endpoint_route(websocket: WebSocket, user_id: str):
    await websocket_endpoint_teacher(websocket, user_id)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )