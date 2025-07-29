"""
AttentionPulse Backend - FastAPI Application
"""

from contextlib import asynccontextmanager

from app.api import auth, student, teacher
from app.core.config import settings
from app.core.database import Base, engine
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles


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
