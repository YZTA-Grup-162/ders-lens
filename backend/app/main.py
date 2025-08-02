"""
DersLens Backend - Main FastAPI Application
Clean and Final Version
"""

import logging
from contextlib import asynccontextmanager

import httpx
import sentry_sdk
from aioredis import from_url as redis_from_url
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Import routers
from app.api import auth, demo, student, teacher
from app.core.config import settings
from app.core.security import get_current_username

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    logger.info("🚀 DersLens Backend başlatılıyor...")
    yield
    logger.info("📴 DersLens Backend kapatılıyor...")

# init Sentry
if settings.SENTRY_DSN:
    sentry_sdk.init(dsn=settings.SENTRY_DSN, traces_sample_rate=0.2)

# Create FastAPI app
app = FastAPI(
    title="DersLens API",
    description="AI Destekli Sınıf Etkileşim Analizi Sistemi - Final Version",
    version="1.0.0",
    lifespan=lifespan,
    dependencies=[Depends(get_current_username)]
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers
app.include_router(auth.router, prefix="/api/auth", tags=["authentication"])
app.include_router(student.router, prefix="/api/student", tags=["student"])
app.include_router(teacher.router, prefix="/api/teacher", tags=["teacher"])
app.include_router(demo.router, prefix="/api/demo", tags=["demo"])

# Static files (commented out for now since directory doesn't exist)
# app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "DersLens API - Final Version",
        "version": "1.0.0",
        "status": "healthy",
        "docs": "/docs"
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    try:
        # Check AI service connection
        async with httpx.AsyncClient() as client:
            ai_response = await client.get(
                f"{settings.AI_SERVICE_URL}/health", 
                timeout=5.0
            )
            ai_status = "connected" if ai_response.status_code == 200 else "disconnected"
    except:
        ai_status = "disconnected"
    
    return {
        "status": "healthy",
        "backend": "online",
        "ai_service": ai_status,
        "database": "connected",
        "version": "1.0.0"
    }

@app.post("/api/analyze/frame")
async def analyze_frame(request: dict):
    """Proxy to AI service for frame analysis"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{settings.AI_SERVICE_URL}/api/v1/analyze/frame",
                json=request,
                timeout=30.0
            )
            return response.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI service error: {str(e)}")

@app.post("/api/analyze")
async def analyze(frame: UploadFile = File(...)):
    """Proxy to AI service for analysis (frontend compatibility)"""
    print(f"🔄 Backend received analyze request:")
    print(f"  - Filename: {frame.filename}")
    print(f"  - Content Type: {frame.content_type}")
    print(f"  - File Size: {frame.size if hasattr(frame, 'size') else 'unknown'}")
    
    try:
        frame_data = await frame.read()
        print(f"  - Read {len(frame_data)} bytes from uploaded file")
        
        async with httpx.AsyncClient() as client:
            files = {"frame": (frame.filename or "frame.jpg", frame_data, frame.content_type or "image/jpeg")}
            
            print(f"📤 Forwarding to AI service: {settings.AI_SERVICE_URL}/api/v1/analyze/upload")
            print(f"  - Files param: {list(files.keys())}")
            
            response = await client.post(
                f"{settings.AI_SERVICE_URL}/api/v1/analyze/upload",
                files=files,
                timeout=30.0
            )
            
            print(f"📨 AI service response:")
            print(f"  - Status: {response.status_code}")
            print(f"  - Headers: {dict(response.headers)}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ AI analysis successful: {result}")
                return result
            else:
                error_text = response.text
                print(f"❌ AI service error: {response.status_code} - {error_text}")
                raise HTTPException(status_code=response.status_code, detail=f"AI service error: {error_text}")
                
    except httpx.TimeoutException:
        print("⏰ AI service timeout")
        raise HTTPException(status_code=504, detail="AI service timeout")
    except httpx.RequestError as e:
        print(f"🔌 AI service connection error: {e}")
        raise HTTPException(status_code=503, detail=f"AI service connection error: {str(e)}")
    except Exception as e:
        print(f"💥 Unexpected error in backend: {e}")
        raise HTTPException(status_code=500, detail=f"Backend error: {str(e)}")

# create a Redis client and attach to app.state
@app.on_event("startup")
async def startup_event():
    app.state.redis = await redis_from_url(settings.REDIS_URL, encoding="utf-8", decode_responses=True)

@app.on_event("shutdown")
async def shutdown_event():
    await app.state.redis.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
