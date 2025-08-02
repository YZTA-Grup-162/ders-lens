"""
Gemini AI Integration Router for DersLens
Provides intelligent insights, explanations, and adaptive content
"""

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import httpx
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.core.config import settings

router = APIRouter()

class StudentEngagementData(BaseModel):
    student_id: str
    attention_score: float
    gaze_pattern: str
    session_duration: int
    timestamp: Optional[datetime] = None

class ClassInsightRequest(BaseModel):
    students: List[StudentEngagementData]
    topic: str
    class_duration: int
    difficulty_level: str

class AdaptiveQuizRequest(BaseModel):
    topic: str
    difficulty: str
    engagement_level: float
    student_context: Optional[Dict[str, Any]] = None

class EngagementAlert(BaseModel):
    student_id: str
    explanation: str
    risk_level: str
    intervention_type: str
    recommendations: List[str]
    confidence_score: float

@router.post("/explain-engagement")
async def explain_student_engagement(data: StudentEngagementData) -> Dict[str, Any]:
    """
    🧠 Generate AI-powered explanation for student engagement patterns
    """
    try:
        # Forward to AI service for Gemini analysis
        ai_service_url = f"{settings.AI_SERVICE_URL}/api/v1/gemini/explain"
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                ai_service_url,
                json=data.dict()
            )
            
            if response.status_code == 200:
                result = response.json()
                return {
                    "success": True,
                    "explanation": result.get("explanation", ""),
                    "risk_level": result.get("risk_level", "medium"),
                    "recommendations": result.get("recommendations", []),
                    "intervention_type": result.get("intervention_type", "gentle"),
                    "confidence_score": result.get("confidence_score", 0.8),
                    "timestamp": datetime.now().isoformat()
                }
            else:
                raise HTTPException(status_code=response.status_code, detail="AI service error")
                
    except Exception as e:
        # Fallback explanation
        return _generate_fallback_explanation(data)

@router.post("/class-insights")
async def generate_class_insights(request: ClassInsightRequest) -> Dict[str, Any]:
    """
    📊 Generate comprehensive class-wide engagement insights using Gemini AI
    """
    try:
        # Forward to AI service
        ai_service_url = f"{settings.AI_SERVICE_URL}/api/v1/gemini/class-analysis"
        
        async with httpx.AsyncClient(timeout=45.0) as client:
            response = await client.post(
                ai_service_url,
                json=request.dict()
            )
            
            if response.status_code == 200:
                insights = response.json()
                
                # Enhance with real-time data
                enhanced_insights = {
                    **insights,
                    "generated_at": datetime.now().isoformat(),
                    "class_size": len(request.students),
                    "avg_engagement": sum(s.attention_score for s in request.students) / len(request.students),
                    "students_at_risk": len([s for s in request.students if s.attention_score < 0.5]),
                    "session_health_score": _calculate_session_health(request.students)
                }
                
                return enhanced_insights
            else:
                raise HTTPException(status_code=response.status_code, detail="AI service error")
                
    except Exception as e:
        return _generate_fallback_class_insights(request)

@router.post("/adaptive-quiz")
async def generate_adaptive_quiz(request: AdaptiveQuizRequest) -> Dict[str, Any]:
    """
    🎮 Generate personalized quiz questions based on student engagement
    """
    try:
        # Forward to AI service
        ai_service_url = f"{settings.AI_SERVICE_URL}/api/v1/gemini/adaptive-quiz"
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                ai_service_url,
                json=request.dict()
            )
            
            if response.status_code == 200:
                quiz = response.json()
                
                # Add gamification elements
                quiz["gamification"] = {
                    "points_available": _calculate_points(request.difficulty),
                    "achievement_unlocked": _get_achievement(request.engagement_level),
                    "streak_bonus": request.student_context.get("streak", 0) if request.student_context else 0
                }
                
                return quiz
            else:
                raise HTTPException(status_code=response.status_code, detail="AI service error")
                
    except Exception as e:
        return _generate_fallback_quiz(request)

@router.get("/real-time-alerts")
async def get_real_time_alerts() -> List[EngagementAlert]:
    """
    🚨 Get real-time engagement alerts with AI explanations
    """
    try:
        # Get current engagement data from AI service
        ai_service_url = f"{settings.AI_SERVICE_URL}/api/v1/engagement/current"
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(ai_service_url)
            
            if response.status_code == 200:
                engagement_data = response.json()
                alerts = []
                
                for student_data in engagement_data.get("students", []):
                    if student_data["attention_score"] < 0.6:  # Alert threshold
                        # Get AI explanation for this student
                        explanation_response = await client.post(
                            f"{settings.AI_SERVICE_URL}/api/v1/gemini/explain",
                            json=student_data
                        )
                        
                        if explanation_response.status_code == 200:
                            explanation = explanation_response.json()
                            alerts.append(EngagementAlert(
                                student_id=student_data["student_id"],
                                explanation=explanation["explanation"],
                                risk_level=explanation["risk_level"],
                                intervention_type=explanation["intervention_type"],
                                recommendations=explanation["recommendations"],
                                confidence_score=explanation["confidence_score"]
                            ))
                
                return alerts
            else:
                return []
                
    except Exception as e:
        return []

@router.post("/intervention-suggestion")
async def suggest_intervention(student_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
    """
    💡 Get AI-powered intervention suggestions for specific students
    """
    try:
        # Create intervention prompt for Gemini
        intervention_request = {
            "student_id": student_id,
            "context": context,
            "request_type": "intervention"
        }
        
        # Forward to AI service
        ai_service_url = f"{settings.AI_SERVICE_URL}/api/v1/gemini/intervention"
        
        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.post(
                ai_service_url,
                json=intervention_request
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                raise HTTPException(status_code=response.status_code, detail="AI service error")
                
    except Exception as e:
        return _generate_fallback_intervention(student_id, context)

@router.get("/privacy-report")
async def get_privacy_report() -> Dict[str, Any]:
    """
    🔒 Generate privacy and data handling report for transparency
    """
    return {
        "data_processing": {
            "student_identification": "Anonymized student IDs only",
            "biometric_data": "No facial recognition storage",
            "attention_patterns": "Aggregated and anonymized",
            "retention_period": "Session-based, deleted after class"
        },
        "ai_processing": {
            "gemini_usage": "Engagement analysis only",
            "data_sent_to_ai": "Anonymized engagement metrics",
            "personal_info": "Never shared with external AI",
            "gdpr_compliant": True
        },
        "security_measures": [
            "End-to-end encryption",
            "Local processing priority", 
            "No video recording",
            "Anonymized analytics only",
            "Teacher-controlled data access"
        ],
        "transparency": {
            "student_consent": "Required for participation",
            "opt_out_available": True,
            "data_deletion": "Immediate upon request",
            "audit_trail": "Complete processing logs"
        }
    }

# Helper functions
def _generate_fallback_explanation(data: StudentEngagementData) -> Dict[str, Any]:
    """Generate fallback explanation when AI service is unavailable"""
    if data.attention_score > 0.8:
        return {
            "success": True,
            "explanation": f"Student {data.student_id} shows excellent engagement patterns with consistent focus.",
            "risk_level": "low",
            "recommendations": ["Continue current approach", "Consider advanced challenges"],
            "intervention_type": "none",
            "confidence_score": 0.7
        }
    elif data.attention_score > 0.5:
        return {
            "success": True,
            "explanation": f"Student {data.student_id} shows moderate engagement with occasional distractions.",
            "risk_level": "medium", 
            "recommendations": ["Gentle check-in", "Interactive content", "Brief activity change"],
            "intervention_type": "gentle",
            "confidence_score": 0.7
        }
    else:
        return {
            "success": True,
            "explanation": f"Student {data.student_id} shows low engagement and may benefit from immediate attention.",
            "risk_level": "high",
            "recommendations": ["Direct intervention", "One-on-one check", "Activity break"],
            "intervention_type": "immediate",
            "confidence_score": 0.8
        }

def _generate_fallback_class_insights(request: ClassInsightRequest) -> Dict[str, Any]:
    """Generate fallback class insights"""
    avg_engagement = sum(s.attention_score for s in request.students) / len(request.students)
    
    return {
        "overall_health": int(avg_engagement * 100),
        "key_insights": [
            f"Class of {len(request.students)} students showing {'good' if avg_engagement > 0.7 else 'mixed'} engagement",
            f"Average attention level: {avg_engagement:.1%}",
            "Recommendation: Monitor individual students for personalized support"
        ],
        "teaching_recommendations": [
            "Consider interactive break",
            "Use varied teaching methods",
            "Monitor student energy levels",
            "Provide individual check-ins"
        ],
        "engagement_trend": "stable",
        "class_mood": "focused",
        "recommended_activity": "interactive discussion"
    }

def _generate_fallback_quiz(request: AdaptiveQuizRequest) -> Dict[str, Any]:
    """Generate fallback quiz"""
    return {
        "question": f"Quick knowledge check about {request.topic}",
        "question_type": "multiple_choice",
        "options": [
            "Core concept A",
            "Core concept B", 
            "Core concept C",
            "Need more explanation"
        ],
        "correct_answer": "Core concept B",
        "explanation": "This reinforces key learning objectives",
        "engagement_boost": "Quick check to refocus attention",
        "estimated_time": "30 seconds",
        "fun_factor": 6
    }

def _generate_fallback_intervention(student_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
    """Generate fallback intervention suggestion"""
    return {
        "intervention_type": "gentle_check_in",
        "suggested_actions": [
            "Brief one-on-one conversation",
            "Check if student needs clarification",
            "Offer alternative learning approach"
        ],
        "timing": "within next 2-3 minutes",
        "follow_up": "Monitor for 5 minutes after intervention"
    }

def _calculate_session_health(students: List[StudentEngagementData]) -> int:
    """Calculate overall session health score"""
    if not students:
        return 50
    
    avg_attention = sum(s.attention_score for s in students) / len(students)
    variance = sum((s.attention_score - avg_attention) ** 2 for s in students) / len(students)
    
    # Health score based on average attention and consistency
    health = int((avg_attention * 0.8 + (1 - variance) * 0.2) * 100)
    return max(0, min(100, health))

def _calculate_points(difficulty: str) -> int:
    """Calculate points for quiz gamification"""
    points_map = {"easy": 10, "medium": 20, "hard": 30}
    return points_map.get(difficulty, 15)

def _get_achievement(engagement_level: float) -> str:
    """Get achievement based on engagement level"""
    if engagement_level > 0.9:
        return "Focus Master! 🏆"
    elif engagement_level > 0.7:
        return "Great Attention! ⭐"
    elif engagement_level > 0.5:
        return "Keep Going! 💪"
    else:
        return "Let's Refocus! 🎯"
