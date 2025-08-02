

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import google.generativeai as genai
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from gemini_explainer import GeminiExplainer

# Initialize router
router = APIRouter(prefix="/api/v1/gemini", tags=["gemini"])

# Initialize Gemini explainer
explainer = None

try:
    explainer = GeminiExplainer()
    print("✅ Gemini explainer initialized successfully")
except Exception as e:
    print(f"⚠️ Gemini explainer initialization failed: {e}")

# Pydantic models
class StudentEngagementData(BaseModel):
    student_id: str
    attention_score: float
    gaze_pattern: str
    session_duration: int
    historical_data: Optional[List[Dict[str, Any]]] = []
    timestamp: Optional[datetime] = None

class ClassAnalysisRequest(BaseModel):
    students: List[Dict[str, Any]]
    topic: str
    class_duration: int
    difficulty_level: str

class AdaptiveQuizRequest(BaseModel):
    topic: str
    difficulty: str
    engagement_level: float
    student_context: Optional[Dict[str, Any]] = None

class InterventionRequest(BaseModel):
    student_id: str
    context: Dict[str, Any]
    request_type: str = "intervention"

@router.post("/explain")
async def explain_engagement(data: StudentEngagementData):
    """
    Generate AI explanation for student engagement patterns
    """
    try:
        if explainer is None:
            # Fallback explanation
            return _create_simple_explanation(data.dict())
        
        # Generate insight using Gemini
        insight = explainer.explain_attention_pattern(
            attention_data=data.dict(),
            historical_data=data.historical_data or []
        )
        
        return {
            "success": True,
            "explanation": insight.explanation,
            "risk_level": insight.risk_level,
            "intervention_type": insight.intervention_type,
            "recommendations": insight.recommendations,
            "confidence_score": 0.85,
            "timestamp": insight.timestamp.isoformat()
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "fallback_explanation": _create_simple_explanation(data.dict())
        }

@router.post("/class-analysis")
async def analyze_class_engagement(request: ClassAnalysisRequest):
    """
    Generate comprehensive class-wide engagement insights
    """
    try:
        if explainer is None:
            return _create_fallback_class_analysis(request)
        
        # Generate class insights using Gemini
        insights = explainer.analyze_class_engagement(
            students_data=request.students,
            topic=request.topic,
            duration=request.class_duration,
            difficulty=request.difficulty_level
        )
        
        return {
            "success": True,
            "overall_health": insights.get("overall_health", 75),
            "engagement_trend": insights.get("engagement_trend", "stable"),
            "class_mood": insights.get("class_mood", "focused"),
            "key_insights": insights.get("key_insights", []),
            "teaching_recommendations": insights.get("teaching_recommendations", []),
            "recommended_activity": insights.get("recommended_activity", "Continue current approach"),
            "optimal_break_time": insights.get("optimal_break_time", "In 10-15 minutes"),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return _create_fallback_class_analysis(request)

@router.post("/adaptive-quiz")
async def generate_adaptive_quiz(request: AdaptiveQuizRequest):
    """
    Generate personalized quiz questions based on engagement
    """
    try:
        if explainer is None:
            return _create_fallback_quiz(request)
        
        # Generate quiz using Gemini
        quiz = explainer.generate_adaptive_quiz(
            topic=request.topic,
            difficulty=request.difficulty,
            engagement_level=request.engagement_level,
            context=request.student_context or {}
        )
        
        return {
            "success": True,
            "question": quiz.get("question", ""),
            "question_type": quiz.get("question_type", "multiple_choice"),
            "options": quiz.get("options", []),
            "correct_answer": quiz.get("correct_answer", ""),
            "explanation": quiz.get("explanation", ""),
            "engagement_boost": quiz.get("engagement_boost", ""),
            "estimated_time": quiz.get("estimated_time", "1 minute"),
            "fun_factor": quiz.get("fun_factor", 7)
        }
        
    except Exception as e:
        return _create_fallback_quiz(request)

@router.post("/intervention")
async def suggest_intervention(request: InterventionRequest):
    """
    Get AI-powered intervention suggestions
    """
    try:
        if explainer is None:
            return _create_fallback_intervention(request)
        
        # Generate intervention using Gemini
        intervention = explainer.suggest_intervention(
            student_id=request.student_id,
            context=request.context
        )
        
        return {
            "success": True,
            "intervention_type": intervention.get("intervention_type", "gentle_check_in"),
            "suggested_actions": intervention.get("suggested_actions", []),
            "timing": intervention.get("timing", "within next 2-3 minutes"),
            "follow_up": intervention.get("follow_up", "Monitor for 5 minutes")
        }
        
    except Exception as e:
        return _create_fallback_intervention(request)

@router.get("/status")
async def get_gemini_status():
    """
    Check Gemini integration status
    """
    return {
        "gemini_available": explainer is not None,
        "api_key_configured": bool(os.getenv("GEMINI_API_KEY")),
        "status": "operational" if explainer else "fallback_mode",
        "timestamp": datetime.now().isoformat()
    }

# Fallback functions
def _create_simple_explanation(data: Dict[str, Any]):
    """Create a simple explanation when Gemini is unavailable"""
    attention_score = data.get('attention_score', 0.5)
    
    if attention_score > 0.8:
        return {
            "explanation": f"Student shows excellent engagement with high attention focus.",
            "risk_level": "low",
            "intervention_type": "none",
            "recommendations": ["Continue current approach"]
        }
    elif attention_score > 0.5:
        return {
            "explanation": f"Student shows moderate engagement with some distractions.",
            "risk_level": "medium",
            "intervention_type": "gentle",
            "recommendations": ["Gentle check-in", "Interactive content"]
        }
    else:
        return {
            "explanation": f"Student shows low engagement and may need attention.",
            "risk_level": "high",
            "intervention_type": "immediate", 
            "recommendations": ["Direct intervention", "Activity break"]
        }

def _create_fallback_class_analysis(request: ClassAnalysisRequest):
    """Create fallback class analysis"""
    avg_engagement = sum(s.get('attention_score', 0.5) for s in request.students) / len(request.students)
    
    return {
        "success": True,
        "overall_health": int(avg_engagement * 100),
        "engagement_trend": "stable",
        "class_mood": "focused" if avg_engagement > 0.6 else "needs_attention",
        "key_insights": [
            f"Class average engagement: {avg_engagement:.1%}",
            f"Topic difficulty appears {'appropriate' if avg_engagement > 0.5 else 'challenging'}",
            f"Session duration: {request.class_duration} minutes"
        ],
        "teaching_recommendations": [
            "Monitor low-engagement students",
            "Consider interactive activities",
            "Check pace appropriateness"
        ],
        "recommended_activity": "Continue monitoring",
        "optimal_break_time": "In 10-15 minutes"
    }

def _create_fallback_quiz(request: AdaptiveQuizRequest):
    """Create fallback quiz"""
    return {
        "success": True,
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

def _create_fallback_intervention(request: InterventionRequest):
    """Create fallback intervention"""
    return {
        "success": True,
        "intervention_type": "gentle_check_in",
        "suggested_actions": [
            "Brief one-on-one conversation",
            "Check if student needs clarification",
            "Offer alternative learning approach"
        ],
        "timing": "within next 2-3 minutes",
        "follow_up": "Monitor for 5 minutes after intervention"
    }
