

from flask import Blueprint, request, jsonify
import google.generativeai as genai
import json
import os
from datetime import datetime
from typing import Dict, List, Any

from gemini_explainer import GeminiExplainer

gemini_bp = Blueprint('gemini', __name__)

# Initialize Gemini
explainer = GeminiExplainer()

@gemini_bp.route('/explain', methods=['POST'])
def explain_engagement():
    """
    Generate AI explanation for student engagement patterns
    """
    try:
        data = request.get_json()
        
        # Generate insight using Gemini
        insight = explainer.explain_attention_pattern(
            attention_data=data,
            historical_data=data.get('historical_data', [])
        )
        
        return jsonify({
            "success": True,
            "explanation": insight.explanation,
            "risk_level": insight.risk_level,
            "intervention_type": insight.intervention_type,
            "recommendations": insight.recommendations,
            "confidence_score": 0.85,
            "timestamp": insight.timestamp.isoformat()
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "fallback_explanation": _create_simple_explanation(data)
        }), 500

@gemini_bp.route('/class-analysis', methods=['POST'])
def analyze_class():
    """
    Generate comprehensive class engagement analysis
    """
    try:
        data = request.get_json()
        students = data.get('students', [])
        
        # Convert to format expected by explainer
        class_data = []
        for student in students:
            class_data.append({
                "student_id": student.get('student_id'),
                "attention_score": student.get('attention_score', 0.0),
                "gaze_pattern": student.get('gaze_pattern', 'unknown'),
                "session_duration": student.get('session_duration', 0)
            })
        
        # Generate class summary using Gemini
        summary = explainer.generate_class_summary(class_data)
        
        return jsonify({
            "success": True,
            **summary,
            "analysis_timestamp": datetime.now().isoformat(),
            "total_students": len(students)
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "fallback_summary": _create_simple_class_summary(data)
        }), 500

@gemini_bp.route('/adaptive-quiz', methods=['POST'])
def generate_quiz():
    """
    Generate adaptive quiz based on student engagement
    """
    try:
        data = request.get_json()
        
        topic = data.get('topic', 'current lesson')
        difficulty = data.get('difficulty', 'medium')
        engagement_level = data.get('engagement_level', 0.5)
        
        # Generate adaptive quiz using Gemini
        quiz = explainer.generate_adaptive_quiz(topic, difficulty, engagement_level)
        
        # Add engagement-specific modifications
        if engagement_level < 0.4:
            # For low engagement, make it more interactive
            quiz["engagement_boost"] = "🎯 Quick focus booster - you've got this!"
            quiz["encouragement"] = "Let's get back on track together!"
        elif engagement_level > 0.8:
            # For high engagement, make it more challenging
            quiz["engagement_boost"] = "🚀 Challenge mode activated!"
            quiz["encouragement"] = "You're doing amazing - let's push further!"
        
        return jsonify({
            "success": True,
            **quiz,
            "generated_at": datetime.now().isoformat(),
            "personalized_for_engagement": engagement_level
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "fallback_quiz": _create_simple_quiz(data)
        }), 500

@gemini_bp.route('/intervention', methods=['POST'])
def suggest_intervention():
    """
    Generate personalized intervention suggestions
    """
    try:
        data = request.get_json()
        student_id = data.get('student_id')
        context = data.get('context', {})
        
        # Create intervention prompt
        attention_score = context.get('attention_score', 0.5)
        time_distracted = context.get('time_distracted', 0)
        subject = context.get('subject', 'lesson')
        
        prompt = f"""
        Student intervention needed:
        - Student ID: {student_id}
        - Current attention score: {attention_score}
        - Time distracted: {time_distracted} seconds
        - Current subject: {subject}
        - Time of day: {datetime.now().strftime('%H:%M')}
        
        Provide a specific, actionable intervention strategy:
        
        Response in JSON:
        {{
            "intervention_type": "immediate/gentle/delayed",
            "specific_action": "Exact action for teacher",
            "duration": "How long to implement",
            "follow_up": "What to check after",
            "alternative_if_unsuccessful": "Backup plan",
            "privacy_respectful": true
        }}
        """
        
        response = explainer.model.generate_content(prompt)
        intervention = json.loads(response.text)
        
        return jsonify({
            "success": True,
            **intervention,
            "generated_at": datetime.now().isoformat(),
            "student_id": student_id
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "fallback_intervention": _create_simple_intervention(data)
        }), 500

@gemini_bp.route('/gamification', methods=['POST'])
def generate_gamification():
    """
    Generate gamified engagement activities
    """
    try:
        data = request.get_json()
        
        class_mood = data.get('class_mood', 'neutral')
        energy_level = data.get('energy_level', 'medium')
        subject = data.get('subject', 'general')
        duration_available = data.get('duration_available', 5)  # minutes
        
        prompt = f"""
        Create a gamified classroom activity:
        
        Context:
        - Class mood: {class_mood}
        - Energy level: {energy_level}
        - Subject: {subject}
        - Time available: {duration_available} minutes
        
        Generate an engaging activity that will re-energize the class:
        
        Response in JSON:
        {{
            "activity_name": "Fun name for the activity",
            "description": "How to implement it",
            "duration": "{duration_available} minutes",
            "materials_needed": ["item1", "item2"],
            "instructions": ["step1", "step2", "step3"],
            "success_metrics": "How to know it worked",
            "adaptations": "How to modify if needed",
            "fun_factor": 9
        }}
        """
        
        response = explainer.model.generate_content(prompt)
        activity = json.loads(response.text)
        
        return jsonify({
            "success": True,
            **activity,
            "generated_at": datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "fallback_activity": _create_simple_activity(data)
        }), 500

@gemini_bp.route('/privacy-explanation', methods=['GET'])
def explain_privacy():
    """
    Generate clear, understandable privacy explanation
    """
    try:
        prompt = """
        Explain how DersLens protects student privacy in simple, clear terms:
        
        Cover these points:
        1. What data we collect
        2. How we process it
        3. What we DON'T store
        4. Student rights
        5. Teacher controls
        
        Make it parent and student friendly.
        
        Response in JSON:
        {
            "simple_explanation": "Easy to understand summary",
            "data_collected": ["item1", "item2"],
            "data_not_stored": ["item1", "item2"], 
            "student_rights": ["right1", "right2"],
            "teacher_controls": ["control1", "control2"],
            "compliance": ["GDPR", "COPPA", "etc"]
        }
        """
        
        response = explainer.model.generate_content(prompt)
        privacy_info = json.loads(response.text)
        
        return jsonify({
            "success": True,
            **privacy_info,
            "last_updated": datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "fallback_privacy": {
                "simple_explanation": "DersLens only analyzes attention patterns without storing personal data",
                "data_collected": ["Anonymized attention scores", "Gaze direction (not images)"],
                "data_not_stored": ["Student faces", "Personal information", "Video recordings"],
                "student_rights": ["Opt out anytime", "Data deletion on request"],
                "teacher_controls": ["Full access control", "Session-based data only"]
            }
        }), 500

# Helper functions for fallbacks
def _create_simple_explanation(data):
    """Simple fallback explanation"""
    attention_score = data.get('attention_score', 0.5)
    
    if attention_score > 0.8:
        return "Student showing excellent focus and engagement"
    elif attention_score > 0.5:
        return "Student showing moderate engagement with some distraction"
    else:
        return "Student may benefit from attention and support"

def _create_simple_class_summary(data):
    """Simple fallback class summary"""
    students = data.get('students', [])
    if not students:
        return {"overall_health": 50, "key_insights": ["No student data available"]}
    
    avg_attention = sum(s.get('attention_score', 0.5) for s in students) / len(students)
    
    return {
        "overall_health": int(avg_attention * 100),
        "key_insights": [
            f"Class average attention: {avg_attention:.1%}",
            f"Total students: {len(students)}",
            "Monitor individual students for personalized support"
        ],
        "teaching_recommendations": [
            "Consider interactive break",
            "Check in with students",
            "Use varied teaching methods"
        ]
    }

def _create_simple_quiz(data):
    """Simple fallback quiz"""
    topic = data.get('topic', 'current lesson')
    
    return {
        "question": f"Quick check: What's the main point about {topic}?",
        "question_type": "multiple_choice",
        "options": [
            "Main concept A",
            "Main concept B",
            "Main concept C", 
            "Need clarification"
        ],
        "correct_answer": "Main concept B",
        "explanation": "This helps reinforce learning",
        "estimated_time": "30 seconds"
    }

def _create_simple_intervention(data):
    """Simple fallback intervention"""
    return {
        "intervention_type": "gentle",
        "specific_action": "Brief check-in with student",
        "duration": "1-2 minutes",
        "follow_up": "Monitor engagement for next 5 minutes",
        "alternative_if_unsuccessful": "Consider break or activity change"
    }

def _create_simple_activity(data):
    """Simple fallback activity"""
    return {
        "activity_name": "Quick Energizer",
        "description": "Simple interactive activity to re-engage class",
        "duration": "3-5 minutes",
        "instructions": [
            "Ask students to stand and stretch",
            "Quick review question",
            "Partner discussion for 2 minutes"
        ],
        "success_metrics": "Increased student participation",
        "fun_factor": 6
    }
