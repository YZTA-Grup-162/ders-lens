
import google.generativeai as genai
import json
from typing import Dict, List, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

load_dotenv()

@dataclass
class EngagementInsight:
    """Structure for engagement insights"""
    student_id: str
    timestamp: datetime
    attention_score: float
    gaze_pattern: str
    explanation: str
    recommendations: List[str]
    risk_level: str  # "low", "medium", "high"
    intervention_type: str  # "none", "gentle", "immediate"

class GeminiExplainer:
    """Gemini-powered explainable AI for engagement analysis"""
    
    def __init__(self):
        genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        
    def explain_attention_pattern(self, 
                                attention_data: Dict[str, Any], 
                                historical_data: List[Dict] = None) -> EngagementInsight:
        """
        Generate intelligent explanation for student attention patterns
        """
        
        # Build context for Gemini
        context = self._build_context(attention_data, historical_data)
        
        prompt = f"""
        You are an expert educational psychologist analyzing student engagement patterns.
        
        STUDENT ENGAGEMENT DATA:
        {json.dumps(context, indent=2)}
        
        TASK: Provide a comprehensive analysis with:
        1. EXPLANATION: Why is this student showing this engagement pattern?
        2. RISK_ASSESSMENT: Rate the risk level (low/medium/high)
        3. INTERVENTION: What type of intervention is needed?
        4. RECOMMENDATIONS: 3 specific actionable recommendations
        5. POSITIVE_ASPECTS: What's working well?
        
        RESPOND IN JSON FORMAT:
        {{
            "explanation": "Clear, empathetic explanation of the engagement pattern",
            "risk_level": "low/medium/high",
            "intervention_type": "none/gentle/immediate",
            "recommendations": ["rec1", "rec2", "rec3"],
            "positive_aspects": ["aspect1", "aspect2"],
            "confidence_score": 0.85,
            "suggested_check_in_time": "in 5 minutes"
        }}
        
        Keep explanations educational and supportive, not judgmental.
        """
        
        try:
            response = self.model.generate_content(prompt)
            result = json.loads(response.text)
            
            return EngagementInsight(
                student_id=attention_data.get('student_id', 'unknown'),
                timestamp=datetime.now(),
                attention_score=attention_data.get('attention_score', 0.0),
                gaze_pattern=attention_data.get('gaze_pattern', 'unknown'),
                explanation=result['explanation'],
                recommendations=result['recommendations'],
                risk_level=result['risk_level'],
                intervention_type=result['intervention_type']
            )
            
        except Exception as e:
            # Fallback explanation
            return self._create_fallback_insight(attention_data)
    
    def generate_class_summary(self, class_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate intelligent class-wide engagement summary
        """
        
        prompt = f"""
        You are analyzing engagement data for an entire class session.
        
        CLASS ENGAGEMENT DATA:
        {json.dumps(class_data, indent=2)}
        
        TASK: Create a comprehensive class analysis with:
        1. OVERALL_HEALTH: Class engagement health score (0-100)
        2. KEY_INSIGHTS: 3 main insights about class dynamics
        3. TEACHING_RECOMMENDATIONS: 4 actionable teaching strategies
        4. STUDENT_HIGHLIGHTS: Students who need attention vs. those excelling
        5. OPTIMAL_BREAK_TIME: When should the next break be?
        6. ENGAGEMENT_TREND: Is engagement improving or declining?
        
        RESPOND IN JSON FORMAT:
        {{
            "overall_health": 85,
            "key_insights": ["insight1", "insight2", "insight3"],
            "teaching_recommendations": ["rec1", "rec2", "rec3", "rec4"],
            "students_needing_attention": ["student1", "student2"],
            "students_excelling": ["student3", "student4"],
            "optimal_break_time": "in 8 minutes",
            "engagement_trend": "improving",
            "class_mood": "focused but tired",
            "recommended_activity": "interactive quiz"
        }}
        """
        
        try:
            response = self.model.generate_content(prompt)
            return json.loads(response.text)
        except Exception as e:
            return self._create_fallback_class_summary(class_data)
    
    def generate_adaptive_quiz(self, topic: str, difficulty: str, student_engagement: float) -> Dict[str, Any]:
        """
        Generate personalized quiz questions based on engagement level
        """
        
        engagement_context = "highly engaged" if student_engagement > 0.8 else \
                           "moderately engaged" if student_engagement > 0.5 else "struggling to focus"
        
        prompt = f"""
        Create an adaptive quiz question for a {engagement_context} student.
        
        PARAMETERS:
        - Topic: {topic}
        - Difficulty: {difficulty}
        - Student engagement level: {student_engagement}
        
        TASK: Generate an engaging question that will re-capture attention:
        
        RESPOND IN JSON FORMAT:
        {{
            "question": "Interactive question text",
            "question_type": "multiple_choice/true_false/interactive",
            "options": ["option1", "option2", "option3", "option4"],
            "correct_answer": "option2",
            "explanation": "Why this answer is correct",
            "engagement_boost": "How this question helps attention",
            "estimated_time": "30 seconds",
            "fun_factor": 8
        }}
        
        Make it engaging and appropriate for the student's current attention state.
        """
        
        try:
            response = self.model.generate_content(prompt)
            return json.loads(response.text)
        except Exception as e:
            return self._create_fallback_quiz(topic, difficulty)
    
    def _build_context(self, attention_data: Dict, historical_data: List[Dict] = None) -> Dict:
        """Build rich context for Gemini analysis"""
        
        context = {
            "current_session": attention_data,
            "session_duration": attention_data.get('session_duration', 0),
            "time_of_day": datetime.now().strftime("%H:%M"),
            "day_of_week": datetime.now().strftime("%A")
        }
        
        if historical_data:
            context["historical_patterns"] = historical_data[-5:]  # Last 5 sessions
            
        return context
    
    def _create_fallback_insight(self, attention_data: Dict) -> EngagementInsight:
        """Create fallback insight when Gemini fails"""
        
        attention_score = attention_data.get('attention_score', 0.0)
        
        if attention_score > 0.8:
            explanation = "Student shows excellent engagement with consistent focus patterns."
            risk_level = "low"
            intervention_type = "none"
            recommendations = ["Continue current approach", "Consider advanced challenges", "Maintain positive reinforcement"]
        elif attention_score > 0.5:
            explanation = "Student shows moderate engagement with some distraction periods."
            risk_level = "medium"
            intervention_type = "gentle"
            recommendations = ["Brief check-in", "Interactive element", "Variety in content delivery"]
        else:
            explanation = "Student shows low engagement and may need immediate attention."
            risk_level = "high"
            intervention_type = "immediate"
            recommendations = ["Direct teacher intervention", "Break or activity change", "One-on-one check"]
        
        return EngagementInsight(
            student_id=attention_data.get('student_id', 'unknown'),
            timestamp=datetime.now(),
            attention_score=attention_score,
            gaze_pattern=attention_data.get('gaze_pattern', 'unknown'),
            explanation=explanation,
            recommendations=recommendations,
            risk_level=risk_level,
            intervention_type=intervention_type
        )
    
    def _create_fallback_class_summary(self, class_data: List[Dict]) -> Dict:
        """Create fallback class summary"""
        
        return {
            "overall_health": 75,
            "key_insights": [
                "Class showing mixed engagement levels",
                "Some students may need attention",
                "Overall learning environment is stable"
            ],
            "teaching_recommendations": [
                "Consider interactive break",
                "Check in with quieter students",
                "Use varied teaching methods",
                "Monitor energy levels"
            ],
            "students_needing_attention": [],
            "students_excelling": [],
            "optimal_break_time": "in 10 minutes",
            "engagement_trend": "stable",
            "class_mood": "focused",
            "recommended_activity": "group discussion"
        }
    
    def _create_fallback_quiz(self, topic: str, difficulty: str) -> Dict:
        """Create fallback quiz question"""
        
        return {
            "question": f"Quick check: What's the main concept we just covered about {topic}?",
            "question_type": "multiple_choice",
            "options": [
                "Core concept A",
                "Core concept B", 
                "Core concept C",
                "Need clarification"
            ],
            "correct_answer": "Core concept B",
            "explanation": "This helps reinforce the main learning objective",
            "engagement_boost": "Quick knowledge check to refocus attention",
            "estimated_time": "30 seconds",
            "fun_factor": 6
        }
