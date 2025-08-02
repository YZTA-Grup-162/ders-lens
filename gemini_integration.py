
import asyncio
import base64
import json
import logging
from datetime import datetime, timedelta
from io import BytesIO
from typing import Dict, List, Optional, Tuple

import cv2
import google.generativeai as genai
import numpy as np
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GeminiTeachingAssistant:
    """AI Teaching Assistant powered by Gemini for contextual analysis"""
    
    def __init__(self, api_key: str, model_name: str = "gemini-1.5-pro"):
        """Initialize Gemini Teaching Assistant"""
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        self.session_data = []
        self.current_insights = {}
        
    def encode_frame_for_gemini(self, frame: np.ndarray) -> str:
        """Convert OpenCV frame to base64 for Gemini vision"""
        _, buffer = cv2.imencode('.jpg', frame)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        return img_base64
    
    async def analyze_engagement_context(self, 
                                       frame: np.ndarray,
                                       cv_results: Dict,
                                       timestamp: datetime) -> Dict:
        """
        Use Gemini to provide contextual analysis of student engagement
        
        Args:
            frame: Current video frame
            cv_results: Computer vision results (emotions, gaze, etc.)
            timestamp: Current timestamp
            
        Returns:
            Dict with Gemini's contextual insights
        """
        
        # Prepare context for Gemini
        context_prompt = self._build_engagement_prompt(cv_results, timestamp)
        
        try:
            # For multi-modal analysis, include both frame and context
            img_base64 = self.encode_frame_for_gemini(frame)
            
            response = await self.model.generate_content_async([
                context_prompt,
                {
                    "mime_type": "image/jpeg",
                    "data": img_base64
                }
            ])
            
            # Parse Gemini response
            insights = self._parse_gemini_response(response.text)
            
            # Store for historical analysis
            self.session_data.append({
                "timestamp": timestamp,
                "cv_results": cv_results,
                "gemini_insights": insights
            })
            
            return insights
            
        except Exception as e:
            logger.error(f"Gemini analysis error: {e}")
            return {"error": str(e), "fallback_analysis": self._fallback_analysis(cv_results)}
    
    def _build_engagement_prompt(self, cv_results: Dict, timestamp: datetime) -> str:
        """Build comprehensive prompt for Gemini analysis"""
        
        # Extract key metrics
        emotions = cv_results.get('emotions', {})
        gaze_data = cv_results.get('gaze', {})
        attention_score = cv_results.get('attention_score', 0)
        
        prompt = f"""
        Analyze this student's engagement state as an AI teaching assistant:
        
        COMPUTER VISION DATA:
        - Timestamp: {timestamp.strftime('%H:%M:%S')}
        - Detected Emotions: {emotions}
        - Gaze Direction: {gaze_data}
        - Attention Score: {attention_score}/100
        - Head Pose: {cv_results.get('head_pose', 'Unknown')}
        
        CONTEXT:
        - Class Duration: {self._get_class_duration()}
        - Recent Trend: {self._get_recent_trend()}
        
        Please provide:
        1. ENGAGEMENT_LEVEL: Scale 1-10 with reasoning
        2. EMOTIONAL_STATE: Primary emotion and confidence
        3. ATTENTION_INDICATORS: Specific behavioral observations
        4. RECOMMENDATIONS: Immediate teaching suggestions
        5. ALERTS: Any concerning patterns (yes/no with reason)
        
        Respond in JSON format for easy parsing.
        Be concise but insightful. Focus on actionable teaching insights.
        """
        
        return prompt
    
    def _parse_gemini_response(self, response_text: str) -> Dict:
        """Parse Gemini's response into structured insights"""
        try:
            # Try to parse as JSON first
            if response_text.strip().startswith('{'):
                return json.loads(response_text)
            
            # Fallback: Extract key insights with regex/parsing
            insights = {
                "engagement_level": self._extract_engagement_level(response_text),
                "emotional_state": self._extract_emotional_state(response_text),
                "recommendations": self._extract_recommendations(response_text),
                "alerts": self._extract_alerts(response_text),
                "raw_response": response_text
            }
            
            return insights
            
        except Exception as e:
            logger.warning(f"Failed to parse Gemini response: {e}")
            return {
                "error": "parsing_failed",
                "raw_response": response_text
            }
    
    def _extract_engagement_level(self, text: str) -> Dict:
        """Extract engagement level from text response"""
        # Simple regex patterns for extraction
        import re
        
        level_match = re.search(r'engagement.*?(\d+)\/10', text, re.IGNORECASE)
        if level_match:
            return {
                "score": int(level_match.group(1)),
                "max_score": 10
            }
        return {"score": 5, "max_score": 10, "confidence": "low"}
    
    def _extract_emotional_state(self, text: str) -> Dict:
        """Extract emotional state analysis"""
        emotions = ['focused', 'confused', 'bored', 'engaged', 'frustrated', 'curious']
        detected_emotions = []
        
        for emotion in emotions:
            if emotion.lower() in text.lower():
                detected_emotions.append(emotion)
        
        return {
            "primary_emotion": detected_emotions[0] if detected_emotions else "neutral",
            "detected_emotions": detected_emotions
        }
    
    def _extract_recommendations(self, text: str) -> List[str]:
        """Extract teaching recommendations"""
        # Look for recommendation sections
        import re
        
        rec_pattern = r'recommend.*?:\s*(.+?)(?:\n|$)'
        matches = re.findall(rec_pattern, text, re.IGNORECASE | re.DOTALL)
        
        recommendations = []
        for match in matches:
            # Clean and split recommendations
            recs = [r.strip() for r in match.split('.') if r.strip()]
            recommendations.extend(recs)
        
        return recommendations[:3]  # Limit to top 3
    
    def _extract_alerts(self, text: str) -> Dict:
        """Extract any alerts or concerns"""
        alert_keywords = ['concern', 'alert', 'warning', 'attention', 'urgent']
        
        has_alerts = any(keyword in text.lower() for keyword in alert_keywords)
        
        return {
            "has_alerts": has_alerts,
            "alert_text": text if has_alerts else None
        }
    
    def _get_class_duration(self) -> str:
        """Calculate current class duration"""
        if not self.session_data:
            return "Just started"
        
        start_time = self.session_data[0]["timestamp"]
        duration = datetime.now() - start_time
        
        minutes = int(duration.total_seconds() / 60)
        return f"{minutes} minutes"
    
    def _get_recent_trend(self) -> str:
        """Analyze recent engagement trend"""
        if len(self.session_data) < 3:
            return "Insufficient data"
        
        recent_scores = [
            entry["cv_results"].get("attention_score", 50) 
            for entry in self.session_data[-5:]
        ]
        
        if len(recent_scores) >= 2:
            trend = recent_scores[-1] - recent_scores[0]
            if trend > 10:
                return "Improving"
            elif trend < -10:
                return "Declining"
        
        return "Stable"
    
    def _fallback_analysis(self, cv_results: Dict) -> Dict:
        """Provide basic analysis when Gemini fails"""
        attention_score = cv_results.get('attention_score', 50)
        
        if attention_score > 80:
            engagement = "High"
            recommendation = "Student is highly engaged. Continue current approach."
        elif attention_score > 60:
            engagement = "Moderate"
            recommendation = "Good engagement. Consider interactive elements."
        else:
            engagement = "Low"
            recommendation = "Low engagement detected. Try interactive question or break."
        
        return {
            "engagement_level": {"score": attention_score // 10, "max_score": 10},
            "emotional_state": {"primary_emotion": "neutral"},
            "recommendations": [recommendation],
            "alerts": {"has_alerts": attention_score < 40},
            "source": "fallback_analysis"
        }

class GeminiDashboardAssistant:
    """Conversational AI assistant for teacher dashboard"""
    
    def __init__(self, api_key: str, teaching_assistant: GeminiTeachingAssistant):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-1.5-pro")
        self.teaching_assistant = teaching_assistant
        self.conversation_history = []
    
    async def answer_teacher_query(self, query: str) -> Dict:
        """
        Answer teacher's conversational queries about class engagement
        
        Examples:
        - "Who looks disengaged right now?"
        - "Summarize the class mood today"
        - "Should I take a break?"
        """
        
        # Build context from recent session data
        context = self._build_dashboard_context()
        
        conversation_prompt = f"""
        You are an AI teaching assistant analyzing student engagement data.
        
        CURRENT CLASS CONTEXT:
        {context}
        
        TEACHER QUESTION: "{query}"
        
        Provide a helpful, actionable response as a teaching assistant would.
        Be specific, concise, and focus on practical teaching suggestions.
        Use the engagement data to support your recommendations.
        
        If asked about specific students, refer to them by position (e.g., "Student in front row, left side").
        """
        
        try:
            response = await self.model.generate_content_async(conversation_prompt)
            
            # Store conversation
            self.conversation_history.append({
                "timestamp": datetime.now(),
                "query": query,
                "response": response.text
            })
            
            return {
                "response": response.text,
                "confidence": "high",
                "suggestions": self._extract_action_items(response.text)
            }
            
        except Exception as e:
            logger.error(f"Dashboard query error: {e}")
            return {
                "response": "I'm having trouble accessing the analysis right now. Please try again.",
                "confidence": "low",
                "error": str(e)
            }
    
    def _build_dashboard_context(self) -> str:
        """Build context summary for dashboard queries"""
        recent_data = self.teaching_assistant.session_data[-10:]  # Last 10 entries
        
        if not recent_data:
            return "No recent engagement data available."
        
        # Aggregate recent insights
        avg_attention = np.mean([
            entry["cv_results"].get("attention_score", 50) 
            for entry in recent_data
        ])
        
        recent_emotions = []
        for entry in recent_data:
            emotions = entry["cv_results"].get("emotions", {})
            if emotions:
                top_emotion = max(emotions.items(), key=lambda x: x[1])
                recent_emotions.append(top_emotion[0])
        
        context = f"""
        CLASS SUMMARY:
        - Duration: {self.teaching_assistant._get_class_duration()}
        - Average Attention: {avg_attention:.1f}/100
        - Trend: {self.teaching_assistant._get_recent_trend()}
        - Common Emotions: {', '.join(set(recent_emotions[-5:]))}
        - Total Students Monitored: {len(recent_data)}
        """
        
        return context
    
    def _extract_action_items(self, response: str) -> List[str]:
        """Extract actionable suggestions from response"""
        action_words = ['suggest', 'recommend', 'try', 'consider', 'should']
        sentences = response.split('.')
        
        action_items = []
        for sentence in sentences:
            if any(word in sentence.lower() for word in action_words):
                action_items.append(sentence.strip())
        
        return action_items[:3]  # Top 3 suggestions

class GeminiSmartAlerts:
    """Intelligent alert system using Gemini's reasoning"""
    
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-1.5-flash")  # Faster model for alerts
        self.alert_history = []
        self.alert_thresholds = {
            "low_engagement": 40,
            "sustained_distraction": 30,  # seconds
            "emotional_concern": 0.8  # confidence threshold
        }
    
    async def check_smart_alerts(self, session_data: List[Dict]) -> List[Dict]:
        """Generate intelligent alerts based on engagement patterns"""
        
        if len(session_data) < 5:  # Need some data to analyze
            return []
        
        recent_data = session_data[-10:]
        alerts = []
        
        # 1. Check for sustained low engagement
        low_engagement_alert = await self._check_sustained_low_engagement(recent_data)
        if low_engagement_alert:
            alerts.append(low_engagement_alert)
        
        # 2. Check for emotional concerns
        emotional_alert = await self._check_emotional_concerns(recent_data)
        if emotional_alert:
            alerts.append(emotional_alert)
        
        # 3. Check for learning opportunity
        learning_opportunity = await self._check_learning_opportunity(recent_data)
        if learning_opportunity:
            alerts.append(learning_opportunity)
        
        return alerts
    
    async def _check_sustained_low_engagement(self, data: List[Dict]) -> Optional[Dict]:
        """Check for sustained periods of low engagement"""
        
        low_scores = [
            entry for entry in data 
            if entry["cv_results"].get("attention_score", 50) < self.alert_thresholds["low_engagement"]
        ]
        
        if len(low_scores) >= 3:  # 3+ consecutive low scores
            context = f"Student showing {len(low_scores)} consecutive low engagement readings"
            
            prompt = f"""
            ALERT ANALYSIS REQUEST:
            {context}
            
            Recent attention scores: {[entry["cv_results"].get("attention_score", 50) for entry in data]}
            
            Should this trigger a teaching intervention alert?
            Respond with: YES/NO and brief reasoning (max 50 words).
            """
            
            try:
                response = await self.model.generate_content_async(prompt)
                
                if "YES" in response.text.upper():
                    return {
                        "type": "sustained_low_engagement",
                        "severity": "medium",
                        "message": "Student showing sustained low engagement",
                        "recommendation": "Consider interactive question or brief check-in",
                        "timestamp": datetime.now(),
                        "reasoning": response.text
                    }
            except Exception as e:
                logger.error(f"Alert check error: {e}")
        
        return None
    
    async def _check_emotional_concerns(self, data: List[Dict]) -> Optional[Dict]:
        """Check for concerning emotional patterns"""
        
        recent_emotions = []
        for entry in data:
            emotions = entry["cv_results"].get("emotions", {})
            if emotions:
                # Get emotions with high confidence
                high_conf_emotions = {
                    emotion: score for emotion, score in emotions.items() 
                    if score > self.alert_thresholds["emotional_concern"]
                }
                recent_emotions.extend(high_conf_emotions.keys())
        
        concerning_emotions = ['angry', 'frustrated', 'sad', 'confused']
        concern_count = sum(1 for emotion in recent_emotions if emotion in concerning_emotions)
        
        if concern_count >= 2:  # Multiple concerning emotions
            return {
                "type": "emotional_concern",
                "severity": "high",
                "message": f"Detected concerning emotions: {', '.join(set(recent_emotions))}",
                "recommendation": "Consider checking if student needs help or clarification",
                "timestamp": datetime.now()
            }
        
        return None
    
    async def _check_learning_opportunity(self, data: List[Dict]) -> Optional[Dict]:
        """Identify good moments for interactive teaching"""
        
        # Check for high engagement + neutral/positive emotion
        recent_entry = data[-1]
        attention_score = recent_entry["cv_results"].get("attention_score", 50)
        emotions = recent_entry["cv_results"].get("emotions", {})
        
        if attention_score > 75:  # High attention
            positive_emotions = ['happy', 'focused', 'engaged', 'curious']
            has_positive_emotion = any(
                emotion in emotions and emotions[emotion] > 0.7 
                for emotion in positive_emotions
            )
            
            if has_positive_emotion:
                return {
                    "type": "learning_opportunity",
                    "severity": "low",
                    "message": "High engagement detected - great time for interaction!",
                    "recommendation": "Consider asking a question or introducing new concept",
                    "timestamp": datetime.now()
                }
        
        return None

# Example usage and integration helpers
class GeminiIntegrationManager:
    """Main manager for all Gemini integrations"""
    
    def __init__(self, api_key: str):
        self.teaching_assistant = GeminiTeachingAssistant(api_key)
        self.dashboard_assistant = GeminiDashboardAssistant(api_key, self.teaching_assistant)
        self.smart_alerts = GeminiSmartAlerts(api_key)
        
    async def process_frame_analysis(self, 
                                   frame: np.ndarray, 
                                   cv_results: Dict) -> Dict:
        """Complete analysis pipeline for a single frame"""
        
        timestamp = datetime.now()
        
        # Get Gemini insights
        insights = await self.teaching_assistant.analyze_engagement_context(
            frame, cv_results, timestamp
        )
        
        # Check for alerts
        alerts = await self.smart_alerts.check_smart_alerts(
            self.teaching_assistant.session_data
        )
        
        return {
            "timestamp": timestamp,
            "cv_results": cv_results,
            "gemini_insights": insights,
            "alerts": alerts
        }
    
    async def handle_teacher_query(self, query: str) -> Dict:
        """Handle conversational queries from teacher dashboard"""
        return await self.dashboard_assistant.answer_teacher_query(query)
    
    def get_session_summary(self) -> Dict:
        """Generate comprehensive session summary"""
        if not self.teaching_assistant.session_data:
            return {"error": "No session data available"}
        
        data = self.teaching_assistant.session_data
        
        # Calculate summary statistics
        attention_scores = [entry["cv_results"].get("attention_score", 50) for entry in data]
        avg_attention = np.mean(attention_scores)
        min_attention = np.min(attention_scores)
        max_attention = np.max(attention_scores)
        
        # Get trend
        trend = self.teaching_assistant._get_recent_trend()
        duration = self.teaching_assistant._get_class_duration()
        
        return {
            "session_duration": duration,
            "average_attention": avg_attention,
            "attention_range": {"min": min_attention, "max": max_attention},
            "trend": trend,
            "total_alerts": len([
                alert for entry in data 
                for alert in entry.get("alerts", [])
            ]),
            "recommendations": self._get_top_recommendations()
        }
    
    def _get_top_recommendations(self) -> List[str]:
        """Get most common recommendations from session"""
        all_recommendations = []
        
        for entry in self.teaching_assistant.session_data:
            insights = entry.get("gemini_insights", {})
            recommendations = insights.get("recommendations", [])
            all_recommendations.extend(recommendations)
        
        # Count frequency and return top ones
        from collections import Counter
        common_recs = Counter(all_recommendations).most_common(3)
        
        return [rec[0] for rec in common_recs]

# Configuration helper
def setup_gemini_integration(api_key: str) -> GeminiIntegrationManager:
    """Setup complete Gemini integration for DersLens"""
    
    if not api_key:
        raise ValueError("Gemini API key is required")
    
    manager = GeminiIntegrationManager(api_key)
    
    logger.info("Gemini integration initialized successfully")
    logger.info("Available features:")
    logger.info("- Multi-modal engagement analysis")
    logger.info("- Conversational teacher assistant")
    logger.info("- Smart alerts and recommendations")
    logger.info("- Session summaries and insights")
    
    return manager

if __name__ == "__main__":
    # Example usage
    import os

    api_key = os.getenv("GEMINI_API_KEY")
    
    if api_key:
        manager = setup_gemini_integration(api_key)
        print("Gemini integration ready for DersLens!")
    else:
        print("Please set GEMINI_API_KEY environment variable")
