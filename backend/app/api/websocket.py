"""
WebSocket endpoints for real-time video streaming and attention analysis
"""
import base64
import json
import logging
from typing import Any, Dict, List

import cv2
import numpy as np
from fastapi import Depends, WebSocket, WebSocketDisconnect
from sqlalchemy.orm import Session

from app.ai.attention_detector import AttentionData
from app.ai.working_inference import HighFidelityAttentionEngine
from app.core.auth import verify_token
from app.core.database import User, get_db
from app.models.session import SessionData

logger = logging.getLogger(__name__)
def convert_to_json_safe(obj: Any) -> Any:
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_safe(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_json_safe(v) for v in obj)
    else:
        return obj
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {
            "students": [],
            "teachers": []
        }
        self.student_data: Dict[str, AttentionData] = {}
        self.attention_detector = HighFidelityAttentionEngine()
    async def connect(self, websocket: WebSocket, user_type: str, user_id: str):
        logger.info(f"🔌 Attempting to accept WebSocket connection for {user_type}: {user_id}")
        await websocket.accept()
        logger.info(f"✅ WebSocket accepted for {user_type}: {user_id}")
        self.active_connections[user_type].append({
            "socket": websocket,
            "user_id": user_id
        })
        total_connections = len(self.active_connections["students"]) + len(self.active_connections["teachers"])
        logger.info(f"📊 New {user_type} connection: {user_id} (Total: {total_connections})")
        logger.info(f"📈 Active students: {len(self.active_connections['students'])}, teachers: {len(self.active_connections['teachers'])}")
    def disconnect(self, websocket: WebSocket, user_type: str):
        self.active_connections[user_type] = [
            conn for conn in self.active_connections[user_type] 
            if conn["socket"] != websocket
        ]
    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)
    async def broadcast_to_teachers(self, message: dict):
        message_str = json.dumps(message)
        for connection in self.active_connections["teachers"]:
            try:
                await connection["socket"].send_text(message_str)
            except Exception as e:
                logger.error(f"Failed to send message to teacher: {e}")
    async def process_video_frame(self, websocket: WebSocket, user_id: str, frame_data: str):
        try:
            if ',' in frame_data:
                image_data = base64.b64decode(frame_data.split(',')[1])
            else:
                image_data = base64.b64decode(frame_data)
            nparr = np.frombuffer(image_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is None:
                logger.warning("Could not decode frame")
                await self.send_personal_message(
                    json.dumps({"type": "error", "message": "Invalid frame data"}),
                    websocket
                )
                return
            try:
                prediction_result = await self.attention_detector.process_frame(frame)
                import time

                from app.ai.attention_detector import (AttentionLevel,
                                                       DistractionType)
                attention_score = prediction_result.get('attention_score', 0.5)
                if attention_score > 0.8:
                    attention_level = AttentionLevel.VERY_HIGH
                elif attention_score > 0.6:
                    attention_level = AttentionLevel.HIGH
                elif attention_score > 0.3:
                    attention_level = AttentionLevel.LOW
                else:
                    attention_level = AttentionLevel.VERY_LOW
                if prediction_result.get('phone_detected', False):
                    distraction_type = DistractionType.PHONE_USE
                elif not prediction_result.get('face_detected', True):
                    distraction_type = DistractionType.NOT_VISIBLE
                elif prediction_result.get('emotions', {}).get('boredom', 0) > 0.7:
                    distraction_type = DistractionType.HEAD_TURNED
                else:
                    distraction_type = DistractionType.NONE
                features_dict = prediction_result.get('features_used', [])
                if len(features_dict) >= 9:
                    head_pose = {
                        'yaw': float(features_dict[7]) if len(features_dict) > 7 else 0.0,
                        'pitch': float(features_dict[8]) if len(features_dict) > 8 else 0.0,
                        'roll': 0.0,  # Not available in current features
                        'pose_confidence': float(features_dict[15]) if len(features_dict) > 15 else 0.5
                    }
                else:
                    head_pose = {'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0, 'pose_confidence': 0.5}
                attention_data = AttentionData(
                    attention_level=attention_level,
                    confidence=convert_to_json_safe(prediction_result.get('confidence', 0.0)),
                    distraction_type=distraction_type,
                    face_detected=convert_to_json_safe(prediction_result.get('face_detected', False)),
                    head_pose=convert_to_json_safe(head_pose),
                    timestamp=time.time(),
                    features=convert_to_json_safe({
                        k: v for k, v in prediction_result.items() if k != 'emotions'
                    })
                )
                self.student_data[user_id] = attention_data
                emotions = prediction_result.get('emotions', {})
                safe_emotions = convert_to_json_safe(emotions)
                features_used = prediction_result.get('features_used', [])
                safe_features = convert_to_json_safe(features_used)
                student_feedback = {
                    "type": "attention_feedback",
                    "data": convert_to_json_safe({
                        "attention_level": attention_data.attention_level.name,
                        "attention_score": prediction_result.get('attention_score', 0.5),
                        "confidence": attention_data.confidence,
                        "distraction_type": attention_data.distraction_type.value,
                        "timestamp": attention_data.timestamp,
                        "face_detected": attention_data.face_detected,
                        "head_pose": attention_data.head_pose,
                        "emotions": safe_emotions,
                        "model_used": prediction_result.get('model_used', 'unknown'),
                        "temporal_smoothed": prediction_result.get('temporal_smoothed', False),
                        "raw_attention_score": prediction_result.get('raw_attention_score', 0.5),
                        "calibrated_score": prediction_result.get('calibrated_score', 0.5),
                        "emotion_attention_score": prediction_result.get('emotion_attention_score', 0.5),
                        "combined_score": prediction_result.get('combined_score', 0.5),
                        "fallback_used": prediction_result.get('fallback_used', False),
                        "fallback_reason": prediction_result.get('fallback_reason', None),
                        "comprehensive_analysis": prediction_result.get('comprehensive_analysis', {}),
                        "features_summary": {
                            "face_confidence": safe_features[5] if len(safe_features) > 5 else 0.0,
                            "face_area_ratio": safe_features[16] if len(safe_features) > 16 else 0.0,
                            "phone_detected": prediction_result.get('phone_detected', False),
                            "hands_detected": safe_features[6] if len(safe_features) > 6 else 0.0
                        },
                        "daisee_metrics": {
                            "engagement_level": prediction_result.get('comprehensive_analysis', {}).get('engagement_level', 'Unknown'),
                            "boredom_level": prediction_result.get('comprehensive_analysis', {}).get('boredom_level', 'Unknown'),
                            "confusion_level": prediction_result.get('comprehensive_analysis', {}).get('confusion_level', 'Unknown'),
                            "frustration_level": prediction_result.get('comprehensive_analysis', {}).get('frustration_level', 'Unknown'),
                            "emotional_state": prediction_result.get('comprehensive_analysis', {}).get('emotional_state', 'Unknown'),
                            "attention_quality": prediction_result.get('comprehensive_analysis', {}).get('attention_quality', 'Unknown'),
                            "overall_engagement": prediction_result.get('comprehensive_analysis', {}).get('overall_engagement', 0.5),
                            "distraction_risk": prediction_result.get('comprehensive_analysis', {}).get('distraction_risk', 0.3)
                        }
                    })
                }
                if prediction_result.get('fallback_used', False):
                    fallback_alert = {
                        "type": "fallback_alert",
                        "data": {
                            "message": "System using fallback analysis",
                            "reason": prediction_result.get('fallback_reason', 'Unknown'),
                            "timestamp": time.time(),
                            "severity": "warning"
                        }
                    }
                    await self.send_personal_message(json.dumps(fallback_alert), websocket)
                await self.send_personal_message(json.dumps(student_feedback), websocket)
                teacher_update = {
                    "type": "student_update",
                    "data": convert_to_json_safe({
                        "student_id": user_id,
                        "attention_level": attention_data.attention_level.name,
                        "confidence": attention_data.confidence,
                        "distraction_type": attention_data.distraction_type.value,
                        "face_detected": attention_data.face_detected,
                        "timestamp": attention_data.timestamp,
                        "features": attention_data.features
                    })
                }
                await self.broadcast_to_teachers(teacher_update)
            except Exception as analysis_error:
                logger.error(f"Attention analysis failed: {analysis_error}")
                await self.send_personal_message(
                    json.dumps({
                        "type": "error", 
                        "message": "Failed to analyze frame",
                        "details": str(analysis_error)
                    }),
                    websocket
                )
        except Exception as e:
            logger.error(f"Video frame processing failed: {e}")
            await self.send_personal_message(
                json.dumps({
                    "type": "error", 
                    "message": "Failed to process video frame",
                    "details": str(e)
                }),
                websocket
            )
    def get_classroom_summary(self) -> Dict:
        summary = {
            "total_students": len(self.student_data),
            "attention_distribution": {
                "VERY_HIGH": 0,
                "HIGH": 0,
                "LOW": 0,
                "VERY_LOW": 0
            },
            "distracted_students": [],
            "average_attention": 0.0
        }
        if not self.student_data:
            return convert_to_json_safe(summary)
        total_attention = 0
        for student_id, data in self.student_data.items():
            summary["attention_distribution"][data.attention_level.name] += 1
            if data.distraction_type.value != "none":
                summary["distracted_students"].append({
                    "student_id": student_id,
                    "distraction_type": data.distraction_type.value
                })
            attention_numeric = data.attention_level.value / 3.0
            total_attention += attention_numeric
        summary["average_attention"] = total_attention / len(self.student_data)
        return convert_to_json_safe(summary)
manager = ConnectionManager()
async def websocket_endpoint_student(websocket: WebSocket, user_id: str):
    logger.info(f"🔌 Student WebSocket connection attempt from {user_id}")
    try:
        await manager.connect(websocket, "students", user_id)
        logger.info(f"✅ Student {user_id} connected successfully")
        await manager.send_personal_message(
            json.dumps({
                "type": "connection_status",
                "status": "connected",
                "message": "Successfully connected to the DersLens WebSocket server"
            }),
            websocket
        )
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            logger.debug(f"📩 Received from {user_id}: {message.get('type', 'unknown')}")
            if message.get("type") == "video_frame":
                await manager.process_video_frame(
                    websocket, 
                    user_id, 
                    message.get("frame_data")
                )
            elif message.get("type") == "ping":
                logger.debug(f"🏓 Ping from {user_id}")
                await manager.send_personal_message(
                    json.dumps({"type": "pong"}), 
                    websocket
                )
    except WebSocketDisconnect:
        manager.disconnect(websocket, "students")
        logger.info(f"🔌 Student {user_id} disconnected")
    except Exception as e:
        logger.error(f"🚨 WebSocket error for student {user_id}: {e}")
        manager.disconnect(websocket, "students")
async def websocket_endpoint_teacher(websocket: WebSocket, user_id: str):
    await manager.connect(websocket, "teachers", user_id)
    try:
        summary = manager.get_classroom_summary()
        await manager.send_personal_message(
            json.dumps({
                "type": "classroom_summary",
                "data": summary
            }), 
            websocket
        )
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            if message.get("type") == "get_summary":
                summary = manager.get_classroom_summary()
                await manager.send_personal_message(
                    json.dumps({
                        "type": "classroom_summary",
                        "data": summary
                    }), 
                    websocket
                )
            elif message.get("type") == "ping":
                await manager.send_personal_message(
                    json.dumps({"type": "pong"}), 
                    websocket
                )
    except WebSocketDisconnect:
        manager.disconnect(websocket, "teachers")
        logger.info(f"Teacher {user_id} disconnected")
    except Exception as e:
        logger.error(f"WebSocket error for teacher {user_id}: {e}")
        manager.disconnect(websocket, "teachers")