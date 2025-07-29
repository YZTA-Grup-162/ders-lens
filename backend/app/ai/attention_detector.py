"""
Core attention detection module using OpenCV and MediaPipe
"""

import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

try:
    from backend.app.ai.enhanced_dataset_integration import \
        EnhancedAttentionDetector
    ENHANCED_FEATURES_AVAILABLE = True
except ImportError:
    try:
        from enhanced_dataset_integration import EnhancedAttentionDetector
        ENHANCED_FEATURES_AVAILABLE = True
    except ImportError:
        ENHANCED_FEATURES_AVAILABLE = False
logger = logging.getLogger(__name__)
if not ENHANCED_FEATURES_AVAILABLE:
    logger.warning("Enhanced features not available. Install scikit-learn for full functionality.")
class AttentionLevel(Enum):
    VERY_LOW = 0
    LOW = 1
    HIGH = 2
    VERY_HIGH = 3
class DistractionType(Enum):
    NONE = "none"
    PHONE_USE = "phone_use"
    HEAD_TURNED = "head_turned"
    EYES_AWAY = "eyes_away"
    NOT_VISIBLE = "not_visible"
@dataclass
class AttentionData:
    attention_level: AttentionLevel
    confidence: float
    distraction_type: DistractionType
    face_detected: bool
    head_pose: Dict[str, float]
    timestamp: float
    features: Dict[str, float]
class AttentionDetector:

    def __init__(self, confidence_threshold: float = 0.5, use_enhanced: bool = True):
        self.confidence_threshold = confidence_threshold
        self.use_enhanced = use_enhanced and ENHANCED_FEATURES_AVAILABLE
        if self.use_enhanced:
            try:
                self.enhanced_detector = EnhancedAttentionDetector(use_daisee=False)
                logger.info("✅ Enhanced attention detector initialized")
            except Exception as e:
                logger.warning(f"⚠️ Could not initialize enhanced detector: {e}")
                self.enhanced_detector = None
                self.use_enhanced = False
        else:
            self.enhanced_detector = None
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_hands = mp.solutions.hands
        self.mp_face_detection = mp.solutions.face_detection
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=0.5
        )
        self.attention_history: List[float] = []
        self.max_history = 10
    def detect_face_and_pose(self, frame: np.ndarray) -> Tuple[bool, Dict[str, float]]:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        if not results.multi_face_landmarks:
            return False, {}
        face_landmarks = results.multi_face_landmarks[0]
        h, w = frame.shape[:2]
        landmarks_3d = []
        landmarks_2d = []
        key_indices = [1, 33, 263, 61, 291, 199]
        for idx in key_indices:
            landmark = face_landmarks.landmark[idx]
            x, y = int(landmark.x * w), int(landmark.y * h)
            landmarks_2d.append([x, y])
            landmarks_3d.append([landmark.x * w, landmark.y * h, landmark.z * 3000])
        landmarks_3d = np.array(landmarks_3d, dtype=np.float32)
        landmarks_2d = np.array(landmarks_2d, dtype=np.float32)
        focal_length = w
        cam_matrix = np.array([[focal_length, 0, w/2],
                              [0, focal_length, h/2],
                              [0, 0, 1]], dtype=np.float32)
        dist_coeffs = np.zeros((4, 1), dtype=np.float32)
        object_3d = np.array([
            [0.0, 0.0, 0.0],
            [0.0, -330.0, -65.0],
            [-225.0, 170.0, -135.0],
            [225.0, 170.0, -135.0],
            [-150.0, -150.0, -125.0],
            [150.0, -150.0, -125.0]
        ], dtype=np.float32)
        try:
            success, rotation_vec, translation_vec = cv2.solvePnP(
                object_3d, landmarks_2d, cam_matrix, dist_coeffs
            )
            if success:
                rotation_mat, _ = cv2.Rodrigues(rotation_vec)
                angles = self._rotation_matrix_to_euler_angles(rotation_mat)
                return True, {
                    'pitch': angles[0],  # Up/down
                    'yaw': angles[1],    # Left/right
                    'roll': angles[2]    # Tilt
                }
        except Exception as e:
            logger.warning(f"Head pose estimation failed: {e}")
        return True, {}
    def detect_phone_usage(self, frame: np.ndarray) -> bool:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        if not results.multi_hand_landmarks:
            return False
        h, w = frame.shape[:2]
        for hand_landmarks in results.multi_hand_landmarks:
            hand_points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks.landmark]
            avg_y = sum(p[1] for p in hand_points) / len(hand_points)
            avg_x = sum(p[0] for p in hand_points) / len(hand_points)
            if avg_y < h * 0.6 and w * 0.3 < avg_x < w * 0.7:
                return True
        return False
    def analyze_attention(self, frame: np.ndarray) -> AttentionData:
        timestamp = time.time()
        if self.use_enhanced and self.enhanced_detector is not None:
            try:
                enhanced_result = self.enhanced_detector.predict_attention(frame)
                return self._convert_enhanced_to_attention_data(enhanced_result, timestamp)
            except Exception as e:
                logger.warning(f"Enhanced detector failed, falling back to basic: {e}")
        return self._analyze_attention_basic(frame, timestamp)
    def _convert_enhanced_to_attention_data(self, enhanced_result: Dict, timestamp: float) -> AttentionData:
        attention_score = enhanced_result.get('attention_score', 0.5)
        prediction = enhanced_result.get('prediction', 'unknown')
        confidence = enhanced_result.get('confidence', 0.5)
        face_detected = enhanced_result.get('face_detected', False)
        phone_detected = enhanced_result.get('phone_detected', False)
        if attention_score >= 0.8:
            attention_level = AttentionLevel.VERY_HIGH
        elif attention_score >= 0.6:
            attention_level = AttentionLevel.HIGH
        elif attention_score >= 0.4:
            attention_level = AttentionLevel.LOW
        else:
            attention_level = AttentionLevel.VERY_LOW
        if not face_detected:
            distraction_type = DistractionType.NOT_VISIBLE
        elif phone_detected:
            distraction_type = DistractionType.PHONE_USE
        elif prediction == 'distracted':
            distraction_type = DistractionType.HEAD_TURNED
        else:
            distraction_type = DistractionType.NONE
        features = {
            'enhanced_attention_score': attention_score,
            'enhanced_confidence': confidence,
            'face_detected': face_detected,
            'phone_detected': phone_detected,
            'model_used': enhanced_result.get('model_used', 'unknown')
        }
        pose_weight = enhanced_result.get('pose_attention_weight', 0.5)
        # Extract actual head pose from enhanced result if available
        head_pose_data = enhanced_result.get('head_pose', {})
        head_pose = {
            'pose_attention_weight': pose_weight,
            'yaw': head_pose_data.get('yaw', 0.0),
            'pitch': head_pose_data.get('pitch', 0.0),
            'roll': head_pose_data.get('roll', 0.0)
        }
        return AttentionData(
            attention_level=attention_level,
            confidence=confidence,
            distraction_type=distraction_type,
            face_detected=face_detected,
            head_pose=head_pose,
            timestamp=timestamp,
            features=features
        )
    def _analyze_attention_basic(self, frame: np.ndarray, timestamp: float) -> AttentionData:
        features = {
            'face_area': 0.0,
            'face_center_x': 0.0,
            'face_center_y': 0.0,
            'num_faces': 0,
            'num_hands': 0,
            'phone_detected': 0.0
        }
        face_detected, head_pose = self.detect_face_and_pose(frame)
        phone_usage = self.detect_phone_usage(frame)
        features['phone_detected'] = 1.0 if phone_usage else 0.0
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_results = self.face_detection.process(rgb_frame)
        if face_results.detections:
            features['num_faces'] = len(face_results.detections)
            detection = face_results.detections[0]
            bbox = detection.location_data.relative_bounding_box
            h, w = frame.shape[:2]
            features['face_area'] = bbox.width * bbox.height
            features['face_center_x'] = bbox.xmin + bbox.width / 2
            features['face_center_y'] = bbox.ymin + bbox.height / 2
        distraction_type = DistractionType.NONE
        if not face_detected:
            distraction_type = DistractionType.NOT_VISIBLE
        elif phone_usage:
            distraction_type = DistractionType.PHONE_USE
        elif head_pose and abs(head_pose.get('yaw', 0)) > 30:
            distraction_type = DistractionType.HEAD_TURNED
        attention_score = self._calculate_attention_score(
            face_detected, head_pose, phone_usage, features
        )
        if attention_score >= 0.8:
            attention_level = AttentionLevel.VERY_HIGH
        elif attention_score >= 0.6:
            attention_level = AttentionLevel.HIGH
        elif attention_score >= 0.4:
            attention_level = AttentionLevel.LOW
        else:
            attention_level = AttentionLevel.VERY_LOW
        self.attention_history.append(attention_score)
        if len(self.attention_history) > self.max_history:
            self.attention_history.pop(0)
        confidence = self._calculate_confidence()
        return AttentionData(
            attention_level=attention_level,
            confidence=confidence,
            distraction_type=distraction_type,
            face_detected=face_detected,
            head_pose=head_pose,
            timestamp=timestamp,
            features=features
        )
    def _calculate_attention_score(self, face_detected: bool, head_pose: Dict[str, float], phone_usage: bool, features: Dict[str, float]) -> float:
        score = 0.0
        if face_detected:
            score += 0.3
        if head_pose:
            yaw = abs(head_pose.get('yaw', 0))
            pitch = abs(head_pose.get('pitch', 0))
            if yaw < 15 and pitch < 20:
                score += 0.4
            elif yaw < 30 and pitch < 30:
                score += 0.2
        if not phone_usage:
            score += 0.3
        if features['face_area'] > 0.05:  # Face is large enough (student is close):
            score += 0.1
        return max(0.0, min(1.0, score))
    def _calculate_confidence(self) -> float:
        if len(self.attention_history) < 3:
            return 0.5
        recent_scores = self.attention_history[-5:]
        variance = np.var(recent_scores)
        confidence = max(0.1, 1.0 - variance * 2)
        return min(1.0, confidence)
    def _rotation_matrix_to_euler_angles(self, R: np.ndarray) -> np.ndarray:
        sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        singular = sy < 1e-6
        if not singular:
            x = np.arctan2(R[2, 1], R[2, 2])
            y = np.arctan2(-R[2, 0], sy)
            z = np.arctan2(R[1, 0], R[0, 0])
        else:
            x = np.arctan2(-R[1, 2], R[1, 1])
            y = np.arctan2(-R[2, 0], sy)
            z = 0
        return np.degrees(np.array([x, y, z]))
    def get_attention_summary(self, duration_minutes: int = 5) -> Dict[str, float]:
        if not self.attention_history:
            return {
                'average_attention': 0.0,
                'attention_percentage': 0.0,
                'distraction_count': 0,
                'engagement_trend': 'stable'
            }
        recent_scores = self.attention_history[-duration_minutes * 60:]
        avg_attention = np.mean(recent_scores)
        attention_percentage = (avg_attention * 100)
        distraction_count = sum(1 for score in recent_scores if score < 0.3)
        if len(recent_scores) >= 10:
            first_half = np.mean(recent_scores[:len(recent_scores)//2])
            second_half = np.mean(recent_scores[len(recent_scores)//2:])
            if second_half > first_half + 0.1:
                trend = 'improving'
            elif second_half < first_half - 0.1:
                trend = 'declining'
            else:
                trend = 'stable'
        else:
            trend = 'stable'
        return {
            'average_attention': avg_attention,
            'attention_percentage': attention_percentage,
            'distraction_count': distraction_count,
            'engagement_trend': trend
        }   