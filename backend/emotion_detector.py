"""
🎭 Advanced Emotion Detection for AttentionPulse
"""
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
import torch
import torch.nn as nn
logger = logging.getLogger(__name__)

@dataclass
class EmotionState:
    engagement: float = 0.0
    boredom: float = 0.0  
    confusion: float = 0.0
    frustration: float = 0.0
    happiness: float = 0.0
    sadness: float = 0.0
    surprise: float = 0.0
    fear: float = 0.0
    anger: float = 0.0
    neutral: float = 0.0
    attention_level: str = "medium"
    confidence: float = 0.0
    dominant_emotion: str = "neutral"
    emotional_arousal: float = 0.0
    emotional_valence: float = 0.0
    timestamp: float = 0.0
    frame_quality: float = 1.0
class DAISEEEmotionModel(nn.Module):
    def __init__(self, input_size=512, hidden_size=256, num_classes=4):
        super(DAISEEEmotionModel, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.engagement_head = nn.Linear(hidden_size // 2, 4)
        self.boredom_head = nn.Linear(hidden_size // 2, 4)
        self.confusion_head = nn.Linear(hidden_size // 2, 4)
        self.frustration_head = nn.Linear(hidden_size // 2, 4)
    def forward(self, x):
        features = self.feature_extractor(x)
        engagement = torch.softmax(self.engagement_head(features), dim=1)
        boredom = torch.softmax(self.boredom_head(features), dim=1)
        confusion = torch.softmax(self.confusion_head(features), dim=1)
        frustration = torch.softmax(self.frustration_head(features), dim=1)
        return {
            'engagement': engagement,
            'boredom': boredom,
            'confusion': confusion,
            'frustration': frustration
        }
class AdvancedEmotionDetector:
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.daisee_model = None
        self.emotion_model_loaded = False
        self.face_cascade = None
        self.emotion_classifier = None
        self.emotion_history = []
        self.emotion_smoothing_window = 5
        self.load_emotion_models()
        self.setup_facial_analysis()
        logger.info("🎭 Advanced Emotion Detector initialized")
    def load_emotion_models(self):
        try:
            daisee_model_path = self.models_dir / "daisee_emotional_model_best.pth"
            if daisee_model_path.exists():
                self.daisee_model = DAISEEEmotionModel()
                checkpoint = torch.load(daisee_model_path, map_location=self.device)
                if 'model_state_dict' in checkpoint:
                    self.daisee_model.load_state_dict(checkpoint['model_state_dict'])
                elif 'state_dict' in checkpoint:
                    self.daisee_model.load_state_dict(checkpoint['state_dict'])
                else:
                    self.daisee_model.load_state_dict(checkpoint)
                self.daisee_model.to(self.device)
                self.daisee_model.eval()
                self.emotion_model_loaded = True
                logger.info("✅ DAISEE emotion model loaded successfully")
            else:
                logger.warning("⚠️ DAISEE model not found, using fallback emotion detection")
        except Exception as e:
            logger.error(f"❌ Failed to load DAISEE emotion model: {e}")
    def setup_facial_analysis(self):
        try:
            try:
                import mediapipe as mp
                self.mp_face_mesh = mp.solutions.face_mesh
                self.face_mesh = self.mp_face_mesh.FaceMesh(
                    static_image_mode=False,
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )
                self.mp_drawing = mp.solutions.drawing_utils
                self.use_mediapipe = True
                logger.info("✅ MediaPipe facial analysis initialized")
            except ImportError:
                logger.warning("⚠️ MediaPipe not available, using OpenCV fallback")
                self.use_mediapipe = False
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            logger.info("✅ Facial analysis setup complete")
        except Exception as e:
            logger.error(f"❌ Facial analysis setup failed: {e}")
            self.use_mediapipe = False
    def extract_facial_features_for_emotion(self, frame: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        try:
            features = []
            if landmarks is not None and len(landmarks) > 0:
                if len(landmarks) < 400:
                    logger.warning(f"⚠️ Insufficient landmarks: {len(landmarks)}, using fallback features")
                    return np.zeros(512, dtype=np.float32)
                try:
                    left_eye_indices = [33, 133, 160, 158, 144, 153]
                    right_eye_indices = [362, 263, 387, 385, 373, 380]
                    if all(idx < len(landmarks) for idx in left_eye_indices + right_eye_indices):
                        left_eye = landmarks[left_eye_indices]
                        right_eye = landmarks[right_eye_indices]
                    else:
                        left_eye = landmarks[33:39] if len(landmarks) > 39 else landmarks[:6]
                        right_eye = landmarks[362:368] if len(landmarks) > 368 else landmarks[:6]
                    left_eyebrow_indices = [70, 63, 105, 66, 107]
                    right_eyebrow_indices = [296, 334, 293, 300, 276]
                    if all(idx < len(landmarks) for idx in left_eyebrow_indices + right_eyebrow_indices):
                        left_eyebrow = landmarks[left_eyebrow_indices]
                        right_eyebrow = landmarks[right_eyebrow_indices]
                    else:
                        left_eyebrow = landmarks[70:75] if len(landmarks) > 75 else landmarks[:5]
                        right_eyebrow = landmarks[296:301] if len(landmarks) > 301 else landmarks[:5]
                    mouth_outer_indices = [61, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318]
                    mouth_inner_indices = [78, 81, 13, 82, 312, 311, 310, 415]
                    if all(idx < len(landmarks) for idx in mouth_outer_indices + mouth_inner_indices):
                        mouth_outer = landmarks[mouth_outer_indices]
                        mouth_inner = landmarks[mouth_inner_indices]
                    else:
                        mouth_outer = landmarks[61:73] if len(landmarks) > 73 else landmarks[:12]
                        mouth_inner = landmarks[78:86] if len(landmarks) > 86 else landmarks[:8]
                    nose_indices = [1, 2, 5, 4, 6, 19, 94, 168]
                    if all(idx < len(landmarks) for idx in nose_indices):
                        nose = landmarks[nose_indices]
                    else:
                        nose = landmarks[1:9] if len(landmarks) > 9 else landmarks[:8]
                except IndexError as e:
                    logger.warning(f"⚠️ Landmark indexing error: {e}, using fallback")
                    return np.zeros(512, dtype=np.float32)
                try:
                    left_eye_ratio = self._calculate_eye_aspect_ratio(left_eye)
                    right_eye_ratio = self._calculate_eye_aspect_ratio(right_eye)
                    features.extend([left_eye_ratio, right_eye_ratio])
                    left_brow_height = np.mean(left_eyebrow[:, 1]) if len(left_eyebrow) > 0 else 0.0
                    right_brow_height = np.mean(right_eyebrow[:, 1]) if len(right_eyebrow) > 0 else 0.0
                    eye_center_y = (np.mean(left_eye[:, 1]) + np.mean(right_eye[:, 1])) / 2 if len(left_eye) > 0 and len(right_eye) > 0 else 0.0
                    brow_eye_distance = eye_center_y - (left_brow_height + right_brow_height) / 2
                    features.append(brow_eye_distance)
                    if len(mouth_outer) >= 7:
                        mouth_width = np.linalg.norm(mouth_outer[0] - mouth_outer[6])
                        mouth_height = np.linalg.norm(mouth_outer[3] - mouth_outer[9]) if len(mouth_outer) > 9 else np.linalg.norm(mouth_outer[3] - mouth_outer[-1])
                        mouth_ratio = mouth_height / mouth_width if mouth_width > 0 else 0
                    else:
                        mouth_ratio = 0.0
                    features.append(mouth_ratio)
                    if len(mouth_outer) >= 7:
                        left_mouth_corner = mouth_outer[0]
                        right_mouth_corner = mouth_outer[6]
                        mouth_center = mouth_outer[3]
                        left_corner_angle = (mouth_center[1] - left_mouth_corner[1]) / frame.shape[0]
                        right_corner_angle = (mouth_center[1] - right_mouth_corner[1]) / frame.shape[0]
                        smile_indicator = (left_corner_angle + right_corner_angle) / 2
                        smile_indicator = smile_indicator * 2.0
                    else:
                        smile_indicator = 0.0
                    features.append(smile_indicator)
                    face_center_x = frame.shape[1] / 2
                    landmark_center_x = np.mean(landmarks[:, 0])
                    asymmetry = abs(landmark_center_x - face_center_x) / frame.shape[1]
                    features.append(asymmetry)
                    if len(self.emotion_history) > 0:
                        prev_entry = self.emotion_history[-1]
                        prev_landmarks = prev_entry.get('landmarks')
                        if prev_landmarks is not None and len(prev_landmarks) == len(landmarks):
                            movement_energy = np.mean(np.linalg.norm(landmarks - prev_landmarks, axis=1))
                            features.append(movement_energy)
                        else:
                            features.append(0.0)
                    else:
                        features.append(0.0)
                    jaw_indices = [172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 361, 323]
                    valid_jaw_indices = [idx for idx in jaw_indices if idx < len(landmarks)]
                    if len(valid_jaw_indices) > 5:
                        jaw_line = landmarks[valid_jaw_indices]
                        jaw_tension = np.std(jaw_line[:, 1])
                    else:
                        jaw_tension = 0.0
                    features.append(jaw_tension)
                    logger.debug(f"👄 Mouth analysis: width={mouth_width:.3f}, height={mouth_height:.3f}, ratio={mouth_ratio:.3f}")
                    logger.debug(f"😊 Smile indicator: left_angle={left_corner_angle:.3f}, right_angle={right_corner_angle:.3f}, combined={smile_indicator:.3f}")
                except Exception as e:
                    logger.warning(f"⚠️ Feature calculation error: {e}")
                    if len(features) < 8:
                        features.extend([0.0] * (8 - len(features)))
                while len(features) < 512:
                    features.extend(features[:min(512-len(features), len(features))])
                return np.array(features[:512], dtype=np.float32)
            logger.debug("No landmarks provided, returning zero features")
            return np.zeros(512, dtype=np.float32)
        except Exception as e:
            logger.error(f"❌ Feature extraction error: {e}")
            return np.zeros(512, dtype=np.float32)
    def _calculate_eye_aspect_ratio(self, eye_landmarks: np.ndarray) -> float:
        try:
            A = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
            B = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
            C = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
            ear = (A + B) / (2.0 * C) if C > 0 else 0.3
            return ear
        except:
            return 0.3
    def predict_daisee_emotions(self, features: np.ndarray) -> Dict[str, float]:
        if not self.emotion_model_loaded or self.daisee_model is None:
            return self._fallback_emotion_prediction(features)
        try:
            features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
            with torch.no_grad():
                predictions = self.daisee_model(features_tensor)
            engagement_score = torch.sum(predictions['engagement'] * torch.arange(4).float().to(self.device)).item()
            boredom_score = torch.sum(predictions['boredom'] * torch.arange(4).float().to(self.device)).item()
            confusion_score = torch.sum(predictions['confusion'] * torch.arange(4).float().to(self.device)).item()
            frustration_score = torch.sum(predictions['frustration'] * torch.arange(4).float().to(self.device)).item()
            return {
                'engagement': engagement_score / 3.0,  # Normalize to 0-1
                'boredom': boredom_score / 3.0,
                'confusion': confusion_score / 3.0,
                'frustration': frustration_score / 3.0
            }
        except Exception as e:
            logger.warning(f"DAISEE prediction failed: {e}")
            return self._fallback_emotion_prediction(features)
    def _fallback_emotion_prediction(self, features: np.ndarray) -> Dict[str, float]:
        try:
            left_eye_ratio = features[0] if len(features) > 0 else 0.35
            right_eye_ratio = features[1] if len(features) > 1 else 0.35
            brow_eye_distance = features[2] if len(features) > 2 else 0.0
            mouth_ratio = features[3] if len(features) > 3 else 0.0
            smile_indicator = features[4] if len(features) > 4 else 0.0
            asymmetry = features[5] if len(features) > 5 else 0.0
            movement_energy = features[6] if len(features) > 6 else 0.0
            avg_eye_ratio = (left_eye_ratio + right_eye_ratio) / 2
            eye_symmetry = 1.0 - abs(left_eye_ratio - right_eye_ratio)
            engagement_base = avg_eye_ratio * 1.2
            if movement_energy > 0.1:
                engagement_base += movement_energy * 0.8
            if eye_symmetry > 0.8:
                engagement_base += 0.2
            if smile_indicator > 0:
                engagement_base += smile_indicator * 0.5
            engagement = np.clip(engagement_base, 0.0, 1.0)
            boredom_base = 0.0
            if avg_eye_ratio < 0.3:
                boredom_base += 0.4
            if movement_energy < 0.05:
                boredom_base += 0.3
            if smile_indicator < -0.1:
                boredom_base += 0.2
            if brow_eye_distance < -0.1:
                boredom_base += 0.1
            boredom = np.clip(max(boredom_base, 1.0 - engagement * 0.8), 0.0, 1.0)
            confusion_base = 0.0
            if asymmetry > 0.1:
                confusion_base += asymmetry * 2.0
            if abs(left_eye_ratio - right_eye_ratio) > 0.1:
                confusion_base += 0.3
            if brow_eye_distance > 0.1:
                confusion_base += 0.3
            if 0.05 < movement_energy < 0.3:
                confusion_base += movement_energy * 0.5
            confusion = np.clip(confusion_base, 0.0, 1.0)
            frustration_base = 0.0
            if movement_energy > 0.3:
                frustration_base += movement_energy * 0.6
            if smile_indicator < -0.15:
                frustration_base += abs(smile_indicator) * 0.8
            if brow_eye_distance > 0.15:
                frustration_base += 0.4
            if asymmetry > 0.15:
                frustration_base += 0.3
            frustration = np.clip(frustration_base, 0.0, 1.0)
            total = engagement + boredom + confusion + frustration
            if total > 1.5:
                factor = 1.2 / total
                engagement *= factor
                boredom *= factor
                confusion *= factor
                frustration *= factor
            if engagement > 0.7:
                boredom = max(0.0, boredom - 0.3)
            if boredom > 0.7:
                engagement = max(0.0, engagement - 0.3)
            return {
                'engagement': np.clip(engagement, 0.0, 1.0),
                'boredom': np.clip(boredom, 0.0, 1.0),
                'confusion': np.clip(confusion, 0.0, 1.0),
                'frustration': np.clip(frustration, 0.0, 1.0)
            }
        except Exception as e:
            logger.warning(f"Fallback emotion prediction failed: {e}")
            return {
                'engagement': 0.5,
                'boredom': 0.3,
                'confusion': 0.2,
                'frustration': 0.1
            }
    def analyze_basic_emotions(self, features: np.ndarray) -> Dict[str, float]:
        try:
            smile_indicator = features[4] if len(features) > 4 else 0.0
            left_eye_ratio = features[0] if len(features) > 0 else 0.35
            right_eye_ratio = features[1] if len(features) > 1 else 0.35
            brow_eye_distance = features[2] if len(features) > 2 else 0.0
            mouth_ratio = features[3] if len(features) > 3 else 0.0
            asymmetry = features[5] if len(features) > 5 else 0.0
            movement_energy = features[6] if len(features) > 6 else 0.0
            avg_eye_ratio = (left_eye_ratio + right_eye_ratio) / 2
            happiness_score = 0.0
            if smile_indicator > 0:
                happiness_score += smile_indicator * 4.0
            if avg_eye_ratio > 0.25 and avg_eye_ratio < 0.45:
                happiness_score += 0.3
            if 0.05 < asymmetry < 0.15 and smile_indicator > 0:
                happiness_score += 0.2
            if movement_energy > 0.1 and smile_indicator > -0.1:
                happiness_score += movement_energy * 0.3
            if smile_indicator > 0.02:
                happiness_score += 0.15
            happiness = np.clip(happiness_score, 0.0, 1.0)
            sadness_score = 0.0
            if smile_indicator < -0.05:
                sadness_score += abs(smile_indicator) * 2.0
            if avg_eye_ratio < 0.25:
                sadness_score += 0.3
            if brow_eye_distance < -0.1:
                sadness_score += 0.2
            sadness = np.clip(sadness_score, 0.0, 1.0)
            surprise_score = 0.0
            if avg_eye_ratio > 0.45:
                surprise_score += (avg_eye_ratio - 0.45) * 3.0
            if brow_eye_distance > 0.15:
                surprise_score += 0.4
            if mouth_ratio > 0.5:
                surprise_score += 0.3
            surprise = np.clip(surprise_score, 0.0, 1.0)
            fear_score = 0.0
            if movement_energy > 0.3:
                fear_score += movement_energy * 0.5
            if asymmetry > 0.2:
                fear_score += 0.3
            if avg_eye_ratio > 0.4 and brow_eye_distance > 0.1:
                fear_score += 0.3
            fear = np.clip(fear_score, 0.0, 1.0)
            anger_score = 0.0
            if movement_energy > 0.2 and smile_indicator < -0.1:
                anger_score += movement_energy * 0.4
            if brow_eye_distance > 0.1 and smile_indicator < 0:
                anger_score += 0.4
            if asymmetry < 0.1 and movement_energy > 0.25:
                anger_score += 0.3
            anger = np.clip(anger_score, 0.0, 1.0)
            other_emotions_total = happiness + sadness + surprise + fear + anger
            neutral = max(0.0, 1.0 - other_emotions_total * 0.8)
            total = happiness + sadness + surprise + fear + anger + neutral
            if total > 1.5:
                factor = 1.2 / total
                happiness *= factor
                sadness *= factor
                surprise *= factor
                fear *= factor
                anger *= factor
                neutral *= factor
            emotions = {
                'happiness': happiness,
                'sadness': sadness,
                'surprise': surprise,
                'fear': fear,
                'anger': anger,
                'neutral': neutral
            }
            
            total = sum(emotions.values())
            if total > 1.5:  # Only normalize if the total is significantly above 1.0
                emotions = {k: v/total for k, v in emotions.items()}
                
            logger.info(f"🎭 Emotion analysis: smile_indicator={smile_indicator:.3f}, happiness={emotions['happiness']:.3f}")
            return emotions
        except Exception as e:
            logger.warning(f"Basic emotion analysis failed: {e}")
            return {
                'happiness': 0.3,  # Default higher happiness
                'sadness': 0.1,
                'surprise': 0.1,
                'fear': 0.1,
                'anger': 0.1,
                'neutral': 0.3
            }
    def detect_emotions(self, frame: np.ndarray, landmarks: np.ndarray = None) -> EmotionState:
        """🎭 Main emotion detection function
        
        Args:
            frame: Input image frame
            landmarks: Optional facial landmarks
            
        Returns:
            EmotionState: Comprehensive emotional state analysis
        """
        start_time = time.time()
        try:
            features = self.extract_facial_features_for_emotion(frame, landmarks)
            daisee_emotions = self.predict_daisee_emotions(features)
            basic_emotions = self.analyze_basic_emotions(features)
            dominant_emotion = max(basic_emotions.items(), key=lambda x: x[1])[0]
            arousal = daisee_emotions['engagement'] + basic_emotions['surprise'] + basic_emotions['anger']
            valence = basic_emotions['happiness'] - basic_emotions['sadness'] - daisee_emotions['frustration']
            attention_level = self._determine_attention_level(daisee_emotions, basic_emotions)
            confidence = 1.0 - daisee_emotions['confusion']
            emotion_state = EmotionState(
                engagement=daisee_emotions['engagement'],
                boredom=daisee_emotions['boredom'],
                confusion=daisee_emotions['confusion'],
                frustration=daisee_emotions['frustration'],
                happiness=basic_emotions['happiness'],
                sadness=basic_emotions['sadness'],
                surprise=basic_emotions['surprise'],
                fear=basic_emotions['fear'],
                anger=basic_emotions['anger'],
                neutral=basic_emotions['neutral'],
                attention_level=attention_level,
                confidence=confidence,
                dominant_emotion=dominant_emotion,
                emotional_arousal=np.clip(arousal, 0, 1),
                emotional_valence=np.clip(valence, -1, 1),
                timestamp=time.time(),
                frame_quality=1.0
            )
            
            self._update_emotion_history(emotion_state, landmarks)
            smoothed_state = self._apply_temporal_smoothing()
            processing_time = time.time() - start_time
            logger.debug(f"🎭 Emotion detection completed in {processing_time*1000:.1f}ms")
            logger.info(f"🎭 Real-time emotions: happiness={smoothed_state.happiness:.3f}, engagement={smoothed_state.engagement:.3f}, boredom={smoothed_state.boredom:.3f}")
            return smoothed_state
            
        except Exception as e:
            logger.error(f"❌ Emotion detection failed: {e}", exc_info=True)
            return EmotionState(
                engagement=0.5,
                boredom=0.3,
                confusion=0.2,
                frustration=0.1,
                happiness=0.4,
                sadness=0.1,
                surprise=0.1,
                fear=0.1,
                anger=0.1,
                neutral=0.3,
                attention_level="medium",
                confidence=0.7,
                dominant_emotion="neutral",
                emotional_arousal=0.5,
                emotional_valence=0.0,
                timestamp=time.time(),
                frame_quality=0.5
            )
    def _determine_attention_level(self, daisee_emotions: Dict, basic_emotions: Dict) -> str:
        engagement = daisee_emotions['engagement']
        boredom = daisee_emotions['boredom']
        confusion = daisee_emotions['confusion']
        if engagement > 0.7 and boredom < 0.3:
            return "very_high"
        elif engagement > 0.5 and boredom < 0.5:
            return "high"
        elif confusion > 0.6 or boredom > 0.6:
            return "low"
        else:
            return "medium"
    def _update_emotion_history(self, emotion_state: EmotionState, landmarks: np.ndarray):
        history_entry = {
            'emotion_state': emotion_state,
            'landmarks': landmarks,
            'timestamp': time.time()
        }
        self.emotion_history.append(history_entry)
        if len(self.emotion_history) > self.emotion_smoothing_window * 2:
            self.emotion_history = self.emotion_history[-self.emotion_smoothing_window:]
    def _apply_temporal_smoothing(self) -> EmotionState:
        if len(self.emotion_history) == 0:
            return EmotionState()
        if len(self.emotion_history) == 1:
            return self.emotion_history[-1]['emotion_state']
        smoothing_window = min(3, len(self.emotion_history))
        recent_states = [entry['emotion_state'] for entry in self.emotion_history[-smoothing_window:]]
        if len(recent_states) == 2:
            weights = np.array([0.3, 0.7])
        elif len(recent_states) == 3:
            weights = np.array([0.2, 0.3, 0.5])
        else:
            weights = np.linspace(0.1, 0.6, len(recent_states))
            weights = weights / np.sum(weights)
        smoothed = EmotionState()
        current_state = recent_states[-1]
        dynamic_alpha = 0.7
        static_alpha = 0.5
        smoothed.engagement = dynamic_alpha * current_state.engagement + (1 - dynamic_alpha) * np.average([s.engagement for s in recent_states[:-1]], weights=weights[:-1]) if len(recent_states) > 1 else current_state.engagement
        smoothed.boredom = static_alpha * current_state.boredom + (1 - static_alpha) * np.average([s.boredom for s in recent_states[:-1]], weights=weights[:-1]) if len(recent_states) > 1 else current_state.boredom
        smoothed.confusion = static_alpha * current_state.confusion + (1 - static_alpha) * np.average([s.confusion for s in recent_states[:-1]], weights=weights[:-1]) if len(recent_states) > 1 else current_state.confusion
        smoothed.frustration = static_alpha * current_state.frustration + (1 - static_alpha) * np.average([s.frustration for s in recent_states[:-1]], weights=weights[:-1]) if len(recent_states) > 1 else current_state.frustration
        smoothed.happiness = 0.8 * current_state.happiness + 0.2 * np.average([s.happiness for s in recent_states[:-1]], weights=weights[:-1]) if len(recent_states) > 1 else current_state.happiness
        smoothed.sadness = static_alpha * current_state.sadness + (1 - static_alpha) * np.average([s.sadness for s in recent_states[:-1]], weights=weights[:-1]) if len(recent_states) > 1 else current_state.sadness
        smoothed.surprise = dynamic_alpha * current_state.surprise + (1 - dynamic_alpha) * np.average([s.surprise for s in recent_states[:-1]], weights=weights[:-1]) if len(recent_states) > 1 else current_state.surprise
        smoothed.fear = static_alpha * current_state.fear + (1 - static_alpha) * np.average([s.fear for s in recent_states[:-1]], weights=weights[:-1]) if len(recent_states) > 1 else current_state.fear
        smoothed.anger = static_alpha * current_state.anger + (1 - static_alpha) * np.average([s.anger for s in recent_states[:-1]], weights=weights[:-1]) if len(recent_states) > 1 else current_state.anger
        smoothed.neutral = static_alpha * current_state.neutral + (1 - static_alpha) * np.average([s.neutral for s in recent_states[:-1]], weights=weights[:-1]) if len(recent_states) > 1 else current_state.neutral
        smoothed.emotional_arousal = static_alpha * current_state.emotional_arousal + (1 - static_alpha) * np.average([s.emotional_arousal for s in recent_states[:-1]], weights=weights[:-1]) if len(recent_states) > 1 else current_state.emotional_arousal
        smoothed.emotional_valence = static_alpha * current_state.emotional_valence + (1 - static_alpha) * np.average([s.emotional_valence for s in recent_states[:-1]], weights=weights[:-1]) if len(recent_states) > 1 else current_state.emotional_valence
        smoothed.confidence = static_alpha * current_state.confidence + (1 - static_alpha) * np.average([s.confidence for s in recent_states[:-1]], weights=weights[:-1]) if len(recent_states) > 1 else current_state.confidence
        smoothed.attention_level = current_state.attention_level
        smoothed.dominant_emotion = current_state.dominant_emotion
        smoothed.timestamp = current_state.timestamp
        smoothed.frame_quality = current_state.frame_quality
        return smoothed
    def get_emotion_insights(self) -> Dict:
        if len(self.emotion_history) < 5:
            return {"status": "insufficient_data"}
        recent_states = [entry['emotion_state'] for entry in self.emotion_history[-10:]]
        engagement_trend = np.polyfit(range(len(recent_states)), [s.engagement for s in recent_states], 1)[0]
        boredom_trend = np.polyfit(range(len(recent_states)), [s.boredom for s in recent_states], 1)[0]
        dominant_emotions = [s.dominant_emotion for s in recent_states]
        emotion_stability = len(set(dominant_emotions)) / len(dominant_emotions)
        return {
            "engagement_trend": "increasing" if engagement_trend > 0.01 else "decreasing" if engagement_trend < -0.01 else "stable",
            "boredom_trend": "increasing" if boredom_trend > 0.01 else "decreasing" if boredom_trend < -0.01 else "stable",
            "emotion_stability": emotion_stability,
            "average_engagement": np.mean([s.engagement for s in recent_states]),
            "average_arousal": np.mean([s.emotional_arousal for s in recent_states]),
            "average_valence": np.mean([s.emotional_valence for s in recent_states])
        }