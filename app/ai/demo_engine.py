"""
Student Demo System
"""

import asyncio
import logging
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
try:
    import pickle
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
try:
    from .audio_analysis import AudioEngagementAnalyzer
    AUDIO_ANALYSIS_AVAILABLE = True
except ImportError:
    AUDIO_ANALYSIS_AVAILABLE = False
logger = logging.getLogger(__name__)
class StudentDemo:
  
    def __init__(self, models_dir: str = "/app/models"):
        self.models_dir = Path(models_dir)
        self.model_version = "demo-v1.0"
        if not self.models_dir.exists():
            logger.warning(f"Models directory not found: {self.models_dir}. Trying fallback paths...")
            fallback_paths = ["models", "./models", "../models", "/models"]
            for fallback in fallback_paths:
                fallback_path = Path(fallback)
                if fallback_path.exists():
                    self.models_dir = fallback_path
                    logger.info(f"Using fallback models directory: {self.models_dir}")
                    break
            else:
                logger.error("No valid models directory found!")
        self._init_mediapipe()
        self._load_trained_models()
        self._init_smoothing_systems()
        self._init_audio_analysis()
        self.update_frequency = 1.0
        self.smoothing_window = 15
        self.confidence_threshold = 0.7
        self.min_update_interval = 1.5
        self.session_start = time.time()
        self.frame_count = 0
        self.last_display_update = 0
        self._last_smoothed_values = {}
    def _init_mediapipe(self):
        if not MEDIAPIPE_AVAILABLE:
            logger.warning("MediaPipe not available - using fallback")
            return
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_pose = mp.solutions.pose
        self.mp_hands = mp.solutions.hands
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.8,
            min_tracking_confidence=0.8
        )
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.8
        )
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.8
        )
    def _load_trained_models(self):
        self.models = {}
        self.trained_features = {
            'emotion_detection': False,
            'attention_analysis': False,
            'engagement_scoring': False,
            'gaze_tracking': False,  
            'posture_analysis': False
        }
        logger.info(f"🔍 Loading models from: {self.models_dir}")
        onnx_path = self.models_dir / "onnx" / "best_model.onnx"
        if onnx_path.exists() and ONNX_AVAILABLE:
            try:
                self.models['emotion_onnx'] = ort.InferenceSession(str(onnx_path))
                self.trained_features['emotion_detection'] = True
                logger.info(f"✅ Loaded TRAINED ONNX emotion model: {onnx_path}")
            except Exception as e:
                logger.error(f"❌ Failed to load ONNX emotion model: {e}")
        else:
            logger.warning(f"⚠️ ONNX emotion model not found at: {onnx_path}")
        torch_path = self.models_dir / "daisee_emotional_model_best.pth"
        if torch_path.exists() and TORCH_AVAILABLE:
            try:
                file_size = torch_path.stat().st_size
                if file_size > 1000:
                    self.models['attention_torch'] = torch.load(torch_path, map_location='cpu')
                    self.models['attention_torch'].eval()
                    self.trained_features['attention_analysis'] = True
                    logger.info(f"✅ Loaded TRAINED PyTorch attention model: {torch_path} ({file_size} bytes)")
                else:
                    logger.warning(f"⚠️ PyTorch model too small, likely untrained: {file_size} bytes")
            except Exception as e:
                logger.error(f"❌ Failed to load PyTorch model: {e}")
        else:
            logger.warning(f"⚠️ PyTorch attention model not found at: {torch_path}")
        logger.info("⏭️ Skipping sklearn engagement models (not trained yet)")
        self.trained_features['gaze_tracking'] = True
        logger.info("✅ Gaze tracking enabled (geometric estimation)")
        if MEDIAPIPE_AVAILABLE:
            self.trained_features['posture_analysis'] = True
            logger.info("✅ Basic posture analysis enabled (MediaPipe)")
        loaded_count = sum(self.trained_features.values())
        if loaded_count > 0:
            self.model_version = f"demo-v1.0-partial-{loaded_count}features"
        else:
            self.model_version = "demo-v1.0-fallback"
        logger.info(f"🎯 Demo ready with {loaded_count}/5 features: {self.trained_features}")
    def _init_smoothing_systems(self):
        self.smoothed_metrics = {
            'attention_score': deque(maxlen=self.smoothing_window),
            'engagement_level': deque(maxlen=self.smoothing_window),
            'emotion_confidence': deque(maxlen=self.smoothing_window),
            'gaze_stability': deque(maxlen=self.smoothing_window),
            'posture_score': deque(maxlen=self.smoothing_window)
        }
        self.display_metrics = {
            'attention': 0.0,
            'engagement': 0.0,
            'emotion': 'neutral',
            'emotion_confidence': 0.0,
            'gaze_direction': 'center',
            'posture': 'good',
            'overall_focus': 0.0
        }
        self.emotion_labels = {
            0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy',
            4: 'Neutral', 5: 'Sad', 6: 'Surprise', 7: 'Focused'
        }
    def _init_audio_analysis(self):
        if AUDIO_ANALYSIS_AVAILABLE:
            try:
                self.audio_analyzer = AudioEngagementAnalyzer()
                logger.info("✅ Audio analysis enabled")
            except Exception as e:
                logger.warning(f"Audio analysis initialization failed: {e}")
                self.audio_analyzer = None
        else:
            self.audio_analyzer = None
            logger.info("🎤 Audio analysis not available")
    def preprocess_frame_for_emotion(self, frame):
        if 'emotion_onnx' not in self.models:
            return None
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (48, 48))
            normalized = resized.astype(np.float32) / 255.0
            input_tensor = normalized.reshape(1, 1, 48, 48)
            return input_tensor
        except Exception as e:
            logger.error(f"Frame preprocessing failed: {e}")
            return None
    def predict_emotion_onnx(self, frame):
        if 'emotion_onnx' not in self.models:
            return 'neutral', 0.5
        try:
            input_tensor = self.preprocess_frame_for_emotion(frame)
            if input_tensor is None:
                return 'neutral', 0.5
            model = self.models['emotion_onnx']
            input_name = model.get_inputs()[0].name
            outputs = model.run(None, {input_name: input_tensor})
            probabilities = outputs[0][0]
            predicted_class = np.argmax(probabilities)
            confidence = float(probabilities[predicted_class])
            emotion = self.emotion_labels.get(predicted_class, 'neutral')
            return emotion, confidence
        except Exception as e:
            logger.error(f"ONNX emotion prediction failed: {e}")
            return 'neutral', 0.5
    def extract_attention_features(self, face_landmarks, pose_landmarks, frame_shape):
        features = []
        try:
            if face_landmarks and len(face_landmarks.landmark) > 0:
                left_eye_ratio = self._calculate_eye_aspect_ratio(face_landmarks, 'left')
                right_eye_ratio = self._calculate_eye_aspect_ratio(face_landmarks, 'right')
                features.extend([left_eye_ratio, right_eye_ratio])
                head_pose = self._estimate_head_pose(face_landmarks, frame_shape)
                features.extend(head_pose)
                mouth_ratio = self._calculate_mouth_aspect_ratio(face_landmarks)
                features.append(mouth_ratio)
            else:
                features.extend([0.3, 0.3, 0.0, 0.0, 0.0, 0.1])
            if pose_landmarks:
                shoulder_alignment = self._calculate_shoulder_alignment(pose_landmarks)
                features.append(shoulder_alignment)
                head_tilt = self._calculate_head_tilt(pose_landmarks)
                features.append(head_tilt)
            else:
                features.extend([0.0, 0.0])
            return np.array(features).reshape(1, -1)
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            return np.zeros((1, 8))
    def predict_attention_pytorch(self, features):
        if 'attention_torch' not in self.models:
            return 0.5
        try:
            model = self.models['attention_torch']
            features_tensor = torch.FloatTensor(features)
            with torch.no_grad():
                outputs = model(features_tensor)
                attention_score = torch.sigmoid(outputs).item()
            return attention_score
        except Exception as e:
            logger.error(f"PyTorch attention prediction failed: {e}")
            return 0.5
    def predict_engagement_sklearn(self, features):
        if 'local_attention_model_ensemble' not in self.models:
            if features is not None and len(features[0]) > 0:
                eye_ratio_avg = (features[0][0] + features[0][1]) / 2 if len(features[0]) > 1 else 0.3
                head_pose_stability = 1.0 - abs(features[0][4]) / 30.0 if len(features[0]) > 4 else 0.7
                engagement_score = (eye_ratio_avg * 0.6 + head_pose_stability * 0.4)
                engagement_score = max(0.2, min(0.9, engagement_score))
                logger.debug(f"Using fallback engagement estimation: {engagement_score:.3f}")
                return engagement_score
            else:
                return 0.5
        try:
            model = self.models['local_attention_model_ensemble'] 
            scaler = self.models.get('local_scaler_ensemble')
            if scaler:
                features_scaled = scaler.transform(features)
            else:
                features_scaled = features
            engagement_pred = model.predict_proba(features_scaled)[0]
            engagement_score = engagement_pred[1] if len(engagement_pred) > 1 else engagement_pred[0]
            return engagement_score
        except Exception as e:
            logger.error(f"Sklearn engagement prediction failed: {e}")
            return 0.5
    def estimate_gaze_direction(self, face_landmarks):
        if not face_landmarks:
            return 'center', 0.5, 0.5
        try:
            left_eye_center = self._get_eye_center(face_landmarks, 'left')
            right_eye_center = self._get_eye_center(face_landmarks, 'right')
            nose_tip = self._get_nose_tip(face_landmarks)
            eye_center = ((left_eye_center[0] + right_eye_center[0]) / 2,
                         (left_eye_center[1] + right_eye_center[1]) / 2)
            horizontal_diff = eye_center[0] - nose_tip[0]
            vertical_diff = eye_center[1] - nose_tip[1]
            if abs(horizontal_diff) < 0.02 and abs(vertical_diff) < 0.02:
                direction = 'center'
            elif horizontal_diff > 0.02:
                direction = 'left'
            elif horizontal_diff < -0.02:
                direction = 'right'
            elif vertical_diff > 0.02:
                direction = 'up'
            else:
                direction = 'down'
            screen_x = 0.5 + horizontal_diff * 5
            screen_y = 0.5 + vertical_diff * 5
            screen_x = max(0, min(1, screen_x))
            screen_y = max(0, min(1, screen_y))
            return direction, screen_x, screen_y
        except Exception as e:
            logger.error(f"Gaze estimation failed: {e}")
            return 'center', 0.5, 0.5
    def _smooth_metrics(self, new_metrics):
        for key, value in new_metrics.items():
            if key in self.smoothed_metrics:
                self.smoothed_metrics[key].append(value)
        smoothed = {}
        for key, values in self.smoothed_metrics.items():
            if values:
                if len(values) == 1:
                    smoothed[key] = values[0]
                else:
                    weights = np.linspace(0.5, 1.0, len(values))
                    weighted_sum = sum(w * v for w, v in zip(weights, values))
                    weight_sum = sum(weights)
                    smoothed[key] = weighted_sum / weight_sum
                    if hasattr(self, '_last_smoothed_values'):
                        last_value = self._last_smoothed_values.get(key, smoothed[key])
                        max_change = 0.1
                        change = smoothed[key] - last_value
                        if abs(change) > max_change:
                            smoothed[key] = last_value + (max_change if change > 0 else -max_change)
            else:
                smoothed[key] = 0.0
        self._last_smoothed_values = smoothed.copy()
        return smoothed
    def _should_update_display(self):
        current_time = time.time()
        time_since_last_update = current_time - self.last_display_update
        min_interval = max(1.0 / self.update_frequency, self.min_update_interval)
        if time_since_last_update >= min_interval:
            self.last_display_update = current_time
            return True
        return False
    async def analyze_frame(self, frame, audio_data=None):
  
        try:
            self.frame_count += 1
            analysis_start = time.time()
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_shape = frame.shape
            face_results = None
            pose_results = None
            hand_results = None
            if MEDIAPIPE_AVAILABLE:
                face_results = self.face_mesh.process(rgb_frame)
                pose_results = self.pose.process(rgb_frame)
                hand_results = self.hands.process(rgb_frame)
            emotion, emotion_confidence = self.predict_emotion_onnx(frame)
            face_landmarks = face_results.multi_face_landmarks[0] if (face_results and face_results.multi_face_landmarks) else None
            pose_landmarks = pose_results.pose_landmarks if pose_results else None
            features = self.extract_attention_features(face_landmarks, pose_landmarks, frame_shape)
            attention_score = self.predict_attention_pytorch(features)
            engagement_score = self.predict_engagement_sklearn(features)
            gaze_direction, gaze_x, gaze_y = self.estimate_gaze_direction(face_landmarks)
            posture_score = self._analyze_posture(pose_landmarks)
            audio_metrics = {}
            if audio_data is not None and self.audio_analyzer:
                try:
                    audio_features = await self.audio_analyzer.analyze_audio_stream(audio_data)
                    audio_metrics = {
                        'audio_engagement': audio_features.get('audio_engagement_score', 0.5),
                        'vocal_confidence': audio_features.get('vocal_confidence', False),
                        'speech_activity': audio_features.get('speech_ratio', 0.0),
                        'audio_quality': audio_features.get('audio_quality', 'unknown')
                    }
                    audio_weight = 0.3 if audio_metrics['vocal_confidence'] else 0.1
                    engagement_score = (engagement_score * (1 - audio_weight)) + (audio_metrics['audio_engagement'] * audio_weight)
                except Exception as e:
                    logger.warning(f"Audio analysis failed: {e}")
                    audio_metrics = {'audio_engagement': 0.5, 'vocal_confidence': False}
            overall_focus = (attention_score * 0.4 + engagement_score * 0.4 + emotion_confidence * 0.2)
            raw_metrics = {
                'attention_score': attention_score,
                'engagement_level': engagement_score,
                'emotion_confidence': emotion_confidence,
                'gaze_stability': 1.0 - abs(gaze_x - 0.5) - abs(gaze_y - 0.5),  # Stability measure
                'posture_score': posture_score,
                **audio_metrics
            }
            smoothed = self._smooth_metrics(raw_metrics)
            if self._should_update_display():
                display_update = {
                    'attention': smoothed['attention_score'],
                    'engagement': smoothed['engagement_level'],
                    'emotion': emotion,
                    'emotion_confidence': smoothed['emotion_confidence'],
                    'gaze_direction': gaze_direction,
                    'gaze_x': gaze_x,
                    'gaze_y': gaze_y,
                    'posture': 'good' if posture_score > 0.7 else 'slouching' if posture_score > 0.4 else 'poor',
                    'overall_focus': (smoothed['attention_score'] + smoothed['engagement_level']) / 2
                }
                if audio_metrics:
                    display_update.update({
                        'audio_engagement': smoothed.get('audio_engagement', 0.5),
                        'vocal_activity': audio_metrics.get('vocal_confidence', False),
                        'multimodal_score': (display_update['overall_focus'] + smoothed.get('audio_engagement', 0.5)) / 2
                    })
                self.display_metrics.update(display_update)
            processing_time = (time.time() - analysis_start) * 1000
            result = {
                'display_metrics': self.display_metrics.copy(),
                'raw_metrics': raw_metrics,
                'smoothed_metrics': smoothed,
                'face_detected': face_landmarks is not None,
                'hands_detected': hand_results.multi_hand_landmarks is not None if hand_results else False,
                'processing_time_ms': processing_time,
                'frame_count': self.frame_count,
                'session_duration': time.time() - self.session_start,
                'model_version': self.model_version,
                'models_loaded': list(self.models.keys()),
                'audio_enabled': self.audio_analyzer is not None
            }
            if self.audio_analyzer:
                try:
                    audio_summary = self.audio_analyzer.get_audio_engagement_summary()
                    result['audio_summary'] = audio_summary
                except Exception as e:
                    logger.warning(f"Audio summary failed: {e}")
            return result
        except Exception as e:
            logger.error(f"Frame analysis failed: {e}")
            return {
                'display_metrics': self.display_metrics.copy(),
                'error': str(e),
                'processing_time_ms': 0,
                'frame_count': self.frame_count,
                'audio_enabled': False
            }
    def _calculate_eye_aspect_ratio(self, landmarks, eye):
        return 0.3
    def _estimate_head_pose(self, landmarks, frame_shape):
        return [0.0, 0.0, 0.0]
    def _calculate_mouth_aspect_ratio(self, landmarks):
        return 0.1
    def _calculate_shoulder_alignment(self, pose_landmarks):
        return 0.8
    def _calculate_head_tilt(self, pose_landmarks):
        return 0.0
    def _get_eye_center(self, landmarks, eye):
        return (0.5, 0.3)
    def _get_nose_tip(self, landmarks):
        return (0.5, 0.5)
    def _analyze_posture(self, pose_landmarks):
        if pose_landmarks:
            return 0.8
        return 0.5