"""
AttentionPulse - Production AI Detector
"""
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)
class SimpleAttentionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((7, 7))
        )
        self.feature_fusion = nn.Sequential(
            nn.Linear(128 * 7 * 7 + 7, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.attention_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        self.engagement_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 4)
        )
        self.emotion_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 7)
        )
    def forward(self, image, features):
        visual_features = self.backbone(image)
        visual_features = visual_features.view(visual_features.size(0), -1)
        combined_features = torch.cat([visual_features, features], dim=1)
        fused_features = self.feature_fusion(combined_features)
        attention_score = self.attention_head(fused_features)
        engagement_logits = self.engagement_head(fused_features)
        emotion_logits = self.emotion_head(fused_features)
        return {
            'attention_score': attention_score.squeeze(),
            'engagement_logits': engagement_logits,
            'emotion_logits': emotion_logits
        }
class AIAttentionDetectorProduction:
    def __init__(self, model_path: Optional[str] = None):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = None
        self.model_loaded = False
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.load_model(model_path)
        logger.info(f"AI Detector initialized with device: {self.device}")
        logger.info(f"Model loaded: {self.model_loaded}")
    def load_model(self, model_path: Optional[str] = None):
        try:
            if model_path is None:
                possible_paths = [
                    'models/best_attention_model.pth',
                    'models/minimal_attention_model.pth',
                    'models_trained/attention_model.h5',
                    'production_models/attention_best.h5'
                ]
                for path in possible_paths:
                    if os.path.exists(path):
                        model_path = path
                        break
            if model_path and os.path.exists(model_path):
                if model_path.endswith('.pth'):
                    self.model = SimpleAttentionModel()
                    self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                    self.model.eval()
                    self.model_loaded = True
                    logger.info(f"Loaded PyTorch model from: {model_path}")
                else:
                    logger.warning(f"Model format not supported: {model_path}")
            else:
                logger.warning("No trained model found, using rule-based fallback")
                self.model_loaded = False
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.model_loaded = False
    def extract_handcrafted_features(self, image: np.ndarray, faces: list) -> np.ndarray:
        height, width = image.shape[:2]
        if len(faces) == 0:
            return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        face = max(faces, key=lambda f: f[2] * f[3])
        x, y, w, h = face
        face_confidence = min(1.0, (w * h) / (width * height * 0.1))
        face_roi = image[y:y+h, x:x+w]
        if face_roi.size > 0:
            eye_region = face_roi[int(h*0.2):int(h*0.6), int(w*0.2):int(w*0.8)]
            if eye_region.size > 0:
                eye_openness = min(1.0, np.std(eye_region) / 50.0)
            else:
                eye_openness = 0.5
        else:
            eye_openness = 0.5
        center_x = x + w // 2
        center_y = y + h // 2
        head_pose_yaw = (center_x - width // 2) / (width // 2)
        head_pose_pitch = (center_y - height // 2) / (height // 2)
        gaze_focus = max(0.0, 1.0 - abs(head_pose_yaw) - abs(head_pose_pitch))
        motion_magnitude = 0.3
        blink_frequency = 0.3
        return np.array([
            face_confidence,
            eye_openness,
            head_pose_yaw,
            head_pose_pitch,
            gaze_focus,
            motion_magnitude,
            blink_frequency
        ], dtype=np.float32)
    def analyze_frame(self, image: np.ndarray) -> Dict:
        start_time = time.time()
        try:
            if len(image.shape) == 3 and image.shape[2] == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image
            gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            handcrafted_features = self.extract_handcrafted_features(image_rgb, faces)
            if self.model_loaded and self.model is not None:
                attention_score, engagement_level, emotion_class = self._predict_with_model(
                    image_rgb, handcrafted_features
                )
            else:
                attention_score, engagement_level, emotion_class = self._rule_based_prediction(
                    image_rgb, faces, handcrafted_features
                )
            processing_time = time.time() - start_time
            return {
                'attention_score': float(attention_score),
                'engagement_level': int(engagement_level),
                'emotion_class': int(emotion_class),
                'face_detected': len(faces) > 0,
                'face_count': len(faces),
                'processing_time_ms': processing_time * 1000,
                'model_used': 'trained' if self.model_loaded else 'rule_based',
                'features': handcrafted_features.tolist()
            }
        except Exception as e:
            logger.error(f"Error in analyze_frame: {e}")
            return {
                'attention_score': 0.5,
                'engagement_level': 1,
                'emotion_class': 2,  
                'face_detected': False,
                'face_count': 0,
                'processing_time_ms': 0,
                'model_used': 'error',
                'error': str(e)
            }
    def _predict_with_model(self, image: np.ndarray, features: np.ndarray) -> Tuple[float, int, int]:
        try:
            image_resized = cv2.resize(image, (224, 224))
            image_tensor = torch.from_numpy(image_resized).float().permute(2, 0, 1) / 255.0
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            image_tensor = (image_tensor - mean) / std
            image_tensor = image_tensor.unsqueeze(0).to(self.device)
            features_tensor = torch.from_numpy(features).unsqueeze(0).to(self.device)
            with torch.no_grad():
                outputs = self.model(image_tensor, features_tensor)
                attention_score = outputs['attention_score'].item()
                engagement_level = outputs['engagement_logits'].argmax().item()
                emotion_class = outputs['emotion_logits'].argmax().item()
                return attention_score, engagement_level, emotion_class
        except Exception as e:
            logger.error(f"Model prediction error: {e}")
            return self._rule_based_prediction(image, [], features)
    def _rule_based_prediction(self, image: np.ndarray, faces: list, features: np.ndarray) -> Tuple[float, int, int]:
        face_confidence = features[0]
        eye_openness = features[1]
        head_pose_yaw = features[2]
        head_pose_pitch = features[3]
        gaze_focus = features[4]
        attention_score = 0.5
        if len(faces) == 0:
            attention_score = 0.1
        else:
            attention_score += face_confidence * 0.3
            if eye_openness < 0.3:
                attention_score -= 0.2
            else:
                attention_score += eye_openness * 0.2
            head_deviation = abs(head_pose_yaw) + abs(head_pose_pitch)
            if head_deviation < 0.3:
                attention_score += 0.2
            elif head_deviation > 0.7:
                attention_score -= 0.3
            attention_score += gaze_focus * 0.2
        attention_score = max(0.0, min(1.0, attention_score))
        if attention_score >= 0.8:
            engagement_level = 3
        elif attention_score >= 0.6:
            engagement_level = 2
        elif attention_score >= 0.3:
            engagement_level = 1
        else:
            engagement_level = 0
        if len(faces) == 0:
            emotion_class = 6
        elif eye_openness < 0.3:
            emotion_class = 4
        elif attention_score > 0.7:
            emotion_class = 1
        else:
            emotion_class = 2
        return attention_score, engagement_level, emotion_class
    def get_model_info(self) -> Dict:
        return {
            'model_loaded': self.model_loaded,
            'device': self.device,
            'model_type': 'SimpleAttentionModel' if self.model_loaded else 'rule_based'
        }
ai_detector = None
def get_ai_detector() -> AIAttentionDetectorProduction:
    global ai_detector
    if ai_detector is None:
        ai_detector = AIAttentionDetectorProduction()
    return ai_detector
if __name__ == "__main__":
    detector = AIAttentionDetectorProduction()
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    result = detector.analyze_frame(test_image)
    print("Test Results:")
    print(json.dumps(result, indent=2))