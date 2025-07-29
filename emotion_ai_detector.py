"""
Emotion-based Attention Detection
"""
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
class EmotionModel(nn.Module):
    def __init__(self, num_classes=7, input_size=(48, 48)):
        super().__init__()
        self.num_classes = num_classes
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
        )
        self.feature_size = 128 * 6 * 6
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.feature_size, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, self.feature_size)
        x = self.classifier(x)
        return x
class EmotionAIDetector:
    def __init__(self, models_dir: str = "models", model_path: str = None):
        self.models_dir = Path(models_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.emotion_labels = [
            'neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'contempt'
        ]
        self.attention_mapping = {
            'Angry': 0.2,      # Low attention when angry
            'Disgust': 0.15,   # Very low attention when disgusted
            'Fear': 0.3,       # Low attention when fearful
            'Happy': 0.9,      # High attention when happy
            'Sad': 0.25,       # Low attention when sad
            'Surprise': 0.8,   # High attention when surprised
            'Neutral': 0.6,     # Medium attention when neutral,
            'Contempt': 0.2,   # Low attention when contemptuous
            
        }
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.model = None
        self.model_loaded = False
        if model_path:
            self.load_model(model_path)
        else:
            default_paths = [
                "../attention_pulse/emotion_output/emotion_model_best.pth",
                "emotion_output/emotion_model_best.pth",
                "../attention_pulse/simple_90_output/model_best.pth"  # Fallback
            ]
            for path in default_paths:
                if Path(path).exists():
                    if "emotion" in path:
                        self.load_emotion_model(path)
                    else:
                        self.load_binary_model(path)
                    break
    def load_emotion_model(self, model_path: str) -> bool:
        try:
            model_path = Path(model_path)
            if not model_path.exists():
                logger.error(f"Model path does not exist: {model_path}")
                return False
            self.model = EmotionModel(num_classes=7)
            checkpoint = torch.load(model_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()
            self.model_loaded = True
            self.is_emotion_model = True
            logger.info(f"Successfully loaded emotion model from {model_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load emotion model: {e}")
            self.model_loaded = False
            return False
    def load_binary_model(self, model_path: str) -> bool:
        try:
            logger.info(f"Binary model fallback not implemented yet")
            return False
        except Exception as e:
            logger.error(f"Failed to load binary model: {e}")
            return False
    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        
        try:
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (48, 48))
            image = image.astype(np.float32) / 255.0
            tensor = torch.FloatTensor(image).permute(2, 0, 1)
            tensor = tensor.unsqueeze(0)
            return tensor.to(self.device)
        except Exception as e:
            logger.error(f"Image preprocessing failed: {e}")
            return None
    def detect_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.1, 
                minNeighbors=5, 
                minSize=(30, 30)
            )
            return faces
        except Exception as e:
            logger.error(f"Face detection failed: {e}")
            return []
    def predict_attention(self, frame: np.ndarray) -> Dict:
        try:
            if not self.model_loaded:
                return self._get_error_result("Model not loaded")
            faces = self.detect_faces(frame)
            if len(faces) == 0:
                return self._get_no_face_result()
            face = max(faces, key=lambda f: f[2] * f[3])
            x, y, w, h = face
            face_region = frame[y:y+h, x:x+w]
            input_tensor = self.preprocess_image(face_region)
            if input_tensor is None:
                return self._get_error_result("Image preprocessing failed")
            with torch.no_grad():
                output = self.model(input_tensor)
                probabilities = F.softmax(output, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0, predicted_class].item()
            predicted_emotion = self.emotion_labels[predicted_class]
            attention_score = self.attention_mapping[predicted_emotion]
            emotion_probs = probabilities[0].cpu().numpy()
            emotions = {}
            for i, emotion in enumerate(self.emotion_labels):
                emotions[emotion.lower()] = round(float(emotion_probs[i]), 3)
            emotions['primary_emotion'] = predicted_emotion.lower()
            if attention_score >= 0.8:
                attention_level = "High"
            elif attention_score >= 0.6:
                attention_level = "Medium"
            elif attention_score >= 0.4:
                attention_level = "Low"
            else:
                attention_level = "Very Low"
            result = {
                'attention_score': round(attention_score * 100, 1),
                'attention_percentage': f"{round(attention_score * 100, 1)}%",
                'attention_level': attention_level,
                'confidence': round(confidence, 3),
                'prediction_method': '7_emotion_classification',
                'model_name': 'Emotion Attention Detector',
                'face_detected': True,
                'timestamp': time.time(),
                'processing_time_ms': 25,
                'emotions': emotions,
                'primary_emotion': predicted_emotion.lower(),
                'emotion_confidence': round(confidence, 3)
            }
            return result
        except Exception as e:
            logger.error(f"Attention prediction failed: {e}")
            return self._get_error_result(f"Prediction failed: {str(e)}")
    def _get_no_face_result(self) -> Dict:
        return {
            'attention_score': 0.0,
            'attention_percentage': "0.0%",
            'attention_level': "No Face",
            'confidence': 0.0,
            'prediction_method': '7_emotion_classification',
            'model_name': 'Emotion Attention Detector',
            'face_detected': False,
            'timestamp': time.time(),
            'processing_time_ms': 5,
            'emotions': {
                'angry': 0.0,
                'disgust': 0.0,
                'fear': 0.0,
                'happy': 0.0,
                'sad': 0.0,
                'surprise': 0.0,
                'neutral': 0.0,
                'primary_emotion': 'none'
            },
            'primary_emotion': 'none',
            'emotion_confidence': 0.0
        }
    def _get_error_result(self, error_msg: str) -> Dict:
        return {
            'attention_score': 0.0,
            'attention_percentage': "0.0%",
            'attention_level': "Error",
            'confidence': 0.0,
            'prediction_method': '7_emotion_classification',
            'model_name': 'Emotion Attention Detector',
            'face_detected': False,
            'timestamp': time.time(),
            'processing_time_ms': 5,
            'emotions': {
                'angry': 0.0,
                'disgust': 0.0,
                'fear': 0.0,
                'happy': 0.0,
                'sad': 0.0,
                'surprise': 0.0,
                'neutral': 0.0,
                'primary_emotion': 'error'
            },
            'primary_emotion': 'error',
            'emotion_confidence': 0.0,
            'error': error_msg
        }
    def get_model_info(self) -> Dict:
        return {
            'model_type': 'Emotion Attention Detector',
            'accuracy': 'Emotion-based mapping',
            'method': '7_emotion_classification',
            'emotions': self.emotion_labels,
            'attention_mapping': self.attention_mapping,
            'real_time': True,
            'expression_aware': True,
            'model_loaded': self.model_loaded
        }