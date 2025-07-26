"""
Emotion Detector
"""

import logging
import os
import time
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from torchvision.models import ResNet18_Weights, resnet18
logger = logging.getLogger(__name__)
class UltimateEmotionModel(nn.Module):
  
    def __init__(self, num_classes=7, dropout_rate=0.5):
        super().__init__()
        self.backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        return self.backbone(x)
class UltimateEmotionDetector:
    def __init__(self, model_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        if model_path is None:
            model_path = Path("../attention_pulse/ultimate_emotion_output/ultimate_model_best.pth")
        self.model_loaded = self.load_model(model_path)
        self.emotion_to_attention = {
            'Happy': 0.9,      # Very attentive
            'Surprise': 0.8,   # High attention
            'Neutral': 0.6,    # Moderate attention
            'Angry': 0.4,      # Low attention (distracted by anger)
            'Fear': 0.3,       # Low attention (distracted by fear)
            'Sad': 0.3,        # Low attention (distracted by sadness)
            'Disgust': 0.2     # Very low attention (strong negative emotion)
        }
        logger.info(f"🎭 Ultimate Emotion Detector initialized (device: {self.device})")
        logger.info(f"📊 Model loaded: {self.model_loaded}")
    def load_model(self, model_path):
        try:
            if not os.path.exists(model_path):
                logger.warning(f"⚠️ Model file not found: {model_path}")
                return False
            self.model = UltimateEmotionModel(num_classes=7, dropout_rate=0.4)
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            val_accuracy = checkpoint.get('val_accuracy', 0)
            epoch = checkpoint.get('epoch', 0)
            logger.info(f"✅ Ultimate emotion model loaded successfully")
            logger.info(f"📈 Model accuracy: {val_accuracy:.2f}% (epoch {epoch})")
            logger.info(f"🧠 Transfer learning: ResNet18 + fine-tuning")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to load ultimate emotion model: {e}")
            return False
    def detect_faces(self, frame):
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
    def preprocess_face(self, face_img):
        try:
            face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(face_rgb)
            tensor_img = self.transform(pil_img)
            tensor_img = tensor_img.unsqueeze(0)
            return tensor_img.to(self.device)
        except Exception as e:
            logger.error(f"Face preprocessing failed: {e}")
            return None
    def predict_emotion(self, face_tensor):
        try:
            if not self.model_loaded:
                return None
            with torch.no_grad():
                outputs = self.model(face_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                return {
                    'emotion_index': predicted.item(),
                    'emotion': self.emotion_labels[predicted.item()],
                    'confidence': confidence.item(),
                    'probabilities': probabilities.cpu().numpy()[0]
                }
        except Exception as e:
            logger.error(f"Emotion prediction failed: {e}")
            return None
    def get_emotion_distribution(self, probabilities):
        emotion_dist = {}
        for i, emotion in enumerate(self.emotion_labels):
            emotion_dist[emotion.lower()] = float(probabilities[i])
        return emotion_dist
    def map_emotion_to_attention(self, emotion, confidence):
        base_attention = self.emotion_to_attention.get(emotion, 0.5)
        confidence_factor = confidence * 0.3
        attention_score = base_attention + confidence_factor
        attention_score = max(0.0, min(1.0, attention_score))
        return attention_score
    def classify_attention_level(self, attention_score):
        if attention_score >= 0.7:
            return "highly_attentive"
        elif attention_score >= 0.5:
            return "attentive"
        elif attention_score >= 0.3:
            return "distracted"
        else:
            return "not_attentive"
    def predict_attention(self, frame):
        try:
            start_time = time.time()
            faces = self.detect_faces(frame)
            if len(faces) == 0:
                return {
                    "attention_score": 0.5,
                    "attention_level": "no_face_detected",
                    "confidence": 0.0,
                    "emotions": {emotion.lower(): 0.0 for emotion in self.emotion_labels},
                    "primary_emotion": "unknown",
                    "emotion_confidence": 0.0,
                    "face_detected": False,
                    "model_type": "UltimateEmotionResNet18",
                    "processing_time": time.time() - start_time,
                    "distractions": ["No face visible"],
                    "timestamp": time.time()
                }
            (x, y, w, h) = faces[0]
            face_img = frame[y:y+h, x:x+w]
            face_tensor = self.preprocess_face(face_img)
            if face_tensor is None:
                return self._default_response("preprocessing_failed", start_time)
            emotion_result = self.predict_emotion(face_tensor)
            if emotion_result is None:
                return self._default_response("prediction_failed", start_time)
            primary_emotion = emotion_result['emotion']
            emotion_confidence = emotion_result['confidence']
            probabilities = emotion_result['probabilities']
            emotions = self.get_emotion_distribution(probabilities)
            attention_score = self.map_emotion_to_attention(primary_emotion, emotion_confidence)
            attention_level = self.classify_attention_level(attention_score)
            distractions = []
            if attention_score < 0.5:
                distractions.append(f"Showing {primary_emotion.lower()} emotion")
            if emotion_confidence < 0.5:
                distractions.append("Uncertain emotion detection")
            return {
                "attention_score": attention_score,
                "attention_level": attention_level,
                "confidence": emotion_confidence,
                "emotions": emotions,
                "primary_emotion": primary_emotion.lower(),
                "emotion_confidence": emotion_confidence,
                "face_detected": True,
                "model_type": "UltimateEmotionResNet18",
                "model_accuracy": "34%+ (Transfer Learning)",
                "processing_time": time.time() - start_time,
                "distractions": distractions if distractions else ["None detected"],
                "timestamp": time.time(),
                "face_count": len(faces),
                "face_box": [int(x), int(y), int(w), int(h)]
            }
        except Exception as e:
            logger.error(f"Attention prediction failed: {e}")
            return self._default_response("error", start_time, str(e))
    def _default_response(self, reason, start_time, error_msg=None):
        return {
            "attention_score": 0.5,
            "attention_level": reason,
            "confidence": 0.0,
            "emotions": {emotion.lower(): 0.0 for emotion in self.emotion_labels},
            "primary_emotion": "unknown",
            "emotion_confidence": 0.0,
            "face_detected": False,
            "model_type": "UltimateEmotionResNet18",
            "processing_time": time.time() - start_time,
            "distractions": [error_msg] if error_msg else ["Processing failed"],
            "timestamp": time.time(),
            "error": error_msg
        }
    def get_model_info(self):
        return {
            "model_type": "Ultimate Emotion ResNet18",
            "model_loaded": self.model_loaded,
            "device": str(self.device),
            "emotion_classes": len(self.emotion_labels),
            "emotions": self.emotion_labels,
            "architecture": "ResNet18 + Transfer Learning",
            "accuracy": "34%+ (improving)",
            "features": [
                "7-emotion classification",
                "Transfer learning from ImageNet",
                "Two-phase training (frozen + fine-tuning)",
                "Class balancing",
                "Intelligent attention mapping"
            ]
        }
if __name__ == "__main__":
    detector = UltimateEmotionDetector()
    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = detector.predict_attention(test_frame)
    print("Test result:", result)