"""
ONNX Emotion Detection Integration
"""
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image
logger = logging.getLogger(__name__)
class ONNXEmotionDetector:
    ['neutral', 'sadness', 'happiness', 'surprise', 'anger', 'fear', 'contempt', 'disgust']
    def __init__(self, model_path: str = None):
        possible_paths = [
            "backend/models/onnx/best_model.onnx",
            "models/onnx/best_model.onnx", 
            "best_model.onnx",
            "../best_model.onnx",
            "backend/best_model.onnx"
        ]
        if model_path:
            possible_paths.insert(0, model_path)
        self.model_path = None
        for path in possible_paths:
            if Path(path).exists():
                self.model_path = Path(path)
                break
        self.session = None
        self.model_loaded = False
        self.emotion_labels_8 = [
            'neutral', 'sadness', 'happiness', 'surprise', 
            'anger', 'fear', 'contempt', 'disgust'
        ]
        self.emotion_labels_7 = [
            'angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'
        ]
        self.emotion_labels = self.emotion_labels_8
        self.num_emotions = 8
        self.emotion_to_attention = {
            'happiness': 0.85,    
            'surprise': 0.75,     
            'neutral': 0.65,      
            'contempt': 0.35,     
            'anger': 0.25,        
            'fear': 0.30,         
            'sadness': 0.15,      
            'disgust': 0.10,
            'happy': 0.85,
            'angry': 0.25,
            'sad': 0.15
        }
        try:
            import mediapipe as mp
            self.mp_face_detection = mp.solutions.face_detection
            self.face_detection = self.mp_face_detection.FaceDetection(
                model_selection=1, min_detection_confidence=0.7
            )
            self.use_mediapipe = True
        except ImportError:
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            self.use_mediapipe = False
        self.load_model()
        logger.info(f"🎭 ONNX Emotion Detector initialized")
        logger.info(f"📊 Model loaded: {self.model_loaded}")
        logger.info(f"🏷️ Emotions: {self.emotion_labels}")
    def load_model(self) -> bool:
        try:
            if not self.model_path or not self.model_path.exists():
                logger.error(f"❌ ONNX model not found at any expected location")
                return False
            providers = ['CPUExecutionProvider']
            if ort.get_available_providers():
                available_providers = ort.get_available_providers()
                if 'CUDAExecutionProvider' in available_providers:
                    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                elif 'DirectMLExecutionProvider' in available_providers:
                    providers = ['DirectMLExecutionProvider', 'CPUExecutionProvider']
            self.session = ort.InferenceSession(
                str(self.model_path), 
                providers=providers
            )
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            self.input_shape = self.session.get_inputs()[0].shape
            self.output_shape = self.session.get_outputs()[0].shape
            if len(self.output_shape) > 1:
                self.num_emotions = self.output_shape[-1]
            else:
                self.num_emotions = self.output_shape[0]
            if self.num_emotions == 7:
                self.emotion_labels = self.emotion_labels_7
                logger.info("🔍 Using 7-emotion model (FER2013 format)")
            elif self.num_emotions == 8:
                self.emotion_labels = self.emotion_labels_8
                logger.info("🔍 Using 8-emotion model")
            else:
                logger.warning(f"⚠️ Unexpected number of emotions: {self.num_emotions}")
                self.emotion_labels = [f"emotion_{i}" for i in range(self.num_emotions)]
            self.model_loaded = True
            logger.info(f"✅ ONNX model loaded successfully")
            logger.info(f"📐 Input shape: {self.input_shape}")
            logger.info(f"📊 Output shape: {self.output_shape}")
            logger.info(f"🎭 Emotions ({self.num_emotions}): {self.emotion_labels}")
            logger.info(f"🔧 Providers: {providers}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to load ONNX model: {e}")
            self.model_loaded = False
            return False
    def detect_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        try:
            if self.use_mediapipe:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.face_detection.process(rgb_frame)
                faces = []
                if results.detections:
                    h, w = frame.shape[:2]
                    for detection in results.detections:
                        bbox = detection.location_data.relative_bounding_box
                        x = int(bbox.xmin * w)
                        y = int(bbox.ymin * h)
                        width = int(bbox.width * w)
                        height = int(bbox.height * h)
                        x = max(0, x)
                        y = max(0, y)
                        width = min(width, w - x)
                        height = min(height, h - y)
                        if width > 0 and height > 0:
                            faces.append((x, y, width, height))
                return faces
            else:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(
                    gray, 
                    scaleFactor=1.1, 
                    minNeighbors=5, 
                    minSize=(30, 30)
                )
                return [tuple(face) for face in faces]
        except Exception as e:
            logger.error(f"Face detection failed: {e}")
            return []
    def preprocess_face(self, face_img: np.ndarray) -> Optional[np.ndarray]:
        try:
            face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            if len(self.input_shape) == 4:
                if self.input_shape[1] == 3:
                    target_size = (self.input_shape[2], self.input_shape[3])
                else:
                    target_size = (self.input_shape[1], self.input_shape[2])
            else:
                target_size = (48, 48)
            face_resized = cv2.resize(face_rgb, target_size)
            face_normalized = face_resized.astype(np.float32) / 255.0
            if len(self.input_shape) == 4:
                if self.input_shape[1] == 3:
                    face_tensor = np.transpose(face_normalized, (2, 0, 1))
                    face_tensor = np.expand_dims(face_tensor, axis=0)
                else:
                    face_tensor = np.expand_dims(face_normalized, axis=0)
            else:
                face_tensor = face_normalized.flatten()
                face_tensor = np.expand_dims(face_tensor, axis=0)
            return face_tensor
        except Exception as e:
            logger.error(f"Face preprocessing failed: {e}")
            return None
    def predict_emotion(self, face_tensor: np.ndarray) -> Optional[Dict]:
        try:
            if not self.model_loaded or self.session is None:
                logger.error("Model not loaded or session is None")
                return None
            outputs = self.session.run(
                [self.output_name], 
                {self.input_name: face_tensor}
            )
            raw_output = outputs[0]
            if len(raw_output.shape) == 2:
                probabilities = raw_output[0]
            elif len(raw_output.shape) == 1:
                probabilities = raw_output
            else:
                logger.error(f"Unexpected output shape: {raw_output.shape}")
                return None
            if len(probabilities) != len(self.emotion_labels):
                logger.error(f"Mismatch: model output {len(probabilities)} vs emotion labels {len(self.emotion_labels)}")
                return None
            if abs(np.sum(probabilities) - 1.0) > 0.1:
                probabilities = self._softmax(probabilities)
            predicted_idx = np.argmax(probabilities)
            if predicted_idx >= len(self.emotion_labels):
                logger.error(f"Predicted index {predicted_idx} out of range for {len(self.emotion_labels)} emotions")
                return None
            predicted_emotion = self.emotion_labels[predicted_idx]
            confidence = float(probabilities[predicted_idx])
            return {
                'emotion_index': int(predicted_idx),
                'emotion': predicted_emotion,
                'confidence': confidence,
                'probabilities': probabilities.tolist()
            }
        except Exception as e:
            logger.error(f"Emotion prediction failed: {e}")
            return None
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)
    def get_emotion_distribution(self, probabilities: List[float]) -> Dict[str, float]:
        emotion_dist = {}
        for i, emotion in enumerate(self.emotion_labels):
            emotion_dist[emotion] = float(probabilities[i])
        return emotion_dist
    def map_emotion_to_attention(self, emotion: str, confidence: float) -> float:
        base_attention = self.emotion_to_attention.get(emotion, 0.5)
        confidence_factor = (confidence - 0.5) * 0.3
        attention_score = base_attention + confidence_factor
        attention_score = max(0.0, min(1.0, attention_score))
        return attention_score
    def classify_attention_level(self, attention_score: float) -> str:
        if attention_score >= 0.8:
            return "highly_attentive"
        elif attention_score >= 0.6:
            return "attentive"
        elif attention_score >= 0.4:
            return "moderately_attentive"
        elif attention_score >= 0.2:
            return "distracted"
        else:
            return "not_attentive"
    def predict_attention(self, frame: np.ndarray) -> Dict:
        try:
            start_time = time.time()
            faces = self.detect_faces(frame)
            if len(faces) == 0:
                return {
                    "attention_score": 0.3,
                    "attention_percentage": "30.0%",
                    "attention_level": "no_face_detected",
                    "confidence": 0.0,
                    "emotions": {emotion: 0.0 for emotion in self.emotion_labels},
                    "primary_emotion": "unknown",
                    "emotion_confidence": 0.0,
                    "face_detected": False,
                    "model_type": "ONNX_8_Emotions",
                    "processing_time": time.time() - start_time,
                    "distractions": ["No face visible"],
                    "timestamp": time.time()
                }
            face_data = faces[0]
            if len(face_data) != 4:
                logger.error(f"Invalid face data: {face_data}")
                return self._default_response("invalid_face_data", start_time)
            (x, y, w, h) = face_data
            if x < 0 or y < 0 or w <= 0 or h <= 0:
                logger.error(f"Invalid face coordinates: x={x}, y={y}, w={w}, h={h}")
                return self._default_response("invalid_coordinates", start_time)
            frame_h, frame_w = frame.shape[:2]
            if x + w > frame_w or y + h > frame_h:
                logger.warning(f"Face region exceeds frame bounds, clipping")
                w = min(w, frame_w - x)
                h = min(h, frame_h - y)
            face_img = frame[y:y+h, x:x+w]
            if face_img.size == 0:
                logger.error("Extracted face image is empty")
                return self._default_response("empty_face", start_time)
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
            attention_percentage = attention_score * 100
            distractions = []
            if attention_score < 0.5:
                distractions.append(f"Showing {primary_emotion} emotion")
            if emotion_confidence < 0.6:
                distractions.append("Uncertain emotion detection")
            return {
                "attention_score": attention_percentage,
                "attention_percentage": f"{attention_percentage:.1f}%",
                "attention_level": attention_level,
                "confidence": emotion_confidence,
                "emotions": emotions,
                "primary_emotion": primary_emotion,
                "emotion_confidence": emotion_confidence,
                "face_detected": True,
                "model_type": "ONNX_8_Emotions",
                "model_accuracy": "High-Performance ONNX",
                "processing_time": time.time() - start_time,
                "distractions": distractions if distractions else ["None detected"],
                "timestamp": time.time(),
                "face_count": len(faces),
                "face_box": [int(x), int(y), int(w), int(h)]
            }
        except Exception as e:
            logger.error(f"Attention prediction failed: {e}")
            return self._default_response("error", start_time, str(e))
    def _default_response(self, reason: str, start_time: float, error_msg: str = None) -> Dict:
        return {
            "attention_score": 30.0,
            "attention_percentage": "30.0%",
            "attention_level": reason,
            "confidence": 0.0,
            "emotions": {emotion: 0.0 for emotion in self.emotion_labels},
            "primary_emotion": "unknown",
            "emotion_confidence": 0.0,
            "face_detected": False,
            "model_type": "ONNX_8_Emotions",
            "processing_time": time.time() - start_time,
            "distractions": [error_msg] if error_msg else ["Processing failed"],
            "timestamp": time.time(),
            "error": error_msg
        }
    def get_model_info(self) -> Dict:
        return {
            "model_type": "ONNX 8-Emotion Detector",
            "model_loaded": self.model_loaded,
            "emotion_classes": len(self.emotion_labels),
            "emotions": self.emotion_labels,
            "architecture": "ONNX Runtime",
            "model_path": str(self.model_path),
            "features": [
                "8-emotion classification",
                "ONNX Runtime optimization",
                "Cross-platform inference",
                "GPU acceleration support",
                "Real-time processing",
                "MediaPipe face detection"
            ]
        }
if __name__ == "__main__":
    detector = ONNXEmotionDetector()
    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = detector.predict_attention(test_frame)
    print("Test result:", result)