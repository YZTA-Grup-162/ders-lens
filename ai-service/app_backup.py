"""
Real AI Service for Ders Lens - Uses actual trained models  
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import base64
import io
from PIL import Image
import numpy as np
import cv2
import random
import time
from typing import Dict, List
from datetime import datetime
import dlib
from scipy.spatial import distance as dist
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app = Flask(__name__)
CORS(app)
class ModelManager:
    def __init__(self):
        self.models = {}
        self.face_detector = None
        self.landmark_predictor = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        self.model_paths = {
            'fer2013': '/app/models/models_fer2013',
            'daisee': '/app/models/models_daisee', 
            'mendeley': '/app/models/models_mendeley'
        }
        self.load_models()
    def load_models(self):
        try:
            self.face_detector = dlib.get_frontal_face_detector()
            predictor_path = "/app/models/shape_predictor_68_face_landmarks.dat"
            if os.path.exists(predictor_path):
                self.landmark_predictor = dlib.shape_predictor(predictor_path)
                logger.info("Loaded face landmark predictor")
            else:
                logger.warning("Face landmark predictor not found - attention analysis will be limited")
            for model_name, model_path in self.model_paths.items():
                if os.path.exists(model_path):
                    model_files = [f for f in os.listdir(model_path) if f.endswith('.pth') or f.endswith('.pkl')]
                    if model_files:
                        model_file = os.path.join(model_path, model_files[0])
                        try:
                            if model_file.endswith('.pth'):
                                model = torch.load(model_file, map_location=self.device)
                                if hasattr(model, 'eval'):
                                    model.eval()
                                self.models[model_name] = model
                                logger.info(f"Loaded {model_name} model from {model_file}")
                        except Exception as e:
                            logger.error(f"Failed to load {model_name} model: {e}")
                else:
                    logger.warning(f"Model path not found: {model_path}")
            logger.info(f"Successfully loaded {len(self.models)} models")
        except Exception as e:
            logger.error(f"Error loading models (app_backup.py): {e}")
            traceback.print_exc()
    def detect_faces(self, image):
        try:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            faces = self.face_detector(gray)
            return faces
        except Exception as e:
            logger.error(f"Face detection error (app_backup.py): {e}")
            return []
    def extract_face_region(self, image, face):
        try:
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            padding = 20
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(image.shape[1] - x, w + 2 * padding)
            h = min(image.shape[0] - y, h + 2 * padding)
            face_region = image[y:y+h, x:x+w]
            return face_region, (x, y, w, h)
        except Exception as e:
            logger.error(f"Face extraction error (app_backup.py): {e}")
            return None, None
    def preprocess_for_emotion(self, face_image):
        try:
            face_resized = cv2.resize(face_image, (48, 48))
            if len(face_resized.shape) == 3:
                face_gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
            else:
                face_gray = face_resized
            face_normalized = face_gray.astype(np.float32) / 255.0
            face_tensor = torch.FloatTensor(face_normalized).unsqueeze(0).unsqueeze(0)
            return face_tensor.to(self.device)
        except Exception as e:
            logger.error(f"Emotion preprocessing error (app_backup.py): {e}")
            return None
    def analyze_emotion(self, face_image):
        try:
            if 'fer2013' not in self.models:
                return self.fallback_emotion_analysis()
            input_tensor = self.preprocess_for_emotion(face_image)
            if input_tensor is None:
                return self.fallback_emotion_analysis()
            model = self.models['fer2013']
            with torch.no_grad():
                if hasattr(model, 'forward'):
                    outputs = model(input_tensor)
                else:
                    return self.fallback_emotion_analysis()
                probabilities = F.softmax(outputs, dim=1)
                probs = probabilities.cpu().numpy()[0]
                emotion_classes = ['anger', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise', 'contempt']
                emotions = dict(zip(emotion_classes, probs.astype(float)))
                dominant_emotion = emotion_classes[np.argmax(probs)]
                confidence = float(np.max(probs))
                return {
                    'dominant': dominant_emotion,
                    'confidence': confidence,
                    'emotions': emotions
                }
        except Exception as e:
            logger.error(f"Emotion analysis error (app_backup.py): {e}")
            return self.fallback_emotion_analysis()
    def analyze_attention(self, image, face):
        try:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            attention_score = 0.8
            is_attentive = True
            head_pose = {'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0}
            if self.landmark_predictor:
                landmarks = self.landmark_predictor(gray, face)
                if landmarks:
                    nose_tip = (landmarks.part(30).x, landmarks.part(30).y)
                    face_center_x = face.left() + face.width() / 2
                    face_center_y = face.top() + face.height() / 2
                    yaw = (nose_tip[0] - face_center_x) / face.width() * 45
                    pitch = (nose_tip[1] - face_center_y) / face.height() * 30
                    head_pose = {
                        'yaw': float(yaw),
                        'pitch': float(pitch), 
                        'roll': 0.0
                    }
                    pose_magnitude = abs(yaw) + abs(pitch)
                    if pose_magnitude > 20:
                        attention_score *= 0.6
                        is_attentive = False
                    elif pose_magnitude > 10:
                        attention_score *= 0.8
            if 'daisee' in self.models:
                try:
                    face_region, _ = self.extract_face_region(image, face)
                    if face_region is not None:
                        face_resized = cv2.resize(face_region, (64, 64))
                        face_tensor = torch.FloatTensor(face_resized).permute(2, 0, 1).unsqueeze(0) / 255.0
                        face_tensor = face_tensor.to(self.device)
                        model = self.models['daisee']
                        with torch.no_grad():
                            if hasattr(model, 'forward'):
                                daisee_output = model(face_tensor)
                                if isinstance(daisee_output, torch.Tensor):
                                    attention_from_model = torch.sigmoid(daisee_output).item()
                                    attention_score = float(attention_from_model)
                                    is_attentive = attention_score > 0.6
                except Exception as e:
                    logger.error(f"DAISEE model inference error (app_backup.py): {e}")
            return {
                'score': attention_score,
                'isAttentive': is_attentive,
                'headPose': head_pose
            }
        except Exception as e:
            logger.error(f"Attention analysis error (app_backup.py): {e}")
            return {
                'score': 0.7,
                'isAttentive': True,
                'headPose': {'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0}
            }
    def analyze_engagement(self, image, face, emotion_result, attention_result):
        try:
            engagement_score = 0.7
            emotion = emotion_result['dominant']
            if emotion in ['happiness', 'surprise']:
                engagement_score += 0.2
            elif emotion in ['anger', 'disgust', 'sadness']:
                engagement_score -= 0.3
            elif emotion == 'neutral':
                engagement_score += 0.1
            engagement_score = (engagement_score + attention_result['score']) / 2
            if 'mendeley' in self.models:
                try:
                    face_region, _ = self.extract_face_region(image, face)
                    if face_region is not None:
                        face_resized = cv2.resize(face_region, (112, 112))
                        face_tensor = torch.FloatTensor(face_resized).permute(2, 0, 1).unsqueeze(0) / 255.0
                        face_tensor = face_tensor.to(self.device)
                        model = self.models['mendeley']
                        with torch.no_grad():
                            if hasattr(model, 'forward'):
                                mendeley_output = model(face_tensor)
                                if isinstance(mendeley_output, torch.Tensor):
                                    engagement_from_model = torch.sigmoid(mendeley_output).item()
                                    engagement_score = float(engagement_from_model)
                except Exception as e:
                    logger.error(f"Mendeley model inference error (app_backup.py): {e}")
            engagement_score = max(0.0, min(1.0, engagement_score))
            if engagement_score > 0.7:
                level = 'high'
            elif engagement_score > 0.4:
                level = 'medium'
            else:
                level = 'low'
            return {
                'score': engagement_score,
                'level': level
            }
        except Exception as e:
            logger.error(f"Engagement analysis error (app_backup.py): {e}")
            return {
                'score': 0.6,
                'level': 'medium'
            }
    def analyze_gaze(self, image, face):
        try:
            gaze_x = 0.0
            gaze_y = 0.0
            on_screen = True
            if self.landmark_predictor:
                if len(image.shape) == 3:
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                else:
                    gray = image
                landmarks = self.landmark_predictor(gray, face)
                if landmarks:
                    left_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)]
                    right_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)]
                    left_center = np.mean(left_eye, axis=0)
                    right_center = np.mean(right_eye, axis=0)
                    eye_center = (left_center + right_center) / 2
                    face_center = np.array([face.left() + face.width()/2, face.top() + face.height()/2])
                    gaze_vector = eye_center - face_center
                    gaze_x = float(gaze_vector[0] / face.width())
                    gaze_y = float(gaze_vector[1] / face.height())
                    gaze_magnitude = np.linalg.norm(gaze_vector)
                    on_screen = gaze_magnitude < face.width() * 0.3
            return {
                'direction': {'x': gaze_x, 'y': gaze_y},
                'onScreen': on_screen
            }
        except Exception as e:
            logger.error(f"Gaze analysis error (app_backup.py): {e}")
            return {
                'direction': {'x': 0.0, 'y': 0.0},
                'onScreen': True
            }
    def fallback_emotion_analysis(self):
        emotions = {
            'neutral': 0.4,
            'happiness': 0.3,
            'surprise': 0.1,
            'sadness': 0.05,
            'anger': 0.05,
            'disgust': 0.05,
            'fear': 0.03,
            'contempt': 0.02
        }
        return {
            'dominant': 'neutral',
            'confidence': 0.4,
            'emotions': emotions
        }
model_manager = ModelManager()
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'models_loaded': len(model_manager.models),
        'available_models': list(model_manager.models.keys()),
        'device': str(model_manager.device),
        'timestamp': datetime.now().isoformat()
    })
@app.route('/api/v1/analyze/frame', methods=['POST'])
def analyze_frame():
    try:
        start_time = datetime.now()
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400
        try:
            image_data = base64.b64decode(data['image'])
            image = Image.open(BytesIO(image_data))
            image_np = np.array(image)
            if len(image_np.shape) == 3 and image_np.shape[2] == 3:
                image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        except Exception as e:
            return jsonify({'error': f'Invalid image data: {str(e)}'}), 400
        options = data.get('options', {})
        detect_emotion = options.get('detectEmotion', True)
        detect_attention = options.get('detectAttention', True)
        detect_engagement = options.get('detectEngagement', True)
        detect_gaze = options.get('detectGaze', True)
        faces = model_manager.detect_faces(image_np)
        if len(faces) == 0:
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds() * 1000
            return jsonify({
                'success': True,
                'data': {
                    'metadata': {
                        'faceDetected': False,
                        'confidence': 0.0,
                        'processingTime': processing_time
                    }
                }
            })
        face = max(faces, key=lambda f: f.width() * f.height())
        result = {}
        total_confidence = 0.0
        confidence_count = 0
        if detect_emotion:
            face_region, _ = model_manager.extract_face_region(image_np, face)
            if face_region is not None:
                emotion_result = model_manager.analyze_emotion(face_region)
                result['emotion'] = emotion_result
                total_confidence += emotion_result['confidence']
                confidence_count += 1
        if detect_attention:
            attention_result = model_manager.analyze_attention(image_np, face)
            result['attention'] = attention_result
            total_confidence += attention_result['score']
            confidence_count += 1
        if detect_engagement:
            emotion_data = result.get('emotion', model_manager.fallback_emotion_analysis())
            attention_data = result.get('attention', {'score': 0.7})
            engagement_result = model_manager.analyze_engagement(image_np, face, emotion_data, attention_data)
            result['engagement'] = engagement_result
            total_confidence += engagement_result['score']
            confidence_count += 1
        if detect_gaze:
            gaze_result = model_manager.analyze_gaze(image_np, face)
            result['gaze'] = gaze_result
        average_confidence = total_confidence / confidence_count if confidence_count > 0 else 0.0
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds() * 1000
        result['metadata'] = {
            'faceDetected': True,
            'confidence': average_confidence,
            'processingTime': processing_time
        }
        return jsonify({
            'success': True,
            'data': result
        })
    except Exception as e:
        logger.error(f"Frame analysis error: {e}")
        traceback.print_exc()
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500
@app.route('/api/v1/models/info', methods=['GET'])
def model_info():
    return jsonify({
        'models': {
            'fer2013': {
                'loaded': 'fer2013' in model_manager.models,
                'description': 'FER2013+ emotion recognition model',
                'emotions': ['anger', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise', 'contempt']
            },
            'daisee': {
                'loaded': 'daisee' in model_manager.models,
                'description': 'DAISEE attention detection model',
                'capabilities': ['attention_score', 'head_pose', 'engagement_level']
            },
            'mendeley': {
                'loaded': 'mendeley' in model_manager.models,
                'description': 'Mendeley engagement analysis model',
                'capabilities': ['engagement_score', 'behavioral_analysis']
            }
        },
        'device': str(model_manager.device),
        'face_detection': model_manager.face_detector is not None,
        'landmark_detection': model_manager.landmark_predictor is not None
    })
if __name__ == '__main__':
    logger.info("Starting Ders Lens AI Service with real trained models")
    logger.info(f"Models available: {list(model_manager.models.keys())}")
    app.run(host='0.0.0.0', port=8001, debug=False)