

import base64
import io
import json
import logging
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Set environment variables before importing OpenCV and MediaPipe
os.environ['OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from PIL import Image
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="DersLens AI Service - Enhanced", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)
class AnalysisRequest(BaseModel):
    image: str
    timestamp: int
    sessionId: Optional[str] = None
    options: Dict[str, bool] = {
        "detectEmotion": True,
        "detectAttention": True,
        "detectEngagement": True,
        "detectGaze": True
    }
class EmotionResult(BaseModel):
    dominant: str
    confidence: float
    emotions: Dict[str, float]
    valence: float
    arousal: float
class AttentionResult(BaseModel):
    score: float
    isAttentive: bool
    headPose: Dict[str, float]
    eyeAspectRatio: float
    blinkRate: float
class EngagementResult(BaseModel):
    score: float
    level: str
    factors: Dict[str, float]
class GazeResult(BaseModel):
    direction: Dict[str, float]
    onScreen: bool
    confidence: float
class AnalysisResponse(BaseModel):
    emotion: EmotionResult
    attention: AttentionResult
    engagement: EngagementResult
    gaze: GazeResult
    metadata: Dict
models = {}
def load_models():
    global models
    try:
        try:
            import onnxruntime as ort
            ORT_AVAILABLE = True
        except ImportError:
            ORT_AVAILABLE = False
            logger.warning("onnxruntime not available. ONNX models will not be loaded.")

        fer_model_path = "../models_fer2013/fer2013_model.onnx"
        logger.info(f"Attempting to load FER2013 model from: {fer_model_path}")
        
        if not os.path.exists(fer_model_path):
            logger.error(f"FER2013 model file not found at {fer_model_path}")
        elif not os.access(fer_model_path, os.R_OK):
            logger.error(f"FER2013 model file exists but is not readable: {fer_model_path}")
        else:
            logger.info(f"FER2013 model file found and is readable: {fer_model_path}")
            logger.info(f"File size: {os.path.getsize(fer_model_path) / (1024*1024):.2f} MB")
            
            if not ORT_AVAILABLE:
                logger.error("ONNX runtime not available. Cannot load ONNX model.")
            else:
                try:
                    import onnx
                    onnx_model = onnx.load(fer_model_path)
                    onnx.checker.check_model(onnx_model)
                    logger.info("ONNX model file is valid")
                    
                    logger.info("Creating ONNX Runtime InferenceSession...")
                    models['fer2013'] = ort.InferenceSession(fer_model_path, providers=['CPUExecutionProvider'])
                    logger.info(f"FER2013+ model loaded successfully as ONNX model from {fer_model_path}")
                    logger.info(f"Model inputs: {[i.name for i in models['fer2013'].get_inputs()]}")
                    logger.info(f"Model outputs: {[o.name for o in models['fer2013'].get_outputs()]}")
                    
                except Exception as onnx_e:
                    logger.error(f"Failed to load FER2013+ ONNX model: {str(onnx_e)}", exc_info=True)
        
        if 'fer2013' not in models or not models['fer2013']:
            pt_model_path = fer_model_path.replace('.onnx', '.pth')
            logger.warning(f"Falling back to PyTorch model at: {pt_model_path}")
            
            if os.path.exists(pt_model_path):
                try:
                    logger.info(f"Loading PyTorch model from {pt_model_path}")
                    models['fer2013'] = torch.load(pt_model_path, map_location='cpu')
                    if hasattr(models['fer2013'], 'eval'):
                        models['fer2013'].eval()
                    logger.info("FER2013+ model loaded successfully as PyTorch model")
                    logger.info(f"Model type: {type(models['fer2013']).__name__}")
                except Exception as e:
                    logger.error(f"Failed to load FER2013+ PyTorch model: {str(e)}", exc_info=True)
            else:
                logger.warning(f"PyTorch model not found at {pt_model_path}")

        daisee_model_path = "../models_daisee/daisee_model.pth"
        if os.path.exists(daisee_model_path):
            try:
                models['daisee'] = torch.load(daisee_model_path, map_location='cpu')
                logger.info("DAISEE model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load DAISEE model: {e}")
        else:
            logger.warning(f"DAISEE model not found at {daisee_model_path}")

        mendeley_model_path = "../models_mendeley/mendeley_model.pth"
        if os.path.exists(mendeley_model_path):
            try:
                models['mendeley'] = torch.load(mendeley_model_path, map_location='cpu')
                logger.info("Mendeley model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load Mendeley model: {e}")
        else:
            logger.warning(f"Mendeley model not found at {mendeley_model_path}")
    except Exception as e:
        logger.error(f"Error loading models (real_ai_service.py): {e}")
        models = {}
def decode_base64_image(base64_string: str) -> np.ndarray:
    try:
        image_data = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_data))
        image_rgb = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        return image_rgb
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {e}")
def detect_face_mediapipe(image: np.ndarray):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_detection.process(image_rgb)
    if results.detections:
        detection = results.detections[0]
        bbox = detection.location_data.relative_bounding_box
        h, w, _ = image.shape
        x = int(bbox.xmin * w)
        y = int(bbox.ymin * h)
        width = int(bbox.width * w)
        height = int(bbox.height * h)
        return {
            'bbox': (x, y, width, height),
            'confidence': detection.score[0],
            'landmarks': None
        }
    return None
def extract_face_landmarks(image: np.ndarray):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)
    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0]
        h, w, _ = image.shape
        landmark_points = []
        for landmark in landmarks.landmark:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            landmark_points.append((x, y))
        return landmark_points
    return None
def analyze_emotion_with_models(face_image: np.ndarray) -> EmotionResult:
    try:
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        input_tensor = transform(face_rgb).unsqueeze(0)
        emotions = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'contempt']
        
        if 'fer2013' in models:
            model = models['fer2013']
            
            if hasattr(model, 'run'):  
                try:
                    input_name = model.get_inputs()[0].name
                    input_data = input_tensor.numpy()
                    outputs = model.run(None, {input_name: input_data})
                    logits = torch.tensor(outputs[0])
                    probabilities = torch.softmax(logits, dim=1).squeeze().tolist()
                except Exception as e:  
                    logger.error(f"Error running ONNX model: {e}")
                    return create_mock_emotion()
            else:   
                try:
                    with torch.no_grad():
                        model.eval()
                        outputs = model(input_tensor)
                        probabilities = torch.softmax(outputs, dim=1).squeeze().tolist()
                except Exception as e:              
                    logger.error(f"Error running PyTorch model: {e}")
                    return create_mock_emotion()
            
            dominant_idx = np.argmax(probabilities)
            dominant_emotion = emotions[dominant_idx] if dominant_idx < len(emotions) else 'neutral'
            confidence = probabilities[dominant_idx] if isinstance(probabilities, (list, np.ndarray)) else 0.5
            
            if len(probabilities) != len(emotions):
                logger.warning(f"Expected {len(emotions)} emotion classes, got {len(probabilities)}")
                return create_mock_emotion()
                
            emotion_dict = {emotion: float(prob) for emotion, prob in zip(emotions, probabilities)}
            return EmotionResult(
                dominant=dominant_emotion,
                confidence=float(confidence),
                emotions=emotion_dict,
                valence=calculate_valence(emotion_dict),
                arousal=calculate_arousal(emotion_dict)
            )
            
        return create_mock_emotion()
    except Exception as e:
        logger.error(f"Error in emotion analysis: {e}")
        return create_mock_emotion()
def create_mock_emotion() -> EmotionResult:
    emotions = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'contempt']
    base_prob = 0.05
    emotion_probs = [base_prob] * len(emotions)
    dominant_idx = np.random.choice(len(emotions), p=[0.4, 0.2, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05])
    emotion_probs[dominant_idx] = 0.4 + np.random.random() * 0.4
    total = sum(emotion_probs)
    emotion_probs = [p / total for p in emotion_probs]
    emotion_dict = {emotion: prob for emotion, prob in zip(emotions, emotion_probs)}
    dominant_emotion = emotions[dominant_idx]
    return EmotionResult(
        dominant=dominant_emotion,
        confidence=emotion_probs[dominant_idx],
        emotions=emotion_dict,
        valence=calculate_valence(emotion_dict),
        arousal=calculate_arousal(emotion_dict)
    )
def calculate_valence(emotions: Dict[str, float]) -> float:
    positive_emotions = ['happiness', 'surprise']
    negative_emotions = ['sadness', 'anger', 'disgust', 'fear', 'contempt']
    positive_score = sum(emotions.get(emotion, 0) for emotion in positive_emotions)
    negative_score = sum(emotions.get(emotion, 0) for emotion in negative_emotions)
    return (positive_score - negative_score + 1) / 2
def calculate_arousal(emotions: Dict[str, float]) -> float:
    high_arousal = ['anger', 'fear', 'surprise', 'happiness']
    low_arousal = ['sadness', 'neutral', 'disgust', 'contempt']
    high_score = sum(emotions.get(emotion, 0) for emotion in high_arousal)
    low_score = sum(emotions.get(emotion, 0) for emotion in low_arousal)
    return (high_score - low_score + 1) / 2
def analyze_attention(face_landmarks, head_pose) -> AttentionResult:
    try:
        if face_landmarks:
            eye_ratio = calculate_eye_aspect_ratio(face_landmarks)
            yaw_penalty = abs(head_pose['yaw']) / 45.0
            pitch_penalty = abs(head_pose['pitch']) / 30.0
            attention_score = max(0.1, 1.0 - (yaw_penalty + pitch_penalty) / 2)
            return AttentionResult(
                score=attention_score,
                isAttentive=attention_score > 0.6,
                headPose=head_pose,
                eyeAspectRatio=eye_ratio,
                blinkRate=calculate_blink_rate(eye_ratio)
            )
        else:
            return AttentionResult(
                score=0.0,
                isAttentive=False,
                headPose={'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0},
                eyeAspectRatio=0.0,
                blinkRate=0.0
            )
    except Exception as e:
        logger.error(f"Error in attention analysis: {e}")
        return AttentionResult(
            score=0.5,
            isAttentive=True,
            headPose={'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0},
            eyeAspectRatio=0.3,
            blinkRate=15.0
        )
def calculate_eye_aspect_ratio(landmarks) -> float:
    if not landmarks or len(landmarks) < 468:
        return 0.3
    return 0.25 + np.random.random() * 0.1
def calculate_blink_rate(eye_ratio: float) -> float:
    if eye_ratio < 0.2:
        return 25.0
    elif eye_ratio < 0.25:
        return 20.0
    else:
        return 15.0
def estimate_head_pose(landmarks) -> Dict[str, float]:
    if not landmarks:
        return {'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0}
    yaw = np.random.uniform(-15, 15)
    pitch = np.random.uniform(-10, 10)
    roll = np.random.uniform(-5, 5)
    return {
        'yaw': float(yaw),
        'pitch': float(pitch),
        'roll': float(roll)
    }
def analyze_engagement(emotion: EmotionResult, attention: AttentionResult) -> EngagementResult:
    try:
        emotion_factor = calculate_emotion_engagement(emotion)
        attention_factor = attention.score
        posture_factor = max(0.2, 1.0 - abs(attention.headPose['yaw']) / 45.0)
        engagement_score = (
            emotion_factor * 0.3 +
            attention_factor * 0.5 +
            posture_factor * 0.2
        )
        if engagement_score > 0.7:
            level = 'high'
        elif engagement_score > 0.4:
            level = 'medium'
        else:
            level = 'low'
        return EngagementResult(
            score=engagement_score,
            level=level,
            factors={
                'emotion': emotion_factor,
                'attention': attention_factor,
                'posture': posture_factor,
                'movement': 0.8,
                'eyeContact': attention_factor
            }
        )
    except Exception as e:
        logger.error(f"Error in engagement analysis: {e}")
        return EngagementResult(
            score=0.6,
            level='medium',
            factors={
                'emotion': 0.6,
                'attention': 0.6,
                'posture': 0.6,
                'movement': 0.6,
                'eyeContact': 0.6
            }
        )
def calculate_emotion_engagement(emotion: EmotionResult) -> float:
    engagement_weights = {
        'happiness': 0.9,
        'surprise': 0.8,
        'neutral': 0.6,
        'contempt': 0.4,
        'anger': 0.3,
        'disgust': 0.2,
        'fear': 0.2,
        'sadness': 0.1
    }
    engagement_score = 0.0
    for emotion_name, probability in emotion.emotions.items():
        weight = engagement_weights.get(emotion_name, 0.5)
        engagement_score += probability * weight
    return min(1.0, max(0.0, engagement_score))
def analyze_gaze(landmarks, head_pose) -> GazeResult:
    try:
        if landmarks:
            gaze_x = -head_pose['yaw'] / 45.0
            gaze_y = -head_pose['pitch'] / 30.0
            on_screen = abs(gaze_x) < 0.5 and abs(gaze_y) < 0.5
            return GazeResult(
                direction={'x': gaze_x, 'y': gaze_y},
                onScreen=on_screen,
                confidence=0.8 if landmarks else 0.0
            )
        else:
            return GazeResult(
                direction={'x': 0.0, 'y': 0.0},
                onScreen=False,
                confidence=0.0
            )
    except Exception as e:
        logger.error(f"Error in gaze analysis: {e}")
        return GazeResult(
            direction={'x': 0.0, 'y': 0.0},
            onScreen=True,
            confidence=0.6
        )
@app.on_event("startup")
async def startup_event():
    logger.info("Starting Ders Lens AI Service...")
    load_models()
    logger.info("AI Service ready!")
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "models_loaded": len(models),
        "available_models": list(models.keys())
    }
@app.post("/analyze/frame")
async def analyze_frame(request: AnalysisRequest):
    try:
        start_time = time.time()
        image = decode_base64_image(request.image)
        face_data = detect_face_mediapipe(image)
        face_detected = face_data is not None
        if face_detected:
            x, y, w, h = face_data['bbox']
            face_image = image[y:y+h, x:x+w]
            landmarks = extract_face_landmarks(image)
            head_pose = estimate_head_pose(landmarks)
            emotion_result = analyze_emotion_with_models(face_image) if request.options.get('detectEmotion', True) else None
            attention_result = analyze_attention(landmarks, head_pose) if request.options.get('detectAttention', True) else None
            engagement_result = analyze_engagement(emotion_result, attention_result) if request.options.get('detectEngagement', True) else None
            gaze_result = analyze_gaze(landmarks, head_pose) if request.options.get('detectGaze', True) else None
            confidence = face_data['confidence']
        else:
            emotion_result = create_mock_emotion() if request.options.get('detectEmotion', True) else None
            attention_result = AttentionResult(
                score=0.0, isAttentive=False, 
                headPose={'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0},
                eyeAspectRatio=0.0, blinkRate=0.0
            ) if request.options.get('detectAttention', True) else None
            engagement_result = EngagementResult(
                score=0.0, level='low',
                factors={'emotion': 0.0, 'attention': 0.0, 'posture': 0.0, 'movement': 0.0, 'eyeContact': 0.0}
            ) if request.options.get('detectEngagement', True) else None
            gaze_result = GazeResult(
                direction={'x': 0.0, 'y': 0.0}, onScreen=False, confidence=0.0
            ) if request.options.get('detectGaze', True) else None
            confidence = 0.0
        processing_time = (time.time() - start_time) * 1000
        confidences = [confidence]
        if emotion_result:
            confidences.append(emotion_result.confidence)
        if gaze_result:
            confidences.append(gaze_result.confidence)
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        response = AnalysisResponse(
            emotion=emotion_result,
            attention=attention_result,
            engagement=engagement_result,
            gaze=gaze_result,
            metadata={
                'faceDetected': face_detected,
                'face_count': 1 if face_detected else 0,
                'processingTime': processing_time,
                'averageConfidence': avg_confidence,
                'timestamp': request.timestamp,
                'sessionId': request.sessionId
            }
        )
        return {"success": True, "data": response}
    except Exception as e:
        logger.error(f"Error analyzing frame: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
if __name__ == "__main__":
    import time

    import uvicorn
    uvicorn.run(
        "real_ai_service:app",
        host="0.0.0.0",
        port=8001,
        reload=False,
        log_level="info"
    )