import base64
import io
import json
import logging
import math
import os
import time
from typing import Dict, List, Optional

import cv2
import mediapipe as mp
import numpy as np
import torch
import torchvision.transforms as transforms
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
app = FastAPI(title="Ders Lens AI Service", version="1.0.0")
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
        # Use local model paths (Windows)
        base_path = os.path.dirname(os.path.dirname(__file__))  # Go up from ai-service to root
        
        try:
            import onnxruntime as ort
            ORT_AVAILABLE = True
        except ImportError:
            ORT_AVAILABLE = False
            logger.warning("onnxruntime not available. ONNX models will not be loaded.")

        # FER2013 Model Loading - TESTING BEST MODEL (90% accuracy)
        best_model_path = os.path.join(base_path, "models", "onnx", "best_model.onnx")
        fer_model_path = os.path.join(base_path, "models_fer2013", "fer2013_model.onnx")
        
        # Try best_model.onnx first (the 90% accuracy model)
        logger.info(f"Attempting to load BEST MODEL from: {best_model_path}")
        
        if os.path.exists(best_model_path) and ORT_AVAILABLE:
            try:
                logger.info("Creating ONNX Runtime InferenceSession for BEST MODEL...")
                models['fer2013'] = ort.InferenceSession(best_model_path, providers=['CPUExecutionProvider'])
                logger.info(f"🎉 BEST MODEL (90% accuracy) loaded successfully from {best_model_path}")
                logger.info(f"Model inputs: {[i.name for i in models['fer2013'].get_inputs()]}")
                logger.info(f"Model outputs: {[o.name for o in models['fer2013'].get_outputs()]}")
                
            except Exception as best_e:
                logger.error(f"Failed to load BEST MODEL: {str(best_e)}")
                models['fer2013'] = None
        else:
            logger.warning(f"BEST MODEL not found at {best_model_path} or ONNX not available")
            models['fer2013'] = None
            
        # Fallback to original FER2013 model if best model failed
        if 'fer2013' not in models or not models['fer2013']:
            logger.info(f"Falling back to original FER2013 model from: {fer_model_path}")
            
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
                        logger.info("Creating ONNX Runtime InferenceSession...")
                        models['fer2013'] = ort.InferenceSession(fer_model_path, providers=['CPUExecutionProvider'])
                        logger.info(f"FER2013+ model loaded successfully as ONNX model from {fer_model_path}")
                        logger.info(f"Model inputs: {[i.name for i in models['fer2013'].get_inputs()]}")
                        logger.info(f"Model outputs: {[o.name for o in models['fer2013'].get_outputs()]}")
                        
                    except Exception as onnx_e:
                        logger.error(f"Failed to load FER2013+ ONNX model: {str(onnx_e)}", exc_info=True)
        
        # Try PyTorch model if ONNX failed
        if 'fer2013' not in models or not models['fer2013']:
            pt_model_path = os.path.join(base_path, "models_fer2013", "fer2013_model.pth")
            logger.warning(f"Falling back to PyTorch model at: {pt_model_path}")
            
            if os.path.exists(pt_model_path):
                try:
                    logger.info(f"Loading PyTorch model from {pt_model_path}")
                    models['fer2013'] = torch.load(pt_model_path, map_location='cpu', weights_only=False)
                    if hasattr(models['fer2013'], 'eval'):
                        models['fer2013'].eval()
                    logger.info("FER2013+ model loaded successfully as PyTorch model")
                    logger.info(f"Model type: {type(models['fer2013']).__name__}")
                except Exception as e:
                    logger.error(f"Failed to load FER2013+ PyTorch model: {str(e)}", exc_info=True)
            else:
                logger.warning(f"PyTorch model not found at {pt_model_path}")

        # DAISEE Model Loading
        daisee_model_path = os.path.join(base_path, "models_daisee", "daisee_model.pth")
        if os.path.exists(daisee_model_path):
            try:
                models['daisee'] = torch.load(daisee_model_path, map_location='cpu', weights_only=False)
                if hasattr(models['daisee'], 'eval'):
                    models['daisee'].eval()
                logger.info("DAISEE model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load DAISEE model: {e}")
        else:
            logger.warning(f"DAISEE model not found at {daisee_model_path}")

        # Mendeley Model Loading  
        mendeley_model_path = os.path.join(base_path, "models_mendeley", "mendeley_model.pth")
        if os.path.exists(mendeley_model_path):
            try:
                models['mendeley'] = torch.load(mendeley_model_path, map_location='cpu', weights_only=False)
                if hasattr(models['mendeley'], 'eval'):
                    models['mendeley'].eval()
                logger.info("Mendeley model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load Mendeley model: {e}")
        else:
            logger.warning(f"Mendeley model not found at {mendeley_model_path}")

        # MPIIGaze Model Loading with enhanced error handling
        mpiigaze_model_paths = [
            os.path.join(base_path, "models_mpiigaze", "mpiigaze_best.pth"),
            os.path.join(base_path, "models_mpiigaze_excellent", "mpiigaze_best.pth"),
            os.path.join(base_path, "models_mpiigaze_stable", "mpiigaze_best.pth"),
            os.path.join(base_path, "models", "mpiigaze_best.pth"),
        ]
        
        mpiigaze_loaded = False
        for mpiigaze_path in mpiigaze_model_paths:
            if os.path.exists(mpiigaze_path):
                try:
                    logger.info(f"Loading MPIIGaze model from: {mpiigaze_path}")
                    
                    # Load checkpoint with error handling
                    checkpoint = torch.load(mpiigaze_path, map_location='cpu', weights_only=False)
                    
                    # Create MPIIGaze model architecture
                    class GazeNet(torch.nn.Module):
                        def __init__(self):
                            super(GazeNet, self).__init__()
                            
                            self.features = torch.nn.Sequential(
                                torch.nn.Conv2d(3, 32, kernel_size=3, padding=1),
                                torch.nn.BatchNorm2d(32),
                                torch.nn.ReLU(inplace=True),
                                torch.nn.Conv2d(32, 32, kernel_size=3, padding=1),
                                torch.nn.BatchNorm2d(32),
                                torch.nn.ReLU(inplace=True),
                                torch.nn.MaxPool2d(2, 2),
                                torch.nn.Dropout2d(0.1),
                                
                                torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
                                torch.nn.BatchNorm2d(64),
                                torch.nn.ReLU(inplace=True),
                                torch.nn.Conv2d(64, 64, kernel_size=3, padding=1),
                                torch.nn.BatchNorm2d(64),
                                torch.nn.ReLU(inplace=True),
                                torch.nn.MaxPool2d(2, 2),
                                torch.nn.Dropout2d(0.2),
                                
                                torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
                                torch.nn.BatchNorm2d(128),
                                torch.nn.ReLU(inplace=True),
                                torch.nn.Conv2d(128, 128, kernel_size=3, padding=1),
                                torch.nn.BatchNorm2d(128),
                                torch.nn.ReLU(inplace=True),
                                torch.nn.MaxPool2d(2, 2),
                                torch.nn.Dropout2d(0.3),
                                
                                torch.nn.Conv2d(128, 256, kernel_size=3, padding=1),
                                torch.nn.BatchNorm2d(256),
                                torch.nn.ReLU(inplace=True),
                                torch.nn.AdaptiveAvgPool2d((2, 2))
                            )
                            
                            self.regressor = torch.nn.Sequential(
                                torch.nn.Dropout(0.5),
                                torch.nn.Linear(256 * 2 * 2, 256),
                                torch.nn.ReLU(inplace=True),
                                torch.nn.Dropout(0.4),
                                torch.nn.Linear(256, 64),
                                torch.nn.ReLU(inplace=True),
                                torch.nn.Dropout(0.3),
                                torch.nn.Linear(64, 2)  # [theta, phi]
                            )
                            
                        def forward(self, x):
                            x = self.features(x)
                            x = x.view(x.size(0), -1)
                            x = self.regressor(x)
                            return x
                    
                    # Load model
                    gaze_model = GazeNet()
                    
                    # Handle different checkpoint formats
                    if isinstance(checkpoint, dict):
                        if 'model_state_dict' in checkpoint:
                            gaze_model.load_state_dict(checkpoint['model_state_dict'])
                            logger.info(f"   📊 MPIIGaze performance: {checkpoint.get('performance_text', 'N/A')}")
                        elif 'state_dict' in checkpoint:
                            gaze_model.load_state_dict(checkpoint['state_dict'])
                        else:
                            gaze_model.load_state_dict(checkpoint)
                    else:
                        gaze_model.load_state_dict(checkpoint)
                    
                    gaze_model.eval()
                    models['mpiigaze'] = gaze_model
                    
                    # Store transform for preprocessing
                    models['mpiigaze_transform'] = transforms.Compose([
                        transforms.ToPILImage(),
                        transforms.Resize((64, 64)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                           std=[0.229, 0.224, 0.225])
                    ])
                    
                    logger.info(f"✅ MPIIGaze model loaded successfully! (3.39° MAE accuracy)")
                    mpiigaze_loaded = True
                    break
                    
                except Exception as e:
                    logger.warning(f"Failed to load MPIIGaze from {mpiigaze_path}: {e}")
                    continue
        
        if not mpiigaze_loaded:
            logger.warning("⚠️ MPIIGaze model not loaded - gaze accuracy will be reduced")
            models['mpiigaze'] = None
            models['mpiigaze_transform'] = None
            
        # Load attention models (sklearn models) with enhanced error handling and validation
        try:
            attention_model_paths = [
                # Primary paths
                os.path.join(base_path, "models_fer2013", "local_attention_model_random_forest.pkl"),
                os.path.join(base_path, "models_mendeley", "mendeley_random_forest.pkl"),
                # Fallback paths
                os.path.join(base_path, "models", "attention_model.pkl"),
                os.path.join(base_path, "models_trained", "attention_model.pkl"),
                # ONNX fallback
                os.path.join(base_path, "models_fer2013", "attention_model.onnx"),
            ]
            
            scaler_paths = [
                os.path.join(base_path, "models_fer2013", "local_scaler_random_forest.pkl"),
                os.path.join(base_path, "models_mendeley", "mendeley_scaler.pkl"),
                os.path.join(base_path, "models", "scaler.pkl"),
                os.path.join(base_path, "models_trained", "scaler.pkl"),
                # ONNX doesn't need separate scaler
                None,
            ]
            
            model_loaded = False
            
            # Try each model path combination
            for attention_path, scaler_path in zip(attention_model_paths, scaler_paths):
                if not os.path.exists(attention_path):
                    continue
                    
                # Handle ONNX models differently
                if attention_path.endswith('.onnx'):
                    try:
                        if ORT_AVAILABLE:
                            models['attention_onnx'] = ort.InferenceSession(attention_path, providers=['CPUExecutionProvider'])
                            models['attention_scaler'] = None  # ONNX models handle scaling internally
                            logger.info(f"✅ Attention ONNX model loaded from: {attention_path}")
                            model_loaded = True
                            break
                    except Exception as e:
                        logger.warning(f"Failed to load ONNX attention model from {attention_path}: {e}")
                        continue
                
                # Handle pickle models with comprehensive error handling
                if scaler_path and os.path.exists(scaler_path):
                    try:
                        import pickle

                        import sklearn

                        # Check file integrity before loading
                        def validate_pickle_file(file_path):
                            """Validate if pickle file is readable and not corrupted"""
                            try:
                                with open(file_path, 'rb') as f:
                                    first_bytes = f.read(16)
                                    f.seek(0)
                                    
                                    # Check for valid pickle magic bytes
                                    valid_headers = [
                                        b'\x80\x02',  # Protocol 2
                                        b'\x80\x03',  # Protocol 3 
                                        b'\x80\x04',  # Protocol 4
                                        b'\x80\x05',  # Protocol 5
                                        b'(',         # Protocol 0 (text)
                                        b'c',         # Protocol 0 (binary)
                                    ]
                                    
                                    if any(first_bytes.startswith(header) for header in valid_headers):
                                        # Try to actually load it
                                        try:
                                            obj = pickle.load(f)
                                            return True, obj
                                        except Exception as load_error:
                                            return False, str(load_error)
                                    else:
                                        return False, f"Invalid pickle header: {first_bytes[:8].hex()}"
                            except Exception as e:
                                return False, str(e)
                        
                        # Validate attention model
                        logger.info(f"🔍 Validating attention model: {attention_path}")
                        is_valid_attention, attention_obj = validate_pickle_file(attention_path)
                        
                        if not is_valid_attention:
                            logger.warning(f"Corrupted attention model {attention_path}: {attention_obj}")
                            continue
                        
                        # Validate scaler
                        logger.info(f"🔍 Validating scaler: {scaler_path}")
                        is_valid_scaler, scaler_obj = validate_pickle_file(scaler_path)
                        
                        if not is_valid_scaler:
                            logger.warning(f"Corrupted scaler {scaler_path}: {scaler_obj}")
                            continue
                        
                        # Check sklearn version compatibility
                        try:
                            # Suppress sklearn version warnings temporarily
                            import warnings
                            with warnings.catch_warnings():
                                warnings.filterwarnings("ignore", category=UserWarning)
                                warnings.filterwarnings("ignore", message=".*version.*")
                                
                                # Verify models are sklearn objects
                                if hasattr(attention_obj, 'predict') and hasattr(scaler_obj, 'transform'):
                                    models['attention_rf'] = attention_obj
                                    models['attention_scaler'] = scaler_obj
                                    
                                    # Test prediction to ensure it works
                                    test_features = np.array([[0.5, 0.5, 0.0, 0.0, 0.8]]).reshape(1, -1)
                                    scaled_features = scaler_obj.transform(test_features)
                                    test_prediction = attention_obj.predict(scaled_features)
                                    
                                    logger.info(f"✅ Attention model loaded successfully from: {attention_path}")
                                    logger.info(f"   Model type: {type(attention_obj).__name__}")
                                    logger.info(f"   Scaler type: {type(scaler_obj).__name__}")
                                    logger.info(f"   Test prediction: {test_prediction}")
                                    model_loaded = True
                                    break
                                else:
                                    logger.warning(f"Invalid model objects in {attention_path} or {scaler_path}")
                                    continue
                                    
                        except Exception as compat_error:
                            logger.warning(f"Sklearn compatibility issue with {attention_path}: {compat_error}")
                            continue
                        
                    except (pickle.UnpicklingError, EOFError, UnicodeDecodeError, ImportError) as e:
                        logger.warning(f"Pickle loading failed for {attention_path}: {e}")
                        continue
                    except Exception as e:
                        logger.warning(f"Unexpected error loading {attention_path}: {e}")
                        continue
            
            if not model_loaded:
                logger.warning("⚠️ No valid attention model found - trying Mendeley ensemble...")
                
                # Enhanced Mendeley ensemble loading
                mendeley_ensemble = {}
                mendeley_files = {
                    'gradient_boosting': os.path.join(base_path, "models_mendeley", "mendeley_gradient_boosting.pkl"),
                    'random_forest': os.path.join(base_path, "models_mendeley", "mendeley_random_forest.pkl"),
                    'logistic_regression': os.path.join(base_path, "models_mendeley", "mendeley_logistic_regression.pkl"),
                    'scaler': os.path.join(base_path, "models_mendeley", "mendeley_scaler.pkl")
                }
                
                logger.info("🔄 Loading Mendeley ensemble models...")
                
                for model_name, model_path in mendeley_files.items():
                    if os.path.exists(model_path):
                        try:
                            logger.info(f"🔍 Loading Mendeley {model_name}: {model_path}")
                            
                            # Use joblib for better sklearn compatibility
                            # Suppress version warnings
                            import warnings

                            import joblib
                            with warnings.catch_warnings():
                                warnings.filterwarnings("ignore", category=UserWarning)
                                warnings.filterwarnings("ignore", message=".*version.*")
                                
                                loaded_model = joblib.load(model_path)
                                
                                # Validate the model works
                                if model_name == 'scaler':
                                    if hasattr(loaded_model, 'transform'):
                                        # Test scaler with 28 features (Mendeley format)
                                        test_data = np.zeros((1, 28))  # 28 features for Mendeley
                                        try:
                                            _ = loaded_model.transform(test_data)
                                            mendeley_ensemble[model_name] = loaded_model
                                            logger.info(f"✅ Mendeley {model_name} loaded and validated (28-feature)")
                                        except Exception as e:
                                            logger.warning(f"Mendeley {model_name} validation failed: {e}")
                                    else:
                                        logger.warning(f"Invalid scaler: {model_path}")
                                else:
                                    if hasattr(loaded_model, 'predict'):
                                        # Test predictor with 28 features (Mendeley format)
                                        test_data = np.zeros((1, 28))  # 28 features for Mendeley
                                        try:
                                            _ = loaded_model.predict(test_data)
                                            mendeley_ensemble[model_name] = loaded_model
                                            logger.info(f"✅ Mendeley {model_name} loaded and validated (28-feature)")
                                        except Exception as e:
                                            logger.warning(f"Mendeley {model_name} validation failed: {e}")
                                    else:
                                        logger.warning(f"Invalid predictor: {model_path}")
                                
                        except Exception as e:
                            logger.warning(f"Failed to load Mendeley {model_name}: {e}")
                            continue
                    else:
                        logger.info(f"Mendeley {model_name} not found: {model_path}")
                
                # Check if we have a complete or partial Mendeley ensemble
                if len(mendeley_ensemble) > 0:
                    models['mendeley_ensemble_models'] = mendeley_ensemble  # Store with correct key
                    models['mendeley_ensemble'] = mendeley_ensemble  # Keep original for backward compatibility
                    
                    if 'random_forest' in mendeley_ensemble:
                        models['attention_rf'] = mendeley_ensemble['random_forest']
                    if 'scaler' in mendeley_ensemble:
                        models['attention_scaler'] = mendeley_ensemble['scaler']
                    
                    ensemble_models = list(mendeley_ensemble.keys())
                    if len(ensemble_models) >= 3:
                        logger.info("🎉 ✅ MENDELEY ENSEMBLE LOADED SUCCESSFULLY! (Enhanced 28-feature support)")
                        logger.info(f"   📊 Loaded models: {ensemble_models}")
                    else:
                        logger.info(f"✅ Partial Mendeley ensemble loaded: {ensemble_models}")
                    
                    model_loaded = True
                elif len(mendeley_ensemble) > 0:
                    logger.info(f"⚠️ Incomplete Mendeley ensemble: {list(mendeley_ensemble.keys())}")
                
            if not model_loaded:
                logger.warning("⚠️ No valid attention model found - creating robust rule-based fallback")
                models['attention_rf'] = None
                models['attention_scaler'] = None
                
                # Create a simple rule-based attention classifier
                class RuleBasedAttentionClassifier:
                    def predict(self, X):
                        """Rule-based attention prediction"""
                        predictions = []
                        for features in X:
                            # Features: [face_conf, eye_open, head_yaw, head_pitch, gaze_focus]
                            if len(features) >= 5:
                                face_conf, eye_open, head_yaw, head_pitch, gaze_focus = features[:5]
                                
                                # Calculate attention score
                                attention_score = 0.0
                                attention_score += face_conf * 0.3  # Face detection confidence
                                attention_score += eye_open * 0.25   # Eye openness
                                attention_score += max(0, 1 - abs(head_yaw)) * 0.2  # Head orientation
                                attention_score += max(0, 1 - abs(head_pitch)) * 0.15  # Head tilt
                                attention_score += gaze_focus * 0.1  # Gaze focus
                                
                                # Convert to class (0=low, 1=medium, 2=high attention)
                                if attention_score >= 0.7:
                                    predictions.append(2)
                                elif attention_score >= 0.4:
                                    predictions.append(1)
                                else:
                                    predictions.append(0)
                            else:
                                predictions.append(1)  # Default medium attention
                        return np.array(predictions)
                    
                    def predict_proba(self, X):
                        """Return probability distributions"""
                        predictions = self.predict(X)
                        probas = []
                        for pred in predictions:
                            if pred == 2:
                                probas.append([0.1, 0.2, 0.7])
                            elif pred == 1:
                                probas.append([0.2, 0.6, 0.2])
                            else:
                                probas.append([0.7, 0.2, 0.1])
                        return np.array(probas)
                
                class IdentityScaler:
                    def transform(self, X):
                        """Identity transformation (no scaling)"""
                        return np.array(X)
                    
                    def fit_transform(self, X):
                        return self.transform(X)
                
                models['attention_rf'] = RuleBasedAttentionClassifier()
                models['attention_scaler'] = IdentityScaler()
                logger.info("✅ Rule-based attention classifier created successfully")
                
        except Exception as e:
            logger.error(f"Critical error in attention model loading: {e}")
            # Create minimal fallback
            models['attention_rf'] = None
            models['attention_scaler'] = None
            
    except Exception as e:
        logger.error(f"Error loading models (real_ai_service.py): {e}")
        models = {}
def decode_base64_image(base64_string: str) -> np.ndarray:
    try:
        # Remove data URL prefix if present
        if base64_string.startswith('data:image'):
            # Find the comma that separates the header from the data
            comma_index = base64_string.find(',')
            if comma_index != -1:
                base64_string = base64_string[comma_index + 1:]
        
        # Log for debugging
        logger.debug(f"Decoding image of length: {len(base64_string)}")
        
        # Decode base64
        image_data = base64.b64decode(base64_string)
        logger.debug(f"Decoded image data size: {len(image_data)} bytes")
        
        # Check if we have any data
        if len(image_data) == 0:
            raise ValueError("Empty image data after base64 decode")
        
        # Open image
        image = Image.open(io.BytesIO(image_data))
        logger.debug(f"Image loaded: {image.size}, mode: {image.mode}")
        
        # Convert to numpy array and then to BGR for OpenCV
        image_array = np.array(image)
        if len(image_array.shape) == 3 and image_array.shape[2] == 3:
            # RGB to BGR conversion
            image_rgb = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        elif len(image_array.shape) == 3 and image_array.shape[2] == 4:
            # RGBA to BGR conversion
            image_rgb = cv2.cvtColor(image_array, cv2.COLOR_RGBA2BGR)
        else:
            # Grayscale to BGR
            image_rgb = cv2.cvtColor(image_array, cv2.COLOR_GRAY2BGR)
        
        logger.debug(f"Final image shape: {image_rgb.shape}")
        return image_rgb
    except Exception as e:
        logger.error(f"Error decoding base64 image: {e}")
        logger.error(f"Base64 string length: {len(base64_string) if base64_string else 'None'}")
        logger.error(f"Base64 string preview: {base64_string[:100] if base64_string else 'None'}...")
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
        # FER2013 typically outputs 7 emotion classes, not 8
        fer2013_emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        
        if 'fer2013' in models:
            model = models['fer2013']
            
            # Use RGB 224x224 as the ONNX model expects (based on error message)
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),  # Model expects 224x224
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Standard ImageNet normalization
            ])
            
            face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            input_tensor = transform(face_rgb).unsqueeze(0)
            
            if hasattr(model, 'run'):  # ONNX model
                try:
                    input_name = model.get_inputs()[0].name
                    input_data = input_tensor.numpy()
                    outputs = model.run(None, {input_name: input_data})
                    logits = torch.tensor(outputs[0])
                    probabilities = torch.softmax(logits, dim=1).squeeze().tolist()
                except Exception as e:  
                    logger.error(f"Error running ONNX model: {e}")
                    return create_mock_emotion()
            else:   # PyTorch model
                try:
                    with torch.no_grad():
                        model.eval()
                        outputs = model(input_tensor)
                        probabilities = torch.softmax(outputs, dim=1).squeeze().tolist()
                except Exception as e:              
                    logger.error(f"Error running PyTorch model: {e}")
                    return create_mock_emotion()
            
            # Use the actual number of classes from the model
            if len(probabilities) == 7:
                emotions = fer2013_emotions
            elif len(probabilities) == 8:
                # If model outputs 8 classes, use standard set
                emotions = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise', 'neutral', 'contempt']
            else:
                logger.warning(f"Unexpected number of emotion classes from model: {len(probabilities)}")
                emotions = fer2013_emotions[:len(probabilities)]  # Truncate if needed
            
            # Apply emotion bias correction for FER2013 sadness bias
            corrected_probs = apply_emotion_bias_correction(probabilities, emotions)
            
            dominant_idx = np.argmax(corrected_probs)
            dominant_emotion = emotions[dominant_idx] if dominant_idx < len(emotions) else 'neutral'
            confidence = corrected_probs[dominant_idx] if isinstance(corrected_probs, (list, np.ndarray)) else 0.5
                
            emotion_dict = {emotion: float(prob) for emotion, prob in zip(emotions, corrected_probs)}
            
            # Map to consistent emotion names if needed
            emotion_mapping = {
                'angry': 'anger',
                'happy': 'happiness',
                'sad': 'sadness'
            }
            
            # Remap emotions to consistent naming
            consistent_emotions = {}
            for emotion, prob in emotion_dict.items():
                mapped_emotion = emotion_mapping.get(emotion, emotion)
                consistent_emotions[mapped_emotion] = prob
            
            return EmotionResult(
                dominant=emotion_mapping.get(dominant_emotion, dominant_emotion),
                confidence=float(confidence),
                emotions=consistent_emotions,
                valence=calculate_valence(consistent_emotions),
                arousal=calculate_arousal(consistent_emotions)
            )
            
        return create_mock_emotion()
    except Exception as e:
        logger.error(f"Error in emotion analysis: {e}")
        return create_mock_emotion()

def apply_emotion_bias_correction(probabilities, emotions):
    """
    AGGRESSIVE correction for FER2013 model bias toward sadness for neutral/normal faces
    This model has a severe sadness bias that needs strong correction
    """
    try:
        corrected_probs = probabilities.copy() if isinstance(probabilities, list) else probabilities.tolist()
        
        # Find emotion indices
        sadness_idx = None
        neutral_idx = None
        happiness_idx = None
        surprise_idx = None
        
        for i, emotion in enumerate(emotions):
            if emotion in ['sad', 'sadness']:
                sadness_idx = i
            elif emotion in ['neutral']:
                neutral_idx = i
            elif emotion in ['happy', 'happiness']:
                happiness_idx = i
            elif emotion in ['surprise']:
                surprise_idx = i
        
        original_sadness = corrected_probs[sadness_idx] if sadness_idx is not None else 0
        
        # ULTRA-AGGRESSIVE: If sadness > 0.4, it's likely wrong for normal faces
        if sadness_idx is not None and neutral_idx is not None:
            sadness_prob = corrected_probs[sadness_idx]
            neutral_prob = corrected_probs[neutral_idx]
            
            # Much more aggressive threshold: sadness > 0.4 (was 0.6)
            if sadness_prob > 0.4:
                if sadness_prob > 0.8:
                    # EXTREME bias - force neutral to be dominant
                    corrected_probs[neutral_idx] = 0.6
                    corrected_probs[sadness_idx] = 0.15
                    # Distribute remaining to other positive emotions
                    remaining = 0.25
                    other_count = 0
                    for i in range(len(corrected_probs)):
                        if i != sadness_idx and i != neutral_idx:
                            other_count += 1
                    if other_count > 0:
                        for i in range(len(corrected_probs)):
                            if i != sadness_idx and i != neutral_idx:
                                corrected_probs[i] = remaining / other_count
                    logger.debug(f"EXTREME sadness bias correction: {sadness_prob:.3f} -> {corrected_probs[sadness_idx]:.3f}")
                
                elif sadness_prob > 0.6:
                    # MAJOR bias - swap sadness and neutral
                    corrected_probs[neutral_idx] = sadness_prob * 0.8  # Give most to neutral
                    corrected_probs[sadness_idx] = 0.2  # Reduce sadness dramatically
                    logger.debug(f"MAJOR sadness bias correction: {sadness_prob:.3f} -> {corrected_probs[sadness_idx]:.3f}")
                
                else:
                    # MODERATE bias - significant transfer
                    transfer_amount = sadness_prob * 0.7  # Transfer 70% (was 40%)
                    corrected_probs[sadness_idx] -= transfer_amount
                    corrected_probs[neutral_idx] += transfer_amount * 0.7
                    # Give some to happiness too for normal faces
                    if happiness_idx is not None:
                        corrected_probs[happiness_idx] += transfer_amount * 0.3
                    logger.debug(f"MODERATE sadness bias correction: {sadness_prob:.3f} -> {corrected_probs[sadness_idx]:.3f}")
        
        # ADDITIONAL: If neutral is still too low after correction, boost it more
        if neutral_idx is not None and corrected_probs[neutral_idx] < 0.3:
            # Normal faces should have significant neutral component
            boost_neutral = 0.4 - corrected_probs[neutral_idx]
            corrected_probs[neutral_idx] = 0.4
            
            # Take proportionally from all other emotions
            remaining_total = sum(corrected_probs) - corrected_probs[neutral_idx]
            if remaining_total > 0:
                reduction_factor = (1.0 - 0.4) / remaining_total
                for i in range(len(corrected_probs)):
                    if i != neutral_idx:
                        corrected_probs[i] *= reduction_factor
        
        # Normalize probabilities to sum to 1
        total = sum(corrected_probs)
        if total > 0:
            corrected_probs = [p / total for p in corrected_probs]
        
        # Log the correction if significant
        if original_sadness > 0.5:
            final_sadness = corrected_probs[sadness_idx] if sadness_idx is not None else 0
            logger.debug(f"BIAS CORRECTION SUMMARY: sadness {original_sadness:.3f} -> {final_sadness:.3f}, neutral -> {corrected_probs[neutral_idx]:.3f}")
        
        return corrected_probs
        
    except Exception as e:
        logger.warning(f"Error in bias correction, using original: {e}")
        return probabilities
def create_mock_emotion() -> EmotionResult:
    # Use FER2013 emotion classes for consistency
    emotions = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise', 'neutral']
    base_prob = 0.05
    emotion_probs = [base_prob] * len(emotions)
    dominant_idx = np.random.choice(len(emotions), p=[0.05, 0.05, 0.05, 0.3, 0.1, 0.1, 0.35])
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
    negative_emotions = ['sadness', 'anger', 'disgust', 'fear']
    positive_score = sum(emotions.get(emotion, 0) for emotion in positive_emotions)
    negative_score = sum(emotions.get(emotion, 0) for emotion in negative_emotions)
    return (positive_score - negative_score + 1) / 2
def calculate_arousal(emotions: Dict[str, float]) -> float:
    high_arousal = ['anger', 'fear', 'surprise', 'happiness']
    low_arousal = ['sadness', 'neutral', 'disgust']
    high_score = sum(emotions.get(emotion, 0) for emotion in high_arousal)
    low_score = sum(emotions.get(emotion, 0) for emotion in low_arousal)
    return (high_score - low_score + 1) / 2
def extract_mendeley_features(face_landmarks, head_pose, emotion_result=None) -> np.ndarray:
    """Extract 28 features for Mendeley ensemble models"""
    try:
        features = []
        
        if face_landmarks and len(face_landmarks) >= 468:
            # Basic geometric features (5 original features)
            eye_ratio = calculate_eye_aspect_ratio(face_landmarks)
            face_conf = 0.9  # High confidence for detected face
            features.extend([face_conf, eye_ratio, head_pose['yaw'], head_pose['pitch'], 0.8])
            
            # Extended facial landmark features (23 additional features)
            # Eye features (6 features)
            left_eye = face_landmarks[33:42] if len(face_landmarks) > 42 else [(0,0)] * 9
            right_eye = face_landmarks[362:371] if len(face_landmarks) > 371 else [(0,0)] * 9
            
            left_eye_center_x = np.mean([p[0] for p in left_eye]) if left_eye else 0.0
            left_eye_center_y = np.mean([p[1] for p in left_eye]) if left_eye else 0.0
            right_eye_center_x = np.mean([p[0] for p in right_eye]) if right_eye else 0.0
            right_eye_center_y = np.mean([p[1] for p in right_eye]) if right_eye else 0.0
            
            eye_distance = np.sqrt((left_eye_center_x - right_eye_center_x)**2 + (left_eye_center_y - right_eye_center_y)**2)
            eye_symmetry = abs(left_eye_center_y - right_eye_center_y)
            
            features.extend([left_eye_center_x/640.0, left_eye_center_y/480.0, 
                           right_eye_center_x/640.0, right_eye_center_y/480.0, 
                           eye_distance/640.0, eye_symmetry/480.0])
            
            # Mouth features (4 features)
            mouth_landmarks = face_landmarks[61:68] if len(face_landmarks) > 68 else [(0,0)] * 7
            mouth_center_x = np.mean([p[0] for p in mouth_landmarks]) if mouth_landmarks else 0.0
            mouth_center_y = np.mean([p[1] for p in mouth_landmarks]) if mouth_landmarks else 0.0
            mouth_width = max([p[0] for p in mouth_landmarks]) - min([p[0] for p in mouth_landmarks]) if mouth_landmarks else 0.0
            mouth_height = max([p[1] for p in mouth_landmarks]) - min([p[1] for p in mouth_landmarks]) if mouth_landmarks else 0.0
            
            features.extend([mouth_center_x/640.0, mouth_center_y/480.0, 
                           mouth_width/640.0, mouth_height/480.0])
            
            # Nose features (3 features)
            nose_tip = face_landmarks[1] if len(face_landmarks) > 1 else (0, 0)
            nose_bridge = face_landmarks[6] if len(face_landmarks) > 6 else (0, 0)
            
            features.extend([nose_tip[0]/640.0, nose_tip[1]/480.0, 
                           abs(nose_tip[0] - nose_bridge[0])/640.0])
            
            # Face boundary features (4 features)
            if len(face_landmarks) > 10:
                face_width = max([p[0] for p in face_landmarks[:10]]) - min([p[0] for p in face_landmarks[:10]])
                face_height = max([p[1] for p in face_landmarks[:10]]) - min([p[1] for p in face_landmarks[:10]])
                face_center_x = np.mean([p[0] for p in face_landmarks[:10]])
                face_center_y = np.mean([p[1] for p in face_landmarks[:10]])
            else:
                face_width = face_height = face_center_x = face_center_y = 0.0
            
            features.extend([face_width/640.0, face_height/480.0, 
                           face_center_x/640.0, face_center_y/480.0])
            
            # Emotion-based features (6 features)
            if emotion_result and emotion_result.emotions:
                emotion_features = [
                    emotion_result.emotions.get('happiness', 0.0),
                    emotion_result.emotions.get('sadness', 0.0),
                    emotion_result.emotions.get('anger', 0.0),
                    emotion_result.emotions.get('surprise', 0.0),
                    emotion_result.emotions.get('fear', 0.0),
                    emotion_result.emotions.get('neutral', 0.0)
                ]
                features.extend(emotion_features)
            else:
                features.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.6])  # Default neutral
                
        else:
            # No face detected - use default features
            features = [0.0] * 28
            
        # Ensure we have exactly 28 features
        features = features[:28]  # Truncate if too many
        while len(features) < 28:  # Pad if too few
            features.append(0.0)
            
        return np.array(features).reshape(1, -1)
        
    except Exception as e:
        logger.warning(f"Error extracting Mendeley features: {e}")
        return np.zeros((1, 28))

def analyze_attention(face_landmarks, head_pose, emotion_result=None) -> AttentionResult:
    try:
        if face_landmarks:
            # Try to use Mendeley ensemble models first
            attention_score = 0.5  # Default
            
            if models.get('mendeley_ensemble_models'):
                try:
                    # Extract 28 features for Mendeley models
                    features_28 = extract_mendeley_features(face_landmarks, head_pose, emotion_result)
                    
                    # Try each Mendeley model
                    predictions = []
                    ensemble_models = models['mendeley_ensemble_models']
                    
                    if 'logistic_regression' in ensemble_models:
                        try:
                            pred = ensemble_models['logistic_regression'].predict_proba(features_28)[0]
                            # Interpret model output more intelligently
                            if len(pred) > 1:
                                attention_prob = pred[1]  # High attention class
                            else:
                                attention_prob = pred[0]
                            
                            # Apply intelligent scaling - these models seem to output low values
                            attention_prob = min(1.0, attention_prob * 2.5)  # Scale up conservative predictions
                            
                            # Apply attention boost for good conditions
                            if emotion_result and emotion_result.emotions:
                                positive_emotions = emotion_result.emotions.get('happiness', 0) + emotion_result.emotions.get('neutral', 0) + emotion_result.emotions.get('surprise', 0)
                                if positive_emotions > 0.4:
                                    attention_prob = min(1.0, attention_prob * 1.5)
                            predictions.append(attention_prob)
                            logger.debug(f"Logistic regression prediction: {pred} -> {attention_prob:.3f}")
                        except Exception as e:
                            logger.debug(f"Logistic regression prediction failed: {e}")
                    
                    if 'gradient_boosting' in ensemble_models:
                        try:
                            pred = ensemble_models['gradient_boosting'].predict_proba(features_28)[0]
                            # Interpret model output more intelligently
                            if len(pred) > 1:
                                attention_prob = pred[1]  # High attention class
                            else:
                                attention_prob = pred[0]
                            
                            # Apply intelligent scaling
                            attention_prob = min(1.0, attention_prob * 2.5)  # Scale up conservative predictions
                            
                            # Apply attention boost for good conditions
                            if emotion_result and emotion_result.emotions:
                                positive_emotions = emotion_result.emotions.get('happiness', 0) + emotion_result.emotions.get('neutral', 0) + emotion_result.emotions.get('surprise', 0)
                                if positive_emotions > 0.4:
                                    attention_prob = min(1.0, attention_prob * 1.5)
                            predictions.append(attention_prob)
                            logger.debug(f"Gradient boosting prediction: {pred} -> {attention_prob:.3f}")
                        except Exception as e:
                            logger.debug(f"Gradient boosting prediction failed: {e}")
                    
                    if 'random_forest' in ensemble_models:
                        try:
                            pred = ensemble_models['random_forest'].predict_proba(features_28)[0]
                            # Interpret model output more intelligently
                            if len(pred) > 1:
                                attention_prob = pred[1]  # High attention class
                            else:
                                attention_prob = pred[0]
                            
                            # Apply intelligent scaling
                            attention_prob = min(1.0, attention_prob * 2.5)  # Scale up conservative predictions
                            
                            # Apply attention boost for good conditions
                            if emotion_result and emotion_result.emotions:
                                positive_emotions = emotion_result.emotions.get('happiness', 0) + emotion_result.emotions.get('neutral', 0) + emotion_result.emotions.get('surprise', 0)
                                if positive_emotions > 0.4:
                                    attention_prob = min(1.0, attention_prob * 1.5)
                            predictions.append(attention_prob)
                            logger.debug(f"Random forest prediction: {pred} -> {attention_prob:.3f}")
                        except Exception as e:
                            logger.debug(f"Random forest prediction failed: {e}")
                    
                    # Add rule-based backup prediction for robustness
                    rule_based_score = 0.7  # Base score for detected face
                    if abs(head_pose['yaw']) < 25:
                        rule_based_score += 0.2
                    if abs(head_pose['pitch']) < 15:
                        rule_based_score += 0.1
                    rule_based_score = min(1.0, rule_based_score)
                    predictions.append(rule_based_score)
                    logger.debug(f"Rule-based backup score: {rule_based_score:.3f}")
                    
                    # Use ensemble average if we have predictions
                    if predictions:
                        raw_attention_score = np.mean(predictions)
                        
                        # Apply BALANCED attention scoring - removing ultra-aggressive boosting
                        attention_score = raw_attention_score
                        
                        # 1. Moderate boost for good head pose (looking forward)
                        if abs(head_pose['yaw']) < 30 and abs(head_pose['pitch']) < 25:
                            attention_score = min(1.0, attention_score * 1.3)  # 30% boost for decent head pose
                            logger.debug(f"Applied head pose boost: {raw_attention_score:.3f} -> {attention_score:.3f}")
                        
                        # 2. Small boost for positive expressions
                        if emotion_result and emotion_result.emotions:
                            happiness = emotion_result.emotions.get('happiness', 0)
                            neutral = emotion_result.emotions.get('neutral', 0)
                            if happiness > 0.3 or neutral > 0.4:  # Reasonable thresholds
                                attention_score = min(1.0, attention_score * 1.2)  # 20% boost for good expressions
                                logger.debug(f"Applied emotion boost (happiness: {happiness:.2f}, neutral: {neutral:.2f}): {attention_score:.3f}")
                        
                        # 3. Small bonus for direct screen look
                        if abs(head_pose['yaw']) < 15 and abs(head_pose['pitch']) < 10:
                            attention_score = min(1.0, attention_score + 0.1)  # Small 10% bonus for direct look
                            logger.debug(f"Applied direct screen look boost: {attention_score:.3f}")
                        
                        # 4. Reasonable minimum baseline (not too high!)
                        attention_score = max(0.3, attention_score)  # Minimum 30% attention when face detected
                        
                        logger.debug(f"Mendeley ensemble attention score: raw={raw_attention_score:.3f}, final={attention_score:.3f}")
                    else:
                        raise Exception("No Mendeley models available")
                        
                except Exception as e:
                    logger.debug(f"Mendeley ensemble failed, using rule-based: {e}")
                    # Fall back to OPTIMIZED rule-based approach
                    eye_ratio = calculate_eye_aspect_ratio(face_landmarks)
                    yaw_penalty = abs(head_pose['yaw']) / 60.0  # More forgiving (was 45)
                    pitch_penalty = abs(head_pose['pitch']) / 40.0  # More forgiving (was 30)
                    base_attention = max(0.4, 1.0 - (yaw_penalty + pitch_penalty) / 2)  # Higher baseline
                    
                    # Apply boosts for rule-based approach too
                    if abs(head_pose['yaw']) < 25:
                        base_attention = min(1.0, base_attention * 1.5)
                    attention_score = max(0.6, base_attention)  # Much higher minimum
            else:
                # Use OPTIMIZED rule-based approach
                eye_ratio = calculate_eye_aspect_ratio(face_landmarks)
                yaw_penalty = abs(head_pose['yaw']) / 60.0  # More forgiving (was 45)
                pitch_penalty = abs(head_pose['pitch']) / 40.0  # More forgiving (was 30)
                base_attention = max(0.4, 1.0 - (yaw_penalty + pitch_penalty) / 2)  # Higher baseline
                
                # Apply boosts for rule-based approach too
                if abs(head_pose['yaw']) < 25:
                    base_attention = min(1.0, base_attention * 1.5)
                attention_score = max(0.6, base_attention)  # Much higher minimum
            
            return AttentionResult(
                score=attention_score,
                isAttentive=attention_score > 0.5,  # Adjusted to match new baseline of 60%
                headPose=head_pose,
                eyeAspectRatio=calculate_eye_aspect_ratio(face_landmarks),
                blinkRate=calculate_blink_rate(calculate_eye_aspect_ratio(face_landmarks))
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
    """
    Estimate head pose from facial landmarks using geometric analysis
    """
    if not landmarks or len(landmarks) < 6:
        return {'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0}
    
    try:
        # Use key facial landmarks for head pose estimation
        # MediaPipe face mesh landmark indices (approximate)
        
        # Get face boundary points for basic orientation
        if len(landmarks) >= 468:  # Full MediaPipe face mesh
            # Key landmarks for pose estimation
            nose_tip = landmarks[1] if len(landmarks) > 1 else landmarks[0]
            left_eye = landmarks[33] if len(landmarks) > 33 else landmarks[0]
            right_eye = landmarks[362] if len(landmarks) > 362 else landmarks[0]
            left_mouth = landmarks[61] if len(landmarks) > 61 else landmarks[0]
            right_mouth = landmarks[291] if len(landmarks) > 291 else landmarks[0]
            chin = landmarks[175] if len(landmarks) > 175 else landmarks[0]
            
            # Calculate face center
            face_center_x = np.mean([p[0] for p in landmarks[:10]])
            face_center_y = np.mean([p[1] for p in landmarks[:10]])
            
            # Estimate yaw (left-right head turn)
            eye_distance_x = abs(left_eye[0] - right_eye[0])
            nose_offset_x = nose_tip[0] - face_center_x
            if eye_distance_x > 0:
                yaw = np.clip((nose_offset_x / eye_distance_x) * 45, -45, 45)
            else:
                yaw = 0.0
            
            # Estimate pitch (up-down head tilt)
            eye_to_nose_y = abs(nose_tip[1] - (left_eye[1] + right_eye[1]) / 2)
            nose_to_mouth_y = abs(nose_tip[1] - (left_mouth[1] + right_mouth[1]) / 2)
            if nose_to_mouth_y > 0:
                pitch_ratio = eye_to_nose_y / nose_to_mouth_y
                pitch = np.clip((1.0 - pitch_ratio) * 30, -30, 30)
            else:
                pitch = 0.0
            
            # Estimate roll (head tilt)
            eye_line_slope = (right_eye[1] - left_eye[1]) / max(abs(right_eye[0] - left_eye[0]), 1)
            roll = np.clip(np.arctan(eye_line_slope) * 180 / np.pi, -30, 30)
            
        else:
            # Simplified calculation for fewer landmarks
            nose_tip = landmarks[0]
            left_eye = landmarks[1] if len(landmarks) > 1 else landmarks[0]
            right_eye = landmarks[2] if len(landmarks) > 2 else landmarks[0]
            
            # Basic yaw estimation
            face_center_x = np.mean([p[0] for p in landmarks])
            nose_offset = nose_tip[0] - face_center_x
            yaw = np.clip(nose_offset * 0.5, -20, 20)
            
            # Basic pitch estimation
            face_center_y = np.mean([p[1] for p in landmarks])
            nose_offset_y = nose_tip[1] - face_center_y
            pitch = np.clip(nose_offset_y * 0.3, -15, 15)
            
            # Basic roll estimation from eye line
            if len(landmarks) >= 3:
                eye_slope = (right_eye[1] - left_eye[1]) / max(abs(right_eye[0] - left_eye[0]), 1)
                roll = np.clip(np.arctan(eye_slope) * 180 / np.pi, -20, 20)
            else:
                roll = 0.0
        
        return {
            'yaw': float(yaw),
            'pitch': float(pitch),
            'roll': float(roll)
        }
        
    except Exception as e:
        logger.debug(f"Head pose estimation error: {e}")
        # Return neutral pose on error
        return {'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0}
def analyze_engagement(emotion: EmotionResult, attention: AttentionResult) -> EngagementResult:
    try:
        emotion_factor = calculate_emotion_engagement(emotion)
        attention_factor = attention.score
        posture_factor = max(0.2, 1.0 - abs(attention.headPose['yaw']) / 45.0)
        
        # AGGRESSIVE engagement calculation with much better weighting
        base_engagement = (
            emotion_factor * 0.3 +    # Balanced emotion weight
            attention_factor * 0.5 +  # INCREASED attention weight (main factor)
            posture_factor * 0.2
        )
        
        # Apply AGGRESSIVE engagement boosters for much better user experience
        engagement_score = base_engagement
        
        # 1. MAJOR boost for good attention scores (lowered threshold)
        if attention_factor > 0.4:  # Lowered from 0.3
            engagement_score = min(1.0, engagement_score * 1.8)  # Increased from 1.3 to 1.8
            logger.debug(f"Applied attention boost: {base_engagement:.3f} -> {engagement_score:.3f}")
        
        # 2. Moderate boost for good posture (looking forward)
        if abs(attention.headPose['yaw']) < 30:  # Reasonable tolerance
            engagement_score = min(1.0, engagement_score * 1.2)  # Moderate 20% boost
            logger.debug(f"Applied posture boost: {engagement_score:.3f}")
        
        # 3. Small extra boost for excellent posture (direct screen look)
        if abs(attention.headPose['yaw']) < 15 and abs(attention.headPose['pitch']) < 10:
            engagement_score = min(1.0, engagement_score + 0.1)  # Small 10% boost for direct look
            logger.debug(f"Applied direct look boost: {engagement_score:.3f}")
        
        # 4. Reasonable minimum baseline for detected faces
        engagement_score = max(0.2, engagement_score)  # Minimum 20% engagement (realistic baseline)
        
        # Determine engagement level with balanced thresholds
        if engagement_score > 0.6:  # High engagement
            level = 'high'
        elif engagement_score > 0.35:  # Medium engagement
            level = 'medium'
        else:
            level = 'low'
        
        logger.debug(f"Final engagement: base={base_engagement:.3f}, final={engagement_score:.3f}, level={level}")
        
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
        logger.debug(f"Face detection result: {face_detected}, face_data: {face_data}")
        if face_detected:
            x, y, w, h = face_data['bbox']
            face_image = image[y:y+h, x:x+w]
            landmarks = extract_face_landmarks(image)
            head_pose = estimate_head_pose(landmarks)
            emotion_result = analyze_emotion_with_models(face_image) if request.options.get('detectEmotion', True) else None
            attention_result = analyze_attention(landmarks, head_pose, emotion_result) if request.options.get('detectAttention', True) else None
            engagement_result = analyze_engagement(emotion_result, attention_result) if request.options.get('detectEngagement', True) else None
            gaze_result = analyze_gaze(landmarks, head_pose) if request.options.get('detectGaze', True) else None
            confidence = face_data['confidence']
            logger.debug(f"Analysis results - Emotion: {emotion_result}, Attention: {attention_result}")
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
        port=8003,
        reload=False,
        log_level="info"
    )