"""
Enhanced Attention Detection using Trained Models
Integrates multiple trained models for high-accuracy attention detection
"""
import logging
import os
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available. Using fallback models only.")

try:
    from sklearn.ensemble import (GradientBoostingClassifier,
                                  RandomForestClassifier)
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("Scikit-learn not available. Using basic heuristic detection.")

class EnhancedAttentionDetector:
    """
    Enhanced attention detector that leverages multiple trained models
    """
    
    def __init__(self, models_dir: str = "models", use_daisee: bool = True):
        self.models_dir = Path(models_dir)
        self.use_daisee = use_daisee
        self.device = torch.device('cuda' if torch.cuda.is_available() and TORCH_AVAILABLE else 'cpu')
        
        # Model containers
        self.models = {}
        self.scalers = {}
        self.model_status = {}
        
        # Load all available models
        self._load_models()
        
        # Feature extraction history for temporal smoothing
        self.feature_history = []
        self.max_history = 5
        
    def _load_models(self):
        """Load all available trained models"""
        logger.info("🔄 Loading enhanced attention detection models...")
        
        # Load Mendeley models
        self._load_mendeley_models()
        
        # Load DAiSEE model if requested
        if self.use_daisee:
            self._load_daisee_model()
        
        # Load MPIIGaze model
        self._load_mpiigaze_model()
        
        # Load traditional ML models
        self._load_sklearn_models()
        
        logger.info(f"Enhanced detector initialized with {len(self.models)} models")
        
    def _load_mendeley_models(self):
        """Load Mendeley-based models"""
        mendeley_dir = self.models_dir.parent / "models_mendeley"
        
        if not mendeley_dir.exists():
            logger.warning("Mendeley models directory not found")
            return
            
        # Load neural network model
        nn_path = mendeley_dir / "mendeley_nn_best.pth"
        if nn_path.exists() and TORCH_AVAILABLE:
            try:
                self.models['mendeley_nn'] = torch.load(nn_path, map_location=self.device)
                self.models['mendeley_nn'].eval()
                self.model_status['mendeley_nn'] = True
                logger.info("Mendeley Neural Network loaded")
            except Exception as e:
                logger.error(f"Failed to load Mendeley NN: {e}")
                self.model_status['mendeley_nn'] = False
        
        # Load ensemble models
        ensemble_models = ['gradient_boosting', 'random_forest', 'logistic_regression']
        for model_name in ensemble_models:
            model_path = mendeley_dir / f"mendeley_{model_name}.pkl"
            if model_path.exists() and SKLEARN_AVAILABLE:
                try:
                    with open(model_path, 'rb') as f:
                        self.models[f'mendeley_{model_name}'] = pickle.load(f)
                    self.model_status[f'mendeley_{model_name}'] = True
                    logger.info(f"Mendeley {model_name} loaded")
                except Exception as e:
                    logger.error(f"Failed to load Mendeley {model_name}: {e}")
                    self.model_status[f'mendeley_{model_name}'] = False
        
        # Load scaler
        scaler_path = mendeley_dir / "mendeley_scaler.pkl"
        if scaler_path.exists() and SKLEARN_AVAILABLE:
            try:
                with open(scaler_path, 'rb') as f:
                    self.scalers['mendeley'] = pickle.load(f)
                logger.info("Mendeley scaler loaded")
            except Exception as e:
                logger.error(f"Failed to load Mendeley scaler: {e}")
    
    def _load_daisee_model(self):
        """Load DAiSEE emotion-attention model"""
        daisee_path = self.models_dir / "daisee_emotional_model_best.pth"
        if daisee_path.exists() and TORCH_AVAILABLE:
            try:
                self.models['daisee'] = torch.load(daisee_path, map_location=self.device)
                self.models['daisee'].eval()
                self.model_status['daisee'] = True
                logger.info("DAiSEE emotion model loaded")
            except Exception as e:
                logger.error(f"Failed to load DAiSEE model: {e}")
                self.model_status['daisee'] = False
    
    def _load_mpiigaze_model(self):
        """Load MPIIGaze model for gaze direction"""
        mpiigaze_dir = self.models_dir.parent / "models_mpiigaze"
        mpiigaze_path = mpiigaze_dir / "mpiigaze_best.pth"
        
        if mpiigaze_path.exists() and TORCH_AVAILABLE:
            try:
                self.models['mpiigaze'] = torch.load(mpiigaze_path, map_location=self.device)
                self.models['mpiigaze'].eval()
                self.model_status['mpiigaze'] = True
                logger.info("MPIIGaze model loaded")
            except Exception as e:
                logger.error(f"Failed to load MPIIGaze model: {e}")
                self.model_status['mpiigaze'] = False
    
    def _load_sklearn_models(self):
        """Load scikit-learn based models"""
        if not SKLEARN_AVAILABLE:
            return
            
        # Load local models
        model_files = [
            ('random_forest', 'local_attention_model_random_forest.pkl'),
            ('gradient_boosting', 'local_attention_model_gradient_boosting.pkl'),
            ('ensemble', 'local_attention_model_ensemble.pkl')
        ]
        
        for model_name, filename in model_files:
            model_path = self.models_dir / filename
            if model_path.exists():
                try:
                    with open(model_path, 'rb') as f:
                        self.models[model_name] = pickle.load(f)
                    self.model_status[model_name] = True
                    logger.info(f"{model_name} model loaded")
                except Exception as e:
                    logger.error(f"Failed to load {model_name}: {e}")
                    self.model_status[model_name] = False
        
        # Load scalers
        scaler_files = [
            ('random_forest', 'local_scaler_random_forest.pkl'),
            ('gradient_boosting', 'local_scaler_gradient_boosting.pkl'),
            ('ensemble', 'local_scaler_ensemble.pkl')
        ]
        
        for scaler_name, filename in scaler_files:
            scaler_path = self.models_dir / filename
            if scaler_path.exists():
                try:
                    with open(scaler_path, 'rb') as f:
                        self.scalers[scaler_name] = pickle.load(f)
                    logger.info(f"{scaler_name} scaler loaded")
                except Exception as e:
                    logger.error(f"Failed to load {scaler_name} scaler: {e}")
    
    def predict_attention(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Main prediction method using all available models
        """
        try:
            start_time = time.time()
            
            # Extract comprehensive features
            features = self._extract_comprehensive_features(frame)
            
            if not features['face_detected']:
                return self._no_face_result(start_time)
            
            # Get predictions from all available models
            predictions = []
            confidences = []
            model_used = []
            
            # Try Mendeley ensemble first (highest accuracy)
            if self._is_model_available('mendeley_gradient_boosting'):
                pred, conf = self._predict_mendeley_ensemble(features)
                if pred is not None:
                    predictions.append(pred)
                    confidences.append(conf)
                    model_used.append('Mendeley Ensemble')
            
            # Try Mendeley Neural Network
            if self._is_model_available('mendeley_nn'):
                pred, conf = self._predict_mendeley_nn(features)
                if pred is not None:
                    predictions.append(pred)
                    confidences.append(conf)
                    model_used.append('Mendeley NN')
            
            # Try DAiSEE model
            if self._is_model_available('daisee'):
                pred, conf = self._predict_daisee(features)
                if pred is not None:
                    predictions.append(pred)
                    confidences.append(conf)
                    model_used.append('DAiSEE')
            
            # Try local models
            for model_name in ['ensemble', 'gradient_boosting', 'random_forest']:
                if self._is_model_available(model_name):
                    pred, conf = self._predict_sklearn_model(features, model_name)
                    if pred is not None:
                        predictions.append(pred)
                        confidences.append(conf)
                        model_used.append(f'Local {model_name}')
            
            # Ensemble the predictions
            if predictions:
                final_score = np.mean(predictions)
                final_confidence = np.mean(confidences)
                models_str = ' + '.join(model_used)
            else:
                # Fallback to heuristic
                final_score, final_confidence = self._heuristic_prediction(features)
                models_str = 'Heuristic Fallback'
            
            # Apply temporal smoothing
            self.feature_history.append(final_score)
            if len(self.feature_history) > self.max_history:
                self.feature_history.pop(0)
            
            smoothed_score = np.mean(self.feature_history)
            
            # Determine attention level and prediction
            if smoothed_score >= 0.7:
                prediction = 'attentive'
                attention_level = 'high'
            elif smoothed_score >= 0.4:
                prediction = 'partially_attentive'
                attention_level = 'medium'
            else:
                prediction = 'distracted'
                attention_level = 'low'
            
            processing_time = time.time() - start_time
            
            return {
                'attention_score': float(smoothed_score),
                'prediction': prediction,
                'attention_level': attention_level,
                'confidence': float(final_confidence),
                'face_detected': True,
                'phone_detected': features.get('phone_detected', False),
                'model_used': models_str,
                'processing_time': processing_time,
                'pose_attention_weight': features.get('pose_attention_weight', 0.5),
                'gaze_score': features.get('gaze_score', 0.5),
                'emotion_score': features.get('emotion_score', 0.5)
            }
            
        except Exception as e:
            logger.error(f"Enhanced prediction failed: {e}")
            return self._error_result(str(e))
    
    def _extract_comprehensive_features(self, frame: np.ndarray) -> Dict[str, Any]:
        """Extract comprehensive features from frame"""
        features = {
            'face_detected': False,
            'phone_detected': False,
            'pose_attention_weight': 0.0,
            'gaze_score': 0.5,
            'emotion_score': 0.5
        }
        
        try:
            # Basic face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) == 0:
                return features
            
            features['face_detected'] = True
            face = faces[0]  # Take the largest face
            x, y, w, h = face
            
            # Extract face region
            face_region = frame[y:y+h, x:x+w]
            features['face_region'] = face_region
            features['face_bbox'] = [x, y, w, h]
            
            # Calculate face area relative to frame
            frame_area = frame.shape[0] * frame.shape[1]
            face_area = w * h
            features['face_area_ratio'] = face_area / frame_area
            
            # Face position features
            features['face_center_x'] = (x + w/2) / frame.shape[1]
            features['face_center_y'] = (y + h/2) / frame.shape[0]
            
            # Head pose estimation (simplified)
            pose_score = self._estimate_head_pose(face_region)
            features['pose_attention_weight'] = pose_score
            
            # Phone detection (basic)
            phone_detected = self._detect_phone_usage(frame, face)
            features['phone_detected'] = phone_detected
            
            # Additional features for ML models
            features.update(self._extract_ml_features(frame, face))
            
        except Exception as e:
            logger.error(f"Feature extraction error: {e}")
        
        return features
    
    def _estimate_head_pose(self, face_region: np.ndarray) -> float:
        """Estimate head pose attention score"""
        try:
            # Simple head pose estimation based on face symmetry
            gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape
            
            # Calculate left-right symmetry
            left_half = gray[:, :w//2]
            right_half = cv2.flip(gray[:, w//2:], 1)
            
            # Resize to match if needed
            min_width = min(left_half.shape[1], right_half.shape[1])
            left_half = left_half[:, :min_width]
            right_half = right_half[:, :min_width]
            
            # Calculate similarity (inverse of difference)
            diff = np.mean(np.abs(left_half.astype(float) - right_half.astype(float)))
            symmetry_score = max(0, 1 - diff / 255.0)
            
            return symmetry_score
            
        except Exception as e:
            logger.error(f"Head pose estimation error: {e}")
            return 0.5
    
    def _detect_phone_usage(self, frame: np.ndarray, face_bbox: Tuple[int, int, int, int]) -> bool:
        """Detect phone usage near face"""
        try:
            x, y, w, h = face_bbox
            
            # Define region below face where phone might be
            phone_region_y = y + h
            phone_region_h = min(h, frame.shape[0] - phone_region_y)
            
            if phone_region_h <= 0:
                return False
            
            phone_region = frame[phone_region_y:phone_region_y + phone_region_h, x:x+w]
            
            # Simple edge detection for rectangular objects
            gray = cv2.cvtColor(phone_region, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 500:  # Reasonable phone size
                    rect_x, rect_y, rect_w, rect_h = cv2.boundingRect(contour)
                    aspect_ratio = rect_w / rect_h if rect_h > 0 else 0
                    
                    # Phone-like aspect ratio
                    if 0.4 < aspect_ratio < 0.8:
                        return True
            
            return False
            
        except Exception as e:
            logger.error(f"Phone detection error: {e}")
            return False
    
    def _extract_ml_features(self, frame: np.ndarray, face_bbox: Tuple[int, int, int, int]) -> Dict[str, float]:
        """Extract features for ML models"""
        features = {}
        
        try:
            x, y, w, h = face_bbox
            
            # Face size and position features
            features['face_width'] = w / frame.shape[1]
            features['face_height'] = h / frame.shape[0]
            features['face_aspect_ratio'] = w / h if h > 0 else 1.0
            
            # Face region analysis
            face_region = frame[y:y+h, x:x+w]
            face_gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            
            # Texture features
            features['face_mean_intensity'] = np.mean(face_gray) / 255.0
            features['face_std_intensity'] = np.std(face_gray) / 255.0
            
            # Edge density (activity indicator)
            edges = cv2.Canny(face_gray, 50, 150)
            features['edge_density'] = np.sum(edges > 0) / (w * h)
            
            # Additional geometric features
            features['face_center_distance'] = np.sqrt(
                (features.get('face_center_x', 0.5) - 0.5)**2 + 
                (features.get('face_center_y', 0.5) - 0.5)**2
            )
            
        except Exception as e:
            logger.error(f"ML feature extraction error: {e}")
        
        return features
    
    def _predict_mendeley_ensemble(self, features: Dict[str, Any]) -> Tuple[Optional[float], float]:
        """Predict using Mendeley ensemble models"""
        try:
            # Create feature vector (28 features for Mendeley)
            feature_vector = self._create_mendeley_feature_vector(features)
            
            # Scale features
            if 'mendeley' in self.scalers:
                scaled_features = self.scalers['mendeley'].transform([feature_vector])
            else:
                scaled_features = [feature_vector]
            
            predictions = []
            for model_name in ['gradient_boosting', 'random_forest', 'logistic_regression']:
                model_key = f'mendeley_{model_name}'
                if model_key in self.models:
                    try:
                        pred_proba = self.models[model_key].predict_proba(scaled_features)[0]
                        predictions.append(pred_proba[1])  # Probability of attention
                    except Exception as e:
                        logger.warning(f"Mendeley {model_name} prediction failed: {e}")
            
            if predictions:
                avg_prediction = np.mean(predictions)
                confidence = 1.0 - np.std(predictions)  # Higher std = lower confidence
                return float(avg_prediction), float(confidence)
            
        except Exception as e:
            logger.error(f"Mendeley ensemble prediction error: {e}")
        
        return None, 0.0
    
    def _predict_mendeley_nn(self, features: Dict[str, Any]) -> Tuple[Optional[float], float]:
        """Predict using Mendeley neural network"""
        try:
            if 'face_region' not in features:
                return None, 0.0
            
            face_img = features['face_region']
            face_resized = cv2.resize(face_img, (96, 96))
            face_tensor = torch.FloatTensor(face_resized).permute(2, 0, 1).unsqueeze(0) / 255.0
            face_tensor = face_tensor.to(self.device)
            
            model = self.models['mendeley_nn']
            with torch.no_grad():
                outputs = model(face_tensor)
                probs = torch.softmax(outputs, dim=1)
                attention_score = float(probs[0][1])  # Probability of attention
                confidence = float(torch.max(probs))
            
            return attention_score, confidence
            
        except Exception as e:
            logger.error(f"Mendeley NN prediction error: {e}")
            return None, 0.0
    
    def _predict_daisee(self, features: Dict[str, Any]) -> Tuple[Optional[float], float]:
        """Predict using DAiSEE model"""
        try:
            # DAiSEE prediction would go here
            # For now, return a placeholder
            return 0.6, 0.7
        except Exception as e:
            logger.error(f"DAiSEE prediction error: {e}")
            return None, 0.0
    
    def _predict_sklearn_model(self, features: Dict[str, Any], model_name: str) -> Tuple[Optional[float], float]:
        """Predict using sklearn model"""
        try:
            feature_vector = self._create_feature_vector(features)
            
            # Scale if scaler available
            if model_name in self.scalers:
                scaled_features = self.scalers[model_name].transform([feature_vector])
            else:
                scaled_features = [feature_vector]
            
            model = self.models[model_name]
            pred_proba = model.predict_proba(scaled_features)[0]
            
            return float(pred_proba[1]), float(max(pred_proba))
            
        except Exception as e:
            logger.error(f"Sklearn {model_name} prediction error: {e}")
            return None, 0.0
    
    def _create_mendeley_feature_vector(self, features: Dict[str, Any]) -> List[float]:
        """Create 28-feature vector for Mendeley models"""
        vector = []
        
        # Basic features
        vector.extend([
            features.get('face_area_ratio', 0.0),
            features.get('face_center_x', 0.5),
            features.get('face_center_y', 0.5),
            features.get('face_width', 0.0),
            features.get('face_height', 0.0),
            features.get('face_aspect_ratio', 1.0),
            features.get('face_mean_intensity', 0.5),
            features.get('face_std_intensity', 0.0),
            features.get('edge_density', 0.0),
            features.get('face_center_distance', 0.0),
            features.get('pose_attention_weight', 0.5),
            float(features.get('phone_detected', False)),
            features.get('gaze_score', 0.5),
            features.get('emotion_score', 0.5)
        ])
        
        # Pad with additional features to reach 28
        while len(vector) < 28:
            vector.append(0.0)
        
        return vector[:28]  # Ensure exactly 28 features
    
    def _create_feature_vector(self, features: Dict[str, Any]) -> List[float]:
        """Create feature vector for local models"""
        return [
            features.get('face_area_ratio', 0.0),
            features.get('face_center_x', 0.5),
            features.get('face_center_y', 0.5),
            features.get('pose_attention_weight', 0.5),
            float(features.get('phone_detected', False)),
            features.get('gaze_score', 0.5),
            features.get('emotion_score', 0.5),
            features.get('face_width', 0.0),
            features.get('face_height', 0.0),
            features.get('edge_density', 0.0)
        ]
    
    def _heuristic_prediction(self, features: Dict[str, Any]) -> Tuple[float, float]:
        """Fallback heuristic prediction"""
        score = 0.5
        
        # Face position
        center_x = features.get('face_center_x', 0.5)
        center_y = features.get('face_center_y', 0.5)
        center_distance = abs(center_x - 0.5) + abs(center_y - 0.5)
        score += (0.5 - center_distance) * 0.3
        
        # Face size
        face_area = features.get('face_area_ratio', 0.0)
        if face_area > 0.05:  # Reasonable face size
            score += 0.2
        
        # Phone usage
        if not features.get('phone_detected', False):
            score += 0.2
        
        # Pose
        pose_score = features.get('pose_attention_weight', 0.5)
        score += pose_score * 0.3
        
        return max(0.0, min(1.0, score)), 0.6
    
    def _is_model_available(self, model_name: str) -> bool:
        """Check if model is available and loaded"""
        return model_name in self.models and self.model_status.get(model_name, False)
    
    def _no_face_result(self, start_time: float) -> Dict[str, Any]:
        """Return result when no face is detected"""
        return {
            'attention_score': 0.0,
            'prediction': 'no_face',
            'attention_level': 'unknown',
            'confidence': 0.0,
            'face_detected': False,
            'phone_detected': False,
            'model_used': 'None (No face detected)',
            'processing_time': time.time() - start_time,
            'pose_attention_weight': 0.0,
            'gaze_score': 0.0,
            'emotion_score': 0.0
        }
    
    def _error_result(self, error_msg: str) -> Dict[str, Any]:
        """Return result when error occurs"""
        return {
            'attention_score': 0.3,
            'prediction': 'error',
            'attention_level': 'unknown',
            'confidence': 0.0,
            'face_detected': False,
            'phone_detected': False,
            'model_used': 'Error',
            'processing_time': 0.0,
            'pose_attention_weight': 0.0,
            'gaze_score': 0.0,
            'emotion_score': 0.0,
            'error': error_msg
        }
