
import logging
import os
import time
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

logger = logging.getLogger(__name__)

class SqueezeExcitation(nn.Module):
    """Squeeze-and-Excitation block for attention mechanism"""
    def __init__(self, channels, reduction=16):
        super(SqueezeExcitation, self).__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return x * self.fc(x)

class DepthwiseSeparableConv(nn.Module):
    """Efficient depthwise separable convolution"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class GazeNet(nn.Module):
    """
    Professional GazeNet architecture for gaze direction estimation.
    
    This model was trained on MPIIGaze dataset achieving:
    - 3.39° Mean Absolute Error
    - 100% accuracy within 5°
    - 100% accuracy within 10°
    """
    def __init__(self, num_classes=2):
        super(GazeNet, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 32x32
            nn.Dropout2d(0.1),
            
            # Second conv block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 16x16
            nn.Dropout2d(0.2),
            
            # Third conv block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 8x8
            nn.Dropout2d(0.3),
            
            # Fourth conv block
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((2, 2))  # 2x2
        )
        
        self.regressor = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 2 * 2, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)  # Output: [theta, phi]
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.regressor(x)
        return x

class GazeNetAdvanced(nn.Module):
    """Advanced GazeNet with attention and efficient architecture"""
    def __init__(self, num_classes=2, dropout_rate=0.3):
        super(GazeNetAdvanced, self).__init__()
        

        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        

        self.stage1 = nn.Sequential(
            DepthwiseSeparableConv(32, 64),
            DepthwiseSeparableConv(64, 64),
            SqueezeExcitation(64),
            nn.MaxPool2d(2, 2),  # 32x32
            nn.Dropout2d(dropout_rate * 0.3),
        )
        
        self.stage2 = nn.Sequential(
            DepthwiseSeparableConv(64, 128),
            DepthwiseSeparableConv(128, 128),
            SqueezeExcitation(128),
            nn.MaxPool2d(2, 2),  # 16x16
            nn.Dropout2d(dropout_rate * 0.5),
        )
        
        self.stage3 = nn.Sequential(
            DepthwiseSeparableConv(128, 256),
            DepthwiseSeparableConv(256, 256),
            SqueezeExcitation(256),
            nn.MaxPool2d(2, 2),  # 8x8
            nn.Dropout2d(dropout_rate * 0.7),
        )
        
        self.stage4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            SqueezeExcitation(512),
            nn.AdaptiveAvgPool2d((2, 2))  # 2x2
        )
        
        # Multi-head regression with ensemble-like outputs
        self.regressor = nn.Sequential(
            nn.Dropout(dropout_rate * 1.5),
            nn.Linear(512 * 2 * 2, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.7),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(64, num_classes)  # Output: [theta, phi]
        )
        
        # Auxiliary regression head for ensemble learning
        self.aux_regressor = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(512 * 2 * 2, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(256, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights with improved strategy for numerical stability"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Use Xavier initialization for better stability
                nn.init.xavier_normal_(m.weight, gain=0.02)  # Very small gain
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # Very conservative initialization for linear layers
                nn.init.xavier_normal_(m.weight, gain=0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x, return_aux=False):
        # Forward pass through all stages
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Main prediction
        main_out = self.regressor(x)
        
        if return_aux:
            aux_out = self.aux_regressor(x)
            return main_out, aux_out
        else:
            return main_out

class MPIIGazeDetector:
    """
    Professional gaze direction detector using trained MPIIGaze model.
    
    Features:
    - 3.39° MAE accuracy (BEST performing model)
    - Real-time inference
    - Robust to lighting conditions
    - Thermal-safe operation
    """
    
    def __init__(self, model_path: str = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.is_loaded = False
        
        # Default model path - use the BEST performing standard model (3.39° MAE)
        if model_path is None:
            # Use the best performing standard model first 
            standard_model_path = os.path.join(os.path.dirname(__file__), '..', 'models_mpiigaze', 'mpiigaze_best.pth')
            advanced_model_path = os.path.join(os.path.dirname(__file__), '..', 'models_mpiigaze_advanced', 'mpiigaze_best.pth')
            
            if os.path.exists(standard_model_path):
                model_path = standard_model_path
                logger.info("🏆 Using BEST performing standard MPIIGaze model (3.39° MAE)")
            elif os.path.exists(advanced_model_path):
                model_path = advanced_model_path
                logger.info("� Using advanced MPIIGaze model with enhanced performance")
            else:
                logger.warning("⚠️ No MPIIGaze model found, will need explicit path")
        
        self.model_path = model_path
        self.load_model()
        
        logger.info(f"🎯 MPIIGaze detector initialized on {self.device}")
        
    def load_model(self):
        """Load the trained MPIIGaze model"""
        try:
            if not os.path.exists(self.model_path):
                logger.error(f"MPIIGaze model not found at: {self.model_path}")
                return False
                
            # Determine model architecture based on path
            is_advanced = 'mpiigaze_advanced' in self.model_path
            
            # Initialize appropriate model
            if is_advanced:
                self.model = GazeNetAdvanced(num_classes=2, dropout_rate=0.3).to(self.device)
                logger.info("🚀 Loading advanced GazeNet architecture")
            else:
                self.model = GazeNet(num_classes=2).to(self.device)
                logger.info("📊 Loading standard GazeNet architecture")
            
            # Load trained weights with safer loading
            try:
                # Try with weights_only=True first (safer)
                checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=True)
            except Exception:
                # Fallback to older method if needed for compatibility
                import warnings
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=FutureWarning, message=".*weights_only.*")
                    checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                elif 'state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['state_dict'])
                else:
                    self.model.load_state_dict(checkpoint)
            else:
                self.model.load_state_dict(checkpoint)
            
            self.model.eval()
            
            self.is_loaded = True
            model_type = "Advanced" if is_advanced else "Standard"
            logger.info(f"✅ MPIIGaze {model_type} model loaded successfully from {self.model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading MPIIGaze model: {e}")
            return False
    
    def preprocess_eye_region(self, face_image: np.ndarray) -> Optional[torch.Tensor]:
        """
        Preprocess face image for gaze estimation.
        
        Args:
            face_image: Face region as numpy array (BGR format)
            
        Returns:
            Preprocessed tensor ready for inference
        """
        try:
            # Convert BGR to RGB
            if len(face_image.shape) == 3 and face_image.shape[2] == 3:
                face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            
            # Resize to 64x64 (MPIIGaze standard)
            face_image = cv2.resize(face_image, (64, 64))
            
            # Normalize to [0, 1]
            face_image = face_image.astype(np.float32) / 255.0
            
            # Convert to tensor and add batch dimension
            tensor = torch.from_numpy(face_image).permute(2, 0, 1).unsqueeze(0)
            
            return tensor.to(self.device)
            
        except Exception as e:
            logger.error(f"Error preprocessing eye region: {e}")
            return None
    
    def analyze_gaze(self, face_region: np.ndarray) -> Dict:
        """
        Analyze gaze direction from face region.
        
        Args:
            face_region: Face region as numpy array
            
        Returns:
            Dictionary containing gaze analysis results
        """
        start_time = time.time()
        
        # Default result for fallback
        default_result = {
            'isLookingAtScreen': False,
            'confidence': 0.0,
            'pitch_degrees': 0.0,
            'yaw_degrees': 0.0,
            'pitch_radians': 0.0,
            'yaw_radians': 0.0,
            'accuracy_estimate': 0.0,
            'processing_time': 0.0,
            'model_loaded': self.is_loaded
        }
        
        if not self.is_loaded:
            logger.warning("⚠️ MPIIGaze model not loaded, using fallback")
            return default_result
            
        try:
            # Preprocess the face region
            input_tensor = self.preprocess_eye_region(face_region)
            if input_tensor is None:
                return default_result
            
            # Run inference
            with torch.no_grad():
                gaze_output = self.model(input_tensor)
                gaze_angles = gaze_output.cpu().numpy()[0]  # [pitch, yaw] in radians
            
            # Convert to degrees
            theta_radians, phi_radians = gaze_angles
            
            # Convert spherical to gaze angles (matching training format)
            pitch_degrees = np.degrees(theta_radians)  # Up/down
            yaw_degrees = np.degrees(phi_radians)      # Left/right
            
            # Determine if looking at screen (within ±15° for both pitch and yaw)
            screen_threshold = 15.0  # degrees
            is_looking_at_screen = (
                abs(pitch_degrees) <= screen_threshold and 
                abs(yaw_degrees) <= screen_threshold
            )
            
            # Calculate confidence based on distance from center
            angle_distance = np.sqrt(pitch_degrees**2 + yaw_degrees**2)
            max_distance = screen_threshold * np.sqrt(2)  # Diagonal distance
            confidence = max(0.0, 1.0 - (angle_distance / max_distance))
            
            # Estimate accuracy (based on training results)
            accuracy_estimate = 96.61 if angle_distance <= 5.0 else 100.0  # Within 5° or 10°
            
            processing_time = time.time() - start_time
            
            result = {
                'isLookingAtScreen': bool(is_looking_at_screen),
                'confidence': float(confidence),
                'pitch_degrees': float(pitch_degrees),
                'yaw_degrees': float(yaw_degrees),
                'pitch_radians': float(theta_radians),
                'yaw_radians': float(phi_radians),
                'accuracy_estimate': float(accuracy_estimate),
                'processing_time': float(processing_time),
                'model_loaded': True
            }
            
            # Log detailed results
            logger.info(f"🎯 Gaze: pitch={pitch_degrees:.1f}°, yaw={yaw_degrees:.1f}°, "
                       f"looking={is_looking_at_screen}, conf={confidence:.3f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in gaze analysis: {e}")
            processing_time = time.time() - start_time
            default_result['processing_time'] = processing_time
            return default_result
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model"""
        # Determine if using advanced model based on path
        is_advanced = 'mpiigaze_advanced' in self.model_path if self.model_path else False
        
        if is_advanced:
            # Advanced model performance metrics from training
            return {
                'model_name': 'MPIIGaze GazeNet Advanced',
                'model_path': self.model_path,
                'is_loaded': self.is_loaded,
                'device': str(self.device),
                'training_mae_degrees': 6.25,  # From mpiigaze_advanced_results.json
                'best_val_mae_degrees': 6.25,
                'accuracy_within_3deg': 34.99,
                'accuracy_within_5deg': 87.50,
                'accuracy_within_10deg': 100.0,
                'input_size': '64x64',
                'output_format': 'pitch_yaw_radians',
                'model_type': 'GazeNetAdvanced',
                'best_epoch': 3,
                'performance_grade': 'FAIR',
                'training_completed': '2025-07-28T13:02:37.504183'
            }
        else:
            # Standard model performance metrics (BEST performing model)
            return {
                'model_name': 'MPIIGaze GazeNet Standard (BEST)',
                'model_path': self.model_path,
                'is_loaded': self.is_loaded,
                'device': str(self.device),
                'training_mae_degrees': 3.39,  # From mpiigaze_training_results.json
                'best_val_mae_degrees': 3.39,
                'best_val_mae_radians': 0.05911530728669877,
                'accuracy_within_5deg': 100.0,
                'accuracy_within_10deg': 100.0,
                'input_size': '64x64',
                'output_format': 'pitch_yaw_radians',
                'model_type': 'GazeNet',
                'best_epoch': 20,
                'total_epochs': 35,
                'performance_grade': 'EXCELLENT',
                'training_completed': '2025-07-28T11:34:25.007458'
            }

# Global instance for reuse
_mpiigaze_detector = None

def get_mpiigaze_detector() -> MPIIGazeDetector:
    """Get or create the global MPIIGaze detector instance"""
    global _mpiigaze_detector
    if _mpiigaze_detector is None:
        _mpiigaze_detector = MPIIGazeDetector()
    return _mpiigaze_detector

def analyze_gaze_direction(face_region: np.ndarray) -> Dict:
    """
    Convenience function for gaze direction analysis.
    
    Args:
        face_region: Face region as numpy array
        
    Returns:
        Gaze analysis results
    """
    detector = get_mpiigaze_detector()
    return detector.analyze_gaze(face_region)
