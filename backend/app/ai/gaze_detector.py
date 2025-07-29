"""
Enhanced Gaze Detection for DersLens
Real-time gaze estimation with excellent performance
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

class DersLensGazeDetector:
    """
    Enhanced gaze detector for DersLens with excellent performance
    Optimized for real-time classroom monitoring
    """
    
    def __init__(self, model_path: Optional[str] = None, device: str = "auto"):
        """
        Initialize the DersLens gaze detector
        
        Args:
            model_path: Path to trained model (optional)
            device: Device to use ('auto', 'cuda', 'cpu')
        """
        # Device setup
        if device == "auto":
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        logger.info(f"🎯 DersLens Gaze Detector initializing on {self.device}")
        
        # Model setup
        self.model = None
        self.model_info = {}
        
        # Face detection setup
        self.setup_face_detection()
        
        # Performance tracking
        self.inference_times = []
        self.frame_count = 0
        
        # Load model if provided
        if model_path:
            self.load_model(model_path)
        
        logger.info("✅ DersLens Gaze Detector ready")
    
    def setup_face_detection(self):
        """Setup face detection for preprocessing"""
        try:
            # Try to load OpenCV's DNN face detector for better performance
            self.face_net = cv2.dnn.readNetFromTensorflow(
                cv2.data.haarcascades + 'opencv_face_detector_uint8.pb',
                cv2.data.haarcascades + 'opencv_face_detector.pbtxt'
            )
            self.use_dnn_face = True
            logger.info("✅ Using DNN face detector")
        except:
            # Fallback to Haar cascades
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            self.use_dnn_face = False
            logger.info("✅ Using Haar cascade face detector")
    
    def create_model_architecture(self) -> nn.Module:
        """Create the enhanced stable model architecture"""
        
        class EnhancedStableGazeNet(nn.Module):
            """Enhanced stable architecture for excellent gaze estimation"""
            
            def __init__(self):
                super(EnhancedStableGazeNet, self).__init__()
                
                # Enhanced feature extraction
                self.features = nn.Sequential(
                    # First block
                    nn.Conv2d(3, 32, 3, padding=1),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 32, 3, padding=1),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),  # 32x32
                    
                    # Second block
                    nn.Conv2d(32, 64, 3, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(64, 64, 3, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),  # 16x16
                    
                    # Third block
                    nn.Conv2d(64, 128, 3, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(128, 128, 3, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),  # 8x8
                    
                    # Global average pooling
                    nn.AdaptiveAvgPool2d((2, 2))
                )
                
                # Enhanced regression head
                self.regressor = nn.Sequential(
                    nn.Dropout(0.3),
                    nn.Linear(128 * 2 * 2, 256),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.3),
                    nn.Linear(256, 128),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.2),
                    nn.Linear(128, 64),
                    nn.ReLU(inplace=True),
                    nn.Linear(64, 2)
                )
            
            def forward(self, x):
                x = self.features(x)
                x = x.view(x.size(0), -1)
                x = self.regressor(x)
                return x
        
        return EnhancedStableGazeNet()
    
    def load_model(self, model_path: str) -> bool:
        """
        Load trained gaze model
        
        Args:
            model_path: Path to model checkpoint
            
        Returns:
            True if loaded successfully
        """
        try:
            model_path = Path(model_path)
            if not model_path.exists():
                logger.error(f"Model not found at {model_path}")
                return False
            
            # Create model architecture
            self.model = self.create_model_architecture()
            
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            # Store model info
            self.model_info = {
                'epoch': checkpoint.get('epoch', 'unknown'),
                'val_mae_degrees': checkpoint.get('val_mae_degrees', 'unknown'),
                'performance_class': checkpoint.get('performance_class', 'unknown'),
                'model_architecture': checkpoint.get('model_architecture', 'EnhancedStableGazeNet')
            }
            
            logger.info(f"✅ Model loaded successfully")
            logger.info(f"📊 Epoch: {self.model_info['epoch']}")
            logger.info(f"🎯 VAL MAE: {self.model_info['val_mae_degrees']}°")
            logger.info(f"⭐ Performance: {self.model_info['performance_class']}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def detect_faces_dnn(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces using DNN detector"""
        try:
            h, w = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123])
            self.face_net.setInput(blob)
            detections = self.face_net.forward()
            
            faces = []
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.5:  # Confidence threshold
                    x1 = int(detections[0, 0, i, 3] * w)
                    y1 = int(detections[0, 0, i, 4] * h)
                    x2 = int(detections[0, 0, i, 5] * w)
                    y2 = int(detections[0, 0, i, 6] * h)
                    faces.append((x1, y1, x2 - x1, y2 - y1))
            
            return faces
        except:
            return []
    
    def detect_faces_haar(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces using Haar cascade"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )
            return [(x, y, w, h) for x, y, w, h in faces]
        except:
            return []
    
    def detect_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces in frame"""
        if self.use_dnn_face:
            return self.detect_faces_dnn(frame)
        else:
            return self.detect_faces_haar(frame)
    
    def preprocess_face(self, face_crop: np.ndarray) -> torch.Tensor:
        """
        Preprocess face crop for gaze estimation
        
        Args:
            face_crop: Face crop (BGR format)
            
        Returns:
            Preprocessed tensor
        """
        try:
            # Resize to model input size (64x64)
            face_resized = cv2.resize(face_crop, (64, 64))
            
            # Convert BGR to RGB
            face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
            
            # Normalize to [0, 1]
            face_normalized = face_rgb.astype(np.float32) / 255.0
            
            # Convert to tensor and add batch dimension
            face_tensor = torch.from_numpy(face_normalized).permute(2, 0, 1).unsqueeze(0)
            
            return face_tensor.to(self.device)
            
        except Exception as e:
            logger.error(f"Preprocessing failed: {e}")
            return None
    
    def predict_gaze(self, face_tensor: torch.Tensor) -> Optional[Tuple[float, float]]:
        """
        Predict gaze direction from face tensor
        
        Args:
            face_tensor: Preprocessed face tensor
            
        Returns:
            (yaw, pitch) in degrees or None if failed
        """
        if self.model is None:
            logger.error("No model loaded")
            return None
        
        try:
            start_time = time.time()
            
            with torch.no_grad():
                outputs = self.model(face_tensor)
                gaze_vector = outputs.squeeze().cpu().numpy()
            
            # Convert radians to degrees
            yaw_deg = float(gaze_vector[0] * 180 / np.pi)
            pitch_deg = float(gaze_vector[1] * 180 / np.pi)
            
            # Track inference time
            inference_time = time.time() - start_time
            self.inference_times.append(inference_time)
            
            # Keep only last 100 times for average
            if len(self.inference_times) > 100:
                self.inference_times.pop(0)
            
            return (yaw_deg, pitch_deg)
            
        except Exception as e:
            logger.error(f"Gaze prediction failed: {e}")
            return None
    
    def classify_attention_zone(self, yaw: float, pitch: float) -> str:
        """
        Classify gaze direction into attention zones for classroom monitoring
        
        Args:
            yaw: Horizontal gaze angle in degrees
            pitch: Vertical gaze angle in degrees
            
        Returns:
            Attention zone classification
        """
        # Define attention zones for classroom
        if abs(yaw) < 15 and abs(pitch) < 10:
            return "focused"  # Looking at teacher/board
        elif abs(yaw) < 30 and abs(pitch) < 20:
            return "attentive"  # Generally looking forward
        elif abs(yaw) < 45:
            return "distracted"  # Looking to sides
        else:
            return "off_task"  # Looking away
    
    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Process a single frame for gaze detection
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            Dictionary with detection results
        """
        self.frame_count += 1
        results = {
            'frame_id': self.frame_count,
            'faces': [],
            'timestamp': time.time(),
            'processing_time': 0
        }
        
        start_time = time.time()
        
        try:
            # Detect faces
            faces = self.detect_faces(frame)
            
            for i, (x, y, w, h) in enumerate(faces):
                face_result = {
                    'face_id': i,
                    'bbox': (x, y, w, h),
                    'gaze': None,
                    'attention_zone': 'unknown',
                    'confidence': 0.0
                }
                
                # Extract face crop
                face_crop = frame[y:y+h, x:x+w]
                
                if face_crop.size > 0:
                    # Preprocess face
                    face_tensor = self.preprocess_face(face_crop)
                    
                    if face_tensor is not None:
                        # Predict gaze
                        gaze_prediction = self.predict_gaze(face_tensor)
                        
                        if gaze_prediction is not None:
                            yaw, pitch = gaze_prediction
                            attention_zone = self.classify_attention_zone(yaw, pitch)
                            
                            face_result.update({
                                'gaze': {'yaw': yaw, 'pitch': pitch},
                                'attention_zone': attention_zone,
                                'confidence': 1.0  # Could be enhanced with uncertainty estimation
                            })
                
                results['faces'].append(face_result)
            
            results['processing_time'] = time.time() - start_time
            
        except Exception as e:
            logger.error(f"Frame processing failed: {e}")
            results['error'] = str(e)
        
        return results
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        avg_inference_time = np.mean(self.inference_times) if self.inference_times else 0
        fps = 1.0 / avg_inference_time if avg_inference_time > 0 else 0
        
        return {
            'avg_inference_time_ms': avg_inference_time * 1000,
            'estimated_fps': fps,
            'total_frames_processed': self.frame_count,
            'model_info': self.model_info,
            'device': str(self.device)
        }
    
    def visualize_results(self, frame: np.ndarray, results: Dict[str, Any]) -> np.ndarray:
        """
        Visualize gaze detection results on frame
        
        Args:
            frame: Input frame
            results: Detection results from process_frame
            
        Returns:
            Annotated frame
        """
        annotated_frame = frame.copy()
        
        # Color mapping for attention zones
        zone_colors = {
            'focused': (0, 255, 0),      # Green
            'attentive': (0, 255, 255),  # Yellow
            'distracted': (0, 165, 255), # Orange
            'off_task': (0, 0, 255),     # Red
            'unknown': (128, 128, 128)   # Gray
        }
        
        for face in results.get('faces', []):
            x, y, w, h = face['bbox']
            attention_zone = face['attention_zone']
            color = zone_colors.get(attention_zone, (128, 128, 128))
            
            # Draw face bounding box
            cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), color, 2)
            
            # Draw gaze information
            if face['gaze']:
                yaw = face['gaze']['yaw']
                pitch = face['gaze']['pitch']
                
                # Draw gaze vector
                center_x = x + w // 2
                center_y = y + h // 2
                
                # Calculate gaze direction
                scale = 50
                end_x = int(center_x + yaw * scale / 45)  # Scale to pixel coords
                end_y = int(center_y + pitch * scale / 45)
                
                cv2.arrowedLine(annotated_frame, (center_x, center_y), (end_x, end_y), color, 2)
                
                # Add text
                text = f"{attention_zone}: ({yaw:.1f}°, {pitch:.1f}°)"
                cv2.putText(annotated_frame, text, (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Add performance info
        if hasattr(self, 'inference_times') and self.inference_times:
            avg_time = np.mean(self.inference_times[-10:]) * 1000  # Last 10 frames
            fps = 1000 / avg_time if avg_time > 0 else 0
            cv2.putText(annotated_frame, f"FPS: {fps:.1f} | Inference: {avg_time:.1f}ms", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return annotated_frame

def create_derslens_gaze_detector(model_path: str = "models_mpiigaze_derslens/derslens_gaze_best.pth") -> Optional[DersLensGazeDetector]:
    """
    Convenience function to create DersLens gaze detector
    
    Args:
        model_path: Path to trained model
        
    Returns:
        DersLensGazeDetector instance or None if failed
    """
    try:
        detector = DersLensGazeDetector()
        
        if Path(model_path).exists():
            if detector.load_model(model_path):
                logger.info("✅ DersLens gaze detector ready for inference")
                return detector
            else:
                logger.error("Failed to load model")
                return None
        else:
            logger.warning(f"⚠️ Model not found at {model_path}")
            logger.info("💡 Use gaze_trainer.py to train a model first")
            return detector  # Return detector without model for face detection
            
    except Exception as e:
        logger.error(f"Failed to create detector: {e}")
        return None

if __name__ == "__main__":
    # Test the detector
    detector = create_derslens_gaze_detector()
    
    if detector:
        logger.info("🎯 Testing with webcam...")
        cap = cv2.VideoCapture(0)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            results = detector.process_frame(frame)
            
            # Visualize results
            annotated_frame = detector.visualize_results(frame, results)
            
            cv2.imshow('DersLens Gaze Detection', annotated_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Print performance stats
        stats = detector.get_performance_stats()
        logger.info(f"📊 Performance: {stats}")
