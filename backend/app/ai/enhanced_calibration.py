"""
Enhanced Camera Calibration System for DersLens
Provides accurate camera parameters for gaze tracking and head pose estimation.
Fixes the hardcoded camera matrix issues causing inaccurate measurements.
"""

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

class CameraCalibrationSystem:
    """
    Enhanced camera calibration system for accurate gaze tracking.
    
    Fixes the major issue where hardcoded camera parameters were causing
    inaccurate gaze and head pose measurements.
    """
    
    def __init__(self):
        self.camera_matrix = None
        self.distortion_coeffs = None
        self.calibration_data = {}
        self.is_calibrated = False
        self.frame_size = None
        
        # Auto-calibration parameters
        self.auto_calibration_frames = []
        self.max_calibration_frames = 50
        self.min_calibration_frames = 20
        
    def auto_calibrate_from_face(self, frame: np.ndarray, face_landmarks) -> bool:
        """
        Auto-calibrate camera using face landmarks.
        This is more practical than checkerboard calibration for real-time use.
        """
        try:
            if self.is_calibrated:
                return True
                
            h, w = frame.shape[:2]
            
            # Estimate camera parameters from face geometry
            if face_landmarks and len(face_landmarks) > 10:
                # Use face landmarks to estimate focal length
                # Average distance between eyes as reference
                if hasattr(face_landmarks, 'landmark'):
                    # MediaPipe format
                    landmarks = [(lm.x * w, lm.y * h) for lm in face_landmarks.landmark]
                else:
                    # Already in pixel coordinates
                    landmarks = face_landmarks
                
                if len(landmarks) >= 468:  # MediaPipe face mesh
                    # Get eye corners for reference
                    left_eye_corner = landmarks[33]   # Left eye left corner
                    right_eye_corner = landmarks[263] # Right eye right corner
                    
                    # Average inter-pupillary distance is ~63mm
                    pixel_distance = np.sqrt((left_eye_corner[0] - right_eye_corner[0])**2 + 
                                           (left_eye_corner[1] - right_eye_corner[1])**2)
                    
                    if pixel_distance > 20:  # Reasonable face size
                        # Estimate focal length based on face geometry
                        # Typical IPD is 63mm, typical viewing distance is 60cm
                        estimated_focal_length = (pixel_distance * 600) / 63  # mm to pixels
                        
                        # Create camera matrix
                        self.camera_matrix = np.array([
                            [estimated_focal_length, 0, w / 2],
                            [0, estimated_focal_length, h / 2],
                            [0, 0, 1]
                        ], dtype=np.float64)
                        
                        # Assume minimal distortion for most webcams
                        self.distortion_coeffs = np.array([0.1, -0.2, 0, 0, 0], dtype=np.float64)
                        
                        self.frame_size = (w, h)
                        self.is_calibrated = True
                        
                        logger.info(f"Auto-calibrated camera:")
                        logger.info(f"   Focal length: {estimated_focal_length:.1f}")
                        logger.info(f"   Frame size: {w}x{h}")
                        logger.info(f"   IPD pixels: {pixel_distance:.1f}")
                        
                        return True
                        
        except Exception as e:
            logger.error(f"Auto-calibration failed: {e}")
        
        return False
    
    def manual_calibrate_from_gaze_points(self, gaze_data: List[Tuple]) -> bool:
        """
        Calibrate using known gaze points (user looks at specific screen locations).
        More accurate for gaze tracking applications.
        """
        try:
            if len(gaze_data) < 9:  # Need at least 9 points for good calibration
                logger.warning(f"Need at least 9 gaze points, got {len(gaze_data)}")
                return False
            
            # gaze_data format: [(screen_x, screen_y, measured_gaze_x, measured_gaze_y), ...]
            screen_points = np.array([[point[0], point[1]] for point in gaze_data])
            gaze_points = np.array([[point[2], point[3]] for point in gaze_data])
            
            # Calculate transformation matrix from gaze to screen coordinates
            # Using homography for better accuracy
            if len(gaze_data) >= 4:
                homography_matrix, mask = cv2.findHomography(
                    gaze_points, screen_points, cv2.RANSAC
                )
                
                if homography_matrix is not None:
                    self.calibration_data['gaze_to_screen_homography'] = homography_matrix
                    self.is_calibrated = True
                    
                    # Calculate calibration accuracy
                    transformed_points = cv2.perspectiveTransform(
                        gaze_points.reshape(-1, 1, 2), homography_matrix
                    ).reshape(-1, 2)
                    
                    errors = np.linalg.norm(transformed_points - screen_points, axis=1)
                    mean_error = np.mean(errors)
                    
                    logger.info(f"Gaze calibration complete:")
                    logger.info(f"   Mean error: {mean_error:.2f} pixels")
                    logger.info(f"   Calibration points: {len(gaze_data)}")
                    
                    return True
                    
        except Exception as e:
            logger.error(f"Manual gaze calibration failed: {e}")
        
        return False
    
    def get_camera_matrix(self, frame_shape: Tuple[int, int]) -> np.ndarray:
        """Get camera matrix, auto-calibrating if needed."""
        h, w = frame_shape[:2]
        
        if self.camera_matrix is not None and self.frame_size == (w, h):
            return self.camera_matrix
        
        # Fallback: estimate reasonable camera matrix
        focal_length = max(w, h) * 1.2  # Common approximation
        
        camera_matrix = np.array([
            [focal_length, 0, w / 2],
            [0, focal_length, h / 2], 
            [0, 0, 1]
        ], dtype=np.float64)
        
        logger.info(f"Using estimated camera matrix for {w}x{h}")
        return camera_matrix
    
    def get_distortion_coeffs(self) -> np.ndarray:
        """Get distortion coefficients."""
        if self.distortion_coeffs is not None:
            return self.distortion_coeffs
        
        # Minimal distortion assumption for most webcams
        return np.array([0.1, -0.2, 0, 0, 0], dtype=np.float64)
    
    def correct_gaze_point(self, raw_gaze_x: float, raw_gaze_y: float) -> Tuple[float, float]:
        """Apply calibration correction to gaze point."""
        if not self.is_calibrated or 'gaze_to_screen_homography' not in self.calibration_data:
            return raw_gaze_x, raw_gaze_y
        
        try:
            # Apply homography transformation
            gaze_point = np.array([[[raw_gaze_x, raw_gaze_y]]], dtype=np.float32)
            corrected_point = cv2.perspectiveTransform(
                gaze_point, self.calibration_data['gaze_to_screen_homography']
            )
            
            return float(corrected_point[0, 0, 0]), float(corrected_point[0, 0, 1])
            
        except Exception as e:
            logger.error(f"Gaze correction failed: {e}")
            return raw_gaze_x, raw_gaze_y
    
    def validate_calibration(self, test_points: List[Tuple]) -> Dict[str, float]:
        """Validate calibration accuracy with test points."""
        if not self.is_calibrated:
            return {'status': 'not_calibrated'}
        
        try:
            errors = []
            for screen_x, screen_y, measured_x, measured_y in test_points:
                corrected_x, corrected_y = self.correct_gaze_point(measured_x, measured_y)
                error = np.sqrt((corrected_x - screen_x)**2 + (corrected_y - screen_y)**2)
                errors.append(error)
            
            return {
                'status': 'validated',
                'mean_error_pixels': np.mean(errors),
                'max_error_pixels': np.max(errors),
                'std_error_pixels': np.std(errors),
                'test_points': len(test_points)
            }
            
        except Exception as e:
            logger.error(f"Calibration validation failed: {e}")
            return {'status': 'validation_failed', 'error': str(e)}
    
    def save_calibration(self, filepath: str) -> bool:
        """Save calibration data to file."""
        try:
            calibration_data = {
                'camera_matrix': self.camera_matrix.tolist() if self.camera_matrix is not None else None,
                'distortion_coeffs': self.distortion_coeffs.tolist() if self.distortion_coeffs is not None else None,
                'frame_size': self.frame_size,
                'is_calibrated': self.is_calibrated,
                'calibration_data': self.calibration_data,
                'timestamp': time.time()
            }
            
            import json
            with open(filepath, 'w') as f:
                json.dump(calibration_data, f, indent=2)
            
            logger.info(f"Calibration saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save calibration: {e}")
            return False
    
    def load_calibration(self, filepath: str) -> bool:
        """Load calibration data from file."""
        try:
            import json
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            if data.get('camera_matrix'):
                self.camera_matrix = np.array(data['camera_matrix'])
            if data.get('distortion_coeffs'):
                self.distortion_coeffs = np.array(data['distortion_coeffs'])
            
            self.frame_size = data.get('frame_size')
            self.is_calibrated = data.get('is_calibrated', False)
            self.calibration_data = data.get('calibration_data', {})
            
            logger.info(f"Calibration loaded from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load calibration: {e}")
            return False
    
    def get_status(self) -> Dict[str, any]:
        """Get calibration status and info."""
        return {
            'is_calibrated': self.is_calibrated,
            'camera_matrix_available': self.camera_matrix is not None,
            'distortion_coeffs_available': self.distortion_coeffs is not None,
            'frame_size': self.frame_size,
            'calibration_methods': list(self.calibration_data.keys()),
            'focal_length': float(self.camera_matrix[0, 0]) if self.camera_matrix is not None else None
        }


class EnhancedHeadPoseEstimator:
    """
    Enhanced head pose estimation using proper camera calibration.
    Fixes the inaccurate head pose calculations.
    """
    
    def __init__(self, calibration_system: CameraCalibrationSystem):
        self.calibration = calibration_system
        
        # 3D model points for face landmarks (in mm)
        # More accurate model based on anthropometric data
        self.model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Left eye left corner
            (225.0, 170.0, -135.0),      # Right eye right corner
            (-150.0, -150.0, -125.0),    # Left mouth corner
            (150.0, -150.0, -125.0)      # Right mouth corner
        ], dtype=np.float64)
        
        # Corresponding MediaPipe landmark indices
        self.landmark_indices = [1, 152, 33, 263, 61, 291]
    
    def estimate_head_pose(self, landmarks, frame_shape: Tuple[int, int]) -> Dict[str, float]:
        """
        Estimate head pose with proper camera calibration.
        Returns angles in degrees and confidence score.
        """
        try:
            h, w = frame_shape[:2]
            
            # Get calibrated camera parameters
            camera_matrix = self.calibration.get_camera_matrix(frame_shape)
            dist_coeffs = self.calibration.get_distortion_coeffs()
            
            # Extract 2D landmark points
            if hasattr(landmarks, 'landmark'):
                # MediaPipe format
                image_points = []
                for idx in self.landmark_indices:
                    if idx < len(landmarks.landmark):
                        lm = landmarks.landmark[idx]
                        image_points.append([lm.x * w, lm.y * h])
                    else:
                        logger.warning(f"Landmark index {idx} out of range")
                        return self._get_default_pose()
            else:
                # Assume already in correct format
                image_points = [landmarks[idx] for idx in self.landmark_indices if idx < len(landmarks)]
            
            if len(image_points) < 6:
                return self._get_default_pose()
            
            image_points = np.array(image_points, dtype=np.float64)
            
            # Solve PnP for head pose
            success, rotation_vector, translation_vector = cv2.solvePnP(
                self.model_points, image_points, camera_matrix, dist_coeffs
            )
            
            if not success:
                return self._get_default_pose()
            
            # Convert rotation vector to Euler angles
            rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
            
            # Extract pitch, yaw, roll (in degrees)
            sy = np.sqrt(rotation_matrix[0, 0]**2 + rotation_matrix[1, 0]**2)
            singular = sy < 1e-6
            
            if not singular:
                pitch = np.arctan2(-rotation_matrix[2, 0], sy) * 180 / np.pi
                yaw = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0]) * 180 / np.pi
                roll = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2]) * 180 / np.pi
            else:
                pitch = np.arctan2(-rotation_matrix[2, 0], sy) * 180 / np.pi
                yaw = 0
                roll = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1]) * 180 / np.pi
            
            # Calculate confidence based on pose stability and landmark quality
            confidence = self._calculate_pose_confidence(pitch, yaw, roll, image_points)
            
            return {
                'pitch': float(pitch),      # Up/down rotation
                'yaw': float(yaw),          # Left/right rotation  
                'roll': float(roll),        # Tilt rotation
                'confidence': float(confidence),
                'translation': translation_vector.flatten().tolist(),
                'is_calibrated': self.calibration.is_calibrated
            }
            
        except Exception as e:
            logger.error(f"Head pose estimation failed: {e}")
            return self._get_default_pose()
    
    def _calculate_pose_confidence(self, pitch: float, yaw: float, roll: float, 
                                 image_points: np.ndarray) -> float:
        """Calculate confidence score for head pose estimate."""
        try:
            # Base confidence
            confidence = 0.8
            
            # Reduce confidence for extreme angles
            max_angle = max(abs(pitch), abs(yaw), abs(roll))
            if max_angle > 60:
                confidence *= 0.3
            elif max_angle > 45:
                confidence *= 0.6
            elif max_angle > 30:
                confidence *= 0.8
            
            # Check landmark point distribution
            if len(image_points) == 6:
                # Points should be well distributed
                center = np.mean(image_points, axis=0)
                distances = np.linalg.norm(image_points - center, axis=1)
                if np.std(distances) < 10:  # Points too close together
                    confidence *= 0.7
            
            # Camera calibration bonus
            if self.calibration.is_calibrated:
                confidence *= 1.1
            
            return np.clip(confidence, 0.0, 1.0)
            
        except Exception:
            return 0.5
    
    def _get_default_pose(self) -> Dict[str, float]:
        """Return default pose when estimation fails."""
        return {
            'pitch': 0.0,
            'yaw': 0.0, 
            'roll': 0.0,
            'confidence': 0.0,
            'translation': [0, 0, 0],
            'is_calibrated': False
        }


# Global calibration system instance
_calibration_system = None

def get_calibration_system() -> CameraCalibrationSystem:
    """Get the global calibration system instance."""
    global _calibration_system
    if _calibration_system is None:
        _calibration_system = CameraCalibrationSystem()
    return _calibration_system

def get_head_pose_estimator() -> EnhancedHeadPoseEstimator:
    """Get head pose estimator with calibration."""
    calibration = get_calibration_system()
    return EnhancedHeadPoseEstimator(calibration)


if __name__ == "__main__":
    # Test the calibration system
    import logging
    logging.basicConfig(level=logging.INFO)
    
    calibration = get_calibration_system()
    estimator = get_head_pose_estimator()
    
    print("Camera Calibration System Test")
    print(f"Status: {calibration.get_status()}")
    
    # Test with dummy frame
    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    camera_matrix = calibration.get_camera_matrix(test_frame.shape)
    print(f"Camera matrix shape: {camera_matrix.shape}")
    print(f"Focal length: {camera_matrix[0, 0]:.1f}")
