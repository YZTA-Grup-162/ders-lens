
import math
import time
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
import onnxruntime as ort
import torch

from gemini_integration import GeminiIntegrationManager


class AttentionState(Enum):
    """Attention state classifications"""
    HIGHLY_FOCUSED = "highly_focused"
    FOCUSED = "focused"
    NEUTRAL = "neutral"
    DISTRACTED = "distracted"
    HIGHLY_DISTRACTED = "highly_distracted"

@dataclass
class GazeData:
    """Gaze tracking results"""
    gaze_vector: Tuple[float, float, float]
    gaze_angles: Tuple[float, float]  # yaw, pitch
    on_screen: bool
    confidence: float

@dataclass
class HeadPoseData:
    """Head pose estimation results"""
    rotation_vector: Tuple[float, float, float]
    translation_vector: Tuple[float, float, float]
    euler_angles: Tuple[float, float, float]  # roll, pitch, yaw

@dataclass
class EmotionData:
    """Emotion recognition results"""
    emotions: Dict[str, float]
    primary_emotion: str
    confidence: float

@dataclass
class EngagementMetrics:
    """Complete engagement analysis"""
    attention_score: float  # 0-100
    gaze_data: GazeData
    head_pose: HeadPoseData
    emotion_data: EmotionData
    attention_state: AttentionState
    timestamp: float

class EnhancedComputerVision:
    """Enhanced CV pipeline with MediaPipe and trained models"""
    
    def __init__(self, 
                 fer_model_path: str = "models_fer2013/fer2013_pytorch.onnx",
                 use_gpu: bool = True):
        
        # Initialize MediaPipe
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_pose = mp.solutions.pose
        self.mp_hands = mp.solutions.hands
        
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Initialize emotion model
        self.emotion_model = None
        if fer_model_path and self._load_emotion_model(fer_model_path):
            print(f"✅ Loaded emotion model: {fer_model_path}")
        else:
            print("⚠️ Emotion model not loaded - using fallback detection")
        
        # FER2013 emotion labels
        self.emotion_labels = [
            'Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise'
        ]
        
        # Calibration parameters
        self.screen_width = 1920  # Default screen width
        self.screen_height = 1080  # Default screen height
        self.camera_matrix = None
        self.dist_coeffs = None
        
        # Attention tracking
        self.attention_history = []
        self.max_history = 30  # Keep last 30 frames (1 second at 30fps)
        
    def _load_emotion_model(self, model_path: str) -> bool:
        """Load ONNX emotion recognition model"""
        try:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            self.emotion_model = ort.InferenceSession(model_path, providers=providers)
            return True
        except Exception as e:
            print(f"Failed to load emotion model: {e}")
            return False
    
    def set_screen_dimensions(self, width: int, height: int):
        """Set screen dimensions for gaze estimation"""
        self.screen_width = width
        self.screen_height = height
    
    def calibrate_camera(self, frame: np.ndarray) -> bool:
        """Simple camera calibration (basic implementation)"""
        # For production, use proper camera calibration
        h, w = frame.shape[:2]
        
        # Approximate camera matrix (you should calibrate properly)
        self.camera_matrix = np.array([
            [w, 0, w/2],
            [0, w, h/2],
            [0, 0, 1]
        ], dtype=np.float32)
        
        self.dist_coeffs = np.zeros((4, 1))  # Assume no distortion
        return True
    
    def extract_face_landmarks(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Extract 468 face landmarks using MediaPipe"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            h, w = frame.shape[:2]
            
            # Convert normalized coordinates to pixel coordinates
            face_landmarks = []
            for landmark in landmarks.landmark:
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                z = landmark.z
                face_landmarks.append([x, y, z])
            
            return np.array(face_landmarks)
        
        return None
    
    def estimate_head_pose(self, landmarks: np.ndarray, frame_shape: Tuple[int, int]) -> HeadPoseData:
        """Estimate head pose from facial landmarks"""
        
        if self.camera_matrix is None:
            self.calibrate_camera(np.zeros((*frame_shape, 3), dtype=np.uint8))
        
        # 3D model points for key facial features
        model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Left eye left corner
            (225.0, 170.0, -135.0),      # Right eye right corner
            (-150.0, -150.0, -125.0),    # Left mouth corner
            (150.0, -150.0, -125.0)      # Right mouth corner
        ], dtype=np.float32)
        
        # 2D image points from landmarks
        image_points = np.array([
            landmarks[1],      # Nose tip
            landmarks[152],    # Chin
            landmarks[226],    # Left eye left corner
            landmarks[446],    # Right eye right corner
            landmarks[57],     # Left mouth corner
            landmarks[287]     # Right mouth corner
        ], dtype=np.float32)[:, :2]  # Only x, y coordinates
        
        # Solve PnP
        success, rotation_vector, translation_vector = cv2.solvePnP(
            model_points, image_points, self.camera_matrix, self.dist_coeffs
        )
        
        if success:
            # Convert rotation vector to euler angles
            rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
            euler_angles = self._rotation_matrix_to_euler_angles(rotation_matrix)
            
            return HeadPoseData(
                rotation_vector=tuple(rotation_vector.flatten()),
                translation_vector=tuple(translation_vector.flatten()),
                euler_angles=euler_angles
            )
        
        return HeadPoseData((0, 0, 0), (0, 0, 0), (0, 0, 0))
    
    def _rotation_matrix_to_euler_angles(self, R: np.ndarray) -> Tuple[float, float, float]:
        """Convert rotation matrix to euler angles"""
        sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
        
        singular = sy < 1e-6
        
        if not singular:
            x = math.atan2(R[2,1], R[2,2])
            y = math.atan2(-R[2,0], sy)
            z = math.atan2(R[1,0], R[0,0])
        else:
            x = math.atan2(-R[1,2], R[1,1])
            y = math.atan2(-R[2,0], sy)
            z = 0
        
        return (math.degrees(x), math.degrees(y), math.degrees(z))
    
    def estimate_gaze(self, landmarks: np.ndarray, head_pose: HeadPoseData) -> GazeData:
        """Estimate gaze direction using eye landmarks and head pose"""
        
        # Get eye landmarks
        left_eye_landmarks = landmarks[362:374]  # Left eye region
        right_eye_landmarks = landmarks[33:42]   # Right eye region
        
        # Calculate eye centers
        left_eye_center = np.mean(left_eye_landmarks[:, :2], axis=0)
        right_eye_center = np.mean(right_eye_landmarks[:, :2], axis=0)
        
        # Estimate gaze vector (simplified)
        # In production, use more sophisticated gaze estimation
        eye_center = (left_eye_center + right_eye_center) / 2
        
        # Combine with head pose for gaze direction
        roll, pitch, yaw = head_pose.euler_angles
        
        # Adjust gaze based on head pose
        gaze_yaw = yaw + (eye_center[0] - landmarks[1][0]) * 0.1
        gaze_pitch = pitch + (eye_center[1] - landmarks[1][1]) * 0.1
        
        # Convert to 3D gaze vector
        gaze_vector = (
            math.sin(math.radians(gaze_yaw)),
            math.sin(math.radians(gaze_pitch)),
            math.cos(math.radians(gaze_yaw)) * math.cos(math.radians(gaze_pitch))
        )
        
        # Check if looking at screen (simplified)
        on_screen = abs(gaze_yaw) < 25 and abs(gaze_pitch) < 20
        
        # Calculate confidence based on head pose stability
        confidence = max(0, 1 - (abs(yaw) + abs(pitch)) / 90)
        
        return GazeData(
            gaze_vector=gaze_vector,
            gaze_angles=(gaze_yaw, gaze_pitch),
            on_screen=on_screen,
            confidence=confidence
        )
    
    def recognize_emotion(self, frame: np.ndarray, face_bbox: Tuple[int, int, int, int]) -> EmotionData:
        """Recognize emotion from face region"""
        
        if self.emotion_model is None:
            return self._fallback_emotion_detection()
        
        try:
            # Extract face region
            x, y, w, h = face_bbox
            face_region = frame[y:y+h, x:x+w]
            
            # Preprocess for emotion model
            face_resized = cv2.resize(face_region, (48, 48))
            face_gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
            face_normalized = face_gray.astype(np.float32) / 255.0
            
            # Convert to RGB and normalize like training
            face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
            face_input = face_rgb.astype(np.float32) / 255.0
            
            # Normalize with ImageNet stats (same as training)
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            face_input = (face_input - mean) / std
            
            # Add batch dimension and transpose to CHW
            face_input = np.transpose(face_input, (2, 0, 1))
            face_input = np.expand_dims(face_input, axis=0)
            
            # Run inference
            input_name = self.emotion_model.get_inputs()[0].name
            outputs = self.emotion_model.run(None, {input_name: face_input})
            predictions = outputs[0][0]
            
            # Apply softmax
            exp_preds = np.exp(predictions - np.max(predictions))
            probabilities = exp_preds / np.sum(exp_preds)
            
            # Create emotion dictionary
            emotions = {
                self.emotion_labels[i]: float(probabilities[i]) 
                for i in range(len(self.emotion_labels))
            }
            
            # Get primary emotion
            primary_idx = np.argmax(probabilities)
            primary_emotion = self.emotion_labels[primary_idx]
            confidence = float(probabilities[primary_idx])
            
            return EmotionData(
                emotions=emotions,
                primary_emotion=primary_emotion,
                confidence=confidence
            )
            
        except Exception as e:
            print(f"Emotion recognition error: {e}")
            return self._fallback_emotion_detection()
    
    def _fallback_emotion_detection(self) -> EmotionData:
        """Fallback emotion detection when model fails"""
        return EmotionData(
            emotions={label: 1.0/len(self.emotion_labels) for label in self.emotion_labels},
            primary_emotion="Neutral",
            confidence=0.5
        )
    
    def calculate_attention_score(self, 
                                gaze_data: GazeData, 
                                head_pose: HeadPoseData, 
                                emotion_data: EmotionData) -> float:
        """Calculate comprehensive attention score (0-100)"""
        
        score = 50.0  # Base score
        
        # Gaze contribution (40% of score)
        if gaze_data.on_screen:
            score += 20 * gaze_data.confidence
        else:
            score -= 15
        
        # Head pose contribution (30% of score)
        roll, pitch, yaw = head_pose.euler_angles
        head_stability = max(0, 1 - (abs(yaw) + abs(pitch)) / 60)
        score += 15 * head_stability
        
        # Emotion contribution (30% of score)
        positive_emotions = ['Happy', 'Neutral']
        negative_emotions = ['Angry', 'Sad', 'Fear']
        focus_emotions = ['Neutral']  # Neutral often indicates focus
        
        emotion_score = 0
        for emotion, prob in emotion_data.emotions.items():
            if emotion in focus_emotions:
                emotion_score += prob * 15
            elif emotion in positive_emotions:
                emotion_score += prob * 10
            elif emotion in negative_emotions:
                emotion_score -= prob * 10
        
        score += emotion_score
        
        return max(0, min(100, score))
    
    def determine_attention_state(self, attention_score: float) -> AttentionState:
        """Determine attention state from score"""
        if attention_score >= 85:
            return AttentionState.HIGHLY_FOCUSED
        elif attention_score >= 70:
            return AttentionState.FOCUSED
        elif attention_score >= 50:
            return AttentionState.NEUTRAL
        elif attention_score >= 30:
            return AttentionState.DISTRACTED
        else:
            return AttentionState.HIGHLY_DISTRACTED
    
    def get_face_bbox_from_landmarks(self, landmarks: np.ndarray) -> Tuple[int, int, int, int]:
        """Get bounding box from face landmarks"""
        x_coords = landmarks[:, 0]
        y_coords = landmarks[:, 1]
        
        x_min, x_max = int(np.min(x_coords)), int(np.max(x_coords))
        y_min, y_max = int(np.min(y_coords)), int(np.max(y_coords))
        
        # Add some padding
        padding = 10
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        
        width = x_max - x_min + 2 * padding
        height = y_max - y_min + 2 * padding
        
        return (x_min, y_min, width, height)
    
    def process_frame(self, frame: np.ndarray) -> Optional[EngagementMetrics]:
        """Process single frame and return complete engagement metrics"""
        
        # Extract face landmarks
        landmarks = self.extract_face_landmarks(frame)
        if landmarks is None:
            return None
        
        # Get face bounding box
        face_bbox = self.get_face_bbox_from_landmarks(landmarks)
        
        # Estimate head pose
        head_pose = self.estimate_head_pose(landmarks, frame.shape[:2])
        
        # Estimate gaze
        gaze_data = self.estimate_gaze(landmarks, head_pose)
        
        # Recognize emotion
        emotion_data = self.recognize_emotion(frame, face_bbox)
        
        # Calculate attention score
        attention_score = self.calculate_attention_score(gaze_data, head_pose, emotion_data)
        
        # Determine attention state
        attention_state = self.determine_attention_state(attention_score)
        
        # Create engagement metrics
        metrics = EngagementMetrics(
            attention_score=attention_score,
            gaze_data=gaze_data,
            head_pose=head_pose,
            emotion_data=emotion_data,
            attention_state=attention_state,
            timestamp=time.time()
        )
        
        # Update history
        self.attention_history.append(metrics)
        if len(self.attention_history) > self.max_history:
            self.attention_history.pop(0)
        
        return metrics
    
    def get_smoothed_attention_score(self, window_size: int = 5) -> float:
        """Get smoothed attention score over recent frames"""
        if len(self.attention_history) < window_size:
            return self.attention_history[-1].attention_score if self.attention_history else 50.0
        
        recent_scores = [m.attention_score for m in self.attention_history[-window_size:]]
        return sum(recent_scores) / len(recent_scores)
    
    def draw_debug_info(self, frame: np.ndarray, metrics: EngagementMetrics) -> np.ndarray:
        """Draw debug information on frame"""
        debug_frame = frame.copy()
        
        # Draw attention score
        cv2.putText(debug_frame, f"Attention: {metrics.attention_score:.1f}%", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Draw attention state
        state_color = {
            AttentionState.HIGHLY_FOCUSED: (0, 255, 0),
            AttentionState.FOCUSED: (0, 200, 0),
            AttentionState.NEUTRAL: (0, 255, 255),
            AttentionState.DISTRACTED: (0, 165, 255),
            AttentionState.HIGHLY_DISTRACTED: (0, 0, 255)
        }
        
        color = state_color.get(metrics.attention_state, (255, 255, 255))
        cv2.putText(debug_frame, metrics.attention_state.value.replace('_', ' ').title(), 
                   (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # Draw primary emotion
        cv2.putText(debug_frame, f"Emotion: {metrics.emotion_data.primary_emotion}", 
                   (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
        
        # Draw gaze status
        gaze_status = "On Screen" if metrics.gaze_data.on_screen else "Off Screen"
        gaze_color = (0, 255, 0) if metrics.gaze_data.on_screen else (0, 0, 255)
        cv2.putText(debug_frame, f"Gaze: {gaze_status}", 
                   (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, gaze_color, 2)
        
        return debug_frame

class DersLensEnhancedPipeline:
    """Complete DersLens pipeline with Gemini integration"""
    
    def __init__(self, gemini_api_key: str, fer_model_path: str = None):
        self.cv_pipeline = EnhancedComputerVision(fer_model_path)
        self.gemini_manager = GeminiIntegrationManager(_api_key)
        self.frame_count = 0
        self.analysis_interval = 5  # Analyze with Gemini every 5 frames
        
    def process_video_frame(self, frame: np.ndarray, analyze_with_gemini: bool = None) -> Dict:
        """Process video frame with complete pipeline"""
        
        self.frame_count += 1
        
        # Get CV metrics
        cv_metrics = self.cv_pipeline.process_frame(frame)
        
        if cv_metrics is None:
            return {"error": "No face detected", "frame_count": self.frame_count}
        
        # Convert metrics to dictionary for Gemini
        cv_results = {
            "attention_score": cv_metrics.attention_score,
            "attention_state": cv_metrics.attention_state.value,
            "emotions": cv_metrics.emotion_data.emotions,
            "primary_emotion": cv_metrics.emotion_data.primary_emotion,
            "gaze": {
                "on_screen": cv_metrics.gaze_data.on_screen,
                "confidence": cv_metrics.gaze_data.confidence,
                "angles": cv_metrics.gaze_data.gaze_angles
            },
            "head_pose": {
                "euler_angles": cv_metrics.head_pose.euler_angles
            }
        }
        
        # Decide whether to use Gemini analysis
        if analyze_with_gemini is None:
            analyze_with_gemini = (self.frame_count % self.analysis_interval == 0)
        
        result = {
            "frame_count": self.frame_count,
            "cv_results": cv_results,
            "gemini_analysis": None,
            "debug_frame": None
        }
        
        # Add Gemini analysis if requested
        if analyze_with_gemini:
            try:
                import asyncio

                # Run Gemini analysis
                gemini_analysis = asyncio.run(
                    self.gemini_manager.process_frame_analysis(frame, cv_results)
                )
                result["gemini_analysis"] = gemini_analysis
                
            except Exception as e:
                result["gemini_error"] = str(e)
        
        # Add debug visualization
        result["debug_frame"] = self.cv_pipeline.draw_debug_info(frame, cv_metrics)
        
        return result
    
    async def handle_teacher_query(self, query: str) -> Dict:
        """Handle teacher dashboard queries"""
        return await self.gemini_manager.handle_teacher_query(query)
    
    def get_session_summary(self) -> Dict:
        """Get comprehensive session summary"""
        return self.gemini_manager.get_session_summary()

if __name__ == "__main__":
    # Example usage
    import os

    # Setup
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    fer_model_path = "models_fer2013/fer2013_pytorch.onnx"
    
    if gemini_api_key:
        # Initialize pipeline
        pipeline = DersLensEnhancedPipeline(gemini_api_key, fer_model_path)
        
        # Test with webcam
        cap = cv2.VideoCapture(0)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            result = pipeline.process_video_frame(frame)
            
            # Display debug frame
            if "debug_frame" in result:
                cv2.imshow("DersLens Enhanced", result["debug_frame"])
            
            # Print insights
            if result.get("gemini_analysis"):
                insights = result["gemini_analysis"].get("gemini_insights", {})
                if insights:
                    print(f"Gemini Insights: {insights}")
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
    else:
        print("Please set GEMINI_API_KEY environment variable")
