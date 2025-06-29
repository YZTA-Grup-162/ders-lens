"""
OpenCV utilities for attention detection
"""

import math
from typing import Dict, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np


class AttentionDetector:
    """OpenCV based attention detection class"""
    
    def __init__(self):
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Initialize MediaPipe drawing utilities
        self.mp_drawing = mp.solutions.drawing_utils
        self.drawing_spec = self.mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
        
        # Face detection
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
    def detect_face(self, frame: np.ndarray) -> Tuple[bool, Optional[Tuple[int, int, int, int]]]:
        """
        Detect face in frame
        Returns: (face_detected, face_coordinates)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) > 0:
            return True, faces[0]  # Return first face
        return False, None
    
    def calculate_head_pose(self, frame: np.ndarray) -> Dict[str, float]:
        """
        Calculate head pose angles using MediaPipe
        Returns: {x_angle, y_angle, z_angle, confidence}
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        if not results.multi_face_landmarks:
            return {"x_angle": 0, "y_angle": 0, "z_angle": 0, "confidence": 0}
        
        face_landmarks = results.multi_face_landmarks[0]
        
        # Get specific landmarks for head pose calculation
        # Nose tip, chin, left ear, right ear
        h, w = frame.shape[:2]
        
        # Convert normalized coordinates to pixel coordinates
        landmarks_3d = []
        landmarks_2d = []
        
        for landmark in face_landmarks.landmark:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            landmarks_2d.append([x, y])
            landmarks_3d.append([x, y, landmark.z])
        
        # Calculate head pose (simplified)
        # This is a basic implementation - can be improved
        nose_tip = landmarks_2d[1]
        chin = landmarks_2d[175]
        
        # Calculate head tilt
        head_tilt = math.atan2(chin[1] - nose_tip[1], chin[0] - nose_tip[0])
        head_tilt_degrees = math.degrees(head_tilt)
        
        return {
            "x_angle": head_tilt_degrees,
            "y_angle": 0,  # TODO: Implement proper y-axis calculation
            "z_angle": 0,  # TODO: Implement proper z-axis calculation
            "confidence": 0.8
        }
    
    def detect_phone_usage(self, frame: np.ndarray) -> Dict[str, any]:
        
        # This is a placeholder implementation
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Simple edge detection to find rectangular objects (phones)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        phone_detected = False
        phone_area = 0
        
        for contour in contours:
            # Filter contours by area and aspect ratio
            area = cv2.contourArea(contour)
            if area > 1000:  # Minimum area threshold
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h
                
                # Phone-like aspect ratio
                if 0.4 < aspect_ratio < 0.8:
                    phone_detected = True
                    phone_area = area
                    break
        
        return {
            "phone_detected": phone_detected,
            "confidence": 0.6 if phone_detected else 0.9,
            "area": phone_area
        }
    
    def calculate_attention_score(self, 
                                face_detected: bool,
                                head_pose: Dict[str, float],
                                phone_usage: Dict[str, any],
                                eye_contact_duration: float = 0) -> Dict[str, float]:
        """
        Calculate overall attention score based on various factors
        """
        if not face_detected:
            return {
                "attention_score": 0.0,
                "engagement_score": 0.0,
                "confidence": 0.0,
                "factors": {
                    "face_detection": 0.0,
                    "head_pose": 0.0,
                    "phone_usage": 1.0,
                    "eye_contact": 0.0
                }
            }
        
        # Face detection factor (30%)
        face_factor = 1.0
        
        # Head pose factor (40%)
        head_x_angle = abs(head_pose["x_angle"])
        head_pose_factor = max(0, 1 - (head_x_angle / 45))  # Penalize head turns > 45 degrees
        
        # Phone usage factor (20%)
        phone_factor = 0.0 if phone_usage["phone_detected"] else 1.0
        
        # Eye contact factor (10%) - placeholder
        eye_contact_factor = min(1.0, eye_contact_duration / 5.0)  # Normalize to 5 seconds
        
        # Weighted attention score
        attention_score = (
            face_factor * 0.3 +
            head_pose_factor * 0.4 +
            phone_factor * 0.2 +
            eye_contact_factor * 0.1
        )
        
        # Engagement score (similar but different weights)
        engagement_score = (
            face_factor * 0.2 +
            head_pose_factor * 0.5 +
            phone_factor * 0.3
        )
        
        # Overall confidence
        confidence = (
            head_pose["confidence"] * 0.6 +
            phone_usage["confidence"] * 0.4
        )
        
        return {
            "attention_score": round(attention_score, 2),
            "engagement_score": round(engagement_score, 2),
            "confidence": round(confidence, 2),
            "factors": {
                "face_detection": face_factor,
                "head_pose": head_pose_factor,
                "phone_usage": phone_factor,
                "eye_contact": eye_contact_factor
            }
        }
    
    def process_frame(self, frame: np.ndarray) -> Dict[str, any]:
        """
        Process a single frame and return attention analysis
        """
        # Detect face
        face_detected, face_coords = self.detect_face(frame)
        
        # Calculate head pose
        head_pose = self.calculate_head_pose(frame)
        
        # Detect phone usage
        phone_usage = self.detect_phone_usage(frame)
        
        # Calculate attention score
        attention_data = self.calculate_attention_score(
            face_detected, head_pose, phone_usage
        )
        
        # Determine distraction type
        distraction_type = None
        if not face_detected:
            distraction_type = "face_not_visible"
        elif phone_usage["phone_detected"]:
            distraction_type = "phone_usage"
        elif abs(head_pose["x_angle"]) > 30:
            distraction_type = "looking_away"
        
        return {
            "face_detected": face_detected,
            "face_coordinates": face_coords,
            "head_pose": head_pose,
            "phone_usage": phone_usage,
            "attention_score": attention_data["attention_score"],
            "engagement_score": attention_data["engagement_score"],
            "confidence": attention_data["confidence"],
            "distraction_type": distraction_type,
            "factors": attention_data["factors"]
        }


# Example usage function for testing
def test_attention_detector():
    """Test function for attention detector"""
    detector = AttentionDetector()
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    print("Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Could not read frame")
            break
        
        # Process frame
        results = detector.process_frame(frame)
        
        # Draw results on frame
        if results["face_detected"]:
            x, y, w, h = results["face_coordinates"]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Display attention score
        attention_text = f"Attention: {results['attention_score']:.2f}"
        engagement_text = f"Engagement: {results['engagement_score']:.2f}"
        
        cv2.putText(frame, attention_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, engagement_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        if results["distraction_type"]:
            distraction_text = f"Distraction: {results['distraction_type']}"
            cv2.putText(frame, distraction_text, (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Show frame
        cv2.imshow('Attention Detection POC', frame)
        
        # Break on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    test_attention_detector()
