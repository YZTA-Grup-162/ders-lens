#!/usr/bin/env python3
"""
Camera Utility for DersLens
Provides robust camera initialization with fallbacks
"""

import cv2
import logging

logger = logging.getLogger(__name__)

class RobustCamera:
    """Robust camera class with multiple backend fallbacks"""
    
    def __init__(self, camera_index=0, width=640, height=480, fps=15):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.fps = fps
        self.cap = None
        
    def initialize(self):
        """Initialize camera with fallback backends"""
        backends = [
            (cv2.CAP_DSHOW, "DirectShow"),
            (cv2.CAP_MSMF, "Media Foundation"),
            (cv2.CAP_ANY, "Any Backend")
        ]
        
        for backend_id, backend_name in backends:
            try:
                logger.info(f"Trying {backend_name}...")
                self.cap = cv2.VideoCapture(self.camera_index, backend_id)
                
                if self.cap.isOpened():
                    # Set camera properties
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                    self.cap.set(cv2.CAP_PROP_FPS, self.fps)
                    
                    # Test if we can actually read frames
                    ret, frame = self.cap.read()
                    if ret and frame is not None:
                        logger.info(f"Camera initialized with {backend_name}")
                        return True
                
                self.cap.release()
                self.cap = None
                
            except Exception as e:
                logger.warning(f"{backend_name} failed: {e}")
                if self.cap:
                    self.cap.release()
                    self.cap = None
        
        logger.error("All camera backends failed")
        return False
    
    def read(self):
        """Read frame from camera"""
        if self.cap and self.cap.isOpened():
            return self.cap.read()
        return False, None
    
    def release(self):
        """Release camera"""
        if self.cap:
            self.cap.release()
            self.cap = None
    
    def is_opened(self):
        """Check if camera is opened"""
        return self.cap and self.cap.isOpened()

def get_working_camera(camera_index=0):
    """Get a working camera instance"""
    camera = RobustCamera(camera_index)
    if camera.initialize():
        return camera
    return None

# For backward compatibility
def create_robust_videocapture(camera_index=0):
    """Create VideoCapture with robust backend selection"""
    backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    
    for backend in backends:
        try:
            cap = cv2.VideoCapture(camera_index, backend)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS, 15)
                
                ret, frame = cap.read()
                if ret and frame is not None:
                    return cap
            cap.release()
        except:
            continue
    
    return None
