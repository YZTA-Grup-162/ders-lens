#!/usr/bin/env python3
"""
Startup fix script for Ders Lens AI Service
Helps resolve common startup issues
"""
import os
import sys
import subprocess
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_camera_access():
    """Check if camera is accessible"""
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            if ret:
                logger.info("✅ Camera access OK")
                return True
            else:
                logger.warning("Camera opened but can't read frames")
                return False
        else:
            logger.error("Camera cannot be opened")
            return False
    except Exception as e:
        logger.error(f"Camera check failed: {e}")
        return False

def kill_camera_processes():
    """Kill processes that might be using the camera"""
    try:
        if os.name == 'nt':  # Windows
            # Kill common processes that use camera
            processes = ['Teams.exe', 'Skype.exe', 'Zoom.exe', 'discord.exe', 'chrome.exe --use-fake-ui-for-media-stream']
            for proc in processes:
                try:
                    subprocess.run(f'taskkill /F /IM {proc}', shell=True, capture_output=True)
                except:
                    pass
            logger.info("🔄 Attempted to close camera-using processes")
        return True
    except Exception as e:
        logger.error(f"Failed to kill camera processes: {e}")
        return False

def check_dependencies():
    """Check if all dependencies are installed"""
    package_imports = {
        'opencv-python': 'cv2',
        'torch': 'torch',
        'torchvision': 'torchvision', 
        'numpy': 'numpy',
        'fastapi': 'fastapi',
        'uvicorn': 'uvicorn',
        'scikit-learn': 'sklearn',
        'joblib': 'joblib'
    }
    
    missing = []
    for package, import_name in package_imports.items():
        try:
            __import__(import_name)
            logger.info(f"✅ {package} installed")
        except ImportError:
            missing.append(package)
            logger.warning(f"{package} missing")
    
    if missing:
        logger.error(f"Missing packages: {missing}")
        logger.info("Install with: pip install " + " ".join(missing))
        return False
    return True

def main():
    logger.info("🔧 Starting Ders Lens AI Service diagnostics...")
    
    issues = []
    
    # Check dependencies
    if not check_dependencies():
        issues.append("Missing dependencies")
    
    # Check camera access
    if not check_camera_access():
        logger.info("🔄 Trying to fix camera access...")
        kill_camera_processes()
        # Wait a bit and try again
        import time
        time.sleep(2)
        if not check_camera_access():
            issues.append("Camera access failed")
    
    if issues:
        logger.error(f"Found {len(issues)} issues: {', '.join(issues)}")
        logger.info("Try restarting the computer or check camera permissions")
        return False
    else:
        logger.info("✅ All checks passed! AI Service should start properly.")
        return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
