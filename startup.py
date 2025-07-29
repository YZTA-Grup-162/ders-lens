#!/usr/bin/env python3
"""
AttentionPulse Startup Script
Handles dependency issues and starts the backend server
"""

import logging
import os
import subprocess
import sys
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
def fix_pytorch_dll_issue():
    """
    Uninstalls any existing PyTorch packages and reinstalls the CPU-only versions to resolve potential DLL conflicts.
    Side effects: This will remove GPU-enabled PyTorch packages and may affect environments relying on CUDA.
    """
    try:
        logger.info("🔧 Fixing PyTorch DLL issues...")
        subprocess.run([
            sys.executable, "-m", "pip", "uninstall", 
            "torch", "torchvision", "torchaudio", "-y"
        ], capture_output=True)
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", 
            "torch", "torchvision", "torchaudio", 
            "--index-url", "https://download.pytorch.org/whl/cpu"
        ], capture_output=True, text=True)
        if result.returncode == 0:
            logger.info("PyTorch CPU version installed successfully")
            return True
        else:
            #logger.warning(f"⚠️ PyTorch installation warning: {result.stderr}")
            return False
    except Exception as e:
        logger.error(f"PyTorch fix failed: {e}")
        return False
def install_dependencies():
    try:
        logger.info("📦 Installing required dependencies...")
        dependencies = [
            "mediapipe>=0.10.0",
            "opencv-python>=4.7.0",
            "numpy>=1.21.0", 
            "scikit-learn>=1.0.0",
            "fastapi>=0.100.0",
            "uvicorn>=0.20.0",
            "websockets>=11.0",
            "python-multipart",
            "aiofiles",
            "requests",
            "pillow"
        ]
        for dep in dependencies:
            try:
                result = subprocess.run([
                    sys.executable, "-m", "pip", "install", dep
                ], capture_output=True, text=True)
                if result.returncode == 0:
                    logger.info(f"Installed {dep}")
                else:
                    logger.warning(f"⚠️ Failed to install {dep}: {result.stderr}")
            except Exception as e:
                logger.warning(f"⚠️ Error installing {dep}: {e}")
        logger.info("Dependency installation completed")
        return True
    except Exception as e:
        logger.error(f"Dependency installation failed: {e}")
        return False
def test_imports():
    try:
        #logger.info("🧪 Testing critical imports...")
        import cv2
        import numpy as np
        logger.info("OpenCV and NumPy working")
        try:
            import mediapipe as mp
            logger.info("MediaPipe available")
        except ImportError:
            logger.warning("⚠️ MediaPipe not available")
        try:
            import torch
            logger.info("PyTorch available")
        except ImportError as e:
            logger.warning(f"⚠️ PyTorch not available: {e}")
        try:
            import sklearn
            logger.info("Scikit-learn available")
        except ImportError:
            logger.warning("⚠️ Scikit-learn not available")
        try:
            import fastapi
            import uvicorn
            logger.info("FastAPI and Uvicorn available")
        except ImportError:
            logger.warning("⚠️ FastAPI/Uvicorn not available")
        return True
    except Exception as e:
        #logger.error(f"Import test failed: {e}")
        return False
def start_backend():
    original_dir = os.getcwd()
    try:
        logger.info("🚀 Starting AttentionPulse backend...")
        backend_dir = os.path.join(os.path.dirname(__file__), "backend")
        if not os.path.exists(backend_dir):
            logger.error("Backend directory not found")
            return False
        
        os.chdir(backend_dir)
        try:
            logger.info("🎯 Starting main FastAPI server...")
            subprocess.run([
                sys.executable, "-m", "uvicorn", 
                "app.main:app", 
                "--host", "0.0.0.0", 
                "--port", "8000", 
                "--reload"
            ])
        except KeyboardInterrupt:
            logger.info("🛑 Server stopped by user")
        except Exception as e:
            logger.error(f"Main server failed: {e}")
            logger.info("🔄 Trying fallback server...")
            try:
                subprocess.run([
                    sys.executable, "app/main_fallback.py"
                ])
            except Exception as e2:
                logger.error(f"Fallback server also failed: {e2}")
                return False
        return True
    except Exception as e:
        logger.error(f"Backend startup failed: {e}")
        return False
    finally:
        # Restore original directory
        os.chdir(original_dir)
def main():
    print("="*60)
    print("🚀 ATTENTIONPULSE STARTUP SCRIPT")
    print("="*60)
    logger.info("Step 1: Fixing PyTorch installation...")
    pytorch_fixed = fix_pytorch_dll_issue()
    logger.info("Step 2: Installing dependencies...")
    deps_installed = install_dependencies()
    #logger.info("Step 3: Testing imports...")
    imports_ok = test_imports()
    logger.info("Step 4: Starting backend server...")
    #if not imports_ok:
        #logger.warning("⚠️ Some imports failed, but trying to start anyway...")
    print("\n" + "="*60)
    print("STARTUP SUMMARY")
    print("="*60)
    print(f"PyTorch Fix: {'SUCCESS' if pytorch_fixed else '⚠️ ISSUES'}")
    print(f"Dependencies: {'INSTALLED' if deps_installed else '⚠️ ISSUES'}")
    print(f"Imports: {'WORKING' if imports_ok else '⚠️ ISSUES'}")
    print("="*60)
    if not (pytorch_fixed or deps_installed or imports_ok):
        print("Too many issues detected. Please check Python environment.")
        input("Press Enter to continue anyway or Ctrl+C to exit...")
    start_backend()
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 Startup cancelled by user")
    except Exception as e:
        print(f"\nStartup failed: {e}")
        input("Press Enter to exit...")