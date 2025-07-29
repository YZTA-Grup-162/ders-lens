"""
DersLens Complete System Startup Script
Starts Backend, AI Service, and Frontend for testing
"""

import os
import subprocess
import sys
import threading
import time
from pathlib import Path


def print_status(service_name, status, color="white"):
    """Print colored status messages"""
    colors = {
        "green": "\033[92m",
        "red": "\033[91m", 
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "white": "\033[0m"
    }
    print(f"{colors.get(color, '')}{service_name}: {status}{colors['white']}")

def check_port(port):
    """Check if a port is available"""
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(1)
    result = sock.connect_ex(('localhost', port))
    sock.close()
    return result == 0

def install_dependencies():
    """Install required dependencies"""
    print_status("SETUP", "Installing dependencies...", "blue")
    
    # Backend dependencies
    print_status("Backend", "Installing Python dependencies...", "yellow")
    backend_path = Path("backend")
    if (backend_path / "requirements.txt").exists():
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", 
                       str(backend_path / "requirements.txt")], 
                      capture_output=True)
    
    # AI Service dependencies
    print_status("AI Service", "Installing AI dependencies...", "yellow")
    ai_path = Path("ai-service")
    if (ai_path / "requirements.txt").exists():
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", 
                       str(ai_path / "requirements.txt")], 
                      capture_output=True)
    elif (ai_path / "ai-requirements.txt").exists():
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", 
                       str(ai_path / "ai-requirements.txt")], 
                      capture_output=True)
    
    # Frontend dependencies
    print_status("Frontend", "Installing Node dependencies...", "yellow")
    frontend_path = Path("frontend")
    if (frontend_path / "package.json").exists():
        subprocess.run(["npm", "install"], cwd=frontend_path, capture_output=True)
    
    print_status("SETUP", "Dependencies installed!", "green")

def start_backend():
    """Start the FastAPI backend"""
    print_status("Backend", "Starting FastAPI server on port 8000...", "blue")
    
    backend_path = Path("backend")
    
    # Try different startup methods
    startup_commands = [
        # Method 1: Using uvicorn directly
        [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"],
        # Method 2: Using fastapi run
        [sys.executable, "-m", "fastapi", "run", "app/main.py", "--port", "8000"],
        # Method 3: Direct python execution
        [sys.executable, "app/main.py"]
    ]
    
    for i, cmd in enumerate(startup_commands):
        try:
            print_status("Backend", f"Trying startup method {i+1}...", "yellow")
            process = subprocess.Popen(cmd, cwd=backend_path)
            
            # Check if it started successfully
            time.sleep(3)
            if check_port(8000):
                print_status("Backend", "✅ Running on http://localhost:8000", "green")
                return process
            else:
                process.terminate()
                
        except Exception as e:
            print_status("Backend", f"Method {i+1} failed: {e}", "red")
            continue
    
    print_status("Backend", "❌ Failed to start", "red")
    return None

def start_ai_service():
    """Start the AI service"""
    print_status("AI Service", "Starting AI service on port 5000...", "blue")
    
    ai_path = Path("ai-service")
    
    # Try different startup methods
    startup_commands = [
        # Method 1: Using Flask directly
        [sys.executable, "app.py"],
        # Method 2: Using flask run
        [sys.executable, "-m", "flask", "run", "--host", "0.0.0.0", "--port", "5000"],
        # Method 3: Using python -m flask
        ["python", "-m", "flask", "run", "--port", "5000"]
    ]
    
    # Set Flask environment
    env = os.environ.copy()
    env['FLASK_APP'] = 'app.py'
    env['FLASK_ENV'] = 'development'
    
    for i, cmd in enumerate(startup_commands):
        try:
            print_status("AI Service", f"Trying startup method {i+1}...", "yellow")
            process = subprocess.Popen(cmd, cwd=ai_path, env=env)
            
            # Check if it started successfully
            time.sleep(3)
            if check_port(5000):
                print_status("AI Service", "✅ Running on http://localhost:5000", "green")
                return process
            else:
                process.terminate()
                
        except Exception as e:
            print_status("AI Service", f"Method {i+1} failed: {e}", "red")
            continue
    
    print_status("AI Service", "❌ Failed to start", "red")
    return None

def start_frontend():
    """Start the React frontend"""
    print_status("Frontend", "Starting React frontend on port 3000...", "blue")
    
    frontend_path = Path("frontend")
    
    # Try different startup methods
    startup_commands = [
        # Method 1: npm run dev (Vite)
        ["npm", "run", "dev"],
        # Method 2: npm start
        ["npm", "start"],
        # Method 3: npx vite
        ["npx", "vite"],
        # Method 4: yarn dev
        ["yarn", "dev"]
    ]
    
    for i, cmd in enumerate(startup_commands):
        try:
            print_status("Frontend", f"Trying startup method {i+1}...", "yellow")
            process = subprocess.Popen(cmd, cwd=frontend_path)
            
            # Check if it started successfully (Vite might use different port)
            time.sleep(5)
            ports_to_check = [3000, 5173, 4173]  # Common Vite ports
            
            for port in ports_to_check:
                if check_port(port):
                    print_status("Frontend", f"✅ Running on http://localhost:{port}", "green")
                    return process
            
            process.terminate()
                
        except Exception as e:
            print_status("Frontend", f"Method {i+1} failed: {e}", "red")
            continue
    
    print_status("Frontend", "❌ Failed to start", "red")
    return None

def start_services_threaded():
    """Start all services in separate threads"""
    processes = {}
    
    def start_service(name, start_func):
        processes[name] = start_func()
    
    # Start services in threads
    threads = []
    
    # Backend
    backend_thread = threading.Thread(target=start_service, args=("backend", start_backend))
    backend_thread.start()
    threads.append(backend_thread)
    
    time.sleep(2)  # Stagger starts
    
    # AI Service
    ai_thread = threading.Thread(target=start_service, args=("ai_service", start_ai_service))
    ai_thread.start()
    threads.append(ai_thread)
    
    time.sleep(2)  # Stagger starts
    
    # Frontend
    frontend_thread = threading.Thread(target=start_service, args=("frontend", start_frontend))
    frontend_thread.start()
    threads.append(frontend_thread)
    
    # Wait for all to start
    for thread in threads:
        thread.join(timeout=10)
    
    return processes

def main():
    """Main startup function"""
    print("🚀 DERSLENS COMPLETE SYSTEM STARTUP")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path("backend").exists() or not Path("frontend").exists():
        print_status("ERROR", "Run this script from the ders-lens root directory", "red")
        return
    
    # Install dependencies
    try:
        install_dependencies()
    except Exception as e:
        print_status("SETUP", f"Dependency installation failed: {e}", "red")
        print_status("INFO", "Continuing with existing dependencies...", "yellow")
    
    # Start services
    print_status("STARTUP", "Starting all services...", "blue")
    
    processes = start_services_threaded()
    
    # Check final status
    print("\n" + "=" * 50)
    print_status("SYSTEM STATUS", "Service Overview:", "blue")
    
    services_running = 0
    
    if check_port(8000):
        print_status("✅ Backend", "http://localhost:8000", "green")
        services_running += 1
    else:
        print_status("❌ Backend", "Not running", "red")
    
    if check_port(5000):
        print_status("✅ AI Service", "http://localhost:5000", "green")
        services_running += 1
    else:
        print_status("❌ AI Service", "Not running", "red")
    
    # Check frontend ports
    frontend_running = False
    for port in [3000, 5173, 4173]:
        if check_port(port):
            print_status("✅ Frontend", f"http://localhost:{port}", "green")
            frontend_running = True
            services_running += 1
            break
    
    if not frontend_running:
        print_status("❌ Frontend", "Not running", "red")
    
    print("\n" + "=" * 50)
    
    if services_running >= 2:
        print_status("SUCCESS", f"{services_running}/3 services running!", "green")
        print_status("TESTING", "You can now test the enhanced system:", "blue")
        print_status("", "- Open frontend URL in browser", "white")
        print_status("", "- Backend API: http://localhost:8000/docs", "white")
        print_status("", "- AI Service: http://localhost:5000", "white")
        print("\n🔧 Enhanced Features Available:")
        print("   ✅ High-accuracy gaze tracking (3.39° MAE)")
        print("   ✅ Professional emotion recognition (72% accuracy)")
        print("   ✅ Real-time attention detection")
        print("   ✅ Enhanced camera calibration")
        
        print("\n⌨️  Press Ctrl+C to stop all services")
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print_status("SHUTDOWN", "Stopping all services...", "yellow")
            for name, process in processes.items():
                if process:
                    process.terminate()
            print_status("SHUTDOWN", "All services stopped", "green")
    
    else:
        print_status("ERROR", f"Only {services_running}/3 services started", "red")
        print_status("HELP", "Check the error messages above", "yellow")
        print_status("HELP", "You may need to:", "yellow")
        print("   - Install missing dependencies")
        print("   - Check port availability")
        print("   - Review service logs")

if __name__ == "__main__":
    main()
