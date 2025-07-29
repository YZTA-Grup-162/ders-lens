@echo off
echo Starting DersLens AI Service (Flask)...
echo.

cd ai-service

echo Installing AI dependencies...
if exist ai-requirements.txt (
    pip install -r ai-requirements.txt
) else if exist requirements.txt (
    pip install -r requirements.txt
) else (
    echo Installing basic AI dependencies...
    pip install flask flask-cors opencv-python mediapipe torch numpy scikit-learn
)

echo.
echo Starting AI Service on http://localhost:5000
echo Enhanced models available:
echo   - MPIIGaze: 3.39° MAE gaze accuracy
echo   - FER2013: 72%% emotion accuracy
echo.

set FLASK_APP=app.py
set FLASK_ENV=development
python app.py

pause
