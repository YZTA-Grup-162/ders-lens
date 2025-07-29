@echo off
echo Starting DersLens Backend (FastAPI)...
echo.

cd backend

echo Installing dependencies...
pip install -r requirements.txt

echo.
echo Starting FastAPI server on http://localhost:8000
echo API docs will be available at: http://localhost:8000/docs
echo.

REM Try different startup methods
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

pause
