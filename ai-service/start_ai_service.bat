@echo off
echo Starting Ders Lens AI Service...
echo ================================

cd /d "d:\YZTA\ders-lens\ai-service"

echo Checking environment...
python fix_startup.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Environment check failed. Please fix the issues above.
    pause
    exit /b 1
)

echo.
echo Starting AI Service...
python start_service.py

pause
