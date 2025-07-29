@echo off
echo Starting DersLens Frontend (React + Vite)...
echo.

cd frontend

echo Installing Node.js dependencies...
npm install

echo.
echo Starting development server...
echo Frontend will be available at: http://localhost:3000 or http://localhost:5173
echo.

npm run dev

pause
