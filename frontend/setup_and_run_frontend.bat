@echo off
REM === setup_and_run_frontend.bat ===
REM Installs dependencies and runs the Vite dev server for the frontend

REM Step 1: Check for Node.js
where node >nul 2>&1
if errorlevel 1 (
    echo Node.js is not installed or not in PATH. Please install Node.js from https://nodejs.org/
    pause
    exit /b 1
)

REM Step 2: Install dependencies
if exist node_modules (
    echo node_modules already exists. Skipping npm install.
) else (
    echo Installing npm dependencies...
    call npm install
    if errorlevel 1 (
        echo npm install failed. Please check for errors above.
        pause
        exit /b 1
    )
)

REM Step 3: Start the Vite dev server
call npm run dev

REM Keep window open if server fails
if errorlevel 1 (
    echo Frontend dev server failed to start.
    pause
) 