@echo off
REM ====================================================================
REM CycloneAI — Windows Production Server Launcher
REM Smart India Hackathon 2026
REM Launches unified FastAPI application on http://localhost:8000
REM ====================================================================

echo ====================================================================
echo Starting CycloneAI Production Server (SIH 2026 Prototype)
echo ====================================================================

set PROJECT_ROOT=%~dp0
cd /d "%PROJECT_ROOT%"

set PYTHON_EXEC="%PROJECT_ROOT%backend\.venv\Scripts\python.exe"

if not exist %PYTHON_EXEC% (
    echo [ERROR] Virtual environment not found at %PYTHON_EXEC%
    echo Please ensure the backend virtual environment is created.
    pause
    exit /b 1
)

echo [INFO] Launching unified production server...
%PYTHON_EXEC% run_production.py

pause
