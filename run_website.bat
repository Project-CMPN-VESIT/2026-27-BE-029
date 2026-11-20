@echo off
title VANI-ISL AI Web Studio
color 0B
echo ===================================================================
echo             VANI-ISL: Indian Sign Language AI Web Studio
echo ===================================================================
echo.
echo Starting FastAPI High-Performance WebSocket Backend Server...
echo Opening Web Application in your default browser...
echo.
echo URL: http://localhost:8000
echo.
echo Press Ctrl+C in this console to stop the web server.
echo ===================================================================
echo.

start "" "http://localhost:8000"

cd /d "%~dp0web"
"C:\Users\USER\Desktop\lakhan\Indian-Sign-Language-Translator\.venv311\Scripts\python.exe" -m uvicorn app:app --host 127.0.0.1 --port 8000

pause
