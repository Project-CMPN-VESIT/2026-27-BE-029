@echo off
title Indian Sign Language (ISL) Real-Time Recognizer
echo ======================================================================
echo           Starting Indian Sign Language (ISL) Recognizer
echo ======================================================================
echo.
cd /d "%~dp0"
"C:\Users\USER\Desktop\lakhan\Indian-Sign-Language-Translator\.venv311\Scripts\python.exe" camera_recognizer.py
echo.
echo Application stopped.
pause
