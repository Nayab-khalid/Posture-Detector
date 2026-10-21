@echo off
cd /d "%~dp0"
venv\Scripts\python.exe posture_detector.py
if errorlevel 1 pause
