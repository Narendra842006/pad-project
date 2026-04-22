@echo off
echo Starting ECG Anomaly Detection Website...
start "backend" /D "%~dp0backend" cmd /k "pip install -r requirements.txt && uvicorn main:app --reload --port 8000"
start "frontend" /D "%~dp0frontend" cmd /k "python -m http.server 5500"
timeout /t 3 >nul
start "" http://localhost:5500
echo Done! Frontend: http://localhost:5500  Backend: http://localhost:8000
