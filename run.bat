@echo off
echo Starting ECG Anomaly Detection Website...
cd backend
start cmd /k "pip install -r requirements.txt && uvicorn main:app --reload --port 8000"
timeout /t 3
start "" ..\frontend\index.html
echo Done! Website opening in browser...
