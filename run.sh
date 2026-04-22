#!/bin/bash
echo "Starting ECG Anomaly Detection Website..."
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000 &
cd ../frontend
python3 -m http.server 5500 &
sleep 2
echo "Done! Frontend: http://localhost:5500  Backend: http://localhost:8000"
