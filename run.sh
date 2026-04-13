#!/bin/bash
echo "Starting ECG Anomaly Detection Website..."
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000 &
sleep 2
open ../frontend/index.html    # Mac
# xdg-open ../frontend/index.html  # Linux
echo "Done! Website open in browser."
