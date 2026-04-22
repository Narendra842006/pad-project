from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
import torch
import torch.nn.functional as F
import numpy as np
from model import CNNAttentionAutoencoder

app = FastAPI(title="ECG Anomaly Detection API")

# Allow frontend to call this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model once at startup.
# This default threshold is calibrated for the shipped model weights and current
# per-sample min-max normalization. You can override with ECG_THRESHOLD env var.
THRESHOLD = float(os.getenv("ECG_THRESHOLD", "0.108"))

model = CNNAttentionAutoencoder()
model.load_state_dict(torch.load("cnn_attention_ae.pth", map_location=torch.device("cpu")))
model.eval()
print("Model loaded successfully")


def normalize(signal):
    mn, mx = signal.min(), signal.max()
    if mx - mn == 0:
        return signal
    return (signal - mn) / (mx - mn)


@app.get("/")
def root():
    return {"status": "ECG Anomaly Detection API is running"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        text = contents.decode("utf-8")
        lines = [l.strip() for l in text.strip().split("\n") if l.strip()]

        if not lines:
            raise HTTPException(status_code=400, detail="Empty file")

        # Take the first data row (ignore header if present)
        first_row = lines[0]
        values = [v.strip() for v in first_row.split(",")]

        # Remove label column if present (last column)
        if len(values) == 141:
            values = values[:140]
        elif len(values) != 140:
            raise HTTPException(
                status_code=400,
                detail=f"Expected 140 values, got {len(values)}. Please upload a valid ECG CSV.",
            )

        signal = np.array([float(v) for v in values], dtype=np.float32)
        signal = normalize(signal)

        # Prepare tensor: (1, 1, 140)
        tensor = torch.tensor(signal).unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            recon = model(tensor)
            error = F.mse_loss(recon, tensor).item()

        is_normal = error <= THRESHOLD
        confidence = max(0, min(100, (1 - error / (THRESHOLD * 3)) * 100))

        return {
            "prediction": "NORMAL" if is_normal else "ABNORMAL",
            "is_normal": is_normal,
            "reconstruction_error": round(error, 8),
            "threshold": THRESHOLD,
            "confidence": round(confidence if is_normal else 100 - confidence, 1),
            "signal": signal.tolist(),
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid data: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict-raw")
async def predict_raw(data: dict):
    try:
        values = data.get("values", [])
        if len(values) != 140:
            raise HTTPException(status_code=400, detail="Need exactly 140 values")

        signal = np.array(values, dtype=np.float32)
        signal = normalize(signal)
        tensor = torch.tensor(signal).unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            recon = model(tensor)
            error = F.mse_loss(recon, tensor).item()

        is_normal = error <= THRESHOLD
        confidence = max(0, min(100, (1 - error / (THRESHOLD * 3)) * 100))

        return {
            "prediction": "NORMAL" if is_normal else "ABNORMAL",
            "is_normal": is_normal,
            "reconstruction_error": round(error, 8),
            "threshold": THRESHOLD,
            "confidence": round(confidence if is_normal else 100 - confidence, 1),
            "signal": signal.tolist(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
