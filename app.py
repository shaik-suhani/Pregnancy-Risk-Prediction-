from fastapi import FastAPI
from pydantic import BaseModel
import joblib, os
import numpy as np

MODEL_PATH = os.path.join("models", "pipeline.joblib")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Model not found. Run: python src/train.py")

bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
le = bundle["label_encoder"]
feature_order = bundle["feature_order"]

class Patient(BaseModel):
    Age: float
    BloodPressure: float
    BloodSugar: float
    Hemoglobin: float
    BMI: float
    Parity: float

app = FastAPI(title="Pregnancy Risk Predictor")

@app.get("/health")
def health():
    return {"status": "ok", "classes": list(le.classes_)}

@app.post("/predict")
def predict(p: Patient):
    # Order strictly as in training
    arr = np.array([[getattr(p, f) for f in feature_order]], dtype=float)
    pred_idx = int(model.predict(arr)[0])
    pred_label = le.inverse_transform([pred_idx])[0]
    proba = None
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(arr)[0]
        proba = {cls: float(probs[i]) for i, cls in enumerate(le.classes_)}
    return {"predicted_label": pred_label, "probabilities": proba}