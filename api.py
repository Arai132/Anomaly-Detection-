"""FastAPI prediction endpoint serving the trained fraud-detection ensemble."""
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

MODEL_PATH = "models/fraud_pipeline.joblib"

app = FastAPI(title="Anomaly Detection API")
_artifact = None


def _load_artifact():
    global _artifact
    if _artifact is None:
        try:
            _artifact = joblib.load(MODEL_PATH)
        except FileNotFoundError:
            raise HTTPException(
                status_code=503,
                detail=f"no trained model found at {MODEL_PATH} — run benchmark_fraud.py first",
            )
    return _artifact


class PredictRequest(BaseModel):
    records: list[dict[str, float]]


class Prediction(BaseModel):
    prediction: int
    anomaly_score: float


class PredictResponse(BaseModel):
    predictions: list[Prediction]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    artifact = _load_artifact()
    scaler, model, feature_names = artifact["scaler"], artifact["model"], artifact["feature_names"]

    if not request.records:
        raise HTTPException(status_code=422, detail="records must be non-empty")

    missing_by_row = {
        i: sorted(set(feature_names) - record.keys())
        for i, record in enumerate(request.records)
        if set(feature_names) - record.keys()
    }
    if missing_by_row:
        raise HTTPException(status_code=422, detail=f"records missing required features: {missing_by_row}")

    X = pd.DataFrame(request.records, columns=feature_names)
    X_scaled = pd.DataFrame(scaler.transform(X), columns=feature_names)

    preds = model.predict(X_scaled)
    scores = model.score_samples(X_scaled)

    return PredictResponse(predictions=[
        Prediction(prediction=int(p), anomaly_score=float(s)) for p, s in zip(preds, scores)
    ])
