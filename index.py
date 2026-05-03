from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict
import joblib
import json
import pandas as pd
import numpy as np

MODEL_PATH = "xgb_diabetes_model.joblib"
SCALER_PATH = "scaler.joblib"
METADATA_PATH = "model_metadata.json"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

with open(METADATA_PATH, "r") as f:
    metadata = json.load(f)

FEATURE_ORDER = metadata["feature_order"]
MEDIANS = metadata["median_values"]
INVALID_ZERO_FEATURES = metadata["invalid_zero_features"]
THRESHOLD = metadata.get("decision_threshold", 0.5)

app = FastAPI(
    title="PIMA Diabetes Prediction API",
    description="Educational diabetes risk prediction API using XGBoost. Not for clinical diagnosis.",
    version="1.0"
)

class PatientFeatures(BaseModel):
    Pregnancies: int = Field(..., ge=0, le=20, description="Number of pregnancies")
    Glucose: float = Field(..., ge=40, le=250, description="Plasma glucose concentration")
    BloodPressure: float = Field(..., ge=40, le=140, description="Diastolic blood pressure")
    SkinThickness: float = Field(..., ge=1, le=100, description="Triceps skin fold thickness")
    Insulin: float = Field(..., ge=1, le=900, description="2-hour serum insulin")
    BMI: float = Field(..., ge=10, le=70, description="Body mass index")
    DiabetesPedigreeFunction: float = Field(..., ge=0.0, le=3.0, description="Genetic/family risk score")
    Age: int = Field(..., ge=18, le=100, description="Patient age")

def preprocess(payload: PatientFeatures):
    row = payload.model_dump()
    df = pd.DataFrame([row], columns=FEATURE_ORDER)
    for col in INVALID_ZERO_FEATURES:
        df[col] = df[col].replace(0, np.nan)
    df = df.fillna(MEDIANS)
    return scaler.transform(df)

@app.get("/")
def health_check() -> Dict[str, str]:
    return {
        "status": "ok",
        "message": "Use POST /predict with patient features."
    }

@app.post("/predict")
def predict(payload: PatientFeatures):
    X = preprocess(payload)
    probability = float(model.predict_proba(X)[0, 1])
    prediction = int(probability >= THRESHOLD)
    risk_label = "High diabetes risk" if prediction == 1 else "Low diabetes risk"
    return {
        "prediction": prediction,
        "risk_label": risk_label,
        "confidence": round(probability if prediction == 1 else 1 - probability, 4),
        "diabetes_probability": round(probability, 4),
        "threshold": THRESHOLD,
        "disclaimer": "Educational model only. It supports clinician review and must not replace medical diagnosis."
    }
