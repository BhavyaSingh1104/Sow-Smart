"""
main.py

FastAPI backend for Sow Smart.
Run with: uvicorn api.main:app --reload --port 8000
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.recommend import get_recommendation

app = FastAPI(title="Sow Smart API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL = joblib.load("models/risk_model.joblib")
CROP_ENCODER = joblib.load("models/crop_encoder.joblib")
DISTRICT_ENCODER = joblib.load("models/district_encoder.joblib")
FEATURES_DF = pd.read_csv("data/features.csv")


class SowingRequest(BaseModel):
    district: str
    crop: str
    sow_week: int  # ISO week number, e.g. 24


@app.get("/districts")
def list_districts():
    return sorted(FEATURES_DF.district.unique().tolist())


@app.get("/crops")
def list_crops():
    return sorted(FEATURES_DF.crop.unique().tolist())


@app.post("/predict")
def predict(req: SowingRequest):
    if req.crop not in CROP_ENCODER.classes_:
        raise HTTPException(400, f"Unknown crop: {req.crop}")
    if req.district not in DISTRICT_ENCODER.classes_:
        raise HTTPException(400, f"Unknown district: {req.district}")

    # Use the district's historical average deficit as a stand-in for
    # "current season outlook" -- in production this would be replaced
    # with live IMD seasonal forecast data for the current year.
    district_rows = FEATURES_DF[FEATURES_DF.district == req.district]
    district_avg_deficit_pct = district_rows.district_avg_deficit_pct.mean()
    pre_sowing_deficit_pct = district_rows.pre_sowing_deficit_pct.mean()

    crop_encoded = CROP_ENCODER.transform([req.crop])[0]
    district_encoded = DISTRICT_ENCODER.transform([req.district])[0]

    X = pd.DataFrame([{
        "sow_week": req.sow_week,
        "pre_sowing_deficit_pct": pre_sowing_deficit_pct,
        "district_avg_deficit_pct": district_avg_deficit_pct,
        "crop_encoded": crop_encoded,
        "district_encoded": district_encoded,
    }])

    risk_score = float(MODEL.predict_proba(X)[0][1] * 100)
    recommendation = get_recommendation(
        crop=req.crop,
        risk_score=risk_score,
        pre_sowing_deficit_pct=pre_sowing_deficit_pct,
    )

    return {
        "district": req.district,
        "crop": req.crop,
        "sow_week": req.sow_week,
        **recommendation,
    }


@app.get("/health")
def health():
    return {"status": "ok"}
