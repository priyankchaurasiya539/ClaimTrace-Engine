import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(
    title="ClaimTrace Engine API",
    description="Unified Industrial Fraud & Anomaly Audit Gateway",
    version="4.0.0",
)

# Load Artifacts
try:
    preprocessor = joblib.load("models/preprocessor.joblib")
    clf = joblib.load("models/logistic_model.joblib")
    iso_forest = joblib.load("models/anomaly_model.joblib")
    scaler_num = joblib.load("models/scaler_num.joblib")
    anomaly_threshold = joblib.load("models/anomaly_threshold.joblib")
except Exception as e:
    raise RuntimeError(f"Model Loading Error: {str(e)}")


class ClaimInput(BaseModel):
    claim_amount: float = Field(..., example=4500.0)
    vehicle_age_months: int = Field(..., example=14)
    mileage_at_claim: float = Field(..., example=18000.0)
    past_claims_count: int = Field(..., example=0)
    parts_replaced_count: int = Field(..., example=1)
    mechanic_tenure_months: int = Field(..., example=36)
    region: str = Field(..., example="North")
    dealer_type: str = Field(..., example="Authorized")
    part_category: str = Field(..., example="Engine")
    mechanic_notes: str = Field(..., example="Routine scheduled maintenance.")


class ClaimResponse(BaseModel):
    verdict: str
    fraud_probability: float
    is_anomaly: bool
    anomaly_score: float
    risk_tier: str
    action_notes: str
    status: str


@app.get("/")
def health_check():
    return {"status": "ClaimTrace Engine Operational"}


@app.post("/predict", response_model=ClaimResponse)
def predict_claim(claim: ClaimInput):
    try:
        input_df = pd.DataFrame(
            [
                {
                    "claim_amount": float(claim.claim_amount),
                    "vehicle_age_months": int(claim.vehicle_age_months),
                    "mileage_at_claim": float(claim.mileage_at_claim),
                    "past_claims_count": int(claim.past_claims_count),
                    "parts_replaced_count": int(claim.parts_replaced_count),
                    "mechanic_tenure_months": int(claim.mechanic_tenure_months),
                    "region": str(claim.region),
                    "dealer_type": str(claim.dealer_type),
                    "part_category": str(claim.part_category),
                    "mechanic_notes": str(claim.mechanic_notes),
                }
            ]
        )

        # 1. Supervised Fraud Probability
        X_trans = preprocessor.transform(input_df)
        fraud_prob = float(clf.predict_proba(X_trans)[0][1])

        # 2. Unsupervised Anomaly Detection on Numeric Metrics
        num_cols = [
            "claim_amount",
            "vehicle_age_months",
            "mileage_at_claim",
            "past_claims_count",
            "parts_replaced_count",
            "mechanic_tenure_months",
        ]
        X_num_scaled = scaler_num.transform(input_df[num_cols])
        anomaly_score = float(iso_forest.decision_function(X_num_scaled)[0])
        is_anomaly = bool(anomaly_score <= anomaly_threshold)

        # 3. HARMONIZED ENSEMBLE DECISION LOGIC (No Conflicts)
        if fraud_prob >= 0.70:
            verdict = "FRAUD DETECTED"
            risk_tier = "High Risk / Immediate Audit"
            action_notes = (
                "Severe pattern deviation and high fraudulent probability."
            )
        elif fraud_prob >= 0.40 or is_anomaly:
            verdict = "SUSPICIOUS / AUDIT"
            risk_tier = "Medium Risk / Audit Flag"
            action_notes = (
                "Borderline statistical score or structural feature anomaly."
            )
        else:
            verdict = "GENUINE CLAIM"
            risk_tier = "Low Risk Tier"
            action_notes = "Standard operational profile."

        return ClaimResponse(
            verdict=verdict,
            fraud_probability=round(fraud_prob, 4),
            is_anomaly=is_anomaly,
            anomaly_score=round(anomaly_score, 4),
            risk_tier=risk_tier,
            action_notes=action_notes,
            status="Success",
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Inference Error: {str(e)}"
        )