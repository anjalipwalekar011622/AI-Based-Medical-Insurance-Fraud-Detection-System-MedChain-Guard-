from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
from sklearn.ensemble import IsolationForest
import joblib
import os

app = FastAPI(title="MedChain Guard API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Synthetic training data ──────────────────────────────────────────────────
# Features: [claim_amount, treatment_frequency, num_procedures,
#            patient_age,  days_since_last_claim, provider_claim_count]

NORMAL = np.array([
    [1200, 1, 2, 45, 180, 3],
    [800,  1, 1, 30,  90, 2],
    [2500, 2, 3, 60, 120, 5],
    [500,  1, 1, 25, 365, 1],
    [3000, 2, 4, 55,  60, 4],
    [1500, 1, 2, 40, 200, 3],
    [700,  1, 1, 35, 150, 2],
    [4000, 3, 5, 70,  90, 6],
    [1100, 1, 2, 50, 300, 3],
    [2200, 2, 3, 48,  75, 4],
    [900,  1, 1, 28, 400, 2],
    [3500, 2, 4, 65,  45, 5],
    [600,  1, 1, 22, 500, 1],
    [1800, 2, 3, 52, 100, 4],
    [1300, 1, 2, 38, 250, 3],
])

FRAUD = np.array([
    [85000, 10, 15, 35,   1, 50],   # massive billing
    [50000,  8, 12, 28,   2, 40],
    [1200,   9, 11, 45,   1, 35],   # high frequency
    [75000,  7, 14, 55,   3, 45],
    [2000,  10, 13, 30,   1, 60],
    [60000,  6, 10, 62,   2, 30],
])

X_train = np.vstack([NORMAL, FRAUD])

model = IsolationForest(contamination=0.25, random_state=42, n_estimators=100)
model.fit(X_train)

# ── Request schema ────────────────────────────────────────────────────────────
class Claim(BaseModel):
    claim_amount: float
    treatment_frequency: int
    num_procedures: int
    patient_age: int
    days_since_last_claim: int
    provider_claim_count: int
    patient_name: str = "Anonymous"
    treatment_type: str = "General"

# ── Helpers ───────────────────────────────────────────────────────────────────
def risk_flags(c: Claim) -> list[str]:
    flags = []
    if c.claim_amount > 20000:
        flags.append("Unusually high claim amount")
    if c.treatment_frequency > 5:
        flags.append("Abnormally high treatment frequency")
    if c.num_procedures > 8:
        flags.append("Excessive number of procedures")
    if c.days_since_last_claim < 5:
        flags.append("Claim filed within 5 days of previous claim")
    if c.provider_claim_count > 20:
        flags.append("Provider has unusually high claim volume")
    if c.claim_amount > 10000 and c.treatment_frequency > 4:
        flags.append("High cost combined with high frequency — potential duplicate billing")
    return flags

def compute_risk_score(raw_score: float) -> int:
    """Map Isolation Forest score [-0.5, 0.5] → risk 0-100"""
    clamped = max(-0.5, min(0.5, raw_score))
    return int((0.5 - clamped) * 100)

# ── Endpoint ──────────────────────────────────────────────────────────────────
@app.post("/predict")
def predict(claim: Claim):
    features = np.array([[
        claim.claim_amount,
        claim.treatment_frequency,
        claim.num_procedures,
        claim.patient_age,
        claim.days_since_last_claim,
        claim.provider_claim_count,
    ]])

    raw = model.score_samples(features)[0]
    risk_score = compute_risk_score(raw)
    prediction = model.predict(features)[0]  # -1 = anomaly, 1 = normal

    flags = risk_flags(claim)

    # Final verdict
    is_fraud = (prediction == -1) or (len(flags) >= 2) or (risk_score >= 70)
    verdict = "FRAUD" if is_fraud else "LEGITIMATE"

    return {
        "verdict": verdict,
        "risk_score": risk_score,
        "flags": flags,
        "patient_name": claim.patient_name,
        "treatment_type": claim.treatment_type,
        "claim_amount": claim.claim_amount,
        "confidence": f"{min(risk_score + 10, 99)}%" if is_fraud else f"{100 - risk_score}%",
    }

@app.get("/health")
def health():
    return {"status": "MedChain Guard is running"}

@app.get("/sample-claims")
def sample_claims():
    """Returns pre-built demo claims for quick testing"""
    return [
        {
            "label": "🟢 Routine Checkup",
            "data": {"patient_name": "Rahul Sharma", "treatment_type": "General Checkup",
                     "claim_amount": 1200, "treatment_frequency": 1, "num_procedures": 2,
                     "patient_age": 45, "days_since_last_claim": 180, "provider_claim_count": 3}
        },
        {
            "label": "🔴 Suspicious High Billing",
            "data": {"patient_name": "XYZ Clinic", "treatment_type": "Surgery",
                     "claim_amount": 85000, "treatment_frequency": 10, "num_procedures": 15,
                     "patient_age": 35, "days_since_last_claim": 1, "provider_claim_count": 50}
        },
        {
            "label": "🟡 Borderline Case",
            "data": {"patient_name": "Priya Mehta", "treatment_type": "Physiotherapy",
                     "claim_amount": 8500, "treatment_frequency": 6, "num_procedures": 7,
                     "patient_age": 55, "days_since_last_claim": 10, "provider_claim_count": 18}
        },
    ]
