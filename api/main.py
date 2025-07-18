# main.py: Complete Real-time Fraud Detection API
# This script runs a FastAPI server to provide real-time fraud predictions.
# It loads a pre-trained model and uses a multi-tiered decision engine.

import joblib
import json
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, conlist
from typing import List

# 1. App Initialization and Model Loading
# Initialize the FastAPI app with a title
app = FastAPI(title="Real-time Fraud Detection API")

# Load the trained model and feature list once at startup
# The paths are now relative to the project root, where the server is run from.
try:
    model = joblib.load('artifacts/fraud_model.joblib')
    feature_names = json.load(open('artifacts/model_features.json'))
    print("Model and features loaded successfully.")
except FileNotFoundError:
    print("Error: Required model or feature file not found in the /artifacts/ directory.")
    model = None
    feature_names = []

# 2. Configuration
# Decision thresholds for the multi-tiered engine
LOW_THRESHOLD = 0.1
HIGH_THRESHOLD = 0.6

# 3. Pydantic Model for Input Validation

class Transaction(BaseModel):
    features: conlist(float, min_length=150, max_length=150)

# 4. Decision Engine Logic

def get_decision(fraud_score: float, low_threshold: float, high_threshold: float) -> str:
    """
    Determines the transaction decision based on the fraud score and defined thresholds.
    """
    if fraud_score < low_threshold:
        return "APPROVE"
    elif fraud_score > high_threshold:
        return "BLOCK"
    else:
        return "MANUAL_REVIEW"

# 5. API Endpoints
@app.get("/")
def read_root():
    """A simple health check endpoint to confirm the API is running."""
    return {"status": "ok", "message": "Welcome to the Fraud Detection API"}


@app.post("/predict")
def predict_fraud(transaction: Transaction):
    """
    Predicts the probability of fraud for a single transaction and returns a decision.
    """
    if not model:
        return {"error": "Model not loaded. Please check server logs."}

    input_df = pd.DataFrame([transaction.features], columns=feature_names)
    fraud_score = model.predict_proba(input_df)[:, 1][0]
    decision = get_decision(fraud_score, LOW_THRESHOLD, HIGH_THRESHOLD)
    
    return {
        "fraud_score": float(fraud_score),
        "decision": decision
    }