# app/app.py
import streamlit as st
import requests
import json
import pandas as pd

#  Page Configuration 
st.set_page_config(
    page_title="Real-time Fraud Detection",
    page_icon="🛡️",
    layout="centered"
)

# API Endpoint
API_URL = "http://127.0.0.1:8000/predict"

# UI Elements 
st.title("🛡️ Real-time E-commerce Fraud Detection")
st.write(
    "This is a interactive  machine learning model to predict the likelihood of a transaction being fraudulent. "
    "Adjust the key features below to see how they impact the prediction."
)
st.markdown("---")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Transaction Details")
    transaction_amt = st.slider("Transaction Amount (USD)", 0.0, 1000.0, 150.0)
    transaction_hour = st.slider("Hour of the Day (0-23)", 0, 23, 12)

with col2:
    st.subheader("Card & User Features")
    c1 = st.slider("C1 (Anonymized Count)", 0, 100, 10)
    c13 = st.slider("C13 (Anonymized Count)", 0, 100, 25)

# Backend Communication 
try:
    with open('artifacts/model_features.json', 'r') as f:
        feature_names = json.load(f)
except FileNotFoundError:
    st.error("Error: `model_features.json` not found. Make sure the artifacts folder is one level up.")
    feature_names = []

if st.button("Check for Fraud", type="primary"):
    if feature_names:
        # Use a default vector of zeros and update it with slider values
        default_features = [0.0] * len(feature_names)
        input_df = pd.DataFrame([default_features], columns=feature_names)

        if 'TransactionAmt' in input_df.columns:
            input_df['TransactionAmt'] = transaction_amt
        if 'Transaction_hour' in input_df.columns:
            input_df['Transaction_hour'] = transaction_hour
        if 'C1' in input_df.columns:
            input_df['C1'] = c1
        if 'C13' in input_df.columns:
            input_df['C13'] = c13

        features_list = input_df.values.tolist()[0]
        payload = {"features": features_list}

        try:
            response = requests.post(API_URL, json=payload)
            response.raise_for_status()
            result = response.json()
            score = result.get("fraud_score", 0)
            decision = result.get("decision", "ERROR")

            st.markdown("---")
            st.subheader("Prediction Result")

            if decision == "BLOCK":
                st.error(f"Decision: {decision}", icon="🚫")
            elif decision == "MANUAL_REVIEW":
                st.warning(f"Decision: {decision}", icon="⚠️")
            else:
                st.success(f"Decision: {decision}", icon="✅")

            st.progress(score)
            st.metric(label="Fraud Score", value=f"{score:.4f}")

        except requests.exceptions.RequestException:
            st.error("Could not connect to the API. Please ensure the backend is running.")