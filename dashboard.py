import streamlit as st
import pandas as pd
import joblib
import os

from utils import risk_category, generate_summary
from logger import setup_logger

# ==========================================================
# Setup
# ==========================================================

logger = setup_logger()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "fraud_model.pkl")

model = joblib.load(MODEL_PATH)

EXPECTED_COLUMNS = [
    "transaction_amount",
    "transaction_hour",
    "account_age_months",
    "transactions_today",
    "failed_attempts",
    "is_international",
    "merchant_risk_score",
    "device_risk_score",
    "location_change",
    "avg_spend_ratio"
]

# ==========================================================
# UI
# ==========================================================

st.title("🏦 Enterprise Fraud Detection System")

uploaded_file = st.file_uploader("Upload Transaction CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Data Preview")
    st.dataframe(df.head())

    # ======================================================
    # Validate Columns
    # ======================================================

    if set(df.columns) != set(EXPECTED_COLUMNS):
        st.error("❌ Uploaded file columns do not match expected format.")
        st.write("Expected columns:")
        st.write(EXPECTED_COLUMNS)
        st.stop()

    # Enforce correct column order
    df = df[EXPECTED_COLUMNS]

    # ======================================================
    # Prediction
    # ======================================================

    probabilities = model.predict_proba(df)[:, 1]
    predictions = model.predict(df)

    df["Fraud_Prediction"] = predictions
    df["Risk_Score"] = probabilities
    df["Risk_Level"] = df["Risk_Score"].apply(risk_category)

    total, fraud_count, fraud_percentage = generate_summary(df)

    st.subheader("📊 Risk Summary")
    st.write(f"Total Transactions: {total}")
    st.write(f"Fraudulent Transactions: {fraud_count}")
    st.write(f"Fraud Percentage: {fraud_percentage:.2f}%")

    st.subheader("🚨 High Risk Transactions")
    st.dataframe(df[df["Risk_Level"] == "High"])

    logger.info(f"Processed {total} records. Fraud detected: {fraud_count}")

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download Results", csv, "fraud_results.csv", "text/csv")