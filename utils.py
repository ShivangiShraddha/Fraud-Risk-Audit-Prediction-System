import numpy as np

def risk_category(score):
    if score < 0.3:
        return "Low"
    elif score < 0.7:
        return "Medium"
    else:
        return "High"

def preprocess_dataframe(df):
    """
    Ensure dataframe matches model input format.
    Adjust here if feature engineering needed.
    """
    return df

def generate_summary(df):
    total = len(df)
    fraud_count = sum(df["Fraud_Prediction"])
    fraud_percentage = (fraud_count / total) * 100

    return total, fraud_count, fraud_percentage