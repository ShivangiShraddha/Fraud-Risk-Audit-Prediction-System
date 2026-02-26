import pandas as pd
import numpy as np
import joblib
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ==========================================================
# 1. Setup Base Directory (Corporate Safe Path Handling)
# ==========================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "fraud_model.pkl")

# Ensure models folder exists
os.makedirs(MODEL_DIR, exist_ok=True)

# ==========================================================
# 2. Generate Corporate Synthetic Fraud Dataset
# ==========================================================

np.random.seed(42)
n_samples = 5000

data = pd.DataFrame({
    "transaction_amount": np.random.uniform(10, 20000, n_samples),
    "transaction_hour": np.random.randint(0, 24, n_samples),
    "account_age_months": np.random.randint(1, 120, n_samples),
    "transactions_today": np.random.randint(1, 20, n_samples),
    "failed_attempts": np.random.randint(0, 6, n_samples),
    "is_international": np.random.randint(0, 2, n_samples),
    "merchant_risk_score": np.random.uniform(0, 1, n_samples),
    "device_risk_score": np.random.uniform(0, 1, n_samples),
    "location_change": np.random.randint(0, 2, n_samples),
    "avg_spend_ratio": np.random.uniform(0, 3, n_samples)
})

# ==========================================================
# 3. Inject Fraud Logic (Realistic Corporate Pattern)
# ==========================================================

data["fraud"] = (
    (data["transaction_amount"] > 10000) &
    (data["is_international"] == 1) &
    (data["failed_attempts"] > 2)
).astype(int)

# ==========================================================
# 4. Train/Test Split
# ==========================================================

X = data.drop("fraud", axis=1)
y = data["fraud"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ==========================================================
# 5. Train Model
# ==========================================================

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

# ==========================================================
# 6. Evaluate Model
# ==========================================================

y_pred = model.predict(X_test)

print("\n📊 Model Evaluation Report:\n")
print(classification_report(y_test, y_pred))

# ==========================================================
# 7. Save Model (Corporate Safe)
# ==========================================================

joblib.dump(model, MODEL_PATH)

print(f"\n✅ Corporate Fraud Model Saved Successfully!")
print(f"📁 Saved At: {MODEL_PATH}")