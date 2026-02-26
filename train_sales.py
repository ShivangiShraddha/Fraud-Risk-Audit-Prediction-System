import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(BASE_DIR, "data", "Online Retail.xlsx")
model_path = os.path.join(BASE_DIR, "models", "sales_model.pkl")

# Load data
df = pd.read_excel(data_path)

df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
df['Revenue'] = df['Quantity'] * df['UnitPrice']

monthly = df.resample('M', on='InvoiceDate')['Revenue'].sum().reset_index()

X = [[i] for i in range(len(monthly))]
y = monthly['Revenue']

model = LinearRegression()
model.fit(X, y)

joblib.dump(model, model_path)

print("Sales forecasting model saved successfully!")