import os
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

BASE = r"C:\Users\Abcom\Desktop\Work\Code Projects\ZaraERP"
OUTPUTS = os.path.join(BASE, "outputs")
DATA_PATH = os.path.join(BASE, "data", "zaraSales.csv")

# 1. Load Model, Scaler, and Columns
model = joblib.load(os.path.join(OUTPUTS, "demand_model.pkl"))
scaler = joblib.load(os.path.join(OUTPUTS, "scaler.pkl"))
with open(os.path.join(OUTPUTS, "feature_columns.json"), "r") as f:
    import json
    feature_cols = json.load(f)

# 2. Re-derive the testing set from the original data
df = pd.read_csv(DATA_PATH, sep=';')
df["Promotion"] = pd.to_numeric(df["Promotion"], errors="coerce").fillna(0)
df["Seasonal"] = pd.to_numeric(df["Seasonal"], errors="coerce").fillna(0)

# FORCE CATEGORICAL CODES (Aisle=0, End-cap=1, Front=2)
_pos_map = {"Aisle": 0, "End-cap": 1, "Front of Store": 2}
df["Product Position"] = df["Product Position"].map(_pos_map).fillna(0)

# Section Mapping (Ladies=0, Men=1)
_sec_map = {"ladieswear": 0, "menswear": 1}
df["section"] = df["section"].str.lower().map(_sec_map).fillna(0)

df["price"] = pd.to_numeric(df["price"], errors="coerce").fillna(df["price"].median())
df["price_log"] = np.log1p(df["price"])

# Synthetic Targets (Matching the 6-week recalibration)
def estimate_shelf_life(row):
    base_weeks = 6.0
    if row["Seasonal"] == 1: base_weeks = 3.0
    if row["Promotion"] == 1: base_weeks *= 0.5
    return max(1.0, base_weeks)

df["Shelf_Life_Weeks"] = df.apply(estimate_shelf_life, axis=1)
df["Sales_Volume_Weekly"] = df["Sales Volume"] / df["Shelf_Life_Weeks"]

# NLP stubs (since we are testing the demand model specifically)
df["sentiment_score"] = 0.5
df["recommend_prob"] = 0.8
df["review_volume"] = 100
df["weighted_sentiment"] = 0.4
df["price_x_sentiment"] = df["price_log"] * 0.5
df["promo_x_sentiment"] = df["Promotion"] * 0.5
df["promo_x_position"] = df["Promotion"] * df["Product Position"]
df["sentiment_x_volume"] = 0.5 * np.log1p(100)

X = df[feature_cols]
y = df["Sales_Volume_Weekly"]

# 3. Predict & Evaluate
X_scaled = scaler.transform(X)
preds = model.predict(X_scaled)

r2 = r2_score(y, preds)
rmse = np.sqrt(mean_squared_error(y, preds))

print(f"--- Model Validation Results ---")
print(f"R2 Score: {r2:.4f}")
print(f"RMSE: {rmse:.2f} units/week")
print(f"Target Mean: {y.mean():.2f}")
print(f"Prediction Mean: {preds.mean():.2f}")

# Check specific toggles
promo_indices = df[df["Promotion"] == 1].index
no_promo_indices = df[df["Promotion"] == 0].index

print(f"\n--- Feature Sensitivity ---")
print(f"Avg Prediction (Promotion=1): {preds[promo_indices].mean():.2f}")
print(f"Avg Prediction (Promotion=0): {preds[no_promo_indices].mean():.2f}")
print(f"Boost Ratio: {preds[promo_indices].mean() / preds[no_promo_indices].mean():.2f}x")
