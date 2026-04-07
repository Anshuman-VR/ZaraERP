import os
import json
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

# -----------------------------
# PATHS
# -----------------------------
BASE = r"C:\Users\Abcom\Desktop\Work\Code Projects\ZaraERP"

DATA_PATH = os.path.join(BASE, "data", "zaraSales.csv")
NLP_PATH = os.path.join(BASE, "outputs", "nlp_features.csv")
OUTPUT_PATH = os.path.join(BASE, "outputs")

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv(DATA_PATH, sep=';')
nlp = pd.read_csv(NLP_PATH)

# -----------------------------
# CLEAN + ALIGN KEYS
# -----------------------------
nlp = nlp.rename(columns={
    "Department Name": "section",
    "Class Name": "terms"
})

# lowercase for safe matching
df["section"] = df["section"].astype(str).str.lower()
df["terms"] = df["terms"].astype(str).str.lower()

nlp["section"] = nlp["section"].astype(str).str.lower()
nlp["terms"] = nlp["terms"].astype(str).str.lower()

# -----------------------------
# MERGE NLP FEATURES
# -----------------------------
df = df.merge(nlp, on=["section", "terms"], how="left")

# fill missing (products without reviews)
df["sentiment_score"] = df["sentiment_score"].fillna(0)
df["recommend_prob"] = df["recommend_prob"].fillna(0.5)
df["review_volume"] = df["review_volume"].fillna(0)
df["weighted_sentiment"] = df["weighted_sentiment"].fillna(0)

# -----------------------------
# FIX DATA TYPES
# -----------------------------
df["Promotion"] = pd.to_numeric(df["Promotion"], errors="coerce").fillna(0)
df["Seasonal"] = pd.to_numeric(df["Seasonal"], errors="coerce").fillna(0)

# Encode Product Position properly
df["Product Position"] = df["Product Position"].astype("category").cat.codes

# Encode section
df["section"] = df["section"].astype("category").cat.codes

# -----------------------------
# PRICE SANITY (CRITICAL FIX)
# -----------------------------
# clip extreme values to prevent absurd predictions
df["price"] = pd.to_numeric(df["price"], errors="coerce").fillna(df["price"].median())
df["price"] = df["price"].clip(lower=1, upper=df["price"].quantile(0.99))

# log transform (major stability boost)
df["price_log"] = np.log1p(df["price"])

# -----------------------------
# FEATURE ENGINEERING
# -----------------------------
df["promo_x_position"] = df["Promotion"] * df["Product Position"]

df["price_x_sentiment"] = df["price_log"] * df["sentiment_score"]
df["promo_x_sentiment"] = df["Promotion"] * df["sentiment_score"]

df["sentiment_x_volume"] = df["sentiment_score"] * np.log1p(df["review_volume"])

# -----------------------------
# SYNTHETIC TEMPORAL ENGINEERING
# -----------------------------
# The original dataset lacks a time dimension. We synthetically extrapolate 
# "Units per Week" by estimating shelf-life based on market mechanics.
def estimate_shelf_life(row):
    base_weeks = 12.0  # Recalibrated to 3-month baseline for higher velocity
    
    if row["Seasonal"] == 1:
        base_weeks = 6.0   # Seasonal items have much faster cycles (1.5 months)
        
    if row["Promotion"] == 1:
        base_weeks *= 0.6  # Promotions significantly accelerate sell-through
        
    return max(2.0, base_weeks)

df["Shelf_Life_Weeks"] = df.apply(estimate_shelf_life, axis=1)

# Convert lifetime sales volume into temporal regression target
df["Sales_Volume_Weekly"] = df["Sales Volume"] / df["Shelf_Life_Weeks"]

# Give even higher importance to promotional and seasonal items
df["Sample_Weight"] = (df["Promotion"] * 5) + (df["Seasonal"] * 3) + 1.0

# -----------------------------
# FEATURES
# -----------------------------
feature_cols = [
    "Promotion", "Seasonal",
    "price_log",
    "Product Position", "section",

    # NLP features
    "sentiment_score", "recommend_prob",
    "review_volume", "weighted_sentiment",

    # interactions
    "price_x_sentiment",
    "promo_x_sentiment",
    "promo_x_position",
    "sentiment_x_volume"
]

target = "Sales_Volume_Weekly"

X = df[feature_cols]
y = df[target]
weights = df["Sample_Weight"]

# -----------------------------
# SCALE FEATURES (IMPORTANT)
# -----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------
# TRAIN TEST SPLIT
# -----------------------------
X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
    X_scaled, y, weights, test_size=0.2, random_state=42
)

# -----------------------------
# MODEL (TUNED)
# -----------------------------
model = XGBRegressor(
    n_estimators=400,
    max_depth=6,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.5,
    reg_alpha=0.5
)

# Fit with sample weights to force the model to respect promotion/seasonal controls
model.fit(X_train, y_train, sample_weight=w_train)

# -----------------------------
# EVALUATE
# -----------------------------
preds = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, preds))
r2 = r2_score(y_test, preds)

metrics = {
    "rmse": float(rmse),
    "r2": float(r2)
}

with open(os.path.join(OUTPUT_PATH, "model_metrics.json"), "w") as f:
    json.dump(metrics, f, indent=4)

# -----------------------------
# SAVE EVERYTHING
# -----------------------------
joblib.dump(model, os.path.join(OUTPUT_PATH, "demand_model.pkl"))
joblib.dump(scaler, os.path.join(OUTPUT_PATH, "scaler.pkl"))

with open(os.path.join(OUTPUT_PATH, "feature_columns.json"), "w") as f:
    json.dump(feature_cols, f, indent=4)

print("✅ FUSION MODEL TRAINED SUCCESSFULLY")