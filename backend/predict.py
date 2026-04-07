import os
import json
import joblib
import numpy as np

BASE = r"C:\Users\Abcom\Desktop\Work\Code Projects\ZaraERP"
OUTPUT_PATH = os.path.join(BASE, "outputs")

# -----------------------------
# LOAD MODEL
# -----------------------------
model = joblib.load(os.path.join(OUTPUT_PATH, "demand_model.pkl"))
scaler = joblib.load(os.path.join(OUTPUT_PATH, "scaler.pkl"))

with open(os.path.join(OUTPUT_PATH, "feature_columns.json"), "r") as f:
    feature_cols = json.load(f)

# -----------------------------
# MANUAL INPUT (CHANGE HERE)
# -----------------------------
sample_input = {
    "Promotion": 0,
    "Seasonal": 1,
    "price": 250,   # change this for testing
    "Product Position": 0,
    "section": 1,

    # NLP features
    "sentiment_score": 0.8,
    "recommend_prob": 0.9,
    "review_volume": 120,
    "weighted_sentiment": 0.75
}

# -----------------------------
# FEATURE ENGINEERING
# -----------------------------
price_raw = sample_input["price"]
price = np.log1p(price_raw)

promo_x_position = sample_input["Promotion"] * sample_input["Product Position"]
price_x_sentiment = price * sample_input["sentiment_score"]
promo_x_sentiment = sample_input["Promotion"] * sample_input["sentiment_score"]
sentiment_x_volume = sample_input["sentiment_score"] * np.log1p(sample_input["review_volume"])

# -----------------------------
# FINAL FEATURE VECTOR
# -----------------------------
X = [
    sample_input["Promotion"],
    sample_input["Seasonal"],
    price,
    sample_input["Product Position"],
    sample_input["section"],

    sample_input["sentiment_score"],
    sample_input["recommend_prob"],
    sample_input["review_volume"],
    sample_input["weighted_sentiment"],

    price_x_sentiment,
    promo_x_sentiment,
    promo_x_position,
    sentiment_x_volume
]

X = np.array(X).reshape(1, -1)
X_scaled = scaler.transform(X)

# -----------------------------
# PREDICT
# -----------------------------
prediction = model.predict(X_scaled)[0]

# -----------------------------
# 🔥 BUSINESS LOGIC FIX (CRITICAL)
# -----------------------------
if price_raw > 100000:
    prediction *= 0.1

if price_raw > 500000:
    prediction *= 0.01

# ensure no negative values
prediction = max(prediction, 0)

# -----------------------------
# OUTPUT
# -----------------------------
print("\n==============================")
print("📊 PREDICTED DEMAND:", int(prediction))
print("==============================\n")