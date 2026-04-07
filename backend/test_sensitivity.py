import os
import joblib
import pandas as pd
import numpy as np

BASE = r"C:\Users\Abcom\Desktop\Work\Code Projects\ZaraERP\outputs"
model = joblib.load(os.path.join(BASE, "demand_model.pkl"))
scaler = joblib.load(os.path.join(BASE, "scaler.pkl"))
with open(os.path.join(BASE, "feature_columns.json"), "r") as f:
    import json
    feature_cols = json.load(f)

def get_pred(promo, seasonal, price=30):
    price_log = np.log1p(price)
    feature_map = {
        "Promotion": float(promo),
        "Seasonal": float(seasonal),
        "price_log": price_log,
        "Product Position": 1, # End-cap
        "section": 0, # Women
        "sentiment_score": 0.5,
        "recommend_prob": 0.8,
        "review_volume": 100,
        "weighted_sentiment": 0.4,
        "price_x_sentiment": price_log * 0.5,
        "promo_x_sentiment": float(promo) * 0.5,
        "promo_x_position": float(promo) * 1,
        "sentiment_x_volume": 0.5 * np.log1p(100)
    }
    vec = np.array([feature_map[c] for c in feature_cols]).reshape(1, -1)
    return model.predict(scaler.transform(vec))[0]

base = get_pred(0, 0)
boost = get_pred(1, 1)

print(f"Base Prediction (No Promo/Season): {base:.2f}")
print(f"Boosted Prediction (Promo+Season): {boost:.2f}")
print(f"Impact Multiplier: {boost/base:.2f}x")
