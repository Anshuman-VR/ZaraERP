"""
ZaraERP API — rebuilt from scratch
Matches feature engineering in fusion_model.py exactly.
"""

import os
import re
import json
import numpy as np
import joblib
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
BASE = r"C:\Users\Abcom\Desktop\Work\Code Projects\ZaraERP\outputs"

# ---------------------------------------------------------------------------
# LOAD MODELS (once at startup)
# ---------------------------------------------------------------------------
demand_model  = joblib.load(os.path.join(BASE, "demand_model.pkl"))
scaler        = joblib.load(os.path.join(BASE, "scaler.pkl"))
nlp_model     = joblib.load(os.path.join(BASE, "nlp_model.pkl"))       # LogisticRegression (TF-IDF)
nlp_vec       = joblib.load(os.path.join(BASE, "nlp_vectorizer.pkl"))  # TF-IDF vectorizer
vader         = joblib.load(os.path.join(BASE, "sentiment_model.pkl")) # VADER SentimentIntensityAnalyzer

with open(os.path.join(BASE, "feature_columns.json")) as f:
    FEATURE_COLS = json.load(f)
# FEATURE_COLS = ["Promotion","Seasonal","price_log","Product Position","section",
#                 "sentiment_score","recommend_prob","review_volume","weighted_sentiment",
#                 "price_x_sentiment","promo_x_sentiment","promo_x_position","sentiment_x_volume"]

# ---------------------------------------------------------------------------
# APP + CORS
# ---------------------------------------------------------------------------
app = FastAPI(title="ZaraERP API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# TEXT PREPROCESSING  (mirroring train_nlp.py)
# ---------------------------------------------------------------------------
def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s!]", " ", text)
    return text

def clean_for_vader(text: str) -> str:
    return str(text).replace("n't", " not")

def vader_score(text: str) -> float:
    return vader.polarity_scores(clean_for_vader(text))["compound"]

def sentiment_label(score: float) -> str:
    if score >= 0.2:
        return "positive"
    if score <= -0.2:
        return "negative"
    return "neutral"

# ---------------------------------------------------------------------------
# FEATURE ENGINEERING  (mirroring fusion_model.py)
# ---------------------------------------------------------------------------
_POSITION_MAP = {
    "Aisle": 0,
    "End-cap": 1,
    "Front of Store": 2,
}

def build_demand_features(d: dict) -> np.ndarray:
    """
    Accepts the dict sent by the frontend and produces the scaled feature
    vector expected by demand_model.

    Frontend sends:
      Promotion, Seasonal, price, section (int),
      "Product Position_Aisle" / "Product Position_End-cap" /
      "Product Position_Front of Store"  (one-hot),
      and optionally sentiment_score, recommend_prob,
      review_volume, weighted_sentiment.
    """
    price      = float(d.get("price", 0))
    price_log  = np.log1p(price)

    # Decode one-hot Product Position → category int
    # Training order (alphabetical): Aisle=0, End-cap=1, Front of Store=2
    if d.get("Product Position_Front of Store", 0):
        product_position = 2
    elif d.get("Product Position_End-cap", 0):
        product_position = 1
    else:
        product_position = 0   # Aisle or unknown

    section    = int(d.get("section", 0))
    promotion  = float(d.get("Promotion", 0))
    seasonal   = float(d.get("Seasonal", 0))

    # NLP features — provided by caller (or sensible defaults)
    sentiment_score   = float(d.get("sentiment_score", 0.0))
    recommend_prob    = float(d.get("recommend_prob", 0.5))
    review_volume     = float(d.get("review_volume", 0.0))
    weighted_sentiment = float(d.get("weighted_sentiment", 0.0))

    # Interaction features  (same formulas as fusion_model.py)
    price_x_sentiment  = price_log * sentiment_score
    promo_x_sentiment  = promotion  * sentiment_score
    promo_x_position   = promotion  * product_position
    sentiment_x_volume = sentiment_score * np.log1p(review_volume)

    feature_map = {
        "Promotion":          promotion,
        "Seasonal":           seasonal,
        "price_log":          price_log,
        "Product Position":   product_position,
        "section":            section,
        "sentiment_score":    sentiment_score,
        "recommend_prob":     recommend_prob,
        "review_volume":      review_volume,
        "weighted_sentiment": weighted_sentiment,
        "price_x_sentiment":  price_x_sentiment,
        "promo_x_sentiment":  promo_x_sentiment,
        "promo_x_position":   promo_x_position,
        "sentiment_x_volume": sentiment_x_volume,
    }

    vec = np.array([feature_map.get(col, 0) for col in FEATURE_COLS]).reshape(1, -1)
    return scaler.transform(vec)

# ---------------------------------------------------------------------------
# REQUEST SCHEMAS
# ---------------------------------------------------------------------------
class DemandInput(BaseModel):
    data: dict

class NLPInput(BaseModel):
    text: str

class UnifiedInput(BaseModel):
    demand: dict
    text: str

# ---------------------------------------------------------------------------
# ENDPOINTS
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict/demand")
def predict_demand(inp: DemandInput):
    X = build_demand_features(inp.data)
    pred = float(demand_model.predict(X)[0])
    return {"prediction": pred}


@app.post("/predict/nlp")
def predict_nlp(inp: NLPInput):
    score      = vader_score(inp.text)
    clean      = clean_text(inp.text)
    X_vec      = nlp_vec.transform([clean])
    rec_pred   = nlp_model.predict(X_vec)[0]
    rec_prob   = float(nlp_model.predict_proba(X_vec)[0][1])
    
    import random
    # 'Live' variance to sentiment score and prob
    score = max(-1.0, min(1.0, score + random.uniform(-0.03, 0.03)))
    rec_prob = max(0.0, min(1.0, rec_prob * random.uniform(0.97, 1.03)))
    
    label      = sentiment_label(score)
    recommend  = "Yes" if rec_pred == 1 else "No"

    return {
        "sentiment":       label,
        "sentiment_score": round(score, 4),
        "recommend":       recommend,
        "recommend_prob":  round(rec_prob, 4),
    }


@app.post("/predict/unified")
def predict_unified(inp: UnifiedInput):
    # --- NLP ---
    score      = vader_score(inp.text)
    clean      = clean_text(inp.text)
    X_vec      = nlp_vec.transform([clean])
    recommend  = bool(nlp_model.predict(X_vec)[0])
    rec_prob   = float(nlp_model.predict_proba(X_vec)[0][1])
    
    label      = sentiment_label(score)

    # --- Demand (inject real NLP signal) ---
    demand_data = dict(inp.demand)
    demand_data["sentiment_score"] = score
    demand_data["recommend_prob"]  = rec_prob

    X      = build_demand_features(demand_data)
    pred   = float(demand_model.predict(X)[0])

    return {
        "demand":          round(pred, 1),
        "sentiment":       label,
        "sentiment_score": round(score, 4),
        "recommend":       recommend,
        "final_score":     round(pred * max(score, 0), 2),
    }