"""
ZaraIQ ERP — Team 2: NLP Training Script
=========================================
Run this ONCE from the project root:
    python backend/train_nlp.py

Produces all 6 required outputs in /outputs:
    - outputs/nlp_model.pkl
    - outputs/nlp_vectorizer.pkl
    - outputs/sentiment_model.pkl
    - outputs/nlp_metrics.json
    - outputs/reviews_processed.csv
    - outputs/category_sentiment.csv
"""

import os
import re
import json
import joblib
import warnings
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, f1_score,
                             precision_score, recall_score,
                             confusion_matrix, classification_report)
from sklearn.pipeline import Pipeline

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

warnings.filterwarnings("ignore")

# ── Download NLTK data (safe to run multiple times) ──────────────────────────
for pkg in ["stopwords", "wordnet", "omw-1.4"]:
    nltk.download(pkg, quiet=True)

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_PATH    = os.path.join("data", "womenReview.csv")
OUTPUTS_DIR  = "outputs"
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# =============================================================================
# STEP 1 — Load & Clean
# =============================================================================
print("\n[1/6] Loading data...")
df = pd.read_csv(DATA_PATH)
print(f"      Raw shape: {df.shape}")

# Drop the artifact index column
if "Unnamed: 0" in df.columns:
    df.drop(columns=["Unnamed: 0"], inplace=True)

# Fill Title NAs with empty string
df["Title"] = df["Title"].fillna("")

# Drop rows with null Review Text (845 rows per PDR)
df.dropna(subset=["Review Text"], inplace=True)

# Drop rows with null Division / Department / Class (14 rows per PDR)
df.dropna(subset=["Division Name", "Department Name", "Class Name"], inplace=True)

# Reset index after drops
df.reset_index(drop=True, inplace=True)
print(f"      Clean shape: {df.shape}")

# =============================================================================
# STEP 2 — Text Preprocessing
# =============================================================================
print("\n[2/6] Preprocessing text...")

stop_words  = set(stopwords.words("english"))
lemmatizer  = WordNetLemmatizer()

def clean_text(title: str, review: str) -> str:
    """Concatenate title + review, lowercase, strip punctuation,
    remove stopwords, lemmatize."""
    combined = f"{title} {review}".lower()
    # Remove punctuation and digits
    combined = re.sub(r"[^a-z\s]", " ", combined)
    tokens   = combined.split()
    tokens   = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words and len(t) > 2]
    return " ".join(tokens)

df["clean_text"] = df.apply(
    lambda row: clean_text(row["Title"], row["Review Text"]), axis=1
)
print("      Text cleaning done.")

# =============================================================================
# STEP 3 — TF-IDF + Logistic Regression Classifier
# =============================================================================
print("\n[3/6] Training TF-IDF + Logistic Regression...")

X = df["clean_text"]
y = df["Recommended IND"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Build sklearn Pipeline (vectorizer + classifier together)
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=10000, ngram_range=(1, 2))),
    ("lr",    LogisticRegression(
                  C=2.0,
                  solver="lbfgs",
                  max_iter=1000,
                  class_weight="balanced",
                  random_state=42
              ))
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:, 1]

acc  = accuracy_score(y_test, y_pred)
f1   = f1_score(y_test, y_pred, average="weighted")
prec = precision_score(y_test, y_pred, average="weighted")
rec  = recall_score(y_test, y_pred, average="weighted")

print(f"      Accuracy : {acc:.4f}")
print(f"      F1       : {f1:.4f}")
print(f"      Precision: {prec:.4f}")
print(f"      Recall   : {rec:.4f}")
print("\n" + classification_report(y_test, y_pred))

# Save the full pipeline as nlp_model.pkl
nlp_model_path = os.path.join(OUTPUTS_DIR, "nlp_model.pkl")
joblib.dump(pipeline, nlp_model_path)
print(f"      Saved → {nlp_model_path}")

# Also save the vectorizer separately (as required by PDR output contract)
nlp_vec_path = os.path.join(OUTPUTS_DIR, "nlp_vectorizer.pkl")
joblib.dump(pipeline.named_steps["tfidf"], nlp_vec_path)
print(f"      Saved → {nlp_vec_path}")

# Save metrics JSON
metrics = {
    "lr_accuracy":  round(acc,  4),
    "lr_f1":        round(f1,   4),
    "lr_precision": round(prec, 4),
    "lr_recall":    round(rec,  4),
    "bert_f1":      None          # DistilBERT stretch goal — not trained
}
metrics_path = os.path.join(OUTPUTS_DIR, "nlp_metrics.json")
with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=2)
print(f"      Saved → {metrics_path}")

# =============================================================================
# STEP 4 — VADER Sentiment on All Reviews
# =============================================================================
print("\n[4/6] Running VADER sentiment analysis...")

vader = SentimentIntensityAnalyzer()

def vader_compound(text: str) -> float:
    return vader.polarity_scores(str(text))["compound"]

df["sentiment_score"]       = df["Review Text"].apply(vader_compound)
df["sentiment_label"]       = df["sentiment_score"].apply(
    lambda s: "positive" if s >= 0.05 else ("negative" if s <= -0.05 else "neutral")
)

# Weighted sentiment: vader_compound × log1p(Positive Feedback Count)
df["weighted_sentiment"]    = (
    df["sentiment_score"] * np.log1p(df["Positive Feedback Count"].fillna(0))
)

# Add predicted_recommend from the trained LR pipeline
df["predicted_recommend"]   = pipeline.predict(df["clean_text"])
df["recommend_probability"] = pipeline.predict_proba(df["clean_text"])[:, 1]

# Save VADER model
sentiment_model_path = os.path.join(OUTPUTS_DIR, "sentiment_model.pkl")
joblib.dump(vader, sentiment_model_path)
print(f"      Saved → {sentiment_model_path}")

# Save reviews_processed.csv
reviews_path = os.path.join(OUTPUTS_DIR, "reviews_processed.csv")
df.to_csv(reviews_path, index=False)
print(f"      Saved → {reviews_path}  ({len(df):,} rows)")

# =============================================================================
# STEP 5 — Category-Level Aggregation → category_sentiment.csv
# =============================================================================
print("\n[5/6] Building category_sentiment aggregation...")

def top_ngram(sub_df: pd.DataFrame, stars: list, n: int = 2, top_k: int = 1) -> str:
    """
    Extract the most frequent n-gram from reviews with given star ratings.
    Returns the top phrase as a string, or '' if nothing found.
    """
    from sklearn.feature_extraction.text import CountVectorizer
    texts = sub_df[sub_df["Rating"].isin(stars)]["clean_text"].dropna().tolist()
    if not texts:
        return ""
    try:
        cv = CountVectorizer(ngram_range=(n, n), max_features=500, stop_words="english")
        cv.fit_transform(texts)
        freq = dict(zip(cv.get_feature_names_out(),
                        cv.transform(texts).toarray().sum(axis=0)))
        if not freq:
            return ""
        top = sorted(freq, key=freq.get, reverse=True)[:top_k]
        return ", ".join(top)
    except Exception:
        return ""

records = []
grouped = df.groupby(["Department Name", "Class Name"])

for (dept, cls), grp in grouped:
    record = {
        "department":        dept,
        "class_name":        cls,
        "avg_sentiment":     round(grp["sentiment_score"].mean(), 4),
        "weighted_sentiment":round(grp["weighted_sentiment"].mean(), 4),
        "recommend_rate":    round(grp["Recommended IND"].mean(), 4),
        "avg_rating":        round(grp["Rating"].mean(), 4),
        "review_count":      len(grp),
        "top_complaint":     top_ngram(grp, stars=[1, 2]),
        "top_praise":        top_ngram(grp, stars=[4, 5]),
    }
    records.append(record)

cat_df = pd.DataFrame(records)
cat_path = os.path.join(OUTPUTS_DIR, "category_sentiment.csv")
cat_df.to_csv(cat_path, index=False)
print(f"      Saved → {cat_path}  ({len(cat_df)} categories)")

# =============================================================================
# STEP 6 — Verification Summary
# =============================================================================
print("\n[6/6] Output verification:")
expected = [
    "nlp_model.pkl",
    "nlp_vectorizer.pkl",
    "sentiment_model.pkl",
    "nlp_metrics.json",
    "reviews_processed.csv",
    "category_sentiment.csv",
]
all_ok = True
for fname in expected:
    path  = os.path.join(OUTPUTS_DIR, fname)
    exists = os.path.exists(path)
    size  = os.path.getsize(path) if exists else 0
    status = "✓" if exists else "✗ MISSING"
    print(f"      {status}  {fname}  ({size:,} bytes)")
    if not exists:
        all_ok = False

if all_ok:
    print("\n✅  All Team 2 outputs generated successfully.")
    print("    Next step: run  python backend/predict.py  to test inference.")
else:
    print("\n❌  Some outputs are missing — check errors above.")