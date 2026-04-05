"""
ZaraIQ ERP — Team 2: Inference Helpers
========================================
Used by the Streamlit frontend. Never import train_nlp.py from here.

Functions:
    predict_review(review_text, title="")  → dict
    get_category_stats(department, class_name) → dict
"""

import re
import os
import joblib
import numpy as np
import pandas as pd

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download if not already present
for pkg in ["stopwords", "wordnet", "omw-1.4"]:
    nltk.download(pkg, quiet=True)

OUTPUTS_DIR = "outputs"

# ── Lazy-load models (called once at import time) ─────────────────────────────
_pipeline  = None   # TF-IDF + LR sklearn Pipeline
_vader     = None   # VADER SentimentIntensityAnalyzer
_cat_df    = None   # category_sentiment.csv DataFrame

_stop_words = set(stopwords.words("english"))
_lemmatizer = WordNetLemmatizer()


def _load_models():
    global _pipeline, _vader, _cat_df
    if _pipeline is None:
        _pipeline = joblib.load(os.path.join(OUTPUTS_DIR, "nlp_model.pkl"))
    if _vader is None:
        _vader    = joblib.load(os.path.join(OUTPUTS_DIR, "sentiment_model.pkl"))
    if _cat_df is None:
        _cat_df   = pd.read_csv(os.path.join(OUTPUTS_DIR, "category_sentiment.csv"))


def _clean(title: str, review: str) -> str:
    combined = f"{title} {review}".lower()
    combined = re.sub(r"[^a-z\s]", " ", combined)
    tokens   = combined.split()
    tokens   = [_lemmatizer.lemmatize(t)
                for t in tokens
                if t not in _stop_words and len(t) > 2]
    return " ".join(tokens)


# =============================================================================
# PUBLIC API
# =============================================================================

def predict_review(review_text: str, title: str = "") -> dict:
    """
    Run full NLP inference on a single review.

    Parameters
    ----------
    review_text : str   Raw review body text
    title       : str   Optional review title (boosts accuracy)

    Returns
    -------
    dict with keys:
        sentiment_score       float   VADER compound [-1, +1]
        sentiment_label       str     'positive' | 'neutral' | 'negative'
        recommend_probability float   LR probability of Recommended=1
        recommend             bool    True if probability >= 0.5
        top_phrases           list    Top 5 TF-IDF terms for this text
    """
    _load_models()

    clean  = _clean(title, review_text)

    # VADER sentiment
    scores = _vader.polarity_scores(review_text)
    compound = scores["compound"]
    if compound >= 0.05:
        label = "positive"
    elif compound <= -0.05:
        label = "negative"
    else:
        label = "neutral"

    # LR recommendation probability
    prob    = float(_pipeline.predict_proba([clean])[0][1])
    rec     = prob >= 0.5

    # Top TF-IDF terms (feature names with highest weights for this doc)
    vectorizer = _pipeline.named_steps["tfidf"]
    lr_model   = _pipeline.named_steps["lr"]
    vec_matrix = vectorizer.transform([clean])
    feature_names = vectorizer.get_feature_names_out()

    # Get nonzero feature indices and their tfidf values
    nonzero_indices = vec_matrix.nonzero()[1]
    tfidf_scores    = np.array(vec_matrix[0, nonzero_indices]).flatten()
    top_idx         = nonzero_indices[np.argsort(tfidf_scores)[::-1][:5]]
    top_phrases     = [feature_names[i] for i in top_idx]

    return {
        "sentiment_score":       round(compound, 4),
        "sentiment_label":       label,
        "recommend_probability": round(prob, 4),
        "recommend":             rec,
        "top_phrases":           top_phrases,
    }


def get_category_stats(department: str, class_name: str) -> dict:
    """
    Look up aggregated stats for a product category.

    Parameters
    ----------
    department : str   e.g. 'Tops', 'Jackets', 'Dresses'
    class_name : str   e.g. 'Knits', 'Jeans', 'Sweaters'

    Returns
    -------
    dict with all 9 columns from category_sentiment.csv,
    or a dict of NaN/empty values if category not found.
    """
    _load_models()

    mask = (
        (_cat_df["department"].str.lower() == department.lower()) &
        (_cat_df["class_name"].str.lower() == class_name.lower())
    )
    row = _cat_df[mask]

    if row.empty:
        # Graceful fallback — frontend can show "No data"
        return {
            "department":         department,
            "class_name":         class_name,
            "avg_sentiment":      None,
            "weighted_sentiment": None,
            "recommend_rate":     None,
            "avg_rating":         None,
            "review_count":       0,
            "top_complaint":      "No data",
            "top_praise":         "No data",
        }

    return row.iloc[0].to_dict()


# =============================================================================
# Quick smoke-test (run directly: python backend/predict.py)
# =============================================================================
if __name__ == "__main__":
    print("── predict_review() smoke test ──")

    sample_reviews = [
        ("Love it!", "This dress fits perfectly and the fabric is amazing. Totally recommend!"),
        ("Terrible", "Runs extremely small, poor quality stitching, very disappointed."),
        ("Okay I guess", "It's fine but nothing special. Might return it."),
    ]

    for title, text in sample_reviews:
        result = predict_review(text, title)
        print(f"\n  Title   : {title}")
        print(f"  Review  : {text[:60]}...")
        print(f"  Result  : {result}")

    print("\n── get_category_stats() smoke test ──")
    cats = [("Tops", "Knits"), ("Jackets", "Jackets"), ("Bottoms", "Jeans")]
    for dept, cls in cats:
        stats = get_category_stats(dept, cls)
        print(f"\n  {dept} / {cls} → recommend_rate={stats.get('recommend_rate')}, "
              f"avg_sentiment={stats.get('avg_sentiment')}, "
              f"reviews={stats.get('review_count')}")
        print(f"    top_praise    : {stats.get('top_praise')}")
        print(f"    top_complaint : {stats.get('top_complaint')}")

    print("\n✅ Inference helpers working correctly.")