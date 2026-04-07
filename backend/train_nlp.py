import os
import json
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, roc_auc_score, precision_recall_curve, auc
)
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from gensim.models import Word2Vec

import joblib
import seaborn as sns

# -----------------------------
# PATHS
# -----------------------------
BASE = r"C:\Users\Abcom\Desktop\Work\Code Projects\ZaraERP"
DATA_PATH = os.path.join(BASE, "data", "womenReview.csv")
OUTPUT_PATH = os.path.join(BASE, "outputs")
PLOT_PATH = os.path.join(OUTPUT_PATH, "vader_analysis")

os.makedirs(PLOT_PATH, exist_ok=True)

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv(DATA_PATH)

df = df.drop(columns=["Unnamed: 0"], errors="ignore")
df["Title"] = df["Title"].fillna("")
df = df.dropna(subset=["Review Text", "Department Name", "Class Name"])

df["text"] = df["Title"] + " " + df["Review Text"]

# -----------------------------
# CLEANING FUNCTIONS
# -----------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s!]", " ", text)
    return text

def clean_for_vader(text):
    return str(text).replace("n't", " not")

df["clean_text"] = df["text"].apply(clean_text)
df["vader_text"] = df["text"].apply(clean_for_vader)

# -----------------------------
# VADER SETUP
# -----------------------------
analyzer = SentimentIntensityAnalyzer()

custom_words = {
    "runs small": -2.0,
    "runs large": -1.5,
    "poor quality": -2.5,
    "cheap material": -2.0,
    "perfect fit": 2.5,
    "high quality": 2.5,
    "very comfortable": 2.0,
}
analyzer.lexicon.update(custom_words)

def vader_score(text):
    return analyzer.polarity_scores(text)["compound"]

df["sentiment_score"] = df["vader_text"].apply(vader_score)

def vader_label(score):
    if score >= 0.2:
        return "positive"
    elif score <= -0.2:
        return "negative"
    return "neutral"

df["sentiment_label"] = df["sentiment_score"].apply(vader_label)

# -----------------------------
# TF-IDF MODEL
# -----------------------------
vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1,2))
X_tfidf = vectorizer.fit_transform(df["clean_text"])
y = df["Recommended IND"]

X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y, test_size=0.2, stratify=y, random_state=42
)

lr = LogisticRegression(max_iter=1000, class_weight="balanced", C=2.0)
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)
y_prob = lr.predict_proba(X_test)[:,1]

# -----------------------------
# WORD2VEC MODEL
# -----------------------------
sentences = [text.split() for text in df["clean_text"]]

w2v = Word2Vec(sentences, vector_size=100, window=5, min_count=2)

def get_vec(text):
    words = text.split()
    vecs = [w2v.wv[w] for w in words if w in w2v.wv]
    return np.mean(vecs, axis=0) if vecs else np.zeros(100)

X_vec = np.vstack(df["clean_text"].apply(get_vec))
X_vec = np.hstack([X_vec, df["sentiment_score"].values.reshape(-1,1)])

X_train_v, X_test_v, y_train_v, y_test_v = train_test_split(
    X_vec, y, test_size=0.2, stratify=y, random_state=42
)

lr_vec = LogisticRegression(max_iter=1000, class_weight="balanced")
lr_vec.fit(X_train_v, y_train_v)

y_pred_v = lr_vec.predict(X_test_v)
y_prob_v = lr_vec.predict_proba(X_test_v)[:,1]

# -----------------------------
# METRICS
# -----------------------------
metrics = {
    "lr_accuracy": float(accuracy_score(y_test, y_pred)),
    "lr_f1": float(f1_score(y_test, y_pred)),
    "lr_precision": float(precision_score(y_test, y_pred)),
    "lr_recall": float(recall_score(y_test, y_pred)),
    "lr_auc": float(roc_auc_score(y_test, y_prob)),
    "w2v_f1": float(f1_score(y_test_v, y_pred_v)),
    "w2v_auc": float(roc_auc_score(y_test_v, y_prob_v)),
    "vader_auc": float(roc_auc_score(y, df["sentiment_score"]))
}

with open(os.path.join(OUTPUT_PATH, "nlp_metrics.json"), "w") as f:
    json.dump(metrics, f, indent=4)

# -----------------------------
# CONFUSION MATRIX
# -----------------------------
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d")
plt.savefig(os.path.join(PLOT_PATH, "B1_confusion_matrix.png"))
plt.clf()

# -----------------------------
# ROC + PR CURVE
# -----------------------------
fpr, tpr, _ = precision_recall_curve(y_test, y_prob)
plt.plot(tpr, fpr)
plt.savefig(os.path.join(PLOT_PATH, "B2_roc_pr_curves.png"))
plt.clf()

# -----------------------------
# LEARNING CURVE
# -----------------------------
train_sizes, train_scores, test_scores = learning_curve(
    lr, X_tfidf, y, cv=5
)

plt.plot(train_sizes, np.mean(train_scores, axis=1))
plt.plot(train_sizes, np.mean(test_scores, axis=1))
plt.savefig(os.path.join(PLOT_PATH, "C_learning_curves.png"))
plt.clf()

# -----------------------------
# CORRELATION MATRIX
# -----------------------------
corr = df[["Age", "Positive Feedback Count", "sentiment_score", "Recommended IND"]].corr()
sns.heatmap(corr, annot=True)
plt.savefig(os.path.join(PLOT_PATH, "A3_recommend_rate_and_correlation.png"))
plt.clf()

# -----------------------------
# SAVE MODELS
# -----------------------------
joblib.dump(lr, os.path.join(OUTPUT_PATH, "nlp_model.pkl"))
joblib.dump(vectorizer, os.path.join(OUTPUT_PATH, "nlp_vectorizer.pkl"))
joblib.dump(analyzer, os.path.join(OUTPUT_PATH, "sentiment_model.pkl"))

# -----------------------------
# SAVE PROCESSED DATA
# -----------------------------
df["predicted_recommend"] = lr.predict(X_tfidf)
df.to_csv(os.path.join(OUTPUT_PATH, "reviews_processed.csv"), index=False)

# -----------------------------
# CATEGORY AGGREGATION
# -----------------------------
agg = df.groupby(["Department Name", "Class Name"]).agg(
    avg_sentiment=("sentiment_score", "mean"),
    weighted_sentiment=("sentiment_score", lambda x: np.mean(x * np.log1p(df.loc[x.index, "Positive Feedback Count"]))),
    recommend_rate=("Recommended IND", "mean"),
    avg_rating=("Rating", "mean"),
    review_count=("Rating", "count")
).reset_index()

agg.to_csv(os.path.join(OUTPUT_PATH, "category_sentiment.csv"), index=False)

print("✅ ALL TASKS COMPLETE")


# -----------------------------
# ADD PROBABILITY OUTPUT
# -----------------------------
df["recommend_prob"] = lr.predict_proba(X_tfidf)[:,1]

# -----------------------------
# AGGREGATED NLP FEATURES (FOR DEMAND MODEL)
# -----------------------------
nlp_agg = df.groupby(["Department Name", "Class Name"]).agg(
    sentiment_score=("sentiment_score", "mean"),
    recommend_prob=("recommend_prob", "mean"),
    review_volume=("Review Text", "count"),
    weighted_sentiment=("sentiment_score", lambda x: np.mean(x * np.log1p(df.loc[x.index, "Positive Feedback Count"])))
).reset_index()

nlp_agg.to_csv(os.path.join(OUTPUT_PATH, "nlp_features.csv"), index=False)

print("✅ NLP FEATURES SAVED")