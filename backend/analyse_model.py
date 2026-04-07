import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, roc_auc_score, precision_recall_curve, auc,
    classification_report
)
from sklearn.model_selection import train_test_split

import joblib
from gensim.models import Word2Vec

# -----------------------------
# PATHS
# -----------------------------
BASE = r"C:\Users\Abcom\Desktop\Work\Code Projects\ZaraERP"
DATA_PATH = os.path.join(BASE, "data", "womenReview.csv")
OUTPUT_PATH = os.path.join(BASE, "outputs")
PLOT_PATH = os.path.join(OUTPUT_PATH, "vader_analysis")

# -----------------------------
# LOAD DATA + MODELS
# -----------------------------
df = pd.read_csv(DATA_PATH)

lr_model = joblib.load(os.path.join(OUTPUT_PATH, "nlp_model.pkl"))
vectorizer = joblib.load(os.path.join(OUTPUT_PATH, "nlp_vectorizer.pkl"))
vader = joblib.load(os.path.join(OUTPUT_PATH, "sentiment_model.pkl"))

# -----------------------------
# CLEANING (same as training)
# -----------------------------
import re

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s!]", " ", text)
    return text

def clean_for_vader(text):
    return str(text).replace("n't", " not")

df["text"] = df["Title"].fillna("") + " " + df["Review Text"]
df["clean_text"] = df["text"].apply(clean_text)
df["vader_text"] = df["text"].apply(clean_for_vader)

# -----------------------------
# VADER SCORE
# -----------------------------
df["vader_score"] = df["vader_text"].apply(
    lambda x: vader.polarity_scores(x)["compound"]
)

# -----------------------------
# REBUILD TF-IDF TEST SET
# -----------------------------
X = vectorizer.transform(df["clean_text"])
y = df["Recommended IND"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# -----------------------------
# TF-IDF MODEL EVAL
# -----------------------------
y_pred = lr_model.predict(X_test)
y_prob = lr_model.predict_proba(X_test)[:, 1]

# -----------------------------
# WORD2VEC REBUILD
# -----------------------------
sentences = [text.split() for text in df["clean_text"]]
w2v = Word2Vec(sentences, vector_size=100, window=5, min_count=2)

def get_vec(text):
    words = text.split()
    vecs = [w2v.wv[w] for w in words if w in w2v.wv]
    return np.mean(vecs, axis=0) if vecs else np.zeros(100)

X_vec = np.vstack(df["clean_text"].apply(get_vec))
X_vec = np.hstack([X_vec, df["vader_score"].values.reshape(-1,1)])

X_train_v, X_test_v, y_train_v, y_test_v = train_test_split(
    X_vec, y, test_size=0.2, stratify=y, random_state=42
)

from sklearn.linear_model import LogisticRegression
lr_vec = LogisticRegression(max_iter=1000, class_weight="balanced")
lr_vec.fit(X_train_v, y_train_v)

y_pred_v = lr_vec.predict(X_test_v)
y_prob_v = lr_vec.predict_proba(X_test_v)[:,1]

# -----------------------------
# VADER AS CLASSIFIER
# -----------------------------
y_pred_vader = (df["vader_score"] > 0).astype(int)

# -----------------------------
# METRICS
# -----------------------------
def evaluate(name, y_true, y_pred, y_prob):
    return {
        "model": name,
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_prob)
    }

results = []

results.append(evaluate("TFIDF_LR", y_test, y_pred, y_prob))
results.append(evaluate("W2V_LR", y_test_v, y_pred_v, y_prob_v))
results.append(evaluate("VADER", y, y_pred_vader, df["vader_score"]))

# Save metrics
with open(os.path.join(OUTPUT_PATH, "analysis_summary.json"), "w") as f:
    json.dump(results, f, indent=4)

# -----------------------------
# CONFUSION MATRIX
# -----------------------------
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d")
plt.title("Confusion Matrix - TFIDF")
plt.savefig(os.path.join(PLOT_PATH, "B1_confusion_matrix_analysis.png"))
plt.clf()

# -----------------------------
# ROC + PR CURVES
# -----------------------------
precision, recall, _ = precision_recall_curve(y_test, y_prob)
pr_auc = auc(recall, precision)

plt.plot(recall, precision)
plt.title(f"PR Curve (AUC={pr_auc:.3f})")
plt.savefig(os.path.join(PLOT_PATH, "B2_pr_curve_analysis.png"))
plt.clf()

# -----------------------------
# CLASSIFICATION REPORT
# -----------------------------
report = classification_report(y_test, y_pred)

with open(os.path.join(OUTPUT_PATH, "classification_report.txt"), "w") as f:
    f.write(report)

# -----------------------------
# 🔥 NEW REVIEWS TEST (UNSEEN)
# -----------------------------
new_reviews = [
    "This dress fits perfectly and looks amazing!",
    "Terrible quality, fell apart after one wash.",
    "Very comfortable but runs slightly large.",
    "Not worth the price at all.",
    "Absolutely love this jacket, great material!",
    "The fabric feels cheap and itchy.",
    "Nice design but poor stitching.",
    "Super soft and perfect fit!",
    "Disappointed, expected better quality.",
    "Great purchase, would definitely recommend."
]

df_new = pd.DataFrame({"text": new_reviews})
df_new["clean_text"] = df_new["text"].apply(clean_text)
df_new["vader_score"] = df_new["text"].apply(
    lambda x: vader.polarity_scores(x)["compound"]
)

# TF-IDF predictions
X_new = vectorizer.transform(df_new["clean_text"])
df_new["tfidf_pred"] = lr_model.predict(X_new)
df_new["tfidf_prob"] = lr_model.predict_proba(X_new)[:,1]

# Word2Vec predictions
X_new_vec = np.vstack(df_new["clean_text"].apply(get_vec))
X_new_vec = np.hstack([X_new_vec, df_new["vader_score"].values.reshape(-1,1)])

df_new["w2v_pred"] = lr_vec.predict(X_new_vec)
df_new["w2v_prob"] = lr_vec.predict_proba(X_new_vec)[:,1]

# VADER label
df_new["vader_pred"] = (df_new["vader_score"] > 0).astype(int)

df_new.to_csv(os.path.join(OUTPUT_PATH, "new_reviews_analysis.csv"), index=False)

print("\n=== NEW REVIEWS ANALYSIS ===")
print(df_new)

# -----------------------------
# FINAL PRINT
# -----------------------------
print("\n=== MODEL COMPARISON ===")
for r in results:
    print(r)

print("\nAnalysis complete. Outputs saved.")