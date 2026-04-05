"""
ZaraIQ ERP — Team 2: Model Analysis & Diagnostics (v2)
=======================================================
Run from project root AFTER train_nlp.py has completed:
    python backend/analyse_model.py

FIX vs v1:
  - Reads C directly from the saved pkl (no hardcoded assumption)
  - C-sweep verdict uses BOTH test accuracy AND gap <= 0.05 as criteria
  - Summary correctly reflects the model actually on disk

Produces:  outputs/analysis/*.png
"""

import os, re, warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import (
    train_test_split, learning_curve, StratifiedKFold, cross_val_score
)
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score,
    ConfusionMatrixDisplay
)
from sklearn.pipeline import Pipeline
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

warnings.filterwarnings("ignore")
for pkg in ["stopwords", "wordnet", "omw-1.4"]:
    nltk.download(pkg, quiet=True)

DATA_PATH = os.path.join("data", "womenReview.csv")
OUT_DIR   = os.path.join("outputs", "analysis")
os.makedirs(OUT_DIR, exist_ok=True)

ACCENT  = "#E63946"
GREEN   = "#2D6A4F"
BLUE    = "#457B9D"
BG      = "#F8F9FA"
PALETTE = ["#2D6A4F","#52B788","#B7E4C7","#D8F3DC",
           "#E63946","#F4A261","#457B9D","#1D3557"]

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3, "grid.color": "#CCCCCC",
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.titlesize": 13, "axes.titleweight": "bold", "figure.dpi": 130,
})

def save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  ✓  {path}")

# =============================================================================
# LOAD & PREP
# =============================================================================
print("\n[SETUP] Loading and preprocessing data...")

df = pd.read_csv(DATA_PATH)
if "Unnamed: 0" in df.columns:
    df.drop(columns=["Unnamed: 0"], inplace=True)
df["Title"] = df["Title"].fillna("")
df.dropna(subset=["Review Text"], inplace=True)
df.dropna(subset=["Division Name","Department Name","Class Name"], inplace=True)
df.reset_index(drop=True, inplace=True)

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(title, review):
    combined = f"{title} {review}".lower()
    combined = re.sub(r"[^a-z\s]", " ", combined)
    tokens   = combined.split()
    return " ".join([lemmatizer.lemmatize(t) for t in tokens
                     if t not in stop_words and len(t) > 2])

df["clean_text"] = df.apply(lambda r: clean_text(r["Title"], r["Review Text"]), axis=1)

vader = SentimentIntensityAnalyzer()
df["sentiment_score"] = df["Review Text"].apply(
    lambda t: vader.polarity_scores(str(t))["compound"])

X = df["clean_text"]
y = df["Recommended IND"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y)

pipeline     = joblib.load(os.path.join("outputs", "nlp_model.pkl"))
saved_C      = pipeline.named_steps["lr"].C
y_pred       = pipeline.predict(X_test)
y_prob       = pipeline.predict_proba(X_test)[:, 1]
y_train_pred = pipeline.predict(X_train)

print(f"  Loaded model C = {saved_C}")
print(f"  Dataset: {len(df):,} rows | Train: {len(X_train):,} | Test: {len(X_test):,}")

# =============================================================================
# A. EDA
# =============================================================================
print("\n[A] Dataset EDA plots...")

# A1
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("A1 · Dataset Overview", fontsize=15, fontweight="bold", y=1.02)
rec_counts = y.value_counts()
axes[0].bar(["Not Rec. (0)","Rec. (1)"],
            [rec_counts.get(0,0), rec_counts.get(1,0)],
            color=[ACCENT, GREEN], edgecolor="white", linewidth=1.5)
axes[0].set_title("Target Distribution"); axes[0].set_ylabel("Count")
for bar in axes[0].patches:
    axes[0].annotate(f"{bar.get_height():,.0f}\n({bar.get_height()/len(y)*100:.1f}%)",
                     xy=(bar.get_x()+bar.get_width()/2, bar.get_height()),
                     ha="center", va="bottom", fontsize=10, fontweight="bold")

rating_counts = df["Rating"].value_counts().sort_index()
colors_r = [ACCENT if r<=2 else (PALETTE[6] if r==3 else GREEN) for r in rating_counts.index]
axes[1].bar(rating_counts.index.astype(str), rating_counts.values,
            color=colors_r, edgecolor="white")
axes[1].set_title("Rating Distribution"); axes[1].set_xlabel("Stars"); axes[1].set_ylabel("Count")
for bar in axes[1].patches:
    axes[1].annotate(f"{bar.get_height():,.0f}",
                     xy=(bar.get_x()+bar.get_width()/2, bar.get_height()),
                     ha="center", va="bottom", fontsize=9)

axes[2].hist(df["Age"].dropna(), bins=30, color=BLUE, edgecolor="white", alpha=0.85)
axes[2].set_title("Reviewer Age"); axes[2].set_xlabel("Age"); axes[2].set_ylabel("Count")
axes[2].axvline(df["Age"].mean(), color=ACCENT, linestyle="--", linewidth=2,
                label=f"Mean: {df['Age'].mean():.1f}")
axes[2].legend()
plt.tight_layout(); save(fig, "A1_dataset_overview.png")

# A2
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("A2 · Category Breakdown", fontsize=15, fontweight="bold")
dept_counts = df["Department Name"].value_counts()
axes[0].barh(dept_counts.index, dept_counts.values,
             color=PALETTE[:len(dept_counts)], edgecolor="white")
axes[0].set_title("Reviews by Department"); axes[0].set_xlabel("Count")
for i, v in enumerate(dept_counts.values):
    axes[0].text(v+50, i, f"{v:,}", va="center", fontsize=9)

class_counts = df["Class Name"].value_counts().head(15)
axes[1].barh(class_counts.index, class_counts.values,
             color=sns.color_palette("Blues_r", len(class_counts)), edgecolor="white")
axes[1].set_title("Top 15 Classes by Volume"); axes[1].set_xlabel("Count")
plt.tight_layout(); save(fig, "A2_category_breakdown.png")

# A3
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("A3 · Recommendation Rate & Correlations", fontsize=15, fontweight="bold")
dept_rec = df.groupby("Department Name")["Recommended IND"].mean().sort_values()
colors_dr = [GREEN if v>=0.82 else ACCENT for v in dept_rec.values]
axes[0].barh(dept_rec.index, dept_rec.values, color=colors_dr, edgecolor="white")
axes[0].set_title("Recommend Rate by Department"); axes[0].set_xlabel("Rate")
axes[0].axvline(y.mean(), color="black", linestyle="--", linewidth=1.5,
                label=f"Overall: {y.mean():.3f}")
axes[0].legend(); axes[0].set_xlim(0, 1)
for i, v in enumerate(dept_rec.values):
    axes[0].text(v+0.005, i, f"{v:.3f}", va="center", fontsize=9)

numeric_cols = ["Rating","Age","Positive Feedback Count","Recommended IND","sentiment_score"]
corr = df[numeric_cols].dropna().corr()
cmap = LinearSegmentedColormap.from_list("rg", [ACCENT,"white",GREEN])
sns.heatmap(corr, ax=axes[1], annot=True, fmt=".3f", cmap=cmap,
            vmin=-1, vmax=1, square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
axes[1].set_title("Correlation Matrix — Numeric Features")
plt.tight_layout(); save(fig, "A3_recommend_rate_and_correlation.png")

# A4
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("A4 · Feedback Count & Sentiment Distribution", fontsize=15, fontweight="bold")
pfc = df["Positive Feedback Count"].fillna(0)
axes[0].hist(pfc[pfc>0], bins=50, color=BLUE, edgecolor="white", alpha=0.85)
axes[0].set_title("Positive Feedback Count (>0 only)")
axes[0].set_xlabel("Count"); axes[0].set_ylabel("Reviews"); axes[0].set_yscale("log")
axes[1].hist(df["sentiment_score"], bins=50, color=GREEN, edgecolor="white", alpha=0.85)
axes[1].set_title("VADER Score Distribution")
axes[1].set_xlabel("Compound (−1 to +1)"); axes[1].set_ylabel("Count")
axes[1].axvline(0.05,  color=GREEN, linestyle="--", linewidth=1.5, label="+0.05 threshold")
axes[1].axvline(-0.05, color=ACCENT, linestyle="--", linewidth=1.5, label="-0.05 threshold")
axes[1].legend()
plt.tight_layout(); save(fig, "A4_feedback_and_sentiment_dist.png")

# =============================================================================
# B. MODEL EVALUATION
# =============================================================================
print("\n[B] Model evaluation plots...")

train_acc  = accuracy_score(y_train, y_train_pred)
test_acc   = accuracy_score(y_test,  y_pred)
train_f1   = f1_score(y_train, y_train_pred, average="weighted")
test_f1    = f1_score(y_test,  y_pred,       average="weighted")
train_prec = precision_score(y_train, y_train_pred, average="weighted")
test_prec  = precision_score(y_test,  y_pred,       average="weighted")
train_rec  = recall_score(y_train, y_train_pred, average="weighted")
test_rec   = recall_score(y_test,  y_pred,       average="weighted")
roc_auc    = auc(*roc_curve(y_test, y_prob)[:2])
ap         = average_precision_score(y_test, y_prob)
gap        = train_acc - test_acc

# B1 — Confusion matrix
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(f"B1 · Confusion Matrix  (C={saved_C})", fontsize=15, fontweight="bold")
cm     = confusion_matrix(y_test, y_pred)
cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
ConfusionMatrixDisplay(cm, display_labels=["Not Rec.","Rec."]).plot(
    ax=axes[0], colorbar=False, cmap="Blues")
axes[0].set_title("Raw Counts")
cmap2 = LinearSegmentedColormap.from_list("wp", ["white", GREEN])
im = axes[1].imshow(cm_pct, cmap=cmap2, vmin=0, vmax=100)
axes[1].set_xticks([0,1]); axes[1].set_yticks([0,1])
axes[1].set_xticklabels(["Not Rec.","Rec."])
axes[1].set_yticklabels(["Not Rec.","Rec."])
axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("Actual")
axes[1].set_title("Normalised (%)")
for i in range(2):
    for j in range(2):
        axes[1].text(j, i, f"{cm_pct[i,j]:.1f}%", ha="center", va="center",
                     fontsize=14, fontweight="bold",
                     color="white" if cm_pct[i,j]>50 else "black")
plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
plt.tight_layout(); save(fig, "B1_confusion_matrix.png")

# B2 — ROC + PR
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("B2 · ROC & Precision-Recall Curves", fontsize=15, fontweight="bold")
fpr, tpr, _ = roc_curve(y_test, y_prob)
axes[0].plot(fpr, tpr, color=GREEN, linewidth=2.5, label=f"AUC = {roc_auc:.4f}")
axes[0].plot([0,1],[0,1], color="grey", linestyle="--")
axes[0].fill_between(fpr, tpr, alpha=0.08, color=GREEN)
axes[0].set_xlabel("FPR"); axes[0].set_ylabel("TPR")
axes[0].set_title("ROC Curve"); axes[0].legend(fontsize=12)

prec_arr, rec_arr, _ = precision_recall_curve(y_test, y_prob)
axes[1].plot(rec_arr, prec_arr, color=BLUE, linewidth=2.5, label=f"AP = {ap:.4f}")
axes[1].axhline(y.mean(), color="grey", linestyle="--", label=f"Baseline = {y.mean():.3f}")
axes[1].fill_between(rec_arr, prec_arr, alpha=0.08, color=BLUE)
axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
axes[1].set_title("Precision-Recall Curve"); axes[1].legend(fontsize=12)
plt.tight_layout(); save(fig, "B2_roc_pr_curves.png")

# B3 — Train vs Test bar
metrics_names = ["Accuracy","F1 (weighted)","Precision","Recall"]
train_vals = [train_acc, train_f1, train_prec, train_rec]
test_vals  = [test_acc,  test_f1,  test_prec,  test_rec]

fig, ax = plt.subplots(figsize=(11, 6))
fig.suptitle(f"B3 · Train vs Test Metrics  (C={saved_C})", fontsize=15, fontweight="bold")
x = np.arange(len(metrics_names)); w = 0.35
bars1 = ax.bar(x-w/2, train_vals, w, label="Train", color=BLUE,  edgecolor="white")
bars2 = ax.bar(x+w/2, test_vals,  w, label="Test",  color=GREEN, edgecolor="white")
ax.set_xticks(x); ax.set_xticklabels(metrics_names)
ax.set_ylabel("Score"); ax.set_ylim(0, 1.12); ax.legend()
ax.set_title("Δ < 0.05 = healthy  |  Δ > 0.05 = investigate")
for bar in bars1:
    ax.annotate(f"{bar.get_height():.4f}",
                xy=(bar.get_x()+bar.get_width()/2, bar.get_height()),
                ha="center", va="bottom", fontsize=9, color=BLUE, fontweight="bold")
for bar in bars2:
    ax.annotate(f"{bar.get_height():.4f}",
                xy=(bar.get_x()+bar.get_width()/2, bar.get_height()),
                ha="center", va="bottom", fontsize=9, color=GREEN, fontweight="bold")
for i, (tv, sv) in enumerate(zip(train_vals, test_vals)):
    g = tv - sv
    ax.annotate(f"Δ {g:+.4f}", xy=(x[i], max(tv,sv)+0.03),
                ha="center", fontsize=8.5,
                color=ACCENT if g>0.05 else "black", fontweight="bold")
plt.tight_layout(); save(fig, "B3_train_vs_test_metrics.png")

# =============================================================================
# C. LEARNING CURVES
# =============================================================================
print("\n[C] Learning curves (~2 min)...")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle(f"C · Learning Curves  (C={saved_C})", fontsize=15, fontweight="bold")
train_sizes = np.linspace(0.05, 1.0, 12)

for ax, scoring, label in [
    (axes[0], "accuracy",    "Accuracy"),
    (axes[1], "f1_weighted", "F1 (weighted)"),
]:
    tr_s, tr_sc, cv_sc = learning_curve(
        pipeline, X, y, train_sizes=train_sizes,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring=scoring, n_jobs=-1)
    tr_m = tr_sc.mean(axis=1); tr_std = tr_sc.std(axis=1)
    cv_m = cv_sc.mean(axis=1); cv_std = cv_sc.std(axis=1)
    ax.plot(tr_s, tr_m, "o-",  color=BLUE,  linewidth=2, label="Train")
    ax.fill_between(tr_s, tr_m-tr_std, tr_m+tr_std, alpha=0.12, color=BLUE)
    ax.plot(tr_s, cv_m, "s--", color=GREEN, linewidth=2, label="CV (5-fold)")
    ax.fill_between(tr_s, cv_m-cv_std, cv_m+cv_std, alpha=0.12, color=GREEN)
    g = tr_m[-1] - cv_m[-1]
    ax.set_title(f"{label}  (gap at full: {g:+.4f})")
    ax.set_xlabel("Training Size"); ax.set_ylabel(label)
    ax.set_ylim(0.5, 1.02); ax.legend(fontsize=10)
    ax.axhline(cv_m[-1], color="grey", linestyle=":", linewidth=1)
plt.tight_layout(); save(fig, "C_learning_curves.png")

# =============================================================================
# D. C SWEEP — dual criterion: test acc AND gap <= 0.05
# =============================================================================
print("\n[D] Hyperparameter sweep...")

C_values = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0]
cv_accs, test_accs, train_accs, test_f1s, gaps = [], [], [], [], []

for C in C_values:
    p = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=10000, ngram_range=(1,2))),
        ("lr",    LogisticRegression(C=C, solver="lbfgs", max_iter=1000,
                                     class_weight="balanced", random_state=42))
    ])
    cv_a = cross_val_score(p, X_train, y_train, cv=5, scoring="accuracy", n_jobs=-1)
    cv_accs.append(cv_a.mean())
    p.fit(X_train, y_train)
    tr_a = accuracy_score(y_train, p.predict(X_train))
    te_a = accuracy_score(y_test,  p.predict(X_test))
    train_accs.append(tr_a); test_accs.append(te_a)
    test_f1s.append(f1_score(y_test, p.predict(X_test), average="weighted"))
    gaps.append(tr_a - te_a)

# Best C = highest test acc where gap <= 0.05
candidates = [(C_values[i], test_accs[i]) for i in range(len(C_values)) if gaps[i] <= 0.05]
if candidates:
    best_C_balanced = max(candidates, key=lambda x: x[1])[0]
    best_method = "highest test acc with gap ≤ 0.05"
else:
    best_C_balanced = C_values[np.argmax([test_accs[i] - gaps[i] for i in range(len(C_values))])]
    best_method = "best acc-gap trade-off (no C met gap ≤ 0.05)"

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("D · C Sweep: Test Accuracy vs Generalisation Gap", fontsize=15, fontweight="bold")

ax = axes[0]
ax.semilogx(C_values, train_accs, "o-",  color=BLUE,  linewidth=2,   label="Train acc")
ax.semilogx(C_values, test_accs,  "s--", color=GREEN, linewidth=2,   label="Test acc")
ax.semilogx(C_values, cv_accs,    "^:",  color=BLUE,  linewidth=1.5, alpha=0.55, label="CV acc")
ax.axvline(best_C_balanced, color=ACCENT, linewidth=2.5, label=f"Recommended C={best_C_balanced}")
ax.axvline(saved_C, color="grey", linestyle="--", linewidth=1.5, alpha=0.7, label=f"Current C={saved_C}")
ax.set_xlabel("C  (← more regularised | less regularised →)")
ax.set_ylabel("Accuracy"); ax.set_title("Train / Test / CV Accuracy"); ax.legend(fontsize=8.5)

ax2 = axes[1]
bar_colors = [ACCENT if g>0.05 else GREEN for g in gaps]
ax2.bar([str(c) for c in C_values], gaps, color=bar_colors, edgecolor="white")
ax2.axhline(0.05, color=ACCENT, linestyle="--", linewidth=2, label="Overfit threshold (0.05)")
ax2.set_xlabel("C"); ax2.set_ylabel("Train − Test Gap")
ax2.set_title("Generalisation Gap  (red = overfitting)"); ax2.legend()
for i, g in enumerate(gaps):
    ax2.text(i, g+0.001, f"{g:.3f}", ha="center", fontsize=8.5,
             color="white" if g>0.05 else "black", fontweight="bold")
plt.tight_layout(); save(fig, "D_hyperparam_C_sweep.png")

# =============================================================================
# E. FEATURE IMPORTANCE
# =============================================================================
print("\n[E] Feature importance...")

feature_names = pipeline.named_steps["tfidf"].get_feature_names_out()
coefs         = pipeline.named_steps["lr"].coef_[0]
N = 20
top_pos = np.argsort(coefs)[::-1][:N]
top_neg = np.argsort(coefs)[:N]

fig, axes = plt.subplots(1, 2, figsize=(16, 8))
fig.suptitle("E · LR Coefficients — Words Driving Each Prediction",
             fontsize=15, fontweight="bold")
axes[0].barh([feature_names[i] for i in top_pos[::-1]],
             [coefs[i] for i in top_pos[::-1]], color=GREEN, edgecolor="white")
axes[0].set_title(f"Top {N} → RECOMMEND"); axes[0].set_xlabel("Coefficient (log-odds)")
axes[1].barh([feature_names[i] for i in top_neg],
             [coefs[i] for i in top_neg], color=ACCENT, edgecolor="white")
axes[1].set_title(f"Top {N} → NOT RECOMMEND"); axes[1].set_xlabel("Coefficient (log-odds)")
plt.tight_layout(); save(fig, "E_feature_importance_lr_coefficients.png")

# =============================================================================
# F. SENTIMENT DEEP DIVE
# =============================================================================
print("\n[F] Sentiment plots...")

df["sentiment_label"] = df["sentiment_score"].apply(
    lambda s: "positive" if s>=0.05 else ("negative" if s<=-0.05 else "neutral"))

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("F · Sentiment Deep Dive", fontsize=15, fontweight="bold")

sent_counts = df["sentiment_label"].value_counts()
axes[0,0].bar(sent_counts.index, sent_counts.values,
              color=[GREEN if s=="positive" else (ACCENT if s=="negative" else BLUE)
                     for s in sent_counts.index], edgecolor="white")
axes[0,0].set_title("VADER Sentiment Label Counts")
for bar in axes[0,0].patches:
    axes[0,0].annotate(f"{bar.get_height():,}",
                       xy=(bar.get_x()+bar.get_width()/2, bar.get_height()),
                       ha="center", va="bottom", fontsize=10)

rating_sent = df.groupby("Rating")["sentiment_score"].mean()
axes[0,1].bar(rating_sent.index.astype(str), rating_sent.values,
              color=[ACCENT if r<=2 else (PALETTE[6] if r==3 else GREEN)
                     for r in rating_sent.index], edgecolor="white")
axes[0,1].set_title("Avg VADER Score by Rating")
axes[0,1].set_xlabel("Stars"); axes[0,1].set_ylabel("Avg Compound")

rec_0 = df[df["Recommended IND"]==0]["sentiment_score"].dropna()
rec_1 = df[df["Recommended IND"]==1]["sentiment_score"].dropna()
vp = axes[1,0].violinplot([rec_0, rec_1], positions=[0,1], showmedians=True)
vp["bodies"][0].set_facecolor(ACCENT); vp["bodies"][0].set_alpha(0.6)
vp["bodies"][1].set_facecolor(GREEN);  vp["bodies"][1].set_alpha(0.6)
axes[1,0].set_xticks([0,1]); axes[1,0].set_xticklabels(["Not Rec.","Rec."])
axes[1,0].set_title("Sentiment by Recommendation Label")
axes[1,0].set_ylabel("VADER Compound")

sample = df.sample(min(3000, len(df)), random_state=42).copy()
sample["pred_prob"] = pipeline.predict_proba(sample["clean_text"])[:,1]
sc = axes[1,1].scatter(sample["sentiment_score"], sample["pred_prob"],
                        c=sample["Recommended IND"],
                        cmap=LinearSegmentedColormap.from_list("rg",[ACCENT,GREEN]),
                        alpha=0.25, s=12)
axes[1,1].set_xlabel("VADER Score"); axes[1,1].set_ylabel("LR Recommend Prob")
axes[1,1].set_title("VADER vs LR Probability  (colour=actual label)")
plt.colorbar(sc, ax=axes[1,1], label="Actual Label")
plt.tight_layout(); save(fig, "F_sentiment_deep_dive.png")

# =============================================================================
# G. CATEGORY INSIGHTS
# =============================================================================
print("\n[G] Category insights...")

cat_df = pd.read_csv(os.path.join("outputs","category_sentiment.csv"))

fig, axes = plt.subplots(2, 2, figsize=(18, 13))
fig.suptitle("G · Category-Level Insights", fontsize=15, fontweight="bold")

dept_sent = cat_df.groupby("department")["avg_sentiment"].mean().sort_values()
axes[0,0].barh(dept_sent.index, dept_sent.values,
               color=[GREEN if v>=0.5 else ACCENT for v in dept_sent.values], edgecolor="white")
axes[0,0].set_title("Avg Sentiment by Department"); axes[0,0].set_xlabel("Avg Compound Score")
axes[0,0].axvline(0.5, color="grey", linestyle="--", linewidth=1)

axes[0,1].scatter(cat_df["recommend_rate"], cat_df["avg_sentiment"],
                  s=cat_df["review_count"]/5, alpha=0.65, c=cat_df["avg_rating"],
                  cmap="RdYlGn", edgecolors="white", linewidths=0.5)
axes[0,1].set_xlabel("Recommend Rate"); axes[0,1].set_ylabel("Avg Sentiment")
axes[0,1].set_title("Rec Rate vs Sentiment  (size=volume, colour=rating)")
plt.colorbar(plt.cm.ScalarMappable(cmap="RdYlGn",
    norm=plt.Normalize(cat_df["avg_rating"].min(), cat_df["avg_rating"].max())),
    ax=axes[0,1], label="Avg Rating")

top_cls = cat_df.sort_values("review_count", ascending=False).head(15)
axes[1,0].barh(top_cls["class_name"], top_cls["review_count"],
               color=[GREEN if r>=0.82 else ACCENT for r in top_cls["recommend_rate"]],
               edgecolor="white")
axes[1,0].set_title("Top 15 Classes  (green=rec≥82%)"); axes[1,0].set_xlabel("Reviews")

dept_rating = df.groupby("Department Name")["Rating"].mean().sort_values()
axes[1,1].barh(dept_rating.index, dept_rating.values,
               color=[GREEN if v>=4.0 else (PALETTE[6] if v>=3.5 else ACCENT)
                      for v in dept_rating.values], edgecolor="white")
axes[1,1].set_title("Avg Star Rating by Department"); axes[1,1].set_xlabel("Avg Rating")
axes[1,1].axvline(4.0, color="grey", linestyle="--", linewidth=1, label="4.0")
axes[1,1].legend()
for i, v in enumerate(dept_rating.values):
    axes[1,1].text(v+0.01, i, f"{v:.2f}", va="center", fontsize=9)
plt.tight_layout(); save(fig, "G_category_insights.png")

# =============================================================================
# SUMMARY
# =============================================================================
if gap > 0.05:
    gap_verdict = f"⚠️  GAP={gap:.4f} — mild overfit (train learns more than it generalises)"
elif gap < -0.01:
    gap_verdict = f"⚠️  GAP={gap:.4f} — train < test, check data leakage"
else:
    gap_verdict = f"✅  GAP={gap:.4f} — healthy generalisation"

print("\n" + "="*65)
print("  ANALYSIS SUMMARY")
print("="*65)
print(f"  C loaded from pkl       : {saved_C}")
print(f"  Dataset                 : {len(df):,} reviews")
print(f"  Class balance           : {y.mean()*100:.1f}% recommended")
print(f"\n  ── Model Performance ──")
print(f"  Train accuracy          : {train_acc:.4f}")
print(f"  Test  accuracy          : {test_acc:.4f}")
print(f"  Train F1 (weighted)     : {train_f1:.4f}")
print(f"  Test  F1 (weighted)     : {test_f1:.4f}")
print(f"  ROC-AUC                 : {roc_auc:.4f}")
print(f"  Avg Precision (PR AUC)  : {ap:.4f}")
print(f"\n  {gap_verdict}")
print(f"\n  ── C Sweep ──")
print(f"  Best balanced C         : {best_C_balanced}  ({best_method})")
if best_C_balanced != saved_C:
    print(f"  ACTION → change C={saved_C} to C={best_C_balanced} in train_nlp.py and retrain")
else:
    print(f"  ✅  C={saved_C} is already the optimal value")
print(f"\n  ── VADER ──")
for label, pct in (df["sentiment_label"].value_counts(normalize=True)*100).items():
    print(f"  {label:<12}: {pct:.1f}%")
print(f"\n  Plots saved → outputs/analysis/\n")