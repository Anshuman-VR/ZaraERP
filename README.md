# Zara Intelligence Platform (ZaraERP)

An AI-driven intelligence dashboard for Zara, synthesizing demand forecasting and NLP sentiment analysis into actionable business strategies.

## 🚀 Overview
This platform provides three core analytical engines:
1.  **Demand Forecasting**: Uses a retrained XGBoost model to predict weekly sales volume based on product attributes and marketing levers.
2.  **Sentiment Analysis**: A TF-IDF + Logistic Regression pipeline that processes customer reviews to extract sentiment scores and recommendation probabilities.
3.  **Unified Strategic IQ**: A fusion system that correlates NLP sentiment with demand velocity to generate macroscopic business guidance.

---

## 🧠 Machine Learning Methodology

### Synthetic Temporal Engineering
The original dataset lacked a time dimension (Sales Volume was lifetime total). To enable professional "units/week" forecasting, we engineered a **Synthetic Shelf-Life Pipeline**:
*   **Dynamic Lifecycle**: Products are assigned a base 12-week lifecycle.
*   **Marketing Impact**: Seasonality (`-6 weeks`) and Promotions (`-40% duration`) accelerate lifecycle velocity.
*   **Target Calculation**: Weekly volume is calculated as `Sales_Volume / Shelf_Life_Weeks`.
*   **Retrained Model**: The XGBoost model was retrained on this new high-velocity metric with heavy sample weighting for Promotional and Seasonal features.

### Actionable Insights
The dashboard doesn't just output numbers; it transforms model outputs into structured strategies:
*   **Inventory**: Dynamic scale recommendations for Manufacturing/MOQs.
*   **Logistics**: Multi-tier shipping channel selection (Air vs. Sea).
*   **Pricing**: Inelasticity-based margin optimization guides.

---

## 🛠 Project Structure
```text
ZaraERP/
├── backend/            # FastAPI Predictive API
│   ├── app.py          # Real-time inference engine
│   ├── fusion_model.py # Training pipeline (Retrained with Weekly Metrics)
│   └── outputs/        # Serialized .pkl models & scalers
├── data/               # Zara Sales & Reviews datasets (CSV)
└── frontend/           # Intelligence Dashboard
    ├── dashboard.html  # Glassmorphic UI & Insight Generators
    └── assets/         # UI images & SHAP visualizations
```

---

## ⚙️ Setup & Execution

### 1. Requirements
Ensure you have Python 3.10+ installed. Install all dependencies using the provided package list:

```powershell
pip install -r requirements.txt
```

### 2. Launch Backend
Run the FastAPI server to handle real-time AI inference:
```powershell
cd backend
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

### 3. Open Dashboard
Directly open the HTML file in your browser:
`file:///.../ZaraERP/frontend/dashboard.html`

---

## ⚡ Key Achievements
*   ✅ **Fixed Under-Prediction**: Recalibrated shelf-life from 24 to 12 weeks to double predicted volume scales.
*   ✅ **Aggressive Responsiveness**: The model now reacts heavily to Promotion/Seasonal toggles.
*   ✅ **No More NaN**: Fixed probability rendering anomalies in Unified Analysis.
*   ✅ **Semantic Alignment**: Replaced "Cumulative Sales" with "Weekly Velocity" for industry-standard reporting.
