# ⚙️ SMAT.AI v6.0 — Smart Machine Analysis & Telemetry

> AI-powered predictive maintenance system with Power BI-style dashboards, multi-model ML, SHAP explainability, MLflow tracking, and automated report generation.

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3+-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-FF6B35?style=flat)
![SHAP](https://img.shields.io/badge/SHAP-0.44+-blueviolet?style=flat)
![MLflow](https://img.shields.io/badge/MLflow-2.10+-0194E2?style=flat&logo=mlflow)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=flat&logo=docker&logoColor=white)
![Azure](https://img.shields.io/badge/Azure-Deployed-0078D4?style=flat&logo=microsoft-azure&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-22c55e?style=flat)

---

## 📌 Overview

**SMAT.AI** is a full-stack predictive maintenance platform that uses machine learning to monitor machine health, detect potential failures before they happen, and automatically generate maintenance reports.

Built for industrial use cases where downtime is costly — SMAT.AI turns raw sensor data into actionable intelligence with a **Power BI-style interactive dashboard**, **multi-model ML comparison**, **SHAP explainability**, and **live API prediction**.

---

## 🚀 What's New in v6.0

| Feature | v5.0 | v6.0 |
|---|---|---|
| Power BI Dashboard | ❌ | ✅ Full BI tab with KPIs, trend chart, radar |
| Models | RF + XGB | RF + XGB + GradientBoosting |
| Precision-Recall Curve | ❌ | ✅ |
| Radar Chart Model Comparison | ❌ | ✅ |
| Sensor Boxplot Analysis | ❌ | ✅ |
| 2D Failure Density Heatmap | ❌ | ✅ |
| Random Sample Auto-Fill | ❌ | ✅ |
| Failure Probability in Reports | ❌ | ✅ |
| Filter by Healthy/Failure | ❌ | ✅ |
| SHAP on Live Prediction | ❌ | ✅ |
| Code Split (app.py + utils.py) | ❌ | ✅ |
| Live REST API Prediction Guide | ❌ | ✅ |

---

## 🖥️ App Structure — 6 Tabs

```
Tab 1 — 📊 BI DASHBOARD
   → Power BI-style KPI cards (total machines, failure rate, model accuracy, AUC)
   → Failure rate trend chart across the full dataset
   → Model comparison radar chart (Accuracy vs F1 vs AUC vs CV F1)
   → Model comparison table with best model highlighted
   → Top feature importance bars
   → Sensor boxplots (Healthy vs Failure distributions)
   → 2D failure density heatmap (top 2 sensors)

Tab 2 — 🔬 DATASET & EDA
   → Upload CSV via sidebar
   → Auto-detect target column
   → Dataset overview KPIs (rows, nulls, failure count)
   → Raw data preview + descriptive stats
   → Class distribution chart
   → Missing value map
   → Correlation heatmap (all numeric features)
   → Feature distributions by class (Healthy vs Failure)

Tab 3 — 🤖 MODEL TRAINING
   → Train Random Forest + XGBoost + GradientBoosting simultaneously
   → MLflow auto-logging (params, metrics, models)
   → Confusion matrix per model
   → ROC curve per model
   → Precision-Recall curve per model
   → Classification report per model
   → Cross-validation scores (5-fold)
   → Feature importance bars for best model
   → SHAP values computed automatically

Tab 4 — ⚡ LIVE PREDICTION
   → Enter sensor readings manually (with value ranges shown)
   → Auto-fill random sample from dataset with one click
   → Instant HEALTHY / FAILURE prediction with probability
   → Animated probability breakdown bars
   → SHAP explanation for this specific prediction
   → Auto-generated detailed report for this input

Tab 5 — 🧠 EXPLAINABILITY (SHAP)
   → Beeswarm summary plot (global feature impact)
   → Mean |SHAP| bar chart (global feature importance)
   → Per-sample waterfall-style bar chart
   → Interactive sample selector

Tab 6 — 📋 REPORTS
   → Full report dashboard with failure/healthy KPIs
   → Random machine report browser
   → Filter: All / Only Failures / Only Healthy
   → Download all reports + predictions as CSV
```

---

## 🗂️ Project Structure

```
smatai-gaurav/
├── app.py                # Main Streamlit application (v6.0) — UI + tabs
├── utils.py              # All helpers: ML utils, plots, report generators
├── requirements.txt      # Python dependencies
├── Dockerfile            # Docker deployment config
├── sample_data.csv       # Sample dataset for testing
├── .github/
│   └── workflows/
│       └── main_smatai-gaurav.yml  # Azure CI/CD pipeline
└── README.md             # This file
```

### Why two files?
- **`app.py`** — Only Streamlit UI code, tab layout, session state, user interactions.
- **`utils.py`** — All helper functions: preprocessing, plots, report generation, SHAP. This keeps `app.py` lean and `utils.py` independently testable.

---

## 📦 Sample Data Format

Your CSV should contain machine sensor readings. Required columns:

| Column | Type | Example |
|---|---|---|
| `Air temperature [K]` | float | 298.1 |
| `Process temperature [K]` | float | 308.6 |
| `Rotational speed [rpm]` | int | 1551 |
| `Torque [Nm]` | float | 42.8 |
| `Tool wear [min]` | int | 0 |
| `Machine failure` | int (0/1) | 0 |

**Auto-detected target column names:** `Machine failure`, `Failure`, `Target`, `Anomaly`, `label`

The app auto-drops ID columns (`UDI`, `Product ID`, etc.) and handles missing values automatically.

---

## ⚙️ Local Setup

### 1. Clone the repository

```bash
git clone https://github.com/gaurav-s23/smatai-gaurav.git
cd smatai-gaurav
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the app

```bash
python -m streamlit run app.py
```

App opens at: **http://localhost:8501**

---

## 📡 Live Prediction via REST API

SMAT.AI supports live sensor data prediction through a FastAPI endpoint. Here's how to set it up:

### Step 1 — Train and export the model

After training in the app, export the best model from Python:

```python
import pickle
import streamlit as st

# After training in the app, the model is available as:
# st.session_state.model  →  sklearn/XGBoost classifier

# Export
with open("smat_model.pkl", "wb") as f:
    pickle.dump(model, f)

# Also save the feature column list
import json
with open("feature_cols.json", "w") as f:
    json.dump(feature_cols, f)
```

### Step 2 — Create a FastAPI server

```bash
pip install fastapi uvicorn
```

Create `api.py`:

```python
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd
import json

app = FastAPI(title="SMAT.AI Prediction API")

with open("smat_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("feature_cols.json") as f:
    feature_cols = json.load(f)

class SensorInput(BaseModel):
    data: dict  # {feature_name: value, ...}

@app.post("/predict")
def predict(payload: SensorInput):
    df = pd.DataFrame([payload.data])[feature_cols]
    prediction = int(model.predict(df)[0])
    probability = float(model.predict_proba(df)[0][1])
    return {
        "prediction": prediction,
        "status": "FAILURE" if prediction == 1 else "HEALTHY",
        "failure_probability": round(probability, 4),
    }

@app.get("/health")
def health():
    return {"status": "ok"}
```

### Step 3 — Start the API

```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

API docs available at: **http://localhost:8000/docs**

### Step 4 — Send live sensor data

Using Python:

```python
import requests

sensor_reading = {
    "data": {
        "Air temperature [K]": 298.5,
        "Process temperature [K]": 308.2,
        "Rotational speed [rpm]": 1551,
        "Torque [Nm]": 42.8,
        "Tool wear [min]": 120,
        "Type": 0  # encoded
    }
}

response = requests.post("http://localhost:8000/predict", json=sensor_reading)
print(response.json())
# → {"prediction": 0, "status": "HEALTHY", "failure_probability": 0.023}
```

Using cURL:

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"data": {"Air temperature [K]": 298.5, "Rotational speed [rpm]": 1551}}'
```

### Connecting MQTT / Kafka (Real-time Streaming)

For real-time sensor streams (e.g., from industrial IoT devices via MQTT):

```python
import paho.mqtt.client as mqtt
import requests, json

def on_message(client, userdata, msg):
    sensor_data = json.loads(msg.payload)
    response = requests.post("http://localhost:8000/predict", json={"data": sensor_data})
    result = response.json()
    if result["status"] == "FAILURE":
        print(f"⚠ ALERT: Failure predicted! Prob={result['failure_probability']:.1%}")

client = mqtt.Client()
client.on_message = on_message
client.connect("your-mqtt-broker", 1883)
client.subscribe("factory/machine/sensors")
client.loop_forever()
```

---

## 🐳 Docker Deployment

### Build & run locally

```bash
docker build -t smatai-ai .
docker run -p 8501:8501 smatai-ai
```

### Deploy to Render / Railway

1. Connect your GitHub repo
2. Set build command: `pip install -r requirements.txt`
3. Set start command: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`

### Deploy to Azure Container Apps

```bash
az acr build --registry <your-registry> --image smatai-ai .
az containerapp create --name smatai-ai --image <your-registry>.azurecr.io/smatai-ai --target-port 8501
```

### ☁️ Automated Azure CI/CD

1. Fork this repository
2. Create Azure Web App (Python 3.10+)
3. Add GitHub Secrets:
   - `AZUREAPPSERVICE_CLIENTID_...`
   - `AZUREAPPSERVICE_TENANTID_...`
   - `AZUREAPPSERVICE_SUBSCRIPTIONID_...`
4. Push to `main` → Auto-deploys

---

## 🧠 How It Works

```
CSV Upload
    ↓
Auto Preprocessing
  → Detect target column
  → Drop ID columns
  → Fill missing values (median/mode)
  → Label-encode categoricals
    ↓
Train All Models in Parallel
  → Random Forest (100 estimators, stratified split)
  → XGBoost (configurable LR + depth)
  → GradientBoosting (optional)
  → 5-fold cross-validation per model
  → Log all to MLflow (params + metrics + artifacts)
    ↓
Auto-Select Best Model (by weighted F1)
    ↓
Compute SHAP Values (TreeExplainer on best model)
    ↓
Generate Batch Reports (all rows → short + long + probability)
    ↓
┌───────────────────┬────────────────────┬────────────────────┐
│   BI Dashboard    │   Live Prediction  │   Reports & Filter  │
│   (KPIs + charts) │   (manual input)   │   (CSV download)    │
└───────────────────┴────────────────────┴────────────────────┘
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Frontend / UI | Streamlit 1.32+ |
| ML Models | Scikit-learn (Random Forest, GradientBoosting), XGBoost 2.0+ |
| Data Processing | Pandas 2.0+, NumPy 1.24+ |
| Visualization | Matplotlib 3.7+, Seaborn 0.13+ |
| Model Explainability | SHAP 0.44+ (TreeExplainer) |
| ML Tracking | MLflow 2.10+ |
| Live API | FastAPI + Uvicorn (optional) |
| Real-time Streaming | MQTT (paho-mqtt), Kafka (optional) |
| Deployment | Docker, Render, Azure Web Apps |
| CI/CD | GitHub Actions |
| Language | Python 3.11+ |

---

## 📈 Model Details

| Model | Algorithm | Key Params |
|---|---|---|
| Random Forest | Ensemble (bagging) | 100 trees, stratified split |
| XGBoost | Gradient boosting | Configurable LR + depth |
| GradientBoosting | Gradient boosting (sklearn) | Optional, slower |

**Selection criteria:** Best weighted F1 score on test set (handles class imbalance better than raw accuracy).

**Evaluation suite per model:**
- Accuracy
- Weighted F1
- ROC-AUC
- Precision-Recall (Average Precision)
- 5-Fold Cross-Validation F1
- Confusion Matrix

---

## 🔮 Roadmap

- [x] XGBoost + GradientBoosting integration
- [x] SHAP explainability (global + per-prediction)
- [x] MLflow model tracking
- [x] Automated Azure deployment (CI/CD)
- [x] Power BI-style BI dashboard
- [x] Multi-model comparison (radar chart)
- [x] Sensor boxplot & 2D density heatmap
- [x] Precision-Recall curve
- [x] Live REST API prediction guide (FastAPI)
- [x] Failure probability in all reports
- [x] Filter: All / Only Failures / Only Healthy
- [x] Auto-fill random sample in Live Prediction
- [x] Code split: app.py + utils.py
- [ ] Email alerts on predicted failures (SMTP)
- [ ] PostgreSQL persistent report storage
- [ ] Real-time MQTT/Kafka data streaming (built-in)
- [ ] Multi-machine comparison view
- [ ] Anomaly detection (Isolation Forest, Autoencoder)
- [ ] Time-series forecasting (LSTM)

---

## 👨‍💻 Author

**Gaurav Shukla**
AI & Data Engineer | B.Tech CSE (AI & ML)

[![GitHub](https://img.shields.io/badge/GitHub-gaurav--s23-181717?style=flat&logo=github)](https://github.com/gaurav-s23)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Gaurav%20Shukla-0A66C2?style=flat&logo=linkedin)](https://www.linkedin.com/in/gaurav-shukla-406934290/)

---

## 📄 License

MIT License — free to use, modify, and distribute.