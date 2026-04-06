# ⚙️ SMAT.AI — Smart Machine Analysis & Telemetry

> AI-powered predictive maintenance system with automated short & long report generation.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3+-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=flat&logo=docker&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-22c55e?style=flat)

---

## 📌 Overview

**SMAT.AI** is a predictive maintenance platform that uses machine learning to monitor machine health, detect potential failures before they happen, and automatically generate both short and detailed maintenance reports.

Built for industrial use cases where downtime is costly — SMAT.AI turns raw sensor data into actionable intelligence.

---

## 🚀 Features

| Feature | Description |
|---|---|
| 📂 CSV Upload | Upload any machine sensor dataset via sidebar |
| 🧹 Auto Preprocessing | Automatic encoding, ID column removal, null handling |
| 🤖 ML Model Training | Random Forest classifier trained on your data |
| 📊 Model Evaluation | Accuracy score, classification report, feature importance |
| ⚡ Live Prediction | Enter sensor values manually → instant failure prediction |
| 📋 Report Generation | Auto-generates short & detailed reports for every machine |
| ⬇️ CSV Download | Download all reports with predictions as a CSV file |
| 🐳 Docker Ready | One-command deployment with included Dockerfile |

---

## 🖥️ App Structure

The app is organized into **3 tabs**:

```
Tab 1 — DATASET & MODEL
   → Upload CSV
   → View data preview & stats
   → Train Random Forest model
   → See accuracy + feature importance

Tab 2 — LIVE PREDICTION
   → Enter sensor readings manually
   → Get instant HEALTHY / FAILURE prediction
   → View failure probability score
   → Auto-generate report for that input

Tab 3 — REPORTS
   → View all generated reports
   → Browse random machine reports
   → Download full report CSV
```

---

## 🗂️ Project Structure

```
smatai-gaurav/
├── app.py                # Main Streamlit application (v4.0)
├── requirements.txt      # Python dependencies
├── Dockerfile            # Docker deployment config
├── sample_data.csv       # Sample dataset for testing
└── README.md             # This file
```

---

## 📦 Sample Data Format

Your CSV should contain machine sensor readings. Minimum required columns:

| Column | Type | Example |
|---|---|---|
| `Air temperature [K]` | float | 298.1 |
| `Process temperature [K]` | float | 308.6 |
| `Rotational speed [rpm]` | int | 1551 |
| `Torque [Nm]` | float | 42.8 |
| `Tool wear [min]` | int | 0 |
| `Machine failure` | int (0/1) | 0 |

> The app also auto-detects columns named: `Failure`, `Target`, `Anomaly`

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
streamlit run app.py
```

App opens at: **http://localhost:8501**

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
4. Deploy ✅

### Deploy to Azure Container Apps

```bash
az acr build --registry <your-registry> --image smatai-ai .
az containerapp create --name smatai-ai --image <your-registry>.azurecr.io/smatai-ai --target-port 8501
```

---

## 🧠 How It Works

```
CSV Upload
    ↓
Auto Preprocessing (encode categoricals, drop IDs)
    ↓
Train Random Forest (100 estimators, 80/20 split)
    ↓
Evaluate → Accuracy + Classification Report + Feature Importance
    ↓
┌────────────────────┬──────────────────────────┐
│   Batch Reports    │    Live Manual Prediction │
│   (all rows)       │    (user enters values)   │
└────────────────────┴──────────────────────────┘
    ↓
Short Report + Long Report + Download CSV
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Frontend / UI | Streamlit |
| ML Model | Scikit-learn (Random Forest) |
| Data Processing | Pandas, NumPy |
| Deployment | Docker, Render, Azure |
| Language | Python 3.10+ |

---

## 📈 Model Details

- **Algorithm:** Random Forest Classifier
- **Estimators:** 100 trees
- **Train/Test Split:** 80% / 20%
- **Stratified Split:** Yes (handles class imbalance)
- **Feature Importance:** Displayed as ranked bar chart
- **Output:** Binary classification (0 = Healthy, 1 = Failure)

---

## 🔮 Roadmap

- [ ] XGBoost & model comparison tab
- [ ] Real-time streaming sensor data (MQTT/Kafka)
- [ ] Email alerts on predicted failures
- [ ] PostgreSQL support for persistent ticket storage
- [ ] Multi-machine dashboard view
- [ ] REST API endpoint for external integration

---

## 👨‍💻 Author

**Gaurav Shukla**
AI & Data Engineer | B.Tech CSE (AI & ML)

[![GitHub](https://img.shields.io/badge/GitHub-gaurav--s23-181717?style=flat&logo=github)](https://github.com/gaurav-s23)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Gaurav%20Shukla-0A66C2?style=flat&logo=linkedin)](https://www.linkedin.com/in/gaurav-shukla-406934290/)

---

## 📄 License

This project is licensed under the MIT License — feel free to use, modify, and distribute.

---

<div align="center">
  <sub>Built with ❤️ for industrial AI applications</sub>
</div>

