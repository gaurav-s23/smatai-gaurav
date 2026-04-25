import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import mlflow
import mlflow.sklearn
import random
import io
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve, f1_score, ConfusionMatrixDisplay
)
from xgboost import XGBClassifier

# ─────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="SMAT.AI — Predictive Maintenance",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# MLflow Setup
# ─────────────────────────────────────────────
MLFLOW_TRACKING_URI = "./mlruns"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("SMAT-AI-Predictive-Maintenance")

# ─────────────────────────────────────────────
# Custom CSS — Industrial dark theme (unchanged)
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;700&family=Syne:wght@400;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Syne', sans-serif; }
.stApp { background: #0a0c0f; color: #e8e4dc; }
.main .block-container { padding: 2rem 2.5rem; max-width: 1400px; }
[data-testid="stSidebar"] { background: #0f1218 !important; border-right: 1px solid #1e2530; }
.smat-header {
    background: linear-gradient(135deg, #0f1218 0%, #141920 50%, #0a0c10 100%);
    border: 1px solid #1e2530; border-left: 4px solid #f5a623;
    border-radius: 4px; padding: 1.5rem 2rem; margin-bottom: 2rem;
}
.smat-header h1 { font-family: 'Syne', sans-serif; font-weight: 800; font-size: 2.2rem; color: #f5a623; margin: 0; }
.smat-header p { font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; color: #6b7280; margin: 0.3rem 0 0 0; letter-spacing: 0.08em; }
.version-badge {
    display: inline-block; background: rgba(245,166,35,0.12);
    border: 1px solid rgba(245,166,35,0.3); color: #f5a623;
    font-family: 'JetBrains Mono', monospace; font-size: 0.65rem;
    padding: 2px 10px; border-radius: 2px; margin-left: 1rem; vertical-align: middle;
}
.section-label {
    font-family: 'JetBrains Mono', monospace; font-size: 0.7rem;
    letter-spacing: 0.15em; color: #f5a623; text-transform: uppercase;
    margin-bottom: 0.75rem; border-bottom: 1px solid #1e2530; padding-bottom: 0.4rem;
}
.metric-card {
    background: #0f1218; border: 1px solid #1e2530; border-radius: 4px;
    padding: 1.2rem 1.4rem; position: relative; overflow: hidden;
}
.metric-card::after { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px; background: #f5a623; }
.metric-card.danger::after { background: #ef4444; }
.metric-card.ok::after { background: #22c55e; }
.metric-card.info::after { background: #3b82f6; }
.metric-value { font-family: 'JetBrains Mono', monospace; font-size: 1.8rem; font-weight: 700; color: #e8e4dc; line-height: 1; }
.metric-label { font-size: 0.72rem; color: #6b7280; margin-top: 0.4rem; letter-spacing: 0.05em; text-transform: uppercase; font-family: 'JetBrains Mono', monospace; }
.report-card { background: #0f1218; border: 1px solid #1e2530; border-radius: 4px; padding: 1.4rem; margin-bottom: 1rem; }
.report-card h4 { font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; letter-spacing: 0.1em; color: #f5a623; text-transform: uppercase; margin-bottom: 0.75rem; }
.report-body { font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; color: #9ca3af; line-height: 1.7; white-space: pre-wrap; }
.pred-box { border-radius: 4px; padding: 1.8rem 2rem; text-align: center; margin: 1rem 0; }
.pred-box.fail { background: rgba(239,68,68,0.08); border: 2px solid rgba(239,68,68,0.4); }
.pred-box.ok { background: rgba(34,197,94,0.08); border: 2px solid rgba(34,197,94,0.35); }
.pred-value { font-family: 'Syne', sans-serif; font-size: 2.5rem; font-weight: 800; line-height: 1; }
.pred-value.fail { color: #ef4444; }
.pred-value.ok { color: #22c55e; }
.stButton > button {
    background: #f5a623 !important; color: #0a0c0f !important; border: none !important;
    border-radius: 3px !important; font-family: 'JetBrains Mono', monospace !important;
    font-weight: 700 !important; font-size: 0.8rem !important;
}
.stTabs [data-baseweb="tab-list"] { background: #0f1218; border-bottom: 1px solid #1e2530; gap: 0; }
.stTabs [data-baseweb="tab"] {
    background: transparent !important; color: #6b7280 !important;
    font-family: 'JetBrains Mono', monospace !important; font-size: 0.75rem !important;
    letter-spacing: 0.08em !important; text-transform: uppercase !important;
    border-radius: 0 !important; padding: 0.7rem 1.4rem !important;
    border-bottom: 2px solid transparent !important;
}
.stTabs [aria-selected="true"] { color: #f5a623 !important; border-bottom: 2px solid #f5a623 !important; }
#MainMenu { visibility: hidden; } footer { visibility: hidden; } header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Session State
# ─────────────────────────────────────────────
defaults = {
    "model": None, "best_model_name": None, "feature_cols": None,
    "X_sample": None, "X_train": None, "X_test": None,
    "y_train": None, "y_test": None,
    "accuracy": None, "output_df": None, "col_map": {},
    "label_encoders": {}, "all_results": {}, "raw_data": None,
    "scaler": None, "shap_values": None, "explainer": None
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────
st.markdown("""
<div class="smat-header">
    <h1>⚙ SMAT.AI <span class="version-badge">v5.0</span></h1>
    <p>SMART MACHINE ANALYSIS & TELEMETRY — AI-POWERED PREDICTIVE MAINTENANCE SYSTEM</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="section-label">Dataset Upload</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload CSV file", type="csv", label_visibility="collapsed")
    st.markdown("---")
    st.markdown('<div class="section-label">System Info</div>', unsafe_allow_html=True)
    model_status = "TRAINED" if st.session_state.model else "NO MODEL"
    color = "#22c55e" if st.session_state.model else "#ef4444"
    best_name = st.session_state.best_model_name or "—"
    st.markdown(f"""
    <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; color: #6b7280; line-height: 2;">
        <div>STATUS: <span style="color:{color}; font-weight:700;">{model_status}</span></div>
        <div>BEST MODEL: <span style="color:#e8e4dc;">{best_name}</span></div>
        <div>TEST SPLIT: <span style="color:#e8e4dc;">20%</span></div>
        <div>MLFLOW: <span style="color:#22c55e;">ACTIVE</span></div>
    </div>
    """, unsafe_allow_html=True)
    if st.session_state.accuracy:
        st.markdown(f"""
        <div style="margin-top:1rem; background:#0a0c0f; border:1px solid #1e2530; border-left:3px solid #f5a623; padding:0.8rem 1rem; border-radius:3px;">
            <div style="font-family:'JetBrains Mono',monospace; font-size:0.65rem; color:#6b7280;">BEST MODEL ACCURACY</div>
            <div style="font-family:'JetBrains Mono',monospace; font-size:1.5rem; font-weight:700; color:#f5a623;">{st.session_state.accuracy:.1%}</div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown('<div class="section-label">MLflow</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-family:'JetBrains Mono',monospace; font-size:0.72rem; color:#6b7280; line-height:1.8;">
        Run in terminal to view UI:<br>
        <span style="color:#f5a623;">mlflow ui --port 5000</span><br>
        Then open localhost:5000
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def find_col(col_map, name):
    name = name.lower()
    for c in col_map:
        if name in c:
            return col_map[c]
    return None

def generate_short_report(row, col_map):
    lines = []
    for key in ['temperature', 'pressure', 'vibration', 'speed', 'torque']:
        col = find_col(col_map, key)
        if col and col in row.index:
            lines.append(f"{key.upper()[:4]}: {row[col]:.2f}")
    return " | ".join(lines) if lines else "Sensor readings recorded."

def generate_long_report(row, col_map):
    report = "MACHINE HEALTH REPORT\n\nSENSOR READINGS:\n"
    for orig_name, col in col_map.items():
        if col in row.index:
            val = row[col]
            report += f"  {col}: {val:.3f}\n" if isinstance(val, (int, float, np.number)) else f"  {col}: {val}\n"
    report += "\nRECOMMENDATION:\n  Schedule inspection based on sensor readings above.\n"
    return report

def plot_confusion_matrix(y_test, y_pred, title="Confusion Matrix"):
    fig, ax = plt.subplots(figsize=(4, 3))
    fig.patch.set_facecolor('#0f1218')
    ax.set_facecolor('#0f1218')
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrBr', ax=ax,
                linewidths=0.5, linecolor='#1e2530',
                annot_kws={"size": 12, "color": "white"})
    ax.set_title(title, color='#f5a623', fontsize=10, pad=10)
    ax.set_xlabel("Predicted", color='#9ca3af', fontsize=9)
    ax.set_ylabel("Actual", color='#9ca3af', fontsize=9)
    ax.tick_params(colors='#9ca3af')
    plt.tight_layout()
    return fig

def plot_roc(y_test, y_prob, label):
    fig, ax = plt.subplots(figsize=(5, 3.5))
    fig.patch.set_facecolor('#0f1218')
    ax.set_facecolor('#0f1218')
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    ax.plot(fpr, tpr, color='#f5a623', lw=2, label=f'AUC = {auc:.3f}')
    ax.plot([0,1],[0,1], color='#374151', linestyle='--', lw=1)
    ax.set_xlabel("False Positive Rate", color='#9ca3af', fontsize=9)
    ax.set_ylabel("True Positive Rate", color='#9ca3af', fontsize=9)
    ax.set_title(f"ROC Curve — {label}", color='#f5a623', fontsize=10)
    ax.tick_params(colors='#9ca3af')
    ax.legend(facecolor='#0f1218', edgecolor='#1e2530', labelcolor='#e8e4dc', fontsize=9)
    ax.spines['bottom'].set_color('#1e2530')
    ax.spines['left'].set_color('#1e2530')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    return fig

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "  DATASET & EDA  ",
    "  MODEL TRAINING  ",
    "  LIVE PREDICTION  ",
    "  EXPLAINABILITY  ",
    "  REPORTS  "
])

# ═══════════════════════════════════
# TAB 1: Dataset & EDA
# ═══════════════════════════════════
with tab1:
    if not uploaded_file:
        st.markdown("""
        <div style="text-align:center; padding:4rem; border:1px dashed #1e2530; border-radius:4px; margin-top:1rem;">
            <div style="font-size:3rem;">⚙</div>
            <div style="font-family:'Syne',sans-serif; font-size:1.4rem; font-weight:700; color:#e8e4dc; margin-bottom:0.5rem;">Upload Your Dataset</div>
            <div style="font-family:'JetBrains Mono',monospace; font-size:0.75rem; color:#6b7280;">
                Upload a CSV with machine sensor readings via the sidebar.
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        data = pd.read_csv(uploaded_file)
        st.session_state.raw_data = data

        # Detect target
        possible_targets = ["Machine failure", "machine failure", "Failure", "failure", "Target", "target", "Anomaly"]
        target_col = None
        lower_map = {col.strip().lower(): col for col in data.columns}
        for name in possible_targets:
            if name.strip().lower() in lower_map:
                target_col = lower_map[name.strip().lower()]
                break
        if not target_col:
            st.error(f"Target column not found. Expected: Machine failure, Failure, Target, Anomaly. Found: {list(data.columns)}")
            st.stop()

        # ── Overview metrics ──
        st.markdown('<div class="section-label">Dataset Overview</div>', unsafe_allow_html=True)
        c1, c2, c3, c4, c5 = st.columns(5)
        cols_data = [
            (c1, len(data), "Total Records", ""),
            (c2, len(data.columns), "Features", "info"),
            (c3, data.isnull().sum().sum(), "Missing Values", "danger" if data.isnull().sum().sum() > 0 else "ok"),
            (c4, int(data[target_col].sum()), "Failure Cases", "danger"),
            (c5, len(data.select_dtypes(include=[np.number]).columns), "Numeric Cols", ""),
        ]
        for col, val, label, cls in cols_data:
            with col:
                st.markdown(f'<div class="metric-card {cls}"><div class="metric-value">{val:,}</div><div class="metric-label">{label}</div></div>', unsafe_allow_html=True)

        st.markdown('<div class="section-label" style="margin-top:1.5rem;">Raw Data Preview</div>', unsafe_allow_html=True)
        st.dataframe(data.head(10), use_container_width=True)

        st.markdown('<div class="section-label" style="margin-top:1.5rem;">Descriptive Statistics</div>', unsafe_allow_html=True)
        st.dataframe(data.describe().round(3), use_container_width=True)

        # ── EDA Plots ──
        st.markdown('<div class="section-label" style="margin-top:1.5rem;">Exploratory Data Analysis</div>', unsafe_allow_html=True)

        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        feature_only = [c for c in numeric_cols if c != target_col]

        # Plot 1: Class distribution
        c1, c2 = st.columns(2)
        with c1:
            fig, ax = plt.subplots(figsize=(5, 3.5))
            fig.patch.set_facecolor('#0f1218')
            ax.set_facecolor('#0f1218')
            counts = data[target_col].value_counts()
            colors = ['#22c55e', '#ef4444']
            ax.bar(['Healthy (0)', 'Failure (1)'], counts.values, color=colors, width=0.5, edgecolor='#1e2530')
            ax.set_title('Class Distribution', color='#f5a623', fontsize=10)
            ax.set_ylabel('Count', color='#9ca3af', fontsize=9)
            ax.tick_params(colors='#9ca3af')
            for spine in ax.spines.values():
                spine.set_color('#1e2530')
            for i, v in enumerate(counts.values):
                ax.text(i, v + 10, str(v), color='#e8e4dc', ha='center', fontsize=9)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        # Plot 2: Missing value heatmap
        with c2:
            fig, ax = plt.subplots(figsize=(5, 3.5))
            fig.patch.set_facecolor('#0f1218')
            ax.set_facecolor('#0f1218')
            missing = data.isnull().sum()
            if missing.sum() == 0:
                ax.text(0.5, 0.5, 'No missing values\nDataset is clean ✓',
                        ha='center', va='center', color='#22c55e',
                        fontsize=12, transform=ax.transAxes)
                ax.set_title('Missing Values', color='#f5a623', fontsize=10)
            else:
                missing[missing > 0].plot(kind='bar', ax=ax, color='#ef4444')
                ax.set_title('Missing Values per Column', color='#f5a623', fontsize=10)
            ax.tick_params(colors='#9ca3af')
            for spine in ax.spines.values():
                spine.set_color('#1e2530')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        # Plot 3: Correlation heatmap
        st.markdown('<div class="section-label" style="margin-top:1rem;">Correlation Heatmap</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor('#0f1218')
        ax.set_facecolor('#0f1218')
        corr = data[numeric_cols].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='YlOrBr',
                    ax=ax, linewidths=0.5, linecolor='#0a0c0f',
                    annot_kws={"size": 8}, vmin=-1, vmax=1)
        ax.set_title('Feature Correlation Matrix', color='#f5a623', fontsize=11, pad=10)
        ax.tick_params(colors='#9ca3af', labelsize=8)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Plot 4: Feature distributions
        st.markdown('<div class="section-label" style="margin-top:1rem;">Feature Distributions by Class</div>', unsafe_allow_html=True)
        cols_to_plot = feature_only[:6]
        fig, axes = plt.subplots(2, 3, figsize=(12, 6))
        fig.patch.set_facecolor('#0f1218')
        axes = axes.flatten()
        for i, col in enumerate(cols_to_plot):
            axes[i].set_facecolor('#0f1218')
            for cls_val, color, label in [(0, '#22c55e', 'Healthy'), (1, '#ef4444', 'Failure')]:
                subset = data[data[target_col] == cls_val][col].dropna()
                axes[i].hist(subset, bins=30, alpha=0.6, color=color, label=label, edgecolor='none')
            axes[i].set_title(col, color='#f5a623', fontsize=9)
            axes[i].tick_params(colors='#9ca3af', labelsize=7)
            for spine in axes[i].spines.values():
                spine.set_color('#1e2530')
            axes[i].legend(fontsize=7, facecolor='#0f1218', labelcolor='#9ca3af', edgecolor='#1e2530')
        for j in range(len(cols_to_plot), len(axes)):
            axes[j].set_visible(False)
        plt.suptitle('Feature Distributions (Healthy vs Failure)', color='#e8e4dc', fontsize=11, y=1.02)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Store target for other tabs
        st.session_state.target_col = target_col

# ═══════════════════════════════════
# TAB 2: Model Training (with MLflow)
# ═══════════════════════════════════
with tab2:
    if st.session_state.raw_data is None:
        st.warning("Upload a dataset in the DATASET & EDA tab first.")
    else:
        data = st.session_state.raw_data
        target_col = getattr(st.session_state, 'target_col', None)
        if not target_col:
            st.error("Target column not detected. Go back to EDA tab.")
            st.stop()

        st.markdown('<div class="section-label">Model Configuration</div>', unsafe_allow_html=True)

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            n_estimators = st.slider("RF Estimators", 50, 300, 100, 50)
        with col_b:
            xgb_lr = st.slider("XGBoost Learning Rate", 0.01, 0.3, 0.1, 0.01)
        with col_c:
            test_size = st.slider("Test Split %", 10, 30, 20, 5)

        train_btn = st.button("🚀  TRAIN ALL MODELS + LOG TO MLFLOW", use_container_width=True)

        if train_btn:
            # Preprocess
            proc_data = data.copy()
            label_encoders = {}
            non_numeric = proc_data.select_dtypes(exclude=['int64', 'float64']).columns
            for col in non_numeric:
                if col.strip().lower() in ['udi', 'id', 'serialnumber', 'product id']:
                    proc_data = proc_data.drop(col, axis=1)
                elif col != target_col:
                    le = LabelEncoder()
                    proc_data[col] = le.fit_transform(proc_data[col].astype(str))
                    label_encoders[col] = le

            X = proc_data.drop(target_col, axis=1)
            y = proc_data[target_col]

            col_map = {c.strip().lower(): c for c in X.columns}
            st.session_state.col_map = col_map
            st.session_state.feature_cols = list(X.columns)
            st.session_state.X_sample = X
            st.session_state.label_encoders = label_encoders

            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size/100, random_state=42, stratify=y
                )
            except:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size/100, random_state=42
                )

            st.session_state.X_train = X_train
            st.session_state.X_test = X_test
            st.session_state.y_train = y_train
            st.session_state.y_test = y_test

            models_to_train = {
                "Random Forest": RandomForestClassifier(n_estimators=n_estimators, random_state=42, n_jobs=-1),
                "XGBoost": XGBClassifier(learning_rate=xgb_lr, n_estimators=n_estimators,
                                         use_label_encoder=False, eval_metric='logloss',
                                         random_state=42, verbosity=0),
            }

            all_results = {}
            best_f1 = -1
            best_model = None
            best_model_name = None

            progress_bar = st.progress(0)

            for i, (name, clf) in enumerate(models_to_train.items()):
                with st.spinner(f"Training {name}..."):
                    with mlflow.start_run(run_name=name):
                        # Log params
                        mlflow.log_param("model", name)
                        mlflow.log_param("n_estimators", n_estimators)
                        mlflow.log_param("test_size", test_size)
                        if name == "XGBoost":
                            mlflow.log_param("learning_rate", xgb_lr)

                        clf.fit(X_train, y_train)
                        y_pred = clf.predict(X_test)
                        y_prob = clf.predict_proba(X_test)[:, 1]

                        acc = accuracy_score(y_test, y_pred)
                        f1 = f1_score(y_test, y_pred, average='weighted')
                        try:
                            auc = roc_auc_score(y_test, y_prob)
                        except:
                            auc = 0.0
                        cv_scores = cross_val_score(clf, X, y, cv=5, scoring='f1_weighted')

                        # Log metrics to MLflow
                        mlflow.log_metric("accuracy", acc)
                        mlflow.log_metric("f1_weighted", f1)
                        mlflow.log_metric("roc_auc", auc)
                        mlflow.log_metric("cv_f1_mean", cv_scores.mean())
                        mlflow.log_metric("cv_f1_std", cv_scores.std())
                        mlflow.sklearn.log_model(clf, name.replace(" ", "_"))

                        all_results[name] = {
                            "model": clf, "y_pred": y_pred, "y_prob": y_prob,
                            "accuracy": acc, "f1": f1, "auc": auc,
                            "cv_mean": cv_scores.mean(), "cv_std": cv_scores.std(),
                            "report": classification_report(y_test, y_pred)
                        }

                        if f1 > best_f1:
                            best_f1 = f1
                            best_model = clf
                            best_model_name = name

                progress_bar.progress((i + 1) / len(models_to_train))

            st.session_state.all_results = all_results
            st.session_state.model = best_model
            st.session_state.best_model_name = best_model_name
            st.session_state.accuracy = all_results[best_model_name]["accuracy"]

            # SHAP values for best model
            with st.spinner("Computing SHAP values..."):
                try:
                    explainer = shap.TreeExplainer(best_model)
                    shap_vals = explainer.shap_values(X_test)
                    st.session_state.shap_values = shap_vals
                    st.session_state.explainer = explainer
                except Exception as e:
                    st.session_state.shap_values = None

            # Generate output df
            output_df = data.copy()
            X_full = proc_data.drop(target_col, axis=1)
            output_df['Short_Report'] = X_full.apply(lambda r: generate_short_report(r, col_map), axis=1)
            output_df['Long_Report'] = X_full.apply(lambda r: generate_long_report(r, col_map), axis=1)
            output_df['Predicted_Failure'] = best_model.predict(X_full)
            st.session_state.output_df = output_df

            st.success(f"✅ All models trained & logged to MLflow! Best: {best_model_name} (F1: {best_f1:.3f})")

        # ── Show results ──
        if st.session_state.all_results:
            results = st.session_state.all_results
            st.markdown('<div class="section-label" style="margin-top:1.5rem;">Model Comparison</div>', unsafe_allow_html=True)

            # Comparison table
            comp_data = {
                "Model": list(results.keys()),
                "Accuracy": [f"{r['accuracy']:.3f}" for r in results.values()],
                "F1 (weighted)": [f"{r['f1']:.3f}" for r in results.values()],
                "ROC-AUC": [f"{r['auc']:.3f}" for r in results.values()],
                "CV F1 Mean": [f"{r['cv_mean']:.3f}" for r in results.values()],
                "CV F1 Std": [f"±{r['cv_std']:.3f}" for r in results.values()],
            }
            comp_df = pd.DataFrame(comp_data)
            st.dataframe(comp_df, use_container_width=True, hide_index=True)

            # Per-model details
            for name, res in results.items():
                is_best = name == st.session_state.best_model_name
                badge = " ★ BEST" if is_best else ""
                st.markdown(f'<div class="section-label" style="margin-top:1.5rem;">{name}{badge}</div>', unsafe_allow_html=True)

                c1, c2 = st.columns([1, 1])
                with c1:
                    fig = plot_confusion_matrix(
                        st.session_state.y_test, res["y_pred"], f"Confusion Matrix — {name}"
                    )
                    st.pyplot(fig)
                    plt.close()
                with c2:
                    try:
                        fig2 = plot_roc(st.session_state.y_test, res["y_prob"], name)
                        st.pyplot(fig2)
                        plt.close()
                    except:
                        st.info("ROC curve requires binary classification with probability scores.")

                st.markdown(f'<div class="report-card"><h4>Classification Report — {name}</h4><div class="report-body">{res["report"]}</div></div>', unsafe_allow_html=True)

            # Feature importance for best model
            st.markdown('<div class="section-label" style="margin-top:1.5rem;">Feature Importance (Best Model)</div>', unsafe_allow_html=True)
            best_res = results[st.session_state.best_model_name]
            best_clf = best_res["model"]
            feat_imp = pd.DataFrame({
                'Feature': st.session_state.feature_cols,
                'Importance': best_clf.feature_importances_
            }).sort_values('Importance', ascending=False).head(10)

            for _, row_fi in feat_imp.iterrows():
                pct = row_fi['Importance'] * 100
                st.markdown(f"""
                <div style="margin-bottom:0.5rem;">
                    <div style="display:flex; justify-content:space-between; font-family:'JetBrains Mono',monospace; font-size:0.72rem; color:#9ca3af; margin-bottom:3px;">
                        <span>{row_fi['Feature']}</span><span style="color:#f5a623;">{pct:.1f}%</span>
                    </div>
                    <div style="background:#1e2530; border-radius:2px; height:4px;">
                        <div style="background:#f5a623; width:{pct}%; height:4px; border-radius:2px;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

# ═══════════════════════════════════
# TAB 3: Live Prediction
# ═══════════════════════════════════
with tab3:
    if st.session_state.model is None:
        st.warning("Train models first in the MODEL TRAINING tab.")
    else:
        st.markdown('<div class="section-label">Manual Sensor Input</div>', unsafe_allow_html=True)
        feature_cols = st.session_state.feature_cols
        X_sample = st.session_state.X_sample
        input_values = {}

        cols_per_row = 3
        col_groups = [feature_cols[i:i+cols_per_row] for i in range(0, len(feature_cols), cols_per_row)]
        for group in col_groups:
            form_cols = st.columns(len(group))
            for idx, feat in enumerate(group):
                with form_cols[idx]:
                    col_data = X_sample[feat]
                    min_val, max_val, mean_val = float(col_data.min()), float(col_data.max()), float(col_data.mean())
                    is_int = (col_data.dtype in ['int64', 'int32']) or (col_data == col_data.round()).all()
                    st.markdown(f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:0.65rem;color:#6b7280;margin-bottom:4px;">{feat} · [{min_val:.1f}–{max_val:.1f}]</div>', unsafe_allow_html=True)
                    step = 1.0 if is_int else max(0.001, (max_val - min_val) / 1000)
                    val = st.number_input(feat, min_value=float(min_val), max_value=float(max_val),
                                          value=float(round(mean_val) if is_int else round(mean_val, 3)),
                                          step=float(round(step, 4)), label_visibility="collapsed", key=f"inp_{feat}")
                    input_values[feat] = val

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("⚡  RUN PREDICTION", use_container_width=True):
            input_df = pd.DataFrame([input_values])
            prediction = st.session_state.model.predict(input_df)[0]
            proba = st.session_state.model.predict_proba(input_df)[0]
            fail_prob = proba[1] if len(proba) > 1 else proba[0]
            ok_prob = 1 - fail_prob
            is_fail = bool(prediction == 1)

            if is_fail:
                st.markdown(f'<div class="pred-box fail"><div class="pred-value fail">⚠ FAILURE DETECTED</div><div style="font-family:JetBrains Mono,monospace;font-size:0.78rem;color:#6b7280;margin-top:0.4rem;">Failure probability: {fail_prob:.1%}</div></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="pred-box ok"><div class="pred-value ok">✓ MACHINE HEALTHY</div><div style="font-family:JetBrains Mono,monospace;font-size:0.78rem;color:#6b7280;margin-top:0.4rem;">Normal operation: {ok_prob:.1%}</div></div>', unsafe_allow_html=True)

            st.markdown('<div class="section-label" style="margin-top:1rem;">Probability Breakdown</div>', unsafe_allow_html=True)
            st.markdown(f"""
            <div style="margin-bottom:0.75rem;">
                <div style="display:flex;justify-content:space-between;font-family:'JetBrains Mono',monospace;font-size:0.72rem;color:#9ca3af;margin-bottom:4px;">
                    <span>NORMAL OPERATION</span><span style="color:#22c55e;">{ok_prob:.1%}</span>
                </div>
                <div style="background:#1e2530;border-radius:2px;height:6px;">
                    <div style="background:#22c55e;width:{ok_prob*100:.1f}%;height:6px;border-radius:2px;"></div>
                </div>
            </div>
            <div>
                <div style="display:flex;justify-content:space-between;font-family:'JetBrains Mono',monospace;font-size:0.72rem;color:#9ca3af;margin-bottom:4px;">
                    <span>FAILURE RISK</span><span style="color:#ef4444;">{fail_prob:.1%}</span>
                </div>
                <div style="background:#1e2530;border-radius:2px;height:6px;">
                    <div style="background:#ef4444;width:{fail_prob*100:.1f}%;height:6px;border-radius:2px;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

# ═══════════════════════════════════
# TAB 4: Explainability (SHAP)
# ═══════════════════════════════════
with tab4:
    if st.session_state.shap_values is None:
        st.warning("Train models first. SHAP values are computed during training.")
    else:
        st.markdown('<div class="section-label">SHAP — Model Explainability</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="font-family:'JetBrains Mono',monospace; font-size:0.75rem; color:#6b7280; margin-bottom:1.5rem; line-height:1.8;">
            SHAP (SHapley Additive exPlanations) shows how each feature contributes to each prediction.
            This makes the model interpretable — not a black box.
        </div>
        """, unsafe_allow_html=True)

        shap_values = st.session_state.shap_values
        X_test = st.session_state.X_test
        feature_cols = st.session_state.feature_cols

        # Handle multi-output SHAP (Random Forest returns list)
        if isinstance(shap_values, list):
            sv = shap_values[1]  # class 1 = failure
        else:
            sv = shap_values

        # Plot 1: Summary plot (beeswarm)
        st.markdown('<div class="section-label">Feature Impact Summary (SHAP Beeswarm)</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor('#0f1218')
        shap.summary_plot(sv, X_test, feature_names=feature_cols,
                          show=False, plot_type="dot",
                          color_bar_label="Feature value")
        plt.gca().set_facecolor('#0f1218')
        fig = plt.gcf()
        fig.patch.set_facecolor('#0f1218')
        plt.title("SHAP Summary — Feature Impact on Failure Prediction",
                  color='#f5a623', fontsize=11, pad=10)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Plot 2: Bar plot (mean |SHAP|)
        st.markdown('<div class="section-label" style="margin-top:1rem;">Mean |SHAP| — Global Feature Importance</div>', unsafe_allow_html=True)
        mean_shap = np.abs(sv).mean(axis=0)
        shap_df = pd.DataFrame({'Feature': feature_cols, 'Mean |SHAP|': mean_shap})
        shap_df = shap_df.sort_values('Mean |SHAP|', ascending=True).tail(10)

        fig, ax = plt.subplots(figsize=(8, 4))
        fig.patch.set_facecolor('#0f1218')
        ax.set_facecolor('#0f1218')
        bars = ax.barh(shap_df['Feature'], shap_df['Mean |SHAP|'], color='#f5a623', edgecolor='none')
        ax.set_title('Top 10 Features — Global SHAP Importance', color='#f5a623', fontsize=10)
        ax.tick_params(colors='#9ca3af', labelsize=9)
        for spine in ax.spines.values():
            spine.set_color('#1e2530')
        ax.set_xlabel('Mean |SHAP value|', color='#9ca3af', fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Single prediction explanation
        st.markdown('<div class="section-label" style="margin-top:1rem;">Individual Prediction Explanation</div>', unsafe_allow_html=True)
        sample_idx = st.slider("Select test sample index", 0, len(X_test)-1, 0)
        sample_shap = sv[sample_idx]
        sample_feats = X_test.iloc[sample_idx]

        shap_contrib = pd.DataFrame({
            'Feature': feature_cols,
            'Value': sample_feats.values,
            'SHAP': sample_shap
        }).sort_values('SHAP', key=abs, ascending=False).head(10)

        fig, ax = plt.subplots(figsize=(9, 4))
        fig.patch.set_facecolor('#0f1218')
        ax.set_facecolor('#0f1218')
        colors = ['#ef4444' if v > 0 else '#22c55e' for v in shap_contrib['SHAP']]
        ax.barh(shap_contrib['Feature'], shap_contrib['SHAP'], color=colors, edgecolor='none')
        ax.axvline(0, color='#374151', lw=1)
        ax.set_title(f'SHAP Explanation — Sample #{sample_idx}', color='#f5a623', fontsize=10)
        ax.tick_params(colors='#9ca3af', labelsize=9)
        for spine in ax.spines.values():
            spine.set_color('#1e2530')
        ax.set_xlabel('SHAP value (red = pushes toward failure, green = toward healthy)', color='#9ca3af', fontsize=8)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.dataframe(shap_contrib.round(4), use_container_width=True, hide_index=True)

# ═══════════════════════════════════
# TAB 5: Reports
# ═══════════════════════════════════
with tab5:
    if st.session_state.output_df is None:
        st.warning("Train models first to generate reports.")
    else:
        output_df = st.session_state.output_df
        st.markdown('<div class="section-label">All Machine Reports</div>', unsafe_allow_html=True)

        total = len(output_df)
        predicted_fail = int(output_df['Predicted_Failure'].sum()) if 'Predicted_Failure' in output_df.columns else 0
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{total:,}</div><div class="metric-label">Total Machines</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="metric-card danger"><div class="metric-value">{predicted_fail:,}</div><div class="metric-label">Predicted Failures</div></div>', unsafe_allow_html=True)
        with c3:
            pct = predicted_fail / total * 100 if total > 0 else 0
            st.markdown(f'<div class="metric-card"><div class="metric-value">{pct:.1f}%</div><div class="metric-label">Failure Rate</div></div>', unsafe_allow_html=True)

        if st.button("🎲  Random Report"):
            idx = random.randint(0, len(output_df) - 1)
            row = output_df.iloc[idx]
            pred_val = row.get('Predicted_Failure', 0)
            status = "FAILURE" if pred_val == 1 else "HEALTHY"
            color = "#ef4444" if pred_val == 1 else "#22c55e"
            st.markdown(f'<div style="font-family:JetBrains Mono,monospace;font-size:0.72rem;color:#6b7280;margin-bottom:1rem;">RECORD #{idx} | <span style="color:{color};">● {status}</span></div>', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f'<div class="report-card"><h4>Short Report</h4><div class="report-body">{row["Short_Report"]}</div></div>', unsafe_allow_html=True)
            with c2:
                st.markdown(f'<div class="report-card"><h4>Detailed Report</h4><div class="report-body">{row["Long_Report"]}</div></div>', unsafe_allow_html=True)

        csv = output_df.to_csv(index=False).encode('utf-8')
        st.download_button("⬇  Download All Reports as CSV", data=csv,
                           file_name="smat_ai_v5_reports.csv", mime="text/csv",
                           use_container_width=True)

        preview_cols = [c for c in ['Short_Report', 'Predicted_Failure'] if c in output_df.columns]
        st.dataframe(output_df[preview_cols].head(20), use_container_width=True)