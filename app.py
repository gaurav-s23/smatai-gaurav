"""
SMAT.AI v6.0 — Smart Machine Analysis & Telemetry
AI-powered Predictive Maintenance System
Author: Gaurav Shukla
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import mlflow
import mlflow.sklearn
import random
import io
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, f1_score, roc_auc_score
)
from xgboost import XGBClassifier

# ── Local utils ──
from utils import (
    detect_target_column, preprocess_data, build_output_df,
    generate_short_report, generate_long_report,
    plot_confusion_matrix, plot_roc, plot_precision_recall,
    plot_class_distribution, plot_correlation_heatmap,
    plot_feature_distributions, plot_feature_importance,
    plot_shap_summary, plot_shap_bar, plot_shap_individual,
    plot_failure_trend, plot_model_comparison_radar,
    plot_sensor_boxplots, plot_failure_heatmap_grid,
    compute_dashboard_stats,
    DARK_BG, CARD_BG, ACCENT, BORDER, TEXT_DIM, TEXT_MAIN, GREEN, RED, BLUE, PURPLE
)

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
# MLflow — SQLite backend (FIXED)
# ─────────────────────────────────────────────
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("SMAT-AI-v6")

# ─────────────────────────────────────────────
# CSS — Industrial dark theme + Power BI cards
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;700&family=Syne:wght@400;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Syne', sans-serif; }
.stApp { background: #0a0c0f; color: #e8e4dc; }
.main .block-container { padding: 1.5rem 2.5rem; max-width: 1500px; }
[data-testid="stSidebar"] { background: #0f1218 !important; border-right: 1px solid #1e2530; }

/* ── Header ── */
.smat-header {
    background: linear-gradient(135deg, #0f1218 0%, #141920 60%, #0a0c10 100%);
    border: 1px solid #1e2530; border-left: 4px solid #f5a623;
    border-radius: 4px; padding: 1.2rem 2rem; margin-bottom: 1.5rem;
    display: flex; align-items: center; justify-content: space-between;
}
.smat-header h1 { font-family: 'Syne', sans-serif; font-weight: 800; font-size: 2rem; color: #f5a623; margin: 0; }
.smat-header p { font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; color: #6b7280; margin: 0.3rem 0 0 0; letter-spacing: 0.08em; }
.version-badge {
    display: inline-block; background: rgba(245,166,35,0.12);
    border: 1px solid rgba(245,166,35,0.3); color: #f5a623;
    font-family: 'JetBrains Mono', monospace; font-size: 0.62rem;
    padding: 2px 10px; border-radius: 2px; margin-left: 1rem; vertical-align: middle;
}

/* ── Section label ── */
.section-label {
    font-family: 'JetBrains Mono', monospace; font-size: 0.68rem;
    letter-spacing: 0.15em; color: #f5a623; text-transform: uppercase;
    margin-bottom: 0.75rem; border-bottom: 1px solid #1e2530; padding-bottom: 0.4rem;
    margin-top: 1.2rem;
}

/* ── KPI cards (Power BI style) ── */
.kpi-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; margin-bottom: 1.2rem; }
.kpi-card {
    background: #0f1218; border: 1px solid #1e2530; border-radius: 6px;
    padding: 1rem 1.2rem; position: relative; overflow: hidden; cursor: default;
    transition: border-color 0.2s;
}
.kpi-card:hover { border-color: #f5a623; }
.kpi-card::after { content:''; position:absolute; top:0; left:0; right:0; height:3px; background:#f5a623; border-radius:6px 6px 0 0; }
.kpi-card.danger::after { background: #ef4444; }
.kpi-card.ok::after { background: #22c55e; }
.kpi-card.info::after { background: #3b82f6; }
.kpi-card.purple::after { background: #a855f7; }
.kpi-value { font-family:'JetBrains Mono',monospace; font-size:1.7rem; font-weight:700; color:#e8e4dc; line-height:1; }
.kpi-label { font-size:0.68rem; color:#6b7280; margin-top:0.35rem; letter-spacing:0.06em; text-transform:uppercase; font-family:'JetBrains Mono',monospace; }
.kpi-sub { font-size:0.65rem; color:#374151; margin-top:0.2rem; font-family:'JetBrains Mono',monospace; }

/* ── Chart card ── */
.chart-card { background: #0f1218; border: 1px solid #1e2530; border-radius: 6px; padding: 1.2rem; margin-bottom: 1rem; }
.chart-card h4 { font-family:'JetBrains Mono',monospace; font-size:0.68rem; letter-spacing:0.12em; color:#f5a623; text-transform:uppercase; margin:0 0 0.8rem 0; }

/* ── Report card ── */
.report-card { background: #0f1218; border: 1px solid #1e2530; border-radius: 4px; padding: 1.2rem; margin-bottom: 0.8rem; }
.report-card h4 { font-family:'JetBrains Mono',monospace; font-size:0.68rem; letter-spacing:0.1em; color:#f5a623; text-transform:uppercase; margin-bottom:0.6rem; }
.report-body { font-family:'JetBrains Mono',monospace; font-size:0.78rem; color:#9ca3af; line-height:1.7; white-space:pre-wrap; }

/* ── Prediction box ── */
.pred-box { border-radius:6px; padding:1.6rem 2rem; text-align:center; margin:0.8rem 0; }
.pred-box.fail { background:rgba(239,68,68,0.08); border:2px solid rgba(239,68,68,0.4); }
.pred-box.ok   { background:rgba(34,197,94,0.08);  border:2px solid rgba(34,197,94,0.35); }
.pred-value { font-family:'Syne',sans-serif; font-size:2.2rem; font-weight:800; line-height:1; }
.pred-value.fail { color:#ef4444; }
.pred-value.ok   { color:#22c55e; }

/* ── Buttons ── */
.stButton>button {
    background:#f5a623 !important; color:#0a0c0f !important; border:none !important;
    border-radius:3px !important; font-family:'JetBrains Mono',monospace !important;
    font-weight:700 !important; font-size:0.78rem !important; letter-spacing:0.05em !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] { background:#0f1218; border-bottom:1px solid #1e2530; gap:0; }
.stTabs [data-baseweb="tab"] {
    background:transparent !important; color:#6b7280 !important;
    font-family:'JetBrains Mono',monospace !important; font-size:0.7rem !important;
    letter-spacing:0.08em !important; text-transform:uppercase !important;
    border-radius:0 !important; padding:0.65rem 1.2rem !important;
    border-bottom:2px solid transparent !important;
}
.stTabs [aria-selected="true"] { color:#f5a623 !important; border-bottom:2px solid #f5a623 !important; }

/* ── Feature bar ── */
.feat-bar-wrap { margin-bottom:0.45rem; }
.feat-bar-header { display:flex; justify-content:space-between; font-family:'JetBrains Mono',monospace; font-size:0.68rem; color:#9ca3af; margin-bottom:3px; }
.feat-bar-bg { background:#1e2530; border-radius:2px; height:5px; }
.feat-bar-fill { background:#f5a623; height:5px; border-radius:2px; }

/* ── Alert strip ── */
.alert-strip { border-radius:4px; padding:0.7rem 1rem; font-family:'JetBrains Mono',monospace; font-size:0.75rem; margin-bottom:0.8rem; }
.alert-strip.warn { background:rgba(245,166,35,0.1); border:1px solid rgba(245,166,35,0.3); color:#f5a623; }
.alert-strip.info { background:rgba(59,130,246,0.1); border:1px solid rgba(59,130,246,0.3); color:#3b82f6; }

/* ── Hide streamlit chrome ── */
#MainMenu { visibility:hidden; } footer { visibility:hidden; } header { visibility:hidden; }

/* ── Scrollable table ── */
.dataframe { font-family:'JetBrains Mono',monospace; font-size:0.72rem; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Session State
# ─────────────────────────────────────────────
_defaults = {
    "model": None, "best_model_name": None, "feature_cols": None,
    "X_sample": None, "X_train": None, "X_test": None,
    "y_train": None, "y_test": None, "accuracy": None,
    "output_df": None, "col_map": {}, "label_encoders": {},
    "all_results": {}, "raw_data": None, "target_col": None,
    "shap_values": None, "explainer": None, "proc_X_full": None,
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────
st.markdown("""
<div class="smat-header">
  <div>
    <h1>⚙ SMAT.AI <span class="version-badge">v6.0</span></h1>
    <p>SMART MACHINE ANALYSIS & TELEMETRY — AI-POWERED PREDICTIVE MAINTENANCE SYSTEM</p>
  </div>
  <div style="font-family:'JetBrains Mono',monospace; font-size:0.65rem; color:#374151; text-align:right;">
    MODELS: RF · XGB · GBM<br>TRACKING: MLFLOW<br>EXPLAINABILITY: SHAP
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="section-label" style="margin-top:0;">📂 Dataset Upload</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload CSV file", type="csv", label_visibility="collapsed")

    st.markdown("---")
    st.markdown('<div class="section-label">⚙ System Status</div>', unsafe_allow_html=True)
    model_status = "TRAINED" if st.session_state.model else "NO MODEL"
    color = "#22c55e" if st.session_state.model else "#ef4444"
    best_name = st.session_state.best_model_name or "—"

    st.markdown(f"""
    <div style="font-family:'JetBrains Mono',monospace; font-size:0.7rem; color:#6b7280; line-height:2.2;">
        <div>STATUS : <span style="color:{color}; font-weight:700;">{model_status}</span></div>
        <div>MODEL  : <span style="color:#e8e4dc;">{best_name}</span></div>
        <div>SPLIT  : <span style="color:#e8e4dc;">80 / 20</span></div>
        <div>MLFLOW : <span style="color:#22c55e;">ACTIVE</span></div>
        <div>SHAP   : <span style="color:{'#22c55e' if st.session_state.shap_values is not None else '#374151'};">{'READY' if st.session_state.shap_values is not None else 'PENDING'}</span></div>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.accuracy:
        st.markdown(f"""
        <div style="margin-top:0.8rem; background:#0a0c0f; border:1px solid #1e2530; border-left:3px solid #f5a623;
                    padding:0.7rem 1rem; border-radius:3px;">
            <div style="font-family:'JetBrains Mono',monospace; font-size:0.62rem; color:#6b7280;">BEST ACCURACY</div>
            <div style="font-family:'JetBrains Mono',monospace; font-size:1.4rem; font-weight:700; color:#f5a623;">
                {st.session_state.accuracy:.1%}
            </div>
        </div>
        """, unsafe_allow_html=True)

    if st.session_state.raw_data is not None:
        data_info = st.session_state.raw_data
        st.markdown("---")
        st.markdown('<div class="section-label">📊 Dataset Info</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div style="font-family:'JetBrains Mono',monospace; font-size:0.68rem; color:#6b7280; line-height:2.1;">
            <div>ROWS   : <span style="color:#e8e4dc;">{len(data_info):,}</span></div>
            <div>COLS   : <span style="color:#e8e4dc;">{len(data_info.columns)}</span></div>
            <div>NULLS  : <span style="color:{'#ef4444' if data_info.isnull().sum().sum() > 0 else '#22c55e'};">{data_info.isnull().sum().sum()}</span></div>
            <div>TARGET : <span style="color:#e8e4dc;">{st.session_state.target_col or '—'}</span></div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-label">🔬 MLflow UI</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-family:'JetBrains Mono',monospace; font-size:0.68rem; color:#6b7280; line-height:1.9;">
        Run in terminal:<br>
        <span style="color:#f5a623;">mlflow server --backend-store-uri sqlite:///mlflow.db --port 5000</span><br>
        Then open: <span style="color:#3b82f6;">localhost:5000</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-label">📡 Live API Prediction</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-family:'JetBrains Mono',monospace; font-size:0.65rem; color:#6b7280; line-height:1.8;">
        Export model + run FastAPI:<br>
        <span style="color:#f5a623;">import pickle, uvicorn</span><br>
        POST /predict → JSON payload<br>
        See README for full guide.
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "  📊 BI DASHBOARD  ",
    "  🔬 DATASET & EDA  ",
    "  🤖 MODEL TRAINING  ",
    "  ⚡ LIVE PREDICTION  ",
    "  🧠 EXPLAINABILITY  ",
    "  📋 REPORTS  ",
])

# ══════════════════════════════════════════════
# TAB 1: POWER BI DASHBOARD
# ══════════════════════════════════════════════
with tab1:
    if st.session_state.output_df is None or not st.session_state.all_results:
        st.markdown("""
        <div style="text-align:center; padding:4rem; border:1px dashed #1e2530; border-radius:6px; margin-top:1rem;">
            <div style="font-size:3rem;">📊</div>
            <div style="font-family:'Syne',sans-serif; font-size:1.3rem; font-weight:700; color:#e8e4dc; margin-bottom:0.5rem;">
                BI Dashboard — Train a Model First
            </div>
            <div style="font-family:'JetBrains Mono',monospace; font-size:0.72rem; color:#6b7280;">
                Upload dataset → Dataset & EDA tab → Model Training tab → come back here
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        stats = compute_dashboard_stats(
            st.session_state.output_df,
            st.session_state.target_col,
            st.session_state.all_results
        )

        # ── KPI Row 1 ──
        st.markdown('<div class="section-label" style="margin-top:0;">Key Performance Indicators</div>', unsafe_allow_html=True)
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        kpi_data = [
            (c1, f"{stats['total']:,}", "Total Machines", "", ""),
            (c2, f"{stats['predicted_failures']:,}", "Predicted Failures", "danger", f"Actual: {stats['actual_failures']:,}"),
            (c3, f"{stats['healthy']:,}", "Healthy Machines", "ok", f"{100 - stats['failure_rate']:.1f}% of fleet"),
            (c4, f"{stats['failure_rate']:.1f}%", "Failure Rate", "danger" if stats['failure_rate'] > 10 else "", ""),
            (c5, f"{stats['best_accuracy']:.1%}", "Model Accuracy", "ok", stats['best_model']),
            (c6, f"{stats['best_auc']:.3f}", "ROC AUC", "info", f"F1: {stats['best_f1']:.3f}"),
        ]
        for col, val, label, cls, sub in kpi_data:
            with col:
                st.markdown(f"""
                <div class="kpi-card {cls}">
                    <div class="kpi-value">{val}</div>
                    <div class="kpi-label">{label}</div>
                    <div class="kpi-sub">{sub}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Row 2: Trend + Radar ──
        c_left, c_right = st.columns([2, 1])
        with c_left:
            st.markdown('<div class="chart-card"><h4>Failure Rate Trend</h4>', unsafe_allow_html=True)
            fig = plot_failure_trend(st.session_state.output_df, st.session_state.target_col)
            if fig:
                st.pyplot(fig)
                plt.close()
            st.markdown("</div>", unsafe_allow_html=True)

        with c_right:
            st.markdown('<div class="chart-card"><h4>Model Radar Comparison</h4>', unsafe_allow_html=True)
            fig = plot_model_comparison_radar(st.session_state.all_results)
            st.pyplot(fig)
            plt.close()
            st.markdown("</div>", unsafe_allow_html=True)

        # ── Row 3: Model table + Feature importance ──
        c_left2, c_right2 = st.columns([1, 1])
        with c_left2:
            st.markdown('<div class="chart-card"><h4>Model Comparison Table</h4>', unsafe_allow_html=True)
            res = st.session_state.all_results
            comp = pd.DataFrame({
                "Model": list(res.keys()),
                "Accuracy": [f"{r['accuracy']:.3f}" for r in res.values()],
                "F1": [f"{r['f1']:.3f}" for r in res.values()],
                "AUC": [f"{r['auc']:.3f}" for r in res.values()],
                "CV F1": [f"{r['cv_mean']:.3f} ±{r['cv_std']:.3f}" for r in res.values()],
                "Best?": ["★" if k == st.session_state.best_model_name else "" for k in res.keys()],
            })
            st.dataframe(comp, use_container_width=True, hide_index=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with c_right2:
            st.markdown('<div class="chart-card"><h4>Top Feature Importances</h4>', unsafe_allow_html=True)
            best_clf = st.session_state.all_results[st.session_state.best_model_name]["model"]
            feat_imp = sorted(
                zip(st.session_state.feature_cols, best_clf.feature_importances_),
                key=lambda x: x[1], reverse=True
            )[:8]
            for feat, imp in feat_imp:
                pct = imp * 100
                st.markdown(f"""
                <div class="feat-bar-wrap">
                    <div class="feat-bar-header">
                        <span>{feat[:25]}</span><span style="color:#f5a623;">{pct:.1f}%</span>
                    </div>
                    <div class="feat-bar-bg"><div class="feat-bar-fill" style="width:{pct:.1f}%;"></div></div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # ── Row 4: Sensor boxplots + Heatmap ──
        st.markdown('<div class="section-label">Sensor Analysis</div>', unsafe_allow_html=True)
        fig = plot_sensor_boxplots(
            st.session_state.output_df,
            st.session_state.feature_cols,
            st.session_state.target_col
        )
        st.pyplot(fig)
        plt.close()

        fig2 = plot_failure_heatmap_grid(
            st.session_state.output_df,
            st.session_state.feature_cols
        )
        if fig2:
            st.pyplot(fig2)
            plt.close()


# ══════════════════════════════════════════════
# TAB 2: DATASET & EDA
# ══════════════════════════════════════════════
with tab2:
    if not uploaded_file:
        st.markdown("""
        <div style="text-align:center; padding:4rem; border:1px dashed #1e2530; border-radius:4px; margin-top:1rem;">
            <div style="font-size:3rem;">⚙</div>
            <div style="font-family:'Syne',sans-serif; font-size:1.3rem; font-weight:700; color:#e8e4dc; margin-bottom:0.5rem;">Upload Your Dataset</div>
            <div style="font-family:'JetBrains Mono',monospace; font-size:0.72rem; color:#6b7280;">
                Upload a CSV with machine sensor readings via the sidebar.
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        data = pd.read_csv(uploaded_file)
        st.session_state.raw_data = data
        target_col = detect_target_column(data)

        if not target_col:
            st.error(f"Target column not found. Expected: Machine failure / Failure / Target / Anomaly. Found: {list(data.columns)}")
            st.stop()
        st.session_state.target_col = target_col

        # Overview KPIs
        st.markdown('<div class="section-label" style="margin-top:0;">Dataset Overview</div>', unsafe_allow_html=True)
        c1, c2, c3, c4, c5 = st.columns(5)
        kpis = [
            (c1, f"{len(data):,}", "Total Records", ""),
            (c2, f"{len(data.columns)}", "Features", "info"),
            (c3, f"{data.isnull().sum().sum()}", "Missing Values", "danger" if data.isnull().sum().sum() > 0 else "ok"),
            (c4, f"{int(data[target_col].sum()):,}", "Failure Cases", "danger"),
            (c5, f"{len(data.select_dtypes(include=[np.number]).columns)}", "Numeric Cols", ""),
        ]
        for col, val, label, cls in kpis:
            with col:
                st.markdown(f'<div class="kpi-card {cls}"><div class="kpi-value">{val}</div><div class="kpi-label">{label}</div></div>', unsafe_allow_html=True)

        st.markdown('<div class="section-label">Raw Data Preview</div>', unsafe_allow_html=True)
        st.dataframe(data.head(10), use_container_width=True)

        st.markdown('<div class="section-label">Descriptive Statistics</div>', unsafe_allow_html=True)
        st.dataframe(data.describe().round(3), use_container_width=True)

        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        feature_only = [c for c in numeric_cols if c != target_col]

        # Class distribution + Missing
        st.markdown('<div class="section-label">Exploratory Data Analysis</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            fig = plot_class_distribution(data, target_col)
            st.pyplot(fig); plt.close()
        with c2:
            import matplotlib.pyplot as plt2
            fig2, ax2 = plt.subplots(figsize=(5, 3.5))
            fig2.patch.set_facecolor(CARD_BG); ax2.set_facecolor(CARD_BG)
            missing = data.isnull().sum()
            if missing.sum() == 0:
                ax2.text(0.5, 0.5, "No missing values\nDataset is clean ✓",
                        ha="center", va="center", color="#22c55e", fontsize=12, transform=ax2.transAxes)
            else:
                missing[missing > 0].plot(kind="bar", ax=ax2, color="#ef4444")
            ax2.set_title("Missing Values", color=ACCENT, fontsize=10)
            ax2.tick_params(colors=TEXT_DIM)
            for sp in ax2.spines.values(): sp.set_color(BORDER)
            plt.tight_layout(); st.pyplot(fig2); plt.close()

        # Correlation heatmap
        st.markdown('<div class="section-label">Correlation Heatmap</div>', unsafe_allow_html=True)
        fig = plot_correlation_heatmap(data, numeric_cols)
        st.pyplot(fig); plt.close()

        # Feature distributions
        st.markdown('<div class="section-label">Feature Distributions by Class</div>', unsafe_allow_html=True)
        fig = plot_feature_distributions(data, feature_only, target_col, n=6)
        st.pyplot(fig); plt.close()


# ══════════════════════════════════════════════
# TAB 3: MODEL TRAINING
# ══════════════════════════════════════════════
with tab3:
    if st.session_state.raw_data is None:
        st.warning("Upload a dataset in the Dataset & EDA tab first.")
    else:
        data = st.session_state.raw_data
        target_col = st.session_state.target_col
        if not target_col:
            st.error("Target column not detected. Go to EDA tab first.")
            st.stop()

        st.markdown('<div class="section-label" style="margin-top:0;">Model Configuration</div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        with c1: n_estimators = st.slider("RF / GBM Estimators", 50, 500, 100, 50)
        with c2: xgb_lr = st.slider("XGBoost LR", 0.01, 0.3, 0.1, 0.01)
        with c3: test_size = st.slider("Test Split %", 10, 30, 20, 5)
        with c4: max_depth = st.slider("Max Depth (XGB/GBM)", 2, 10, 5, 1)

        enable_gbm = st.checkbox("Enable GradientBoosting (slower)", value=False)

        train_btn = st.button("🚀  TRAIN ALL MODELS + LOG TO MLFLOW", use_container_width=True)

        if train_btn:
            X, y, col_map, label_encoders = preprocess_data(data, target_col)
            st.session_state.col_map = col_map
            st.session_state.label_encoders = label_encoders

            # ── FIX: Clean column names for XGBoost ──────────────────────
            # XGBoost does not allow [ ] or < in feature names.
            # This happens after one-hot encoding or dtype-based column names.
            X.columns = X.columns.str.replace(r'[\[\]<>]', '_', regex=True)
            # ─────────────────────────────────────────────────────────────

            st.session_state.feature_cols = list(X.columns)
            st.session_state.X_sample = X
            st.session_state.proc_X_full = X

            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size / 100, random_state=42, stratify=y)
            except Exception:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size / 100, random_state=42)

            st.session_state.X_train = X_train
            st.session_state.X_test = X_test
            st.session_state.y_train = y_train
            st.session_state.y_test = y_test

            models_to_train = {
                "Random Forest": RandomForestClassifier(n_estimators=n_estimators, random_state=42, n_jobs=-1),
                "XGBoost": XGBClassifier(
                    learning_rate=xgb_lr, n_estimators=n_estimators,
                    max_depth=max_depth, use_label_encoder=False,
                    eval_metric="logloss", random_state=42, verbosity=0),
            }
            if enable_gbm:
                models_to_train["GradientBoosting"] = GradientBoostingClassifier(
                    n_estimators=n_estimators, max_depth=max_depth, random_state=42)

            all_results = {}
            best_f1 = -1
            best_model = None
            best_model_name = None

            progress = st.progress(0)
            status_box = st.empty()

            for i, (name, clf) in enumerate(models_to_train.items()):
                status_box.markdown(f'<div class="alert-strip info">Training {name}...</div>', unsafe_allow_html=True)
                with mlflow.start_run(run_name=name):
                    mlflow.log_params({
                        "model": name, "n_estimators": n_estimators,
                        "test_size": test_size, "max_depth": max_depth,
                        **({"xgb_lr": xgb_lr} if name == "XGBoost" else {})
                    })
                    clf.fit(X_train, y_train)
                    y_pred = clf.predict(X_test)
                    y_prob = clf.predict_proba(X_test)[:, 1]
                    acc = accuracy_score(y_test, y_pred)
                    f1 = f1_score(y_test, y_pred, average="weighted")
                    try:
                        auc = roc_auc_score(y_test, y_prob)
                    except Exception:
                        auc = 0.0
                    cv_scores = cross_val_score(clf, X, y, cv=5, scoring="f1_weighted")
                    mlflow.log_metrics({
                        "accuracy": acc, "f1_weighted": f1,
                        "roc_auc": auc, "cv_f1_mean": cv_scores.mean(), "cv_f1_std": cv_scores.std()
                    })
                    mlflow.sklearn.log_model(clf, name.replace(" ", "_"))

                    all_results[name] = {
                        "model": clf, "y_pred": y_pred, "y_prob": y_prob,
                        "accuracy": acc, "f1": f1, "auc": auc,
                        "cv_mean": cv_scores.mean(), "cv_std": cv_scores.std(),
                        "report": classification_report(y_test, y_pred)
                    }
                    if f1 > best_f1:
                        best_f1, best_model, best_model_name = f1, clf, name

                progress.progress((i + 1) / len(models_to_train))

            st.session_state.all_results = all_results
            st.session_state.model = best_model
            st.session_state.best_model_name = best_model_name
            st.session_state.accuracy = all_results[best_model_name]["accuracy"]

            # SHAP
            status_box.markdown('<div class="alert-strip info">Computing SHAP values...</div>', unsafe_allow_html=True)
            try:
                explainer = shap.TreeExplainer(best_model)
                shap_vals = explainer.shap_values(X_test)
                st.session_state.shap_values = shap_vals
                st.session_state.explainer = explainer
            except Exception as e:
                st.session_state.shap_values = None

            # Build output
            status_box.markdown('<div class="alert-strip info">Generating reports...</div>', unsafe_allow_html=True)
            out_df = build_output_df(data, X, col_map, best_model)
            out_df[target_col] = y.values
            st.session_state.output_df = out_df

            status_box.markdown(
                f'<div class="alert-strip warn">✅ All models trained! Best: {best_model_name} — Accuracy: {all_results[best_model_name]["accuracy"]:.1%} | F1: {best_f1:.3f} | AUC: {all_results[best_model_name]["auc"]:.3f}</div>',
                unsafe_allow_html=True
            )

        # ── Results ──
        if st.session_state.all_results:
            results = st.session_state.all_results

            st.markdown('<div class="section-label">Detailed Model Results</div>', unsafe_allow_html=True)
            for name, res in results.items():
                is_best = name == st.session_state.best_model_name
                badge = " ★ BEST" if is_best else ""
                with st.expander(f"{name}{badge}", expanded=is_best):
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        fig = plot_confusion_matrix(st.session_state.y_test, res["y_pred"], f"Confusion Matrix — {name}")
                        st.pyplot(fig); plt.close()
                    with c2:
                        try:
                            fig = plot_roc(st.session_state.y_test, res["y_prob"], name)
                            st.pyplot(fig); plt.close()
                        except Exception:
                            st.info("ROC requires binary classification.")
                    with c3:
                        try:
                            fig = plot_precision_recall(st.session_state.y_test, res["y_prob"], name)
                            st.pyplot(fig); plt.close()
                        except Exception:
                            pass
                    st.markdown(f'<div class="report-card"><h4>Classification Report</h4><div class="report-body">{res["report"]}</div></div>', unsafe_allow_html=True)

            # Feature importance bars
            st.markdown('<div class="section-label">Feature Importance — Best Model</div>', unsafe_allow_html=True)
            best_clf = results[st.session_state.best_model_name]["model"]
            feat_imp = sorted(
                zip(st.session_state.feature_cols, best_clf.feature_importances_),
                key=lambda x: x[1], reverse=True
            )[:10]
            for feat, imp in feat_imp:
                pct = imp * 100
                st.markdown(f"""
                <div class="feat-bar-wrap">
                    <div class="feat-bar-header"><span>{feat}</span><span style="color:#f5a623;">{pct:.1f}%</span></div>
                    <div class="feat-bar-bg"><div class="feat-bar-fill" style="width:{pct:.1f}%;"></div></div>
                </div>
                """, unsafe_allow_html=True)


# ══════════════════════════════════════════════
# TAB 4: LIVE PREDICTION
# ══════════════════════════════════════════════
with tab4:
    if st.session_state.model is None:
        st.warning("Train models first in the Model Training tab.")
    else:
        st.markdown('<div class="section-label" style="margin-top:0;">Manual Sensor Input</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="alert-strip info">
            💡 Enter sensor readings manually OR use the Random Sample button to auto-fill from your dataset.
        </div>
        """, unsafe_allow_html=True)

        feature_cols = st.session_state.feature_cols
        X_sample = st.session_state.X_sample
        input_values = {}

        # Random fill button
        if st.button("🎲  Auto-Fill Random Sample from Dataset"):
            rand_row = X_sample.sample(1).iloc[0]
            for feat in feature_cols:
                st.session_state[f"inp_{feat}"] = float(rand_row[feat])

        cols_per_row = 4
        groups = [feature_cols[i:i+cols_per_row] for i in range(0, len(feature_cols), cols_per_row)]
        for group in groups:
            form_cols = st.columns(len(group))
            for idx, feat in enumerate(group):
                with form_cols[idx]:
                    col_data = X_sample[feat]
                    mn, mx, mu = float(col_data.min()), float(col_data.max()), float(col_data.mean())
                    is_int = (col_data.dtype in ["int64","int32"]) or (col_data == col_data.round()).all()
                    step = 1.0 if is_int else max(0.001, (mx - mn) / 1000)
                    default = float(round(mu) if is_int else round(mu, 3))
                    st.markdown(f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:0.62rem;color:#6b7280;margin-bottom:3px;">{feat[:22]} [{mn:.1f}–{mx:.1f}]</div>', unsafe_allow_html=True)
                    val = st.number_input(
                        feat, min_value=float(mn), max_value=float(mx),
                        value=st.session_state.get(f"inp_{feat}", default),
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

            # Prediction box
            if is_fail:
                st.markdown(f'<div class="pred-box fail"><div class="pred-value fail">⚠ FAILURE DETECTED</div><div style="font-family:JetBrains Mono,monospace;font-size:0.78rem;color:#6b7280;margin-top:0.4rem;">Failure probability: {fail_prob:.1%} — Immediate inspection recommended.</div></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="pred-box ok"><div class="pred-value ok">✓ MACHINE HEALTHY</div><div style="font-family:JetBrains Mono,monospace;font-size:0.78rem;color:#6b7280;margin-top:0.4rem;">Normal operation confidence: {ok_prob:.1%}</div></div>', unsafe_allow_html=True)

            # Probability bars
            st.markdown('<div class="section-label">Probability Breakdown</div>', unsafe_allow_html=True)
            for label, pval, color in [("NORMAL OPERATION", ok_prob, "#22c55e"), ("FAILURE RISK", fail_prob, "#ef4444")]:
                st.markdown(f"""
                <div style="margin-bottom:0.65rem;">
                    <div style="display:flex;justify-content:space-between;font-family:'JetBrains Mono',monospace;font-size:0.68rem;color:#9ca3af;margin-bottom:3px;">
                        <span>{label}</span><span style="color:{color};">{pval:.1%}</span>
                    </div>
                    <div style="background:#1e2530;border-radius:2px;height:7px;">
                        <div style="background:{color};width:{pval*100:.1f}%;height:7px;border-radius:2px;transition:width 0.5s;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # SHAP for this prediction
            if st.session_state.shap_values is not None:
                st.markdown('<div class="section-label">SHAP Explanation for This Prediction</div>', unsafe_allow_html=True)
                try:
                    explainer = st.session_state.explainer
                    sv_single = explainer.shap_values(input_df)
                    if isinstance(sv_single, list):
                        sv_single = sv_single[1]
                    sv_single = sv_single[0]
                    shap_df = pd.DataFrame({
                        "Feature": feature_cols,
                        "Value": [input_values[f] for f in feature_cols],
                        "SHAP": sv_single,
                    }).sort_values("SHAP", key=abs, ascending=False).head(10)

                    fig, ax = plt.subplots(figsize=(8, 3.5))
                    fig.patch.set_facecolor(CARD_BG); ax.set_facecolor(CARD_BG)
                    colors = [RED if v > 0 else GREEN for v in shap_df["SHAP"]]
                    ax.barh(shap_df["Feature"], shap_df["SHAP"], color=colors, edgecolor="none", height=0.6)
                    ax.axvline(0, color="#374151", lw=1)
                    ax.set_title("SHAP Feature Contributions for This Input", color=ACCENT, fontsize=10)
                    ax.set_xlabel("SHAP value", color=TEXT_DIM, fontsize=8)
                    for sp in ax.spines.values(): sp.set_color(BORDER)
                    ax.tick_params(colors=TEXT_DIM, labelsize=8)
                    plt.tight_layout()
                    st.pyplot(fig); plt.close()
                    st.dataframe(shap_df.round(4), use_container_width=True, hide_index=True)
                except Exception as e:
                    st.info(f"SHAP for single prediction not available: {e}")

            # Auto-generated report
            st.markdown('<div class="section-label">Auto-Generated Report</div>', unsafe_allow_html=True)
            report = generate_long_report(
                pd.Series(input_values), st.session_state.col_map,
                prediction=int(prediction), probability=float(fail_prob)
            )
            st.markdown(f'<div class="report-card"><div class="report-body">{report}</div></div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════
# TAB 5: EXPLAINABILITY
# ══════════════════════════════════════════════
with tab5:
    if st.session_state.shap_values is None:
        st.warning("Train models first — SHAP values are computed during training.")
    else:
        st.markdown('<div class="section-label" style="margin-top:0;">SHAP — Model Explainability</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="alert-strip info">
            SHAP (SHapley Additive exPlanations) decomposes each prediction into feature contributions — no more black boxes.
        </div>
        """, unsafe_allow_html=True)

        shap_values = st.session_state.shap_values
        X_test = st.session_state.X_test
        feature_cols = st.session_state.feature_cols

        sv = shap_values[1] if isinstance(shap_values, list) else shap_values

        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="section-label">Beeswarm — Feature Impact</div>', unsafe_allow_html=True)
            fig = plot_shap_summary(sv, X_test, feature_cols)
            st.pyplot(fig); plt.close()
        with c2:
            st.markdown('<div class="section-label">Global Mean |SHAP|</div>', unsafe_allow_html=True)
            fig = plot_shap_bar(sv, feature_cols)
            st.pyplot(fig); plt.close()

        st.markdown('<div class="section-label">Individual Prediction Explanation</div>', unsafe_allow_html=True)
        sample_idx = st.slider("Select test sample index", 0, len(X_test) - 1, 0, key="shap_idx")
        fig, shap_contrib = plot_shap_individual(sv, X_test, feature_cols, idx=sample_idx)
        st.pyplot(fig); plt.close()
        st.dataframe(shap_contrib.round(4), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════
# TAB 6: REPORTS
# ══════════════════════════════════════════════
with tab6:
    if st.session_state.output_df is None:
        st.warning("Train models first to generate reports.")
    else:
        output_df = st.session_state.output_df
        st.markdown('<div class="section-label" style="margin-top:0;">All Machine Reports</div>', unsafe_allow_html=True)

        total = len(output_df)
        pred_fail = int(output_df["Predicted_Failure"].sum()) if "Predicted_Failure" in output_df.columns else 0
        healthy = total - pred_fail
        pct = pred_fail / total * 100 if total > 0 else 0
        avg_prob = output_df["Failure_Probability"].mean() * 100 if "Failure_Probability" in output_df.columns else 0

        c1, c2, c3, c4 = st.columns(4)
        for col, val, label, cls in [
            (c1, f"{total:,}", "Total Machines", ""),
            (c2, f"{pred_fail:,}", "Predicted Failures", "danger"),
            (c3, f"{healthy:,}", "Healthy", "ok"),
            (c4, f"{pct:.1f}%", "Failure Rate", "danger" if pct > 10 else ""),
        ]:
            with col:
                st.markdown(f'<div class="kpi-card {cls}"><div class="kpi-value">{val}</div><div class="kpi-label">{label}</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        c_a, c_b = st.columns(2)
        with c_a:
            if st.button("🎲  Random Machine Report", use_container_width=True):
                idx = random.randint(0, len(output_df) - 1)
                row = output_df.iloc[idx]
                pred_val = row.get("Predicted_Failure", 0)
                prob_val = row.get("Failure_Probability", 0)
                status = "FAILURE" if pred_val == 1 else "HEALTHY"
                color = RED if pred_val == 1 else GREEN
                st.markdown(f'<div style="font-family:JetBrains Mono,monospace;font-size:0.7rem;color:#6b7280;margin:0.5rem 0;">RECORD #{idx} | <span style="color:{color};">● {status}</span> | PROB: {prob_val:.1%}</div>', unsafe_allow_html=True)
                rc1, rc2 = st.columns(2)
                with rc1:
                    st.markdown(f'<div class="report-card"><h4>Short Report</h4><div class="report-body">{row["Short_Report"]}</div></div>', unsafe_allow_html=True)
                with rc2:
                    st.markdown(f'<div class="report-card"><h4>Detailed Report</h4><div class="report-body">{row["Long_Report"]}</div></div>', unsafe_allow_html=True)

        with c_b:
            csv = output_df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇  Download All Reports as CSV", data=csv,
                               file_name="smat_ai_v6_reports.csv", mime="text/csv",
                               use_container_width=True)

        # Failures-only filter
        st.markdown('<div class="section-label">Filter & Browse</div>', unsafe_allow_html=True)
        show_only = st.radio("Show:", ["All Machines", "Only Failures", "Only Healthy"], horizontal=True)
        filtered = output_df.copy()
        if show_only == "Only Failures" and "Predicted_Failure" in filtered.columns:
            filtered = filtered[filtered["Predicted_Failure"] == 1]
        elif show_only == "Only Healthy" and "Predicted_Failure" in filtered.columns:
            filtered = filtered[filtered["Predicted_Failure"] == 0]

        preview_cols = [c for c in ["Short_Report", "Predicted_Failure", "Failure_Probability"] if c in filtered.columns]
        st.dataframe(filtered[preview_cols].head(50), use_container_width=True)
