import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import random
import io

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
# Custom CSS — Industrial dark theme
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;700&family=Syne:wght@400;600;700;800&display=swap');

/* Base */
html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
}

/* App background */
.stApp {
    background: #0a0c0f;
    color: #e8e4dc;
}

/* Main content area */
.main .block-container {
    padding: 2rem 2.5rem;
    max-width: 1400px;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0f1218 !important;
    border-right: 1px solid #1e2530;
}
[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    color: #e8e4dc;
}

/* Header banner */
.smat-header {
    background: linear-gradient(135deg, #0f1218 0%, #141920 50%, #0a0c10 100%);
    border: 1px solid #1e2530;
    border-left: 4px solid #f5a623;
    border-radius: 4px;
    padding: 1.5rem 2rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.smat-header::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 300px;
    height: 300px;
    background: radial-gradient(circle, rgba(245,166,35,0.05) 0%, transparent 70%);
    pointer-events: none;
}
.smat-header h1 {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 2.2rem;
    color: #f5a623;
    margin: 0;
    letter-spacing: -0.02em;
}
.smat-header p {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    color: #6b7280;
    margin: 0.3rem 0 0 0;
    letter-spacing: 0.08em;
}
.version-badge {
    display: inline-block;
    background: rgba(245,166,35,0.12);
    border: 1px solid rgba(245,166,35,0.3);
    color: #f5a623;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    padding: 2px 10px;
    border-radius: 2px;
    margin-left: 1rem;
    vertical-align: middle;
    letter-spacing: 0.1em;
}

/* Section headers */
.section-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.15em;
    color: #f5a623;
    text-transform: uppercase;
    margin-bottom: 0.75rem;
    border-bottom: 1px solid #1e2530;
    padding-bottom: 0.4rem;
}

/* Metric cards */
.metric-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
    margin: 1.5rem 0;
}
.metric-card {
    background: #0f1218;
    border: 1px solid #1e2530;
    border-radius: 4px;
    padding: 1.2rem 1.4rem;
    position: relative;
    overflow: hidden;
}
.metric-card::after {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: #f5a623;
}
.metric-card.danger::after { background: #ef4444; }
.metric-card.ok::after { background: #22c55e; }
.metric-card.info::after { background: #3b82f6; }
.metric-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.8rem;
    font-weight: 700;
    color: #e8e4dc;
    line-height: 1;
}
.metric-label {
    font-size: 0.72rem;
    color: #6b7280;
    margin-top: 0.4rem;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    font-family: 'JetBrains Mono', monospace;
}

/* Status indicators */
.status-ok {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(34,197,94,0.1);
    border: 1px solid rgba(34,197,94,0.3);
    color: #22c55e;
    padding: 3px 12px;
    border-radius: 2px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    font-weight: 500;
}
.status-fail {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(239,68,68,0.1);
    border: 1px solid rgba(239,68,68,0.3);
    color: #ef4444;
    padding: 3px 12px;
    border-radius: 2px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    font-weight: 500;
}

/* Report cards */
.report-card {
    background: #0f1218;
    border: 1px solid #1e2530;
    border-radius: 4px;
    padding: 1.4rem;
    margin-bottom: 1rem;
}
.report-card h4 {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.1em;
    color: #f5a623;
    text-transform: uppercase;
    margin-bottom: 0.75rem;
}
.report-body {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    color: #9ca3af;
    line-height: 1.7;
    white-space: pre-wrap;
}

/* Prediction result box */
.pred-box {
    border-radius: 4px;
    padding: 1.8rem 2rem;
    text-align: center;
    margin: 1rem 0;
}
.pred-box.fail {
    background: rgba(239,68,68,0.08);
    border: 2px solid rgba(239,68,68,0.4);
}
.pred-box.ok {
    background: rgba(34,197,94,0.08);
    border: 2px solid rgba(34,197,94,0.35);
}
.pred-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #6b7280;
    margin-bottom: 0.5rem;
}
.pred-value {
    font-family: 'Syne', sans-serif;
    font-size: 2.5rem;
    font-weight: 800;
    line-height: 1;
}
.pred-value.fail { color: #ef4444; }
.pred-value.ok { color: #22c55e; }
.pred-conf {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    color: #6b7280;
    margin-top: 0.4rem;
}

/* Streamlit overrides */
.stButton > button {
    background: #f5a623 !important;
    color: #0a0c0f !important;
    border: none !important;
    border-radius: 3px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-weight: 700 !important;
    letter-spacing: 0.05em !important;
    font-size: 0.8rem !important;
    padding: 0.5rem 1.5rem !important;
    transition: all 0.15s !important;
}
.stButton > button:hover {
    background: #e8961a !important;
    transform: translateY(-1px) !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: #0f1218;
    border-bottom: 1px solid #1e2530;
    gap: 0;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #6b7280 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    border-radius: 0 !important;
    padding: 0.7rem 1.4rem !important;
    border-bottom: 2px solid transparent !important;
}
.stTabs [aria-selected="true"] {
    color: #f5a623 !important;
    border-bottom: 2px solid #f5a623 !important;
    background: transparent !important;
}

/* Inputs */
.stSlider > div > div > div { background: #1e2530 !important; }
.stSlider > div > div > div > div { background: #f5a623 !important; }
.stNumberInput input, .stTextInput input, .stSelectbox select {
    background: #0f1218 !important;
    border: 1px solid #1e2530 !important;
    color: #e8e4dc !important;
    border-radius: 3px !important;
    font-family: 'JetBrains Mono', monospace !important;
}
[data-testid="stFileUploader"] {
    background: #0f1218;
    border: 1px dashed #1e2530;
    border-radius: 4px;
}

/* Dataframe */
.stDataFrame {
    border: 1px solid #1e2530;
    border-radius: 4px;
}

/* Divider */
hr { border-color: #1e2530 !important; }

/* Info/success/error banners */
.stAlert {
    background: #0f1218 !important;
    border-left-color: #f5a623 !important;
}

/* Hide default streamlit elements */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Session State
# ─────────────────────────────────────────────
if "model" not in st.session_state:
    st.session_state.model = None
if "feature_cols" not in st.session_state:
    st.session_state.feature_cols = None
if "X_sample" not in st.session_state:
    st.session_state.X_sample = None
if "accuracy" not in st.session_state:
    st.session_state.accuracy = None
if "output_df" not in st.session_state:
    st.session_state.output_df = None
if "col_map" not in st.session_state:
    st.session_state.col_map = {}
if "label_encoders" not in st.session_state:
    st.session_state.label_encoders = {}

# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────
st.markdown("""
<div class="smat-header">
    <h1>⚙ SMAT.AI <span class="version-badge">v4.0</span></h1>
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
    st.markdown(f"""
    <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; color: #6b7280; line-height: 2;">
        <div>MODEL STATUS: <span style="color:{color}; font-weight:700;">{model_status}</span></div>
        <div>ENGINE: <span style="color:#e8e4dc;">Random Forest</span></div>
        <div>ESTIMATORS: <span style="color:#e8e4dc;">100</span></div>
        <div>TEST SPLIT: <span style="color:#e8e4dc;">20%</span></div>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.accuracy:
        st.markdown(f"""
        <div style="margin-top:1rem; background:#0a0c0f; border:1px solid #1e2530; border-left: 3px solid #f5a623; padding:0.8rem 1rem; border-radius:3px;">
            <div style="font-family:'JetBrains Mono',monospace; font-size:0.65rem; color:#6b7280; letter-spacing:0.1em;">MODEL ACCURACY</div>
            <div style="font-family:'JetBrains Mono',monospace; font-size:1.5rem; font-weight:700; color:#f5a623;">{st.session_state.accuracy:.1%}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style="font-family:'JetBrains Mono',monospace; font-size:0.65rem; color:#374151; line-height:2; letter-spacing:0.05em;">
    GAURAV SHUKLA<br>
    AI & DATA ENGINEER<br>
    B.TECH CSE (AI & ML)
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────
def find_col(col_map, name):
    """Case-insensitive partial column name match."""
    name = name.lower()
    for c in col_map:
        if name in c:
            return col_map[c]
    return None

def generate_short_report(row, col_map):
    lines = []
    temp_col = find_col(col_map, 'temperature')
    pres_col = find_col(col_map, 'pressure')
    vib_col = find_col(col_map, 'vibration')
    speed_col = find_col(col_map, 'speed')
    torque_col = find_col(col_map, 'torque')
    note_col = find_col(col_map, 'operator_note')

    if temp_col and temp_col in row.index:
        val = row[temp_col]
        lines.append(f"TEMP {'HIGH' if val > 80 else 'OK'}: {val:.1f}")
    if pres_col and pres_col in row.index:
        val = row[pres_col]
        lines.append(f"PRESSURE {'LOW' if val < 25 else 'OK'}: {val:.1f}")
    if vib_col and vib_col in row.index:
        val = row[vib_col]
        lines.append(f"VIB {'HIGH' if val > 0.03 else 'OK'}: {val:.4f}")
    if speed_col and speed_col in row.index:
        val = row[speed_col]
        lines.append(f"SPEED: {val:.0f} RPM")
    if torque_col and torque_col in row.index:
        val = row[torque_col]
        lines.append(f"TORQUE: {val:.1f} Nm")
    if note_col and note_col in row.index and pd.notna(row.get(note_col)):
        lines.append(f"NOTE: {row[note_col]}")

    return " | ".join(lines) if lines else "All sensor readings within normal range."

def generate_long_report(row, col_map):
    report = "━━━ MACHINE HEALTH REPORT ━━━\n\n"
    report += "SENSOR READINGS:\n"
    for orig_name, col in col_map.items():
        if col in row.index:
            val = row[col]
            if isinstance(val, (int, float, np.number)):
                report += f"  {col}: {val:.3f}\n"
            else:
                report += f"  {col}: {val}\n"

    report += "\nANALYSIS:\n"

    temp_col = find_col(col_map, 'temperature')
    pres_col = find_col(col_map, 'pressure')
    vib_col = find_col(col_map, 'vibration')
    speed_col = find_col(col_map, 'speed')
    torque_col = find_col(col_map, 'torque')

    if temp_col and temp_col in row.index:
        t = row[temp_col]
        if t > 85:
            report += "  [CRITICAL] Temperature critically high — inspect cooling system immediately.\n"
        elif t > 75:
            report += "  [WARNING]  Temperature elevated — monitor next cycles closely.\n"
        else:
            report += "  [OK]       Temperature within safe operating range.\n"

    if pres_col and pres_col in row.index:
        p = row[pres_col]
        if p < 25:
            report += "  [WARNING]  Low pressure — check valves and pump efficiency.\n"
        elif p > 35:
            report += "  [WARNING]  High pressure — inspect for blockages or leaks.\n"
        else:
            report += "  [OK]       Pressure normal.\n"

    if vib_col and vib_col in row.index:
        v = row[vib_col]
        if v > 0.03:
            report += "  [CRITICAL] Excessive vibration — inspect bearings and rotating parts.\n"
        elif v > 0.02:
            report += "  [WARNING]  Mild vibration — monitor mechanical components.\n"
        else:
            report += "  [OK]       Vibration levels normal.\n"

    if speed_col and speed_col in row.index:
        s = row[speed_col]
        if s > 3000:
            report += "  [WARNING]  High rotational speed detected.\n"
        else:
            report += "  [OK]       Rotational speed normal.\n"

    if torque_col and torque_col in row.index:
        tq = row[torque_col]
        if tq > 60:
            report += "  [WARNING]  High torque — check load conditions.\n"
        else:
            report += "  [OK]       Torque within normal range.\n"

    report += "\nRECOMMENDATION:\n"
    report += "  Schedule inspection based on critical readings above.\n"
    report += "  Refer to maintenance manual for detailed procedures.\n"
    return report

# ─────────────────────────────────────────────
# Main App
# ─────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["  DATASET & MODEL  ", "  LIVE PREDICTION  ", "  REPORTS  "])

# ═══════════════════════════════════════════
# TAB 1: Dataset & Model
# ═══════════════════════════════════════════
with tab1:
    if not uploaded_file:
        st.markdown("""
        <div style="text-align:center; padding: 4rem 2rem; border: 1px dashed #1e2530; border-radius: 4px; margin-top: 1rem;">
            <div style="font-size: 3rem; margin-bottom: 1rem;">⚙</div>
            <div style="font-family:'Syne',sans-serif; font-size:1.4rem; font-weight:700; color:#e8e4dc; margin-bottom:0.5rem;">Upload Your Dataset</div>
            <div style="font-family:'JetBrains Mono',monospace; font-size:0.75rem; color:#6b7280; line-height:1.8;">
                Upload a CSV with machine sensor readings via the sidebar.<br>
                Required columns: Temperature · Pressure · Vibration · Machine failure<br>
                Optional: Speed · Torque · Tool wear · operator_note
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        data = pd.read_csv(uploaded_file)

        # Metrics row
        st.markdown('<div class="section-label">Dataset Overview</div>', unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{len(data):,}</div><div class="metric-label">Total Records</div></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="metric-card info"><div class="metric-value">{len(data.columns)}</div><div class="metric-label">Features</div></div>', unsafe_allow_html=True)
        with col3:
            missing = data.isnull().sum().sum()
            cls = "danger" if missing > 0 else "ok"
            st.markdown(f'<div class="metric-card {cls}"><div class="metric-value">{missing}</div><div class="metric-label">Missing Values</div></div>', unsafe_allow_html=True)
        with col4:
            num_cols = len(data.select_dtypes(include=[np.number]).columns)
            st.markdown(f'<div class="metric-card"><div class="metric-value">{num_cols}</div><div class="metric-label">Numeric Cols</div></div>', unsafe_allow_html=True)

        st.markdown('<div class="section-label" style="margin-top:1.5rem;">Raw Data Preview</div>', unsafe_allow_html=True)
        st.dataframe(data.head(10), use_container_width=True)

        # Detect target column
        possible_targets = ["Machine failure", "machine failure", "Failure", "failure", "Target", "target", "Anomaly"]
        target_col = None
        lower_map = {col.strip().lower(): col for col in data.columns}
        for name in possible_targets:
            if name.strip().lower() in lower_map:
                target_col = lower_map[name.strip().lower()]
                break

        if target_col is None:
            st.error(f"Target column not found. Expected: Machine failure, Failure, Target, Anomaly.\nFound: {list(data.columns)}")
            st.stop()

        st.markdown(f"""
        <div style="font-family:'JetBrains Mono',monospace; font-size:0.75rem; color:#6b7280; margin: 0.75rem 0;">
            TARGET COLUMN DETECTED: <span style="color:#f5a623; font-weight:700;">{target_col}</span> &nbsp;|&nbsp;
            CLASS DISTRIBUTION: <span style="color:#e8e4dc;">{dict(data[target_col].value_counts().to_dict())}</span>
        </div>
        """, unsafe_allow_html=True)

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

        if y.nunique() < 2:
            st.error("Target column has only one class. Need at least two.")
            st.stop()

        col_map = {c.strip().lower(): c for c in X.columns}
        st.session_state.col_map = col_map
        st.session_state.feature_cols = list(X.columns)
        st.session_state.X_sample = X
        st.session_state.label_encoders = label_encoders

        # Train model
        st.markdown('<div class="section-label" style="margin-top:1.5rem;">Model Training</div>', unsafe_allow_html=True)

        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        except ValueError:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        with st.spinner("Training Random Forest model..."):
            model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)

        st.session_state.model = model
        st.session_state.accuracy = acc

        # Show results
        c1, c2 = st.columns([1, 2])
        with c1:
            st.markdown(f"""
            <div class="metric-card ok" style="margin-bottom:1rem;">
                <div class="metric-value">{acc:.1%}</div>
                <div class="metric-label">Model Accuracy</div>
            </div>
            """, unsafe_allow_html=True)

            fail_rate = y.mean() if y.dtype in [int, float] else (y == 1).mean()
            st.markdown(f"""
            <div class="metric-card danger">
                <div class="metric-value">{fail_rate:.1%}</div>
                <div class="metric-label">Failure Rate in Dataset</div>
            </div>
            """, unsafe_allow_html=True)

        with c2:
            report_text = classification_report(y_test, y_pred)
            st.markdown(f'<div class="report-card"><h4>Classification Report</h4><div class="report-body">{report_text}</div></div>', unsafe_allow_html=True)

        # Feature importance
        st.markdown('<div class="section-label" style="margin-top:1.5rem;">Feature Importance</div>', unsafe_allow_html=True)
        feat_imp = pd.DataFrame({
            'Feature': X.columns,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False).head(10)

        for _, row_fi in feat_imp.iterrows():
            pct = row_fi['Importance'] * 100
            st.markdown(f"""
            <div style="margin-bottom:0.5rem;">
                <div style="display:flex; justify-content:space-between; font-family:'JetBrains Mono',monospace; font-size:0.72rem; color:#9ca3af; margin-bottom:3px;">
                    <span>{row_fi['Feature']}</span>
                    <span style="color:#f5a623;">{pct:.1f}%</span>
                </div>
                <div style="background:#1e2530; border-radius:2px; height:4px;">
                    <div style="background:#f5a623; width:{pct}%; height:4px; border-radius:2px;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Generate reports for download
        output_df = data.copy()
        output_df['Short_Report'] = X.apply(lambda r: generate_short_report(r, col_map), axis=1)
        output_df['Long_Report'] = X.apply(lambda r: generate_long_report(r, col_map), axis=1)
        output_df['Predicted_Failure'] = model.predict(X)
        st.session_state.output_df = output_df

        st.success(f"✅ Model trained successfully! Accuracy: {acc:.1%} — Go to LIVE PREDICTION tab to test manually.")

# ═══════════════════════════════════════════
# TAB 2: Live Prediction
# ═══════════════════════════════════════════
with tab2:
    if st.session_state.model is None:
        st.markdown("""
        <div style="text-align:center; padding:3rem; border:1px dashed #1e2530; border-radius:4px;">
            <div style="font-family:'JetBrains Mono',monospace; font-size:0.8rem; color:#6b7280;">
                ⚠ Train a model first — upload a dataset in the DATASET & MODEL tab.
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown('<div class="section-label">Manual Sensor Input — Live Prediction</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="font-family:'JetBrains Mono',monospace; font-size:0.72rem; color:#6b7280; margin-bottom:1.5rem; line-height:1.8;">
            Enter sensor readings manually below. The trained model will predict machine failure probability in real-time.
        </div>
        """, unsafe_allow_html=True)

        feature_cols = st.session_state.feature_cols
        X_sample = st.session_state.X_sample

        # Build input form dynamically based on dataset columns
        input_values = {}

        # Organize into 3 columns
        cols_per_row = 3
        col_groups = [feature_cols[i:i+cols_per_row] for i in range(0, len(feature_cols), cols_per_row)]

        for group in col_groups:
            form_cols = st.columns(len(group))
            for idx, feat in enumerate(group):
                with form_cols[idx]:
                    col_data = X_sample[feat]
                    min_val = float(col_data.min())
                    max_val = float(col_data.max())
                    mean_val = float(col_data.mean())
                    std_val = float(col_data.std())

                    # Detect if integer-like
                    is_int = (col_data.dtype in ['int64', 'int32']) or (col_data == col_data.round()).all()

                    st.markdown(f"""
                    <div style="font-family:'JetBrains Mono',monospace; font-size:0.65rem; color:#6b7280; margin-bottom:4px; letter-spacing:0.05em;">
                        {feat} <span style="color:#374151;">· range [{min_val:.1f} – {max_val:.1f}]</span>
                    </div>
                    """, unsafe_allow_html=True)

                    if is_int and (max_val - min_val) < 10000:
                        val = st.number_input(
                            feat,
                            min_value=float(min_val),
                            max_value=float(max_val),
                            value=float(round(mean_val)),
                            step=1.0,
                            label_visibility="collapsed",
                            key=f"input_{feat}"
                        )
                    else:
                        step = max(0.001, (max_val - min_val) / 1000)
                        val = st.number_input(
                            feat,
                            min_value=float(min_val - std_val),
                            max_value=float(max_val + std_val),
                            value=float(round(mean_val, 3)),
                            step=float(round(step, 4)),
                            label_visibility="collapsed",
                            key=f"input_{feat}"
                        )
                    input_values[feat] = val

        st.markdown("<br>", unsafe_allow_html=True)

        predict_btn = st.button("⚡  RUN PREDICTION", use_container_width=True)

        if predict_btn:
            input_df = pd.DataFrame([input_values])

            # Predict
            prediction = st.session_state.model.predict(input_df)[0]
            proba = st.session_state.model.predict_proba(input_df)[0]
            fail_prob = proba[1] if len(proba) > 1 else proba[0]
            ok_prob = 1 - fail_prob

            is_fail = bool(prediction == 1)

            # Result display
            if is_fail:
                st.markdown(f"""
                <div class="pred-box fail">
                    <div class="pred-title">Prediction Result</div>
                    <div class="pred-value fail">⚠ FAILURE DETECTED</div>
                    <div class="pred-conf">Failure probability: {fail_prob:.1%} · Confidence: {max(fail_prob, ok_prob):.1%}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="pred-box ok">
                    <div class="pred-title">Prediction Result</div>
                    <div class="pred-value ok">✓ MACHINE HEALTHY</div>
                    <div class="pred-conf">Normal operation probability: {ok_prob:.1%} · Confidence: {max(fail_prob, ok_prob):.1%}</div>
                </div>
                """, unsafe_allow_html=True)

            # Probability bars
            st.markdown('<div class="section-label" style="margin-top:1.5rem;">Probability Breakdown</div>', unsafe_allow_html=True)
            st.markdown(f"""
            <div style="margin-bottom:0.75rem;">
                <div style="display:flex; justify-content:space-between; font-family:'JetBrains Mono',monospace; font-size:0.72rem; color:#9ca3af; margin-bottom:4px;">
                    <span>NORMAL OPERATION</span><span style="color:#22c55e;">{ok_prob:.1%}</span>
                </div>
                <div style="background:#1e2530; border-radius:2px; height:6px;">
                    <div style="background:#22c55e; width:{ok_prob*100:.1f}%; height:6px; border-radius:2px;"></div>
                </div>
            </div>
            <div>
                <div style="display:flex; justify-content:space-between; font-family:'JetBrains Mono',monospace; font-size:0.72rem; color:#9ca3af; margin-bottom:4px;">
                    <span>FAILURE RISK</span><span style="color:#ef4444;">{fail_prob:.1%}</span>
                </div>
                <div style="background:#1e2530; border-radius:2px; height:6px;">
                    <div style="background:#ef4444; width:{fail_prob*100:.1f}%; height:6px; border-radius:2px;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Auto-generate report for this input
            st.markdown('<div class="section-label" style="margin-top:1.5rem;">Generated Reports</div>', unsafe_allow_html=True)
            input_row = pd.Series(input_values)
            short_rep = generate_short_report(input_row, st.session_state.col_map)
            long_rep = generate_long_report(input_row, st.session_state.col_map)

            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f'<div class="report-card"><h4>Short Report</h4><div class="report-body">{short_rep}</div></div>', unsafe_allow_html=True)
            with c2:
                st.markdown(f'<div class="report-card"><h4>Detailed Report</h4><div class="report-body">{long_rep}</div></div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════
# TAB 3: Reports
# ═══════════════════════════════════════════
with tab3:
    if st.session_state.output_df is None:
        st.markdown("""
        <div style="text-align:center; padding:3rem; border:1px dashed #1e2530; border-radius:4px;">
            <div style="font-family:'JetBrains Mono',monospace; font-size:0.8rem; color:#6b7280;">
                ⚠ Upload a dataset and train the model first.
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        output_df = st.session_state.output_df

        st.markdown('<div class="section-label">All Machine Reports</div>', unsafe_allow_html=True)

        # Summary stats
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

        st.markdown("<br>", unsafe_allow_html=True)

        # Random report viewer
        col_btn, col_idx = st.columns([1, 3])
        with col_btn:
            show_random = st.button("🎲  Random Report")

        if show_random:
            idx = random.randint(0, len(output_df) - 1)
            row = output_df.iloc[idx]
            pred_val = row.get('Predicted_Failure', 0)
            status_html = '<span class="status-fail">● FAILURE</span>' if pred_val == 1 else '<span class="status-ok">● HEALTHY</span>'

            st.markdown(f"""
            <div style="font-family:'JetBrains Mono',monospace; font-size:0.72rem; color:#6b7280; margin-bottom:1rem;">
                RECORD #{idx} &nbsp;|&nbsp; {status_html}
            </div>
            """, unsafe_allow_html=True)

            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f'<div class="report-card"><h4>Short Report</h4><div class="report-body">{row["Short_Report"]}</div></div>', unsafe_allow_html=True)
            with c2:
                st.markdown(f'<div class="report-card"><h4>Detailed Report</h4><div class="report-body">{row["Long_Report"]}</div></div>', unsafe_allow_html=True)

        st.markdown('<div class="section-label" style="margin-top:1.5rem;">Download</div>', unsafe_allow_html=True)
        csv = output_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇  Download All Reports as CSV",
            data=csv,
            file_name="smat_ai_maintenance_reports.csv",
            mime="text/csv",
            use_container_width=True
        )

        st.markdown('<div class="section-label" style="margin-top:1.5rem;">Reports Table Preview</div>', unsafe_allow_html=True)
        preview_cols = ['Short_Report', 'Long_Report', 'Predicted_Failure']
        available = [c for c in preview_cols if c in output_df.columns]
        st.dataframe(output_df[available].head(20), use_container_width=True)





