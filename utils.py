"""
SMAT.AI v6.0 — Utility Functions
Helper functions for ML, reports, plotting, and preprocessing.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import shap
import io
import random
from sklearn.metrics import (
    confusion_matrix, roc_curve, roc_auc_score,
    classification_report, f1_score, accuracy_score,
    precision_recall_curve, average_precision_score
)

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
DARK_BG   = "#0a0c0f"
CARD_BG   = "#0f1218"
ACCENT    = "#f5a623"
BORDER    = "#1e2530"
TEXT_DIM  = "#6b7280"
TEXT_MAIN = "#e8e4dc"
GREEN     = "#22c55e"
RED       = "#ef4444"
BLUE      = "#3b82f6"
PURPLE    = "#a855f7"

POSSIBLE_TARGETS = [
    "Machine failure", "machine failure", "Failure", "failure",
    "Target", "target", "Anomaly", "anomaly", "label", "Label"
]

ID_COLS = {"udi", "id", "serialnumber", "product id", "product_id", "index"}

# ─────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────

def detect_target_column(data: pd.DataFrame):
    """Auto-detect the target/label column."""
    lower_map = {col.strip().lower(): col for col in data.columns}
    for name in POSSIBLE_TARGETS:
        if name.strip().lower() in lower_map:
            return lower_map[name.strip().lower()]
    return None


def preprocess_data(data: pd.DataFrame, target_col: str):
    """
    Encode categoricals, drop ID-like columns, fill nulls.
    Returns (X, y, col_map, label_encoders).
    """
    from sklearn.preprocessing import LabelEncoder

    proc = data.copy()
    label_encoders = {}

    # Drop ID columns
    drop_cols = [c for c in proc.columns if c.strip().lower() in ID_COLS and c != target_col]
    proc = proc.drop(columns=drop_cols, errors="ignore")

    # Fill nulls
    for col in proc.columns:
        if proc[col].dtype in ["float64", "float32", "int64", "int32"]:
            proc[col] = proc[col].fillna(proc[col].median())
        else:
            proc[col] = proc[col].fillna(proc[col].mode()[0] if len(proc[col].mode()) > 0 else "Unknown")

    # Encode categoricals (except target)
    non_numeric = proc.select_dtypes(exclude=["int64", "float64", "int32", "float32"]).columns
    for col in non_numeric:
        if col != target_col:
            le = LabelEncoder()
            proc[col] = le.fit_transform(proc[col].astype(str))
            label_encoders[col] = le

    X = proc.drop(columns=[target_col])
    y = proc[target_col]
    col_map = {c.strip().lower(): c for c in X.columns}

    return X, y, col_map, label_encoders


# ─────────────────────────────────────────────
# REPORT GENERATION
# ─────────────────────────────────────────────

def find_col(col_map, name):
    name = name.lower()
    for c in col_map:
        if name in c:
            return col_map[c]
    return None


def generate_short_report(row, col_map):
    """One-line sensor summary."""
    lines = []
    for key in ["temperature", "pressure", "vibration", "speed", "torque", "wear"]:
        col = find_col(col_map, key)
        if col and col in row.index:
            val = row[col]
            lines.append(f"{key[:4].upper()}: {val:.2f}" if isinstance(val, (int, float, np.number)) else f"{key[:4].upper()}: {val}")
    return " | ".join(lines) if lines else "Sensor readings recorded."


def generate_long_report(row, col_map, prediction=None, probability=None):
    """Detailed machine health report."""
    status = "FAILURE RISK" if prediction == 1 else "HEALTHY" if prediction == 0 else "UNKNOWN"
    prob_str = f"{probability:.1%}" if probability is not None else "N/A"

    report = (
        f"═══ MACHINE HEALTH REPORT ═══\n"
        f"STATUS  : {status}\n"
        f"FAIL PROB: {prob_str}\n\n"
        f"SENSOR READINGS:\n"
    )
    for orig_name, col in col_map.items():
        if col in row.index:
            val = row[col]
            report += f"  {col:<30}: {val:.3f}\n" if isinstance(val, (int, float, np.number)) else f"  {col:<30}: {val}\n"

    recommendation = (
        "ACTION: Immediate inspection required. Check mechanical components."
        if prediction == 1
        else "ACTION: Continue normal operation. Schedule next routine check."
    )
    report += f"\nRECOMMENDATION:\n  {recommendation}\n"
    return report


# ─────────────────────────────────────────────
# PLOTS — styled for dark industrial theme
# ─────────────────────────────────────────────

def _fig_setup(figsize=(6, 4)):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(CARD_BG)
    ax.set_facecolor(CARD_BG)
    for spine in ax.spines.values():
        spine.set_color(BORDER)
    ax.tick_params(colors=TEXT_DIM, labelsize=8)
    return fig, ax


def plot_confusion_matrix(y_test, y_pred, title="Confusion Matrix"):
    fig, ax = _fig_setup((4, 3.5))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="YlOrBr", ax=ax,
                linewidths=0.5, linecolor=BORDER, annot_kws={"size": 13, "color": "white"})
    ax.set_title(title, color=ACCENT, fontsize=10, pad=10)
    ax.set_xlabel("Predicted", color=TEXT_DIM, fontsize=9)
    ax.set_ylabel("Actual", color=TEXT_DIM, fontsize=9)
    plt.tight_layout()
    return fig


def plot_roc(y_test, y_prob, label="Model"):
    fig, ax = _fig_setup((5, 3.5))
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    ax.plot(fpr, tpr, color=ACCENT, lw=2, label=f"AUC = {auc:.3f}")
    ax.fill_between(fpr, tpr, alpha=0.1, color=ACCENT)
    ax.plot([0, 1], [0, 1], color="#374151", linestyle="--", lw=1)
    ax.set_xlabel("False Positive Rate", color=TEXT_DIM, fontsize=9)
    ax.set_ylabel("True Positive Rate", color=TEXT_DIM, fontsize=9)
    ax.set_title(f"ROC Curve — {label}", color=ACCENT, fontsize=10)
    ax.legend(facecolor=CARD_BG, edgecolor=BORDER, labelcolor=TEXT_MAIN, fontsize=9)
    plt.tight_layout()
    return fig


def plot_precision_recall(y_test, y_prob, label="Model"):
    fig, ax = _fig_setup((5, 3.5))
    prec, rec, _ = precision_recall_curve(y_test, y_prob)
    ap = average_precision_score(y_test, y_prob)
    ax.plot(rec, prec, color=PURPLE, lw=2, label=f"AP = {ap:.3f}")
    ax.fill_between(rec, prec, alpha=0.1, color=PURPLE)
    ax.set_xlabel("Recall", color=TEXT_DIM, fontsize=9)
    ax.set_ylabel("Precision", color=TEXT_DIM, fontsize=9)
    ax.set_title(f"Precision-Recall — {label}", color=ACCENT, fontsize=10)
    ax.legend(facecolor=CARD_BG, edgecolor=BORDER, labelcolor=TEXT_MAIN, fontsize=9)
    plt.tight_layout()
    return fig


def plot_class_distribution(data, target_col):
    fig, ax = _fig_setup((5, 3.5))
    counts = data[target_col].value_counts().sort_index()
    labels = [f"Healthy (0)" if i == 0 else f"Failure (1)" for i in counts.index]
    colors = [GREEN, RED]
    bars = ax.bar(labels, counts.values, color=colors, width=0.45, edgecolor=BORDER, linewidth=0.5)
    ax.set_title("Class Distribution", color=ACCENT, fontsize=10)
    ax.set_ylabel("Count", color=TEXT_DIM, fontsize=9)
    for bar, v in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 5, f"{v:,}",
                color=TEXT_MAIN, ha="center", fontsize=9, fontweight="bold")
    plt.tight_layout()
    return fig


def plot_correlation_heatmap(data, numeric_cols):
    n = min(len(numeric_cols), 12)
    cols = numeric_cols[:n]
    fig, ax = _fig_setup((10, 5))
    corr = data[cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="YlOrBr",
                ax=ax, linewidths=0.4, linecolor=DARK_BG,
                annot_kws={"size": 7}, vmin=-1, vmax=1)
    ax.set_title("Feature Correlation Matrix", color=ACCENT, fontsize=11, pad=10)
    ax.tick_params(colors=TEXT_DIM, labelsize=8)
    plt.tight_layout()
    return fig


def plot_feature_distributions(data, feature_cols, target_col, n=6):
    cols = feature_cols[:n]
    nrows = (len(cols) + 2) // 3
    fig, axes = plt.subplots(nrows, 3, figsize=(12, 4 * nrows))
    fig.patch.set_facecolor(CARD_BG)
    axes = axes.flatten()
    for i, col in enumerate(cols):
        axes[i].set_facecolor(CARD_BG)
        for cls_val, color, lbl in [(0, GREEN, "Healthy"), (1, RED, "Failure")]:
            subset = data[data[target_col] == cls_val][col].dropna()
            axes[i].hist(subset, bins=30, alpha=0.6, color=color, label=lbl, edgecolor="none")
        axes[i].set_title(col, color=ACCENT, fontsize=9)
        axes[i].tick_params(colors=TEXT_DIM, labelsize=7)
        for spine in axes[i].spines.values():
            spine.set_color(BORDER)
        axes[i].legend(fontsize=7, facecolor=CARD_BG, labelcolor=TEXT_DIM, edgecolor=BORDER)
    for j in range(len(cols), len(axes)):
        axes[j].set_visible(False)
    plt.suptitle("Feature Distributions — Healthy vs Failure", color=TEXT_MAIN, fontsize=11, y=1.01)
    plt.tight_layout()
    return fig


def plot_feature_importance(feature_cols, importances, top_n=10):
    df = pd.DataFrame({"Feature": feature_cols, "Importance": importances})
    df = df.sort_values("Importance", ascending=True).tail(top_n)
    fig, ax = _fig_setup((8, 4))
    bars = ax.barh(df["Feature"], df["Importance"], color=ACCENT, edgecolor="none", height=0.6)
    ax.set_title(f"Top {top_n} Feature Importances", color=ACCENT, fontsize=10)
    ax.set_xlabel("Importance", color=TEXT_DIM, fontsize=9)
    for bar in bars:
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{bar.get_width():.3f}", va="center", color=TEXT_DIM, fontsize=7)
    plt.tight_layout()
    return fig


def plot_shap_summary(shap_values, X_test, feature_cols):
    fig, _ = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(CARD_BG)
    shap.summary_plot(shap_values, X_test, feature_names=feature_cols, show=False)
    plt.gca().set_facecolor(CARD_BG)
    fig = plt.gcf()
    fig.patch.set_facecolor(CARD_BG)
    plt.title("SHAP — Feature Impact on Failure Prediction", color=ACCENT, fontsize=11, pad=10)
    plt.tight_layout()
    return fig


def plot_shap_bar(shap_values, feature_cols, top_n=10):
    # SHAP values handle karna: Binary classification mein ye (n, f, 2) ho sakta hai
    # Hamein sirf positive class (Failure) ke impact se matlab hai
    if isinstance(shap_values, np.ndarray):
        if len(shap_values.shape) == 3:  # (samples, features, classes)
            sv_to_plot = shap_values[:, :, 1]
        else:
            sv_to_plot = shap_values
    else:
        # Agar Explainer object hai ya list hai
        sv_to_plot = np.array(shap_values)
        if len(sv_to_plot.shape) == 3:
            sv_to_plot = sv_to_plot[:, :, 1]

    mean_shap = np.abs(sv_to_plot).mean(axis=0)
    
    # Ensure 1D and match length
    mean_shap = np.array(mean_shap).flatten()
    current_features = feature_cols[:len(mean_shap)]
    
    df = pd.DataFrame({"Feature": current_features, "Mean |SHAP|": mean_shap})
    df = df.sort_values("Mean |SHAP|", ascending=True).tail(top_n)
    
    fig, ax = _fig_setup((8, 4))
    ax.barh(df["Feature"], df["Mean |SHAP|"], color=PURPLE, edgecolor="none", height=0.6)
    ax.set_title(f"Top {top_n} Features — Global SHAP Importance", color=ACCENT, fontsize=10)
    ax.set_xlabel("Mean |SHAP value|", color=TEXT_DIM, fontsize=9)
    plt.tight_layout()
    return fig


def plot_shap_individual(shap_values, X_test, feature_cols, idx=0):
    # Same handling for individual sample
    if isinstance(shap_values, np.ndarray):
        if len(shap_values.shape) == 3:
            sample_shap = shap_values[idx, :, 1]
        else:
            sample_shap = shap_values[idx]
    else:
        # Fallback for Explainer objects
        try:
            sample_shap = shap_values.values[idx]
            # Agar output multi-class hai (.values will be 3D)
            if len(sample_shap.shape) == 2:
                sample_shap = sample_shap[:, 1]
        except:
            sample_shap = np.array(shap_values)[idx]
            if len(sample_shap.shape) == 2:
                sample_shap = sample_shap[:, 1]

    sample_shap = np.array(sample_shap).flatten()
    sample_feats = X_test.iloc[idx]
    
    # Dataframe creation with slicing for safety
    df = pd.DataFrame({
        "Feature": feature_cols[:len(sample_shap)], 
        "Value": sample_feats.values[:len(sample_shap)], 
        "SHAP": sample_shap
    }).sort_values("SHAP", key=abs, ascending=False).head(10)

    fig, ax = _fig_setup((9, 4))
    # Red for pushing towards Failure (>0), Green for pushing towards Healthy (<0)
    colors = [RED if v > 0 else GREEN for v in df["SHAP"]]
    ax.barh(df["Feature"], df["SHAP"], color=colors, edgecolor="none")
    ax.axvline(0, color="#374151", lw=1)
    ax.set_title(f"SHAP Explanation — Sample #{idx}", color=ACCENT, fontsize=10)
    ax.set_xlabel("SHAP value (red=failure risk, green=healthy)", color=TEXT_DIM, fontsize=8)
    plt.tight_layout()
    return fig, df


# ─────────────────────────────────────────────
# POWER BI-STYLE DASHBOARD HELPERS
# ─────────────────────────────────────────────

def compute_dashboard_stats(output_df, target_col, results):
    """Return key stats dict for the BI dashboard."""
    total = len(output_df)
    actual_failures = int(output_df[target_col].sum()) if target_col in output_df.columns else 0
    predicted_failures = int(output_df["Predicted_Failure"].sum()) if "Predicted_Failure" in output_df.columns else 0
    healthy = total - predicted_failures
    failure_rate = predicted_failures / total * 100 if total > 0 else 0

    best_name = max(results, key=lambda k: results[k]["f1"])
    best = results[best_name]

    return {
        "total": total,
        "actual_failures": actual_failures,
        "predicted_failures": predicted_failures,
        "healthy": healthy,
        "failure_rate": failure_rate,
        "best_model": best_name,
        "best_accuracy": best["accuracy"],
        "best_f1": best["f1"],
        "best_auc": best["auc"],
        "best_cv_mean": best["cv_mean"],
    }


def plot_failure_trend(output_df, target_col):
    """Simulated rolling failure trend (since we have static data)."""
    if "Predicted_Failure" not in output_df.columns:
        return None
    chunk = max(1, len(output_df) // 50)
    trend = [output_df["Predicted_Failure"].iloc[i:i+chunk].mean() * 100
             for i in range(0, len(output_df), chunk)]
    x = list(range(len(trend)))

    fig, ax = _fig_setup((9, 3))
    ax.plot(x, trend, color=ACCENT, lw=2)
    ax.fill_between(x, trend, alpha=0.15, color=ACCENT)
    ax.axhline(np.mean(trend), color=RED, lw=1, linestyle="--", label=f"Avg {np.mean(trend):.1f}%")
    ax.set_title("Failure Rate Trend Across Dataset", color=ACCENT, fontsize=10)
    ax.set_xlabel("Batch Index", color=TEXT_DIM, fontsize=9)
    ax.set_ylabel("Failure Rate (%)", color=TEXT_DIM, fontsize=9)
    ax.legend(facecolor=CARD_BG, edgecolor=BORDER, labelcolor=TEXT_MAIN, fontsize=9)
    plt.tight_layout()
    return fig


def plot_model_comparison_radar(results):
    """Radar chart comparing models on 4 metrics."""
    labels = ["Accuracy", "F1", "AUC", "CV F1"]
    n = len(labels)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor(CARD_BG)
    ax.set_facecolor(CARD_BG)

    colors_list = [ACCENT, BLUE, GREEN, PURPLE, RED]
    for i, (name, res) in enumerate(results.items()):
        vals = [res["accuracy"], res["f1"], res["auc"], res["cv_mean"]]
        vals += vals[:1]
        color = colors_list[i % len(colors_list)]
        ax.plot(angles, vals, color=color, lw=2, label=name)
        ax.fill(angles, vals, alpha=0.08, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, color=TEXT_MAIN, fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["25%", "50%", "75%", "100%"], color=TEXT_DIM, fontsize=7)
    ax.tick_params(colors=TEXT_DIM)
    ax.spines["polar"].set_color(BORDER)
    ax.grid(color=BORDER, linestyle="--", linewidth=0.5)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1),
              facecolor=CARD_BG, edgecolor=BORDER, labelcolor=TEXT_MAIN, fontsize=8)
    ax.set_title("Model Comparison Radar", color=ACCENT, fontsize=10, pad=20)
    return fig


def plot_sensor_boxplots(data, feature_cols, target_col, n=6):
    """Box plots comparing sensor distributions by class."""
    cols = [c for c in feature_cols if data[c].dtype in [np.float64, np.float32, np.int64, np.int32]][:n]
    fig, axes = plt.subplots(1, len(cols), figsize=(14, 4))
    fig.patch.set_facecolor(CARD_BG)
    if len(cols) == 1:
        axes = [axes]

    for ax, col in zip(axes, cols):
        ax.set_facecolor(CARD_BG)
        groups = [data[data[target_col] == 0][col].dropna(),
                  data[data[target_col] == 1][col].dropna()]
        bp = ax.boxplot(groups, patch_artist=True, widths=0.5,
                        medianprops=dict(color=ACCENT, lw=2),
                        whiskerprops=dict(color=TEXT_DIM),
                        capprops=dict(color=TEXT_DIM),
                        flierprops=dict(marker=".", color=TEXT_DIM, markersize=3))
        bp["boxes"][0].set_facecolor(GREEN + "44")
        bp["boxes"][0].set_edgecolor(GREEN)
        if len(bp["boxes"]) > 1:
            bp["boxes"][1].set_facecolor(RED + "44")
            bp["boxes"][1].set_edgecolor(RED)
        ax.set_xticklabels(["Healthy", "Failure"], color=TEXT_DIM, fontsize=8)
        ax.set_title(col[:18], color=ACCENT, fontsize=8)
        for spine in ax.spines.values():
            spine.set_color(BORDER)
        ax.tick_params(colors=TEXT_DIM, labelsize=7)

    plt.suptitle("Sensor Boxplots — Healthy vs Failure", color=TEXT_MAIN, fontsize=10, y=1.02)
    plt.tight_layout()
    return fig


def plot_failure_heatmap_grid(output_df, feature_cols, n=2):
    """2D density heatmap for top 2 features vs failure."""
    if len(feature_cols) < 2 or "Predicted_Failure" not in output_df.columns:
        return None
    f1, f2 = feature_cols[0], feature_cols[1]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.patch.set_facecolor(CARD_BG)
    for ax, cls, color, label in [
        (axes[0], 0, GREEN, "Healthy"),
        (axes[1], 1, RED, "Failure"),
    ]:
        ax.set_facecolor(CARD_BG)
        subset = output_df[output_df["Predicted_Failure"] == cls]
        if len(subset) < 5:
            ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center",
                    color=TEXT_DIM, transform=ax.transAxes)
        else:
            try:
                ax.hist2d(subset[f1], subset[f2], bins=20,
                          cmap="YlOrRd" if cls == 1 else "YlGn")
            except Exception:
                ax.scatter(subset[f1], subset[f2], alpha=0.3, color=color, s=5)
        ax.set_title(f"{label} — {f1} vs {f2}", color=ACCENT, fontsize=9)
        ax.set_xlabel(f1, color=TEXT_DIM, fontsize=8)
        ax.set_ylabel(f2, color=TEXT_DIM, fontsize=8)
        for spine in ax.spines.values():
            spine.set_color(BORDER)
        ax.tick_params(colors=TEXT_DIM, labelsize=7)
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────
# CSV EXPORT — FIXED
# ─────────────────────────────────────────────

def build_output_df(data, X_full, col_map, model):
    """
    Attach predictions + reports to original dataframe.

    FIX 1: Pass DataFrame (not numpy array) to model.predict / predict_proba
            to avoid 'X does not have valid feature names' warning.
    FIX 2: Compute predictions row-by-row using DataFrame so RandomForest
            does not spin up parallel threads (avoids RuntimeError on shutdown).
    """
    # Batch predictions for speed (these are fine as DataFrame)
    preds = model.predict(X_full)
    probs = model.predict_proba(X_full)[:, 1]

    out = data.copy()
    out["Predicted_Failure"] = preds
    out["Failure_Probability"] = probs.round(4)

    # Short report — no model call needed, just sensor values
    out["Short_Report"] = X_full.apply(
        lambda r: generate_short_report(r, col_map), axis=1
    )

    # Long report — use pre-computed batch preds/probs (no per-row model call)
    # This avoids both the feature-name warning AND the thread crash
    def make_long_report(idx):
        r = X_full.iloc[idx]
        pred = int(preds[idx])
        prob = float(probs[idx])
        return generate_long_report(r, col_map, prediction=pred, probability=prob)

    out["Long_Report"] = [make_long_report(i) for i in range(len(X_full))]

    return out
