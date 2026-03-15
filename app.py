import os
import json
import math
import numpy as np
import pandas as pd
from flask import Flask, render_template, jsonify, request
from collections import OrderedDict
import base64
import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

app = Flask(__name__)
TARGET_COLUMN_PREFERRED = "Target_Variable"

# ── Load data once at startup ──────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE, "data", "train.csv")
META_PATH = os.path.join(BASE, "data", "metadata.csv")

df = pd.read_csv(DATA_PATH)
# Drop unnamed index col if present
if "Unnamed: 0" in df.columns:
    df.drop(columns=["Unnamed: 0"], inplace=True)


def _resolve_target_column(columns):
    if TARGET_COLUMN_PREFERRED in columns:
        return TARGET_COLUMN_PREFERRED
    for col in columns:
        if str(col).strip().lower().startswith(TARGET_COLUMN_PREFERRED.lower()):
            return col
    return None


TARGET_COLUMN = _resolve_target_column(df.columns)

# Load metadata descriptions
meta_map = {}
try:
    meta_df = pd.read_csv(META_PATH)
    for _, row in meta_df.iterrows():
        meta_map[str(row.iloc[0]).strip()] = str(row.iloc[1]).strip()
except Exception:
    pass

# ── Helpers ────────────────────────────────────────────────────────────────

def _fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight",
                facecolor="#ffffff", edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _classify_feature(series):
    """Classify a column as categorical, numerical-continuous, numerical-discrete, boolean, or identifier."""
    if series.dtype == "bool" or series.nunique() == 2:
        return "Boolean"
    if pd.api.types.is_numeric_dtype(series):
        if series.nunique() < 30:
            return "Numerical (Discrete)"
        return "Numerical (Continuous)"
    if pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series):
        if series.nunique() > 0.5 * len(series):
            return "Identifier / High-Cardinality Text"
        return "Categorical"
    return "Other"


def _order_of_magnitude_bins(series):
    """Bucket numeric values into orders of magnitude."""
    s = series.dropna()
    if len(s) == 0:
        return {}
    s_abs = s.abs().replace(0, np.nan).dropna()
    if len(s_abs) == 0:
        return {"0": int((s == 0).sum())}
    min_exp = int(np.floor(np.log10(s_abs.min())))
    max_exp = int(np.floor(np.log10(s_abs.max())))
    bins = OrderedDict()
    zero_count = int((s == 0).sum())
    if zero_count > 0:
        bins["0"] = zero_count
    neg_count = int((s < 0).sum())
    if neg_count > 0:
        bins["< 0 (negative)"] = neg_count
    for exp in range(min_exp, max_exp + 1):
        lo, hi = 10 ** exp, 10 ** (exp + 1)
        label = f"10^{exp} – 10^{exp+1}"
        count = int(((s_abs >= lo) & (s_abs < hi)).sum())
        if count > 0:
            bins[label] = count
    # catch overflow at top
    hi = 10 ** (max_exp + 1)
    top = int((s_abs >= hi).sum())
    if top > 0:
        bins[f"≥ 10^{max_exp+1}"] = top
    return bins


def _make_frequency_chart(series, col_name, ftype):
    """Return a base64-encoded frequency/histogram chart."""
    fig, ax = plt.subplots(figsize=(7, 3.5))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")

    if "Numerical" in ftype or ftype == "Boolean":
        s = series.dropna()
        if len(s) == 0:
            plt.close(fig)
            return None
        if ftype == "Numerical (Discrete)" or ftype == "Boolean":
            vc = s.value_counts().sort_index()
            ax.bar(vc.index.astype(str), vc.values, color="#222222", edgecolor="#000000", linewidth=0.5)
            ax.set_xlabel(col_name, fontsize=9)
        else:
            ax.hist(s, bins=min(50, max(10, int(np.sqrt(len(s))))),
                    color="#222222", edgecolor="#000000", linewidth=0.5)
            ax.set_xlabel(col_name, fontsize=9)
    else:
        vc = series.value_counts().head(30)
        labels = [str(l)[:28] for l in vc.index]
        ax.barh(labels[::-1], vc.values[::-1], color="#222222", edgecolor="#000000", linewidth=0.5)
        ax.set_xlabel("Count", fontsize=9)

    ax.set_ylabel("Frequency" if "Numerical" in ftype else "", fontsize=9)
    ax.tick_params(labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#000000")
    ax.set_title(f"Distribution – {col_name}", fontsize=10, fontweight="bold")
    return _fig_to_b64(fig)


def _make_boxplot(series, col_name):
    """Box-plot for numeric columns."""
    s = series.dropna()
    if len(s) == 0:
        return None
    fig, ax = plt.subplots(figsize=(7, 2.5))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")
    bp = ax.boxplot(s, vert=False, patch_artist=True,
                    boxprops=dict(facecolor="#cccccc", edgecolor="#000"),
                    whiskerprops=dict(color="#000"),
                    capprops=dict(color="#000"),
                    medianprops=dict(color="#000", linewidth=2),
                    flierprops=dict(marker="o", markerfacecolor="#888", markersize=3, linestyle="none"))
    ax.set_xlabel(col_name, fontsize=9)
    ax.set_title(f"Box Plot – {col_name}", fontsize=10, fontweight="bold")
    ax.tick_params(labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#000000")
    return _fig_to_b64(fig)


def _make_magnitude_chart(mag_bins, col_name):
    """Bar chart of order-of-magnitude distribution."""
    if not mag_bins:
        return None
    fig, ax = plt.subplots(figsize=(7, 3))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")
    labels = list(mag_bins.keys())
    values = list(mag_bins.values())
    ax.bar(labels, values, color="#222222", edgecolor="#000000", linewidth=0.5)
    ax.set_xlabel("Magnitude Bucket", fontsize=9)
    ax.set_ylabel("Count", fontsize=9)
    ax.set_title(f"Order of Magnitude – {col_name}", fontsize=10, fontweight="bold")
    ax.tick_params(labelsize=7, axis="x", rotation=30)
    for spine in ax.spines.values():
        spine.set_color("#000000")
    return _fig_to_b64(fig)


def _make_target_relationship_chart(feature_series, target_series, feature_name, target_name):
    """Type-aware chart to visualize relationship between a feature and target."""
    pair_df = pd.DataFrame({"feature": feature_series, "target": target_series}).dropna()
    if pair_df.empty:
        return None

    feat_is_num = pd.api.types.is_numeric_dtype(pair_df["feature"])
    target_is_num = pd.api.types.is_numeric_dtype(pair_df["target"])

    fig, ax = plt.subplots(figsize=(7, 4))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")

    if feat_is_num and target_is_num:
        # Clip extreme tails to improve readability while preserving the core relationship.
        x = pair_df["feature"]
        y = pair_df["target"]
        x_lo, x_hi = x.quantile([0.01, 0.99])
        y_lo, y_hi = y.quantile([0.01, 0.99])
        clipped = pair_df[
            pair_df["feature"].between(x_lo, x_hi)
            & pair_df["target"].between(y_lo, y_hi)
        ]
        if len(clipped) < 100:
            clipped = pair_df

        hb = ax.hexbin(
            clipped["feature"],
            clipped["target"],
            gridsize=42,
            cmap="Greys",
            mincnt=1,
            linewidths=0,
        )
        cbar = fig.colorbar(hb, ax=ax, fraction=0.03, pad=0.02)
        cbar.ax.tick_params(labelsize=7)
        cbar.set_label("Point Density", fontsize=8)

        # Add a binned mean trend line to clearly show whether target rises or falls.
        n_quantiles = int(min(12, max(4, clipped["feature"].nunique())))
        if n_quantiles >= 2:
            bin_ids = pd.qcut(clipped["feature"], q=n_quantiles, labels=False, duplicates="drop")
            trend_df = clipped.assign(bin_id=bin_ids).dropna(subset=["bin_id"])
            binned = (
                trend_df.groupby("bin_id", as_index=False)
                .agg(feature_mean=("feature", "mean"), target_mean=("target", "mean"))
                .sort_values("feature_mean")
            )
            if len(binned) > 1:
                ax.plot(
                    binned["feature_mean"],
                    binned["target_mean"],
                    color="#0b4f8a",
                    linewidth=2.4,
                    marker="o",
                    markersize=3.5,
                    label="Binned mean trend",
                )

        # Overlay a simple linear trend line for overall direction.
        if len(clipped) > 1 and clipped["feature"].std() > 0:
            slope, intercept = np.polyfit(clipped["feature"], clipped["target"], 1)
            xs = np.linspace(clipped["feature"].min(), clipped["feature"].max(), 100)
            ys = slope * xs + intercept
            ax.plot(xs, ys, color="#b30000", linewidth=1.6, linestyle="--", label="Linear fit")

        corr = clipped["feature"].corr(clipped["target"]) if len(clipped) > 1 else np.nan
        corr_text = f"Pearson r = {corr:.3f}" if pd.notna(corr) else "Pearson r = N/A"
        if pd.isna(corr):
            trend_text = "Trend: N/A"
        elif corr > 0.05:
            trend_text = "Trend: Upward (feature up -> target up)"
        elif corr < -0.05:
            trend_text = "Trend: Downward (feature up -> target down)"
        else:
            trend_text = "Trend: Weak / Flat"
        ax.text(
            0.02,
            0.98,
            f"{corr_text}\n{trend_text}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            bbox=dict(facecolor="#ffffff", edgecolor="#000000", boxstyle="round,pad=0.2"),
        )
        ax.legend(loc="lower right", fontsize=7, frameon=True, edgecolor="#000000")

        ax.set_xlabel(feature_name, fontsize=9)
        ax.set_ylabel(target_name, fontsize=9)
        ax.set_title(f"Correlation Plot: {feature_name} vs {target_name}", fontsize=10, fontweight="bold")
    elif feat_is_num and not target_is_num:
        top_targets = pair_df["target"].value_counts().head(20).index
        plot_df = pair_df[pair_df["target"].isin(top_targets)]
        grouped = [plot_df[plot_df["target"] == cat]["feature"].values for cat in top_targets]
        ax.boxplot(grouped, labels=[str(c)[:20] for c in top_targets], patch_artist=True,
                   boxprops=dict(facecolor="#cccccc", edgecolor="#000"),
                   whiskerprops=dict(color="#000"),
                   capprops=dict(color="#000"),
                   medianprops=dict(color="#000", linewidth=2),
                   flierprops=dict(marker="o", markerfacecolor="#888", markersize=3, linestyle="none"))
        ax.set_xlabel(target_name, fontsize=9)
        ax.set_ylabel(feature_name, fontsize=9)
        ax.set_title(f"{feature_name} by {target_name}", fontsize=10, fontweight="bold")
        ax.tick_params(axis="x", rotation=25)
    elif not feat_is_num and target_is_num:
        top_features = pair_df["feature"].value_counts().head(20).index
        plot_df = pair_df[pair_df["feature"].isin(top_features)]
        grouped = [plot_df[plot_df["feature"] == cat]["target"].values for cat in top_features]
        ax.boxplot(grouped, labels=[str(c)[:20] for c in top_features], patch_artist=True,
                   boxprops=dict(facecolor="#cccccc", edgecolor="#000"),
                   whiskerprops=dict(color="#000"),
                   capprops=dict(color="#000"),
                   medianprops=dict(color="#000", linewidth=2),
                   flierprops=dict(marker="o", markerfacecolor="#888", markersize=3, linestyle="none"))
        ax.set_xlabel(feature_name, fontsize=9)
        ax.set_ylabel(target_name, fontsize=9)
        ax.set_title(f"{target_name} by {feature_name}", fontsize=10, fontweight="bold")
        ax.tick_params(axis="x", rotation=25)
    else:
        top_features = pair_df["feature"].value_counts().head(15).index
        top_targets = pair_df["target"].value_counts().head(15).index
        mask = pair_df["feature"].isin(top_features) & pair_df["target"].isin(top_targets)
        plot_df = pair_df[mask]
        ctab = pd.crosstab(
            plot_df["feature"],
            plot_df["target"]
        )
        if ctab.empty:
            plt.close(fig)
            return None
        im = ax.imshow(ctab.values, cmap="Greys", aspect="auto")
        ax.set_xticks(range(len(ctab.columns)))
        ax.set_yticks(range(len(ctab.index)))
        ax.set_xticklabels([str(c)[:16] for c in ctab.columns], rotation=35, ha="right", fontsize=7)
        ax.set_yticklabels([str(c)[:16] for c in ctab.index], fontsize=7)
        ax.set_xlabel(target_name, fontsize=9)
        ax.set_ylabel(feature_name, fontsize=9)
        ax.set_title(f"{feature_name} vs {target_name} (count heatmap)", fontsize=10, fontweight="bold")
        fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)

    ax.tick_params(labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#000000")
    return _fig_to_b64(fig)


def _safe(val):
    """Make a value JSON-safe."""
    if val is None or (isinstance(val, float) and (math.isnan(val) or math.isinf(val))):
        return None
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return float(val)
    return val


# ── Routes ─────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/features")
def api_features():
    features = []
    for col in df.columns:
        ftype = _classify_feature(df[col])
        features.append({
            "name": col,
            "type": ftype,
            "description": meta_map.get(col, ""),
        })
    return jsonify(features)


@app.route("/api/feature/<path:col_name>")
def api_feature_detail(col_name):
    if col_name not in df.columns:
        return jsonify({"error": "Column not found"}), 404

    series = df[col_name]
    ftype = _classify_feature(series)
    total = len(series)
    null_count = int(series.isnull().sum())
    non_null = total - null_count
    n_unique = int(series.nunique())

    result = {
        "name": col_name,
        "description": meta_map.get(col_name, "No description available."),
        "dtype": str(series.dtype),
        "feature_type": ftype,
        "total_rows": total,
        "non_null_count": non_null,
        "null_count": null_count,
        "null_percentage": round(null_count / total * 100, 2) if total else 0,
        "unique_values": n_unique,
        "unique_percentage": round(n_unique / total * 100, 2) if total else 0,
        "target_column": TARGET_COLUMN if TARGET_COLUMN else TARGET_COLUMN_PREFERRED,
        "target_exists": bool(TARGET_COLUMN),
    }

    # Value counts (top 50)
    vc = series.value_counts()
    top_vc = vc.head(50)
    result["value_counts"] = {str(k): int(v) for k, v in top_vc.items()}
    result["value_counts_total_shown"] = len(top_vc)
    result["value_counts_total"] = len(vc)

    # Numeric stats
    if pd.api.types.is_numeric_dtype(series):
        s = series.dropna()
        result["min"] = _safe(s.min()) if len(s) else None
        result["max"] = _safe(s.max()) if len(s) else None
        result["mean"] = _safe(s.mean()) if len(s) else None
        result["median"] = _safe(s.median()) if len(s) else None
        result["mode"] = _safe(s.mode().iloc[0]) if len(s.mode()) else None
        result["std_dev"] = _safe(s.std()) if len(s) else None
        result["variance"] = _safe(s.var()) if len(s) else None
        result["sum"] = _safe(s.sum()) if len(s) else None
        result["range"] = _safe(s.max() - s.min()) if len(s) else None
        result["skewness"] = _safe(s.skew()) if len(s) else None
        result["kurtosis"] = _safe(s.kurtosis()) if len(s) else None
        result["coeff_of_variation"] = _safe(s.std() / s.mean() * 100) if len(s) and s.mean() != 0 else None

        # Percentiles
        for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
            result[f"p{p}"] = _safe(np.percentile(s, p)) if len(s) else None

        # IQR & outlier bounds
        q1 = np.percentile(s, 25) if len(s) else 0
        q3 = np.percentile(s, 75) if len(s) else 0
        iqr = q3 - q1
        result["iqr"] = _safe(iqr)
        result["outlier_lower_bound"] = _safe(q1 - 1.5 * iqr)
        result["outlier_upper_bound"] = _safe(q3 + 1.5 * iqr)
        outlier_mask = (s < (q1 - 1.5 * iqr)) | (s > (q3 + 1.5 * iqr))
        result["outlier_count"] = int(outlier_mask.sum())
        result["outlier_percentage"] = round(outlier_mask.sum() / len(s) * 100, 2) if len(s) else 0

        # Zero count
        result["zero_count"] = int((s == 0).sum())
        result["zero_percentage"] = round((s == 0).sum() / len(s) * 100, 2) if len(s) else 0

        # Negative count
        result["negative_count"] = int((s < 0).sum())
        result["positive_count"] = int((s > 0).sum())

        # Order of magnitude distribution
        mag_bins = _order_of_magnitude_bins(s)
        result["magnitude_distribution"] = mag_bins

        # Charts
        result["chart_frequency"] = _make_frequency_chart(series, col_name, ftype)
        result["chart_boxplot"] = _make_boxplot(series, col_name)
        result["chart_magnitude"] = _make_magnitude_chart(mag_bins, col_name)
    else:
        # Categorical stats
        result["mode"] = _safe(series.mode().iloc[0]) if len(series.mode()) else None
        result["chart_frequency"] = _make_frequency_chart(series, col_name, ftype)
        result["chart_boxplot"] = None
        result["chart_magnitude"] = None

    # Entropy (for categorical / low-cardinality)
    if n_unique > 0 and n_unique < 500:
        probs = vc.values / vc.values.sum()
        result["entropy"] = _safe(float(-np.sum(probs * np.log2(probs + 1e-12))))
    else:
        result["entropy"] = None

    # Feature vs target relationship chart
    if TARGET_COLUMN and col_name != TARGET_COLUMN:
        target_series = df[TARGET_COLUMN]
        result["chart_target_relationship"] = _make_target_relationship_chart(
            series,
            target_series,
            col_name,
            TARGET_COLUMN,
        )
        if pd.api.types.is_numeric_dtype(series) and pd.api.types.is_numeric_dtype(target_series):
            valid_pair = pd.DataFrame({"x": series, "y": target_series}).dropna()
            corr_val = valid_pair["x"].corr(valid_pair["y"]) if len(valid_pair) > 1 else None
            result["pearson_corr_with_target"] = _safe(corr_val) if corr_val is not None else None
            if corr_val is None or np.isnan(corr_val):
                result["trend_with_target"] = "N/A"
                result["trend_note"] = "Insufficient numeric data to estimate trend."
            elif corr_val > 0.05:
                result["trend_with_target"] = "Upward"
                result["trend_note"] = "As feature increases, target tends to increase."
            elif corr_val < -0.05:
                result["trend_with_target"] = "Downward"
                result["trend_note"] = "As feature increases, target tends to decrease."
            else:
                result["trend_with_target"] = "Weak / Flat"
                result["trend_note"] = "No strong up/down trend detected."
        else:
            result["pearson_corr_with_target"] = None
            result["trend_with_target"] = None
            result["trend_note"] = None
    elif TARGET_COLUMN and col_name == TARGET_COLUMN:
        result["chart_target_relationship"] = None
        result["pearson_corr_with_target"] = None
        result["trend_with_target"] = None
        result["trend_note"] = None
        result["target_relationship_note"] = "Selected feature is the target itself."
    else:
        result["chart_target_relationship"] = None
        result["pearson_corr_with_target"] = None
        result["trend_with_target"] = None
        result["trend_note"] = None
        result["target_relationship_note"] = f"Target column '{TARGET_COLUMN}' not found in dataset."

    # Sample values
    result["sample_values"] = [str(v) for v in series.dropna().sample(min(10, non_null), random_state=42).tolist()]

    return jsonify(result)


@app.route("/api/overview")
def api_overview():
    """Quick summary stats for the whole dataset."""
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    return jsonify({
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "numeric_columns": len(num_cols),
        "categorical_columns": len(cat_cols),
        "total_missing": int(df.isnull().sum().sum()),
        "total_cells": int(df.size),
        "missing_percentage": round(df.isnull().sum().sum() / df.size * 100, 2),
        "memory_usage_mb": round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2),
        "duplicate_rows": int(df.duplicated().sum()),
    })


if __name__ == "__main__":
    app.run(debug=True, port=5050, host="0.0.0.0")
