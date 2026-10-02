"""
Deterministic Statistical Tools for Automated Exploratory Data Analysis
Provides mathematical calculations for trend, stationarity, changepoints, anomalies, and correlations.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List


def evaluate_trend(series: pd.Series) -> Dict[str, Any]:
    """Computes deterministic linear slope and trend significance."""
    clean = series.dropna().values
    if len(clean) < 3:
        return {"status": "insufficient_data"}
    x = np.arange(len(clean))
    slope, intercept = np.polyfit(x, clean, 1)

    std_val = np.std(clean) + 1e-6
    normalized_slope = slope / std_val

    direction = "STABLE"
    if normalized_slope > 0.02:
        direction = "UPWARD TREND"
    elif normalized_slope < -0.02:
        direction = "DOWNWARD TREND"

    return {
        "slope": round(float(slope), 5),
        "normalized_slope": round(float(normalized_slope), 4),
        "direction": direction,
        "first_value": round(float(clean[0]), 3),
        "last_value": round(float(clean[-1]), 3)
    }


def evaluate_stationarity(series: pd.Series) -> Dict[str, Any]:
    """Evaluates stationarity via rolling mean variance and variance stability ratio."""
    clean = series.dropna()
    if len(clean) < 20:
        return {"status": "insufficient_data"}

    # Split into two halves to test variance shift
    mid = len(clean) // 2
    var1 = np.var(clean.iloc[:mid])
    var2 = np.var(clean.iloc[mid:])
    ratio = float(var2 / (var1 + 1e-6))

    is_stationary = bool(0.6 <= ratio <= 1.6)
    return {
        "variance_half1": round(float(var1), 3),
        "variance_half2": round(float(var2), 3),
        "variance_ratio": round(ratio, 3),
        "is_stationary": is_stationary,
        "interpretation": "Series demonstrates variance stationarity" if is_stationary else "Non-stationary variance shift detected"
    }


def detect_changepoints(series: pd.Series, threshold_z: float = 2.5) -> Dict[str, Any]:
    """Detects mean changepoints using cumulative sum deviation (CUSUM proxy)."""
    clean = series.dropna().values
    if len(clean) < 10:
        return {"changepoints_detected": 0, "indices": []}

    diffs = np.diff(clean)
    std_diff = np.std(diffs) + 1e-6
    z_scores = np.abs(diffs - np.mean(diffs)) / std_diff

    jump_indices = np.where(z_scores > threshold_z)[0].tolist()
    return {
        "changepoints_detected": len(jump_indices),
        "indices": jump_indices[:5],  # top 5
        "max_z_jump": round(float(np.max(z_scores)), 2) if len(z_scores) else 0.0
    }


def compute_correlation_matrix(df: pd.DataFrame, threshold: float = 0.70) -> Dict[str, Any]:
    """Calculates Pearson correlation matrix and identifies high collinearity pairs."""
    num_df = df.select_dtypes(include=[np.number])
    if num_df.shape[1] < 2:
        return {"highly_correlated_pairs": []}

    corr = num_df.corr()
    high_pairs = []

    columns = corr.columns
    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            val = corr.iloc[i, j]
            if abs(val) >= threshold:
                high_pairs.append({
                    "feature_1": columns[i],
                    "feature_2": columns[j],
                    "correlation": round(float(val), 3)
                })

    return {
        "highly_correlated_pairs": high_pairs,
        "matrix_features": list(columns)
    }


def detect_anomalies_iqr(series: pd.Series) -> Dict[str, Any]:
    """Detects outlier anomalies using Interquartile Range (IQR)."""
    clean = series.dropna()
    q25, q75 = np.percentile(clean, [25, 75])
    iqr = q75 - q25
    lower_bound = q25 - (1.5 * iqr)
    upper_bound = q75 + (1.5 * iqr)

    outliers = clean[(clean < lower_bound) | (clean > upper_bound)]
    return {
        "outlier_count": int(len(outliers)),
        "outlier_percentage": round(float((len(outliers) / len(clean)) * 100), 2),
        "lower_bound": round(float(lower_bound), 3),
        "upper_bound": round(float(upper_bound), 3)
    }
