"""
AI Data Scientist Agent (Evidence-Driven Orchestration)
Runs automated profiling, calls statistical tools, and synthesizes ML discovery recommendations.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any

from profiler.dataset_profiler import DatasetProfiler
from profiler.statistical_tools import (
    evaluate_trend,
    evaluate_stationarity,
    detect_changepoints,
    compute_correlation_matrix,
    detect_anomalies_iqr
)


class AIDataScientistAgent:
    """
    Automated EDA & ML Discovery Agent.
    Executes deterministic statistical profiling and generates actionable modeling guidelines.
    """

    def analyze_dataset(self, df: pd.DataFrame, dataset_name: str = "Dataset") -> Dict[str, Any]:
        """Runs full automated profiling & statistical discovery workflow."""
        # 1. Base Profiling
        base_profile = DatasetProfiler.profile_dataframe(df)

        # 2. Statistical Analysis on Numeric Features
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_analytics = {}

        for col in numeric_cols:
            s = df[col]
            feature_analytics[col] = {
                "trend": evaluate_trend(s),
                "stationarity": evaluate_stationarity(s),
                "changepoints": detect_changepoints(s),
                "outliers": detect_anomalies_iqr(s)
            }

        # 3. Multicollinearity Analysis
        correlations = compute_correlation_matrix(df)

        # 4. Generate Machine Learning Recommendations
        recommendations = self._generate_ml_recommendations(base_profile, feature_analytics, correlations)

        # 5. Synthesize Executive Markdown Report
        markdown_report = self._generate_markdown_report(dataset_name, base_profile, feature_analytics, correlations, recommendations)

        return {
            "dataset_name": dataset_name,
            "profile": base_profile,
            "feature_analytics": feature_analytics,
            "correlations": correlations,
            "recommendations": recommendations,
            "markdown_report": markdown_report
        }

    def _generate_ml_recommendations(
        self,
        profile: Dict[str, Any],
        analytics: Dict[str, Any],
        correlations: Dict[str, Any]
    ) -> list:
        recs = []
        # Missing values check
        high_nulls = [c for c, d in profile["columns"].items() if d["missing_pct"] > 20.0]
        if high_nulls:
            recs.append(f"Imputation / Dropping: Features {high_nulls} have >20% missing values; consider iterative imputer or column pruning.")

        # Outlier severity check
        high_outliers = [c for c, a in analytics.items() if a["outliers"].get("outlier_percentage", 0) > 5.0]
        if high_outliers:
            recs.append(f"Robust Scaling: Features {high_outliers} exhibit significant outliers (>5%); recommend RobustScaler or Isolation Forest filtering.")

        # Multicollinearity check
        high_corr = correlations.get("highly_correlated_pairs", [])
        if high_corr:
            recs.append(f"Feature Selection: Detected {len(high_corr)} highly correlated pairs (e.g. {high_corr[0]['feature_1']} & {high_corr[0]['feature_2']}); apply VIF filtering or PCA.")

        # Modeling Family Recommendation
        recs.append("Model Exploration: Tree-based ensembles (LightGBM/XGBoost) recommended due to non-linear feature distributions.")
        return recs

    def _generate_markdown_report(
        self,
        name: str,
        profile: Dict[str, Any],
        analytics: Dict[str, Any],
        correlations: Dict[str, Any],
        recommendations: list
    ) -> str:
        report = [
            f"# 🔬 Automated EDA & ML Discovery Report: {name}",
            "",
            "## 1. Dataset Dimensions & Health",
            f"- **Rows:** {profile['total_rows']:,}",
            f"- **Columns:** {profile['total_columns']}",
            f"- **Duplicate Rows:** {profile['duplicate_count']}",
            f"- **Memory Footprint:** {profile['memory_usage_kb']:.2f} KB",
            "",
            "## 2. Statistical & Anomaly Discovery",
            "| Feature | Trend | Stationarity | Outlier % | Changepoints |",
            "| :--- | :---: | :---: | :---: | :---: |"
        ]

        for col, a in analytics.items():
            trend_dir = a["trend"].get("direction", "N/A")
            stat_status = "Stationary" if a["stationarity"].get("is_stationary") else "Non-Stationary"
            outlier_pct = a["outliers"].get("outlier_percentage", 0)
            cps = a["changepoints"].get("changepoints_detected", 0)
            report.append(f"| `{col}` | {trend_dir} | {stat_status} | {outlier_pct}% | {cps} |")

        report.append("")
        report.append("## 3. High Correlation Warnings (|r| >= 0.70)")
        pairs = correlations.get("highly_correlated_pairs", [])
        if pairs:
            for p in pairs:
                report.append(f"- `{p['feature_1']}` ↔ `{p['feature_2']}`: **r = {p['correlation']}**")
        else:
            report.append("- No critical multicollinearity issues found.")

        report.append("")
        report.append("## 4. Evidence-Grounded ML Recommendations")
        for r in recommendations:
            report.append(f"- {r}")

        return "\n".join(report)
