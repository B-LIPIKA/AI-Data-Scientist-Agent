"""
Unit Tests for AI Data Scientist Agent
"""

import pytest
import pandas as pd
import numpy as np

from profiler.dataset_profiler import DatasetProfiler
from profiler.statistical_tools import (
    evaluate_trend,
    evaluate_stationarity,
    detect_changepoints,
    compute_correlation_matrix,
    detect_anomalies_iqr
)
from agent.eda_agent import AIDataScientistAgent


@pytest.fixture
def sample_df():
    np.random.seed(42)
    n = 100
    x = np.linspace(10, 50, n)
    y = x * 2 + np.random.normal(0, 1, n)
    # Inject outlier
    y[10] = 300.0
    return pd.DataFrame({"feature_x": x, "feature_y": y})


def test_dataset_profiler(sample_df):
    profile = DatasetProfiler.profile_dataframe(sample_df)
    assert profile["total_rows"] == 100
    assert profile["total_columns"] == 2
    assert "feature_x" in profile["columns"]
    assert profile["columns"]["feature_x"]["missing_count"] == 0


def test_statistical_tools(sample_df):
    trend = evaluate_trend(sample_df["feature_x"])
    assert "UPWARD TREND" in trend["direction"]
    assert trend["slope"] > 0

    anomalies = detect_anomalies_iqr(sample_df["feature_y"])
    assert anomalies["outlier_count"] >= 1

    corr = compute_correlation_matrix(sample_df)
    assert len(corr["highly_correlated_pairs"]) >= 1


def test_agent_workflow(sample_df):
    agent = AIDataScientistAgent()
    result = agent.analyze_dataset(sample_df, dataset_name="TestDataset")
    assert "markdown_report" in result
    assert "recommendations" in result
    assert len(result["recommendations"]) > 0
    assert "# 🔬 Automated EDA & ML Discovery Report" in result["markdown_report"]
