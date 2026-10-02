"""
CLI Runner for AI Data Scientist Agent
Usage:
  python cli.py
"""

import numpy as np
import pandas as pd
from agent.eda_agent import AIDataScientistAgent


def generate_sample_industrial_dataframe() -> pd.DataFrame:
    """Generates a sample industrial dataset for instant demonstration."""
    np.random.seed(42)
    n = 200
    temp = np.linspace(50, 75, n) + np.random.normal(0, 2, n)
    vibration = 2.0 + 0.05 * temp + np.random.normal(0, 0.4, n)
    pressure = 100.0 - 0.2 * temp + np.random.normal(0, 3, n)
    
    # Introduce some outliers
    vibration[15] = 9.8
    vibration[78] = 12.1

    return pd.DataFrame({
        "temperature_c": np.round(temp, 2),
        "vibration_rms": np.round(vibration, 2),
        "pressure_psi": np.round(pressure, 2),
        "status": ["NORMAL"] * (n - 2) + ["ANOMALY", "ANOMALY"]
    })


def main():
    print("🚀 Initializing AI Data Scientist Agent...")
    df = generate_sample_industrial_dataframe()
    agent = AIDataScientistAgent()
    result = agent.analyze_dataset(df, dataset_name="Industrial_Turbine_Sensors.csv")

    print("\n" + "="*70)
    print(result["markdown_report"])
    print("="*70)
    print("\n✅ EDA & Discovery Analysis Complete.")


if __name__ == "__main__":
    main()
