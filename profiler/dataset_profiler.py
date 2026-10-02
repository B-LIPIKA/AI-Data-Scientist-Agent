"""
Automated Dataset Profiler & Quality Inspection
"""

import pandas as pd
import numpy as np
from typing import Dict, Any


class DatasetProfiler:
    """Profiles tabular datasets to extract schema, null rates, and distribution summaries."""
    
    @staticmethod
    def profile_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
        total_rows, total_cols = df.shape
        duplicate_rows = int(df.duplicated().sum())

        columns_profile = {}
        for col in df.columns:
            s = df[col]
            missing_count = int(s.isna().sum())
            missing_pct = round(float((missing_count / total_rows) * 100), 2)
            dtype_str = str(s.dtype)

            col_info = {
                "dtype": dtype_str,
                "missing_count": missing_count,
                "missing_pct": missing_pct,
                "unique_values": int(s.nunique())
            }

            if np.issubdtype(s.dtype, np.number):
                col_info.update({
                    "mean": round(float(s.mean()), 3) if not s.isna().all() else None,
                    "std": round(float(s.std()), 3) if not s.isna().all() else None,
                    "min": round(float(s.min()), 3) if not s.isna().all() else None,
                    "max": round(float(s.max()), 3) if not s.isna().all() else None,
                    "skew": round(float(s.skew()), 3) if not s.isna().all() else None
                })

            columns_profile[col] = col_info

        return {
            "total_rows": total_rows,
            "total_columns": total_cols,
            "duplicate_count": duplicate_rows,
            "memory_usage_kb": round(float(df.memory_usage(deep=True).sum() / 1024), 2),
            "columns": columns_profile
        }
