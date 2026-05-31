"""Core quality factor screening workflow."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from cpo_phosphorus.workflows.common import numeric_series, read_table, write_dataframe

CORE_QUALITY_FIELDS = [
    "feed_ffa_pct",
    "feed_mi_pct",
    "feed_iv",
    "feed_dobi",
    "feed_car_pv",
]


def _safe_corr(x, y, method):
    data = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(data) < 3 or data["x"].nunique() < 2 or data["y"].nunique() < 2:
        return None, None
    if method == "pearson":
        corr, p_value = stats.pearsonr(data["x"], data["y"])
    else:
        corr, p_value = stats.spearmanr(data["x"], data["y"])
    return float(corr), float(p_value)


def _single_feature_model_r2(x, y):
    data = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(data) < 20 or data["x"].nunique() < 2:
        return None
    train, test = train_test_split(data, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(train[["x"]], train["y"])
    pred = model.predict(test[["x"]])
    return float(r2_score(test["y"], pred))


def _classify_factor(spearman_corr, spearman_p, single_r2):
    abs_corr = abs(spearman_corr) if spearman_corr is not None else 0.0
    r2 = single_r2 if single_r2 is not None else 0.0
    if spearman_p is not None and spearman_p < 0.05 and (abs_corr >= 0.3 or r2 >= 0.1):
        return "direct_supported"
    if spearman_p is not None and spearman_p < 0.1 and (abs_corr >= 0.15 or r2 >= 0.03):
        return "weak_or_context_dependent"
    return "not_supported_current_data"


def _year_stability(df, field, target_col):
    if "date" not in df.columns:
        return []
    data = df.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["year"] = data["date"].dt.year
    rows = []
    for year, group in data.groupby("year"):
        if pd.isna(year):
            continue
        corr, p_value = _safe_corr(numeric_series(group, field), numeric_series(group, target_col), "spearman")
        rows.append(
            {
                "year": int(year),
                "spearman_corr": corr,
                "spearman_p_value": p_value,
                "n": int(group[[field, target_col]].dropna().shape[0]),
            }
        )
    return rows


def run_core_factor_check(input_path, target_col="feed_p_ppm", output_dir=None):
    """Screen core quality fields against the selected target."""
    df = read_table(input_path)
    columns = set(df.columns)
    available_core = [col for col in CORE_QUALITY_FIELDS if col in columns]
    missing_core = [col for col in CORE_QUALITY_FIELDS if col not in columns]
    target_available = target_col in columns
    valid_target_rows = int(df[target_col].notna().sum()) if target_available else 0

    rows = []
    if target_available:
        y = numeric_series(df, target_col)
        for field in available_core:
            x = numeric_series(df, field)
            pearson_corr, pearson_p = _safe_corr(x, y, "pearson")
            spearman_corr, spearman_p = _safe_corr(x, y, "spearman")
            single_r2 = _single_feature_model_r2(x, y)
            rows.append(
                {
                    "factor": field,
                    "n": int(pd.DataFrame({"x": x, "y": y}).dropna().shape[0]),
                    "pearson_corr": pearson_corr,
                    "pearson_p_value": pearson_p,
                    "spearman_corr": spearman_corr,
                    "spearman_p_value": spearman_p,
                    "single_feature_test_r2": single_r2,
                    "evidence_level": _classify_factor(spearman_corr, spearman_p, single_r2),
                    "year_stability": _year_stability(df, field, target_col),
                }
            )

    result_df = pd.DataFrame(rows)
    output_files = {}
    if output_dir:
        output_files["core_factor_screening"] = write_dataframe(
            result_df.drop(columns=["year_stability"], errors="ignore"),
            output_dir,
            "core_factor_screening.csv",
        )

    supported_count = int(result_df["evidence_level"].eq("direct_supported").sum()) if not result_df.empty else 0
    weak_count = (
        int(result_df["evidence_level"].eq("weak_or_context_dependent").sum())
        if not result_df.empty
        else 0
    )
    status = "screened" if target_available and available_core else "not_ready"
    return {
        "workflow": "core_factors",
        "status": status,
        "milestone": 4,
        "input_path": str(input_path),
        "target_col": target_col,
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "target_available": target_available,
        "valid_target_rows": valid_target_rows,
        "available_core_fields": available_core,
        "missing_core_fields": missing_core,
        "supported_factor_count": supported_count,
        "weak_factor_count": weak_count,
        "screening": rows,
        "output_files": output_files,
        "notes": [
            "Core factor screening reports association and single-feature predictive signal.",
            "Association evidence is not causal evidence and does not authorize automatic control.",
        ],
    }
