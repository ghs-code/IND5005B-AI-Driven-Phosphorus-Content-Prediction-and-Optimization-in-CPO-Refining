#!/usr/bin/env python3
"""Validate current-table proxy factors for feed-oil phosphorus.

This current phase has only the 2024/2025 quality tables available.  The
pipeline therefore tests observed proxy groups and process-response fields
without trying to ingest enterprise/source/weather/soil/lab/process sidecars.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from cpo_phosphorus.paths import (
    LOCAL_FACTOR_VALIDATION_REPORTS_DIR,
    LOCAL_PROCESSED_DATA_DIR,
)
from cpo_phosphorus.pipelines.sklearn_preprocessing import (
    build_leakage_safe_preprocessor,
)


RANDOM_STATE = 42
TEST_SIZE = 0.2
DEFAULT_TARGET_COL = "feed_p_ppm"
DATE_COL = "date"
RISK_QUANTILE = 0.80
MIN_TEST_SAMPLES = 20
MIN_RMSE_LIFT = 0.01
OUT_OF_TIME_VALIDATIONS = ("blocked_time_holdout", "year_holdout", "monthly_rolling")

QUALITY_FEATURES = [
    "feed_ffa_pct",
    "feed_mi_pct",
    "feed_iv",
    "feed_dobi",
    "feed_car_pv",
]
HISTORY_STATE_FEATURES = ["feed_p_ppm_lag1", "feed_p_ppm_roll3"]
TIME_CONTEXT_FEATURES = ["time_trend"]
BASE_PROXY_CATEGORICAL = ["source_file", "feed_tank", "feed_type"]
PROCESS_RESPONSE_FEATURES = [
    "acid_dosing_pct",
    "bleaching_earth_dosing_pct",
    "dosage_total_pct",
    "acid_x_bleaching",
]

PREDICTION_EXCLUDED_PREFIXES = ("rbd_",)
PREDICTION_EXCLUDED_COLUMNS = {
    "feed_p_ppm",
    "log_feed_p_ppm",
    "rbd_p_ppm",
    "p_removal_delta",
    *PROCESS_RESPONSE_FEATURES,
}

REPORT_COLUMNS = {
    "availability": [
        "factor_key",
        "category",
        "factor",
        "status",
        "prediction_safe_for_feed_model",
        "current_table_columns_found",
        "proxy_columns_found",
        "missing_fields_or_context",
        "best_non_missing_rate",
        "max_unique_values",
        "verification_method",
        "recommendation",
    ],
    "univariate": [
        "factor_key",
        "category",
        "factor",
        "feature",
        "target",
        "test",
        "n",
        "statistic",
        "p_value",
        "effect_size",
        "notes",
    ],
    "model_comparison": [
        "validation",
        "feature_group",
        "model",
        "train_r2",
        "train_rmse",
        "train_mae",
        "test_r2",
        "test_rmse",
        "test_mae",
        "p80_precision",
        "p80_recall",
        "p80_false_negative_rate",
        "n_train",
        "n_test",
        "metadata",
        "numeric_features",
        "categorical_features",
    ],
    "permutation": [
        "feature_group",
        "model",
        "feature",
        "importance_mean_rmse_reduction",
        "importance_std",
    ],
    "evidence": [
        "factor_key",
        "category",
        "factor",
        "status",
        "evidence_level",
        "verification_summary",
        "best_feature",
        "best_p_value",
        "best_effect_size",
        "related_model_group",
        "next_action",
    ],
    "proxy_impact": [
        "proxy_cluster",
        "feature_group",
        "description",
        "best_random_rmse",
        "best_random_r2",
        "best_out_of_time_rmse",
        "best_out_of_time_r2",
        "baseline_out_of_time_rmse",
        "out_of_time_rmse_delta_vs_baseline",
        "best_p80_recall",
        "interpretation",
    ],
    "proxy_mapping": [
        "proxy_cluster",
        "proxy_features",
        "possible_unobserved_factors",
        "evidence_level",
        "interpretation",
        "allowed_wording",
        "forbidden_wording",
    ],
}


@dataclass(frozen=True)
class FactorSpec:
    key: str
    category: str
    name: str
    current_fields: tuple[str, ...]
    proxy_fields: tuple[str, ...]
    prediction_safe: bool
    verification_method: str
    recommendation: str
    process_response: bool = False


@dataclass(frozen=True)
class FactorFeatureGroup:
    name: str
    numeric_features: tuple[str, ...]
    categorical_features: tuple[str, ...]
    include_month: bool

    @property
    def source_features(self) -> list[str]:
        features = {DATE_COL}
        for feature in self.numeric_features:
            if feature in {"month"}:
                continue
            features.add(feature)
        features.update(self.categorical_features)
        return sorted(features)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    estimator: object
    scale_features: bool = False


FACTOR_SPECS = [
    FactorSpec(
        key="fruit_origin_mill_source",
        category="source_origin",
        name="fruit origin / mill source",
        current_fields=(),
        proxy_fields=("source_file", "feed_tank"),
        prediction_safe=True,
        verification_method="proxy categorical tests and tank/source model lift",
        recommendation="Interpret only as tank/source proxy evidence; do not claim origin is identified.",
    ),
    FactorSpec(
        key="soil_fertility",
        category="source_origin",
        name="soil fertility",
        current_fields=(),
        proxy_fields=(),
        prediction_safe=True,
        verification_method="not assessable with current tables",
        recommendation="Leave as unconfirmed hypothesis under current data constraints.",
    ),
    FactorSpec(
        key="rain_harvest_conditions",
        category="weather_harvest",
        name="rainy season and harvest conditions",
        current_fields=(),
        proxy_fields=("date", "month"),
        prediction_safe=True,
        verification_method="date/month proxy tests and time-context model comparison",
        recommendation="Interpret only as time-context proxy evidence, not rainfall/harvest proof.",
    ),
    FactorSpec(
        key="storage_time_temperature",
        category="storage_transport_mixing",
        name="storage time and temperature",
        current_fields=(),
        proxy_fields=("date", "feed_tank"),
        prediction_safe=True,
        verification_method="date/tank proxy tests and model-group lift",
        recommendation="Interpret only as proxy-supported hidden context.",
    ),
    FactorSpec(
        key="tank_mixing",
        category="storage_transport_mixing",
        name="tank mixing",
        current_fields=(),
        proxy_fields=("feed_tank",),
        prediction_safe=True,
        verification_method="feed-tank proxy tests and model-group lift",
        recommendation="Interpret only as feed-tank proxy evidence.",
    ),
    FactorSpec(
        key="transport_batch_mixing",
        category="storage_transport_mixing",
        name="transport / batch mixing",
        current_fields=(),
        proxy_fields=("source_file", "feed_tank"),
        prediction_safe=True,
        verification_method="source/tank proxy tests and model-group lift",
        recommendation="Interpret only as source/tank proxy evidence.",
    ),
    FactorSpec(
        key="lab_sampling_delay",
        category="lab_quality",
        name="lab sampling delay",
        current_fields=(),
        proxy_fields=(),
        prediction_safe=False,
        verification_method="not assessable with current tables",
        recommendation="Leave as unconfirmed hypothesis under current data constraints.",
    ),
    FactorSpec(
        key="lab_measurement_error",
        category="lab_quality",
        name="lab measurement error",
        current_fields=(),
        proxy_fields=(),
        prediction_safe=False,
        verification_method="not assessable with current tables",
        recommendation="Leave as unconfirmed hypothesis under current data constraints.",
    ),
    FactorSpec(
        key="process_regime_change",
        category="process_response",
        name="process regime change",
        current_fields=(),
        proxy_fields=("date",),
        prediction_safe=False,
        verification_method="date proxy is insufficient for process-regime validation",
        recommendation="Keep as not assessable; date is not direct process-regime evidence.",
        process_response=True,
    ),
    FactorSpec(
        key="acid_bleaching_response",
        category="process_response",
        name="acid / bleaching earth dosage response",
        current_fields=tuple(PROCESS_RESPONSE_FEATURES),
        proxy_fields=(),
        prediction_safe=False,
        verification_method="process-response tests against RBD P or P removal delta",
        recommendation="Discuss only as process-response evidence; exclude from feed P prediction.",
        process_response=True,
    ),
    FactorSpec(
        key="metal_phospholipid_complexes",
        category="chemistry_metals",
        name="metal-phospholipid complexes",
        current_fields=(),
        proxy_fields=(),
        prediction_safe=True,
        verification_method="not assessable with current tables",
        recommendation="Leave as unconfirmed hypothesis under current data constraints.",
    ),
]

MODEL_SPECS = [
    ModelSpec(name="ridge", estimator=Ridge(alpha=10.0), scale_features=True),
    ModelSpec(
        name="rf_regularized",
        estimator=RandomForestRegressor(
            n_estimators=120,
            max_depth=6,
            min_samples_leaf=5,
            min_samples_split=10,
            max_features="sqrt",
            random_state=RANDOM_STATE,
            n_jobs=int(os.getenv("CPO_MODEL_N_JOBS", "1")),
        ),
    ),
]


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if value is pd.NA:
        return None
    return value


def _join_list(values) -> str:
    return "|".join(str(value) for value in values if value is not None)


def _normalize_key_text(series: pd.Series) -> pd.Series:
    return (
        series.where(series.notna(), np.nan)
        .astype(str)
        .str.strip()
        .str.upper()
        .str.replace(r"\s+", "", regex=True)
        .replace({"": np.nan, "NAN": np.nan, "NONE": np.nan})
    )


def _load_model_source(input_path: str | Path, target_col: str) -> pd.DataFrame:
    data = pd.read_csv(input_path)
    if DATE_COL not in data.columns:
        raise ValueError(f"Date column '{DATE_COL}' not found in {input_path}")
    if target_col not in data.columns:
        raise ValueError(f"Target column '{target_col}' not found in {input_path}")

    data[DATE_COL] = pd.to_datetime(data[DATE_COL], errors="coerce")
    for key_col in ["feed_tank", "feed_type", "source_file"]:
        if key_col in data.columns:
            data[key_col] = _normalize_key_text(data[key_col])

    numeric_cols = set(QUALITY_FEATURES) | {
        target_col,
        "rbd_p_ppm",
        "acid_dosing_pct",
        "bleaching_earth_dosing_pct",
        "time_trend",
        "missing_transition_phase",
    }
    for col in numeric_cols:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")

    return data[data[DATE_COL].notna()].sort_values(DATE_COL).reset_index(drop=True)


def add_current_table_features(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    data = df.copy()
    data[DATE_COL] = pd.to_datetime(data[DATE_COL], errors="coerce")
    data["month"] = data[DATE_COL].dt.month
    data["year"] = data[DATE_COL].dt.year
    if "time_trend" not in data.columns:
        data["time_trend"] = np.arange(len(data), dtype=float)

    ordered = data.sort_values(DATE_COL).copy()
    y = pd.to_numeric(ordered[target_col], errors="coerce")
    lag1 = y.shift(1)
    date_diff = ordered[DATE_COL].diff().dt.days
    lag1.loc[date_diff.gt(2)] = np.nan
    ordered["feed_p_ppm_lag1"] = lag1
    ordered["feed_p_ppm_roll3"] = lag1.rolling(window=3, min_periods=2).mean()
    data = ordered.sort_index()

    if target_col in data.columns and "rbd_p_ppm" in data.columns:
        data["p_removal_delta"] = (
            pd.to_numeric(data[target_col], errors="coerce")
            - pd.to_numeric(data["rbd_p_ppm"], errors="coerce")
        )
    if {"acid_dosing_pct", "bleaching_earth_dosing_pct"}.issubset(data.columns):
        acid = pd.to_numeric(data["acid_dosing_pct"], errors="coerce")
        bleaching = pd.to_numeric(data["bleaching_earth_dosing_pct"], errors="coerce")
        data["dosage_total_pct"] = acid + bleaching
        data["acid_x_bleaching"] = acid * bleaching

    return data


def _usable_columns(df: pd.DataFrame, columns: tuple[str, ...] | list[str]) -> list[str]:
    usable = []
    for col in columns:
        if col not in df.columns:
            continue
        series = df[col]
        if series.notna().sum() == 0:
            continue
        if series.nunique(dropna=True) <= 0:
            continue
        usable.append(col)
    return usable


def build_factor_availability(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for spec in FACTOR_SPECS:
        current_found = _usable_columns(df, spec.current_fields)
        proxy_found = _usable_columns(df, spec.proxy_fields)
        if current_found:
            status = "available"
            status_cols = current_found
        elif proxy_found:
            status = "proxy_available"
            status_cols = proxy_found
        else:
            status = "missing"
            status_cols = []

        non_missing_rates = []
        unique_counts = []
        for col in status_cols:
            non_missing_rates.append(float(df[col].notna().mean()))
            unique_counts.append(int(df[col].nunique(dropna=True)))

        rows.append(
            {
                "factor_key": spec.key,
                "category": spec.category,
                "factor": spec.name,
                "status": status,
                "prediction_safe_for_feed_model": bool(spec.prediction_safe),
                "current_table_columns_found": _join_list(current_found),
                "proxy_columns_found": _join_list(proxy_found),
                "missing_fields_or_context": (
                    "" if status != "missing" else "no current-table field or useful proxy"
                ),
                "best_non_missing_rate": (
                    None if not non_missing_rates else round(max(non_missing_rates), 6)
                ),
                "max_unique_values": None if not unique_counts else max(unique_counts),
                "verification_method": spec.verification_method,
                "recommendation": spec.recommendation,
            }
        )

    return pd.DataFrame(rows, columns=REPORT_COLUMNS["availability"])


def _is_prediction_excluded(feature: str) -> bool:
    if feature in PREDICTION_EXCLUDED_COLUMNS:
        return True
    return any(feature.startswith(prefix) for prefix in PREDICTION_EXCLUDED_PREFIXES)


def _is_numeric_feature(df: pd.DataFrame, feature: str) -> bool:
    if feature == DATE_COL:
        return False
    converted = pd.to_numeric(df[feature], errors="coerce")
    non_missing = df[feature].notna().sum()
    return non_missing > 0 and converted.notna().sum() / max(non_missing, 1) >= 0.8


def _numeric_univariate(df: pd.DataFrame, feature: str, target: str) -> dict[str, object]:
    data = df[[feature, target]].copy()
    data[feature] = pd.to_numeric(data[feature], errors="coerce")
    data[target] = pd.to_numeric(data[target], errors="coerce")
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    if len(data) < 8 or data[feature].nunique() < 2 or data[target].nunique() < 2:
        return {
            "test": "spearman",
            "n": int(len(data)),
            "statistic": None,
            "p_value": None,
            "effect_size": None,
            "notes": "insufficient_variation_or_samples",
        }

    rho, p_value = scipy_stats.spearmanr(data[feature], data[target], nan_policy="omit")
    return {
        "test": "spearman",
        "n": int(len(data)),
        "statistic": None if not np.isfinite(rho) else round(float(rho), 6),
        "p_value": None if not np.isfinite(p_value) else round(float(p_value), 6),
        "effect_size": None if not np.isfinite(rho) else round(abs(float(rho)), 6),
        "notes": "",
    }


def _categorical_univariate(df: pd.DataFrame, feature: str, target: str) -> dict[str, object]:
    data = df[[feature, target]].copy()
    if feature == DATE_COL:
        data[feature] = pd.to_datetime(data[feature], errors="coerce").dt.to_period("M").astype(str)
    data[target] = pd.to_numeric(data[target], errors="coerce")
    data = data.dropna()
    if data.empty:
        return {
            "test": "kruskal",
            "n": 0,
            "statistic": None,
            "p_value": None,
            "effect_size": None,
            "notes": "no_complete_rows",
        }

    group_sizes = data.groupby(feature)[target].size()
    valid_levels = group_sizes[group_sizes >= 3].index
    filtered = data[data[feature].isin(valid_levels)]
    groups = [group[target].to_numpy(dtype=float) for _, group in filtered.groupby(feature)]
    if len(groups) < 2 or sum(len(group) for group in groups) < 8:
        return {
            "test": "kruskal",
            "n": int(len(filtered)),
            "statistic": None,
            "p_value": None,
            "effect_size": None,
            "notes": "insufficient_category_support",
        }

    stat, p_value = scipy_stats.kruskal(*groups, nan_policy="omit")
    n = sum(len(group) for group in groups)
    k = len(groups)
    epsilon_sq = (stat - k + 1) / max(n - k, 1)
    return {
        "test": "kruskal",
        "n": int(n),
        "statistic": None if not np.isfinite(stat) else round(float(stat), 6),
        "p_value": None if not np.isfinite(p_value) else round(float(p_value), 6),
        "effect_size": round(float(max(epsilon_sq, 0.0)), 6),
        "notes": f"levels_tested={k}",
    }


def _candidate_columns_for_spec(df: pd.DataFrame, spec: FactorSpec, status: str) -> list[str]:
    if status == "available":
        return _usable_columns(df, spec.current_fields)
    if status == "proxy_available":
        return _usable_columns(df, spec.proxy_fields)
    return []


def build_univariate_tests(
    df: pd.DataFrame, availability: pd.DataFrame, target_col: str
) -> pd.DataFrame:
    rows = []
    status_by_key = dict(zip(availability["factor_key"], availability["status"]))

    for spec in FACTOR_SPECS:
        status = status_by_key.get(spec.key, "missing")
        candidates = _candidate_columns_for_spec(df, spec, status)
        target = target_col
        if spec.process_response:
            if "p_removal_delta" in df.columns and df["p_removal_delta"].notna().any():
                target = "p_removal_delta"
            elif "rbd_p_ppm" in df.columns:
                target = "rbd_p_ppm"

        for feature in candidates:
            if feature == target or target not in df.columns:
                continue
            result = (
                _numeric_univariate(df, feature, target)
                if _is_numeric_feature(df, feature)
                else _categorical_univariate(df, feature, target)
            )
            rows.append(
                {
                    "factor_key": spec.key,
                    "category": spec.category,
                    "factor": spec.name,
                    "feature": feature,
                    "target": target,
                    **result,
                }
            )

    return pd.DataFrame(rows, columns=REPORT_COLUMNS["univariate"])


def _valid_numeric_features(df: pd.DataFrame, columns: list[str]) -> list[str]:
    features = []
    for col in columns:
        if col not in df.columns or _is_prediction_excluded(col):
            continue
        values = pd.to_numeric(df[col], errors="coerce")
        if values.notna().sum() >= 5 and values.nunique(dropna=True) >= 2:
            features.append(col)
    return list(dict.fromkeys(features))


def _valid_categorical_features(df: pd.DataFrame, columns: list[str]) -> list[str]:
    features = []
    for col in columns:
        if col not in df.columns or _is_prediction_excluded(col):
            continue
        if df[col].notna().sum() >= 5 and df[col].nunique(dropna=True) >= 2:
            features.append(col)
    return list(dict.fromkeys(features))


def build_factor_feature_groups(df: pd.DataFrame) -> list[FactorFeatureGroup]:
    quality = _valid_numeric_features(df, QUALITY_FEATURES)
    history = _valid_numeric_features(df, [*QUALITY_FEATURES, *HISTORY_STATE_FEATURES])
    time_context = _valid_numeric_features(df, [*QUALITY_FEATURES, *TIME_CONTEXT_FEATURES])
    combined = _valid_numeric_features(
        df, [*QUALITY_FEATURES, *HISTORY_STATE_FEATURES, *TIME_CONTEXT_FEATURES]
    )
    proxies = _valid_categorical_features(df, BASE_PROXY_CATEGORICAL)

    groups = [
        FactorFeatureGroup(
            name="current_quality_baseline",
            numeric_features=tuple(quality),
            categorical_features=(),
            include_month=False,
        ),
        FactorFeatureGroup(
            name="history_state",
            numeric_features=tuple(history),
            categorical_features=(),
            include_month=False,
        ),
        FactorFeatureGroup(
            name="tank_source_context",
            numeric_features=tuple(quality),
            categorical_features=tuple(proxies),
            include_month=False,
        ),
        FactorFeatureGroup(
            name="time_operating_context",
            numeric_features=tuple(time_context),
            categorical_features=(),
            include_month=True,
        ),
        FactorFeatureGroup(
            name="combined_proxy_context",
            numeric_features=tuple(combined),
            categorical_features=tuple(proxies),
            include_month=True,
        ),
    ]
    return [group for group in groups if group.numeric_features or group.categorical_features]


def _build_model_pipeline(group: FactorFeatureGroup, spec: ModelSpec) -> Pipeline:
    preprocessor = build_leakage_safe_preprocessor(
        numeric_features=list(group.numeric_features),
        categorical_features=list(group.categorical_features),
        include_month=group.include_month,
        iqr_columns=QUALITY_FEATURES,
    )
    steps = [("preprocess", preprocessor)]
    if spec.scale_features:
        steps.append(("scale", StandardScaler()))
    steps.append(("model", clone(spec.estimator)))
    return Pipeline(steps=steps)


def _filter_group_for_training(
    df: pd.DataFrame, group: FactorFeatureGroup, train_idx
) -> FactorFeatureGroup | None:
    numeric = []
    for feature in group.numeric_features:
        if feature not in df.columns:
            continue
        values = pd.to_numeric(df.loc[train_idx, feature], errors="coerce")
        if values.notna().any():
            numeric.append(feature)

    categorical = []
    for feature in group.categorical_features:
        if feature not in df.columns:
            continue
        if df.loc[train_idx, feature].notna().any():
            categorical.append(feature)

    if not numeric and not categorical and not group.include_month:
        return None
    return FactorFeatureGroup(
        name=group.name,
        numeric_features=tuple(numeric),
        categorical_features=tuple(categorical),
        include_month=group.include_month,
    )


def _evaluate_predictions(y_true, y_pred) -> dict[str, object]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) < 2:
        return {"r2": None, "rmse": None, "mae": None, "n_samples": int(len(y_true))}
    return {
        "r2": round(float(r2_score(y_true, y_pred)), 6),
        "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 6),
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 6),
        "n_samples": int(len(y_true)),
    }


def _risk_metrics(y_true, y_score, threshold) -> dict[str, object]:
    actual = np.asarray(y_true, dtype=float)
    score = np.asarray(y_score, dtype=float)
    valid = np.isfinite(actual) & np.isfinite(score)
    actual = actual[valid]
    score = score[valid]
    if len(actual) == 0:
        return {
            "p80_precision": None,
            "p80_recall": None,
            "p80_false_negative_rate": None,
        }
    actual_high = actual >= threshold
    predicted_high = score >= threshold
    tp = int(np.sum(actual_high & predicted_high))
    fp = int(np.sum(~actual_high & predicted_high))
    fn = int(np.sum(actual_high & ~predicted_high))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else None
    fnr = fn / (tp + fn) if (tp + fn) else None
    return {
        "p80_precision": round(float(precision), 6),
        "p80_recall": None if recall is None else round(float(recall), 6),
        "p80_false_negative_rate": None if fnr is None else round(float(fnr), 6),
    }


def _metric_row(
    validation: str,
    group: FactorFeatureGroup,
    spec: ModelSpec,
    train_metrics: dict[str, object],
    test_metrics: dict[str, object],
    risk_metrics: dict[str, object],
    metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "validation": validation,
        "feature_group": group.name,
        "model": spec.name,
        "train_r2": train_metrics.get("r2"),
        "train_rmse": train_metrics.get("rmse"),
        "train_mae": train_metrics.get("mae"),
        "test_r2": test_metrics.get("r2"),
        "test_rmse": test_metrics.get("rmse"),
        "test_mae": test_metrics.get("mae"),
        **risk_metrics,
        "n_train": train_metrics.get("n_samples", 0),
        "n_test": test_metrics.get("n_samples", 0),
        "metadata": json.dumps(_json_safe(metadata or {}), ensure_ascii=False, sort_keys=True),
        "numeric_features": _join_list(group.numeric_features),
        "categorical_features": _join_list(group.categorical_features),
    }


def _fit_evaluate_split(
    df: pd.DataFrame,
    y: pd.Series,
    group: FactorFeatureGroup,
    spec: ModelSpec,
    train_idx,
    test_idx,
    threshold: float,
    validation: str,
    metadata: dict[str, object] | None = None,
) -> tuple[dict[str, object] | None, Pipeline | None]:
    if len(train_idx) < MIN_TEST_SAMPLES or len(test_idx) < 2:
        return None, None

    effective_group = _filter_group_for_training(df, group, train_idx)
    if effective_group is None:
        return None, None

    X = df[effective_group.source_features].copy()
    model = _build_model_pipeline(effective_group, spec)
    model.fit(X.loc[train_idx], y.loc[train_idx])
    train_pred = model.predict(X.loc[train_idx])
    test_pred = model.predict(X.loc[test_idx])
    train_metrics = _evaluate_predictions(y.loc[train_idx], train_pred)
    test_metrics = _evaluate_predictions(y.loc[test_idx], test_pred)
    risk = _risk_metrics(y.loc[test_idx], test_pred, threshold)
    return (
        _metric_row(validation, effective_group, spec, train_metrics, test_metrics, risk, metadata),
        model,
    )


def _evaluate_monthly_rolling(
    df: pd.DataFrame,
    y: pd.Series,
    group: FactorFeatureGroup,
    spec: ModelSpec,
    threshold: float,
    min_train_months: int = 3,
    min_train_samples: int = 60,
    min_test_samples: int = 5,
) -> list[dict[str, object]]:
    data = df.copy()
    data["_period"] = data[DATE_COL].dt.to_period("M")
    periods = sorted(data["_period"].dropna().unique())
    y_true_all = []
    y_pred_all = []
    train_n = 0
    effective_group = None

    for period_index, period in enumerate(periods):
        if period_index < min_train_months:
            continue
        train_periods = periods[:period_index]
        train_idx = data.index[data["_period"].isin(train_periods)]
        test_idx = data.index[data["_period"].eq(period)]
        if len(train_idx) < min_train_samples or len(test_idx) < min_test_samples:
            continue

        effective_group = _filter_group_for_training(data, group, train_idx)
        if effective_group is None:
            continue
        X = data[effective_group.source_features].copy()
        model = _build_model_pipeline(effective_group, spec)
        model.fit(X.loc[train_idx], y.loc[train_idx])
        pred = model.predict(X.loc[test_idx])
        y_true_all.extend(y.loc[test_idx].tolist())
        y_pred_all.extend(pred.tolist())
        train_n = max(train_n, int(len(train_idx)))

    if not y_true_all or effective_group is None:
        return []

    train_metrics = {"r2": None, "rmse": None, "mae": None, "n_samples": train_n}
    test_metrics = _evaluate_predictions(y_true_all, y_pred_all)
    risk = _risk_metrics(y_true_all, y_pred_all, threshold)
    return [
        _metric_row(
            "monthly_rolling",
            effective_group,
            spec,
            train_metrics,
            test_metrics,
            risk,
            {
                "n_windows": len(set(periods[min_train_months:])),
                "test_period_start": str(periods[min_train_months]),
                "test_period_end": str(periods[-1]),
            },
        )
    ]


def _build_permutation_importance(
    df: pd.DataFrame,
    y: pd.Series,
    best_random: tuple[dict[str, object], Pipeline, FactorFeatureGroup, ModelSpec, object] | None,
) -> pd.DataFrame:
    if best_random is None:
        return pd.DataFrame(columns=REPORT_COLUMNS["permutation"])
    row, model, group, spec, test_idx = best_random
    X_test = df.loc[test_idx, group.source_features].copy()
    if len(X_test) < 5:
        return pd.DataFrame(columns=REPORT_COLUMNS["permutation"])

    result = permutation_importance(
        model,
        X_test,
        y.loc[test_idx],
        n_repeats=8,
        random_state=RANDOM_STATE,
        scoring="neg_root_mean_squared_error",
        n_jobs=int(os.getenv("CPO_MODEL_N_JOBS", "1")),
    )
    rows = []
    for feature, mean, std in zip(X_test.columns, result.importances_mean, result.importances_std):
        rows.append(
            {
                "feature_group": row["feature_group"],
                "model": spec.name,
                "feature": feature,
                "importance_mean_rmse_reduction": round(float(mean), 6),
                "importance_std": round(float(std), 6),
            }
        )
    return (
        pd.DataFrame(rows, columns=REPORT_COLUMNS["permutation"])
        .sort_values("importance_mean_rmse_reduction", ascending=False)
        .reset_index(drop=True)
    )


def build_factor_model_comparison(
    df: pd.DataFrame, target_col: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = df[df[DATE_COL].notna()].copy()
    y = pd.to_numeric(data[target_col], errors="coerce")
    valid = y.notna()
    data = data.loc[valid].reset_index(drop=True)
    y = y.loc[valid].reset_index(drop=True)
    if len(data) < 30:
        empty = pd.DataFrame(columns=REPORT_COLUMNS["model_comparison"])
        return empty, pd.DataFrame(columns=REPORT_COLUMNS["permutation"])

    threshold = float(y.quantile(RISK_QUANTILE))
    groups = build_factor_feature_groups(data)
    rows = []
    best_random = None

    all_idx = data.index
    random_train_idx, random_test_idx = train_test_split(
        all_idx, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    for group in groups:
        for spec in MODEL_SPECS:
            row, model = _fit_evaluate_split(
                data,
                y,
                group,
                spec,
                random_train_idx,
                random_test_idx,
                threshold,
                "random_split",
            )
            if row is not None:
                rows.append(row)
                if row["test_rmse"] is not None:
                    effective_group = FactorFeatureGroup(
                        name=group.name,
                        numeric_features=tuple(
                            feature for feature in row["numeric_features"].split("|") if feature
                        ),
                        categorical_features=tuple(
                            feature for feature in row["categorical_features"].split("|") if feature
                        ),
                        include_month=group.include_month,
                    )
                    if best_random is None or row["test_rmse"] < best_random[0]["test_rmse"]:
                        best_random = (row, model, effective_group, spec, random_test_idx)

            ordered = data.sort_values(DATE_COL).reset_index()
            split_idx = int(len(ordered) * 0.8)
            if split_idx >= MIN_TEST_SAMPLES and len(ordered) - split_idx >= 2:
                train_idx = ordered.loc[: split_idx - 1, "index"]
                test_idx = ordered.loc[split_idx:, "index"]
                metadata = {
                    "train_date_min": str(data.loc[train_idx, DATE_COL].min().date()),
                    "train_date_max": str(data.loc[train_idx, DATE_COL].max().date()),
                    "test_date_min": str(data.loc[test_idx, DATE_COL].min().date()),
                    "test_date_max": str(data.loc[test_idx, DATE_COL].max().date()),
                }
                row, _ = _fit_evaluate_split(
                    data,
                    y,
                    group,
                    spec,
                    train_idx,
                    test_idx,
                    threshold,
                    "blocked_time_holdout",
                    metadata,
                )
                if row is not None:
                    rows.append(row)

            years = sorted(data["year"].dropna().astype(int).unique().tolist())
            for train_year in years:
                for test_year in years:
                    if train_year == test_year:
                        continue
                    train_idx = data.index[data["year"].eq(train_year)]
                    test_idx = data.index[data["year"].eq(test_year)]
                    row, _ = _fit_evaluate_split(
                        data,
                        y,
                        group,
                        spec,
                        train_idx,
                        test_idx,
                        threshold,
                        "year_holdout",
                        {"train_year": int(train_year), "test_year": int(test_year)},
                    )
                    if row is not None:
                        rows.append(row)

            rows.extend(_evaluate_monthly_rolling(data, y, group, spec, threshold))

    comparison = pd.DataFrame(rows, columns=REPORT_COLUMNS["model_comparison"])
    permutation = _build_permutation_importance(data, y, best_random)
    return comparison, permutation


def _best_metric(
    comparison: pd.DataFrame,
    group_name: str,
    metric_col: str,
    validations: tuple[str, ...] | None = None,
    maximize: bool = False,
) -> float | None:
    if comparison.empty:
        return None
    subset = comparison[comparison["feature_group"].eq(group_name)]
    if validations is not None:
        subset = subset[subset["validation"].isin(validations)]
    values = pd.to_numeric(subset[metric_col], errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.max() if maximize else values.min())


def _best_out_of_time_rmse(comparison: pd.DataFrame, group_name: str) -> float | None:
    return _best_metric(comparison, group_name, "test_rmse", OUT_OF_TIME_VALIDATIONS)


def _best_out_of_time_r2(comparison: pd.DataFrame, group_name: str) -> float | None:
    return _best_metric(comparison, group_name, "test_r2", OUT_OF_TIME_VALIDATIONS, maximize=True)


def _best_random_rmse(comparison: pd.DataFrame, group_name: str) -> float | None:
    return _best_metric(comparison, group_name, "test_rmse", ("random_split",))


def _best_random_r2(comparison: pd.DataFrame, group_name: str) -> float | None:
    return _best_metric(comparison, group_name, "test_r2", ("random_split",), maximize=True)


def _best_out_of_time_p80_recall(comparison: pd.DataFrame, group_name: str) -> float | None:
    return _best_metric(
        comparison, group_name, "p80_recall", OUT_OF_TIME_VALIDATIONS, maximize=True
    )


def _best_univariate_for_factor(univariate: pd.DataFrame, factor_key: str) -> dict[str, object]:
    if univariate.empty:
        return {}
    subset = univariate[univariate["factor_key"].eq(factor_key)].copy()
    if subset.empty:
        return {}
    subset["p_sort"] = pd.to_numeric(subset["p_value"], errors="coerce").fillna(1.0)
    subset["effect_sort"] = pd.to_numeric(subset["effect_size"], errors="coerce").fillna(0.0)
    subset = subset.sort_values(["p_sort", "effect_sort"], ascending=[True, False])
    return subset.iloc[0].to_dict()


def _has_univariate_signal(best_uni: dict[str, object]) -> bool:
    p_value = best_uni.get("p_value")
    effect_size = best_uni.get("effect_size")
    return (
        p_value is not None
        and not pd.isna(p_value)
        and float(p_value) < 0.05
        and (effect_size is None or pd.isna(effect_size) or float(effect_size) >= 0.05)
    )


def build_evidence_matrix(
    availability: pd.DataFrame,
    univariate: pd.DataFrame,
    comparison: pd.DataFrame,
) -> pd.DataFrame:
    baseline_out = _best_out_of_time_rmse(comparison, "current_quality_baseline")
    history_out = _best_out_of_time_rmse(comparison, "history_state")
    tank_out = _best_out_of_time_rmse(comparison, "tank_source_context")
    time_out = _best_out_of_time_rmse(comparison, "time_operating_context")
    combined_out = _best_out_of_time_rmse(comparison, "combined_proxy_context")

    def improved(group_out: float | None, reference: float | None = baseline_out) -> bool:
        return (
            reference is not None
            and group_out is not None
            and group_out < reference - MIN_RMSE_LIFT
        )

    history_improved = improved(history_out)
    tank_improved = improved(tank_out)
    time_improved = improved(time_out)
    combined_improved = improved(combined_out)
    proxy_signal_by_factor = {
        "fruit_origin_mill_source": tank_improved or combined_improved,
        "rain_harvest_conditions": time_improved or combined_improved,
        "storage_time_temperature": tank_improved or time_improved or combined_improved,
        "tank_mixing": tank_improved or history_improved or combined_improved,
        "transport_batch_mixing": tank_improved or combined_improved,
    }
    related_group_by_factor = {
        "fruit_origin_mill_source": "tank_source_context/combined_proxy_context",
        "rain_harvest_conditions": "time_operating_context/combined_proxy_context",
        "storage_time_temperature": "tank_source_context/time_operating_context/combined_proxy_context",
        "tank_mixing": "history_state/tank_source_context/combined_proxy_context",
        "transport_batch_mixing": "tank_source_context/combined_proxy_context",
    }
    not_assessable = {
        "soil_fertility",
        "lab_sampling_delay",
        "lab_measurement_error",
        "process_regime_change",
        "metal_phospholipid_complexes",
    }

    rows = []
    availability_by_key = availability.set_index("factor_key").to_dict(orient="index")
    for spec in FACTOR_SPECS:
        available = availability_by_key[spec.key]
        status = available["status"]
        best_uni = _best_univariate_for_factor(univariate, spec.key)
        has_signal = _has_univariate_signal(best_uni)

        if spec.key in not_assessable:
            evidence = "not_assessable"
            summary = "Current 2024/2025 tables do not contain direct fields or useful proxy evidence to isolate this factor."
            related_group = ""
        elif spec.process_response:
            if has_signal:
                evidence = "observed_effect"
                summary = "Direct acid/bleaching dosage fields show an observed relationship with RBD P or P removal delta."
            else:
                evidence = "not_assessable"
                summary = "Direct process-response fields are present, but a stable relationship is not established."
            related_group = "process_response_univariate_only"
        elif status == "proxy_available":
            if proxy_signal_by_factor.get(spec.key, False) or has_signal:
                evidence = "proxy_supported"
                summary = "Only proxy fields are present; current-table proxy signals support hidden context but do not identify the exact factor."
            else:
                evidence = "not_assessable"
                summary = "Only proxy fields are present, and current validation does not show a stable enough proxy signal."
            related_group = related_group_by_factor.get(spec.key, "combined_proxy_context")
        else:
            evidence = "not_assessable"
            summary = "Current 2024/2025 tables contain no current-table evidence for this factor."
            related_group = ""

        rows.append(
            {
                "factor_key": spec.key,
                "category": spec.category,
                "factor": spec.name,
                "status": status,
                "evidence_level": evidence,
                "verification_summary": summary,
                "best_feature": best_uni.get("feature"),
                "best_p_value": best_uni.get("p_value"),
                "best_effect_size": best_uni.get("effect_size"),
                "related_model_group": related_group,
                "next_action": spec.recommendation,
            }
        )

    return pd.DataFrame(rows, columns=REPORT_COLUMNS["evidence"])


def _format_value(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return f"{value:.3f}"


def build_proxy_factor_impact_summary(comparison: pd.DataFrame) -> pd.DataFrame:
    baseline_out = _best_out_of_time_rmse(comparison, "current_quality_baseline")
    cluster_specs = [
        (
            "quality_baseline",
            "current_quality_baseline",
            "Current feed quality fields only.",
        ),
        (
            "history_state",
            "history_state",
            "Recent feed P lag/rolling state available before the next prediction.",
        ),
        (
            "tank_source_context",
            "tank_source_context",
            "Feed tank and source-file proxies for hidden source, tank usage, storage, and mixing context.",
        ),
        (
            "time_operating_context",
            "time_operating_context",
            "Date/month/time proxies for seasonality, procurement mix, and operating context.",
        ),
        (
            "combined_proxy_context",
            "combined_proxy_context",
            "Combined history, tank/source, and time proxy context.",
        ),
    ]

    rows = []
    for proxy_cluster, feature_group, description in cluster_specs:
        random_rmse = _best_random_rmse(comparison, feature_group)
        random_r2 = _best_random_r2(comparison, feature_group)
        out_rmse = _best_out_of_time_rmse(comparison, feature_group)
        out_r2 = _best_out_of_time_r2(comparison, feature_group)
        p80_recall = _best_out_of_time_p80_recall(comparison, feature_group)
        delta = None
        if baseline_out is not None and out_rmse is not None:
            delta = baseline_out - out_rmse

        if feature_group == "current_quality_baseline":
            interpretation = "baseline_quality_only"
        elif delta is not None and delta > MIN_RMSE_LIFT:
            interpretation = "proxy_supported_out_of_time_improvement"
        elif random_rmse is not None:
            interpretation = "random_split_or_descriptive_signal_only"
        else:
            interpretation = "not_tested_or_insufficient_data"

        rows.append(
            {
                "proxy_cluster": proxy_cluster,
                "feature_group": feature_group,
                "description": description,
                "best_random_rmse": random_rmse,
                "best_random_r2": random_r2,
                "best_out_of_time_rmse": out_rmse,
                "best_out_of_time_r2": out_r2,
                "baseline_out_of_time_rmse": baseline_out,
                "out_of_time_rmse_delta_vs_baseline": delta,
                "best_p80_recall": p80_recall,
                "interpretation": interpretation,
            }
        )

    return pd.DataFrame(rows, columns=REPORT_COLUMNS["proxy_impact"])


def build_proxy_to_hypothesis_mapping(
    evidence: pd.DataFrame, proxy_impact: pd.DataFrame
) -> pd.DataFrame:
    evidence_by_key = evidence.set_index("factor_key")["evidence_level"].to_dict()
    impact_by_cluster = proxy_impact.set_index("proxy_cluster")["interpretation"].to_dict()

    def proxy_level(cluster: str, related_factor_keys: tuple[str, ...] = ()) -> str:
        interpretation = str(impact_by_cluster.get(cluster, ""))
        if "out_of_time_improvement" in interpretation:
            return "proxy_supported"
        if any(evidence_by_key.get(key) == "proxy_supported" for key in related_factor_keys):
            return "proxy_supported"
        return "not_assessable"

    process_level = evidence_by_key.get("acid_bleaching_response", "not_assessable")
    rows = [
        {
            "proxy_cluster": "history_state",
            "proxy_features": "feed_p_ppm_lag1|feed_p_ppm_roll3",
            "possible_unobserved_factors": "recent feed P state|carryover|unobserved upstream/tank context",
            "evidence_level": proxy_level("history_state"),
            "interpretation": "History features can support prediction, but they do not reveal which hidden upstream factor caused the prior P level.",
            "allowed_wording": "history-state proxies show predictive signal for feed P",
            "forbidden_wording": "storage, origin, or chemistry mechanism is confirmed by lag features",
        },
        {
            "proxy_cluster": "tank_source_context",
            "proxy_features": "feed_tank|source_file|feed_type",
            "possible_unobserved_factors": "tank usage|source mix|storage/mixing|transport/batch context|mill/source origin",
            "evidence_level": proxy_level(
                "tank_source_context",
                (
                    "fruit_origin_mill_source",
                    "storage_time_temperature",
                    "tank_mixing",
                    "transport_batch_mixing",
                ),
            ),
            "interpretation": "Tank/source proxies may capture hidden operational context, but current tables cannot decompose that context.",
            "allowed_wording": "tank/source context is proxy-supported",
            "forbidden_wording": "mill origin, storage time, or mixing is directly confirmed",
        },
        {
            "proxy_cluster": "time_operating_context",
            "proxy_features": "date|month|time_trend",
            "possible_unobserved_factors": "seasonality|weather/harvest conditions|procurement mix|operating context|missing-month transition",
            "evidence_level": proxy_level(
                "time_operating_context",
                ("rain_harvest_conditions", "storage_time_temperature"),
            ),
            "interpretation": "Time proxies can show hidden context shifts, but cannot separate rainfall, harvest, or process-regime causes.",
            "allowed_wording": "time operating context is proxy-supported",
            "forbidden_wording": "rainfall, harvest condition, or process regime is directly confirmed",
        },
        {
            "proxy_cluster": "process_response",
            "proxy_features": "acid_dosing_pct|bleaching_earth_dosing_pct|dosage_total_pct|acid_x_bleaching",
            "possible_unobserved_factors": "downstream process response|RBD P removal behavior",
            "evidence_level": process_level,
            "interpretation": "Dosing fields are direct current-table fields for process-response discussion and are excluded from feed P prediction.",
            "allowed_wording": "acid/bleaching dosage shows observed process-response evidence against P removal metrics",
            "forbidden_wording": "dosing variables prove feed P is predictable or support automatic dosage optimization",
        },
        {
            "proxy_cluster": "not_assessable_current_tables",
            "proxy_features": "",
            "possible_unobserved_factors": "soil fertility|lab sampling delay|lab measurement error|metal-phospholipid complexes|direct process regime",
            "evidence_level": "not_assessable",
            "interpretation": "Current 2024/2025 tables have no direct fields and no useful proxy to isolate these factors.",
            "allowed_wording": "these factors remain plausible but not assessable with current tables",
            "forbidden_wording": "soil, lab error, metals, or process regime effects are confirmed",
        },
    ]
    return pd.DataFrame(rows, columns=REPORT_COLUMNS["proxy_mapping"])


def write_recommendations_md(
    output_path: Path,
    availability: pd.DataFrame,
    evidence: pd.DataFrame,
    comparison: pd.DataFrame,
) -> None:
    status_counts = availability["status"].value_counts().to_dict()
    evidence_counts = evidence["evidence_level"].value_counts().to_dict()

    lines = [
        "# Potential Influencing Factors Validation",
        "",
        "## Data Coverage",
        "- Current conclusion scope: existing 2024/2025 quality tables only.",
        "- No sidecar, weather, soil, lab, source, tank, or process-regime records are ingested.",
        f"- Availability counts: {json.dumps(status_counts, ensure_ascii=False, sort_keys=True)}",
        f"- Evidence counts: {json.dumps(evidence_counts, ensure_ascii=False, sort_keys=True)}",
        "",
        "## Model Boundary",
        "- Feed phosphorus models use only current quality, history-state, source/tank, and time-context fields available before prediction.",
        "- RBD variables, acid dosing, bleaching earth dosing, and P removal delta are excluded from feed prediction.",
        "- Acid/bleaching fields are evaluated only as process-response evidence against RBD P or P removal delta.",
        "",
        "## Not Assessable With Current Tables",
    ]

    gaps = evidence[evidence["evidence_level"].eq("not_assessable")]
    for row in gaps.head(8).to_dict(orient="records"):
        lines.append(f"- `{row['factor_key']}`: {row['next_action']}")

    lines.extend(["", "## Current Evidence Summary"])
    for row in evidence.to_dict(orient="records"):
        lines.append(
            f"- `{row['factor_key']}`: `{row['evidence_level']}` - {row['verification_summary']}"
        )

    if not comparison.empty:
        lines.extend(["", "## Model Group Snapshot"])
        snapshot = (
            comparison.groupby(["feature_group", "validation"], dropna=False)["test_rmse"]
            .min()
            .reset_index()
            .sort_values(["validation", "test_rmse"])
        )
        for row in snapshot.head(16).to_dict(orient="records"):
            rmse = "NA" if pd.isna(row["test_rmse"]) else f"{row['test_rmse']:.3f}"
            lines.append(f"- `{row['validation']}` / `{row['feature_group']}`: best RMSE {rmse}")

    lines.extend(
        [
            "",
            "## Wording Guardrails",
            "- Use `observed_effect` only for direct current-table process-response relationships.",
            "- Use `proxy_supported` for history, tank/source, and time-context signals that cannot identify a hidden cause.",
            "- Use `not_assessable` for soil, lab, metals, process regime, and other factors without direct current-table fields.",
            "- Do not claim rainfall, soil fertility, storage time, mill origin, metal complexes, lab error, production-grade prediction, or automatic dosing optimization are confirmed.",
        ]
    )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_final_factor_conclusion_md(
    output_path: Path,
    evidence: pd.DataFrame,
    proxy_impact: pd.DataFrame,
    proxy_mapping: pd.DataFrame,
) -> None:
    counts = evidence["evidence_level"].value_counts().to_dict()
    impact_by_cluster = proxy_impact.set_index("proxy_cluster").to_dict(orient="index")
    baseline = impact_by_cluster.get("quality_baseline", {})
    combined = impact_by_cluster.get("combined_proxy_context", {})
    history = impact_by_cluster.get("history_state", {})
    tank = impact_by_cluster.get("tank_source_context", {})
    time_context = impact_by_cluster.get("time_operating_context", {})
    process_rows = evidence[evidence["factor_key"].eq("acid_bleaching_response")]
    process_level = (
        "not_assessable" if process_rows.empty else process_rows.iloc[0]["evidence_level"]
    )

    lines = [
        "# Final Factor Conclusion",
        "",
        "## Scope",
        "- This phase uses only the existing 2024 and 2025 quality tables.",
        "- No enterprise sidecar, weather, soil, lab, source, tank, or process-regime records are used.",
        "- The report confirms observed proxy effects where possible, not the exact hidden physical or operational causes.",
        "",
        "## Evidence Classification",
        f"- Evidence counts: {json.dumps(counts, ensure_ascii=False, sort_keys=True)}",
        f"- Quality-only baseline best out-of-time RMSE: {_format_value(baseline.get('best_out_of_time_rmse'))}",
        f"- History-state best out-of-time RMSE delta vs baseline: {_format_value(history.get('out_of_time_rmse_delta_vs_baseline'))}",
        f"- Tank/source best out-of-time RMSE delta vs baseline: {_format_value(tank.get('out_of_time_rmse_delta_vs_baseline'))}",
        f"- Time-context best out-of-time RMSE delta vs baseline: {_format_value(time_context.get('out_of_time_rmse_delta_vs_baseline'))}",
        f"- Combined proxy-context best out-of-time RMSE delta vs baseline: {_format_value(combined.get('out_of_time_rmse_delta_vs_baseline'))}",
        f"- Combined proxy-context best out-of-time p80 recall: {_format_value(combined.get('best_p80_recall'))}",
        "",
        "## Supported Conclusions",
        "- Current quality variables alone remain weak for feed P prediction.",
        "- History-state and combined proxy context show the strongest out-of-time lift; tank/source has smaller standalone lift, while time context is proxy-supported mainly through date/month signals rather than stable standalone model lift.",
        "- These proxies indicate hidden context matters, but they cannot identify which exact unobserved factor is responsible.",
        f"- Acid/bleaching dosage response is classified as `{process_level}` for RBD P or P-removal discussion only.",
        "",
        "## Not Directly Confirmed",
        "- Rainfall, harvest condition, soil fertility, storage time, transport mixing, mill origin, lab error, process regime, and metal-phospholipid mechanisms are not directly confirmed by the current tables.",
        "- The current evidence does not support production-grade automatic phosphorus prediction, lab-test replacement, or automatic acid/bleaching optimization.",
        "",
        "## Proxy Wording Guardrails",
    ]

    for row in proxy_mapping.to_dict(orient="records"):
        lines.append(
            f"- `{row['proxy_cluster']}`: allowed: {row['allowed_wording']}; avoid: {row['forbidden_wording']}."
        )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_pipeline(
    input_path: str | Path,
    output_dir: str | Path,
    target_col: str = DEFAULT_TARGET_COL,
) -> dict[str, object]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    base = _load_model_source(input_path, target_col)
    frame = add_current_table_features(base, target_col)

    availability = build_factor_availability(frame)
    univariate = build_univariate_tests(frame, availability, target_col)
    comparison, permutation = build_factor_model_comparison(frame, target_col)
    evidence = build_evidence_matrix(availability, univariate, comparison)
    proxy_impact = build_proxy_factor_impact_summary(comparison)
    proxy_mapping = build_proxy_to_hypothesis_mapping(evidence, proxy_impact)

    availability.to_csv(output / "factor_availability.csv", index=False)
    univariate.to_csv(output / "factor_univariate_tests.csv", index=False)
    comparison.to_csv(output / "factor_group_model_comparison.csv", index=False)
    permutation.to_csv(output / "factor_permutation_importance.csv", index=False)
    evidence.to_csv(output / "factor_evidence_matrix.csv", index=False)
    proxy_impact.to_csv(output / "proxy_factor_impact_summary.csv", index=False)
    proxy_mapping.to_csv(output / "proxy_to_hypothesis_mapping.csv", index=False)
    write_recommendations_md(
        output / "factor_recommendations.md",
        availability,
        evidence,
        comparison,
    )
    write_final_factor_conclusion_md(
        output / "final_factor_conclusion.md",
        evidence,
        proxy_impact,
        proxy_mapping,
    )

    summary = {
        "target_col": target_col,
        "input_path": str(input_path),
        "n_rows": int(len(frame)),
        "date_min": str(frame[DATE_COL].min().date()) if frame[DATE_COL].notna().any() else None,
        "date_max": str(frame[DATE_COL].max().date()) if frame[DATE_COL].notna().any() else None,
        "availability_counts": availability["status"].value_counts().to_dict(),
        "evidence_counts": evidence["evidence_level"].value_counts().to_dict(),
        "output_files": {
            "availability": "factor_availability.csv",
            "univariate_tests": "factor_univariate_tests.csv",
            "group_model_comparison": "factor_group_model_comparison.csv",
            "permutation_importance": "factor_permutation_importance.csv",
            "evidence_matrix": "factor_evidence_matrix.csv",
            "proxy_factor_impact_summary": "proxy_factor_impact_summary.csv",
            "proxy_to_hypothesis_mapping": "proxy_to_hypothesis_mapping.csv",
            "recommendations": "factor_recommendations.md",
            "final_factor_conclusion": "final_factor_conclusion.md",
        },
        "notes": [
            "Current-phase conclusions are based only on the 2024/2025 quality tables.",
            "Proxy-supported evidence should not be described as direct or causal proof of a specific hidden factor.",
            "Not-assessable factors remain plausible but unconfirmed with current tables.",
            "Process-response features are excluded from feed phosphorus prediction models.",
        ],
    }
    with (output / "factor_validation_summary.json").open("w", encoding="utf-8") as f:
        json.dump(_json_safe(summary), f, ensure_ascii=False, indent=2, allow_nan=False)
    return summary


def parse_args():
    default_target_col = os.getenv("CPO_TARGET_COL", DEFAULT_TARGET_COL)
    parser = argparse.ArgumentParser(description="Validate current-table proxy factors")
    parser.add_argument(
        "--input",
        default=str(LOCAL_PROCESSED_DATA_DIR / "model_source.csv"),
        help=f"Path to model_source.csv (default: {LOCAL_PROCESSED_DATA_DIR / 'model_source.csv'})",
    )
    parser.add_argument(
        "--output-dir",
        default=str(LOCAL_FACTOR_VALIDATION_REPORTS_DIR),
        help=f"Directory for outputs (default: {LOCAL_FACTOR_VALIDATION_REPORTS_DIR})",
    )
    parser.add_argument(
        "--target-col",
        default=default_target_col,
        help=f"Target column to validate (default: {default_target_col})",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    summary = run_pipeline(
        input_path=args.input,
        output_dir=args.output_dir,
        target_col=args.target_col,
    )
    print(json.dumps(_json_safe(summary), ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
