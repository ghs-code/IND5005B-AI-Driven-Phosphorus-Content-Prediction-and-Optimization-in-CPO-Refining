#!/usr/bin/env python3
"""Validate potential influencing factors for feed-oil phosphorus.

The existing project has enough data to test only a few proxy signals
(`source_file`, `feed_tank`, calendar context, and recent phosphorus history).
This module adds a conservative validation layer for the broader factor list
from the final report plan.  Optional sponsor-provided sidecar data can be
joined to `model_source.csv`; when it is absent, the pipeline still produces a
clear evidence matrix showing which factors are proxy-only or missing.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, Ridge
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

QUALITY_FEATURES = [
    "feed_ffa_pct",
    "feed_mi_pct",
    "feed_iv",
    "feed_dobi",
    "feed_car_pv",
]

BASE_PROXY_CATEGORICAL = ["source_file", "feed_tank", "feed_type"]
HISTORY_FEATURES = ["time_trend", "feed_p_ppm_lag1", "feed_p_ppm_roll3"]

PREDICTION_EXCLUDED_PREFIXES = ("rbd_",)
PREDICTION_EXCLUDED_COLUMNS = {
    "feed_p_ppm",
    "log_feed_p_ppm",
    "rbd_p_ppm",
    "p_removal_delta",
    "acid_dosing_pct",
    "bleaching_earth_dosing_pct",
    "dosage_total_pct",
    "acid_x_bleaching",
    "lab_received_timestamp",
    "lab_result_timestamp",
    "sample_timestamp",
    "sampling_timestamp",
    "feed_sampling_timestamp",
    "sampling_delay_hours",
    "lab_turnaround_hours",
    "lab_duplicate_id",
    "lab_replicate_no",
    "lab_p_replicate_ppm",
}

NUMERIC_FACTOR_COLUMNS = {
    "soil_ph",
    "soil_p_mg_kg",
    "soil_organic_matter_pct",
    "fertilizer_p_rate",
    "rainfall_mm",
    "rainfall_7d",
    "rainfall_30d",
    "harvest_maturity",
    "storage_hours",
    "storage_temperature_c",
    "ambient_temperature_c",
    "tank_residence_hours",
    "source_count_in_tank",
    "mixed_batch_flag",
    "batch_count",
    "transport_batch_count",
    "sampling_delay_hours",
    "lab_turnaround_hours",
    "acid_dosing_pct",
    "bleaching_earth_dosing_pct",
    "dosage_total_pct",
    "acid_x_bleaching",
    "fe_ppm",
    "ca_ppm",
    "mg_ppm",
    "metal_ppm",
    "phospholipid_ppm",
}

CATEGORICAL_FACTOR_COLUMNS = {
    "batch_id",
    "fruit_origin",
    "mill_source",
    "estate",
    "supplier",
    "harvest_condition",
    "source_list",
    "transport_batch_id",
    "lorry_id",
    "transport_mixing_flag",
    "tank_mixing_event",
    "lab_duplicate_id",
    "lab_method",
    "lab_operator",
    "lab_instrument",
    "process_regime",
    "line_id",
    "operation_mode",
    "maintenance_flag",
}

SAFE_EXTENDED_NUMERIC = [
    "soil_ph",
    "soil_p_mg_kg",
    "soil_organic_matter_pct",
    "fertilizer_p_rate",
    "rainfall_7d",
    "rainfall_30d",
    "storage_hours",
    "storage_temperature_c",
    "ambient_temperature_c",
    "tank_residence_hours",
    "source_count_in_tank",
    "mixed_batch_flag",
    "batch_count",
    "transport_batch_count",
    "fe_ppm",
    "ca_ppm",
    "mg_ppm",
    "metal_ppm",
    "phospholipid_ppm",
]

SAFE_EXTENDED_CATEGORICAL = [
    "batch_id",
    "fruit_origin",
    "mill_source",
    "estate",
    "supplier",
    "harvest_condition",
    "source_list",
    "transport_batch_id",
    "lorry_id",
    "transport_mixing_flag",
    "tank_mixing_event",
]

REPORT_COLUMNS = {
    "availability": [
        "factor_key",
        "category",
        "factor",
        "status",
        "prediction_safe_for_feed_model",
        "direct_columns_found",
        "proxy_columns_found",
        "missing_direct_fields",
        "best_non_missing_rate",
        "max_unique_values",
        "verification_method",
        "recommendation",
    ],
    "join_diagnostics": [
        "factor_input",
        "status",
        "join_key",
        "merge_validation",
        "base_rows",
        "sidecar_rows",
        "rows_after_merge",
        "matched_rows",
        "unmatched_rows",
        "match_rate",
        "base_duplicate_key_rows",
        "sidecar_duplicate_key_rows",
        "sidecar_columns",
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
        "partial_spearman_rho",
        "partial_spearman_p_value",
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
}


@dataclass(frozen=True)
class FactorSpec:
    key: str
    category: str
    name: str
    direct_fields: tuple[str, ...]
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
            if feature in {"time_trend", "month"}:
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
        direct_fields=("batch_id", "fruit_origin", "mill_source", "estate", "supplier"),
        proxy_fields=("source_file", "feed_tank"),
        prediction_safe=True,
        verification_method="categorical Kruskal/ANOVA proxy and model-group lift",
        recommendation="Capture mill/source or batch origin at each feed-tank draw.",
    ),
    FactorSpec(
        key="soil_fertility",
        category="source_origin",
        name="soil fertility",
        direct_fields=(
            "soil_ph",
            "soil_p_mg_kg",
            "soil_organic_matter_pct",
            "fertilizer_p_rate",
        ),
        proxy_fields=("fruit_origin", "mill_source", "estate"),
        prediction_safe=True,
        verification_method="Spearman and incremental extended-factor model lift",
        recommendation="Collect estate-level soil/fertilizer attributes by origin.",
    ),
    FactorSpec(
        key="rain_harvest_conditions",
        category="weather_harvest",
        name="rainy season and harvest conditions",
        direct_fields=(
            "rainfall_mm",
            "rainfall_7d",
            "rainfall_30d",
            "harvest_date",
            "harvest_condition",
            "harvest_maturity",
        ),
        proxy_fields=("month", "date"),
        prediction_safe=True,
        verification_method="monthly trend, Spearman, and out-of-time model lift",
        recommendation="Add harvest/weather columns or date-level rainfall sidecar.",
    ),
    FactorSpec(
        key="storage_time_temperature",
        category="storage_transport_mixing",
        name="storage time and temperature",
        direct_fields=(
            "storage_start_ts",
            "storage_end_ts",
            "storage_hours",
            "storage_temperature_c",
            "ambient_temperature_c",
        ),
        proxy_fields=("date", "feed_tank"),
        prediction_safe=True,
        verification_method="Spearman and incremental extended-factor model lift",
        recommendation="Record storage start/end timestamps and tank temperature.",
    ),
    FactorSpec(
        key="tank_mixing",
        category="storage_transport_mixing",
        name="tank mixing",
        direct_fields=(
            "tank_residence_hours",
            "source_count_in_tank",
            "mixed_batch_flag",
            "source_list",
            "tank_mixing_event",
        ),
        proxy_fields=("feed_tank",),
        prediction_safe=True,
        verification_method="categorical tests and history/tank proxy model lift",
        recommendation="Track source count, tank residence time, and mixing events.",
    ),
    FactorSpec(
        key="transport_batch_mixing",
        category="storage_transport_mixing",
        name="transport / batch mixing",
        direct_fields=(
            "transport_batch_id",
            "lorry_id",
            "batch_count",
            "transport_batch_count",
            "transport_mixing_flag",
        ),
        proxy_fields=("source_file", "feed_tank"),
        prediction_safe=True,
        verification_method="categorical tests and extended-factor model lift",
        recommendation="Add transport batch and lorry identifiers to the sidecar.",
    ),
    FactorSpec(
        key="lab_sampling_delay",
        category="lab_quality",
        name="lab sampling delay",
        direct_fields=(
            "sample_timestamp",
            "sampling_timestamp",
            "lab_received_timestamp",
            "lab_result_timestamp",
            "sampling_delay_hours",
            "lab_turnaround_hours",
        ),
        proxy_fields=(),
        prediction_safe=False,
        verification_method="Spearman against target/residuals; not used in feed prediction",
        recommendation="Capture sampling, lab receipt, and lab result timestamps.",
    ),
    FactorSpec(
        key="lab_measurement_error",
        category="lab_quality",
        name="lab measurement error",
        direct_fields=(
            "lab_duplicate_id",
            "lab_replicate_no",
            "lab_method",
            "lab_operator",
            "lab_instrument",
            "lab_p_replicate_ppm",
        ),
        proxy_fields=(),
        prediction_safe=False,
        verification_method="replicate variance and same-condition variance",
        recommendation="Collect duplicate/replicate measurements and lab metadata.",
    ),
    FactorSpec(
        key="process_regime_change",
        category="process_response",
        name="process regime change",
        direct_fields=("process_regime", "line_id", "operation_mode", "maintenance_flag"),
        proxy_fields=("date",),
        prediction_safe=False,
        verification_method="process-response tests against RBD P or P removal delta",
        recommendation="Record process regime, line, and maintenance/changeover events.",
        process_response=True,
    ),
    FactorSpec(
        key="acid_bleaching_response",
        category="process_response",
        name="acid / bleaching earth dosage response",
        direct_fields=(
            "acid_dosing_pct",
            "bleaching_earth_dosing_pct",
            "dosage_total_pct",
            "acid_x_bleaching",
        ),
        proxy_fields=(),
        prediction_safe=False,
        verification_method="process-response tests against RBD P or P removal delta",
        recommendation="Treat dosing as downstream response evidence, not feed prediction input.",
        process_response=True,
    ),
    FactorSpec(
        key="metal_phospholipid_complexes",
        category="chemistry_metals",
        name="metal-phospholipid complexes",
        direct_fields=("fe_ppm", "ca_ppm", "mg_ppm", "metal_ppm", "phospholipid_ppm"),
        proxy_fields=(),
        prediction_safe=True,
        verification_method="Spearman and incremental extended-factor model lift",
        recommendation="Add metal/phospholipid assays when available before prediction.",
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


def _normalize_column_name(name: str) -> str:
    text = str(name).strip().lower()
    text = re.sub(r"[^0-9a-zA-Z]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    aliases = {
        "sample_ts": "sample_timestamp",
        "sampling_ts": "sampling_timestamp",
        "lab_received_ts": "lab_received_timestamp",
        "lab_result_ts": "lab_result_timestamp",
        "storage_start_timestamp": "storage_start_ts",
        "storage_end_timestamp": "storage_end_ts",
    }
    return aliases.get(text, text)


def _normalize_key_text(series: pd.Series) -> pd.Series:
    return (
        series.where(series.notna(), np.nan)
        .astype(str)
        .str.strip()
        .str.upper()
        .str.replace(r"\s+", "", regex=True)
        .replace({"": np.nan, "NAN": np.nan, "NONE": np.nan})
    )


def _read_table(path: str | Path) -> pd.DataFrame:
    input_path = Path(path).expanduser()
    if not input_path.exists():
        raise FileNotFoundError(f"Factor sidecar input does not exist: {input_path}")

    if input_path.suffix.lower() in {".xlsx", ".xlsm", ".xls"}:
        data = pd.read_excel(input_path)
    elif input_path.suffix.lower() in {".csv", ".txt"}:
        data = pd.read_csv(input_path)
    else:
        raise ValueError(f"Unsupported factor sidecar format: {input_path.suffix}")

    data = data.rename(columns={col: _normalize_column_name(col) for col in data.columns})
    if DATE_COL in data.columns:
        data[DATE_COL] = pd.to_datetime(data[DATE_COL], errors="coerce")
    for key_col in ["feed_tank", "batch_id"]:
        if key_col in data.columns:
            data[key_col] = _normalize_key_text(data[key_col])
    return data


def _load_model_source(input_path: str | Path, target_col: str) -> pd.DataFrame:
    data = pd.read_csv(input_path)
    if DATE_COL not in data.columns:
        raise ValueError(f"Date column '{DATE_COL}' not found in {input_path}")
    if target_col not in data.columns:
        raise ValueError(f"Target column '{target_col}' not found in {input_path}")

    data[DATE_COL] = pd.to_datetime(data[DATE_COL], errors="coerce")
    for key_col in ["feed_tank", "feed_type", "rbd_tank", "rbd_type"]:
        if key_col in data.columns:
            data[key_col] = _normalize_key_text(data[key_col])

    for col in [
        *QUALITY_FEATURES,
        target_col,
        "rbd_p_ppm",
        "acid_dosing_pct",
        "bleaching_earth_dosing_pct",
    ]:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")

    data = data[data[DATE_COL].notna()].sort_values(DATE_COL).reset_index(drop=True)
    return data


def _choose_join_key(base: pd.DataFrame, sidecar: pd.DataFrame) -> list[str]:
    if "batch_id" in base.columns and "batch_id" in sidecar.columns:
        if base["batch_id"].notna().any() and sidecar["batch_id"].notna().any():
            return ["batch_id"]

    if {DATE_COL, "feed_tank"}.issubset(base.columns) and {DATE_COL, "feed_tank"}.issubset(
        sidecar.columns
    ):
        if sidecar[DATE_COL].notna().any() and sidecar["feed_tank"].notna().any():
            return [DATE_COL, "feed_tank"]

    if DATE_COL in base.columns and DATE_COL in sidecar.columns and sidecar[DATE_COL].notna().any():
        return [DATE_COL]

    return []


def merge_factor_sidecar(
    base: pd.DataFrame, factor_input: str | Path | None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Left-join optional factor sidecar and return merge diagnostics."""

    if not factor_input:
        diagnostics = pd.DataFrame(
            [
                {
                    "factor_input": "",
                    "status": "no_sidecar",
                    "join_key": "",
                    "merge_validation": "",
                    "base_rows": int(len(base)),
                    "sidecar_rows": 0,
                    "rows_after_merge": int(len(base)),
                    "matched_rows": 0,
                    "unmatched_rows": int(len(base)),
                    "match_rate": 0.0,
                    "base_duplicate_key_rows": 0,
                    "sidecar_duplicate_key_rows": 0,
                    "sidecar_columns": "",
                }
            ],
            columns=REPORT_COLUMNS["join_diagnostics"],
        )
        return base.copy(), diagnostics

    sidecar = _read_table(factor_input)
    join_key = _choose_join_key(base, sidecar)
    if not join_key:
        diagnostics = pd.DataFrame(
            [
                {
                    "factor_input": str(factor_input),
                    "status": "no_usable_join_key",
                    "join_key": "",
                    "merge_validation": "",
                    "base_rows": int(len(base)),
                    "sidecar_rows": int(len(sidecar)),
                    "rows_after_merge": int(len(base)),
                    "matched_rows": 0,
                    "unmatched_rows": int(len(base)),
                    "match_rate": 0.0,
                    "base_duplicate_key_rows": 0,
                    "sidecar_duplicate_key_rows": 0,
                    "sidecar_columns": _join_list(sidecar.columns),
                }
            ],
            columns=REPORT_COLUMNS["join_diagnostics"],
        )
        return base.copy(), diagnostics

    base_key = base[join_key]
    sidecar_key = sidecar[join_key]
    base_duplicate_rows = int(base_key.duplicated(keep=False).sum())
    sidecar_duplicate_rows = int(sidecar_key.duplicated(keep=False).sum())
    if sidecar_duplicate_rows:
        raise ValueError(
            "Factor sidecar has duplicate join keys, which could create a one-to-many "
            f"or many-to-many merge. join_key={join_key}, duplicate_rows={sidecar_duplicate_rows}"
        )

    sidecar_payload = sidecar.copy()
    rename_collisions = {
        col: f"factor_{col}"
        for col in sidecar_payload.columns
        if col not in join_key and col in base.columns
    }
    if rename_collisions:
        sidecar_payload = sidecar_payload.rename(columns=rename_collisions)

    merge_validation = "many_to_one" if base_duplicate_rows else "one_to_one"
    merged = base.merge(
        sidecar_payload,
        on=join_key,
        how="left",
        validate=merge_validation,
        indicator=True,
    )
    matched_rows = int(merged["_merge"].eq("both").sum())
    merged = merged.drop(columns=["_merge"])

    diagnostics = pd.DataFrame(
        [
            {
                "factor_input": str(factor_input),
                "status": "merged",
                "join_key": _join_list(join_key),
                "merge_validation": merge_validation,
                "base_rows": int(len(base)),
                "sidecar_rows": int(len(sidecar)),
                "rows_after_merge": int(len(merged)),
                "matched_rows": matched_rows,
                "unmatched_rows": int(len(merged) - matched_rows),
                "match_rate": round(float(matched_rows / max(len(merged), 1)), 6),
                "base_duplicate_key_rows": base_duplicate_rows,
                "sidecar_duplicate_key_rows": sidecar_duplicate_rows,
                "sidecar_columns": _join_list(sidecar.columns),
            }
        ],
        columns=REPORT_COLUMNS["join_diagnostics"],
    )
    return merged, diagnostics


def _parse_datetime_columns(data: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for col in columns:
        if col in data.columns:
            data[col] = pd.to_datetime(data[col], errors="coerce")
    return data


def _derive_hours(data: pd.DataFrame, output_col: str, pairs: list[tuple[str, str]]) -> None:
    if output_col in data.columns and pd.to_numeric(data[output_col], errors="coerce").notna().any():
        data[output_col] = pd.to_numeric(data[output_col], errors="coerce")
        return

    for start_col, end_col in pairs:
        if start_col not in data.columns or end_col not in data.columns:
            continue
        start = pd.to_datetime(data[start_col], errors="coerce")
        end = pd.to_datetime(data[end_col], errors="coerce")
        hours = (end - start).dt.total_seconds() / 3600.0
        if hours.notna().any():
            data[output_col] = hours.where(hours.ge(0))
            return


def _count_delimited_values(value) -> float:
    if pd.isna(value):
        return np.nan
    parts = [part.strip() for part in re.split(r"[;,|+/]", str(value)) if part.strip()]
    return float(len(set(parts))) if parts else np.nan


def add_derived_factor_features(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    data = df.copy()
    data[DATE_COL] = pd.to_datetime(data[DATE_COL], errors="coerce")
    data["month"] = data[DATE_COL].dt.month
    data["year"] = data[DATE_COL].dt.year

    numeric_candidates = set(NUMERIC_FACTOR_COLUMNS) | set(QUALITY_FEATURES) | {
        target_col,
        "rbd_p_ppm",
    }
    for col in sorted(numeric_candidates):
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")

    timestamp_columns = [
        "storage_start_ts",
        "storage_end_ts",
        "harvest_date",
        "harvest_timestamp",
        "feed_sampling_timestamp",
        "sample_timestamp",
        "sampling_timestamp",
        "lab_received_timestamp",
        "lab_result_timestamp",
        "tank_in_timestamp",
        "tank_out_timestamp",
        "tank_fill_timestamp",
        "tank_draw_timestamp",
        "feed_tank_fill_ts",
        "feed_tank_draw_ts",
    ]
    data = _parse_datetime_columns(data, timestamp_columns)

    ordered = data.sort_values(DATE_COL).copy()
    y = pd.to_numeric(ordered[target_col], errors="coerce")
    lag1 = y.shift(1)
    date_diff = ordered[DATE_COL].diff().dt.days
    lag1.loc[date_diff.gt(2)] = np.nan
    ordered[f"{target_col}_lag1"] = lag1
    ordered[f"{target_col}_roll3"] = lag1.rolling(window=3, min_periods=2).mean()
    data = ordered.sort_index()

    _derive_hours(
        data,
        "storage_hours",
        [
            ("storage_start_ts", "storage_end_ts"),
            ("harvest_timestamp", "feed_sampling_timestamp"),
            ("harvest_date", DATE_COL),
        ],
    )
    _derive_hours(
        data,
        "tank_residence_hours",
        [
            ("tank_in_timestamp", "tank_out_timestamp"),
            ("tank_fill_timestamp", "tank_draw_timestamp"),
            ("feed_tank_fill_ts", "feed_tank_draw_ts"),
            ("tank_fill_timestamp", "feed_sampling_timestamp"),
        ],
    )
    _derive_hours(
        data,
        "sampling_delay_hours",
        [
            ("sample_timestamp", "lab_received_timestamp"),
            ("sampling_timestamp", "lab_received_timestamp"),
            ("feed_sampling_timestamp", "lab_received_timestamp"),
        ],
    )
    _derive_hours(
        data,
        "lab_turnaround_hours",
        [
            ("lab_received_timestamp", "lab_result_timestamp"),
            ("sample_timestamp", "lab_result_timestamp"),
            ("sampling_timestamp", "lab_result_timestamp"),
        ],
    )

    if "rainfall_mm" in data.columns:
        rainfall = pd.to_numeric(data["rainfall_mm"], errors="coerce")
        daily = (
            pd.DataFrame({DATE_COL: data[DATE_COL], "rainfall_mm": rainfall})
            .dropna(subset=[DATE_COL])
            .groupby(DATE_COL)["rainfall_mm"]
            .mean()
            .sort_index()
        )
        if "rainfall_7d" not in data.columns and not daily.empty:
            data["rainfall_7d"] = data[DATE_COL].map(daily.rolling("7D", min_periods=1).sum())
        if "rainfall_30d" not in data.columns and not daily.empty:
            data["rainfall_30d"] = data[DATE_COL].map(daily.rolling("30D", min_periods=1).sum())

    if "source_count_in_tank" not in data.columns:
        for count_col in ["source_count", "batch_count", "transport_batch_count"]:
            if count_col in data.columns:
                data["source_count_in_tank"] = pd.to_numeric(data[count_col], errors="coerce")
                break
        if "source_count_in_tank" not in data.columns:
            for list_col in ["source_list", "mill_source_list", "batch_list"]:
                if list_col in data.columns:
                    data["source_count_in_tank"] = data[list_col].map(_count_delimited_values)
                    break

    if "mixed_batch_flag" not in data.columns:
        if "source_count_in_tank" in data.columns:
            data["mixed_batch_flag"] = (
                pd.to_numeric(data["source_count_in_tank"], errors="coerce").gt(1).astype("Int64")
            )
        else:
            for flag_col in ["tank_mixing_event", "transport_mixing_flag"]:
                if flag_col in data.columns:
                    data["mixed_batch_flag"] = data[flag_col]
                    break

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
        direct_found = _usable_columns(df, spec.direct_fields)
        proxy_found = _usable_columns(df, spec.proxy_fields)
        if direct_found:
            status = "available"
            status_cols = direct_found
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
                "direct_columns_found": _join_list(direct_found),
                "proxy_columns_found": _join_list(proxy_found),
                "missing_direct_fields": _join_list(
                    [col for col in spec.direct_fields if col not in direct_found]
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


def _spec_by_key(key: str) -> FactorSpec:
    for spec in FACTOR_SPECS:
        if spec.key == key:
            return spec
    raise KeyError(key)


def _is_prediction_excluded(feature: str) -> bool:
    if feature in PREDICTION_EXCLUDED_COLUMNS:
        return True
    return any(feature.startswith(prefix) for prefix in PREDICTION_EXCLUDED_PREFIXES)


def _is_numeric_feature(df: pd.DataFrame, feature: str) -> bool:
    if feature == DATE_COL:
        return False
    if feature in CATEGORICAL_FACTOR_COLUMNS:
        return False
    if feature in NUMERIC_FACTOR_COLUMNS:
        return True
    converted = pd.to_numeric(df[feature], errors="coerce")
    non_missing = df[feature].notna().sum()
    return non_missing > 0 and converted.notna().sum() / max(non_missing, 1) >= 0.8


def _partial_spearman(
    df: pd.DataFrame, feature: str, target: str, controls: list[str]
) -> tuple[float | None, float | None]:
    controls = [col for col in controls if col in df.columns and col not in {feature, target}]
    cols = list(dict.fromkeys([feature, target] + controls))
    data = df[cols].dropna().copy()
    if len(data) < 12 or data[feature].nunique() < 2 or data[target].nunique() < 2:
        return None, None

    x_rank = pd.to_numeric(data[feature], errors="coerce").rank()
    y_rank = pd.to_numeric(data[target], errors="coerce").rank()
    control_df = pd.DataFrame(index=data.index)
    for col in controls:
        if col not in data.columns:
            continue
        if pd.api.types.is_numeric_dtype(data[col]):
            control_df[col] = pd.to_numeric(data[col], errors="coerce")
        else:
            dummies = pd.get_dummies(data[col].astype(str), prefix=col, drop_first=True)
            control_df = pd.concat([control_df, dummies], axis=1)

    control_df = control_df.replace([np.inf, -np.inf], np.nan).fillna(0)
    if control_df.empty or control_df.shape[1] >= len(data) - 3:
        return None, None

    model_x = LinearRegression().fit(control_df, x_rank)
    model_y = LinearRegression().fit(control_df, y_rank)
    x_resid = x_rank - model_x.predict(control_df)
    y_resid = y_rank - model_y.predict(control_df)
    rho, p_value = scipy_stats.spearmanr(x_resid, y_resid, nan_policy="omit")
    if not np.isfinite(rho):
        return None, None
    return float(rho), None if not np.isfinite(p_value) else float(p_value)


def _numeric_univariate(
    df: pd.DataFrame, feature: str, target: str
) -> dict[str, object]:
    cols = [feature, target]
    if "month" in df.columns:
        cols.append("month")
    data = df[list(dict.fromkeys(cols))].copy()
    data[feature] = pd.to_numeric(data[feature], errors="coerce")
    data[target] = pd.to_numeric(data[target], errors="coerce")
    data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=[feature, target])
    if len(data) < 8 or data[feature].nunique() < 2 or data[target].nunique() < 2:
        return {
            "test": "spearman",
            "n": int(len(data)),
            "statistic": None,
            "p_value": None,
            "effect_size": None,
            "partial_spearman_rho": None,
            "partial_spearman_p_value": None,
            "notes": "insufficient_variation_or_samples",
        }

    rho, p_value = scipy_stats.spearmanr(data[feature], data[target], nan_policy="omit")
    partial_rho, partial_p = _partial_spearman(data, feature, target, ["month"])
    return {
        "test": "spearman",
        "n": int(len(data)),
        "statistic": None if not np.isfinite(rho) else round(float(rho), 6),
        "p_value": None if not np.isfinite(p_value) else round(float(p_value), 6),
        "effect_size": None if not np.isfinite(rho) else round(abs(float(rho)), 6),
        "partial_spearman_rho": None if partial_rho is None else round(partial_rho, 6),
        "partial_spearman_p_value": None if partial_p is None else round(partial_p, 6),
        "notes": "",
    }


def _categorical_univariate(
    df: pd.DataFrame, feature: str, target: str
) -> dict[str, object]:
    data = df[[feature, target]].copy()
    if feature == DATE_COL:
        data[feature] = pd.to_datetime(data[feature], errors="coerce").dt.to_period("M").astype(str)
    data[target] = pd.to_numeric(data[target], errors="coerce")
    data = data.dropna(subset=[feature, target])
    if data.empty:
        return {
            "test": "kruskal",
            "n": 0,
            "statistic": None,
            "p_value": None,
            "effect_size": None,
            "partial_spearman_rho": None,
            "partial_spearman_p_value": None,
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
            "partial_spearman_rho": None,
            "partial_spearman_p_value": None,
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
        "partial_spearman_rho": None,
        "partial_spearman_p_value": None,
        "notes": f"levels_tested={k}",
    }


def _candidate_columns_for_spec(df: pd.DataFrame, spec: FactorSpec, status: str) -> list[str]:
    if status == "available":
        candidates = _usable_columns(df, spec.direct_fields)
        if spec.process_response:
            candidates = [
                col for col in candidates if col not in {"rbd_p_ppm", "p_removal_delta"}
            ]
        return candidates
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
            if _is_numeric_feature(df, feature):
                result = _numeric_univariate(df, feature, target)
            else:
                result = _categorical_univariate(df, feature, target)

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

    if "lab_duplicate_id" in df.columns:
        replicate_target = "lab_p_replicate_ppm" if "lab_p_replicate_ppm" in df.columns else target_col
        if replicate_target in df.columns:
            grouped = (
                df[["lab_duplicate_id", replicate_target]]
                .dropna()
                .groupby("lab_duplicate_id")[replicate_target]
            )
            variances = grouped.var().dropna()
            replicated_groups = int((grouped.size() >= 2).sum())
            rows.append(
                {
                    "factor_key": "lab_measurement_error",
                    "category": "lab_quality",
                    "factor": "lab measurement error",
                    "feature": "lab_duplicate_id",
                    "target": replicate_target,
                    "test": "replicate_variance",
                    "n": replicated_groups,
                    "statistic": None if variances.empty else round(float(variances.mean()), 6),
                    "p_value": None,
                    "effect_size": None if variances.empty else round(float(variances.median()), 6),
                    "partial_spearman_rho": None,
                    "partial_spearman_p_value": None,
                    "notes": "mean_and_median_within_duplicate_variance",
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
    proxies = _valid_categorical_features(df, BASE_PROXY_CATEGORICAL)
    history = _valid_numeric_features(df, [*QUALITY_FEATURES, *HISTORY_FEATURES])
    extended_numeric = _valid_numeric_features(df, SAFE_EXTENDED_NUMERIC)
    extended_categorical = _valid_categorical_features(df, SAFE_EXTENDED_CATEGORICAL)

    groups = [
        FactorFeatureGroup(
            name="current_quality_baseline",
            numeric_features=tuple(quality),
            categorical_features=(),
            include_month=False,
        ),
        FactorFeatureGroup(
            name="current_proxy_context",
            numeric_features=tuple(quality),
            categorical_features=tuple(proxies),
            include_month=True,
        ),
        FactorFeatureGroup(
            name="history_tank_proxy",
            numeric_features=tuple(history),
            categorical_features=tuple(proxies),
            include_month=True,
        ),
        FactorFeatureGroup(
            name="extended_pre_feed_factors",
            numeric_features=tuple(list(dict.fromkeys([*history, *extended_numeric]))),
            categorical_features=tuple(list(dict.fromkeys([*proxies, *extended_categorical]))),
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
        if feature == "time_trend":
            numeric.append(feature)
            continue
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

    all_idx = data.index
    random_train_idx, random_test_idx = train_test_split(
        all_idx, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    best_random = None

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

            monthly_rows = _evaluate_monthly_rolling(data, y, group, spec, threshold)
            rows.extend(monthly_rows)

    comparison = pd.DataFrame(rows, columns=REPORT_COLUMNS["model_comparison"])
    permutation = _build_permutation_importance(data, y, best_random)
    return comparison, permutation


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
    rows = []
    y_true_all = []
    y_pred_all = []
    train_n = 0

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

    if not y_true_all:
        return rows

    train_metrics = {"r2": None, "rmse": None, "mae": None, "n_samples": train_n}
    test_metrics = _evaluate_predictions(y_true_all, y_pred_all)
    risk = _risk_metrics(y_true_all, y_pred_all, threshold)
    rows.append(
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
    )
    return rows


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


def _best_out_of_time_rmse(comparison: pd.DataFrame, group_name: str) -> float | None:
    if comparison.empty:
        return None
    subset = comparison[
        comparison["feature_group"].eq(group_name)
        & comparison["validation"].isin(["blocked_time_holdout", "year_holdout", "monthly_rolling"])
    ]
    values = pd.to_numeric(subset["test_rmse"], errors="coerce").dropna()
    return None if values.empty else float(values.min())


def _best_random_rmse(comparison: pd.DataFrame, group_name: str) -> float | None:
    if comparison.empty:
        return None
    subset = comparison[
        comparison["feature_group"].eq(group_name) & comparison["validation"].eq("random_split")
    ]
    values = pd.to_numeric(subset["test_rmse"], errors="coerce").dropna()
    return None if values.empty else float(values.min())


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


def build_evidence_matrix(
    availability: pd.DataFrame,
    univariate: pd.DataFrame,
    comparison: pd.DataFrame,
) -> pd.DataFrame:
    baseline_out = _best_out_of_time_rmse(comparison, "current_quality_baseline")
    proxy_out = _best_out_of_time_rmse(comparison, "current_proxy_context")
    history_out = _best_out_of_time_rmse(comparison, "history_tank_proxy")
    extended_out = _best_out_of_time_rmse(comparison, "extended_pre_feed_factors")
    baseline_random = _best_random_rmse(comparison, "current_quality_baseline")
    extended_random = _best_random_rmse(comparison, "extended_pre_feed_factors")

    proxy_improved = (
        baseline_out is not None and proxy_out is not None and proxy_out < baseline_out - 0.01
    )
    history_improved = (
        baseline_out is not None and history_out is not None and history_out < baseline_out - 0.01
    )
    extended_improved = (
        history_out is not None and extended_out is not None and extended_out < history_out - 0.01
    )
    extended_random_improved = (
        baseline_random is not None
        and extended_random is not None
        and extended_random < baseline_random - 0.01
    )

    rows = []
    availability_by_key = availability.set_index("factor_key").to_dict(orient="index")
    for spec in FACTOR_SPECS:
        available = availability_by_key[spec.key]
        status = available["status"]
        best_uni = _best_univariate_for_factor(univariate, spec.key)
        p_value = best_uni.get("p_value")
        effect_size = best_uni.get("effect_size")
        has_univariate_signal = (
            p_value is not None
            and not pd.isna(p_value)
            and float(p_value) < 0.05
            and (effect_size is None or pd.isna(effect_size) or float(effect_size) >= 0.05)
        )

        if spec.process_response and status != "available":
            evidence = "needs_data_collection"
            summary = "No direct process-response field is available; date proxy is not enough to validate process regime change."
            related_group = "process_response_univariate_only"
        elif status == "missing":
            evidence = "needs_data_collection"
            summary = "No direct or proxy field is available in the current dataset."
            related_group = ""
        elif status == "proxy_available":
            if proxy_improved or history_improved:
                evidence = "weak_proxy_evidence"
                summary = "Only proxy fields are present; proxy/context groups improve out-of-time RMSE."
            else:
                evidence = "inconclusive_proxy"
                summary = "Only proxy fields are present and stable out-of-time lift is not established."
            related_group = "current_proxy_context/history_tank_proxy"
        elif spec.process_response:
            if has_univariate_signal:
                evidence = "process_response_signal"
                summary = "Direct process-response fields show univariate signal against RBD P or P removal delta."
            else:
                evidence = "inconclusive_process_response"
                summary = "Direct process-response fields are present but signal is not yet robust."
            related_group = "process_response_univariate_only"
        elif extended_improved and has_univariate_signal:
            evidence = "supported"
            summary = "Direct fields are present, show univariate signal, and improve out-of-time model performance."
            related_group = "extended_pre_feed_factors"
        elif extended_random_improved or has_univariate_signal:
            evidence = "weak_evidence"
            summary = "Direct fields are present, but evidence is limited to univariate or random-split lift."
            related_group = "extended_pre_feed_factors"
        else:
            evidence = "inconclusive"
            summary = "Direct fields are present, but no stable relationship has been established."
            related_group = "extended_pre_feed_factors"

        rows.append(
            {
                "factor_key": spec.key,
                "category": spec.category,
                "factor": spec.name,
                "status": status,
                "evidence_level": evidence,
                "verification_summary": summary,
                "best_feature": best_uni.get("feature"),
                "best_p_value": p_value,
                "best_effect_size": effect_size,
                "related_model_group": related_group,
                "next_action": spec.recommendation,
            }
        )

    return pd.DataFrame(rows, columns=REPORT_COLUMNS["evidence"])


def write_recommendations_md(
    output_path: Path,
    join_diag: pd.DataFrame,
    availability: pd.DataFrame,
    evidence: pd.DataFrame,
    comparison: pd.DataFrame,
) -> None:
    status_counts = availability["status"].value_counts().to_dict()
    evidence_counts = evidence["evidence_level"].value_counts().to_dict()
    join_row = join_diag.iloc[0].to_dict()

    lines = [
        "# Potential Influencing Factors Validation",
        "",
        "## Data Coverage",
        f"- Sidecar status: `{join_row['status']}`",
        f"- Join key: `{join_row['join_key'] or 'NA'}`",
        f"- Match rate: {join_row['match_rate']}",
        f"- Availability counts: {json.dumps(status_counts, ensure_ascii=False, sort_keys=True)}",
        f"- Evidence counts: {json.dumps(evidence_counts, ensure_ascii=False, sort_keys=True)}",
        "",
        "## Model Boundary",
        "- Feed phosphorus models use only current quality, source/tank/date proxy, history, and prediction-time pre-feed fields.",
        "- RBD variables, acid dosing, bleaching earth dosing, lab turnaround, and P removal delta are excluded from feed prediction.",
        "- Acid/bleaching and process-regime fields are evaluated only as process-response evidence against RBD P or P removal delta.",
        "",
        "## Highest-Priority Data Gaps",
    ]

    gaps = evidence[evidence["evidence_level"].eq("needs_data_collection")]
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

    lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def run_pipeline(
    input_path: str | Path,
    output_dir: str | Path,
    target_col: str = DEFAULT_TARGET_COL,
    factor_input: str | Path | None = None,
) -> dict[str, object]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    base = _load_model_source(input_path, target_col)
    merged, join_diag = merge_factor_sidecar(base, factor_input)
    frame = add_derived_factor_features(merged, target_col)

    availability = build_factor_availability(frame)
    univariate = build_univariate_tests(frame, availability, target_col)
    comparison, permutation = build_factor_model_comparison(frame, target_col)
    evidence = build_evidence_matrix(availability, univariate, comparison)

    availability.to_csv(output / "factor_availability.csv", index=False)
    join_diag.to_csv(output / "factor_join_diagnostics.csv", index=False)
    univariate.to_csv(output / "factor_univariate_tests.csv", index=False)
    comparison.to_csv(output / "factor_group_model_comparison.csv", index=False)
    permutation.to_csv(output / "factor_permutation_importance.csv", index=False)
    evidence.to_csv(output / "factor_evidence_matrix.csv", index=False)
    write_recommendations_md(
        output / "factor_recommendations.md",
        join_diag,
        availability,
        evidence,
        comparison,
    )

    summary = {
        "target_col": target_col,
        "input_path": str(input_path),
        "factor_input": None if not factor_input else str(factor_input),
        "n_rows": int(len(frame)),
        "date_min": str(frame[DATE_COL].min().date()) if frame[DATE_COL].notna().any() else None,
        "date_max": str(frame[DATE_COL].max().date()) if frame[DATE_COL].notna().any() else None,
        "join_diagnostics": join_diag.to_dict(orient="records"),
        "availability_counts": availability["status"].value_counts().to_dict(),
        "evidence_counts": evidence["evidence_level"].value_counts().to_dict(),
        "output_files": {
            "availability": "factor_availability.csv",
            "join_diagnostics": "factor_join_diagnostics.csv",
            "univariate_tests": "factor_univariate_tests.csv",
            "group_model_comparison": "factor_group_model_comparison.csv",
            "permutation_importance": "factor_permutation_importance.csv",
            "evidence_matrix": "factor_evidence_matrix.csv",
            "recommendations": "factor_recommendations.md",
        },
        "notes": [
            "Missing sidecar fields are reported as data-collection needs, not as negative evidence.",
            "Proxy evidence should not be described as causal proof.",
            "Process-response features are excluded from feed phosphorus prediction models.",
        ],
    }
    with (output / "factor_validation_summary.json").open("w", encoding="utf-8") as f:
        json.dump(_json_safe(summary), f, ensure_ascii=False, indent=2, allow_nan=False)
    return summary


def parse_args():
    default_target_col = os.getenv("CPO_TARGET_COL", DEFAULT_TARGET_COL)
    default_factor_input = os.getenv("CPO_FACTOR_INPUT", "").strip()
    parser = argparse.ArgumentParser(description="Validate potential phosphorus influencing factors")
    parser.add_argument(
        "--input",
        default=str(LOCAL_PROCESSED_DATA_DIR / "model_source.csv"),
        help=f"Path to model_source.csv (default: {LOCAL_PROCESSED_DATA_DIR / 'model_source.csv'})",
    )
    parser.add_argument(
        "--factor-input",
        default=default_factor_input or None,
        help="Optional CSV/XLSX sidecar containing factor fields.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(LOCAL_FACTOR_VALIDATION_REPORTS_DIR),
        help=f"Directory for factor validation outputs (default: {LOCAL_FACTOR_VALIDATION_REPORTS_DIR})",
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
        factor_input=args.factor_input,
    )
    print(json.dumps(_json_safe(summary), ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
