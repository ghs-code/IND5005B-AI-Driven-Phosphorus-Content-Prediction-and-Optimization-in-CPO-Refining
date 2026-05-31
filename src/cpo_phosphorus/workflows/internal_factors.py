"""Direct enterprise internal-factor validation workflow."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from cpo_phosphorus.workflows.common import (
    numeric_series,
    parse_csv_list,
    prepare_join_columns,
    read_table,
    read_tables,
    write_dataframe,
)

DEFAULT_FACTOR_EXCLUDE = {"date", "target", "prediction", "predicted", "actual"}


def _one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _infer_factor_fields(df, join_keys, target_col):
    excluded = set(join_keys) | {target_col} | DEFAULT_FACTOR_EXCLUDE
    return [col for col in df.columns if col not in excluded]


def _date_offset_diagnostics(quality_df, internal_df, merged):
    if "date" not in quality_df.columns or "date" not in internal_df.columns:
        return {}
    quality_dates = pd.to_datetime(quality_df["date"], errors="coerce").dropna()
    internal_dates = pd.to_datetime(internal_df["date"], errors="coerce").dropna()
    if quality_dates.empty or internal_dates.empty:
        return {}

    overlap_start = max(quality_dates.min(), internal_dates.min())
    overlap_end = min(quality_dates.max(), internal_dates.max())
    overlap_days = int((overlap_end - overlap_start).days + 1) if overlap_end >= overlap_start else 0

    unmatched_offsets = []
    if "_join_status" in merged.columns:
        unmatched = merged.loc[merged["_join_status"].eq("left_only"), "date"]
        unmatched_dates = pd.to_datetime(unmatched, errors="coerce").dropna()
        internal_unique = internal_dates.sort_values().drop_duplicates().to_numpy()
        for date in unmatched_dates:
            deltas = np.abs((internal_unique - np.datetime64(date)).astype("timedelta64[D]").astype(int))
            if len(deltas):
                unmatched_offsets.append(int(deltas.min()))

    return {
        "quality_date_min": str(quality_dates.min().date()),
        "quality_date_max": str(quality_dates.max().date()),
        "internal_date_min": str(internal_dates.min().date()),
        "internal_date_max": str(internal_dates.max().date()),
        "date_overlap_days": overlap_days,
        "unmatched_nearest_date_p50_days": (
            float(np.median(unmatched_offsets)) if unmatched_offsets else None
        ),
        "unmatched_nearest_date_max_days": max(unmatched_offsets) if unmatched_offsets else None,
    }


def _field_coverage(data, fields):
    rows = []
    for field in fields:
        if field not in data.columns:
            rows.append({"field": field, "available": False, "non_null_rate": 0.0, "non_null_count": 0})
            continue
        non_null = int(data[field].notna().sum())
        rows.append(
            {
                "field": field,
                "available": True,
                "non_null_rate": float(non_null / len(data)) if len(data) else 0.0,
                "non_null_count": non_null,
            }
        )
    return rows


def _join_diagnostics(quality_df, internal_df, join_keys, factor_fields=None):
    if not join_keys:
        diagnostics = {
            "status": "missing_join_keys",
            "quality_rows": int(len(quality_df)),
            "internal_rows": int(len(internal_df)),
            "matched_rows": 0,
            "match_rate": 0.0,
            "duplicate_internal_key_rows": 0,
            "duplicate_quality_key_rows": 0,
            "unmatched_quality_rows": int(len(quality_df)),
            "field_coverage": _field_coverage(internal_df, factor_fields or []),
        }
        diagnostics.update(_date_offset_diagnostics(quality_df, internal_df, quality_df.assign(_join_status="left_only")))
        return diagnostics, quality_df.assign(_join_status="left_only")

    quality = prepare_join_columns(quality_df, join_keys)
    internal = prepare_join_columns(internal_df, join_keys)
    duplicate_internal = int(internal.duplicated(join_keys, keep=False).sum())
    duplicate_quality = int(quality.duplicated(join_keys, keep=False).sum())
    merged = quality.merge(
        internal,
        on=join_keys,
        how="left",
        suffixes=("", "_internal"),
        indicator="_join_status",
    )
    matched = int(merged["_join_status"].eq("both").sum())
    quality_rows = int(len(quality))
    diagnostics = {
        "status": "joined",
        "quality_rows": quality_rows,
        "internal_rows": int(len(internal)),
        "matched_rows": matched,
        "match_rate": float(matched / quality_rows) if quality_rows else 0.0,
        "duplicate_internal_key_rows": duplicate_internal,
        "duplicate_quality_key_rows": duplicate_quality,
        "unmatched_quality_rows": int(merged["_join_status"].eq("left_only").sum()),
        "field_coverage": _field_coverage(merged.loc[merged["_join_status"].eq("both")], factor_fields or []),
    }
    diagnostics.update(_date_offset_diagnostics(quality_df, internal_df, merged))
    if diagnostics["match_rate"] < 0.5:
        diagnostics["status"] = "low_match_rate"
    if duplicate_internal > 0:
        diagnostics["status"] = "duplicate_internal_keys"
    return diagnostics, merged


def _single_factor_model_r2(data, field, target_col):
    frame = data[[field, target_col]].copy().dropna(subset=[target_col])
    frame[target_col] = pd.to_numeric(frame[target_col], errors="coerce")
    frame = frame.dropna(subset=[target_col])
    if len(frame) < 30 or frame[field].nunique(dropna=True) < 2:
        return None
    numeric = pd.to_numeric(frame[field], errors="coerce").notna().mean() >= 0.8
    if numeric:
        frame[field] = pd.to_numeric(frame[field], errors="coerce")
        transformer = ColumnTransformer(
            transformers=[("numeric", SimpleImputer(strategy="median"), [field])]
        )
    else:
        transformer = ColumnTransformer(
            transformers=[
                (
                    "categorical",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="constant", fill_value="UNKNOWN")),
                            ("onehot", _one_hot_encoder()),
                        ]
                    ),
                    [field],
                )
            ]
        )
    train, test = train_test_split(frame, test_size=0.25, random_state=42)
    if len(test) < 10:
        return None
    model = Pipeline(
        steps=[
            ("prep", transformer),
            ("model", RandomForestRegressor(n_estimators=100, max_depth=3, random_state=42)),
        ]
    )
    model.fit(train[[field]], train[target_col])
    pred = model.predict(test[[field]])
    return float(r2_score(test[target_col], pred))


def _numeric_factor_evidence(data, field, target_col):
    x = numeric_series(data, field)
    y = numeric_series(data, target_col)
    frame = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(frame) < 10 or frame["x"].nunique() < 2:
        return {"method": "numeric", "n": int(len(frame)), "evidence_level": "not_assessable"}
    corr, p_value = stats.spearmanr(frame["x"], frame["y"])
    abs_corr = abs(float(corr))
    if p_value < 0.05 and abs_corr >= 0.3:
        level = "direct_supported"
    elif p_value < 0.1 and abs_corr >= 0.15:
        level = "weak_or_context_dependent"
    else:
        level = "not_supported_current_data"
    return {
        "method": "numeric",
        "n": int(len(frame)),
        "spearman_corr": float(corr),
        "p_value": float(p_value),
        "effect_size": abs_corr,
        "single_factor_test_r2": _single_factor_model_r2(data, field, target_col),
        "evidence_level": level,
    }


def _categorical_factor_evidence(data, field, target_col):
    frame = data[[field, target_col]].copy()
    frame[target_col] = pd.to_numeric(frame[target_col], errors="coerce")
    frame = frame.dropna()
    if len(frame) < 10 or frame[field].nunique() < 2:
        return {"method": "categorical", "n": int(len(frame)), "evidence_level": "not_assessable"}

    groups = [group[target_col].to_numpy() for _, group in frame.groupby(field) if len(group) >= 2]
    if len(groups) < 2:
        return {"method": "categorical", "n": int(len(frame)), "evidence_level": "not_assessable"}
    try:
        stat, p_value = stats.kruskal(*groups)
    except ValueError:
        p_value = 1.0
    means = frame.groupby(field)[target_col].mean()
    spread = float(means.max() - means.min()) if not means.empty else 0.0
    target_std = float(frame[target_col].std()) if frame[target_col].std() else 0.0
    effect = spread / target_std if target_std else 0.0
    if p_value < 0.05 and effect >= 0.3:
        level = "direct_supported"
    elif p_value < 0.1 and effect >= 0.15:
        level = "weak_or_context_dependent"
    else:
        level = "not_supported_current_data"
    return {
        "method": "categorical",
        "n": int(len(frame)),
        "p_value": float(p_value),
        "effect_size": float(effect),
        "group_count": int(frame[field].nunique()),
        "single_factor_test_r2": _single_factor_model_r2(data, field, target_col),
        "evidence_level": level,
    }


def _is_process_response_field(field):
    text = field.lower()
    return any(token in text for token in ["acid", "bleach", "dosing", "temperature", "pressure", "flow", "vacuum"])


def _factor_evidence(data, fields, target_col, process_fields):
    rows = []
    for field in fields:
        if field not in data.columns:
            rows.append(
                {
                    "factor": field,
                    "method": "missing",
                    "evidence_level": "not_assessable",
                    "reason": "field_not_found",
                }
            )
            continue
        series = data[field]
        if pd.api.types.is_numeric_dtype(series) or pd.to_numeric(series, errors="coerce").notna().mean() >= 0.8:
            evidence = _numeric_factor_evidence(data, field, target_col)
        else:
            evidence = _categorical_factor_evidence(data, field, target_col)
        if field in process_fields or _is_process_response_field(field):
            if evidence["evidence_level"] == "direct_supported":
                evidence["evidence_level"] = "observed_process_response"
        evidence["factor"] = field
        evidence["reason"] = ""
        rows.append(evidence)
    return rows


def _incremental_model_lift(data, factor_fields, target_col):
    usable = [field for field in factor_fields if field in data.columns]
    if not usable or target_col not in data.columns:
        return None
    frame = data[usable + [target_col]].copy()
    frame[target_col] = pd.to_numeric(frame[target_col], errors="coerce")
    frame = frame.dropna(subset=[target_col])
    if len(frame) < 40:
        return None

    numeric = [col for col in usable if pd.to_numeric(frame[col], errors="coerce").notna().mean() >= 0.8]
    categorical = [col for col in usable if col not in numeric]
    for col in numeric:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")

    transformers = []
    if numeric:
        transformers.append(("numeric", SimpleImputer(strategy="median"), numeric))
    if categorical:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="constant", fill_value="UNKNOWN")),
                        ("onehot", _one_hot_encoder()),
                    ]
                ),
                categorical,
            )
        )
    if not transformers:
        return None
    model = Pipeline(
        steps=[
            ("prep", ColumnTransformer(transformers=transformers)),
            ("model", RandomForestRegressor(n_estimators=100, max_depth=4, random_state=42)),
        ]
    )
    train, test = train_test_split(frame, test_size=0.25, random_state=42)
    if len(test) < 10:
        return None
    baseline_pred = np.repeat(train[target_col].mean(), len(test))
    model.fit(train[usable], train[target_col])
    pred = model.predict(test[usable])
    return {
        "factor_field_count": len(usable),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "baseline_r2": float(r2_score(test[target_col], baseline_pred)),
        "factor_model_r2": float(r2_score(test[target_col], pred)),
    }


def _basic_factor_level(data, field, target_col):
    if field not in data.columns or target_col not in data.columns:
        return "not_assessable"
    if pd.to_numeric(data[field], errors="coerce").notna().mean() >= 0.8:
        return _numeric_factor_evidence(data, field, target_col).get("evidence_level")
    return _categorical_factor_evidence(data, field, target_col).get("evidence_level")


def _stability_by_group(data, factor_fields, target_col, group_col):
    if group_col not in data.columns:
        return []
    rows = []
    for group_value, group in data.groupby(group_col):
        if len(group) < 20:
            continue
        for field in factor_fields:
            rows.append(
                {
                    "group_column": group_col,
                    "group_value": str(group_value),
                    "factor": field,
                    "n": int(len(group)),
                    "evidence_level": _basic_factor_level(group, field, target_col),
                }
            )
    return rows


def _stability_summary(data, factor_fields, target_col):
    rows = []
    if "date" in data.columns:
        year_data = data.copy()
        year_data["__year"] = pd.to_datetime(year_data["date"], errors="coerce").dt.year
        rows.extend(_stability_by_group(year_data, factor_fields, target_col, "__year"))
    for candidate in ["source", "origin", "source_file", "_source_table"]:
        if candidate in data.columns:
            rows.extend(_stability_by_group(data, factor_fields, target_col, candidate))
            break
    return rows


def _risk_classification_gain(data, factor_fields, target_col):
    usable = [field for field in factor_fields if field in data.columns]
    if not usable or target_col not in data.columns:
        return None
    frame = data[usable + [target_col]].copy()
    frame[target_col] = pd.to_numeric(frame[target_col], errors="coerce")
    frame = frame.dropna(subset=[target_col])
    if len(frame) < 40:
        return None
    threshold = float(frame[target_col].quantile(0.8))
    y_risk = frame[target_col].ge(threshold)
    if y_risk.nunique() < 2:
        return None

    numeric = [col for col in usable if pd.to_numeric(frame[col], errors="coerce").notna().mean() >= 0.8]
    categorical = [col for col in usable if col not in numeric]
    for col in numeric:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    transformers = []
    if numeric:
        transformers.append(("numeric", SimpleImputer(strategy="median"), numeric))
    if categorical:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="constant", fill_value="UNKNOWN")),
                        ("onehot", _one_hot_encoder()),
                    ]
                ),
                categorical,
            )
        )
    if not transformers:
        return None
    train, test = train_test_split(frame, test_size=0.25, random_state=42, stratify=y_risk)
    model = Pipeline(
        steps=[
            ("prep", ColumnTransformer(transformers=transformers)),
            ("model", RandomForestRegressor(n_estimators=100, max_depth=4, random_state=42)),
        ]
    )
    model.fit(train[usable], train[target_col])
    pred = model.predict(test[usable])
    actual_high = test[target_col].ge(threshold)
    predicted_high = pd.Series(pred, index=test.index).ge(threshold)
    positives = int(actual_high.sum())
    recall = float((actual_high & predicted_high).sum() / positives) if positives else None
    baseline_high = pd.Series(False, index=test.index)
    baseline_recall = float((actual_high & baseline_high).sum() / positives) if positives else None
    return {
        "threshold_name": "p80",
        "threshold_ppm": threshold,
        "n_test": int(len(test)),
        "actual_positive_count": positives,
        "factor_model_recall": recall,
        "baseline_recall": baseline_recall,
        "recall_gain_vs_baseline": None if recall is None or baseline_recall is None else recall - baseline_recall,
    }


def run_internal_factor_check(
    input_path=None,
    join_keys=None,
    target_col="feed_p_ppm",
    quality_input=None,
    factor_fields=None,
    process_response_fields=None,
    output_dir=None,
):
    """Validate enterprise internal factors directly against the target."""
    keys = parse_csv_list(join_keys)
    if not input_path:
        return {
            "workflow": "internal_factors",
            "status": "not_configured",
            "milestone": 5,
            "target_col": target_col,
            "join_keys": keys,
            "notes": [
                "No internal factor table was provided.",
                "Provide an enterprise internal factor table to run direct validation.",
            ],
        }

    if isinstance(input_path, (str, Path)):
        input_paths = [input_path]
    else:
        input_paths = list(input_path)
    internal_df, internal_files = read_tables(input_paths)
    columns = set(internal_df.columns)
    available_join_keys = [key for key in keys if key in columns]
    missing_join_keys = [key for key in keys if key not in columns]
    fields = parse_csv_list(factor_fields)
    process_fields = parse_csv_list(process_response_fields)
    if not fields:
        fields = _infer_factor_fields(internal_df, keys, target_col)

    if not quality_input:
        status = "ready_for_mapping" if keys and not missing_join_keys else "needs_mapping"
        return {
            "workflow": "internal_factors",
            "status": status,
            "milestone": 3,
            "input_path": internal_files[0] if len(internal_files) == 1 else internal_files,
            "target_col": target_col,
            "row_count": int(len(internal_df)),
            "column_count": int(len(internal_df.columns)),
            "join_keys": keys,
            "available_join_keys": available_join_keys,
            "missing_join_keys": missing_join_keys,
            "factor_fields": fields,
            "sample_columns": list(internal_df.columns[:20]),
            "notes": [
                "Internal table readiness was checked.",
                "Pass a quality input table to run join diagnostics and direct validation.",
            ],
        }

    quality_df = read_table(quality_input)
    quality_missing_keys = [key for key in keys if key not in quality_df.columns]
    if missing_join_keys or quality_missing_keys:
        diagnostics = {
            "status": "missing_join_keys",
            "missing_internal_join_keys": missing_join_keys,
            "missing_quality_join_keys": quality_missing_keys,
        }
        merged = quality_df.assign(_join_status="not_joined")
    else:
        diagnostics, merged = _join_diagnostics(quality_df, internal_df, keys, factor_fields=fields)

    validation_frame = merged[merged["_join_status"].eq("both")].copy() if "_join_status" in merged.columns else merged
    if target_col not in validation_frame.columns:
        evidence = [
            {
                "factor": field,
                "method": "missing_target",
                "evidence_level": "not_assessable",
                "reason": f"target column '{target_col}' was not found after join",
            }
            for field in fields
        ]
    elif diagnostics.get("match_rate", 0.0) < 0.5:
        evidence = [
            {
                "factor": field,
                "method": "join_blocked",
                "evidence_level": "not_assessable",
                "reason": "join match rate is below 50%",
            }
            for field in fields
        ]
    else:
        evidence = _factor_evidence(validation_frame, fields, target_col, process_fields)

    evidence_df = pd.DataFrame(evidence)
    output_files = {}
    if output_dir:
        output_files["internal_factor_evidence"] = write_dataframe(
            evidence_df, output_dir, "internal_factor_evidence.csv"
        )
        output_files["joined_sample"] = write_dataframe(
            validation_frame.head(500), output_dir, "internal_factor_joined_sample.csv"
        )

    evidence_counts = evidence_df["evidence_level"].value_counts().to_dict() if not evidence_df.empty else {}
    model_lift = (
        _incremental_model_lift(validation_frame, fields, target_col)
        if diagnostics.get("match_rate", 0.0) >= 0.5 and target_col in validation_frame.columns
        else None
    )
    stability = (
        _stability_summary(validation_frame, fields, target_col)
        if diagnostics.get("match_rate", 0.0) >= 0.5 and target_col in validation_frame.columns
        else []
    )
    risk_gain = (
        _risk_classification_gain(validation_frame, fields, target_col)
        if diagnostics.get("match_rate", 0.0) >= 0.5 and target_col in validation_frame.columns
        else None
    )
    status = "validated" if diagnostics.get("status") == "joined" else "needs_attention"

    return {
        "workflow": "internal_factors",
        "status": status,
        "milestone": 5,
        "input_path": internal_files[0] if len(internal_files) == 1 else internal_files,
        "quality_input": str(quality_input),
        "target_col": target_col,
        "row_count": int(len(internal_df)),
        "column_count": int(len(internal_df.columns)),
        "join_keys": keys,
        "available_join_keys": available_join_keys,
        "missing_join_keys": missing_join_keys,
        "factor_fields": fields,
        "process_response_fields": process_fields,
        "join_diagnostics": diagnostics,
        "evidence_counts": evidence_counts,
        "evidence": evidence,
        "incremental_model_lift": model_lift,
        "stability": stability,
        "risk_classification_gain": risk_gain,
        "output_files": output_files,
        "sample_columns": list(internal_df.columns[:20]),
        "notes": [
            "Direct validation reports association and model lift, not causality.",
            "Low match rate or duplicate keys block strong factor conclusions.",
        ],
    }
