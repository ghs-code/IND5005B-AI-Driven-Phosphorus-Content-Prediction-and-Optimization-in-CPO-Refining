"""Data-intake validation for enterprise quality-table inputs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from cpo_phosphorus.pipelines.data_processing import (
    RAW_COLUMNS,
    load_raw_inputs,
    resolve_raw_input_paths,
)
from cpo_phosphorus.workflows.core_factors import CORE_QUALITY_FIELDS
from cpo_phosphorus.workflows.common import parse_csv_list

EXCEL_SUFFIXES = {".xlsx", ".xlsm", ".xls"}
PROCESS_RESPONSE_FIELDS = ["acid_dosing_pct", "bleaching_earth_dosing_pct", "rbd_p_ppm"]
DEFAULT_JOIN_KEYS = ["date", "feed_tank", "source_file"]


def _sheet_template_summary(path: Path, sheet_name: str, target_col: str) -> dict:
    raw = pd.read_excel(path, sheet_name=sheet_name, header=None)
    row_count, col_count = raw.shape
    has_template_block = row_count > 4 and col_count >= 21
    summary = {
        "sheet_name": sheet_name,
        "row_count": int(row_count),
        "column_count": int(col_count),
        "has_expected_b_to_u_block": bool(has_template_block),
        "data_rows": 0,
        "valid_date_rows": 0,
        "valid_target_rows": 0,
        "years_present": [],
        "issues": [],
    }

    if not has_template_block:
        summary["issues"].append(
            "Sheet does not contain the expected data block at rows 5+ and columns B:U."
        )
        return summary

    block = raw.iloc[4:, 1:21].copy()
    block.columns = RAW_COLUMNS
    dates = pd.to_datetime(block["date"], errors="coerce")
    block = block[dates.notna()].copy()
    dates = dates.loc[block.index]
    summary["data_rows"] = int(len(block))
    summary["valid_date_rows"] = int(dates.notna().sum())
    summary["years_present"] = sorted(dates.dt.year.dropna().astype(int).unique().tolist())

    if target_col in block.columns:
        summary["valid_target_rows"] = int(pd.to_numeric(block[target_col], errors="coerce").notna().sum())
    else:
        summary["issues"].append(f"Target column '{target_col}' is not part of the quality-table template.")

    if summary["valid_date_rows"] == 0:
        summary["issues"].append("No valid dates were found in the expected date column.")
    if target_col in block.columns and summary["valid_target_rows"] == 0:
        summary["issues"].append(f"No numeric values were found for target column '{target_col}'.")

    return summary


def _excel_metadata(path: Path, target_col: str) -> dict:
    xls = pd.ExcelFile(path)
    sheet_summaries = [_sheet_template_summary(path, sheet, target_col) for sheet in xls.sheet_names]
    issue_count = sum(len(sheet["issues"]) for sheet in sheet_summaries)
    years = sorted({year for sheet in sheet_summaries for year in sheet["years_present"]})
    return {
        "path": str(path),
        "file_name": path.name,
        "suffix": path.suffix.lower(),
        "sheet_count": len(xls.sheet_names),
        "sheets": xls.sheet_names,
        "years_present": years,
        "data_rows": int(sum(sheet["data_rows"] for sheet in sheet_summaries)),
        "valid_target_rows": int(sum(sheet["valid_target_rows"] for sheet in sheet_summaries)),
        "issue_count": int(issue_count),
        "sheet_summaries": sheet_summaries,
    }


def _field_mapping_summary(columns, target_col, join_keys, factor_fields, process_fields):
    available = set(columns)
    requested_join_keys = join_keys or [key for key in DEFAULT_JOIN_KEYS if key in available]
    requested_factor_fields = factor_fields or [field for field in CORE_QUALITY_FIELDS if field in available]
    requested_process_fields = process_fields or [field for field in PROCESS_RESPONSE_FIELDS if field in available]

    return {
        "target_col": target_col,
        "target_available": target_col in available,
        "join_keys": requested_join_keys,
        "available_join_keys": [key for key in requested_join_keys if key in available],
        "missing_join_keys": [key for key in requested_join_keys if key not in available],
        "factor_fields": requested_factor_fields,
        "available_factor_fields": [field for field in requested_factor_fields if field in available],
        "missing_factor_fields": [field for field in requested_factor_fields if field not in available],
        "process_response_fields": requested_process_fields,
        "available_process_response_fields": [
            field for field in requested_process_fields if field in available
        ],
        "missing_process_response_fields": [
            field for field in requested_process_fields if field not in available
        ],
    }


def run_data_validation(
    input_paths,
    year_filter="all",
    target_col="feed_p_ppm",
    join_keys=None,
    factor_fields=None,
    process_response_fields=None,
):
    """Validate quality-table files against the current template."""
    if isinstance(input_paths, (str, Path)):
        input_paths = [str(input_paths)]
    input_paths = [str(path) for path in input_paths if str(path).strip()]
    if not input_paths:
        raise ValueError("At least one quality-table input path is required.")

    files = resolve_raw_input_paths(input_paths)
    file_summaries = [_excel_metadata(Path(path), target_col) for path in files]
    suffixes = sorted({item["suffix"] for item in file_summaries})
    issue_count = sum(item["issue_count"] for item in file_summaries)

    quality_df, input_metadata = load_raw_inputs(input_paths, year_filter=year_filter)
    years = sorted(quality_df["date"].dt.year.dropna().astype(int).unique().tolist())
    date_min = quality_df["date"].min()
    date_max = quality_df["date"].max()
    valid_target_rows = (
        int(pd.to_numeric(quality_df[target_col], errors="coerce").notna().sum())
        if target_col in quality_df.columns
        else 0
    )
    mapping = _field_mapping_summary(
        quality_df.columns,
        target_col=target_col,
        join_keys=parse_csv_list(join_keys),
        factor_fields=parse_csv_list(factor_fields),
        process_fields=parse_csv_list(process_response_fields),
    )
    status = "ready" if issue_count == 0 and mapping["target_available"] and valid_target_rows > 0 else "needs_attention"

    return {
        "workflow": "validate_data",
        "status": status,
        "milestone": 2,
        "requested_inputs": input_paths,
        "requested_year_filter": year_filter,
        "target_col": target_col,
        "file_count": len(file_summaries),
        "suffixes": suffixes,
        "date_min": None if pd.isna(date_min) else str(date_min.date()),
        "date_max": None if pd.isna(date_max) else str(date_max.date()),
        "years_present": years,
        "row_count": int(len(quality_df)),
        "valid_target_rows": valid_target_rows,
        "input_metadata": input_metadata,
        "field_mapping": mapping,
        "template_issue_count": int(issue_count),
        "files": file_summaries,
        "notes": [
            "Milestone 2 validates the current quality-table template and field mapping readiness.",
            "Join diagnostics and direct internal-factor validation are implemented in later milestones.",
        ],
    }
