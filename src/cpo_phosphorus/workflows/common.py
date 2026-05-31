"""Shared helpers for modular enterprise workflows."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def json_safe(value):
    """Convert common numpy/pandas values into JSON-serializable values."""
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if pd.isna(value) or not np.isfinite(value) else float(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if value is pd.NA or value is pd.NaT:
        return None
    return value


def print_json(summary) -> None:
    print(json.dumps(json_safe(summary), ensure_ascii=False, indent=2))


def read_table(path):
    """Read a CSV or Excel table for workflow-level checks."""
    table_path = Path(path).expanduser()
    if not table_path.exists():
        raise FileNotFoundError(f"Input table does not exist: {table_path}")

    suffix = table_path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(table_path)
    if suffix in {".xlsx", ".xlsm", ".xls"}:
        return pd.read_excel(table_path)

    raise ValueError(f"Unsupported table format: {table_path.suffix}")


def read_tables(paths):
    """Read one or more CSV/Excel files or directories into one table."""
    if isinstance(paths, (str, Path)):
        paths = [paths]

    files = []
    for item in paths:
        if not item:
            continue
        path = Path(item).expanduser()
        if path.is_dir():
            files.extend(
                sorted(
                    child
                    for child in path.iterdir()
                    if child.is_file() and child.suffix.lower() in {".csv", ".xlsx", ".xlsm", ".xls"}
                )
            )
        else:
            files.append(path)

    if not files:
        raise ValueError("At least one input table is required.")

    frames = []
    for file_path in files:
        frame = read_table(file_path)
        frame["_source_table"] = Path(file_path).name
        frames.append(frame)
    return pd.concat(frames, ignore_index=True), [str(path) for path in files]


def parse_csv_list(value):
    if not value:
        return []
    return [item.strip() for item in str(value).split(",") if item.strip()]


def prepare_join_columns(df, join_keys):
    """Normalize common join columns without mutating the caller's frame."""
    data = df.copy()
    for key in join_keys:
        if key not in data.columns:
            continue
        if key == "date" or key.endswith("_date"):
            data[key] = pd.to_datetime(data[key], errors="coerce").dt.strftime("%Y-%m-%d")
        else:
            data[key] = data[key].astype("string").str.strip()
    return data


def numeric_series(df, column):
    return pd.to_numeric(df[column], errors="coerce")


def write_dataframe(df, output_dir, filename):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    path = output / filename
    df.to_csv(path, index=False)
    return str(path)
