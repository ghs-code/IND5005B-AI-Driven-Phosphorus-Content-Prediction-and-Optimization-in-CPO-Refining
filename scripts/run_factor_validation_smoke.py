from pathlib import Path
import argparse
import sys
import tempfile

import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cpo_phosphorus.pipelines.factor_validation import (
    PREDICTION_EXCLUDED_COLUMNS,
    PREDICTION_EXCLUDED_PREFIXES,
    add_derived_factor_features,
    build_factor_feature_groups,
    merge_factor_sidecar,
)


def _assert(condition, message):
    if not condition:
        raise AssertionError(message)


def _write_csv(df, path):
    df.to_csv(path, index=False)
    return path


def _load_base(input_path, target_col):
    data = pd.read_csv(input_path, parse_dates=["date"])
    required = {"date", "feed_tank", target_col}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Smoke input is missing required columns: {sorted(missing)}")
    data = data[data["date"].notna() & data["feed_tank"].notna()].head(80).copy()
    if len(data) < 20:
        raise ValueError("Smoke input needs at least 20 rows with date and feed_tank.")
    return data


def _check_no_prediction_leakage(frame):
    groups = build_factor_feature_groups(frame)
    for group in groups:
        features = set(group.numeric_features) | set(group.categorical_features)
        blocked = features & PREDICTION_EXCLUDED_COLUMNS
        prefixed = [
            feature
            for feature in features
            if any(feature.startswith(prefix) for prefix in PREDICTION_EXCLUDED_PREFIXES)
        ]
        _assert(not blocked, f"Leakage-prone features entered {group.name}: {sorted(blocked)}")
        _assert(not prefixed, f"RBD-prefixed features entered {group.name}: {sorted(prefixed)}")


def run_smoke(input_path, target_col):
    base = _load_base(input_path, target_col)
    with tempfile.TemporaryDirectory(prefix="phosphor_factor_smoke_") as tmp:
        tmp_dir = Path(tmp)

        date_tank = base[["date", "feed_tank"]].head(30).copy()
        date_tank["mill_source"] = ["MILL_A" if i % 2 else "MILL_B" for i in range(len(date_tank))]
        date_tank["rainfall_mm"] = [float(i % 5) for i in range(len(date_tank))]
        date_tank["storage_start_ts"] = date_tank["date"] - pd.to_timedelta(2, unit="h")
        date_tank["storage_end_ts"] = date_tank["date"] + pd.to_timedelta(3, unit="h")
        date_tank["source_count_in_tank"] = [1 + (i % 3 == 0) for i in range(len(date_tank))]
        date_tank_path = _write_csv(date_tank, tmp_dir / "date_tank.csv")
        merged, diag = merge_factor_sidecar(base, date_tank_path)
        _assert(diag.loc[0, "join_key"] == "date|feed_tank", "date+feed_tank join not selected")
        _assert(len(merged) == len(base), "date+feed_tank join changed row count")
        derived = add_derived_factor_features(merged, target_col)
        _assert("storage_hours" in derived.columns, "storage_hours was not derived")
        _assert(derived["storage_hours"].dropna().eq(5).all(), "storage_hours derivation mismatch")
        _check_no_prediction_leakage(derived)

        date_only = base[["date"]].drop_duplicates().head(30).copy()
        date_only["ambient_temperature_c"] = [28 + (i % 4) for i in range(len(date_only))]
        date_only_path = _write_csv(date_only, tmp_dir / "date_only.csv")
        _, diag = merge_factor_sidecar(base, date_only_path)
        _assert(diag.loc[0, "join_key"] == "date", "date-only join not selected")

        batch_base = base.head(30).copy()
        batch_base["batch_id"] = [f"B{i:03d}" for i in range(len(batch_base))]
        batch_sidecar = batch_base[["batch_id"]].copy()
        batch_sidecar["fruit_origin"] = ["ORIGIN_A" if i % 2 else "ORIGIN_B" for i in range(len(batch_sidecar))]
        batch_sidecar_path = _write_csv(batch_sidecar, tmp_dir / "batch.csv")
        _, diag = merge_factor_sidecar(batch_base, batch_sidecar_path)
        _assert(diag.loc[0, "join_key"] == "batch_id", "batch_id join not selected")

        duplicate_sidecar = pd.concat([date_tank.head(2), date_tank.head(1)], ignore_index=True)
        duplicate_path = _write_csv(duplicate_sidecar, tmp_dir / "duplicate.csv")
        try:
            merge_factor_sidecar(base, duplicate_path)
        except ValueError as exc:
            _assert("duplicate join keys" in str(exc), "duplicate-key error message changed")
        else:
            raise AssertionError("Duplicate sidecar keys did not raise ValueError")

    print("factor validation smoke checks passed")


def parse_args():
    parser = argparse.ArgumentParser(description="Run lightweight factor-validation smoke checks.")
    parser.add_argument("--input", required=True, help="Path to model_source.csv")
    parser.add_argument("--target-col", default="feed_p_ppm", help="Target column name")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_smoke(args.input, args.target_col)
