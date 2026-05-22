#!/usr/bin/env python3
"""
Multi-year model dispatcher for the CPO phosphorus project.

This module wraps the existing single-run pipelines for OLS, ACF, and
Random Forest (full-feature) and adds a multi-group scanning loop:

1. Scans `<processed_dir>` for sub-folders produced by multi_year_dispatcher
   (e.g. 2024/, 2025/, overall/).
2. For each group folder that contains the expected input files, runs the
   corresponding analysis pipeline and writes results into the matching
   sub-folder under `<report_dir>`.

Directory layout after a successful run
----------------------------------------
local_runs/<run_id>/
├── processed/
│   ├── 2024/model_ready.csv, model_source.csv, model_ready_lag.csv, ...
│   ├── 2025/...
│   └── overall/...
└── reports/
    ├── ols/
    │   ├── 2024/OLSresult_*.csv, ...
    │   ├── 2025/...
    │   └── overall/...
    ├── rf/
    │   ├── 2024/rf_results.json, rf_feature_importance.*, ...
    │   ├── 2025/...
    │   └── overall/...
    └── acf/
        ├── 2024/acf_plot.jpg
        ├── 2025/...
        └── overall/...

Usage
-----
python scripts/run_models_multi.py
python scripts/run_models_multi.py \\
    --processed-dir local_data/processed \\
    --ols-report-dir  local_reports/ols \\
    --rf-report-dir   local_reports/rf \\
    --acf-report-dir  local_reports/acf \\
    --target-col feed_p_ppm
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from cpo_phosphorus.paths import (
    LOCAL_OLS_REPORTS_DIR,
    LOCAL_PROCESSED_DATA_DIR,
    LOCAL_RF_FULL_REPORTS_DIR,
)
from cpo_phosphorus.models.ols import run_pipeline as ols_run_pipeline
from cpo_phosphorus.models.acf_plot import run_plot as acf_run_plot
from cpo_phosphorus.models.random_forest_full import run_pipeline as rf_run_pipeline


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

_SEPARATOR = "=" * 64


def _print_banner(msg: str) -> None:
    print(f"\n{_SEPARATOR}")
    print(f"  {msg}")
    print(f"{_SEPARATOR}\n")


def _scan_groups(processed_base: Path) -> list[str]:
    """
    Return sorted group names (sub-folder names) found under processed_base.
    Year folders (e.g. '2024', '2025') come before 'overall'.
    """
    if not processed_base.exists():
        return []

    groups = []
    year_groups = []
    has_overall = False

    for child in sorted(processed_base.iterdir()):
        if not child.is_dir():
            continue
        if child.name == "overall":
            has_overall = True
        else:
            year_groups.append(child.name)

    groups = year_groups
    if has_overall:
        groups.append("overall")
    return groups


# ──────────────────────────────────────────────────────────────────────────────
# Per-model dispatchers
# ──────────────────────────────────────────────────────────────────────────────

def _run_ols_for_group(
    group: str,
    processed_base: Path,
    ols_report_base: Path,
    target_col: str,
) -> dict | None:
    """Run OLS pipeline for one group. Returns summary dict or None if skipped."""
    model_ready = processed_base / group / "model_ready.csv"
    if not model_ready.exists():
        print(f"  [SKIP] OLS/{group}: model_ready.csv not found at {model_ready}")
        return None

    group_processed_dir = processed_base / group
    group_report_dir = ols_report_base / group
    group_report_dir.mkdir(parents=True, exist_ok=True)

    print(f"  input         : {model_ready}")
    print(f"  report_dir    : {group_report_dir}\n")

    result = ols_run_pipeline(
        input_path=str(model_ready),
        processed_dir=str(group_processed_dir),
        report_dir=str(group_report_dir),
        target_col=target_col,
    )
    return result


def _run_acf_for_group(
    group: str,
    processed_base: Path,
    acf_report_base: Path,
    target_col: str,
) -> str | None:
    """Run ACF plot for one group. Returns output path string or None if skipped."""
    lag_csv = processed_base / group / "model_ready_lag.csv"
    if not lag_csv.exists():
        print(f"  [SKIP] ACF/{group}: model_ready_lag.csv not found at {lag_csv}")
        return None

    group_report_dir = acf_report_base / group
    group_report_dir.mkdir(parents=True, exist_ok=True)
    output_path = group_report_dir / "acf_plot.jpg"

    print(f"  input         : {lag_csv}")
    print(f"  output        : {output_path}\n")

    acf_run_plot(
        input_path=str(lag_csv),
        output_path=str(output_path),
        target_col=target_col,
    )
    return str(output_path)


def _run_rf_for_group(
    group: str,
    processed_base: Path,
    rf_report_base: Path,
    target_col: str,
) -> dict | None:
    """Run RF full pipeline for one group. Returns results dict or None if skipped."""
    model_source = processed_base / group / "model_source.csv"
    if not model_source.exists():
        print(f"  [SKIP] RF/{group}: model_source.csv not found at {model_source}")
        return None

    group_report_dir = rf_report_base / group
    group_report_dir.mkdir(parents=True, exist_ok=True)

    print(f"  input         : {model_source}")
    print(f"  report_dir    : {group_report_dir}\n")

    result = rf_run_pipeline(
        input_path=str(model_source),
        output_dir=str(group_report_dir),
        target_col=target_col,
    )
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Core multi-group runner
# ──────────────────────────────────────────────────────────────────────────────

def run_models_multi(
    processed_dir: str,
    ols_report_dir: str,
    rf_report_dir: str,
    acf_report_dir: str,
    target_col: str,
    run_ols: bool = True,
    run_acf: bool = True,
    run_rf: bool = True,
) -> dict:
    """
    Scan processed_dir for group sub-folders and run OLS, ACF, and RF for each.

    Parameters
    ----------
    processed_dir   : Root of processed data (contains 2024/, 2025/, overall/).
    ols_report_dir  : Root of OLS reports (group sub-folders created automatically).
    rf_report_dir   : Root of RF reports.
    acf_report_dir  : Root of ACF reports.
    target_col      : Prediction target column name.
    run_ols / run_acf / run_rf : Toggle individual model types.

    Returns
    -------
    dict with keys 'ols', 'acf', 'rf'; each maps group -> result (or None).
    """
    processed_base = Path(processed_dir)
    ols_report_base = Path(ols_report_dir)
    rf_report_base = Path(rf_report_dir)
    acf_report_base = Path(acf_report_dir)

    groups = _scan_groups(processed_base)
    if not groups:
        raise FileNotFoundError(
            f"No group sub-folders found under '{processed_base}'. "
            "Run run_preprocessing_multi.py first."
        )

    _print_banner(f"Groups found: {groups}")

    all_ols: dict = {}
    all_acf: dict = {}
    all_rf: dict = {}

    for group in groups:

        # ── OLS ──────────────────────────────────────────────────────────────
        if run_ols:
            _print_banner(f"OLS - group: {group}")
            all_ols[group] = _run_ols_for_group(
                group, processed_base, ols_report_base, target_col
            )
            if all_ols[group] is not None:
                print(f"  [OK] OLS/{group} complete.")

        # ── ACF ──────────────────────────────────────────────────────────────
        if run_acf:
            _print_banner(f"ACF - group: {group}")
            all_acf[group] = _run_acf_for_group(
                group, processed_base, acf_report_base, target_col
            )
            if all_acf[group] is not None:
                print(f"  [OK] ACF/{group} complete.")

        # ── RF ───────────────────────────────────────────────────────────────
        if run_rf:
            _print_banner(f"Random Forest (full) - group: {group}")
            all_rf[group] = _run_rf_for_group(
                group, processed_base, rf_report_base, target_col
            )
            if all_rf[group] is not None:
                print(f"  [OK] RF/{group} complete.")

    _print_banner("ALL MODELS DONE")
    print(f"  Groups processed: {groups}")
    for group in groups:
        ols_ok = all_ols.get(group) is not None if run_ols else "skipped"
        acf_ok = all_acf.get(group) is not None if run_acf else "skipped"
        rf_ok = all_rf.get(group) is not None if run_rf else "skipped"
        print(f"    [{group}] OLS={ols_ok}  ACF={acf_ok}  RF={rf_ok}")
    print()

    return {"ols": all_ols, "acf": all_acf, "rf": all_rf}


# ──────────────────────────────────────────────────────────────────────────────
# CLI argument parsing
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    default_target_col = os.getenv("CPO_TARGET_COL", "feed_p_ppm")
    default_processed_dir = str(LOCAL_PROCESSED_DATA_DIR)
    default_ols_report_dir = str(LOCAL_OLS_REPORTS_DIR)
    default_rf_report_dir = str(LOCAL_RF_FULL_REPORTS_DIR)
    default_acf_report_dir = str(LOCAL_OLS_REPORTS_DIR)  # ACF lives alongside OLS

    parser = argparse.ArgumentParser(
        description=(
            "Multi-group model dispatcher. "
            "Scans processed_dir sub-folders and runs OLS, ACF, and RF "
            "for each group (year + overall)."
        )
    )
    parser.add_argument(
        "--processed-dir",
        default=default_processed_dir,
        metavar="DIR",
        help=(
            "Root directory containing group sub-folders of processed data. "
            f"Default: {default_processed_dir}"
        ),
    )
    parser.add_argument(
        "--ols-report-dir",
        default=default_ols_report_dir,
        metavar="DIR",
        help=f"Root directory for OLS reports. Default: {default_ols_report_dir}",
    )
    parser.add_argument(
        "--rf-report-dir",
        default=default_rf_report_dir,
        metavar="DIR",
        help=f"Root directory for RF (full) reports. Default: {default_rf_report_dir}",
    )
    parser.add_argument(
        "--acf-report-dir",
        default=default_acf_report_dir,
        metavar="DIR",
        help=f"Root directory for ACF reports. Default: {default_acf_report_dir}",
    )
    parser.add_argument(
        "--target-col",
        default=default_target_col,
        help=f"Target column to predict. Default: {default_target_col}",
    )
    parser.add_argument(
        "--skip-ols",
        action="store_true",
        help="Skip OLS analysis.",
    )
    parser.add_argument(
        "--skip-acf",
        action="store_true",
        help="Skip ACF plot generation.",
    )
    parser.add_argument(
        "--skip-rf",
        action="store_true",
        help="Skip Random Forest analysis.",
    )
    return parser.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """CLI entry point — called by scripts/run_models_multi.py."""
    args = parse_args()
    results = run_models_multi(
        processed_dir=args.processed_dir,
        ols_report_dir=args.ols_report_dir,
        rf_report_dir=args.rf_report_dir,
        acf_report_dir=args.acf_report_dir,
        target_col=args.target_col,
        run_ols=not args.skip_ols,
        run_acf=not args.skip_acf,
        run_rf=not args.skip_rf,
    )
    groups = list({g for d in results.values() for g in d})
    print(json.dumps({"groups": sorted(groups)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
