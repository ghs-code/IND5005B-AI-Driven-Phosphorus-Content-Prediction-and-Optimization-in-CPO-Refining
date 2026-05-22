#!/usr/bin/env python3
"""
Multi-year preprocessing dispatcher for the CPO phosphorus project.

This module contains all business logic for splitting a multi-year raw dataset
into per-year preprocessing runs and an optional combined 'overall' run.

Public interface
----------------
main()          Entry point called by scripts/run_preprocessing_multi.py.
run_multi()     Programmatic entry point (accepts explicit arguments).

Dispatch logic
--------------
1. Load all raw Excel files from the supplied input path(s).
2. Detect which calendar years are present in the data.
3. Run the existing `data_processing.run_pipeline` independently for EACH
   year, writing outputs to:
       <processed_dir>/<year>/
       <report_dir>/<year>/
4. If MORE THAN ONE year is detected, run `run_pipeline` again on the full
   merged dataset and write outputs to:
       <processed_dir>/overall/
       <report_dir>/overall/
5. If only ONE year is detected, the 'overall' pass is skipped (it would be
   identical to the single-year run).

All output paths respect the project's existing archive-run convention
(CPO_ARCHIVE_RUNS / CPO_RUN_ROOT).  Pass --processed-dir and --report-dir
to override defaults, exactly as with the original run_preprocessing.py.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Support running this file directly during development
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from cpo_phosphorus.paths import (
    DEFAULT_RAW_EXCEL,
    LOCAL_PREPROCESSING_REPORTS_DIR,
    LOCAL_PROCESSED_DATA_DIR,
)
from cpo_phosphorus.pipelines.data_processing import (
    _get_env_float,
    _split_env_list,
    load_raw_inputs,
    run_pipeline,
)


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

_SEPARATOR = "=" * 64


def _print_banner(msg: str) -> None:
    print(f"\n{_SEPARATOR}")
    print(f"  {msg}")
    print(f"{_SEPARATOR}\n")


def _detect_years(input_paths: list[str], year_filter: str) -> list[int]:
    """Return a sorted list of unique calendar years present in the raw data."""
    df, _ = load_raw_inputs(input_paths, year_filter=year_filter)
    return sorted(df["date"].dt.year.unique().astype(int).tolist())


# ──────────────────────────────────────────────────────────────────────────────
# Core dispatcher
# ──────────────────────────────────────────────────────────────────────────────

def run_multi(
    input_paths: list[str],
    processed_dir: str,
    report_dir: str,
    target_col: str,
    vif_threshold: float,
    year_filter: str = "all",
) -> dict:
    """
    Run the preprocessing pipeline for each detected year and, when multiple
    years are present, for the full combined dataset.

    Parameters
    ----------
    input_paths   : List of paths to raw Excel files or directories.
    processed_dir : Base directory for processed CSV outputs.
                    Year sub-folders (e.g. 2024/, overall/) are appended.
    report_dir    : Base directory for preprocessing report outputs.
                    Year sub-folders are appended.
    target_col    : Name of the prediction-target column.
    vif_threshold : Severe VIF threshold for iterative feature removal.
    year_filter   : Pre-filter applied before year-splitting.
                    Pass 'all' to include every year found.

    Returns
    -------
    dict keyed by group label ('2024', '2025', 'overall', …), each value
    being the summary dict returned by `data_processing.run_pipeline`.
    """
    base_processed = Path(processed_dir)
    base_report = Path(report_dir)

    # ── Step 1: detect years ──────────────────────────────────────────────────
    _print_banner("STEP 1 - Detecting years in raw data")
    years = _detect_years(input_paths, year_filter=year_filter)
    print(f"  Years detected: {years}")
    multi_year = len(years) > 1

    all_summaries: dict[str, object] = {}

    # ── Step 2: per-year passes ───────────────────────────────────────────────
    for year in years:
        _print_banner(f"STEP 2 - Preprocessing year: {year}")

        year_processed_dir = base_processed / str(year)
        year_report_dir = base_report / str(year)

        print(f"  processed_dir : {year_processed_dir}")
        print(f"  report_dir    : {year_report_dir}\n")

        summary = run_pipeline(
            input_paths=input_paths,
            processed_dir=str(year_processed_dir),
            report_dir=str(year_report_dir),
            target_col=target_col,
            vif_threshold=vif_threshold,
            year_filter=str(year),
        )
        all_summaries[str(year)] = summary
        print(f"\n  [OK] Year {year} preprocessing complete.")

    # ── Step 3: overall pass (multi-year only) ────────────────────────────────
    if multi_year:
        _print_banner("STEP 3 - Preprocessing OVERALL (all years combined)")

        overall_processed_dir = base_processed / "overall"
        overall_report_dir = base_report / "overall"

        print(f"  processed_dir : {overall_processed_dir}")
        print(f"  report_dir    : {overall_report_dir}\n")

        summary = run_pipeline(
            input_paths=input_paths,
            processed_dir=str(overall_processed_dir),
            report_dir=str(overall_report_dir),
            target_col=target_col,
            vif_threshold=vif_threshold,
            year_filter="all",
        )
        all_summaries["overall"] = summary
        print("\n  [OK] Overall pass complete.")
    else:
        _print_banner(
            "STEP 3 - Skipped (only one year detected; overall = single year)"
        )

    # ── Final summary ─────────────────────────────────────────────────────────
    _print_banner("ALL DONE")
    groups = list(all_summaries.keys())
    print(f"  Groups processed: {groups}")
    for group, summary in all_summaries.items():
        rows = summary.get("rows_after", "?")
        date_min = summary.get("date_min", "?")
        date_max = summary.get("date_max", "?")
        print(f"    [{group}] rows={rows}  date={date_min} -> {date_max}")

    print(
        "\n  Full summaries written to each report_dir as "
        "preprocessing_summary.json\n"
    )

    return all_summaries


# ──────────────────────────────────────────────────────────────────────────────
# CLI argument parsing
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    """Parse CLI arguments, falling back to environment variables."""
    default_target_col = os.getenv("CPO_TARGET_COL", "feed_p_ppm")
    default_vif_threshold = _get_env_float("CPO_VIF_THRESHOLD", 10.0)
    default_raw_inputs = _split_env_list(os.getenv("CPO_RAW_INPUTS"))
    if not default_raw_inputs:
        raw_input_env = os.getenv("CPO_RAW_INPUT")
        default_raw_inputs = (
            [raw_input_env] if raw_input_env else [str(DEFAULT_RAW_EXCEL)]
        )
    default_year = os.getenv("CPO_YEAR", "all")
    default_processed_dir = str(LOCAL_PROCESSED_DATA_DIR)
    default_report_dir = str(LOCAL_PREPROCESSING_REPORTS_DIR)

    parser = argparse.ArgumentParser(
        description=(
            "Multi-year preprocessing dispatcher. "
            "Runs the preprocessing pipeline independently for each calendar "
            "year found in the data, plus an 'overall' pass when more than "
            "one year is present."
        )
    )
    parser.add_argument(
        "--input",
        action="append",
        default=None,
        metavar="PATH",
        help=(
            "Path to a raw Excel file or a directory containing raw Excel "
            "files. Can be passed multiple times. "
            f"Default: {default_raw_inputs}"
        ),
    )
    parser.add_argument(
        "--year",
        default=default_year,
        help=(
            "Year pre-filter applied before year-splitting. "
            "Use 'all' to include every year found. "
            f"Default: {default_year}"
        ),
    )
    parser.add_argument(
        "--processed-dir",
        default=default_processed_dir,
        metavar="DIR",
        help=(
            "Base directory for processed datasets. "
            "Year sub-folders are appended automatically. "
            f"Default: {default_processed_dir}"
        ),
    )
    parser.add_argument(
        "--report-dir",
        default=default_report_dir,
        metavar="DIR",
        help=(
            "Base directory for preprocessing reports. "
            "Year sub-folders are appended automatically. "
            f"Default: {default_report_dir}"
        ),
    )
    parser.add_argument(
        "--target-col",
        default=default_target_col,
        help=f"Target column kept in model-ready output. Default: {default_target_col}",
    )
    parser.add_argument(
        "--vif-threshold",
        type=float,
        default=default_vif_threshold,
        help=(
            "Severe VIF threshold for iterative feature removal. "
            f"Default: {default_vif_threshold}"
        ),
    )

    args = parser.parse_args()
    if args.input is None:
        args.input = default_raw_inputs
    return args


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """CLI entry point — called by scripts/run_preprocessing_multi.py."""
    args = parse_args()
    all_summaries = run_multi(
        input_paths=args.input,
        processed_dir=args.processed_dir,
        report_dir=args.report_dir,
        target_col=args.target_col,
        vif_threshold=args.vif_threshold,
        year_filter=args.year,
    )
    print(json.dumps({"groups": list(all_summaries.keys())}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
