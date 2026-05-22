#!/usr/bin/env python3
"""
Cross-year EDA analysis module for the CPO phosphorus project.

This module generates the visualisations and comparison tables required by
Section 2 of the final report:

  2.1  Dataset Overview and Data Preparation
  2.2  Cross-Year Distribution and Trend Comparison
  2.3  Relationship Analysis between Quality Parameters and Phosphorus

It is ONLY executed when more than one year of data is present in the
processed directory.  All outputs are written to sub-folders of
`cross_year_comparison/` using the convention:

    <cross_year_dir>/
    ├── 2.1_Dataset_Overview/
    │   ├── generate_dataset_overview_table.csv
    │   ├── generate_missing_summary_table.csv
    │   ├── plot_monthly_sample_bars.png
    │   └── plot_outlier_boxplots.png
    ├── 2.2_Distribution_Trend/
    │   ├── consolidate_descriptive_stats.csv
    │   ├── plot_cross_year_distribution.png
    │   └── plot_monthly_trend_comparison.png
    └── 2.3_Relationship_Analysis/
        ├── generate_correlation_comparison_table.csv
        ├── generate_vif_comparison_table.csv
        └── plot_cross_year_scatter.png

Target variables (kept consistent with data_processing.py):
  - Dependent  : feed_p_ppm
  - Independent: feed_ffa_pct, feed_mi_pct, feed_iv, feed_dobi, feed_car_pv
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats as scipy_stats

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from cpo_phosphorus.paths import LOCAL_PROCESSED_DATA_DIR, LOCAL_PREPROCESSING_REPORTS_DIR

# ─── Variable configuration ───────────────────────────────────────────────────
TARGET_COL = "feed_p_ppm"
FEATURE_COLS = ["feed_ffa_pct", "feed_mi_pct", "feed_iv", "feed_dobi", "feed_car_pv"]
ANALYSIS_COLS = FEATURE_COLS + [TARGET_COL]

PRETTY_NAMES = {
    "feed_p_ppm": "P (ppm)",
    "feed_ffa_pct": "FFA (%)",
    "feed_mi_pct": "M&I (%)",
    "feed_iv": "IV",
    "feed_dobi": "DOBI",
    "feed_car_pv": "CAR/PV",
}

PALETTE = {
    2024: "#1f77b4",   # blue
    2025: "#ff7f0e",   # orange
    "overall": "#2ca02c",
}

# ─── Internal helpers ─────────────────────────────────────────────────────────

_SEPARATOR = "=" * 60


def _banner(msg: str) -> None:
    print(f"\n{_SEPARATOR}")
    print(f"  {msg}")
    print(f"{_SEPARATOR}\n")


def _save(fig: plt.Figure, path: Path, dpi: int = 150) -> None:
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {path.name}")


def _get_color(year_int: int) -> str:
    return PALETTE.get(year_int, "#9467bd")


def _pretty(col: str) -> str:
    return PRETTY_NAMES.get(col, col)


def _load_overall(processed_base: Path) -> pd.DataFrame:
    """Load the combined multi-year processed dataset."""
    path = processed_base / "overall" / "processed_full.csv"
    df = pd.read_csv(path, parse_dates=["date"])
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    return df


def _load_json(report_base: Path, group: str, filename: str) -> dict:
    p = report_base / group / filename
    return json.loads(p.read_text(encoding="utf-8"))


def _load_csv(report_base: Path, group: str, filename: str) -> pd.DataFrame:
    return pd.read_csv(report_base / group / filename, index_col=0)


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


# ─── Section 2.1 ──────────────────────────────────────────────────────────────

def generate_dataset_overview_table(
    report_base: Path,
    years: list[int],
    out_dir: Path,
) -> pd.DataFrame:
    """
    Compare two years across four dimensions and produce a summary table:
      1. Same columns / fields?
      2. Missing months?
      3. Sample size difference?
      4. Consistent variable definition (unit / dtype)?
    """
    rows = []

    summaries = {
        y: _load_json(report_base, str(y), "preprocessing_summary.json")
        for y in years
    }

    # 1. Field consistency
    field_sets = {}
    for y in years:
        s = summaries[y]
        # infer from missing_values_before keys
        field_sets[y] = set(s["missing_values_before"].keys())
    ref_fields = field_sets[years[0]]
    all_same_fields = all(field_sets[y] == ref_fields for y in years)
    extra = {y: field_sets[y] - ref_fields for y in years}
    missing_fields = {y: ref_fields - field_sets[y] for y in years}
    rows.append({
        "Check": "Same fields / columns",
        **{str(y): "Yes" if (not extra[y] and not missing_fields[y]) else
           f"Extra: {extra[y] or '-'}  Missing: {missing_fields[y] or '-'}"
           for y in years},
        "Consistent": "Yes" if all_same_fields else "No",
    })

    # 2. Missing months
    all_months = set(range(1, 13))
    for y in years:
        present = set(summaries[y]["months_present"])
        missing = sorted(all_months - present)
        rows.append({
            "Check": f"Missing months ({y})",
            **{str(yy): "" for yy in years},
            str(y): f"Month(s) {missing}" if missing else "None",
            "Consistent": "",
        })

    # 3. Sample size
    n_rows = {y: summaries[y]["rows_after"] for y in years}
    rows.append({
        "Check": "Row count (after preprocessing)",
        **{str(y): str(n_rows[y]) for y in years},
        "Consistent": (
            "Similar" if max(n_rows.values()) / max(min(n_rows.values()), 1) < 1.5
            else "Difference > 50%"
        ),
    })

    # 4. Variable definition (presence of analysis columns in dataset)
    for col in ANALYSIS_COLS:
        col_present = {
            y: col in summaries[y]["missing_values_before"]
            for y in years
        }
        rows.append({
            "Check": f"Variable present: {_pretty(col)}",
            **{str(y): "Yes" if col_present[y] else "No" for y in years},
            "Consistent": "Yes" if all(col_present.values()) else "No",
        })

    df = pd.DataFrame(rows)
    out_path = out_dir / "generate_dataset_overview_table.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"    Saved: {out_path.name}")
    return df


def generate_missing_summary_table(
    report_base: Path,
    years: list[int],
    out_dir: Path,
) -> pd.DataFrame:
    """
    Merge per-year missing-value counts into a single comparison table.
    Output columns: feature | missing_count_<year> | missing_rate_<year> ...
    """
    frames = []
    for y in years:
        s = _load_json(report_base, str(y), "preprocessing_summary.json")
        total = s["rows_before"]
        missing_before = s["missing_values_before"]
        sub = pd.DataFrame(
            [{"feature": k, f"missing_count_{y}": v,
              f"missing_rate_{y}": round(v / total * 100, 2) if total else 0}
             for k, v in missing_before.items()
             if k in ANALYSIS_COLS]
        )
        frames.append(sub)

    df = frames[0]
    for sub in frames[1:]:
        df = df.merge(sub, on="feature", how="outer")
    df = df.sort_values("feature").reset_index(drop=True)

    out_path = out_dir / "generate_missing_summary_table.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"    Saved: {out_path.name}")
    return df


def plot_monthly_sample_bars(
    df_overall: pd.DataFrame,
    years: list[int],
    out_dir: Path,
) -> None:
    """
    Grouped bar chart: number of non-null rows per month, coloured by year.
    """
    records = []
    for y in years:
        sub = df_overall[df_overall["year"] == y]
        for m in range(1, 13):
            n = int(sub[sub["month"] == m][TARGET_COL].notna().sum())
            records.append({"year": y, "month": m, "n": n})
    plot_df = pd.DataFrame(records)

    n_years = len(years)
    bar_w = 0.8 / n_years
    months = list(range(1, 13))
    x = np.arange(len(months))

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, y in enumerate(years):
        sub = plot_df[plot_df["year"] == y].set_index("month")
        vals = [sub.loc[m, "n"] if m in sub.index else 0 for m in months]
        offset = (i - (n_years - 1) / 2) * bar_w
        bars = ax.bar(x + offset, vals, width=bar_w, label=str(y),
                      color=_get_color(y), alpha=0.85, edgecolor="white", linewidth=0.5)
        # Annotate each bar with its count value (skip zeros)
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.3,
                    str(val),
                    ha="center", va="bottom",
                    fontsize=7, color="black",
                )

    ax.set_xticks(x)
    ax.set_xticklabels([f"M{m}" for m in months])
    ax.set_xlabel("Month")
    ax.set_ylabel("Number of valid samples")
    ax.set_title(f"Monthly Sample Count by Year — {_pretty(TARGET_COL)}")
    # Place legend outside the plot area (top-right) to avoid overlapping bars
    ax.legend(
        title="Year",
        title_fontsize=7,
        fontsize=7,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0,
        framealpha=0.85,
    )
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    # Add a small top margin so bar labels don't get clipped
    y_max = ax.get_ylim()[1]
    ax.set_ylim(0, y_max * 1.12)
    # subplots_adjust leaves room on the right for the external legend
    fig.tight_layout()
    fig.subplots_adjust(right=0.88)
    _save(fig, out_dir / "plot_monthly_sample_bars.png")


def plot_outlier_boxplots(
    df_overall: pd.DataFrame,
    years: list[int],
    out_dir: Path,
) -> None:
    """
    Box plots of analysis variables grouped by year-month to show outlier
    structure. One subplot per analysis column.
    """
    cols = [c for c in ANALYSIS_COLS if c in df_overall.columns]
    n_cols = len(cols)
    fig, axes = plt.subplots(n_cols, 1, figsize=(14, 4 * n_cols), sharex=False)
    if n_cols == 1:
        axes = [axes]

    df_overall = df_overall.copy()
    df_overall["year_month"] = (
        df_overall["year"].astype(str) + "-"
        + df_overall["month"].astype(str).str.zfill(2)
    )
    palette = {str(y): _get_color(y) for y in years}
    df_overall["year_str"] = df_overall["year"].astype(str)

    for ax, col in zip(axes, cols):
        valid = df_overall[df_overall[col].notna()].copy()
        order = sorted(valid["year_month"].unique())
        hue_map = {ym: str(ym[:4]) for ym in order}
        valid["year_str"] = valid["year_month"].map(hue_map)
        sns.boxplot(
            data=valid,
            x="year_month",
            y=col,
            hue="year_str",
            order=order,
            hue_order=[str(y) for y in years],
            palette=palette,
            ax=ax,
            width=0.6,
            linewidth=0.8,
            fliersize=3,
            dodge=False,
        )
        ax.set_title(f"{_pretty(col)} — Outlier Distribution by Year-Month")
        ax.set_xlabel("")
        ax.set_ylabel(_pretty(col))
        ax.tick_params(axis="x", rotation=70, labelsize=7)
        ax.legend(title="Year", fontsize=8)

    fig.tight_layout()
    _save(fig, out_dir / "plot_outlier_boxplots.png", dpi=120)


# ─── Section 2.2 ──────────────────────────────────────────────────────────────

def consolidate_descriptive_stats(
    report_base: Path,
    years: list[int],
    out_dir: Path,
) -> pd.DataFrame:
    """
    Side-by-side descriptive statistics for each year.
    Columns: feature | mean_<y> | std_<y> | median_<y> | min_<y> | max_<y> ...
    """
    frames = []
    stat_cols = ["mean", "std", "min", "25%", "median", "75%", "max", "missing_count"]
    for y in years:
        desc = pd.read_csv(report_base / str(y) / "descriptive_stats.csv")
        desc = desc[desc["feature"].isin(ANALYSIS_COLS)].copy()
        rename = {c: f"{c}_{y}" for c in stat_cols if c in desc.columns}
        desc = desc.rename(columns=rename)[["feature"] + list(rename.values())]
        frames.append(desc)

    df = frames[0]
    for sub in frames[1:]:
        df = df.merge(sub, on="feature", how="outer")
    df = df.sort_values("feature").reset_index(drop=True)

    out_path = out_dir / "consolidate_descriptive_stats.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"    Saved: {out_path.name}")
    return df


def plot_cross_year_distribution(
    df_overall: pd.DataFrame,
    years: list[int],
    out_dir: Path,
) -> None:
    """
    KDE + rug plots for each analysis variable, one colour per year,
    overlaid on the same axes to show distribution shift.
    """
    cols = [c for c in ANALYSIS_COLS if c in df_overall.columns]
    n_cols = len(cols)
    ncols_grid = 3
    nrows_grid = (n_cols + ncols_grid - 1) // ncols_grid

    fig, axes = plt.subplots(nrows_grid, ncols_grid,
                             figsize=(5 * ncols_grid, 4 * nrows_grid))
    axes_flat = axes.flatten() if n_cols > 1 else [axes]

    for ax, col in zip(axes_flat, cols):
        for y in years:
            sub = df_overall[df_overall["year"] == y][col].dropna()
            if len(sub) < 5:
                continue
            sns.kdeplot(sub, ax=ax, label=str(y), color=_get_color(y),
                        fill=True, alpha=0.25, linewidth=1.5)
        ax.set_title(_pretty(col))
        ax.set_xlabel("")
        ax.legend(title="Year", fontsize=8)

    # hide unused axes
    for ax in axes_flat[n_cols:]:
        ax.set_visible(False)

    fig.suptitle("Cross-Year Distribution Comparison (KDE)", fontsize=13, y=1.01)
    fig.tight_layout()
    _save(fig, out_dir / "plot_cross_year_distribution.png")


def plot_monthly_trend_comparison(
    df_overall: pd.DataFrame,
    years: list[int],
    out_dir: Path,
) -> None:
    """
    Line chart of monthly mean for each analysis variable, one line per year.
    Gaps in data (missing months) appear as natural breaks in the line.
    """
    cols = [c for c in ANALYSIS_COLS if c in df_overall.columns]
    n_cols = len(cols)
    fig, axes = plt.subplots(n_cols, 1, figsize=(12, 4 * n_cols), sharex=False)
    if n_cols == 1:
        axes = [axes]

    for ax, col in zip(axes, cols):
        all_months = list(range(1, 13))
        for y in years:
            sub = df_overall[df_overall["year"] == y]
            monthly_mean = (
                sub.groupby("month")[col].mean().reindex(all_months)
            )
            # Use NaN for missing months — matplotlib renders natural gaps
            ax.plot(all_months, monthly_mean.values,
                    marker="o", markersize=4,
                    label=str(y), color=_get_color(y),
                    linewidth=1.8)

        ax.set_xticks(all_months)
        ax.set_xticklabels([f"M{m}" for m in all_months])
        ax.set_ylabel(_pretty(col))
        ax.set_title(f"Monthly Mean — {_pretty(col)}")
        ax.legend(title="Year", fontsize=8)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.suptitle("Monthly Trend Comparison by Year", fontsize=13, y=1.01)
    fig.tight_layout()
    _save(fig, out_dir / "plot_monthly_trend_comparison.png")


# ─── Section 2.3 ──────────────────────────────────────────────────────────────

def generate_correlation_comparison_table(
    report_base: Path,
    years: list[int],
    out_dir: Path,
) -> pd.DataFrame:
    """
    Merge per-year Pearson and Spearman correlations with feed_p_ppm
    into the table format required by the Final Report Outline:
      Variable | Pearson r (2024) | Pearson r (2025) | Spearman r (2024) | ...
    """
    rows_map: dict[str, dict] = {}

    for y in years:
        pearson = _load_csv(report_base, str(y), "correlation_pearson_core_metrics.csv")
        spearman = _load_csv(report_base, str(y), "correlation_spearman_core_metrics.csv")

        for feat in FEATURE_COLS:
            if feat not in rows_map:
                rows_map[feat] = {"Variable": _pretty(feat)}
            if feat in pearson.index and TARGET_COL in pearson.columns:
                rows_map[feat][f"Pearson r ({y})"] = round(
                    float(pearson.loc[feat, TARGET_COL]), 4
                )
            if feat in spearman.index and TARGET_COL in spearman.columns:
                rows_map[feat][f"Spearman r ({y})"] = round(
                    float(spearman.loc[feat, TARGET_COL]), 4
                )

    df = pd.DataFrame(list(rows_map.values()))
    out_path = out_dir / "generate_correlation_comparison_table.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"    Saved: {out_path.name}")
    return df


def generate_vif_comparison_table(
    report_base: Path,
    years: list[int],
    out_dir: Path,
) -> pd.DataFrame:
    """
    Merge per-year VIF scores into a side-by-side table:
      Variable | VIF (2024) | VIF (2025) | ...
    """
    frames = []
    for y in years:
        vif = pd.read_csv(report_base / str(y) / "vif_core_features.csv")
        vif = vif.rename(columns={"vif": f"VIF ({y})", "feature": "Variable"})
        vif["Variable"] = vif["Variable"].map(lambda x: _pretty(x) if x in PRETTY_NAMES else x)
        frames.append(vif.set_index("Variable"))

    df = pd.concat(frames, axis=1).reset_index()
    df = df.rename(columns={"index": "Variable"})

    out_path = out_dir / "generate_vif_comparison_table.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"    Saved: {out_path.name}")
    return df


def plot_cross_year_scatter(
    df_overall: pd.DataFrame,
    years: list[int],
    out_dir: Path,
) -> None:
    """
    Scatter plot for each feature vs feed_p_ppm, coloured by year,
    with per-year OLS regression lines to show trend stability.
    """
    cols = [c for c in FEATURE_COLS if c in df_overall.columns]
    if TARGET_COL not in df_overall.columns:
        print("  [WARN] Target column not found, skipping scatter plot.")
        return

    n_cols = len(cols)
    ncols_grid = 3
    nrows_grid = (n_cols + ncols_grid - 1) // ncols_grid

    fig, axes = plt.subplots(nrows_grid, ncols_grid,
                             figsize=(5.5 * ncols_grid, 4.5 * nrows_grid))
    axes_flat = axes.flatten() if n_cols > 1 else [axes]

    for ax, col in zip(axes_flat, cols):
        for y in years:
            sub = df_overall[df_overall["year"] == y][[col, TARGET_COL]].dropna()
            if len(sub) < 5:
                continue
            ax.scatter(sub[col], sub[TARGET_COL],
                       color=_get_color(y), alpha=0.45, s=18,
                       edgecolors="none", label=str(y))
            # OLS regression line
            slope, intercept, *_ = scipy_stats.linregress(sub[col], sub[TARGET_COL])
            x_range = np.linspace(sub[col].min(), sub[col].max(), 100)
            ax.plot(x_range, intercept + slope * x_range,
                    color=_get_color(y), linewidth=1.8, linestyle="--")

        ax.set_xlabel(_pretty(col))
        ax.set_ylabel(_pretty(TARGET_COL))
        ax.set_title(f"{_pretty(col)} vs {_pretty(TARGET_COL)}")
        ax.legend(title="Year", fontsize=8, markerscale=1.5)

    for ax in axes_flat[n_cols:]:
        ax.set_visible(False)

    fig.suptitle(
        f"Feature vs {_pretty(TARGET_COL)} — Cross-Year Scatter",
        fontsize=13, y=1.01,
    )
    fig.tight_layout()
    _save(fig, out_dir / "plot_cross_year_scatter.png")


# ─── Orchestration ────────────────────────────────────────────────────────────

def run_final_eda(
    processed_dir: str,
    report_dir: str,
    cross_year_dir: str,
) -> None:
    """
    Main orchestration function called by scripts/run_final_eda.py.

    Parameters
    ----------
    processed_dir  : Root of processed data (contains year sub-folders + overall/).
    report_dir     : Root of preprocessing reports (contains year sub-folders).
    cross_year_dir : Destination for all cross-year EDA outputs.
    """
    processed_base = Path(processed_dir)
    report_base = Path(report_dir)
    cye_base = Path(cross_year_dir)

    # ── Detect years ─────────────────────────────────────────────────────────
    years = sorted(
        int(d.name)
        for d in processed_base.iterdir()
        if d.is_dir() and d.name.isdigit()
    )

    if len(years) < 2:
        print(
            "[INFO] Only one year detected in processed data. "
            "Cross-year EDA requires at least two years. Exiting."
        )
        return

    _banner(f"Cross-Year EDA  |  Years: {years}")

    # ── Load combined dataset ─────────────────────────────────────────────────
    df_overall = _load_overall(processed_base)

    # ── Create output directories ─────────────────────────────────────────────
    dir_21 = _ensure_dir(cye_base / "2.1_Dataset_Overview")
    dir_22 = _ensure_dir(cye_base / "2.2_Distribution_Trend")
    dir_23 = _ensure_dir(cye_base / "2.3_Relationship_Analysis")

    # ── Section 2.1 ───────────────────────────────────────────────────────────
    _banner("Section 2.1 — Dataset Overview and Data Preparation")
    generate_dataset_overview_table(report_base, years, dir_21)
    generate_missing_summary_table(report_base, years, dir_21)
    plot_monthly_sample_bars(df_overall, years, dir_21)
    plot_outlier_boxplots(df_overall, years, dir_21)

    # ── Section 2.2 ───────────────────────────────────────────────────────────
    _banner("Section 2.2 — Cross-Year Distribution and Trend Comparison")
    consolidate_descriptive_stats(report_base, years, dir_22)
    plot_cross_year_distribution(df_overall, years, dir_22)
    plot_monthly_trend_comparison(df_overall, years, dir_22)

    # ── Section 2.3 ───────────────────────────────────────────────────────────
    _banner("Section 2.3 — Relationship Analysis")
    generate_correlation_comparison_table(report_base, years, dir_23)
    generate_vif_comparison_table(report_base, years, dir_23)
    plot_cross_year_scatter(df_overall, years, dir_23)

    _banner("Cross-Year EDA Complete")
    print(f"  All outputs written to: {cye_base}\n")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    default_processed_dir = str(LOCAL_PROCESSED_DATA_DIR)
    default_report_dir = str(LOCAL_PREPROCESSING_REPORTS_DIR)
    default_cross_year_dir = str(
        Path(LOCAL_PREPROCESSING_REPORTS_DIR).parent / "cross_year_comparison"
    )

    parser = argparse.ArgumentParser(
        description=(
            "Cross-year EDA analysis. "
            "Skipped automatically when fewer than two years are present."
        )
    )
    parser.add_argument(
        "--processed-dir",
        default=default_processed_dir,
        metavar="DIR",
        help=f"Root of processed data. Default: {default_processed_dir}",
    )
    parser.add_argument(
        "--report-dir",
        default=default_report_dir,
        metavar="DIR",
        help=f"Root of preprocessing reports. Default: {default_report_dir}",
    )
    parser.add_argument(
        "--cross-year-dir",
        default=default_cross_year_dir,
        metavar="DIR",
        help=f"Output root for cross-year EDA. Default: {default_cross_year_dir}",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point — called by scripts/run_final_eda.py."""
    args = parse_args()
    run_final_eda(
        processed_dir=args.processed_dir,
        report_dir=args.report_dir,
        cross_year_dir=args.cross_year_dir,
    )


if __name__ == "__main__":
    main()
