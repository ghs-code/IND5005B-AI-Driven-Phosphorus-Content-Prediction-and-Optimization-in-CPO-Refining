from pathlib import Path
import argparse
import os
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cpo_phosphorus.paths import LOCAL_PROCESSED_DATA_DIR
from cpo_phosphorus.workflows.common import print_json
from cpo_phosphorus.workflows.core_factors import run_core_factor_check


def parse_args():
    default_target = os.getenv("CPO_TARGET_COL", "feed_p_ppm")
    parser = argparse.ArgumentParser(description="Check core quality factor readiness")
    parser.add_argument(
        "--input",
        default=str(LOCAL_PROCESSED_DATA_DIR / "model_source.csv"),
        help="Processed model_source.csv or compatible table.",
    )
    parser.add_argument(
        "--target-col",
        default=default_target,
        help=f"Target column to validate. Default: {default_target}",
    )
    parser.add_argument(
        "--output-dir",
        default=os.getenv("CPO_CORE_FACTOR_REPORT_DIR", ""),
        help="Optional directory for core factor screening artifacts.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print_json(
        run_core_factor_check(
            args.input,
            target_col=args.target_col,
            output_dir=args.output_dir or None,
        )
    )


if __name__ == "__main__":
    main()
