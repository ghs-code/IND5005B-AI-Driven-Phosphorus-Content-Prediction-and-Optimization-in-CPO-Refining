from pathlib import Path
import argparse
import os
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cpo_phosphorus.paths import LOCAL_PROCESSED_DATA_DIR, LOCAL_REPORTS_DIR
from cpo_phosphorus.workflows.common import print_json
from cpo_phosphorus.workflows.risk_scoring import run_risk_scoring


def parse_args():
    default_target = os.getenv("CPO_TARGET_COL", "feed_p_ppm")
    parser = argparse.ArgumentParser(description="Run standalone feed phosphorus risk scoring")
    parser.add_argument(
        "--input",
        default=str(LOCAL_PROCESSED_DATA_DIR / "model_source.csv"),
        help="Path to model_source.csv.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(LOCAL_REPORTS_DIR / "risk_scoring"),
        help="Directory for risk scoring outputs.",
    )
    parser.add_argument(
        "--target-col",
        default=default_target,
        help=f"Target column to predict. Default: {default_target}",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print_json(
        run_risk_scoring(
            input_path=args.input,
            output_dir=args.output_dir,
            target_col=args.target_col,
        )
    )


if __name__ == "__main__":
    main()
