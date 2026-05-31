from pathlib import Path
import argparse
import os
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cpo_phosphorus.paths import DEFAULT_RAW_EXCEL
from cpo_phosphorus.workflows.common import print_json
from cpo_phosphorus.workflows.validate_data import run_data_validation


def parse_args():
    default_input = os.getenv("CPO_RAW_INPUT", str(DEFAULT_RAW_EXCEL))
    default_target = os.getenv("CPO_TARGET_COL", "feed_p_ppm")
    parser = argparse.ArgumentParser(description="Validate quality-table input discovery")
    parser.add_argument(
        "--input",
        action="append",
        default=None,
        help="Quality-table Excel file or directory. Can be passed multiple times.",
    )
    parser.add_argument(
        "--year",
        default=os.getenv("CPO_YEAR", "all"),
        help="Optional year filter label for downstream workflows.",
    )
    parser.add_argument(
        "--target-col",
        default=default_target,
        help=f"Target column to validate. Default: {default_target}",
    )
    parser.add_argument(
        "--join-keys",
        default=os.getenv("CPO_QUALITY_JOIN_KEYS", ""),
        help="Comma-separated quality-table join keys to check.",
    )
    parser.add_argument(
        "--factor-fields",
        default=os.getenv("CPO_FACTOR_FIELDS", ""),
        help="Comma-separated factor fields to check.",
    )
    parser.add_argument(
        "--process-response-fields",
        default=os.getenv("CPO_PROCESS_RESPONSE_FIELDS", ""),
        help="Comma-separated process-response fields to check.",
    )
    args = parser.parse_args()
    if args.input is None:
        args.input = [default_input]
    return args


def main():
    args = parse_args()
    print_json(
        run_data_validation(
            args.input,
            year_filter=args.year,
            target_col=args.target_col,
            join_keys=args.join_keys,
            factor_fields=args.factor_fields,
            process_response_fields=args.process_response_fields,
        )
    )


if __name__ == "__main__":
    main()
