from pathlib import Path
import argparse
import os
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cpo_phosphorus.workflows.common import print_json
from cpo_phosphorus.workflows.internal_factors import run_internal_factor_check


def parse_args():
    default_target = os.getenv("CPO_TARGET_COL", "feed_p_ppm")
    default_inputs = [
        item
        for item in os.getenv("CPO_INTERNAL_FACTOR_INPUT", "").split(os.pathsep)
        if item
    ]
    parser = argparse.ArgumentParser(description="Check internal factor table readiness")
    parser.add_argument(
        "--quality-input",
        default=os.getenv("CPO_MODEL_SOURCE", ""),
        help="Quality/model table containing the selected target and join keys.",
    )
    parser.add_argument(
        "--input",
        action="append",
        default=None,
        help="Enterprise internal factor CSV/XLSX table.",
    )
    parser.add_argument(
        "--join-keys",
        default=os.getenv("CPO_INTERNAL_JOIN_KEYS", ""),
        help="Comma-separated join keys, such as sample_id,batch_id or date,feed_tank.",
    )
    parser.add_argument(
        "--target-col",
        default=default_target,
        help=f"Target column to validate. Default: {default_target}",
    )
    parser.add_argument(
        "--factor-fields",
        default=os.getenv("CPO_INTERNAL_FACTOR_FIELDS", ""),
        help="Comma-separated internal factor fields to validate.",
    )
    parser.add_argument(
        "--process-response-fields",
        default=os.getenv("CPO_INTERNAL_PROCESS_RESPONSE_FIELDS", ""),
        help="Comma-separated process-response fields.",
    )
    parser.add_argument(
        "--output-dir",
        default=os.getenv("CPO_INTERNAL_FACTOR_REPORT_DIR", ""),
        help="Optional directory for internal factor artifacts.",
    )
    args = parser.parse_args()
    if args.input is None:
        args.input = default_inputs
    return args


def main():
    args = parse_args()
    print_json(
        run_internal_factor_check(
            input_path=args.input,
            join_keys=args.join_keys,
            target_col=args.target_col,
            quality_input=args.quality_input or None,
            factor_fields=args.factor_fields,
            process_response_fields=args.process_response_fields,
            output_dir=args.output_dir or None,
        )
    )


if __name__ == "__main__":
    main()
