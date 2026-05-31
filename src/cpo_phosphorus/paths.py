"""Project paths and enterprise UI defaults."""

from __future__ import annotations

import os
from pathlib import Path


def _env_path(name: str, default: Path) -> Path:
    return Path(os.getenv(name, str(default))).expanduser()


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCAL_DATA_DIR = PROJECT_ROOT / "local_data"

LOCAL_RAW_DATA_DIR = _env_path("PHOSPHORAI_DEFAULT_INPUT_DIR", LOCAL_DATA_DIR / "raw")
LOCAL_DELIVERY_DIR = _env_path("PHOSPHORAI_DEFAULT_OUTPUT_DIR", LOCAL_DATA_DIR / "deliveries")

LOCAL_PROCESSED_DATA_DIR = _env_path("CPO_PROCESSED_DIR", LOCAL_DELIVERY_DIR / "processed")
LOCAL_REPORTS_DIR = _env_path("CPO_REPORTS_DIR", LOCAL_DELIVERY_DIR)
LOCAL_PREPROCESSING_REPORTS_DIR = _env_path(
    "CPO_PREPROCESSING_REPORT_DIR",
    LOCAL_DELIVERY_DIR / "preprocessing",
)
LOCAL_FACTOR_VALIDATION_REPORTS_DIR = _env_path(
    "CPO_FACTOR_REPORT_DIR",
    LOCAL_REPORTS_DIR / "factor_validation",
)

DEFAULT_RAW_EXCEL = _env_path(
    "CPO_RAW_INPUT",
    LOCAL_RAW_DATA_DIR,
)
