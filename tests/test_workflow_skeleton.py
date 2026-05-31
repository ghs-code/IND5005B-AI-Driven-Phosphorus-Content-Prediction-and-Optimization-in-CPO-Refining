import tempfile
import unittest
from pathlib import Path

import pandas as pd

from cpo_phosphorus.workflows.core_factors import run_core_factor_check
from cpo_phosphorus.workflows.internal_factors import run_internal_factor_check
from cpo_phosphorus.workflows.validate_data import run_data_validation


class WorkflowSkeletonTests(unittest.TestCase):
    def _write_quality_workbook(self, path, year, rows=2):
        frame = pd.DataFrame([[None] * 21 for _ in range(4 + rows)])
        for idx in range(rows):
            row = 4 + idx
            values = [
                pd.Timestamp(year=year, month=1, day=idx + 1),
                0.065,
                1.4,
                "NT1",
                4.5 + idx,
                0.2,
                52.0,
                2.3,
                470.0,
                25.0 + idx,
                "CPO",
                "ST14",
                0.06,
                0.01,
                52.0,
                0.0,
                2.2,
                8.0,
                0.5,
                "RBDPO",
            ]
            for col_offset, value in enumerate(values, start=1):
                frame.iat[row, col_offset] = value

        with pd.ExcelWriter(path) as writer:
            frame.to_excel(writer, sheet_name=f"January {year}", header=False, index=False)

    def test_validate_data_accepts_multiple_quality_workbooks(self):
        with tempfile.TemporaryDirectory() as tmp:
            path_2024 = Path(tmp) / "quality_2024.xlsx"
            path_2025 = Path(tmp) / "quality_2025.xlsx"
            self._write_quality_workbook(path_2024, 2024)
            self._write_quality_workbook(path_2025, 2025)

            summary = run_data_validation(
                [path_2024, path_2025],
                target_col="feed_p_ppm",
                join_keys="date,feed_tank",
                factor_fields="feed_ffa_pct,feed_dobi",
            )

        self.assertEqual(summary["status"], "ready")
        self.assertEqual(summary["file_count"], 2)
        self.assertEqual(summary["years_present"], [2024, 2025])
        self.assertEqual(summary["valid_target_rows"], 4)
        self.assertEqual(summary["template_issue_count"], 0)
        self.assertEqual(summary["field_mapping"]["missing_join_keys"], [])
        self.assertEqual(summary["field_mapping"]["missing_factor_fields"], [])

    def test_core_factor_check_reports_screening_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model_source.csv"
            pd.DataFrame(
                {
                    "date": pd.date_range("2025-01-01", periods=40),
                    "feed_p_ppm": list(range(40)),
                    "feed_ffa_pct": list(range(40)),
                    "feed_mi_pct": [0.1] * 40,
                    "feed_iv": [50.0 + (idx % 3) for idx in range(40)],
                    "feed_dobi": [2.0 + (idx % 2) for idx in range(40)],
                    "feed_car_pv": [400.0 + idx for idx in range(40)],
                }
            ).to_csv(path, index=False)

            summary = run_core_factor_check(path)

        self.assertEqual(summary["status"], "screened")
        self.assertEqual(summary["valid_target_rows"], 40)
        self.assertEqual(summary["missing_core_fields"], [])
        self.assertGreaterEqual(len(summary["screening"]), 1)

    def test_internal_factor_check_without_input_is_not_configured(self):
        summary = run_internal_factor_check()

        self.assertEqual(summary["status"], "not_configured")
        self.assertEqual(summary["workflow"], "internal_factors")

    def test_internal_factor_direct_validation_with_join(self):
        with tempfile.TemporaryDirectory() as tmp:
            quality = Path(tmp) / "quality.csv"
            internal = Path(tmp) / "internal.csv"
            dates = pd.date_range("2025-01-01", periods=60)
            pd.DataFrame(
                {
                    "date": dates,
                    "feed_tank": ["NT1"] * 60,
                    "feed_p_ppm": [float(idx) for idx in range(60)],
                }
            ).to_csv(quality, index=False)
            pd.DataFrame(
                {
                    "date": dates,
                    "feed_tank": ["NT1"] * 60,
                    "storage_hours": [float(idx) for idx in range(60)],
                    "source": ["A" if idx < 30 else "B" for idx in range(60)],
                }
            ).to_csv(internal, index=False)

            summary = run_internal_factor_check(
                input_path=internal,
                quality_input=quality,
                join_keys="date,feed_tank",
                factor_fields="storage_hours,source",
            )

        self.assertEqual(summary["status"], "validated")
        self.assertEqual(summary["join_diagnostics"]["matched_rows"], 60)
        self.assertEqual(summary["evidence_counts"].get("direct_supported"), 2)
        self.assertIsNotNone(summary["incremental_model_lift"])
        self.assertIsNotNone(summary["risk_classification_gain"])

    def test_internal_factor_exact_id_and_multiple_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            quality = Path(tmp) / "quality.csv"
            internal_a = Path(tmp) / "internal_a.csv"
            internal_b = Path(tmp) / "internal_b.csv"
            sample_ids = [f"S{idx:03d}" for idx in range(50)]
            pd.DataFrame({"sample_id": sample_ids, "feed_p_ppm": range(50)}).to_csv(quality, index=False)
            pd.DataFrame({"sample_id": sample_ids[:25], "metal_ppm": range(25)}).to_csv(internal_a, index=False)
            pd.DataFrame({"sample_id": sample_ids[25:], "metal_ppm": range(25, 50)}).to_csv(internal_b, index=False)

            summary = run_internal_factor_check(
                input_path=[internal_a, internal_b],
                quality_input=quality,
                join_keys="sample_id",
                factor_fields="metal_ppm",
            )

        self.assertEqual(summary["status"], "validated")
        self.assertEqual(summary["join_diagnostics"]["matched_rows"], 50)
        self.assertEqual(summary["evidence_counts"].get("direct_supported"), 1)

    def test_internal_factor_duplicate_keys_trigger_attention(self):
        with tempfile.TemporaryDirectory() as tmp:
            quality = Path(tmp) / "quality.csv"
            internal = Path(tmp) / "internal.csv"
            pd.DataFrame({"sample_id": ["A", "B"], "feed_p_ppm": [1.0, 2.0]}).to_csv(quality, index=False)
            pd.DataFrame({"sample_id": ["A", "A", "B"], "storage_hours": [1.0, 2.0, 3.0]}).to_csv(
                internal, index=False
            )

            summary = run_internal_factor_check(
                input_path=internal,
                quality_input=quality,
                join_keys="sample_id",
                factor_fields="storage_hours",
            )

        self.assertEqual(summary["status"], "needs_attention")
        self.assertEqual(summary["join_diagnostics"]["status"], "duplicate_internal_keys")

    def test_internal_factor_no_signal_is_not_supported(self):
        with tempfile.TemporaryDirectory() as tmp:
            quality = Path(tmp) / "quality.csv"
            internal = Path(tmp) / "internal.csv"
            sample_ids = [f"S{idx:03d}" for idx in range(60)]
            pd.DataFrame({"sample_id": sample_ids, "feed_p_ppm": [float(idx % 10) for idx in range(60)]}).to_csv(
                quality, index=False
            )
            pd.DataFrame({"sample_id": sample_ids, "constant_factor": [1.0] * 60}).to_csv(internal, index=False)

            summary = run_internal_factor_check(
                input_path=internal,
                quality_input=quality,
                join_keys="sample_id",
                factor_fields="constant_factor",
            )

        self.assertEqual(summary["evidence_counts"].get("not_assessable"), 1)

    def test_internal_factor_missing_join_key_is_not_assessable(self):
        with tempfile.TemporaryDirectory() as tmp:
            quality = Path(tmp) / "quality.csv"
            internal = Path(tmp) / "internal.csv"
            pd.DataFrame({"sample_id": ["A"], "feed_p_ppm": [1.0]}).to_csv(quality, index=False)
            pd.DataFrame({"other_id": ["A"], "factor": [1.0]}).to_csv(internal, index=False)

            summary = run_internal_factor_check(
                input_path=internal,
                quality_input=quality,
                join_keys="sample_id",
                factor_fields="factor",
            )

        self.assertEqual(summary["status"], "needs_attention")
        self.assertEqual(summary["join_diagnostics"]["status"], "missing_join_keys")
        self.assertEqual(summary["evidence_counts"].get("not_assessable"), 1)


if __name__ == "__main__":
    unittest.main()
