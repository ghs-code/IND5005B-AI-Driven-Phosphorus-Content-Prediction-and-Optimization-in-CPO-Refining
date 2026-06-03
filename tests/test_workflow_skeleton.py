import tempfile
import unittest
from pathlib import Path

import pandas as pd

from cpo_phosphorus.workflows.core_factors import run_core_factor_check
from cpo_phosphorus.workflows.delivery import EnterpriseRunConfig, run_enterprise_delivery
from cpo_phosphorus.workflows.internal_factors import run_internal_factor_check
from cpo_phosphorus.workflows.risk_scoring import build_review_priority_ranking
from cpo_phosphorus.workflows.validate_data import run_data_validation


class WorkflowSkeletonTests(unittest.TestCase):
    def _write_quality_workbook(self, path, year, rows=2):
        frame = pd.DataFrame([[None] * 21 for _ in range(4 + rows)])
        for idx in range(rows):
            row = 4 + idx
            date = pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(days=idx)
            values = [
                date,
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

    def test_limited_human_review_ranking_sorts_candidates(self):
        predictions = pd.DataFrame(
            {
                "feature_group": ["context_aware", "context_aware", "context_aware", "quality_only"],
                "model": ["ridge", "ridge", "ridge", "ridge"],
                "date": pd.date_range("2025-01-01", periods=4),
                "feed_tank": ["A", "B", "C", "D"],
                "actual": [9.0, 5.0, 8.0, 99.0],
                "predicted": [7.0, 10.0, 8.5, 100.0],
            }
        )
        summary = {
            "best_random_split_model": {"feature_group": "context_aware", "model": "ridge"},
            "risk_thresholds": [{"threshold_name": "p80", "threshold_ppm": 8.0}],
        }

        ranking, candidates, policy = build_review_priority_ranking(predictions, summary, top_n=2)

        self.assertEqual(ranking["predicted"].tolist(), [10.0, 8.5, 7.0])
        self.assertEqual(candidates["review_priority_rank"].tolist(), [1, 2])
        self.assertEqual(candidates["predicted_high_risk"].tolist(), [True, True])
        self.assertIn("manual review queue", policy["trigger"])
        self.assertIn("descending order", policy["ranking_logic"])
        self.assertIn("does not authorize automatic batch isolation", policy["decision_boundary"])

    def test_enterprise_delivery_writes_to_selected_output_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            quality = root / "quality.xlsx"
            internal = root / "internal.csv"
            output = root / "deliveries"
            self._write_quality_workbook(quality, 2025, rows=60)
            dates = pd.date_range("2025-01-01", periods=60)
            pd.DataFrame(
                {
                    "date": dates,
                    "feed_tank": ["ST14"] * 60,
                    "storage_hours": list(range(60)),
                }
            ).to_csv(internal, index=False)

            result = run_enterprise_delivery(
                EnterpriseRunConfig(
                    quality_inputs=[str(quality)],
                    internal_inputs=[str(internal)],
                    output_dir=str(output),
                    target_col="feed_p_ppm",
                    join_keys="date,feed_tank",
                    internal_factor_fields="storage_hours",
                    run_risk_scoring=False,
                    run_id="test_run",
                )
            )

            run_dir = output / "test_run"
            self.assertEqual(result["status"], "complete")
            self.assertTrue((run_dir / "run_manifest.json").exists())
            self.assertTrue((run_dir / "run_log.txt").exists())
            self.assertTrue((run_dir / "enterprise_summary.html").exists())
            self.assertTrue((run_dir / "core_factor_screening.csv").exists())
            self.assertTrue((run_dir / "internal_factor_evidence.csv").exists())
            self.assertTrue((run_dir / "processed" / "model_source.csv").exists())
            self.assertEqual(result["output_files"]["run_log"], str(run_dir / "run_log.txt"))
            self.assertEqual(result["run_log_path"], str(run_dir / "run_log.txt"))
            self.assertFalse((root / "reports").exists())
            self.assertFalse((root / "outputs").exists())
            run_log = (run_dir / "run_log.txt").read_text(encoding="utf-8")
            self.assertIn("run_started", run_log)
            self.assertIn("data_validation_completed", run_log)
            self.assertIn("internal_factor_validation_completed", run_log)
            self.assertIn("risk_scoring_skipped", run_log)
            self.assertIn("run_completed", run_log)
            html = (run_dir / "enterprise_summary.html").read_text(encoding="utf-8")
            self.assertIn("PhosphorAI Enterprise Summary", html)
            self.assertIn("Quality rows", html)
            self.assertIn("Limited Human Review", html)
            self.assertIn("PGEO Data Roadmap Discussion", html)
            self.assertIn("does not authorize automatic batch isolation", html)
            self.assertIn("supplier/mill", html)


if __name__ == "__main__":
    unittest.main()
