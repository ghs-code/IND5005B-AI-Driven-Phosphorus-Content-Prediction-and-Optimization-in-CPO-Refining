"""Enterprise UI delivery orchestration and report packaging."""

from __future__ import annotations

import html
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pandas as pd

from cpo_phosphorus.pipelines.data_processing import run_pipeline as run_preprocessing
from cpo_phosphorus.workflows.common import json_safe
from cpo_phosphorus.workflows.core_factors import run_core_factor_check
from cpo_phosphorus.workflows.internal_factors import run_internal_factor_check
from cpo_phosphorus.workflows.risk_scoring import limited_human_review_policy, run_risk_scoring
from cpo_phosphorus.workflows.validate_data import run_data_validation


@dataclass
class EnterpriseRunConfig:
    """Configuration for one enterprise UI delivery run."""

    quality_inputs: list[str]
    output_dir: str
    internal_inputs: list[str] = field(default_factory=list)
    target_col: str = "feed_p_ppm"
    year_filter: str = "all"
    join_keys: str = ""
    quality_factor_fields: str = ""
    process_response_fields: str = ""
    internal_factor_fields: str = ""
    internal_process_response_fields: str = ""
    vif_threshold: float = 10.0
    run_internal_factors: bool = True
    run_risk_scoring: bool = True
    run_id: str | None = None


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _write_json(path: Path, payload) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_safe(payload), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return str(path)


class RunLogger:
    """Append a plain-text audit log for one enterprise delivery run."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")

    def write(self, event: str, **details) -> None:
        timestamp = datetime.now().isoformat(timespec="seconds")
        suffix = ""
        if details:
            detail_text = ", ".join(f"{key}={value}" for key, value in details.items())
            suffix = f" | {detail_text}"
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(f"[{timestamp}] {event}{suffix}\n")


def _safe_text(value) -> str:
    if value is None:
        return "Not available"
    return html.escape(str(value))


def _metric(summary: dict | None, key: str):
    if not summary:
        return None
    return summary.get(key)


def _nested_metric(summary: dict | None, *keys):
    value = summary or {}
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


DATA_ROADMAP = [
    {
        "field_group": "Supply context",
        "example_fields": "supplier/mill, harvest period, weather exposure",
        "business_use": "Explain upstream context that quick quality fields only proxy indirectly.",
    },
    {
        "field_group": "Storage and tank history",
        "example_fields": "storage duration, tank residence time, feed tank movement",
        "business_use": "Improve join quality and separate current feed quality from process memory.",
    },
    {
        "field_group": "Lab alignment",
        "example_fields": "sample timestamp, lab result timestamp, retest flag",
        "business_use": "Reduce label timing noise and make false-negative analysis more reliable.",
    },
    {
        "field_group": "Direct chemistry",
        "example_fields": "phospholipid/gum/NHP, trace metals, impurity measurements",
        "business_use": "Validate mechanism hypotheses instead of relying on hidden proxy signals.",
    },
    {
        "field_group": "Process settings",
        "example_fields": "acid dosing, bleaching earth, temperature, pressure, flow",
        "business_use": "Study observed downstream response without turning the model into automatic control.",
    },
]


def _table_rows(path: Path, columns: list[str], limit: int = 10) -> list[dict]:
    if not path.exists():
        return []
    try:
        data = pd.read_csv(path)
    except Exception:
        return []
    available_columns = [col for col in columns if col in data]
    if not available_columns:
        return []
    return data[available_columns].head(limit).to_dict(orient="records")


def _top_review_rows(candidates_path: Path, ranking_path: Path, limit: int = 10) -> list[dict]:
    columns = [
        "review_priority_rank",
        "date",
        "feed_tank",
        "actual",
        "predicted",
        "predicted_high_risk",
        "review_trigger",
    ]
    rows = _table_rows(candidates_path, columns, limit)
    if rows:
        return rows
    return _table_rows(ranking_path, columns, limit)


def _write_enterprise_html(run_dir: Path, payload: dict) -> str:
    quality = payload.get("quality_validation")
    preprocessing = payload.get("preprocessing")
    core = payload.get("core_factors")
    internal = payload.get("internal_factors")
    risk = payload.get("risk_scoring")

    join = _nested_metric(internal, "join_diagnostics") or {}
    decision = _nested_metric(risk, "decision_summary") or {}
    review_policy = _nested_metric(risk, "limited_human_review") or limited_human_review_policy()
    evidence_counts = _metric(internal, "evidence_counts") or {}
    top_review = payload.get("top_review_rows") or []

    rows = [
        ("Quality rows", _metric(quality, "row_count")),
        ("Valid target rows", _metric(quality, "valid_target_rows")),
        ("Template issues", _metric(quality, "template_issue_count")),
        ("Preprocessed samples", _metric(preprocessing, "rows_after")),
        ("Core supported factors", _metric(core, "supported_factor_count")),
        ("Internal match rate", join.get("match_rate")),
        ("Duplicate internal key rows", join.get("duplicate_internal_key_rows")),
        ("Selected p80 recall", decision.get("selected_model_p80_recall")),
        ("Selected p80 false negative rate", decision.get("selected_model_p80_false_negative_rate")),
    ]
    metric_html = "\n".join(
        f"<tr><th>{html.escape(label)}</th><td>{_safe_text(value)}</td></tr>" for label, value in rows
    )
    evidence_html = "\n".join(
        f"<tr><th>{_safe_text(level)}</th><td>{_safe_text(count)}</td></tr>"
        for level, count in evidence_counts.items()
    )
    if not evidence_html:
        evidence_html = "<tr><td colspan=\"2\">No internal factor evidence was generated.</td></tr>"

    review_rows = []
    for row in top_review:
        review_rows.append(
            "<tr>"
            + "".join(f"<td>{_safe_text(value)}</td>" for value in row.values())
            + "</tr>"
        )
    review_header = ""
    if top_review:
        review_header = "<tr>" + "".join(f"<th>{_safe_text(col)}</th>" for col in top_review[0].keys()) + "</tr>"
    review_html = review_header + "\n".join(review_rows)
    if not review_html:
        review_html = "<tr><td>No review priority list was generated for this run.</td></tr>"

    roadmap_rows = "\n".join(
        "<tr>"
        f"<th>{_safe_text(item['field_group'])}</th>"
        f"<td>{_safe_text(item['example_fields'])}</td>"
        f"<td>{_safe_text(item['business_use'])}</td>"
        "</tr>"
        for item in DATA_ROADMAP
    )

    output_files = payload.get("manifest", {}).get("output_files", {})
    file_html = "\n".join(
        f"<li><code>{_safe_text(name)}</code>: {_safe_text(path)}</li>" for name, path in output_files.items()
    )

    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>PhosphorAI Enterprise Summary</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; color: #1f2933; }}
    h1, h2 {{ color: #102a43; }}
    table {{ border-collapse: collapse; width: 100%; margin: 12px 0 24px; }}
    th, td {{ border: 1px solid #d9e2ec; padding: 8px 10px; text-align: left; }}
    th {{ background: #f0f4f8; }}
    code {{ background: #f0f4f8; padding: 2px 4px; }}
    .note {{ color: #52606d; }}
    .policy {{ background: #f8fafc; border: 1px solid #d9e2ec; padding: 16px; margin: 12px 0 24px; }}
    .policy p {{ margin: 6px 0; }}
  </style>
</head>
<body>
  <h1>PhosphorAI Enterprise Summary</h1>
  <p class="note">Generated for manual review and feasibility assessment. This report does not authorize automatic dosing or process control.</p>
  <h2>Run Metrics</h2>
  <table>{metric_html}</table>
  <h2>Internal Evidence Counts</h2>
  <table>{evidence_html}</table>
  <h2>Limited Human Review</h2>
  <div class="policy">
    <p><strong>Trigger:</strong> {_safe_text(review_policy.get("trigger"))}</p>
    <p><strong>Ranking logic:</strong> {_safe_text(review_policy.get("ranking_logic"))}</p>
    <p><strong>Manual action:</strong> {_safe_text(review_policy.get("manual_action"))}</p>
    <p><strong>Operational scenario:</strong> {_safe_text(review_policy.get("operational_scenario"))}</p>
    <p><strong>Decision boundary:</strong> {_safe_text(review_policy.get("decision_boundary"))}</p>
  </div>
  <h2>Top Review Candidates</h2>
  <table>{review_html}</table>
  <h2>PGEO Data Roadmap Discussion</h2>
  <p class="note">Use this field list to confirm which data can be collected, how often it is available, and whether it can be joined reliably to the quality table.</p>
  <table>
    <tr><th>Field group</th><th>Example fields</th><th>Business use</th></tr>
    {roadmap_rows}
  </table>
  <h2>Output Files</h2>
  <ul>{file_html}</ul>
</body>
</html>
"""
    path = run_dir / "enterprise_summary.html"
    path.write_text(document, encoding="utf-8")
    return str(path)


def run_enterprise_delivery(config: EnterpriseRunConfig) -> dict:
    """Run the enterprise UI workflow and package outputs in the selected directory."""
    if not config.quality_inputs:
        raise ValueError("At least one quality-table input is required.")
    if not config.output_dir:
        raise ValueError("An output directory is required.")

    root = Path(config.output_dir).expanduser()
    run_id = config.run_id or f"phosphorai_run_{_timestamp()}"
    run_dir = root / run_id
    processed_dir = run_dir / "processed"
    preprocessing_dir = run_dir / "preprocessing"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "run_log.txt"
    logger = RunLogger(log_path)
    logger.write(
        "run_started",
        run_id=run_id,
        target_col=config.target_col,
        year_filter=config.year_filter,
        quality_inputs=len(config.quality_inputs),
        internal_inputs=len(config.internal_inputs),
    )

    try:
        logger.write("data_validation_started")
        quality_summary = run_data_validation(
            config.quality_inputs,
            year_filter=config.year_filter,
            target_col=config.target_col,
            join_keys=config.join_keys,
            factor_fields=config.quality_factor_fields,
            process_response_fields=config.process_response_fields,
        )
        quality_json = _write_json(run_dir / "quality_validation_summary.json", quality_summary)
        logger.write(
            "data_validation_completed",
            status=quality_summary.get("status"),
            rows=quality_summary.get("row_count"),
            output=Path(quality_json).name,
        )

        logger.write("preprocessing_started")
        preprocessing_summary = run_preprocessing(
            input_paths=config.quality_inputs,
            processed_dir=processed_dir,
            report_dir=preprocessing_dir,
            target_col=config.target_col,
            vif_threshold=config.vif_threshold,
            year_filter=config.year_filter,
        )
        preprocessing_json = _write_json(run_dir / "preprocessing_summary.json", preprocessing_summary)
        model_source_path = processed_dir / "model_source.csv"
        logger.write(
            "preprocessing_completed",
            rows_after=preprocessing_summary.get("rows_after"),
            output=Path(preprocessing_json).name,
        )

        logger.write("core_factor_screening_started")
        core_summary = run_core_factor_check(
            model_source_path,
            target_col=config.target_col,
            output_dir=run_dir,
        )
        core_json = _write_json(run_dir / "core_factor_screening.json", core_summary)
        logger.write(
            "core_factor_screening_completed",
            status=core_summary.get("status"),
            supported=core_summary.get("supported_factor_count"),
            output=Path(core_json).name,
        )

        internal_summary = None
        internal_json = None
        if config.run_internal_factors and config.internal_inputs:
            logger.write("internal_factor_validation_started")
            internal_summary = run_internal_factor_check(
                input_path=config.internal_inputs,
                quality_input=model_source_path,
                join_keys=config.join_keys,
                target_col=config.target_col,
                factor_fields=config.internal_factor_fields,
                process_response_fields=config.internal_process_response_fields,
                output_dir=run_dir,
            )
            internal_json = _write_json(run_dir / "internal_factor_evidence.json", internal_summary)
            join = internal_summary.get("join_diagnostics", {})
            logger.write(
                "internal_factor_validation_completed",
                status=internal_summary.get("status"),
                match_rate=join.get("match_rate"),
                output=Path(internal_json).name,
            )
        else:
            logger.write(
                "internal_factor_validation_skipped",
                enabled=config.run_internal_factors,
                internal_inputs=len(config.internal_inputs),
            )

        risk_summary = None
        risk_json = None
        risk_ranking_path = run_dir / "risk_scoring_batch_ranking.csv"
        review_candidates_path = run_dir / "limited_human_review_top_candidates.csv"
        if config.run_risk_scoring:
            logger.write("risk_scoring_started")
            risk_summary = run_risk_scoring(model_source_path, run_dir, target_col=config.target_col)
            risk_json = _write_json(run_dir / "risk_scoring_summary.json", risk_summary)
            logger.write(
                "risk_scoring_completed",
                threshold=risk_summary.get("limited_human_review", {}).get("threshold_name"),
                output=Path(risk_json).name,
            )
        else:
            logger.write("risk_scoring_skipped", enabled=config.run_risk_scoring)
    except Exception as exc:
        logger.write("run_failed", error_type=type(exc).__name__, error=str(exc))
        raise

    manifest = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "config": json_safe(config.__dict__),
        "output_files": {
            "run_log": str(log_path),
            "quality_validation_summary": quality_json,
            "preprocessing_summary": preprocessing_json,
            "model_source": str(model_source_path),
            "core_factor_screening": str(run_dir / "core_factor_screening.csv"),
            "core_factor_screening_json": core_json,
        },
    }
    if internal_json:
        manifest["output_files"]["internal_factor_evidence"] = str(run_dir / "internal_factor_evidence.csv")
        manifest["output_files"]["internal_factor_evidence_json"] = internal_json
        manifest["output_files"]["internal_factor_joined_sample"] = str(run_dir / "internal_factor_joined_sample.csv")
    if risk_json:
        manifest["output_files"]["risk_scoring_summary"] = risk_json
        manifest["output_files"]["risk_scoring_batch_ranking"] = str(risk_ranking_path)
        if review_candidates_path.exists():
            manifest["output_files"]["limited_human_review_top_candidates"] = str(review_candidates_path)
    manifest["output_files"]["enterprise_summary_html"] = str(run_dir / "enterprise_summary.html")

    payload = {
        "manifest": manifest,
        "quality_validation": quality_summary,
        "preprocessing": preprocessing_summary,
        "core_factors": core_summary,
        "internal_factors": internal_summary,
        "risk_scoring": risk_summary,
        "top_review_rows": _top_review_rows(review_candidates_path, risk_ranking_path),
        "data_roadmap": DATA_ROADMAP,
    }
    logger.write("enterprise_summary_started")
    html_path = _write_enterprise_html(run_dir, payload)
    logger.write("enterprise_summary_completed", output=Path(html_path).name)
    manifest_path = _write_json(run_dir / "run_manifest.json", manifest)
    logger.write("manifest_written", output=Path(manifest_path).name)
    logger.write("run_completed", status="complete")

    return {
        "workflow": "enterprise_delivery",
        "status": "complete",
        "run_id": run_id,
        "run_dir": str(run_dir),
        "model_source_path": str(model_source_path),
        "manifest_path": manifest_path,
        "run_log_path": str(log_path),
        "enterprise_summary_html": html_path,
        "quality_validation": quality_summary,
        "preprocessing": preprocessing_summary,
        "core_factors": core_summary,
        "internal_factors": internal_summary,
        "risk_scoring": risk_summary,
        "top_review_rows": payload["top_review_rows"],
        "data_roadmap": DATA_ROADMAP,
        "output_files": manifest["output_files"],
    }
