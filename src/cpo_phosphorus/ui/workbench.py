"""Streamlit workbench for enterprise modular validation."""

from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

from cpo_phosphorus.workflows.core_factors import run_core_factor_check
from cpo_phosphorus.workflows.internal_factors import run_internal_factor_check
from cpo_phosphorus.workflows.risk_scoring import run_risk_scoring
from cpo_phosphorus.workflows.validate_data import run_data_validation


def main() -> None:
    st.set_page_config(page_title="PhosphorAI Risk Workbench", layout="wide")
    st.title("PhosphorAI Risk Workbench")
    st.caption("Enterprise modular validation prototype")
    st.info("Milestone 2 supports multi-file quality-table intake and template checks.")

    tab_upload, tab_core, tab_internal, tab_risk, tab_export = st.tabs(
        ["Upload & Validate", "Core Factors", "Internal Factors", "Risk Scoring", "Export"]
    )

    with tab_upload:
        st.subheader("Quality Table Inputs")
        uploaded_files = st.file_uploader(
            "Upload one or more quality-table Excel files",
            type=["xlsx", "xlsm", "xls"],
            accept_multiple_files=True,
        )
        path_text = st.text_area(
            "Or enter server-side file/directory paths, one per line",
            value="",
            help="Use this when the app is deployed inside the same environment as the data files.",
        )

        st.subheader("Field Mapping")
        target_col = st.text_input("Target column", value="feed_p_ppm")
        join_keys = st.text_input("Quality-table join keys", value="")
        factor_fields = st.text_input("Factor fields", value="")
        process_fields = st.text_input("Process-response fields", value="")
        year_filter = st.text_input("Optional year filter", value="all")

        if st.button("Validate inputs", type="primary"):
            input_paths = [line.strip() for line in path_text.splitlines() if line.strip()]
            with tempfile.TemporaryDirectory() as tmp:
                for uploaded in uploaded_files:
                    target = Path(tmp) / uploaded.name
                    target.write_bytes(uploaded.getbuffer())
                    input_paths.append(str(target))

                if not input_paths:
                    st.error("Upload at least one Excel file or enter a server-side path.")
                else:
                    try:
                        summary = run_data_validation(
                            input_paths,
                            year_filter=year_filter,
                            target_col=target_col,
                            join_keys=join_keys,
                            factor_fields=factor_fields,
                            process_response_fields=process_fields,
                        )
                    except Exception as exc:  # pragma: no cover - UI guardrail
                        st.exception(exc)
                    else:
                        st.success(f"Validation status: {summary['status']}")
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Files", summary["file_count"])
                        col2.metric("Rows", summary["row_count"])
                        col3.metric("Target rows", summary["valid_target_rows"])
                        col4.metric("Template issues", summary["template_issue_count"])
                        st.json(summary)

    with tab_core:
        st.subheader("Core Quality Factor Screening")
        core_input = st.text_input("Processed quality table", value="local_runs/final_feed_2024_2025/processed/model_source.csv")
        core_target = st.text_input("Core target", value="feed_p_ppm")
        if st.button("Run core factor screening"):
            try:
                summary = run_core_factor_check(core_input, target_col=core_target)
            except Exception as exc:  # pragma: no cover - UI guardrail
                st.exception(exc)
            else:
                st.success(f"Core screening status: {summary['status']}")
                st.metric("Supported factors", summary["supported_factor_count"])
                st.json(summary)

    with tab_internal:
        st.subheader("Direct Internal Factor Validation")
        quality_input = st.text_input("Quality/model table for join", value="local_runs/final_feed_2024_2025/processed/model_source.csv")
        internal_uploads = st.file_uploader(
            "Upload one or more internal factor tables",
            type=["csv", "xlsx", "xlsm", "xls"],
            accept_multiple_files=True,
        )
        internal_input = st.text_area("Or enter internal factor file/directory paths, one per line", value="")
        internal_keys = st.text_input("Join keys", value="date,feed_tank")
        internal_fields = st.text_input("Internal factor fields", value="")
        internal_process = st.text_input("Process-response fields", value="")
        internal_target = st.text_input("Internal validation target", value="feed_p_ppm")
        if st.button("Run internal factor validation"):
            internal_paths = [line.strip() for line in internal_input.splitlines() if line.strip()]
            with tempfile.TemporaryDirectory() as tmp:
                for uploaded in internal_uploads:
                    target = Path(tmp) / uploaded.name
                    target.write_bytes(uploaded.getbuffer())
                    internal_paths.append(str(target))
                try:
                    summary = run_internal_factor_check(
                        input_path=internal_paths or None,
                        quality_input=quality_input or None,
                        join_keys=internal_keys,
                        target_col=internal_target,
                        factor_fields=internal_fields,
                        process_response_fields=internal_process,
                    )
                except Exception as exc:  # pragma: no cover - UI guardrail
                    st.exception(exc)
                else:
                    st.success(f"Internal validation status: {summary['status']}")
                    st.json(summary)

    with tab_risk:
        st.subheader("Risk Scoring")
        risk_input = st.text_input("Risk input model_source.csv", value="local_runs/final_feed_2024_2025/processed/model_source.csv")
        risk_output = st.text_input("Risk output directory", value="local_runs/ui_risk_scoring/reports/risk_scoring")
        risk_target = st.text_input("Risk target", value="feed_p_ppm")
        if st.button("Run risk scoring"):
            try:
                summary = run_risk_scoring(risk_input, risk_output, target_col=risk_target)
            except Exception as exc:  # pragma: no cover - UI guardrail
                st.exception(exc)
            else:
                st.success("Risk scoring complete")
                decision = summary.get("decision_summary", {})
                st.metric("Selected p80 recall", decision.get("selected_model_p80_recall"))
                st.metric("Selected p80 false negative rate", decision.get("selected_model_p80_false_negative_rate"))
                st.json(summary)

    with tab_export:
        st.subheader("Export")
        st.write("Workflow modules write CSV and JSON-compatible summaries to their selected output directories.")
        st.caption("Exports are designed for enterprise review and manual decision support, not automatic process control.")


if __name__ == "__main__":
    main()
