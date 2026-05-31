"""Streamlit workbench for enterprise delivery runs."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import streamlit as st

from cpo_phosphorus.workflows.delivery import EnterpriseRunConfig, run_enterprise_delivery


def _split_paths(value: str) -> list[str]:
    return [line.strip() for line in value.splitlines() if line.strip()]


def _save_uploads(uploaded_files, directory: Path) -> list[str]:
    paths = []
    for uploaded in uploaded_files:
        target = directory / uploaded.name
        target.write_bytes(uploaded.getbuffer())
        paths.append(str(target))
    return paths


def _metric_value(summary: dict | None, key: str, default="n/a"):
    if not summary:
        return default
    value = summary.get(key, default)
    return default if value is None else value


def _nested_value(summary: dict | None, *keys, default="n/a"):
    value = summary or {}
    for key in keys:
        if not isinstance(value, dict):
            return default
        value = value.get(key)
    return default if value is None else value


def _show_quality(summary: dict | None) -> None:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Files", _metric_value(summary, "file_count"))
    col2.metric("Rows", _metric_value(summary, "row_count"))
    col3.metric("Target rows", _metric_value(summary, "valid_target_rows"))
    col4.metric("Template issues", _metric_value(summary, "template_issue_count"))


def _show_internal(summary: dict | None) -> None:
    join = summary.get("join_diagnostics", {}) if summary else {}
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Status", _metric_value(summary, "status"))
    col2.metric("Match rate", join.get("match_rate", "n/a"))
    col3.metric("Matched rows", join.get("matched_rows", "n/a"))
    col4.metric("Duplicate keys", join.get("duplicate_internal_key_rows", "n/a"))


def _show_risk(summary: dict | None) -> None:
    col1, col2, col3 = st.columns(3)
    col1.metric("Samples", _metric_value(summary, "n_samples"))
    col2.metric("p80 recall", _nested_value(summary, "decision_summary", "selected_model_p80_recall"))
    col3.metric(
        "p80 false negative",
        _nested_value(summary, "decision_summary", "selected_model_p80_false_negative_rate"),
    )


def main() -> None:
    st.set_page_config(page_title="PhosphorAI Risk Workbench", layout="wide")
    st.title("PhosphorAI Risk Workbench")
    st.caption("Enterprise UI delivery")

    default_input = os.getenv("PHOSPHORAI_DEFAULT_INPUT_DIR", "local_data/raw")
    default_output = os.getenv("PHOSPHORAI_DEFAULT_OUTPUT_DIR", "local_data/deliveries")
    default_target = os.getenv("CPO_TARGET_COL", "feed_p_ppm")
    default_vif = float(os.getenv("CPO_VIF_THRESHOLD", "10"))

    with st.sidebar:
        st.header("Run")
        output_dir = st.text_input("Output directory", value=default_output)
        target_col = st.text_input("Target column", value=default_target)
        year_filter = st.text_input("Year filter", value="all")
        vif_threshold = st.number_input("VIF threshold", min_value=1.0, max_value=100.0, value=default_vif)
        run_internal = st.checkbox("Internal factors", value=True)
        run_risk = st.checkbox("Risk scoring", value=True)

    tab_run, tab_results, tab_files = st.tabs(["Run", "Results", "Files"])

    with tab_run:
        st.subheader("Quality Tables")
        quality_uploads = st.file_uploader(
            "Upload quality-table Excel files",
            type=["xlsx", "xlsm", "xls"],
            accept_multiple_files=True,
        )
        quality_paths = st.text_area("Server-side quality paths", value=default_input, height=90)

        st.subheader("Internal Factors")
        internal_uploads = st.file_uploader(
            "Upload internal factor tables",
            type=["csv", "xlsx", "xlsm", "xls"],
            accept_multiple_files=True,
        )
        internal_paths = st.text_area("Server-side internal factor paths", value="", height=90)

        st.subheader("Mapping")
        join_keys = st.text_input("Join keys", value="date,feed_tank")
        quality_factor_fields = st.text_input("Quality factor fields", value="")
        process_response_fields = st.text_input("Quality process-response fields", value="")
        internal_factor_fields = st.text_input("Internal factor fields", value="")
        internal_process_fields = st.text_input("Internal process-response fields", value="")

        if st.button("Run delivery", type="primary"):
            with tempfile.TemporaryDirectory() as tmp:
                tmp_path = Path(tmp)
                quality_inputs = _split_paths(quality_paths) + _save_uploads(quality_uploads, tmp_path)
                internal_inputs = _split_paths(internal_paths) + _save_uploads(internal_uploads, tmp_path)
                config = EnterpriseRunConfig(
                    quality_inputs=quality_inputs,
                    internal_inputs=internal_inputs,
                    output_dir=output_dir,
                    target_col=target_col,
                    year_filter=year_filter,
                    join_keys=join_keys,
                    quality_factor_fields=quality_factor_fields,
                    process_response_fields=process_response_fields,
                    internal_factor_fields=internal_factor_fields,
                    internal_process_response_fields=internal_process_fields,
                    vif_threshold=vif_threshold,
                    run_internal_factors=run_internal,
                    run_risk_scoring=run_risk,
                )
                try:
                    with st.spinner("Running PhosphorAI delivery workflow..."):
                        result = run_enterprise_delivery(config)
                except Exception as exc:  # pragma: no cover - UI guardrail
                    st.exception(exc)
                else:
                    st.session_state["delivery_result"] = result
                    st.success(f"Run complete: {result['run_dir']}")

    result = st.session_state.get("delivery_result")
    with tab_results:
        if not result:
            st.info("Run a delivery workflow to view results.")
        else:
            st.subheader("Quality")
            _show_quality(result.get("quality_validation"))
            st.subheader("Core Factors")
            core = result.get("core_factors")
            col1, col2 = st.columns(2)
            col1.metric("Supported", _metric_value(core, "supported_factor_count"))
            col2.metric("Weak", _metric_value(core, "weak_factor_count"))
            st.subheader("Internal Factors")
            _show_internal(result.get("internal_factors"))
            st.subheader("Risk Scoring")
            _show_risk(result.get("risk_scoring"))

    with tab_files:
        if not result:
            st.info("Run a delivery workflow to view generated files.")
        else:
            st.subheader("Report Package")
            st.write(f"Run directory: `{result['run_dir']}`")
            st.write(f"HTML summary: `{result['enterprise_summary_html']}`")
            output_files = result.get("output_files", {})
            if output_files:
                st.dataframe(
                    [{"name": name, "path": path} for name, path in output_files.items()],
                    use_container_width=True,
                )
            with st.expander("Raw summary"):
                st.json(result)


if __name__ == "__main__":
    main()
