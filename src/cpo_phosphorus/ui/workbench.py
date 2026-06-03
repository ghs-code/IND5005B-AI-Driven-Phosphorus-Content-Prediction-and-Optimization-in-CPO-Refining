"""Streamlit workbench for enterprise delivery runs."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import streamlit as st

from cpo_phosphorus.workflows.delivery import EnterpriseRunConfig, run_enterprise_delivery

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SKIP_BROWSER_DIRS = {".git", ".venv", "__pycache__", ".pytest_cache", ".mypy_cache", ".cache"}


def _split_paths(value: str) -> list[str]:
    return [line.strip() for line in value.splitlines() if line.strip()]


def _required_label(label: str) -> None:
    st.markdown(f"{label} <span style='color:#d92d20'>*</span>", unsafe_allow_html=True)


def _optional_label(label: str) -> None:
    st.markdown(label)


def _dedupe_paths(paths: list[str]) -> list[str]:
    seen = set()
    unique = []
    for path in paths:
        if not path:
            continue
        normalized = str(Path(path).expanduser())
        if normalized in seen:
            continue
        seen.add(normalized)
        unique.append(normalized)
    return unique


def _resolve_local_path(value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _display_path(value: str) -> str:
    path = Path(value)
    try:
        label = path.relative_to(PROJECT_ROOT)
    except ValueError:
        label = path
    marker = "/" if path.is_dir() else ""
    return f"{label}{marker}"


def _discover_directories(root_value: str, max_depth: int = 3) -> list[str]:
    root = _resolve_local_path(root_value)
    if not root.exists():
        return [str(root)]

    if root.is_file():
        root = root.parent

    directories = [root]
    pending = [(root, 0)]
    while pending and len(directories) < 250:
        parent, depth = pending.pop(0)
        if depth >= max_depth:
            continue
        try:
            children = sorted(parent.iterdir())
        except OSError:
            continue
        for child in children:
            if child.name.startswith(".") or child.name in SKIP_BROWSER_DIRS:
                continue
            if child.is_dir():
                directories.append(child)
                pending.append((child, depth + 1))
            if len(directories) >= 250:
                break
    return [str(path) for path in directories]


def _output_browser_roots(default_output: str) -> list[str]:
    default_path = _resolve_local_path(default_output)
    candidates = [
        default_path,
        default_path.parent,
        PROJECT_ROOT / "local_data",
        PROJECT_ROOT,
        Path.home(),
    ]
    roots = []
    seen = set()
    for path in candidates:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        roots.append(key)
    return roots


def _directory_picker(label: str, root_value: str, required: bool = False) -> str:
    root = st.selectbox(
        f"{label} browser root",
        options=_output_browser_roots(root_value),
        index=0,
        format_func=_display_path,
    )
    options = _discover_directories(root)
    if required:
        _required_label(label)
    selected = st.selectbox(
        label,
        options=options,
        index=0,
        format_func=_display_path,
        label_visibility="collapsed" if required else "visible",
    )
    selected_path = Path(selected)
    if not selected_path.exists():
        st.caption("Selected directory will be created when the delivery run starts.")
    return selected


def _save_uploads(uploaded_files, directory: Path) -> list[str]:
    paths = []
    for uploaded in uploaded_files or []:
        target = directory / uploaded.name
        target.write_bytes(uploaded.getbuffer())
        paths.append(str(target))
    return paths


def _validate_required_fields(
    output_dir: str,
    target_col: str,
    year_filter: str,
    quality_uploads,
    quality_paths: str,
    run_internal: bool,
    internal_uploads,
    internal_paths: str,
    join_keys: str,
) -> list[str]:
    missing = []
    if not str(output_dir).strip():
        missing.append("Output directory")
    if not _split_paths(quality_paths) and not quality_uploads:
        missing.append("Quality-table Excel input")
    if not target_col.strip():
        missing.append("Target column")
    if not year_filter.strip():
        missing.append("Year filter")
    has_internal_input = bool(_split_paths(internal_paths) or internal_uploads)
    if run_internal and has_internal_input and not join_keys.strip():
        missing.append("Join keys for internal factor validation")
    return missing


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


def _show_limited_review(result: dict) -> None:
    risk = result.get("risk_scoring") or {}
    policy = risk.get("limited_human_review") or {}
    st.subheader("Limited Human Review")
    if policy:
        st.write(f"Trigger: {policy.get('trigger', 'n/a')}")
        st.write(f"Ranking logic: {policy.get('ranking_logic', 'n/a')}")
        st.write(f"Manual action: {policy.get('manual_action', 'n/a')}")
        st.write(f"Decision boundary: {policy.get('decision_boundary', 'n/a')}")
    else:
        st.write("Run risk scoring to generate the manual review policy.")

    top_review = result.get("top_review_rows") or []
    if top_review:
        st.dataframe(top_review, use_container_width=True)


def _show_data_roadmap(result: dict) -> None:
    roadmap = result.get("data_roadmap") or []
    st.subheader("PGEO Data Roadmap Discussion")
    if roadmap:
        st.dataframe(roadmap, use_container_width=True)
    else:
        st.write("No data roadmap was generated for this run.")


def main() -> None:
    st.set_page_config(page_title="PhosphorAI Risk Workbench", layout="wide")
    st.title("PhosphorAI Risk Workbench")
    st.caption("Enterprise UI delivery")

    default_output = os.getenv("PHOSPHORAI_DEFAULT_OUTPUT_DIR", "local_data/deliveries")
    default_target = os.getenv("CPO_TARGET_COL", "feed_p_ppm")
    default_vif = float(os.getenv("CPO_VIF_THRESHOLD", "10"))

    with st.sidebar:
        st.header("Run")
        output_dir = _directory_picker("Output directory", default_output, required=True)
        _required_label("Target column")
        target_col = st.text_input("Target column", value=default_target, label_visibility="collapsed")
        _required_label("Year filter")
        year_filter = st.text_input("Year filter", value="all", label_visibility="collapsed")
        vif_threshold = st.number_input("VIF threshold", min_value=1.0, max_value=100.0, value=default_vif)
        run_internal = st.checkbox("Internal factors", value=True)
        run_risk = st.checkbox("Risk scoring", value=True)
        st.markdown("<span style='color:#d92d20'>*</span> Required", unsafe_allow_html=True)

    tab_run, tab_results, tab_files = st.tabs(["Run", "Results", "Files"])

    with tab_run:
        st.subheader("Quality Tables")
        _required_label("Upload quality-table Excel files")
        quality_uploads = st.file_uploader(
            "Upload quality-table Excel files",
            type=["xlsx", "xlsm", "xls"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )
        with st.expander("Advanced quality path input"):
            quality_paths = st.text_area("Server-side quality paths", value="", height=80)

        st.subheader("Internal Factors")
        _optional_label("Upload internal factor tables")
        internal_uploads = st.file_uploader(
            "Upload internal factor tables",
            type=["csv", "xlsx", "xlsm", "xls"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )
        with st.expander("Advanced internal factor path input"):
            internal_paths = st.text_area("Additional server-side internal factor paths", value="", height=80)

        st.subheader("Mapping")
        join_keys_required = run_internal and bool(_split_paths(internal_paths) or internal_uploads)
        if join_keys_required:
            _required_label("Join keys")
        join_keys = st.text_input(
            "Join keys",
            value="date,feed_tank",
            label_visibility="collapsed" if join_keys_required else "visible",
        )
        quality_factor_fields = st.text_input("Quality factor fields", value="")
        process_response_fields = st.text_input("Quality process-response fields", value="")
        internal_factor_fields = st.text_input("Internal factor fields", value="")
        internal_process_fields = st.text_input("Internal process-response fields", value="")

        missing_required = _validate_required_fields(
            output_dir=output_dir,
            target_col=target_col,
            year_filter=year_filter,
            quality_uploads=quality_uploads,
            quality_paths=quality_paths,
            run_internal=run_internal,
            internal_uploads=internal_uploads,
            internal_paths=internal_paths,
            join_keys=join_keys,
        )
        if missing_required:
            st.warning("Required before run: " + ", ".join(missing_required))

        if st.button("Run delivery", type="primary", disabled=bool(missing_required)):
            with tempfile.TemporaryDirectory() as tmp:
                tmp_path = Path(tmp)
                quality_inputs = _dedupe_paths(
                    _split_paths(quality_paths) + _save_uploads(quality_uploads, tmp_path)
                )
                internal_inputs = _dedupe_paths(
                    _split_paths(internal_paths) + _save_uploads(internal_uploads, tmp_path)
                )
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
            _show_limited_review(result)
            _show_data_roadmap(result)

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
