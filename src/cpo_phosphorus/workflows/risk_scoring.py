"""Independent risk-scoring workflow wrapper."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from cpo_phosphorus.models.feed_model_optimized import run_pipeline

REVIEW_TOP_N = 10


def limited_human_review_policy(threshold_name="p80", threshold_ppm=None, top_n=REVIEW_TOP_N):
    """Describe how the risk ranking should be used by quality teams."""
    threshold_text = threshold_name
    if threshold_ppm is not None:
        threshold_text = f"{threshold_name} ({threshold_ppm:.3f} ppm)"
    return {
        "workflow": "limited_human_review",
        "trigger": (
            "Add a batch/sample to the manual review queue when predicted feed phosphorus "
            f"is at or above the {threshold_text} prototype high-risk cutoff."
        ),
        "ranking_logic": (
            "Sort review candidates by predicted feed phosphorus risk in descending order; "
            "the highest predicted values receive the earliest lab follow-up attention."
        ),
        "manual_action": (
            "Quality or operations teams review the top-ranked candidates for prioritized "
            "lab follow-up, retesting discussion, and operator attention."
        ),
        "operational_scenario": (
            "For a daily or per-batch intake run, the workbench produces a review priority "
            "list. The team checks the top candidates first, then waits for authoritative "
            "laboratory phosphorus confirmation before any process decision."
        ),
        "decision_boundary": (
            "This queue does not authorize automatic batch isolation, acid dosing, "
            "bleaching-earth dosing, or replacement of laboratory phosphorus testing."
        ),
        "threshold_mode": "prototype_quantile",
        "threshold_name": threshold_name,
        "threshold_ppm": threshold_ppm,
        "top_n": top_n,
    }


def build_review_priority_ranking(predictions, summary, threshold_name="p80", top_n=REVIEW_TOP_N):
    """Build a descending manual-review priority list from model predictions."""
    best = summary.get("best_random_split_model", {})
    mask = predictions["feature_group"].eq(best.get("feature_group")) & predictions["model"].eq(best.get("model"))
    ranking = predictions.loc[mask].copy()
    threshold_rows = [row for row in summary.get("risk_thresholds", []) if row.get("threshold_name") == threshold_name]
    threshold = threshold_rows[0]["threshold_ppm"] if threshold_rows else None

    if ranking.empty:
        return ranking, pd.DataFrame(), limited_human_review_policy(threshold_name, threshold, top_n)

    if threshold is not None:
        ranking["risk_threshold_name"] = threshold_name
        ranking["risk_threshold_ppm"] = threshold
        ranking["predicted_high_risk"] = ranking["predicted"] >= threshold
        ranking["actual_high_risk"] = ranking["actual"] >= threshold
        ranking["review_trigger"] = ranking["predicted_high_risk"].map(
            {
                True: f"predicted_at_or_above_{threshold_name}",
                False: f"below_{threshold_name}_prototype_cutoff",
            }
        )
    else:
        ranking["review_trigger"] = "ranked_by_predicted_feed_p"

    ranking = ranking.sort_values("predicted", ascending=False).reset_index(drop=True)
    ranking.insert(0, "review_priority_rank", range(1, len(ranking) + 1))
    ranking["recommended_manual_action"] = (
        "Prioritize lab follow-up or retesting discussion; do not use for automatic control."
    )
    candidates = ranking.head(top_n).copy()
    return ranking, candidates, limited_human_review_policy(threshold_name, threshold, top_n)


def run_risk_scoring(input_path, output_dir, target_col="feed_p_ppm"):
    """Run the existing optimized feed-risk model as a standalone module."""
    summary = run_pipeline(input_path=input_path, output_dir=output_dir, target_col=target_col)
    output = Path(output_dir)
    ranking_path = None
    candidates_path = None
    review_policy = limited_human_review_policy()
    predictions_path = output / "feed_model_optimized_model_predictions.csv"
    if predictions_path.exists():
        predictions = pd.read_csv(predictions_path)
        ranking, candidates, review_policy = build_review_priority_ranking(predictions, summary)
        if not ranking.empty:
            ranking_path = output / "risk_scoring_batch_ranking.csv"
            ranking.to_csv(ranking_path, index=False)
            candidates_path = output / "limited_human_review_top_candidates.csv"
            candidates.to_csv(candidates_path, index=False)

    summary["workflow"] = "risk_scoring"
    summary["milestone"] = 6
    summary["limited_human_review"] = review_policy
    if ranking_path:
        summary.setdefault("output_files", {})["batch_ranking"] = ranking_path.name
    if candidates_path:
        summary.setdefault("output_files", {})["limited_human_review_top_candidates"] = candidates_path.name
    summary["notes"] = [
        "This workflow reuses the existing optimized feed model.",
        "The ranking is a Limited Human Review priority list, not an automatic control queue.",
        "Outputs are for manual review priority and feasibility checks, not automatic control.",
    ]
    return summary
