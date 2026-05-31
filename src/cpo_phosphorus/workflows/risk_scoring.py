"""Independent risk-scoring workflow wrapper."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from cpo_phosphorus.models.feed_model_optimized import run_pipeline


def run_risk_scoring(input_path, output_dir, target_col="feed_p_ppm"):
    """Run the existing optimized feed-risk model as a standalone module."""
    summary = run_pipeline(input_path=input_path, output_dir=output_dir, target_col=target_col)
    output = Path(output_dir)
    ranking_path = None
    predictions_path = output / "feed_model_optimized_model_predictions.csv"
    if predictions_path.exists():
        predictions = pd.read_csv(predictions_path)
        best = summary.get("best_random_split_model", {})
        mask = predictions["feature_group"].eq(best.get("feature_group")) & predictions["model"].eq(best.get("model"))
        ranking = predictions.loc[mask].copy()
        threshold_rows = [
            row for row in summary.get("risk_thresholds", []) if row.get("threshold_name") == "p80"
        ]
        threshold = threshold_rows[0]["threshold_ppm"] if threshold_rows else None
        if not ranking.empty:
            if threshold is not None:
                ranking["risk_threshold_name"] = "p80"
                ranking["risk_threshold_ppm"] = threshold
                ranking["predicted_high_risk"] = ranking["predicted"] >= threshold
                ranking["actual_high_risk"] = ranking["actual"] >= threshold
            ranking = ranking.sort_values("predicted", ascending=False)
            ranking_path = output / "risk_scoring_batch_ranking.csv"
            ranking.to_csv(ranking_path, index=False)

    summary["workflow"] = "risk_scoring"
    summary["milestone"] = 6
    if ranking_path:
        summary.setdefault("output_files", {})["batch_ranking"] = ranking_path.name
    summary["notes"] = [
        "This workflow reuses the existing optimized feed model.",
        "Outputs are for manual review priority and feasibility checks, not automatic control.",
    ]
    return summary
