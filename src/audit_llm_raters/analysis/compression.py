"""Human-baseline compression tests and cross-scale target stability."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _slope(y: np.ndarray, x: np.ndarray) -> float:
    pair = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(pair) < 3 or pair["x"].var() == 0:
        return np.nan
    return float(np.cov(pair["x"], pair["y"], ddof=1)[0, 1] / pair["x"].var())


def compression_audit(scores: pd.DataFrame, min_scale: float | None = None, max_scale: float | None = None) -> pd.DataFrame:
    """Compare LLM→human2 slopes with the human1→human2 attenuation baseline."""
    valid = scores[scores["parse_status"] == "ok"].copy()
    rows = []
    group_columns = ["dataset", "prompt_id", "model"]
    for keys, group in valid.groupby(group_columns, dropna=False):
        dataset, prompt_id, model = keys
        ensemble = group.groupby("essay_id").agg({
            "parsed_score": "mean", "human_r1": "first", "human_r2": "first",
            "scale_min": "first", "scale_max": "first",
        }).dropna()
        if len(ensemble) < 3:
            continue
        h1 = ensemble["human_r1"].to_numpy()
        h2 = ensemble["human_r2"].to_numpy()
        llm = ensemble["parsed_score"].to_numpy()
        baseline = _slope(h1, h2)
        observed = _slope(llm, h2)
        intercept = float(np.mean(llm) - observed * np.mean(h2))
        compression = 1 - observed
        target = intercept / compression if abs(compression) > 1e-8 else np.nan
        scale_min = ensemble["scale_min"].iloc[0] if min_scale is None else min_scale
        scale_max = ensemble["scale_max"].iloc[0] if max_scale is None else max_scale
        rows.append({
            "dataset": dataset, "prompt_id": prompt_id, "model": model, "n_essays": len(ensemble),
            "human_baseline_slope": baseline, "llm_slope": observed,
            "compression_ratio": observed / baseline if baseline else np.nan,
            "excess_compression_vs_human": baseline - observed,
            "implied_target": target,
            "scale_midpoint": (scale_min + scale_max) / 2,
            "human_modal_score": ensemble[["human_r1", "human_r2"]].stack().mode().iloc[0],
            "normalized_target": (target - scale_min) / (scale_max - scale_min),
            "normalized_midpoint": 0.5,
            "normalized_human_mode": (ensemble[["human_r1", "human_r2"]].stack().mode().iloc[0] - scale_min) / (scale_max - scale_min),
        })
    return pd.DataFrame(rows)


def target_stability(compression: pd.DataFrame) -> pd.DataFrame:
    rows = []
    hypotheses = {
        "stable_common_target": lambda group: pd.Series(group["normalized_target"].mean(), index=group.index),
        "scale_midpoint": lambda group: group["normalized_midpoint"],
        "human_modal_score": lambda group: group["normalized_human_mode"],
    }
    for model, group in compression.groupby("model"):
        for target, predicted in hypotheses.items():
            values = predicted(group)
            rows.append({
                "model": model, "target_hypothesis": target, "n_prompts": len(group),
                "observed_target_mean": group["normalized_target"].mean(),
                "observed_target_sd_across_prompts": group["normalized_target"].std(),
                "rmse_to_observed_target": np.sqrt(np.mean((group["normalized_target"] - values) ** 2)),
            })
    return pd.DataFrame(rows)
