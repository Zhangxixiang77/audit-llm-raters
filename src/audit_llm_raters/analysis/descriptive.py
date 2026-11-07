"""Observed-score diagnostics that do not substitute for multivariate G estimates."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score


def _observed_frame(scores: pd.DataFrame, facets: list[str]) -> pd.DataFrame:
    valid = scores[scores["parse_status"] == "ok"].dropna(subset=["parsed_score"]).copy()
    return valid.groupby(["essay_id", *facets], as_index=False).agg(
        llm_score=("parsed_score", "mean"),
        human_r1=("human_r1", "first"),
        human_r2=("human_r2", "first"),
        scale_min=("scale_min", "first"),
        scale_max=("scale_max", "first"),
    ).dropna(subset=["human_r1", "human_r2"])


def _metrics(frame: pd.DataFrame) -> dict[str, float]:
    human = frame[["human_r1", "human_r2"]].mean(axis=1).to_numpy(dtype=float)
    llm = frame["llm_score"].to_numpy(dtype=float)
    error = llm - human
    rounded_human = np.rint(human).astype(int)
    rounded_llm = np.clip(
        np.rint(llm), frame["scale_min"].iloc[0], frame["scale_max"].iloc[0]
    ).astype(int)
    labels = list(range(int(frame["scale_min"].iloc[0]), int(frame["scale_max"].iloc[0]) + 1))
    pearson = float(np.corrcoef(human, llm)[0, 1]) if np.std(human) and np.std(llm) else np.nan
    return {
        "n_essays": len(frame),
        "human_mean": float(np.mean(human)),
        "llm_mean": float(np.mean(llm)),
        "human_sd": float(np.std(human, ddof=1)),
        "llm_sd": float(np.std(llm, ddof=1)),
        "pearson_r_observed": pearson,
        "spearman_rho_observed": float(spearmanr(human, llm).statistic),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "bias_llm_minus_human": float(np.mean(error)),
        "exact_accuracy_rounded": float(np.mean(rounded_llm == rounded_human)),
        "adjacent_accuracy_rounded": float(np.mean(np.abs(rounded_llm - rounded_human) <= 1)),
        "qwk_rounded": float(cohen_kappa_score(rounded_human, rounded_llm, weights="quadratic", labels=labels)),
    }


def observed_metrics(scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize each model overall and within each prompt/rubric configuration."""
    overall_rows = []
    for model, group in scores.groupby("model"):
        frame = _observed_frame(group, [])
        overall_rows.append({"model": model, **_metrics(frame)})
    config_rows = []
    facets = ["model", "prompt_variant", "rubric_variant"]
    for keys, group in scores.groupby(facets):
        frame = _observed_frame(group, ["prompt_variant", "rubric_variant"])
        config_rows.append(dict(zip(facets, keys, strict=True)) | _metrics(frame))
    return pd.DataFrame(overall_rows), pd.DataFrame(config_rows)


def bootstrap_observed_metrics(
    scores: pd.DataFrame, n_boot: int = 1000, seed: int = 20261107
) -> pd.DataFrame:
    """Essay-cluster bootstrap CIs for observed metrics, not universe-score G estimands."""
    rng = np.random.default_rng(seed)
    rows = []
    targets = ["pearson_r_observed", "spearman_rho_observed", "mae", "rmse", "bias_llm_minus_human"]
    for model, group in scores.groupby("model"):
        frame = _observed_frame(group, []).reset_index(drop=True)
        point = _metrics(frame)
        draws = []
        for _ in range(n_boot):
            indices = rng.integers(0, len(frame), len(frame))
            draws.append(_metrics(frame.iloc[indices]))
        boot = pd.DataFrame(draws)
        for target in targets:
            rows.append({
                "model": model,
                "estimand": target,
                "estimate": point[target],
                "ci_lower": boot[target].quantile(0.025),
                "ci_upper": boot[target].quantile(0.975),
                "boot_se": boot[target].std(),
                "n_boot": n_boot,
            })
    return pd.DataFrame(rows)
