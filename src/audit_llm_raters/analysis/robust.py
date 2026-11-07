"""Sensitivity analyses for informative failures, balance, and contamination."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from audit_llm_raters.analysis.gtheory import multivariate_g


def analysis_matrices(scores: pd.DataFrame, failure_policy: str = "missing") -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    if failure_policy not in {"missing", "floor"}:
        raise ValueError("failure_policy must be 'missing' or 'floor'")
    work = scores.copy()
    if failure_policy == "floor":
        work.loc[work["parse_status"] != "ok", "parsed_score"] = work["scale_min"]
    else:
        work = work[work["parse_status"] == "ok"]
    work = work.dropna(subset=["parsed_score"])
    llm_frame = work.groupby(["essay_id", "model"])["parsed_score"].mean().unstack("model")
    human_frame = work.drop_duplicates("essay_id").set_index("essay_id")[["human_r1", "human_r2"]]
    joined = human_frame.join(llm_frame, how="inner").dropna()
    if len(joined) < 3 or llm_frame.shape[1] < 2:
        raise ValueError("Insufficient complete essays or models for sensitivity analysis")
    human = joined[["human_r1", "human_r2"]].to_numpy(dtype=float)
    llm = joined[list(llm_frame.columns)].to_numpy(dtype=float)
    metadata = pd.DataFrame({"n_essays": [len(joined)], "n_models": [llm.shape[1]], "failure_policy": [failure_policy]})
    return human, llm, metadata


def failure_sensitivity(scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for policy in ["missing", "floor"]:
        try:
            human, llm, metadata = analysis_matrices(scores, policy)
            result = multivariate_g(human, llm)
            rows.append({**metadata.iloc[0].to_dict(), **{key: result[key] for key in ["rho_HL", "pi_pt", "Phi_H", "Phi_L"]}})
        except ValueError as exc:
            rows.append({"failure_policy": policy, "error": str(exc)})
    return pd.DataFrame(rows)


def balanced_reestimate(scores: pd.DataFrame, n_essays: int = 100, seed: int = 20261107) -> dict[str, float]:
    human, llm, _ = analysis_matrices(scores, "missing")
    if len(human) < n_essays:
        raise ValueError("Not enough complete cases for requested balanced subsample")
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(len(human), n_essays, replace=False))
    return {key: float(value) for key, value in multivariate_g(human[indices], llm[indices]).items() if np.isscalar(value)}


def contamination_audit(original: pd.DataFrame, rewritten: pd.DataFrame) -> pd.DataFrame:
    """Compare pre-registered, quality-preserving rewrites; does not generate rewrites."""
    keys = ["essay_id", "model", "prompt_variant", "rubric_variant", "run_id"]
    columns = keys + ["parsed_score", "human_r1", "human_r2", "parse_status"]
    left = original[columns].rename(columns={"parsed_score": "original_score", "parse_status": "original_status"})
    right = rewritten[columns].rename(columns={"parsed_score": "rewrite_score", "parse_status": "rewrite_status"})
    pair = left.merge(right, on=keys, suffixes=("_original", "_rewrite"))
    pair = pair[(pair["original_status"] == "ok") & (pair["rewrite_status"] == "ok")].copy()
    pair["human_mean"] = pair[["human_r1_original", "human_r2_original"]].mean(axis=1)
    pair["abs_error_original"] = (pair["original_score"] - pair["human_mean"]).abs()
    pair["abs_error_rewrite"] = (pair["rewrite_score"] - pair["human_mean"]).abs()
    rows = []
    for model, group in pair.groupby("model"):
        difference = group["abs_error_original"] - group["abs_error_rewrite"]
        statistic = wilcoxon(difference) if np.any(difference != 0) else None
        rows.append({
            "model": model, "n": len(group),
            "mae_original": group["abs_error_original"].mean(),
            "mae_rewrite": group["abs_error_rewrite"].mean(),
            "original_minus_rewrite_mae": difference.mean(),
            "wilcoxon_pvalue": statistic.pvalue if statistic else 1.0,
        })
    return pd.DataFrame(rows)
