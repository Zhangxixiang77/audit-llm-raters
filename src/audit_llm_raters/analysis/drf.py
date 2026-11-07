"""Rater-by-ELL differential rater functioning with cluster bootstrap inference."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from audit_llm_raters.analysis.mfrm import MFRMData, fit_mfrm_map


def drf_bootstrap(
    long_frame: pd.DataFrame,
    scale_min: int,
    scale_max: int,
    n_boot: int = 200,
    seed: int = 20261107,
    substantive_threshold: float = 0.5,
) -> pd.DataFrame:
    work = long_frame.copy()
    if "rater_type" in work:
        work = work[work["rater_type"] == "L"]
    if work.empty or work["ell"].nunique() < 2:
        raise ValueError("DRF requires LLM ratings in both ELL groups")
    original_data = MFRMData.from_frame(work, scale_min=scale_min, scale_max=scale_max)
    original = fit_mfrm_map(original_data, drf=True)
    if original.drf is None:
        raise RuntimeError("DRF model did not return interaction estimates")
    rng = np.random.default_rng(seed)
    essays = work["essay_id"].astype(str).unique()
    grouped = {essay: group for essay, group in work.groupby(work["essay_id"].astype(str))}
    draws = np.full((n_boot, len(original_data.rater_levels)), np.nan)
    failures = 0
    for b in range(n_boot):
        sampled = rng.choice(essays, size=len(essays), replace=True)
        pieces = []
        for j, essay in enumerate(sampled):
            piece = grouped[essay].copy()
            piece["essay_id"] = f"boot{b}_{j}_{essay}"
            pieces.append(piece)
        try:
            boot_frame = pd.concat(pieces, ignore_index=True)
            boot_data = MFRMData.from_frame(boot_frame, scale_min=scale_min, scale_max=scale_max)
            if boot_data.rater_levels != original_data.rater_levels:
                raise ValueError("Bootstrap omitted a crossed rater")
            fitted = fit_mfrm_map(boot_data, drf=True, initial=original.x, maxiter=500)
            draws[b] = fitted.drf
        except (ValueError, RuntimeError, FloatingPointError):
            failures += 1
    if failures / n_boot > 0.2:
        raise RuntimeError(f"Too many failed DRF bootstrap fits: {failures}/{n_boot}")
    m = len(original_data.rater_levels)
    alpha = 0.05 / m
    rows = []
    for index, rater in enumerate(original_data.rater_levels):
        values = draws[:, index]
        values = values[np.isfinite(values)]
        p_two_sided = min(1.0, 2 * min((np.sum(values <= 0) + 1) / (len(values) + 1), (np.sum(values >= 0) + 1) / (len(values) + 1)))
        rows.append({
            "rater_id": rater,
            "drf_ell_logit": original.drf[index],
            "ci_lower_bonferroni": np.quantile(values, alpha / 2),
            "ci_upper_bonferroni": np.quantile(values, 1 - alpha / 2),
            "p_bootstrap": p_two_sided,
            "p_bonferroni": min(1.0, p_two_sided * m),
            "substantive_0_5_logit": abs(original.drf[index]) >= substantive_threshold,
            "significant_bonferroni": p_two_sided * m < 0.05,
            "n_boot_valid": len(values),
        })
    return pd.DataFrame(rows)


def ell_group_qwk(scores: pd.DataFrame) -> pd.DataFrame:
    valid = scores[(scores["parse_status"] == "ok") & scores["ell"].notna()].copy()
    rows = []
    for (model, ell), group in valid.groupby(["model", "ell"]):
        ensemble = group.groupby("essay_id").agg({"parsed_score": "mean", "human_r1": "first", "human_r2": "first"}).dropna()
        human_mean = ensemble[["human_r1", "human_r2"]].mean(axis=1).round().astype(int)
        llm_rounded = ensemble["parsed_score"].round().astype(int)
        labels = sorted(set(human_mean) | set(llm_rounded))
        rows.append({
            "model": model, "ell": int(ell), "n": len(ensemble),
            "qwk_llm_human_mean": cohen_kappa_score(llm_rounded, human_mean, weights="quadratic", labels=labels),
        })
    return pd.DataFrame(rows)
