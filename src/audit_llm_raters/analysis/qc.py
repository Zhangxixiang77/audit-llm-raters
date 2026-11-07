"""Quality control, human noise floor, and pilot gates G2/G3."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from sklearn.metrics import cohen_kappa_score

from audit_llm_raters.analysis.gtheory import two_facet_g


def kendall_tau_a(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x)
    y = np.asarray(y)
    if len(x) != len(y) or len(x) < 2:
        return np.nan
    dx = np.sign(x[:, None] - x[None, :])
    dy = np.sign(y[:, None] - y[None, :])
    iu = np.triu_indices(len(x), 1)
    return float(np.sum(dx[iu] * dy[iu]) / (len(x) * (len(x) - 1) / 2))


def human_agreement(human_r1: pd.Series, human_r2: pd.Series) -> dict[str, float]:
    pair = pd.DataFrame({"r1": human_r1, "r2": human_r2}).dropna()
    if len(pair) < 2:
        return {key: np.nan for key in ["qwk", "tau_a", "tau_b", "Phi"]}
    labels = sorted(set(pair["r1"]) | set(pair["r2"]))
    return {
        "qwk": cohen_kappa_score(pair["r1"], pair["r2"], weights="quadratic", labels=labels),
        "tau_a": kendall_tau_a(pair["r1"].to_numpy(), pair["r2"].to_numpy()),
        "tau_b": kendalltau(pair["r1"], pair["r2"]).statistic,
        "Phi": float(two_facet_g(pair.to_numpy(), "h")["Phi"]),
    }


def failure_table(scores: pd.DataFrame, essays: pd.DataFrame | None = None) -> pd.DataFrame:
    work = scores.copy()
    work["failed"] = work["parse_status"] != "ok"
    groupings = [["model"], ["model", "rubric_variant"], ["model", "prompt_variant"]]
    if "ell" in work:
        groupings.append(["model", "ell"])
    if essays is not None and "human_r1" in essays:
        quality = essays[["essay_id", "human_r1", "human_r2"]].copy()
        quality["essay_quality"] = quality[["human_r1", "human_r2"]].mean(axis=1)
        work = work.merge(quality[["essay_id", "essay_quality"]], on="essay_id", how="left")
        work["quality_band"] = pd.qcut(work["essay_quality"].rank(method="first"), 4, labels=False)
        groupings.append(["model", "quality_band"])
    tables = []
    for grouping in groupings:
        if any(column not in work for column in grouping):
            continue
        table = work.groupby(grouping, dropna=False)["failed"].agg(["sum", "count", "mean"]).reset_index()
        table = table.rename(columns={"sum": "n_failed", "count": "n_calls", "mean": "failure_rate"})
        table["grouping"] = ":".join(grouping)
        tables.append(table)
    overall = pd.DataFrame({"n_failed": [work["failed"].sum()], "n_calls": [len(work)], "failure_rate": [work["failed"].mean()], "grouping": ["overall"]})
    return pd.concat([overall, *tables], ignore_index=True)


def residual_correlation(scores: pd.DataFrame) -> dict[str, float | pd.DataFrame]:
    valid = scores[scores["parse_status"] == "ok"].copy()
    valid["residual"] = valid["parsed_score"] - valid[["human_r1", "human_r2"]].mean(axis=1)
    residuals = valid.groupby(["essay_id", "model"])["residual"].mean().unstack("model")
    correlation = residuals.corr(min_periods=3)
    values = correlation.to_numpy()[np.triu_indices_from(correlation, 1)]
    values = values[np.isfinite(values)]
    return {
        "mean_pairwise_residual_correlation": float(values.mean()) if len(values) else np.nan,
        "min_pairwise_residual_correlation": float(values.min()) if len(values) else np.nan,
        "n_pairs": len(values),
        "correlation_matrix": correlation,
    }


def pilot_gates(scores: pd.DataFrame, failure_budget: float = 0.05, g2_threshold: float = 0.2) -> dict[str, object]:
    work = scores.copy()
    work["failed"] = work["parse_status"] != "ok"
    config_failure = work.groupby(["model", "prompt_variant", "rubric_variant"])["failed"].mean()
    g2 = residual_correlation(scores)
    g2_estimable = g2["n_pairs"] > 0
    return {
        "G2_mean_residual_correlation": g2["mean_pairwise_residual_correlation"],
        "G2_pass": bool(g2["mean_pairwise_residual_correlation"] >= g2_threshold) if g2_estimable else None,
        "G2_status": "estimable" if g2_estimable else "not_estimable_single_model",
        "G3_max_config_failure_rate": float(config_failure.max()),
        "G3_pass": bool(config_failure.max() < failure_budget),
        "config_failure_rates": config_failure.reset_index(name="failure_rate"),
        "residual_correlation_matrix": g2["correlation_matrix"],
    }
