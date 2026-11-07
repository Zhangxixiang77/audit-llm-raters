"""Balanced EMS generalizability theory and multivariate universe-score estimates."""

from __future__ import annotations

from collections.abc import Iterable
from itertools import combinations

import numpy as np
import pandas as pd


def _subsets(n: int) -> list[tuple[int, ...]]:
    return [subset for size in range(1, n + 1) for subset in combinations(range(n), size)]


def balanced_ems(values: np.ndarray, facet_names: list[str]) -> pd.DataFrame:
    """Exact balanced random-effects ANOVA for a fully crossed one-cell array.

    The final full interaction is the residual when there is one observation per cell.
    Negative method-of-moments estimates are retained; truncated values are a separate column.
    """
    array = np.asarray(values, dtype=float)
    if array.ndim != len(facet_names) or array.ndim < 2:
        raise ValueError("values.ndim must equal len(facet_names) and be at least two")
    if not np.isfinite(array).all():
        raise ValueError("Balanced EMS requires complete finite observations")
    shape = array.shape
    n = array.size
    subsets = _subsets(array.ndim)
    ms: dict[tuple[int, ...], float] = {}
    rows = []
    for effect in subsets:
        other = tuple(axis for axis in range(array.ndim) if axis not in effect)
        multiplier = int(np.prod([shape[axis] for axis in other])) if other else 1
        contrast = np.zeros(shape, dtype=float)
        for size in range(len(effect) + 1):
            for kept in combinations(effect, size):
                reduce_axes = tuple(axis for axis in range(array.ndim) if axis not in kept)
                marginal = array.mean(axis=reduce_axes, keepdims=True) if reduce_axes else array
                contrast += (-1) ** (len(effect) - size) * marginal
        sum_full = float(np.sum(contrast**2))
        sum_effect_cells = sum_full / multiplier
        ss = multiplier * sum_effect_cells
        df = int(np.prod([shape[axis] - 1 for axis in effect]))
        ms[effect] = ss / df
        rows.append({
            "effect": ":".join(facet_names[axis] for axis in effect),
            "_effect": effect,
            "df": df,
            "ss": ss,
            "ms": ms[effect],
        })
    variance = {effect: 0.0 for effect in subsets}
    for effect in reversed(subsets):
        explained = 0.0
        for superset in subsets:
            if len(superset) > len(effect) and set(effect).issubset(superset):
                coefficient = n / np.prod([shape[axis] for axis in superset])
                explained += coefficient * variance[superset]
        own_coefficient = n / np.prod([shape[axis] for axis in effect])
        variance[effect] = (ms[effect] - explained) / own_coefficient
    for row in rows:
        estimate = variance[row["_effect"]]
        row["variance"] = estimate
        row["variance_nonnegative"] = max(0.0, estimate)
        del row["_effect"]
    return pd.DataFrame(rows)


def _component(components: pd.DataFrame, name: str) -> float:
    return float(components.loc[components["effect"] == name, "variance"].iloc[0])


def two_facet_g(values: np.ndarray, rater_name: str = "r") -> dict[str, float | pd.DataFrame]:
    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or min(array.shape) < 2:
        raise ValueError("Expected a p × r matrix with at least two essays and raters")
    components = balanced_ems(array, ["p", rater_name])
    vp = _component(components, "p")
    vr = _component(components, rater_name)
    vpr = _component(components, f"p:{rater_name}")
    n_r = array.shape[1]
    relative = vpr / n_r
    absolute = relative + vr / n_r
    return {
        "n_essays": array.shape[0],
        "n_raters": n_r,
        "var_p": vp,
        "var_rater": vr,
        "var_pr": vpr,
        "E_rho_squared": vp / (vp + relative) if vp + relative > 0 else np.nan,
        "Phi": vp / (vp + absolute) if vp + absolute > 0 else np.nan,
        "components": components,
    }


def multivariate_g(human: np.ndarray, llm: np.ndarray) -> dict[str, float | pd.DataFrame]:
    human = np.asarray(human, dtype=float)
    llm = np.asarray(llm, dtype=float)
    if human.ndim != 2 or llm.ndim != 2 or human.shape[0] != llm.shape[0]:
        raise ValueError("human and llm must be p × r matrices with the same p")
    h = two_facet_g(human, "h")
    llm_g = two_facet_g(llm, "l")
    h_mean = human.mean(axis=1)
    l_mean = llm.mean(axis=1)
    covariance = float(np.cov(h_mean, l_mean, ddof=1)[0, 1])
    var_h = float(h["var_p"])
    var_l = float(llm_g["var_p"])
    rho = covariance / np.sqrt(var_h * var_l) if var_h > 0 and var_l > 0 else np.nan
    var_pt = (var_h + var_l) / 2 - covariance
    residual_h = float(h["var_pr"]) / human.shape[1]
    residual_l = float(llm_g["var_pr"]) / llm.shape[1]
    residual_mean = (residual_h + residual_l) / 2
    denominator = covariance + var_pt + residual_mean
    return {
        "rho_HL": rho,
        "cov_HL": covariance,
        "var_universe_H": var_h,
        "var_universe_L": var_l,
        "var_p": covariance,
        "var_pt": var_pt,
        "var_residual_H_per_essay": residual_h,
        "var_residual_L_per_essay": residual_l,
        "pi_pt": var_pt / denominator if denominator > 0 else np.nan,
        "Phi_H": float(h["Phi"]),
        "Phi_L": float(llm_g["Phi"]),
        "human_components": h["components"],
        "llm_components": llm_g["components"],
    }


def bootstrap_multivariate_g(
    human: np.ndarray, llm: np.ndarray, n_boot: int = 1000, seed: int = 20261107
) -> pd.DataFrame:
    human = np.asarray(human, dtype=float)
    llm = np.asarray(llm, dtype=float)
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n_boot):
        indices = rng.integers(0, human.shape[0], human.shape[0])
        result = multivariate_g(human[indices], llm[indices])
        rows.append({key: result[key] for key in ["rho_HL", "pi_pt", "Phi_H", "Phi_L"]})
    draws = pd.DataFrame(rows)
    estimates = multivariate_g(human, llm)
    summary = []
    for column in draws:
        summary.append({
            "estimand": column,
            "estimate": estimates[column],
            "ci_lower": draws[column].quantile(0.025),
            "ci_upper": draws[column].quantile(0.975),
            "boot_se": draws[column].std(),
            "n_boot": n_boot,
        })
    return pd.DataFrame(summary)


def d_study(human: np.ndarray, llm: np.ndarray, model_counts: Iterable[int]) -> pd.DataFrame:
    h = two_facet_g(human, "h")
    llm_g = two_facet_g(llm, "l")
    multivariate = multivariate_g(human, llm)
    rows = []
    for k in model_counts:
        if k < 1:
            raise ValueError("D-study model counts must be positive")
        phi = float(llm_g["var_p"]) / (
            float(llm_g["var_p"]) + (float(llm_g["var_pr"]) + float(llm_g["var_rater"])) / k
        )
        var_ensemble = float(llm_g["var_p"]) + float(llm_g["var_pr"]) / k
        correlation = float(multivariate["cov_HL"]) / np.sqrt(
            float(h["var_p"]) * var_ensemble
        ) if var_ensemble > 0 and float(h["var_p"]) > 0 else np.nan
        rows.append({
            "n_models": k,
            "Phi_L": phi,
            "ensemble_human_correlation": correlation,
            "asymptotic_rho_HL": multivariate["rho_HL"],
        })
    return pd.DataFrame(rows)
