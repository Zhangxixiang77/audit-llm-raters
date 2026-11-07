"""Synthetic shared-error data for estimator validation, not empirical evidence."""

from __future__ import annotations

import numpy as np


def simulate_shared_error(
    n_essays: int = 400,
    n_human: int = 2,
    n_llm: int = 6,
    c: float = 0.25,
    u_sd: float = 0.5,
    midpoint: float = 3.0,
    human_noise_sd: float = 0.45,
    llm_noise_sd: float = 0.35,
    seed: int = 20261107,
) -> dict[str, np.ndarray | float]:
    if n_human < 2 or n_llm < 2:
        raise ValueError("At least two human and two LLM raters are required")
    rng = np.random.default_rng(seed)
    nu = rng.normal(0, 1, n_essays)
    u = rng.normal(0, u_sd, n_essays)
    human_severity = rng.normal(0, 0.12, n_human)
    llm_severity = rng.normal(0, 0.12, n_llm)
    human = (
        nu[:, None]
        + human_severity[None, :]
        + rng.normal(0, human_noise_sd, (n_essays, n_human))
    )
    shared = u - c * (nu - midpoint)
    llm = (
        nu[:, None]
        + shared[:, None]
        + llm_severity[None, :]
        + rng.normal(0, llm_noise_sd, (n_essays, n_llm))
    )
    true_rho = (1 - c) / np.sqrt((1 - c) ** 2 + u_sd**2)
    var_h = 1.0
    var_l = (1 - c) ** 2 + u_sd**2
    covariance = 1 - c
    true_var_pt = (var_h + var_l) / 2 - covariance
    true_pi = true_var_pt / (
        covariance + true_var_pt + (human_noise_sd**2 / n_human + llm_noise_sd**2 / n_llm) / 2
    )
    return {
        "human": human,
        "llm": llm,
        "nu": nu,
        "u": u,
        "c": c,
        "u_sd": u_sd,
        "true_rho_HL": true_rho,
        "true_pi_pt": true_pi,
    }


def simulate_crossed_facets(
    shape: tuple[int, int, int, int] = (40, 6, 2, 2),
    variances: dict[str, float] | None = None,
    seed: int = 20261107,
) -> np.ndarray:
    """Generate p × model × prompt × run data with known main/interaction components."""
    if len(shape) != 4:
        raise ValueError("shape must contain p, model, prompt, run")
    variances = variances or {
        "p": 1.0, "m": 0.04, "v": 0.01, "e": 0.0,
        "p:m": 0.16, "p:v": 0.09, "p:m:v": 0.04, "p:m:v:e": 0.25,
    }
    rng = np.random.default_rng(seed)
    names = ["p", "m", "v", "e"]
    from itertools import combinations
    result = np.zeros(shape)
    for size in range(1, 5):
        for effect in combinations(range(4), size):
            key = ":".join(names[index] for index in effect)
            effect_shape = tuple(shape[index] for index in effect)
            effects = rng.normal(0, np.sqrt(variances.get(key, 0.0)), effect_shape)
            broadcast_shape = [1] * 4
            for position, axis in enumerate(effect):
                broadcast_shape[axis] = effect_shape[position]
            result += effects.reshape(broadcast_shape)
    return result
