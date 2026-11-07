"""Simulation gates for universe-score estimators and MFRM parameter recovery."""

from __future__ import annotations

import numpy as np
import pandas as pd

from audit_llm_raters.analysis.gtheory import multivariate_g
from audit_llm_raters.analysis.mfrm import MFRMData, fit_mfrm_map, simulate_mfrm
from audit_llm_raters.sim.generate import simulate_shared_error


def validate_estimators(replications: int = 30, n_essays: int = 400, seed: int = 20261107) -> pd.DataFrame:
    rows = []
    for replication in range(replications):
        simulated = simulate_shared_error(n_essays=n_essays, seed=seed + replication)
        estimated = multivariate_g(simulated["human"], simulated["llm"])
        rows.append({
            "replication": replication,
            "rho_estimated": estimated["rho_HL"], "rho_true": simulated["true_rho_HL"],
            "pi_estimated": estimated["pi_pt"], "pi_true": simulated["true_pi_pt"],
        })
    draws = pd.DataFrame(rows)
    summary = []
    for name in ["rho", "pi"]:
        estimated = draws[f"{name}_estimated"]
        truth = draws[f"{name}_true"].iloc[0]
        bias = estimated.mean() - truth
        summary.append({
            "estimand": name, "true": truth, "mean_estimate": estimated.mean(),
            "bias": bias, "relative_bias_abs": abs(bias) / abs(truth),
            "rmse": np.sqrt(np.mean((estimated - truth) ** 2)),
            "passes_10_percent_gate": abs(bias) / abs(truth) < 0.1,
            "replications": replications, "n_essays": n_essays,
        })
    return pd.DataFrame(summary)


def validate_mfrm(seed: int = 20261107) -> dict[str, float | bool]:
    frame, truth = simulate_mfrm(seed=seed)
    data = MFRMData.from_frame(frame, scale_min=0, scale_max=3)
    fitted = fit_mfrm_map(data)
    return {
        "severity_correlation": float(np.corrcoef(fitted.severity, truth["severity"])[0, 1]),
        "severity_max_abs_error": float(np.max(np.abs(fitted.severity - truth["severity"]))),
        "threshold_max_abs_error": float(np.max(np.abs(fitted.thresholds - truth["thresholds"]))),
        "converged": fitted.success,
        "passes_recovery_gate": bool(
            np.corrcoef(fitted.severity, truth["severity"])[0, 1] > 0.9
            and np.max(np.abs(fitted.severity - truth["severity"])) < 0.25
            and np.max(np.abs(fitted.thresholds - truth["thresholds"])) < 0.25
        ),
    }
