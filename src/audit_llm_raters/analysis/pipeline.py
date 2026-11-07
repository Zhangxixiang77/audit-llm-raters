"""Ordered analysis pipeline: QC → G/D → MFRM → compression → DRF → robustness."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from audit_llm_raters.adapters.tabular import dataset_gate_audit
from audit_llm_raters.analysis.compression import compression_audit, target_stability
from audit_llm_raters.analysis.descriptive import bootstrap_observed_metrics, observed_metrics
from audit_llm_raters.analysis.drf import drf_bootstrap, ell_group_qwk
from audit_llm_raters.analysis.gtheory import (
    balanced_ems,
    bootstrap_multivariate_g,
    d_study,
    multivariate_g,
)
from audit_llm_raters.analysis.mfrm import MFRMData, fit_mfrm_map, scores_to_mfrm_long
from audit_llm_raters.analysis.qc import failure_table, human_agreement, pilot_gates
from audit_llm_raters.analysis.robust import analysis_matrices, failure_sensitivity


def _json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, pd.DataFrame):
        return value.to_dict("records")
    return value


def _save(frame: pd.DataFrame, directory: Path, name: str) -> None:
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
    frame.to_csv(directory / f"{safe_name}.csv", index=False)


def factorial_ems(scores: pd.DataFrame) -> pd.DataFrame:
    facets = ["essay_id", "model", "prompt_variant", "rubric_variant", "run_id"]
    valid = scores[scores["parse_status"] == "ok"].copy()
    levels = [sorted(valid[facet].unique()) for facet in facets]
    expected = int(np.prod([len(level) for level in levels]))
    cell = valid.groupby(facets)["parsed_score"].mean()
    if len(cell) != expected:
        raise ValueError(f"Unbalanced design: observed {len(cell)} of {expected} cells")
    full_index = pd.MultiIndex.from_product(levels, names=facets)
    values = cell.reindex(full_index).to_numpy().reshape([len(level) for level in levels])
    if not np.isfinite(values).all():
        raise ValueError("Balanced EMS received missing cells")
    return balanced_ems(values, ["p", "m", "v", "q", "e"])


def configuration_ems(scores: pd.DataFrame) -> pd.DataFrame:
    """Fixed-model p × prompt × rubric × repeat EMS; model variance is excluded."""
    if scores["model"].nunique() != 1:
        raise ValueError("Configuration EMS is only for a single fixed model")
    facets = ["essay_id", "prompt_variant", "rubric_variant", "run_id"]
    valid = scores[scores["parse_status"] == "ok"].copy()
    levels = [sorted(valid[facet].unique()) for facet in facets]
    expected = int(np.prod([len(level) for level in levels]))
    cell = valid.groupby(facets)["parsed_score"].mean()
    if len(cell) != expected:
        raise ValueError(f"Unbalanced design: observed {len(cell)} of {expected} cells")
    full_index = pd.MultiIndex.from_product(levels, names=facets)
    values = cell.reindex(full_index).to_numpy().reshape([len(level) for level in levels])
    if not np.isfinite(values).all():
        raise ValueError("Balanced EMS received missing cells")
    return balanced_ems(values, ["p", "v", "q", "e"])


def _plot_d_study(frame: pd.DataFrame, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(7, 4.5))
    axis.plot(frame["n_models"], frame["Phi_L"], marker="o", label="LLM-only Φ")
    axis.plot(frame["n_models"], frame["ensemble_human_correlation"], marker="s", label="Ensemble–human correlation")
    axis.axhline(0.9, color="grey", linestyle="--", linewidth=1, label="Φ = .90")
    axis.set(xlabel="Number of LLM models", ylabel="Coefficient", ylim=(0, 1.05))
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def run_analysis(
    scores: pd.DataFrame,
    output_dir: str | Path,
    bootstrap: int = 500,
    drf_boot: int = 0,
    seed: int = 20261107,
    bayes_g: bool = False,
) -> dict[str, Any]:
    output = Path(output_dir)
    tables = output / "tables"
    figures = output / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {}

    failures = failure_table(scores)
    _save(failures, tables, "01_parse_failures")
    _save(scores.groupby(["model", "parse_status"]).size().rename("n").reset_index(), tables, "01_status_counts")
    valid = scores[scores["parse_status"] == "ok"]
    _save(valid.groupby(["model", "parsed_score"]).size().rename("n").reset_index(), tables, "01_score_distribution")
    essays = scores.drop_duplicates("essay_id")
    agreement = human_agreement(essays["human_r1"], essays["human_r2"])
    summary["human_agreement"] = agreement
    gates = pilot_gates(scores)
    summary["pilot_gates"] = {key: value for key, value in gates.items() if not isinstance(value, pd.DataFrame)}
    _save(gates["config_failure_rates"], tables, "01_G3_config_failures")
    _save(gates["residual_correlation_matrix"].reset_index(), tables, "01_G2_residual_correlations")

    n_models = scores["model"].nunique()
    if n_models >= 2:
        human, llm, metadata = analysis_matrices(scores, "missing")
        multivariate = multivariate_g(human, llm)
        summary["multivariate_g"] = {key: value for key, value in multivariate.items() if not isinstance(value, pd.DataFrame)}
        _save(multivariate["human_components"], tables, "02_G_human_components")
        _save(multivariate["llm_components"], tables, "02_G_llm_components")
        if bootstrap:
            _save(bootstrap_multivariate_g(human, llm, bootstrap, seed), tables, "02_G_bootstrap_ci")
        if bayes_g:
            from audit_llm_raters.analysis.bayes_g import (
                fit_bayesian_multivariate_g,
                posterior_summary,
            )
            bayesian = fit_bayesian_multivariate_g(human, llm, seed=seed)
            posterior = posterior_summary(bayesian)
            _save(posterior, tables, "02_G_bayesian_posterior")
            comparison = posterior[["estimand", "posterior_mean"]].copy()
            comparison["EMS_estimate"] = comparison["estimand"].map(multivariate)
            _save(comparison, tables, "06_EMS_vs_bayesian")
        curves = d_study(human, llm, range(1, max(10, llm.shape[1] * 2) + 1))
        _save(curves, tables, "02_D_study")
        _plot_d_study(curves, figures / "02_D_study.png")
        try:
            _save(factorial_ems(scores), tables, "02_G_full_factorial_EMS")
        except ValueError as exc:
            summary["full_factorial_EMS_note"] = str(exc)
    else:
        reason = "A single fixed model cannot estimate a model facet, cross-model shared error, or a multi-model ensemble D-study."
        summary["multivariate_g"] = {"status": "not_estimable", "n_models": int(n_models), "reason": reason}
        summary["d_study"] = {"status": "not_estimable", "reason": reason}
        _save(pd.DataFrame({
            "estimand": ["rho_HL", "pi_pt", "Phi_L", "multi_model_D_study", "G2_cross_model_residual_correlation"],
            "status": ["not_estimable"] * 5,
            "reason": [reason] * 5,
        }), tables, "02_G_D_not_estimable")
        overall, configurations = observed_metrics(scores)
        _save(overall, tables, "02_single_model_observed_metrics")
        _save(configurations, tables, "02_single_model_configuration_metrics")
        if bootstrap:
            _save(bootstrap_observed_metrics(scores, bootstrap, seed), tables, "02_single_model_observed_bootstrap_ci")
        _save(configuration_ems(scores), tables, "02_G_single_model_configuration_EMS")

    human_gate = dataset_gate_audit(essays)
    summary["human_design"] = human_gate
    long_frame = scores_to_mfrm_long(
        scores, include_human=True, human_crossed=human_gate["human_design"] == "crossed"
    )
    mfrm_summaries = []
    for prompt_id, group in long_frame.groupby("item_id"):
        scale_min = int(group["scale_min"].iloc[0])
        scale_max = int(group["scale_max"].iloc[0])
        if group["scale_min"].nunique() != 1 or group["scale_max"].nunique() != 1:
            raise ValueError("Fit MFRM separately for each scale; inconsistent scale within prompt")
        data = MFRMData.from_frame(group, scale_min=scale_min, scale_max=scale_max)
        fitted = fit_mfrm_map(data)
        fit_stats = fitted.fit_statistics()
        fit_stats.insert(0, "prompt_id", prompt_id)
        mfrm_summaries.append({"prompt_id": prompt_id, "converged": fitted.success, "message": fitted.message, "objective": fitted.objective})
        _save(fit_stats, tables, f"03_MFRM_raters_{prompt_id}")
        _save(fitted.category_usage(), tables, f"03_MFRM_categories_{prompt_id}")
    summary["MFRM"] = mfrm_summaries

    compression = compression_audit(scores)
    _save(compression, tables, "04_compression")
    if not compression.empty:
        _save(target_stability(compression), tables, "04_target_stability")

    ell = scores.get("ell", pd.Series(dtype=float))
    if ell.notna().any():
        _save(ell_group_qwk(scores), tables, "05_QWK_by_ELL")
    if ell.notna().sum() == 0 or ell.nunique() < 2:
        summary["DRF"] = {
            "status": "not_estimable",
            "reason": "ELL versus non-ELL DRF requires both groups with observed indicators; this sample does not contain both groups.",
        }
    elif drf_boot:
        drf_outputs = []
        for prompt_id, group in long_frame.groupby("item_id"):
            if group["ell"].nunique() < 2:
                continue
            drf = drf_bootstrap(group, int(group["scale_min"].iloc[0]), int(group["scale_max"].iloc[0]), drf_boot, seed)
            drf.insert(0, "prompt_id", prompt_id)
            drf_outputs.append(drf)
        if drf_outputs:
            _save(pd.concat(drf_outputs, ignore_index=True), tables, "05_DRF")

    if n_models >= 2:
        _save(failure_sensitivity(scores), tables, "06_failure_sensitivity")
    else:
        sensitivity = []
        for policy in ["missing", "floor"]:
            work = scores.copy()
            if policy == "floor":
                work.loc[work["parse_status"] != "ok", "parsed_score"] = work["scale_min"]
                work.loc[work["parse_status"] != "ok", "parse_status"] = "ok"
            overall, _ = observed_metrics(work)
            overall.insert(1, "failure_policy", policy)
            overall["multivariate_g_status"] = "not_estimable_single_model"
            sensitivity.append(overall)
        _save(pd.concat(sensitivity, ignore_index=True), tables, "06_failure_sensitivity")
    (output / "summary.json").write_text(json.dumps(_json_value(summary), indent=2, ensure_ascii=False), encoding="utf-8")
    return summary
