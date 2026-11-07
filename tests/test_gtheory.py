import numpy as np
import pytest

from audit_llm_raters.analysis.gtheory import balanced_ems, multivariate_g, two_facet_g
from audit_llm_raters.sim.generate import simulate_crossed_facets, simulate_shared_error


def test_balanced_two_facet_components_and_phi():
    rng = np.random.default_rng(4)
    p = rng.normal(size=(100, 1))
    r = rng.normal(scale=0.2, size=(1, 4))
    error = rng.normal(scale=0.4, size=(100, 4))
    result = two_facet_g(p + r + error)
    assert result["var_p"] > 0.7
    assert result["var_pr"] > 0
    assert 0 < result["Phi"] <= 1.05


def test_full_crossed_ss_decomposes_total():
    rng = np.random.default_rng(8)
    values = rng.normal(size=(6, 3, 2, 2))
    components = balanced_ems(values, ["p", "m", "v", "e"])
    centered = values - values.mean()
    assert np.isclose(components["ss"].sum(), np.sum(centered**2))
    assert np.isfinite(components["variance"]).all()


def test_multivariate_recovers_shared_error_estimands():
    simulated = simulate_shared_error(n_essays=1500, seed=12)
    result = multivariate_g(simulated["human"], simulated["llm"])
    assert abs(result["rho_HL"] - simulated["true_rho_HL"]) < 0.04
    assert abs(result["pi_pt"] - simulated["true_pi_pt"]) < 0.04


def test_full_factorial_simulation_recovers_major_components():
    values = simulate_crossed_facets(shape=(300, 6, 2, 2), seed=44)
    components = balanced_ems(values, ["p", "m", "v", "e"]).set_index("effect")["variance"]
    assert abs(components["p"] - 1.0) < 0.15
    assert abs(components["p:m"] - 0.16) < 0.06
    assert abs(components["p:v"] - 0.09) < 0.06
    assert abs(components["p:m:v:e"] - 0.25) < 0.05


def test_bayesian_multivariate_g_smoke():
    pytest.importorskip("numpyro")
    from audit_llm_raters.analysis.bayes_g import fit_bayesian_multivariate_g, posterior_summary

    rng = np.random.default_rng(20261107)
    universe = rng.normal(size=(6, 2))
    human = universe[:, 0, None] + rng.normal(scale=0.3, size=(6, 2))
    llm = universe[:, 1, None] + rng.normal(scale=0.3, size=(6, 3))
    mcmc = fit_bayesian_multivariate_g(human, llm, num_warmup=5, num_samples=5, num_chains=1)
    summary = posterior_summary(mcmc)
    assert np.isfinite(summary[["posterior_mean", "ci_lower", "ci_upper"]].to_numpy()).all()
    assert summary["n_draws"].eq(5).all()
