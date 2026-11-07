"""Optional Bayesian p × (r:t) multivariate G model using NumPyro."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def fit_bayesian_multivariate_g(
    human: np.ndarray,
    llm: np.ndarray,
    num_warmup: int = 1000,
    num_samples: int = 2000,
    num_chains: int = 4,
    seed: int = 20261107,
) -> Any:
    human = np.asarray(human, dtype=float)
    llm = np.asarray(llm, dtype=float)
    if human.ndim != 2 or llm.ndim != 2 or human.shape[0] != llm.shape[0]:
        raise ValueError("human and llm must be p × r matrices with the same p")
    if not np.isfinite(human).all() or not np.isfinite(llm).all():
        raise ValueError("Bayesian balanced G model requires complete matrices")
    try:
        import jax.numpy as jnp
        import numpyro
        import numpyro.distributions as dist
        from jax import random
        from numpyro.infer import MCMC, NUTS
    except ImportError as exc:
        raise ImportError("Install the 'bayes' extra: pip install -e '.[bayes]'") from exc

    n_p, n_h = human.shape
    n_l = llm.shape[1]

    def model():
        location = numpyro.sample("location", dist.Normal(0, 5).expand([2]))
        scales = numpyro.sample("universe_scales", dist.HalfNormal(5).expand([2]))
        cholesky = numpyro.sample("universe_cholesky", dist.LKJCholesky(2, concentration=2))
        covariance = jnp.diag(scales) @ cholesky @ cholesky.T @ jnp.diag(scales)
        universe = numpyro.sample(
            "universe", dist.MultivariateNormal(location, covariance_matrix=covariance).expand([n_p])
        )
        h_raw = numpyro.sample("human_severity_raw", dist.Normal(0, 0.5).expand([n_h - 1]))
        l_raw = numpyro.sample("llm_severity_raw", dist.Normal(0, 0.5).expand([n_l - 1]))
        h_severity = jnp.concatenate([h_raw, -h_raw.sum(keepdims=True)])
        l_severity = jnp.concatenate([l_raw, -l_raw.sum(keepdims=True)])
        sigma_h = numpyro.sample("sigma_H", dist.HalfNormal(2))
        sigma_l = numpyro.sample("sigma_L", dist.HalfNormal(2))
        numpyro.sample(
            "human_obs",
            dist.Normal(universe[:, 0, None] + h_severity[None, :], sigma_h).to_event(2),
            obs=jnp.asarray(human),
        )
        numpyro.sample(
            "llm_obs",
            dist.Normal(universe[:, 1, None] + l_severity[None, :], sigma_l).to_event(2),
            obs=jnp.asarray(llm),
        )
        var_h = covariance[0, 0]
        var_l = covariance[1, 1]
        cov_hl = covariance[0, 1]
        var_pt = (var_h + var_l) / 2 - cov_hl
        residual = (sigma_h**2 / n_h + sigma_l**2 / n_l) / 2
        numpyro.deterministic("rho_HL", cov_hl / jnp.sqrt(var_h * var_l))
        numpyro.deterministic("pi_pt", var_pt / (cov_hl + var_pt + residual))
        numpyro.deterministic("var_pt", var_pt)
        numpyro.deterministic("cov_HL", cov_hl)

    mcmc = MCMC(NUTS(model), num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains)
    mcmc.run(random.PRNGKey(seed))
    return mcmc


def posterior_summary(mcmc: Any) -> pd.DataFrame:
    samples = mcmc.get_samples(group_by_chain=False)
    rows = []
    for name in ["rho_HL", "pi_pt", "var_pt", "cov_HL", "sigma_H", "sigma_L"]:
        values = np.asarray(samples[name])
        rows.append({
            "estimand": name, "posterior_mean": values.mean(), "posterior_sd": values.std(),
            "ci_lower": np.quantile(values, 0.025), "ci_upper": np.quantile(values, 0.975),
            "n_draws": len(values),
        })
    return pd.DataFrame(rows)
