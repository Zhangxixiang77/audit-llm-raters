"""Adjacent-category many-facet Rasch model with MAP and optional NumPyro NUTS."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp


@dataclass
class MFRMData:
    essay: np.ndarray
    rater: np.ndarray
    item: np.ndarray
    score: np.ndarray
    ell: np.ndarray
    essay_levels: list[str]
    rater_levels: list[str]
    item_levels: list[str]
    n_categories: int

    @classmethod
    def from_frame(
        cls,
        frame: pd.DataFrame,
        score_col: str = "score",
        scale_min: int = 0,
        scale_max: int | None = None,
    ) -> "MFRMData":
        required = {"essay_id", "rater_id", "item_id", score_col}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"Missing MFRM columns: {sorted(missing)}")
        work = frame.dropna(subset=[score_col]).copy()
        scores = work[score_col].to_numpy(dtype=float)
        if not np.equal(scores, np.floor(scores)).all():
            raise ValueError("MFRM scores must be integers")
        shifted = scores.astype(int) - scale_min
        scale_max = int(shifted.max()) if scale_max is None else scale_max - scale_min
        if (shifted < 0).any() or (shifted > scale_max).any():
            raise ValueError("Observed score outside the declared scale")
        essay_levels = sorted(work["essay_id"].astype(str).unique())
        rater_levels = sorted(work["rater_id"].astype(str).unique())
        item_levels = sorted(work["item_id"].astype(str).unique())
        def encode(column: str, levels: list[str]) -> np.ndarray:
            mapping = {level: index for index, level in enumerate(levels)}
            return work[column].astype(str).map(mapping).to_numpy(dtype=int)
        ell = work.get("ell", pd.Series(0, index=work.index)).fillna(0).to_numpy(dtype=int)
        return cls(
            encode("essay_id", essay_levels), encode("rater_id", rater_levels),
            encode("item_id", item_levels), shifted, ell,
            essay_levels, rater_levels, item_levels, scale_max + 1,
        )


@dataclass
class MFRMResult:
    x: np.ndarray
    theta: np.ndarray
    severity: np.ndarray
    difficulty: np.ndarray
    thresholds: np.ndarray
    drf: np.ndarray | None
    data: MFRMData
    success: bool
    message: str
    objective: float
    expected: np.ndarray
    variance: np.ndarray

    def fit_statistics(self) -> pd.DataFrame:
        standardized_sq = (self.data.score - self.expected) ** 2 / self.variance
        rows = []
        for index, rater in enumerate(self.data.rater_levels):
            mask = self.data.rater == index
            rows.append({
                "rater_id": rater,
                "severity": self.severity[index],
                "drf_ell": None if self.drf is None else self.drf[index],
                "n": int(mask.sum()),
                "infit": float(np.sum((self.data.score[mask] - self.expected[mask]) ** 2) / np.sum(self.variance[mask])),
                "outfit": float(np.mean(standardized_sq[mask])),
            })
        return pd.DataFrame(rows)

    def category_usage(self) -> pd.DataFrame:
        frame = pd.DataFrame({"rater_id": np.asarray(self.data.rater_levels)[self.data.rater], "score": self.data.score})
        usage = frame.groupby(["rater_id", "score"]).size().rename("n").reset_index()
        usage["proportion"] = usage["n"] / usage.groupby("rater_id")["n"].transform("sum")
        return usage


def _centered(raw: np.ndarray, n: int) -> np.ndarray:
    result = np.empty(n)
    result[:-1] = raw
    result[-1] = -np.sum(raw)
    return result


def _centered_gradient(gradient: np.ndarray) -> np.ndarray:
    return gradient[:-1] - gradient[-1]


class _MAPFitter:
    def __init__(self, data: MFRMData, drf: bool, priors: dict[str, float]):
        self.data = data
        self.drf = drf
        self.priors = priors
        self.n_p = len(data.essay_levels)
        self.n_r = len(data.rater_levels)
        self.n_i = len(data.item_levels)
        self.k = data.n_categories - 1
        self.slices = {}
        start = 0
        for name, size in [
            ("theta", self.n_p), ("severity", self.n_r - 1),
            ("difficulty", self.n_i - 1), ("thresholds", self.k - 1),
            ("drf", self.n_r if drf else 0),
        ]:
            self.slices[name] = slice(start, start + size)
            start += size
        self.n_params = start

    def unpack(self, x: np.ndarray) -> tuple[np.ndarray, ...]:
        return (
            x[self.slices["theta"]],
            _centered(x[self.slices["severity"]], self.n_r),
            _centered(x[self.slices["difficulty"]], self.n_i),
            _centered(x[self.slices["thresholds"]], self.k),
            x[self.slices["drf"]] if self.drf else None,
        )

    def probabilities(self, eta: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
        k = self.k
        utilities = np.zeros((len(eta), k + 1))
        cumulative = np.zeros(len(eta))
        for category in range(1, k + 1):
            cumulative = cumulative + eta - thresholds[category - 1]
            utilities[:, category] = cumulative
        return np.exp(utilities - logsumexp(utilities, axis=1, keepdims=True))

    def evaluate(self, x: np.ndarray, gradient: bool = True):
        data = self.data
        theta, severity, difficulty, thresholds, drf = self.unpack(x)
        eta = theta[data.essay] - severity[data.rater] - difficulty[data.item]
        if drf is not None:
            eta = eta - drf[data.rater] * data.ell
        probabilities = self.probabilities(eta, thresholds)
        categories = np.arange(self.k + 1)
        expected_category = probabilities @ categories
        nll = -np.sum(np.log(np.maximum(probabilities[np.arange(len(eta)), data.score], 1e-300)))
        priors = {"theta": 2.0, "severity": 1.0, "difficulty": 1.0, "thresholds": 1.0, "drf": 1.0, **self.priors}
        nll += 0.5 * np.sum(theta**2) / priors["theta"]**2
        nll += 0.5 * np.sum(severity**2) / priors["severity"]**2
        nll += 0.5 * np.sum(difficulty**2) / priors["difficulty"]**2
        nll += 0.5 * np.sum(thresholds**2) / priors["thresholds"]**2
        if drf is not None:
            nll += 0.5 * np.sum(drf**2) / priors["drf"]**2
        if not gradient:
            return float(nll)
        d_eta = expected_category - data.score
        grad_theta = np.zeros(self.n_p)
        grad_severity = np.zeros(self.n_r)
        grad_difficulty = np.zeros(self.n_i)
        np.add.at(grad_theta, data.essay, d_eta)
        np.add.at(grad_severity, data.rater, -d_eta)
        np.add.at(grad_difficulty, data.item, -d_eta)
        grad_theta += theta / priors["theta"]**2
        grad_severity += severity / priors["severity"]**2
        grad_difficulty += difficulty / priors["difficulty"]**2
        grad = np.zeros(self.n_params)
        grad[self.slices["theta"]] = grad_theta
        grad[self.slices["severity"]] = _centered_gradient(grad_severity)
        grad[self.slices["difficulty"]] = _centered_gradient(grad_difficulty)
        grad_thresholds = np.zeros(self.k)
        for threshold in range(1, self.k + 1):
            observed = (data.score >= threshold).astype(float)
            tail_probability = probabilities[:, threshold:].sum(axis=1)
            grad_thresholds[threshold - 1] = np.sum(observed - tail_probability)
        grad_thresholds += thresholds / priors["thresholds"]**2
        grad[self.slices["thresholds"]] = _centered_gradient(grad_thresholds)
        if drf is not None:
            grad_drf = np.zeros(self.n_r)
            np.add.at(grad_drf, data.rater, -d_eta * data.ell)
            grad_drf += drf / priors["drf"]**2
            grad[self.slices["drf"]] = grad_drf
        return float(nll), grad


def fit_mfrm_map(
    data: MFRMData,
    drf: bool = False,
    priors: dict[str, float] | None = None,
    initial: np.ndarray | None = None,
    maxiter: int = 1000,
) -> MFRMResult:
    fitter = _MAPFitter(data, drf, priors or {})
    x0 = np.zeros(fitter.n_params) if initial is None else initial
    if len(x0) != fitter.n_params:
        raise ValueError("Initial parameter vector has the wrong length")
    result = minimize(
        fitter.evaluate, x0, jac=True, method="L-BFGS-B",
        options={"maxiter": maxiter, "ftol": 1e-10, "gtol": 1e-6, "maxls": 40},
    )
    theta, severity, difficulty, thresholds, drf_values = fitter.unpack(result.x)
    eta = theta[data.essay] - severity[data.rater] - difficulty[data.item]
    if drf_values is not None:
        eta = eta - drf_values[data.rater] * data.ell
    probabilities = fitter.probabilities(eta, thresholds)
    categories = np.arange(data.n_categories)
    expected = probabilities @ categories
    variance = probabilities @ (categories**2) - expected**2
    return MFRMResult(
        result.x, theta, severity, difficulty, thresholds, drf_values, data,
        bool(result.success), str(result.message), float(result.fun), expected, np.maximum(variance, 1e-8),
    )


def simulate_mfrm(
    n_essays: int = 120,
    rater_severity: np.ndarray | None = None,
    item_difficulty: np.ndarray | None = None,
    thresholds: np.ndarray | None = None,
    seed: int = 20261107,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    severity = np.asarray(rater_severity if rater_severity is not None else [-0.8, -0.3, 0.2, 0.9])
    difficulty = np.asarray(item_difficulty if item_difficulty is not None else [0.0])
    thresholds = np.asarray(thresholds if thresholds is not None else [-0.7, 0.0, 0.7])
    severity = severity - severity.mean()
    difficulty = difficulty - difficulty.mean()
    thresholds = thresholds - thresholds.mean()
    theta = rng.normal(0, 1, n_essays)
    rows = []
    for p in range(n_essays):
        for r in range(len(severity)):
            for i in range(len(difficulty)):
                eta = theta[p] - severity[r] - difficulty[i]
                utilities = np.concatenate([[0], np.cumsum(eta - thresholds)])
                probabilities = np.exp(utilities - logsumexp(utilities))
                score = rng.choice(len(probabilities), p=probabilities)
                rows.append({"essay_id": f"p{p}", "rater_id": f"r{r}", "item_id": f"i{i}", "score": score, "ell": 0})
    return pd.DataFrame(rows), {"theta": theta, "severity": severity, "difficulty": difficulty, "thresholds": thresholds}


def scores_to_mfrm_long(scores: pd.DataFrame, include_human: bool = True, human_crossed: bool = False) -> pd.DataFrame:
    """Convert canonical score records; missing human IDs become essay-nested raters."""
    valid = scores[scores["parse_status"] == "ok"].copy()
    rows = []
    for row in valid.itertuples():
        rows.append({
            "essay_id": str(row.essay_id), "rater_id": f"LLM:{row.model}:{row.prompt_variant}:{row.rubric_variant}",
            "item_id": str(row.prompt_id), "score": int(row.parsed_score), "scale_min": int(row.scale_min),
            "scale_max": int(row.scale_max), "ell": getattr(row, "ell", 0), "rater_type": "L",
        })
    if include_human:
        essays = valid.drop_duplicates("essay_id")
        for row in essays.itertuples():
            try:
                ids = json.loads(row.human_rater_ids)
            except (TypeError, json.JSONDecodeError):
                ids = []
            for position, score in enumerate([row.human_r1, row.human_r2], start=1):
                if pd.isna(score):
                    continue
                supplied = ids[position - 1] if len(ids) >= position else ""
                if human_crossed and supplied:
                    rater_id = f"H:{supplied}"
                else:
                    rater_id = f"H:{row.essay_id}:{position}"
                rows.append({
                    "essay_id": str(row.essay_id), "rater_id": rater_id, "item_id": str(row.prompt_id),
                    "score": int(score), "scale_min": int(row.scale_min), "scale_max": int(row.scale_max),
                    "ell": getattr(row, "ell", 0), "rater_type": "H",
                })
    return pd.DataFrame(rows)


def fit_mfrm_numpyro(
    data: MFRMData,
    drf: bool = False,
    num_warmup: int = 500,
    num_samples: int = 1000,
    num_chains: int = 2,
    seed: int = 20261107,
) -> Any:
    """Optional full posterior. Returns ArviZ-compatible NumPyro MCMC object."""
    try:
        import jax.numpy as jnp
        from jax import random
        import numpyro
        import numpyro.distributions as dist
        from numpyro.infer import MCMC, NUTS
    except ImportError as exc:
        raise ImportError("Install the 'bayes' extra: pip install -e '.[bayes]'") from exc

    def model():
        theta = numpyro.sample("theta", dist.Normal(0, 2).expand([len(data.essay_levels)]))
        severity_raw = numpyro.sample("severity_raw", dist.Normal(0, 1).expand([len(data.rater_levels) - 1]))
        difficulty_raw = numpyro.sample("difficulty_raw", dist.Normal(0, 1).expand([len(data.item_levels) - 1]))
        threshold_raw = numpyro.sample("threshold_raw", dist.Normal(0, 1).expand([data.n_categories - 2]))
        severity = jnp.concatenate([severity_raw, -severity_raw.sum(keepdims=True)])
        difficulty = jnp.concatenate([difficulty_raw, -difficulty_raw.sum(keepdims=True)])
        thresholds = jnp.concatenate([threshold_raw, -threshold_raw.sum(keepdims=True)])
        eta = theta[jnp.asarray(data.essay)] - severity[jnp.asarray(data.rater)] - difficulty[jnp.asarray(data.item)]
        if drf:
            gamma = numpyro.sample("drf", dist.Normal(0, 1).expand([len(data.rater_levels)]))
            eta = eta - gamma[jnp.asarray(data.rater)] * jnp.asarray(data.ell)
        logits = jnp.concatenate([jnp.zeros((len(eta), 1)), jnp.cumsum(eta[:, None] - thresholds[None, :], axis=1)], axis=1)
        numpyro.sample("obs", dist.Categorical(logits=logits), obs=jnp.asarray(data.score))

    mcmc = MCMC(NUTS(model), num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains)
    mcmc.run(random.PRNGKey(seed))
    return mcmc
