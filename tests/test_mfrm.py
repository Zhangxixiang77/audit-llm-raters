import numpy as np
import pytest

from audit_llm_raters.analysis.mfrm import MFRMData, fit_mfrm_map, simulate_mfrm


def test_mfrm_parameter_recovery_and_fit_statistics():
    frame, truth = simulate_mfrm(n_essays=150, seed=19)
    data = MFRMData.from_frame(frame, scale_min=0, scale_max=3)
    result = fit_mfrm_map(data)
    assert result.success
    assert np.corrcoef(result.severity, truth["severity"])[0, 1] > 0.9
    assert np.max(np.abs(result.severity - truth["severity"])) < 0.25
    assert np.max(np.abs(result.thresholds - truth["thresholds"])) < 0.25
    stats = result.fit_statistics()
    assert set(["infit", "outfit", "severity"]).issubset(stats.columns)
    assert np.isfinite(stats[["infit", "outfit"]]).all().all()


def test_nested_human_raters_are_not_treated_as_crossed():
    frame, _ = simulate_mfrm(n_essays=5, rater_severity=[0])
    data = MFRMData.from_frame(frame, scale_min=0, scale_max=3)
    assert len(data.rater_levels) == 1


def test_bayesian_mfrm_smoke():
    pytest.importorskip("numpyro")
    from audit_llm_raters.analysis.mfrm import fit_mfrm_numpyro

    frame, _ = simulate_mfrm(
        n_essays=5, rater_severity=np.array([-0.4, 0.4]),
        item_difficulty=np.array([0.0]), thresholds=np.array([-0.7, 0.0, 0.7]),
    )
    data = MFRMData.from_frame(frame)
    mcmc = fit_mfrm_numpyro(data, num_warmup=5, num_samples=5, num_chains=1)
    samples = mcmc.get_samples()
    assert samples["theta"].shape == (5, 5)
    assert np.isfinite(np.asarray(samples["severity_raw"])).all()
