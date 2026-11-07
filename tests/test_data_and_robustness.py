import pandas as pd
import pytest

from audit_llm_raters.adapters.tabular import (
    dataset_gate_audit,
    normalize_essays,
    stratified_sample,
)
from audit_llm_raters.analysis.pipeline import run_analysis
from audit_llm_raters.analysis.robust import analysis_matrices


def test_dataset_normalization_sampling_and_gate(tmp_path):
    source = pd.DataFrame({
        "id": [str(i) for i in range(20)], "set": [1] * 10 + [2] * 10,
        "text": [f"Essay {i}" for i in range(20)], "r1": [i % 4 for i in range(20)],
        "r2": [i % 4 for i in range(20)],
    })
    path = tmp_path / "source.csv"
    source.to_csv(path, index=False)
    config = {
        "name": "toy", "columns": {"essay_id": "id", "prompt_id": "set", "essay_text": "text", "human_r1": "r1", "human_r2": "r2"},
        "scale": {"min": 0, "max": 3}, "scales_by_prompt": {"2": {"min": 1, "max": 6}},
    }
    essays = normalize_essays(path, config)
    assert essays.loc[essays["prompt_id"] == "2", "scale_max"].eq(6).all()
    assert len(stratified_sample(essays, 10)) == 10
    gate = dataset_gate_audit(essays)
    assert gate["G1_pass"]
    assert gate["human_design"] == "nested_within_essay"

    filtered = normalize_essays(path, {**config, "prompt_ids": ["2"]})
    assert set(filtered["prompt_id"]) == {"2"}
    assert len(filtered) == 10
    with pytest.raises(ValueError, match="prompt_ids"):
        normalize_essays(path, {**config, "prompt_ids": ["9"]})


def test_failure_floor_policy():
    rows = []
    for essay in range(6):
        for model in ["a", "b"]:
            rows.append({"essay_id": str(essay), "model": model, "parsed_score": None if essay == 0 and model == "b" else 3,
                         "parse_status": "invalid_json" if essay == 0 and model == "b" else "ok", "scale_min": 1,
                         "human_r1": 3, "human_r2": 4})
    frame = pd.DataFrame(rows)
    _, _, missing = analysis_matrices(frame, "missing")
    human, llm, floor = analysis_matrices(frame, "floor")
    assert missing["n_essays"].iloc[0] == 5
    assert floor["n_essays"].iloc[0] == 6
    assert llm[0, 1] == 1


def test_single_model_analysis_is_explicitly_limited(tmp_path):
    rows = []
    for essay in range(12):
        for prompt_variant in ["zero_shot", "anchored"]:
            for rubric_variant in ["brief", "detailed"]:
                for run_id in [1, 2]:
                    rows.append({
                        "essay_id": str(essay), "dataset": "toy", "prompt_id": "overall",
                        "model": "only_model", "prompt_variant": prompt_variant,
                        "rubric_variant": rubric_variant, "run_id": run_id,
                        "parsed_score": 1 + (essay + run_id) % 4, "parse_status": "ok",
                        "scale_min": 1, "scale_max": 5, "human_r1": 1 + essay % 4,
                        "human_r2": 1 + (essay + 1) % 4, "human_rater_ids": "[]", "ell": 1,
                    })
    summary = run_analysis(pd.DataFrame(rows), tmp_path, bootstrap=20)
    assert summary["multivariate_g"]["status"] == "not_estimable"
    assert summary["d_study"]["status"] == "not_estimable"
    assert summary["pilot_gates"]["G2_pass"] is None
    assert summary["DRF"]["status"] == "not_estimable"
    assert (tmp_path / "tables" / "02_single_model_observed_metrics.csv").exists()
    assert (tmp_path / "tables" / "02_G_single_model_configuration_EMS.csv").exists()
    assert not (tmp_path / "tables" / "02_G_bootstrap_ci.csv").exists()
