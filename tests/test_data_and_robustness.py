import pandas as pd

from audit_llm_raters.adapters.tabular import dataset_gate_audit, normalize_essays, stratified_sample
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
