from pathlib import Path

import pandas as pd

from audit_llm_raters.scoring.prompts import PromptLibrary
from audit_llm_raters.scoring.run import build_tasks, run_scoring
from audit_llm_raters.store.sqlite import ScoreStore

ROOT = Path(__file__).parents[1]


def _essays(prefix="e"):
    return pd.DataFrame([
        {"essay_id": f"{prefix}{i}", "dataset": "test", "prompt_id": "p", "essay_text": f"Essay {i} with evidence.",
         "human_r1": i + 1, "human_r2": i + 1, "human_rater_ids": "[]", "scale_min": 1, "scale_max": 6,
         "ell": i % 2, "demographics_json": "{}"}
        for i in range(4)
    ])


def test_mock_scoring_is_resumable(tmp_path):
    essays = _essays()
    anchors = _essays("a")
    models = [{"name": "mock", "provider": "mock", "model_version": "v1"}]
    tasks = build_tasks(essays, models, PromptLibrary(ROOT / "prompts/v1"), anchors, ["zero_shot", "anchored"], ["brief"], 2)
    assert len(tasks) == 16
    database = tmp_path / "scores.sqlite"
    first = run_scoring(tasks, database, workers=2, execute=True)
    second = run_scoring(tasks, database, workers=2, execute=True)
    assert first["completed"] == 16
    assert second["completed"] == 0
    with ScoreStore(database) as store:
        frame = store.to_frame()
    assert len(frame) == 16
    assert frame["raw_response"].str.len().gt(0).all()
    assert frame["request_id"].nunique() == 16
