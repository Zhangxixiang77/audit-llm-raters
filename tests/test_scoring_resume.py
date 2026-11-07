from pathlib import Path

import pandas as pd
import pytest

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


def test_mixed_cost_currencies_are_itemized_but_never_executed_together(tmp_path):
    essays = _essays().iloc[:1]
    anchors = _essays("a")
    models = [
        {
            "name": "usd_model", "provider": "openai_compatible", "model_version": "usd-v1",
            "currency": "USD", "pricing": {"input_per_million": 1, "output_per_million": 5},
        },
        {
            "name": "cny_model", "provider": "openai_compatible", "model_version": "cny-v1",
            "currency": "CNY", "pricing": {"input_per_million": 2, "output_per_million": 8},
        },
    ]
    tasks = build_tasks(
        essays, models, PromptLibrary(ROOT / "prompts/v1"), anchors,
        ["zero_shot"], ["brief"], 1,
    )
    plan = run_scoring(tasks, tmp_path / "plan.sqlite", execute=False)
    assert plan["estimated_cost_no_cache_or_batch_discount"] is None
    assert set(plan["estimated_cost_by_currency"]) == {"USD", "CNY"}
    with pytest.raises(ValueError, match="mixes cost currencies"):
        run_scoring(tasks, tmp_path / "scores.sqlite", execute=True, max_cost=1)
