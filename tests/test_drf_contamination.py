import numpy as np
import pandas as pd

from audit_llm_raters.analysis.drf import drf_bootstrap
from audit_llm_raters.analysis.robust import contamination_audit


def test_drf_bootstrap_schema():
    rows = []
    for essay in range(16):
        for rater in ["a", "b"]:
            rows.append({"essay_id": str(essay), "rater_id": rater, "item_id": "p", "score": (essay + (rater == "b")) % 4,
                         "ell": essay % 2, "rater_type": "L"})
    result = drf_bootstrap(pd.DataFrame(rows), scale_min=0, scale_max=3, n_boot=5, seed=3)
    assert len(result) == 2
    assert {"drf_ell_logit", "p_bonferroni", "substantive_0_5_logit"}.issubset(result.columns)


def test_contamination_comparison():
    rows = []
    rewritten = []
    for essay in range(8):
        base = {"essay_id": str(essay), "model": "m", "prompt_variant": "zero_shot", "rubric_variant": "brief", "run_id": 1,
                "human_r1": 3, "human_r2": 3, "parse_status": "ok"}
        rows.append({**base, "parsed_score": 3})
        rewritten.append({**base, "parsed_score": 1})
    result = contamination_audit(pd.DataFrame(rows), pd.DataFrame(rewritten)).iloc[0]
    assert result["mae_original"] == 0
    assert result["mae_rewrite"] == 2
    assert result["original_minus_rewrite_mae"] == -2
