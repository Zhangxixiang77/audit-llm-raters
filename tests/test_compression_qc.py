import numpy as np
import pandas as pd

from audit_llm_raters.analysis.compression import compression_audit
from audit_llm_raters.analysis.qc import kendall_tau_a


def test_kendall_tau_a_counts_ties_as_zero():
    x = np.array([1, 1, 2, 3])
    y = np.array([1, 2, 3, 4])
    assert kendall_tau_a(x, y) == 5 / 6


def test_compression_uses_human_baseline():
    rows = []
    for essay in range(100):
        h2 = essay % 6 + 1
        h1 = h2 + (1 if essay % 11 == 0 else 0)
        llm = 3 + 0.5 * (h2 - 3)
        for model in ["m"]:
            rows.append({"dataset": "d", "prompt_id": "p", "model": model, "essay_id": str(essay),
                         "parsed_score": llm, "parse_status": "ok", "human_r1": h1, "human_r2": h2,
                         "scale_min": 1, "scale_max": 6})
    result = compression_audit(pd.DataFrame(rows)).iloc[0]
    assert result["human_baseline_slope"] > 0.9
    assert abs(result["llm_slope"] - 0.5) < 0.01
    assert result["excess_compression_vs_human"] > 0.4
