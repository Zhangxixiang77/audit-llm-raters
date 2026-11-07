import pandas as pd

from audit_llm_raters.adapters.ellipse import (
    DIMENSIONS,
    prepare_ellipse_overall,
)
from audit_llm_raters.adapters.tabular import dataset_gate_audit, normalize_essays


def test_ellipse_preparation_preserves_raw_scores_and_excludes_invalid(tmp_path):
    raw_rows = []
    for essay_id, overall_1, overall_2, identifying in [
        ("A", 2, 4, 0),
        ("B", 0, 3, 0),
        ("C", 3, 3, 1),
        ("D", 3, 4, 0),
    ]:
        row = {
            "Filename": f"file_{essay_id}",
            "text_id_kaggle": essay_id,
            "Text": f"Essay {essay_id}",
            "Rater_1": "rater_a",
            "Rater_2": "rater_b",
            "Identifying_Info_1": identifying,
            "Identifying_Info_2": 0,
        }
        for dimension in DIMENSIONS:
            row[f"{dimension}_1"] = overall_1
            row[f"{dimension}_2"] = overall_2
        raw_rows.append(row)
    raw = pd.DataFrame(raw_rows)
    metadata = pd.DataFrame(
        {
            "text_id_kaggle": ["A", "B", "C", "E"],
            "full_text": ["Essay A", "Essay B", "Essay C", "Essay E"],
            "gender": ["F"] * 4,
            "grade": [8] * 4,
            "race_ethnicity": ["group"] * 4,
            "SES": ["status"] * 4,
            "prompt": ["topic"] * 4,
            "task": ["Independent"] * 4,
            "Overall": [3.0, 1.5, 3.0, 2.0],
        }
    )
    raw_path = tmp_path / "raw.csv"
    metadata_path = tmp_path / "metadata.csv"
    raw.to_csv(raw_path, index=False)
    metadata.to_csv(metadata_path, index=False)

    prepared, audit = prepare_ellipse_overall(raw_path, metadata_path)
    assert prepared["text_id_kaggle"].tolist() == ["A"]
    assert prepared[["Overall_1", "Overall_2"]].iloc[0].tolist() == [2, 4]
    assert audit["unmatched_metadata_rows"] == 1
    assert audit["matched_rows_with_any_zero_or_out_of_range_score"] == 1
    assert audit["matched_rows_with_identifying_info_flag"] == 1
    assert audit["eligible_rows"] == 1
    assert audit["human_rater_ids_link_across_essays"] is False

    prepared_path = tmp_path / "prepared.csv"
    prepared.to_csv(prepared_path, index=False)
    essays = normalize_essays(
        prepared_path,
        {
            "name": "ellipse",
            "prompt_id": "overall",
            "columns": {
                "essay_id": "text_id_kaggle",
                "prompt_id": None,
                "essay_text": "Text",
                "human_r1": "Overall_1",
                "human_r2": "Overall_2",
                "human_rater1": "Rater_1",
                "human_rater2": "Rater_2",
            },
            "scale": {"min": 1, "max": 5},
            "demographics": ["prompt"],
        },
    )
    gate = dataset_gate_audit(essays)
    assert essays.loc[0, "prompt_id"] == "overall"
    assert essays.loc[0, "human_r1"] == 2
    assert essays.loc[0, "human_r2"] == 4
    assert gate["G1_pass"]
    assert gate["n_unique_rater_ids"] == 2
