"""Prepare reliable ELLIPSE essays while preserving independent raw ratings."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from audit_llm_raters.adapters.tabular import read_table

DIMENSIONS = [
    "Overall",
    "Cohesion",
    "Syntax",
    "Vocabulary",
    "Phraseology",
    "Grammar",
    "Conventions",
]
RAW_SCORE_COLUMNS = [f"{dimension}_{rater}" for dimension in DIMENSIONS for rater in (1, 2)]
RAW_REQUIRED_COLUMNS = [
    "Filename",
    "text_id_kaggle",
    "Text",
    "Rater_1",
    "Rater_2",
    "Identifying_Info_1",
    "Identifying_Info_2",
    *RAW_SCORE_COLUMNS,
]
METADATA_REQUIRED_COLUMNS = [
    "text_id_kaggle",
    "full_text",
    "gender",
    "grade",
    "race_ethnicity",
    "SES",
    "prompt",
    "task",
]


def _require_columns(frame: pd.DataFrame, required: list[str], label: str) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing {label} columns: {missing}")


def prepare_ellipse_overall(
    raw_ratings_path: str | Path,
    metadata_path: str | Path,
    *,
    exclude_identifying_info: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Join raw scores to reliable-corpus metadata and return an auditable source table.

    The raw table contains a small number of zero-coded ratings although the official
    rubric ranges from 1 to 5. Those records are excluded rather than treated as the
    lowest valid score. Essays without a reliable-corpus ID cannot be joined to the
    metadata table and remain unused. No adjudicated or published average replaces
    either independent human rating.
    """
    raw = read_table(raw_ratings_path)
    metadata = read_table(metadata_path)
    _require_columns(raw, RAW_REQUIRED_COLUMNS, "ELLIPSE raw-rating")
    _require_columns(metadata, METADATA_REQUIRED_COLUMNS, "ELLIPSE metadata")

    raw = raw.copy()
    metadata = metadata.copy()
    raw["text_id_kaggle"] = raw["text_id_kaggle"].astype("string")
    metadata["text_id_kaggle"] = metadata["text_id_kaggle"].astype("string")
    if metadata["text_id_kaggle"].isna().any() or metadata["text_id_kaggle"].duplicated().any():
        raise ValueError("ELLIPSE metadata text_id_kaggle values must be nonmissing and unique")

    raw_without_id = raw["text_id_kaggle"].isna() | raw["text_id_kaggle"].str.strip().eq("")
    identified = raw.loc[~raw_without_id].copy()
    if identified["text_id_kaggle"].duplicated().any():
        raise ValueError("ELLIPSE raw text_id_kaggle values must be unique when present")

    joined = metadata.merge(
        identified,
        on="text_id_kaggle",
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    matched = joined["_merge"].eq("both")
    valid_scores = joined[RAW_SCORE_COLUMNS].isin(range(1, 6)).all(axis=1)
    no_identifying_info = (
        joined["Identifying_Info_1"].eq(0) & joined["Identifying_Info_2"].eq(0)
    )
    nonempty_text = joined["Text"].notna() & joined["Text"].astype("string").str.strip().ne("")
    keep = matched & valid_scores & nonempty_text
    if exclude_identifying_info:
        keep &= no_identifying_info

    selected = joined.loc[keep].drop(columns=["_merge"]).copy()
    selected = selected.sort_values("text_id_kaggle").reset_index(drop=True)
    raters = pd.concat([selected["Rater_1"], selected["Rater_2"]], ignore_index=True)
    rater_counts = raters.value_counts()
    published_mean_difference = (
        selected["Overall"] - selected[["Overall_1", "Overall_2"]].mean(axis=1)
    ).abs()
    audit: dict[str, Any] = {
        "raw_rows": int(len(raw)),
        "raw_rows_without_reliable_corpus_id": int(raw_without_id.sum()),
        "metadata_rows": int(len(metadata)),
        "matched_metadata_rows": int(matched.sum()),
        "unmatched_metadata_rows": int((~matched).sum()),
        "matched_rows_with_any_zero_or_out_of_range_score": int((matched & ~valid_scores).sum()),
        "matched_rows_with_identifying_info_flag": int((matched & ~no_identifying_info).sum()),
        "excluded_identifying_info": bool(exclude_identifying_info),
        "eligible_rows": int(len(selected)),
        "eligible_score_min": int(selected[RAW_SCORE_COLUMNS].min().min()) if len(selected) else None,
        "eligible_score_max": int(selected[RAW_SCORE_COLUMNS].max().max()) if len(selected) else None,
        "n_unique_human_raters": int(raters.nunique()),
        "human_rater_ids_link_across_essays": bool((rater_counts > 1).any()),
        "human_rater_essay_count_min": int(rater_counts.min()) if len(rater_counts) else 0,
        "human_rater_essay_count_median": float(rater_counts.median()) if len(rater_counts) else None,
        "human_rater_essay_count_max": int(rater_counts.max()) if len(rater_counts) else 0,
        "eligible_published_overall_not_equal_raw_mean": int(
            (published_mean_difference > 1e-8).sum()
        ),
        "human_score_source": "Overall_1 and Overall_2; never the published average",
        "score_scale": {"min": 1, "max": 5},
        "zero_coded_scores_are_valid": False,
    }
    return selected, audit
