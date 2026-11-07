"""Canonical data contracts shared by scoring and analysis."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

SCORE_COLUMNS = [
    "request_id", "essay_id", "dataset", "prompt_id", "human_r1", "human_r2",
    "human_rater_ids", "model", "model_version", "prompt_variant",
    "rubric_variant", "run_id", "raw_response", "parsed_score", "parse_status",
    "parse_detail", "tokens_in", "tokens_out", "cost", "timestamp", "scale_min",
    "scale_max", "ell", "error", "latency_ms", "metadata_json",
]

ESSAY_COLUMNS = [
    "essay_id", "dataset", "prompt_id", "essay_text", "human_r1", "human_r2",
    "human_rater_ids", "scale_min", "scale_max", "ell", "demographics_json",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class ScoreRecord:
    request_id: str
    essay_id: str
    dataset: str
    prompt_id: str
    human_r1: float | None
    human_r2: float | None
    human_rater_ids: str
    model: str
    model_version: str
    prompt_variant: str
    rubric_variant: str
    run_id: int
    raw_response: str
    parsed_score: int | None
    parse_status: str
    tokens_in: int | None
    tokens_out: int | None
    cost: float | None
    timestamp: str = field(default_factory=utc_now)
    scale_min: int = 0
    scale_max: int = 6
    parse_detail: str = ""
    ell: int | None = None
    error: str = ""
    latency_ms: float | None = None
    metadata_json: str = "{}"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)
