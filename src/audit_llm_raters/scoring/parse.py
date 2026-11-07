"""Strict JSON score parsing with informative-failure taxonomy."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

REFUSAL_RE = re.compile(
    r"\b(can(?:not|'t)|unable to|refuse|not able to)\b|无法|不能|拒绝|不便评价",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ParseResult:
    score: int | None
    status: str
    detail: str = ""


def _extract_json(text: str) -> tuple[dict | None, str]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        value = json.loads(stripped)
        return (value if isinstance(value, dict) else None), ""
    except json.JSONDecodeError:
        pass
    start = stripped.find("{")
    if start < 0:
        return None, "no JSON object found"
    try:
        value, end = json.JSONDecoder().raw_decode(stripped[start:])
        if not isinstance(value, dict):
            return None, "JSON value is not an object"
        suffix = stripped[start + end :].strip()
        return value, "extra text after JSON" if suffix else ""
    except json.JSONDecodeError as exc:
        return None, f"invalid JSON: {exc.msg}"


def parse_score(text: str, scale_min: int, scale_max: int) -> ParseResult:
    if not text or not text.strip():
        return ParseResult(None, "empty_response")
    if scale_min > scale_max:
        raise ValueError("scale_min must not exceed scale_max")
    value, detail = _extract_json(text)
    if value is None:
        status = "refusal" if REFUSAL_RE.search(text) else "invalid_json"
        return ParseResult(None, status, detail)
    if "score" not in value:
        return ParseResult(None, "missing_score", detail)
    score = value["score"]
    if isinstance(score, bool) or not isinstance(score, int):
        return ParseResult(None, "non_integer_score", f"score={score!r}")
    if not scale_min <= score <= scale_max:
        return ParseResult(None, "out_of_range", f"score={score}, allowed={scale_min}..{scale_max}")
    if "reason" not in value:
        return ParseResult(None, "missing_reason", detail)
    reason = value["reason"]
    if not isinstance(reason, str):
        return ParseResult(None, "invalid_reason", "reason must be a string")
    return ParseResult(score, "ok", detail)
