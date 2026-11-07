"""Configurable adapters for ASAP, ASAP 2.0, PERSUADE, and ELLIPSE tables."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from audit_llm_raters.config import require_keys
from audit_llm_raters.schema import ESSAY_COLUMNS


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    readers = {
        ".csv": pd.read_csv,
        ".tsv": lambda p: pd.read_csv(p, sep="\t"),
        ".parquet": pd.read_parquet,
        ".json": pd.read_json,
        ".jsonl": lambda p: pd.read_json(p, lines=True),
    }
    if path.suffix.lower() not in readers:
        raise ValueError(f"Unsupported table format: {path.suffix}")
    return readers[path.suffix.lower()](path)


def normalize_essays(path: str | Path, dataset_config: dict[str, Any]) -> pd.DataFrame:
    require_keys(dataset_config, ["name", "columns", "scale"], "dataset config")
    frame = read_table(path)
    columns = dataset_config["columns"]
    missing = [source for source in columns.values() if source and source not in frame.columns]
    if missing:
        raise ValueError(f"Missing source columns: {sorted(set(missing))}")

    def get(key: str, default: Any = None) -> Any:
        source = columns.get(key)
        return frame[source] if source else pd.Series(default, index=frame.index)

    scale = dataset_config["scale"]
    prompt_ids = get("prompt_id", dataset_config.get("prompt_id", "all")).astype(str)
    default_min = int(scale["min"])
    default_max = int(scale["max"])
    result = pd.DataFrame(
        {
            "essay_id": get("essay_id").astype(str),
            "dataset": dataset_config["name"],
            "prompt_id": prompt_ids,
            "essay_text": get("essay_text").astype(str),
            "human_r1": pd.to_numeric(get("human_r1"), errors="coerce"),
            "human_r2": pd.to_numeric(get("human_r2"), errors="coerce"),
            "scale_min": pd.to_numeric(get("scale_min", default_min), errors="coerce").fillna(default_min).astype(int),
            "scale_max": pd.to_numeric(get("scale_max", default_max), errors="coerce").fillna(default_max).astype(int),
            "ell": pd.to_numeric(get("ell"), errors="coerce").astype("Int64"),
        }
    )
    for prompt_id, prompt_scale in dataset_config.get("scales_by_prompt", {}).items():
        mask = result["prompt_id"] == str(prompt_id)
        result.loc[mask, "scale_min"] = int(prompt_scale["min"])
        result.loc[mask, "scale_max"] = int(prompt_scale["max"])
    rater1 = get("human_rater1", "").fillna("").astype(str)
    rater2 = get("human_rater2", "").fillna("").astype(str)
    result["human_rater_ids"] = pd.Series(
        [json.dumps([a, b]) for a, b in zip(rater1, rater2)], index=frame.index
    )
    demographics = dataset_config.get("demographics", [])
    missing_demo = [column for column in demographics if column not in frame.columns]
    if missing_demo:
        raise ValueError(f"Missing demographic columns: {missing_demo}")
    result["demographics_json"] = pd.Series(
        [json.dumps({key: row[key] for key in demographics}, default=str) for _, row in frame.iterrows()],
        index=frame.index,
    ) if demographics else "{}"
    result = result[ESSAY_COLUMNS]
    if result["essay_id"].duplicated().any():
        raise ValueError("essay_id must be unique after normalization")
    if result["essay_text"].str.strip().eq("").any():
        raise ValueError("Empty essay_text values are not allowed")
    return result


def stratified_sample(frame: pd.DataFrame, n: int, seed: int = 20261107) -> pd.DataFrame:
    """Sample without replacement, prioritizing low/high score strata within each prompt."""
    work = frame.dropna(subset=["human_r1", "human_r2"]).copy().reset_index(drop=True)
    if n > len(work):
        raise ValueError(f"Requested {n} double-scored essays from a pool of {len(work)}")
    rng = np.random.default_rng(seed)
    work["_score"] = work[["human_r1", "human_r2"]].mean(axis=1)
    work["_stratum"] = work["_score"].round().astype(str) + "|" + work["prompt_id"].astype(str)
    strata = {name: indices for name, indices in work.groupby("_stratum").indices.items()}
    selected: list[int] = []
    if len(strata) > n:
        queues = []
        for _, prompt in work.groupby("prompt_id"):
            names = sorted(prompt["_stratum"].unique(), key=lambda name: work.loc[work["_stratum"] == name, "_score"].mean())
            queue = []
            low, high = 0, len(names) - 1
            while low <= high:
                queue.append(names[low])
                low += 1
                if low <= high:
                    queue.append(names[high])
                    high -= 1
            queues.append(queue)
        ordered = []
        while any(queues):
            for queue in queues:
                if queue:
                    ordered.append(queue.pop(0))
        strata_order = ordered[:n]
    else:
        strata_order = list(strata)
    for name in strata_order:
        selected.append(int(rng.choice(strata[name])))
    selected_set = set(selected)
    remaining = [index for index in rng.permutation(work.index) if index not in selected_set]
    selected.extend(remaining[: n - len(selected)])
    return work.loc[selected].drop(columns=["_score", "_stratum"]).sort_values("essay_id").reset_index(drop=True)


def choose_anchors(pool: pd.DataFrame, prompt_id: str, n: int = 4) -> list[dict]:
    candidates = pool[pool["prompt_id"].astype(str) == str(prompt_id)].copy()
    if len(candidates) < n:
        raise ValueError(f"Need at least {n} held-out anchors for prompt {prompt_id}")
    candidates["_score"] = candidates[["human_r1", "human_r2"]].mean(axis=1)
    quantiles = np.linspace(0.1, 0.9, n)
    chosen = []
    used: set[str] = set()
    for quantile in quantiles:
        target = candidates["_score"].quantile(quantile)
        ordered = candidates.assign(_distance=(candidates["_score"] - target).abs()).sort_values("_distance")
        row = next(row for _, row in ordered.iterrows() if row["essay_id"] not in used)
        used.add(row["essay_id"])
        resolved_score = int(np.clip(np.rint(row["_score"]), row["scale_min"], row["scale_max"]))
        chosen.append({"essay_id": row["essay_id"], "essay_text": row["essay_text"], "score": resolved_score})
    return chosen


def dataset_gate_audit(frame: pd.DataFrame) -> dict[str, Any]:
    """Report G1 and cross-essay rater-ID diagnostics before paid scoring."""
    paired = frame[["human_r1", "human_r2"]].notna().all(axis=1)
    ids: list[str] = []
    rater_essays: dict[str, set[str]] = {}
    for row in frame[["essay_id", "human_rater_ids"]].itertuples(index=False):
        try:
            values = json.loads(row.human_rater_ids)
            for value in values:
                if value is None or not str(value).strip():
                    continue
                rater_id = str(value)
                ids.append(rater_id)
                rater_essays.setdefault(rater_id, set()).add(str(row.essay_id))
        except (TypeError, json.JSONDecodeError):
            continue
    repeated = sum(len(essays) > 1 for essays in rater_essays.values())
    return {
        "n_essays": len(frame),
        "n_with_two_independent_scores": int(paired.sum()),
        "double_score_coverage": float(paired.mean()) if len(frame) else 0.0,
        "G1_pass": bool(paired.sum() > 0),
        "n_nonempty_rater_ids": len(ids),
        "n_unique_rater_ids": len(set(ids)),
        "human_rater_ids_link_across_essays": bool(repeated > 0),
        "human_design": "crossed" if repeated > 0 else "nested_within_essay",
    }
