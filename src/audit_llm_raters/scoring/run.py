"""Resumable factorial scoring runner with explicit execution safeguards."""

from __future__ import annotations

import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from audit_llm_raters.adapters.tabular import choose_anchors
from audit_llm_raters.schema import ScoreRecord
from audit_llm_raters.scoring.client import ProviderResponse, make_provider
from audit_llm_raters.scoring.parse import parse_score
from audit_llm_raters.scoring.prompts import PromptLibrary, format_anchors
from audit_llm_raters.store.sqlite import ScoreStore


@dataclass(frozen=True)
class ScoringTask:
    request_id: str
    essay: dict[str, Any]
    model: dict[str, Any]
    prompt_variant: str
    rubric_variant: str
    run_id: int
    prompt_version: str
    system: str
    user: str


def _request_id(essay: dict, model: dict, prompt_variant: str, rubric_variant: str, run_id: int, version: str) -> str:
    identity = {
        "essay_id": essay["essay_id"],
        "essay_sha256": hashlib.sha256(essay["essay_text"].encode()).hexdigest(),
        "model": model["name"],
        "model_version": model["model_version"],
        "prompt_variant": prompt_variant,
        "rubric_variant": rubric_variant,
        "run_id": run_id,
        "prompt_version": version,
    }
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def build_tasks(
    essays: pd.DataFrame,
    models: list[dict[str, Any]],
    prompt_library: PromptLibrary,
    anchor_pool: pd.DataFrame | None,
    prompt_variants: list[str],
    rubric_variants: list[str],
    repeats: int,
) -> list[ScoringTask]:
    tasks = []
    anchor_cache: dict[str, str] = {}
    for essay in essays.to_dict("records"):
        prompt_id = str(essay["prompt_id"])
        if "anchored" in prompt_variants and prompt_id not in anchor_cache:
            if anchor_pool is None:
                raise ValueError("anchor_pool is required for anchored prompts")
            anchor_cache[prompt_id] = format_anchors(choose_anchors(anchor_pool, prompt_id))
        for model in models:
            for prompt_variant in prompt_variants:
                for rubric_variant in rubric_variants:
                    rendered = prompt_library.render(
                        rubric_variant=rubric_variant,
                        prompt_variant=prompt_variant,
                        scale_min=int(essay["scale_min"]),
                        scale_max=int(essay["scale_max"]),
                        essay_text=essay["essay_text"],
                        anchors=anchor_cache.get(prompt_id, ""),
                        prompt_id=prompt_id,
                    )
                    for run_id in range(1, repeats + 1):
                        request_id = _request_id(
                            essay, model, prompt_variant, rubric_variant, run_id, rendered.version
                        )
                        tasks.append(ScoringTask(
                            request_id, essay, model, prompt_variant, rubric_variant,
                            run_id, rendered.version, rendered.system, rendered.user,
                        ))
    return tasks


def _execute(task: ScoringTask) -> ScoreRecord:
    started = time.monotonic()
    essay = task.essay
    base = {
        "request_id": task.request_id,
        "essay_id": str(essay["essay_id"]),
        "dataset": str(essay["dataset"]),
        "prompt_id": str(essay["prompt_id"]),
        "human_r1": essay.get("human_r1"),
        "human_r2": essay.get("human_r2"),
        "human_rater_ids": str(essay.get("human_rater_ids", "[]")),
        "model": task.model["name"],
        "model_version": task.model["model_version"],
        "prompt_variant": task.prompt_variant,
        "rubric_variant": task.rubric_variant,
        "run_id": task.run_id,
        "scale_min": int(essay["scale_min"]),
        "scale_max": int(essay["scale_max"]),
        "ell": None if pd.isna(essay.get("ell")) else int(essay["ell"]),
    }
    try:
        response: ProviderResponse = make_provider(task.model).generate(task.system, task.user)
        parsed = parse_score(response.content, base["scale_min"], base["scale_max"])
        metadata = {"prompt_version": task.prompt_version, **response.metadata}
        return ScoreRecord(
            **base,
            raw_response=response.raw_response,
            parsed_score=parsed.score,
            parse_status=parsed.status,
            parse_detail=parsed.detail,
            tokens_in=response.tokens_in,
            tokens_out=response.tokens_out,
            cost=response.cost,
            latency_ms=(time.monotonic() - started) * 1000,
            metadata_json=json.dumps(metadata, ensure_ascii=False),
        )
    except Exception as exc:  # stored as a retryable call failure, not silently dropped
        return ScoreRecord(
            **base,
            raw_response=f"{type(exc).__name__}: {exc}",
            parsed_score=None,
            parse_status="api_error",
            parse_detail="",
            tokens_in=None,
            tokens_out=None,
            cost=0.0,
            error=f"{type(exc).__name__}: {exc}",
            latency_ms=(time.monotonic() - started) * 1000,
            metadata_json=json.dumps({"prompt_version": task.prompt_version}),
        )


def run_scoring(
    tasks: list[ScoringTask],
    store_path: str | Path,
    workers: int = 4,
    execute: bool = False,
    max_cost: float | None = None,
) -> dict[str, Any]:
    real_tasks = [task for task in tasks if task.model.get("provider") != "mock"]
    estimated_by_currency: dict[str, float] = {}
    pricing_complete = True
    unpriced_tasks = 0
    for task in tasks:
        pricing = task.model.get("pricing")
        if not pricing:
            pricing_complete = False
            unpriced_tasks += 1
            continue
        currency = str(task.model.get("currency", "USD"))
        tokens_in = max(1, (len(task.system) + len(task.user)) // 4)
        tokens_out = int(task.model.get("max_output_tokens", 200))
        estimated_by_currency[currency] = estimated_by_currency.get(currency, 0.0) + (
            tokens_in * float(pricing.get("input_per_million", 0))
            + tokens_out * float(pricing.get("output_per_million", 0))
        ) / 1_000_000
    generic_rubrics = any("generic rubric" in task.system or "generic template" in task.system for task in tasks)
    if not execute:
        one_currency_estimate = (
            next(iter(estimated_by_currency.values()))
            if pricing_complete and len(estimated_by_currency) == 1 else None
        )
        return {
            "planned": len(tasks), "completed": 0, "cost": 0.0,
            "estimated_cost_no_cache_or_batch_discount": one_currency_estimate,
            "estimated_cost_by_currency": estimated_by_currency,
            "pricing_complete": pricing_complete, "unpriced_tasks": unpriced_tasks,
            "uses_generic_test_rubrics": generic_rubrics,
        }
    real_currencies = {
        str(task.model.get("currency", "USD")) for task in real_tasks if task.model.get("pricing")
    }
    if len(real_currencies) > 1:
        currencies = ", ".join(sorted(real_currencies))
        raise ValueError(
            f"Real scoring mixes cost currencies ({currencies}). Run each currency in a separate "
            "database so costs and hard caps are never added across currencies."
        )
    cost_currency = next(iter(real_currencies), None)
    if any(
        "REPLACE_WITH" in value
        for task in real_tasks for value in task.model.values() if isinstance(value, str)
    ):
        raise ValueError("Replace model-version and source-revision placeholders before real scoring")
    if generic_rubrics and real_tasks:
        raise ValueError("Official prompt-specific rubrics are required before real scoring")
    if max_cost is not None and not pricing_complete:
        raise ValueError("A cost cap requires input/output pricing for every planned model")
    completed = 0
    session_cost = 0.0
    with ScoreStore(store_path) as store:
        pending = [task for task in tasks if task.request_id not in store.completed_ids()]
        total_cost = store.total_cost()
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for start in range(0, len(pending), workers):
                batch = pending[start:start + workers]
                futures = [executor.submit(_execute, task) for task in batch]
                for future in futures:
                    record = future.result()
                    store.add(record)
                    completed += 1
                    session_cost += record.cost or 0.0
                    total_cost += record.cost or 0.0
                if max_cost is not None and total_cost > max_cost:
                    unit = cost_currency or "pricing units"
                    raise RuntimeError(
                        f"Cost cap exceeded after a bounded worker batch: "
                        f"{total_cost:.4f} {unit} > {max_cost:.4f} {unit}. "
                        "Completed calls are saved; inspect pricing before resuming."
                    )
    return {
        "planned": len(tasks), "completed": completed,
        "cost": session_cost, "cost_currency": cost_currency,
    }
