"""Command-line entry points for data preparation, scoring, validation, and analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from audit_llm_raters.adapters.tabular import (
    dataset_gate_audit,
    normalize_essays,
    stratified_sample,
)
from audit_llm_raters.analysis.pipeline import run_analysis
from audit_llm_raters.config import load_yaml
from audit_llm_raters.scoring.prompts import PromptLibrary
from audit_llm_raters.scoring.run import build_tasks, run_scoring
from audit_llm_raters.sim.validate import validate_estimators, validate_mfrm
from audit_llm_raters.store.sqlite import ScoreStore


def _read_scores(path: str) -> pd.DataFrame:
    source = Path(path)
    if source.suffix in {".db", ".sqlite", ".sqlite3"}:
        with ScoreStore(source) as store:
            return store.to_frame()
    if source.suffix == ".parquet":
        return pd.read_parquet(source)
    if source.suffix == ".csv":
        return pd.read_csv(source)
    raise ValueError("Scores must be SQLite, parquet, or CSV")


def _prepare(args: argparse.Namespace) -> None:
    essays = normalize_essays(args.input, load_yaml(args.dataset_config))
    anchors = stratified_sample(essays, min(len(essays), args.anchor_pool), args.seed + 1)
    remaining = essays[~essays["essay_id"].isin(anchors["essay_id"])]
    pilot = stratified_sample(remaining, args.n, args.seed)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    pilot.to_parquet(args.output, index=False)
    anchors.to_parquet(args.anchor_output, index=False)
    print(json.dumps({"pilot_essays": len(pilot), "held_out_anchors": len(anchors), "output": args.output, "data_gate": dataset_gate_audit(essays)}, ensure_ascii=False))


def _score(args: argparse.Namespace) -> None:
    essays = pd.read_parquet(args.essays)
    anchors = pd.read_parquet(args.anchors) if args.anchors else None
    models = load_yaml(args.models)["models"]
    tasks = build_tasks(
        essays, models, PromptLibrary(args.prompt_dir), anchors,
        args.prompt_variants, args.rubric_variants, args.repeats,
    )
    result = run_scoring(tasks, args.db, args.workers, args.execute, args.max_cost)
    if not args.execute:
        result["note"] = "Plan only. Re-run with --execute after checking model versions and prices."
    print(json.dumps(result, ensure_ascii=False))


def _analyze(args: argparse.Namespace) -> None:
    summary = run_analysis(_read_scores(args.scores), args.output, args.bootstrap, args.drf_boot, args.seed, args.bayes_g)
    print(json.dumps({"output": args.output, "rho_HL": summary.get("multivariate_g", {}).get("rho_HL")}, ensure_ascii=False))


def _validate(args: argparse.Namespace) -> None:
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    estimators = validate_estimators(args.replications, args.n_essays, args.seed)
    estimators.to_csv(output / "estimator_validation.csv", index=False)
    mfrm = validate_mfrm(args.seed)
    (output / "mfrm_validation.json").write_text(json.dumps(mfrm, indent=2), encoding="utf-8")
    print(json.dumps({"estimators": estimators.to_dict("records"), "MFRM": mfrm}))


def _demo(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(20261107)
    rows = []
    for index in range(args.n_essays):
        score = int(rng.integers(1, 7))
        rows.append({
            "essay_id": f"demo_{index}", "dataset": "synthetic_demo", "prompt_id": "demo_prompt",
            "essay_text": f"Synthetic essay {index}. " + "A measured claim requires evidence. " * (index + 3),
            "human_r1": score, "human_r2": min(6, max(1, score + int(rng.choice([-1, 0, 0, 1])))),
            "human_rater_ids": "[]", "scale_min": 1, "scale_max": 6, "ell": index % 2,
            "demographics_json": "{}",
        })
    essays = pd.DataFrame(rows)
    anchor_rows = []
    for index in range(4):
        row = rows[index].copy()
        row["essay_id"] = f"held_out_anchor_{index}"
        row["essay_text"] = f"Held-out synthetic calibration essay {index}. Evidence supports a clear position."
        anchor_rows.append(row)
    anchors = pd.DataFrame(anchor_rows)
    models = [{"name": f"mock_{i}", "provider": "mock", "model_version": f"mock-2026-{i}"} for i in range(2)]
    tasks = build_tasks(essays, models, PromptLibrary(args.prompt_dir), anchors, ["zero_shot", "anchored"], ["brief", "detailed"], 2)
    result = run_scoring(tasks, Path(args.output) / "demo.sqlite", workers=2, execute=True)
    with ScoreStore(Path(args.output) / "demo.sqlite") as store:
        scores = store.to_frame()
    summary = run_analysis(scores, Path(args.output) / "analysis", bootstrap=50, drf_boot=0)
    print(json.dumps({"scoring": result, "analysis": summary.get("multivariate_g", {})}, default=str))


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="audit-raters", description="Measurement audit of LLM essay raters")
    sub = root.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare", help="Normalize data and draw a stratified pilot")
    prepare.add_argument("--input", required=True)
    prepare.add_argument("--dataset-config", required=True)
    prepare.add_argument("--output", default="data/processed/pilot.parquet")
    prepare.add_argument("--anchor-output", default="data/processed/anchors.parquet")
    prepare.add_argument("--n", type=int, default=50)
    prepare.add_argument("--anchor-pool", type=int, default=100)
    prepare.add_argument("--seed", type=int, default=20261107)
    prepare.set_defaults(func=_prepare)

    score = sub.add_parser("score", help="Preview or execute a resumable scoring plan")
    score.add_argument("--essays", required=True)
    score.add_argument("--anchors")
    score.add_argument("--models", required=True)
    score.add_argument("--prompt-dir", default="prompts/v1")
    score.add_argument("--db", default="data/interim/scores.sqlite")
    score.add_argument("--prompt-variants", nargs="+", default=["zero_shot", "anchored"])
    score.add_argument("--rubric-variants", nargs="+", default=["brief", "detailed"])
    score.add_argument("--repeats", type=int, default=2)
    score.add_argument("--workers", type=int, default=4)
    score.add_argument("--execute", action="store_true", help="Actually incur provider API calls")
    score.add_argument("--max-cost", type=float)
    score.set_defaults(func=_score)

    analyze = sub.add_parser("analyze", help="Run the ordered statistical pipeline")
    analyze.add_argument("--scores", required=True)
    analyze.add_argument("--output", default="out/pilot")
    analyze.add_argument("--bootstrap", type=int, default=500)
    analyze.add_argument("--drf-boot", type=int, default=0)
    analyze.add_argument("--bayes-g", action="store_true", help="Also fit optional NumPyro multivariate G posterior")
    analyze.add_argument("--seed", type=int, default=20261107)
    analyze.set_defaults(func=_analyze)

    validate = sub.add_parser("validate", help="Run simulation-based estimator recovery gates")
    validate.add_argument("--output", default="out/validation")
    validate.add_argument("--replications", type=int, default=30)
    validate.add_argument("--n-essays", type=int, default=400)
    validate.add_argument("--seed", type=int, default=20261107)
    validate.set_defaults(func=_validate)

    demo = sub.add_parser("demo", help="Offline mock end-to-end test; not empirical evidence")
    demo.add_argument("--output", default="out/demo")
    demo.add_argument("--prompt-dir", default="prompts/v1")
    demo.add_argument("--n-essays", type=int, default=12)
    demo.set_defaults(func=_demo)
    return root


def main() -> None:
    args = parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
