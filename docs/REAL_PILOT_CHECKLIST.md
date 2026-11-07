# Real-pilot go/no-go checklist

## Before any paid call

- [ ] Source license permits local research use; no essay text is committed.
- [ ] G1: at least one dataset has two independent human scores, not only an adjudicated score.
- [ ] Cross-essay human rater IDs have been checked; nested/crossed handling matches the result.
- [ ] All selected prompts have official brief and detailed rubric files.
- [ ] Pilot and anchor pools are disjoint; extreme score strata are adequately represented.
- [ ] Six model families and dated versions are frozen in the local model config.
- [ ] Reasoning/thinking disable parameters are verified for each dated model.
- [ ] Official current token prices and batch/cache assumptions are recorded.
- [ ] Provider keys exist only in environment variables; no real response files are tracked.
- [ ] A scoring plan was reviewed without `--execute`; a cost cap is set.

## During the 50-essay pilot

- [ ] Every raw provider response is present in the local SQLite store.
- [ ] API errors are distinguished from returned-but-invalid ratings.
- [ ] Failures are reported by model, rubric, prompt variant, essay quality, and ELL where available.
- [ ] G2: mean cross-family residual correlation is computed and interpreted as a directional gate, not a significance test.
- [ ] G3: every configuration has failure rate below 5%, and measured cost stays within budget.
- [ ] Any factor degradation follows the protocol order; model families are not removed.

## Before full scoring

- [ ] `audit-raters validate` passes both primary-estimator and MFRM recovery checks.
- [ ] EMS negative components and bootstrap diagnostics are inspected rather than hidden.
- [ ] MFRM convergence, severity/threshold recovery, infit/outfit, and category use are reviewed.
- [ ] Human baseline compression slopes and ELL cell sizes are adequate.
- [ ] Analysis choices and model versions are frozen before the full run.
- [ ] A small MFRM subsample is reserved for independent TAM cross-checking before submission.

## Reporting

- [ ] Simulation results are labeled as validation, never empirical evidence.
- [ ] Human Φ noise floor accompanies LLM-only Φ.
- [ ] Both missing and floor failure sensitivity analyses are reported.
- [ ] Public-data contamination and nested human-rater limitations are disclosed.
- [ ] Public artifacts contain no source essays, secrets, or deanonymized review material.
