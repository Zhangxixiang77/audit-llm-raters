# audit-llm-raters

Reproducible Python pipeline for **When Raters Agree for the Wrong Reasons: A Generalizability-Theory and Many-Facet Rasch Audit of LLM Essay Scoring**.

The repository implements the proposal's mechanism-first route: distinguish shared error uncorrelated with true ability (`u`) from linear central compression (`c`). LLM-only agreement is never treated as sufficient evidence of score validity.

## What is implemented

- Configurable CSV/TSV/parquet adapters for ASAP-style and related essay datasets, per-prompt scales, score-stratified pilot sampling, held-out anchors, and G1 data diagnostics.
- Versioned zero-shot/anchored and brief/detailed prompts; prompt-specific official rubric overrides.
- OpenAI-compatible, Anthropic, Gemini, local vLLM, and offline mock providers.
- Deterministic request IDs, bounded concurrency, retries, resumable SQLite storage, full raw provider responses, token/cost fields, and an explicit `--execute` cost safeguard.
- Strict JSON parsing with separate refusal, invalid JSON, missing score, non-integer, out-of-range, empty-response, and API-error categories.
- QC, QWK, Kendall τa/τb, human Φ noise floor, G2 residual-correlation and G3 failure-rate gates.
- Balanced crossed EMS for the full facet design; two-facet G studies, multivariate `ρ_HL`, heterogeneous-residual `π_pt`, bootstrap intervals, and D-study saturation curves.
- Adjacent-category MFRM with severity, difficulty, thresholds, infit/outfit, category use, nested human-rater handling, regularized MAP, and optional NumPyro NUTS.
- Human-baseline compression slopes, implied compression targets, rater×ELL DRF with cluster bootstrap/Bonferroni inference, failure-policy sensitivity, balanced subsampling, and paraphrase contamination comparisons.
- Simulation-based recovery checks for both primary estimands and MFRM parameters.

**Status:** the software and synthetic recovery checks do not constitute empirical findings. A real 50-essay pilot requires licensed source data, official prompt-specific rubrics, dated model IDs, current prices, and provider credentials. No real scores are claimed in this repository.

## Install

Python 3.10+ is required.

```bash
git clone https://github.com/Zhangxixiang77/audit-llm-raters.git
cd audit-llm-raters
python -m pip install -e '.[dev]'
pytest
```

Optional full Bayesian MFRM and multivariate G estimation:

```bash
python -m pip install -e '.[bayes]'
```

## Five-minute offline verification

```bash
audit-raters validate --replications 30 --output out/validation
audit-raters demo --output out/demo
```

The demo uses a deterministic mock provider and synthetic essays. Its output is only an integration test and must never be reported as an empirical result. Provider contract tests use local mock HTTP transports and never make paid calls; the two NumPyro smoke tests are skipped automatically when the `bayes` extra is absent.

## Run the real 50-essay pilot

### 1. Prepare source data locally

Place the source table under `data/raw/`. The official Kaggle download can require login and acceptance of the competition terms; complete that access step rather than substituting an unverified mirror. These files are ignored by git and are not redistributed. Edit `configs/datasets/asap.yaml`, or copy `generic.example.yaml`, to match the actual columns. The ASAP config freezes the proposal's four-prompt pilot to prompts 1, 3, 7, and 8; change `prompt_ids` only if the protocol changes. For another dataset, verify that scores are two independent human ratings rather than only an adjudicated score.

ELLIPSE is directly available from its [official repository](https://github.com/scrosseye/ELLIPSE-Corpus) under CC BY 4.0. Its raw-rating archive supplies two independent 1–5 scores and cross-essay rater IDs. After extracting that archive with the authors' published password `ellipse_raw_data`, run:

```bash
audit-raters prepare-ellipse \
  --raw-ratings data/raw/ellipse/raw_raters/ellipsis_raw_rater_scores_anon_all_essay.csv \
  --metadata data/raw/ellipse/ELLIPSE_Final_github_train.csv

audit-raters prepare \
  --input data/interim/ellipse_overall.csv \
  --dataset-config configs/datasets/ellipse.yaml \
  --n 50 --anchor-pool 100 \
  --output data/processed/ellipse_pilot50.parquet \
  --anchor-output data/processed/ellipse_anchors100.parquet
```

The ELLIPSE preparation excludes unmatched records, zero/out-of-range raw scores, and identifying-information flags; it never substitutes the published average for either independent rating. See `docs/DATA_SOURCES.md` for provenance, source hashes, aggregate diagnostics, and the all-ELL limitation for RQ3.

For a custom prompt, put the official rubrics here:

```bash
mkdir -p prompts/v1/rubrics
```

```text
prompts/v1/rubrics/<prompt_id>_brief.txt
prompts/v1/rubrics/<prompt_id>_detailed.txt
```

Include the complete assignment and any required source text in these files. Prompt-specific rubric files are local, git-ignored research inputs. The generic rubrics shipped at the top level are software-test placeholders, not valid substitutes for an empirical rubric.

```bash
audit-raters prepare \
  --input data/raw/training_set_rel3.tsv \
  --dataset-config configs/datasets/asap.yaml \
  --n 50 \
  --output data/processed/pilot.parquet \
  --anchor-output data/processed/anchors.parquet
```

Check `data_gate` in the output: G1 requires independent double scores. If human IDs do not repeat across essays, MFRM treats human raters as nested within essays.

### 2. Configure dated models and secrets locally

```bash
cp configs/models.example.yaml configs/local.models.yaml
```

Replace every `REPLACE_WITH_*` placeholder and verify current official API pricing. Set keys only in the shell environment, for example `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `DEEPSEEK_API_KEY`, or `DASHSCOPE_API_KEY`. Do not paste keys into YAML, commits, issues, or chat. Reasoning-disable parameters are provider/model-specific; set them only after checking support for the selected dated model. The preregistered six-family candidate, per-model budgets, smoke-test gates, and external-GPU requirement are documented in `docs/SIX_MODEL_PILOT.md`.

### 3. Preview, then explicitly execute

```bash
# Plan only: no provider calls and no cost
audit-raters score \
  --essays data/processed/pilot.parquet \
  --anchors data/processed/anchors.parquet \
  --models configs/local.models.yaml \
  --db data/interim/pilot.sqlite

# Execute one model/currency in its own database after reviewing its plan
audit-raters score \
  --essays data/processed/pilot.parquet \
  --anchors data/processed/anchors.parquet \
  --models configs/local.one_model.yaml \
  --db data/interim/one_model.sqlite \
  --repeats 2 --workers 4 --max-cost 2 --execute
```

After merging the per-model databases, the full plan has `50 × 6 models × 2 prompts × 2 rubrics × 2 runs = 2,400` calls. Repeating a command resumes non-API-error records. Never combine CNY and USD costs in one execution or cost cap; self-hosted GPU time needs a separate runtime budget. If budget is tight, follow the protocol's degradation order, starting with `--rubric-variants brief`; do not remove model families.

### 4. Analyze in the frozen order

```bash
audit-raters analyze \
  --scores data/interim/pilot.sqlite \
  --output out/pilot \
  --bootstrap 1000 --drf-boot 500
```

Use `--drf-boot 0` when ELL is unavailable. After installing the `bayes` extra, add `--bayes-g` to fit the NumPyro multivariate G posterior and write an EMS-versus-Bayes comparison. Main outputs are written to `out/pilot/tables/`, figures to `out/pilot/figures/`, and a compact summary to `summary.json`. Real output directories and SQLite files are ignored by default because they can contain essay-linked information and raw responses.

A one-model engineering pilot is supported without weakening the estimand definitions: the pipeline writes observed-score associations, configuration-level metrics, fixed-model configuration EMS, MFRM, compression, and failure-policy sensitivity, while marking `ρ_HL`, `π_pt`, cross-model G2, and multi-model D-study outputs as `not_estimable`. Observed Pearson/Spearman correlations from this branch are not substitutes for universe-score `ρ_HL`.

## Interpretation guardrails

- `ρ_HL` is the correlation between human and LLM universe-score constructs after removing within-type random-rater error. It is not the raw correlation between one human and one LLM.
- `π_pt` uses separate human and LLM residual estimates. Negative method-of-moments components are retained in `variance`; `variance_nonnegative` is explicitly separate.
- Human and LLM Φ coefficients answer different design questions. Report human Φ as the noise floor rather than interpreting the Φ difference as pure shared bias.
- Compression is evaluated relative to the human1→human2 baseline slope, not by directly interpreting an attenuated LLM→human slope.
- Parse failures may be informative. Compare `missing` and `floor` sensitivity analyses; never silently delete failures.
- Public-data memorization is not ruled out by high agreement. The contamination module compares pre-registered quality-preserving rewrites but does not automatically paraphrase essays.
- MFRM fits must be separated by original scale. The pipeline groups by prompt and rejects mixed scales within a fit.

## Repository layout

```text
src/audit_llm_raters/
  adapters/       source-table normalization and pilot sampling
  scoring/        providers, prompts, strict parser, resumable runner
  store/          append-only local SQLite score store
  sim/            shared-error simulation and recovery gates
  analysis/       QC, G/D studies, MFRM, compression, DRF, robustness
prompts/v1/       versioned templates and prompt-specific rubric overrides
configs/          non-secret model and dataset examples
scripts/          reproducibility helpers
tests/            unit and integration tests
data/             local-only raw/interim/processed data
out/              local-only generated figures and tables
```

See `docs/METHODS.md` for estimators, `docs/DATA_SOURCES.md` for ELLIPSE provenance and preparation, and `docs/REAL_PILOT_CHECKLIST.md` for the go/no-go checklist. All synthetic defaults use seed `20261107`; empirical reports should disclose every override.
