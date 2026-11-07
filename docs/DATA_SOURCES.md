# Local empirical-data preparation

This document records source provenance and aggregate preparation diagnostics only. Essay text, individual ratings, processed samples, and model responses remain local and are excluded by `.gitignore`.

## ELLIPSE

Official source: [scrosseye/ELLIPSE-Corpus](https://github.com/scrosseye/ELLIPSE-Corpus), inspected at commit `710963bbf1acf23a28e40643d9f937d06f047dd4`. The repository README states that the data are provided under **CC BY 4.0**. The raw-rating archive's published password is `ellipse_raw_data`; this is an access password documented by the authors, not a credential.

Required local files:

- `ELLIPSE_Final_github_train.csv`: reliable-corpus training metadata and writing topics.
- `ellipsis_raw_rater_scores_anon_all_essay.zip`: two independent ratings per essay for Overall and six analytic dimensions.
- `ELL_Rubrics.docx`: official holistic and analytic rubrics.

SHA-256 values for the downloaded source files:

| File | SHA-256 |
| --- | --- |
| `ELLIPSE_Final_github_train.csv` | `782344e99668a3ff508d7410c0eb6e36da70f3b28f81c96e367f1ca04924b06c` |
| `ellipsis_raw_rater_scores_anon_all_essay.zip` | `f9c3cede2d54144225e0651af6415137f7da07f2d2955e8f8587896c293d5e85` |
| `ELL_Rubrics.docx` | `0c0ac5dfa6ae89c9c99ffe483c13ea4a57e8152a1472797e1fe361c2c203efdc` |

The preparation command joins metadata to the raw ratings by `text_id_kaggle`, retains `Overall_1` and `Overall_2` as the two human scores, and retains `Rater_1` and `Rater_2` as cross-essay person IDs. Published average scores never replace the independent raw ratings. The common `prompt_id` is `overall` because all writing topics use the same holistic language-proficiency rubric; the original writing topic remains in `demographics_json`.

Zero is outside the official 1–5 rubric and is treated as an invalid raw score, not as the lowest valid score. Records with either rater's identifying-information flag are excluded by default as a privacy precaution. Use `--keep-identifying-info` only after a separate privacy review.

Aggregate diagnostics from the executed local preparation:

| Diagnostic | Count/value |
| --- | ---: |
| Raw-rating rows | 8,890 |
| Raw rows without a reliable-corpus ID | 2,408 |
| Training-metadata rows | 3,911 |
| Metadata rows joined to raw ratings | 3,897 |
| Unmatched metadata rows | 14 |
| Matched rows containing a zero/out-of-range score | 6 |
| Matched rows with an identifying-information flag | 85 |
| Eligible double-scored essays after exclusions | 3,806 |
| Unique human raters in the eligible pool | 27 |
| Cross-essay rater IDs | Yes |
| Eligible raw-score range | 1–5 |

The eligible pool also contains 22 essays whose published Overall average differs from the mean of the two raw Overall ratings. This discrepancy is retained in the audit rather than silently repaired: the study uses the independent raw scores required by G1 and does not treat the published average as a substitute.

All ELLIPSE writers are English language learners. This source therefore supports the mechanism and halo-effect analyses but cannot estimate an ELL-versus-non-ELL contrast; RQ3 still requires ASAP 2.0 or PERSUADE.

## Reproduce the local ELLIPSE pilot

From the repository root, after downloading and extracting the official files:

```bash
audit-raters prepare-ellipse \
  --raw-ratings data/raw/ellipse/raw_raters/ellipsis_raw_rater_scores_anon_all_essay.csv \
  --metadata data/raw/ellipse/ELLIPSE_Final_github_train.csv \
  --output data/interim/ellipse_overall.csv \
  --audit-output data/interim/ellipse_preparation_audit.json

audit-raters prepare \
  --input data/interim/ellipse_overall.csv \
  --dataset-config configs/datasets/ellipse.yaml \
  --n 50 --anchor-pool 100 \
  --output data/processed/ellipse_pilot50.parquet \
  --anchor-output data/processed/ellipse_anchors100.parquet
```

The executed local draw contains 50 pilot essays and 100 disjoint held-out anchors. Every pilot essay has two independent scores. The pilot covers all nine possible half-point means from 1.0 through 5.0, 28 original writing topics, and 23 human raters. These are sampling diagnostics, not empirical LLM results.

Local brief and detailed holistic rubrics are derived from `ELL_Rubrics.docx` and stored as `prompts/v1/rubrics/overall_{brief,detailed}.txt`. They are intentionally git-ignored with the other official prompt materials. The no-provider planning run expanded the full design to 2,400 tasks and confirmed that no generic test rubric was used. No paid provider call has been made.

## ASAP 2.0 audit

Official source: [scrosseye/ASAP_2.0](https://github.com/scrosseye/ASAP_2.0), inspected at commit `060ab225ca0984c0bf43ff222bf6bb2bc6195b6f`. The repository README states that the data are provided under **CC BY 4.0**. The training archive, seven source-text PDFs, and official 1–6 holistic rubric were downloaded locally.

| File | SHA-256 |
| --- | --- |
| `ASAP_2_Final_github_train.zip` | `aec01e98eb16cc3bc3828010ca053a09daecc2821dd2105f8a7333089271beb4` |
| `ASAP_2_source_texts.zip` | `8655d7df4f78871cba6b8afc83a6a9df796be89f86d93c54f10dd61dd1d65d71` |
| `asap_scoring_rubric.docx` | `578edb71fd66c5f51e99b8d8bc24b038fcf819d91ad896d118b3736711c46403` |

The training table contains 17,307 unique essays across seven source-based prompts. Every row has one holistic `score` on the 1–6 scale, but there are no independent rater-1/rater-2 columns or human rater IDs. It therefore does **not** independently satisfy G1 and must not be converted into a pseudo-double-scored dataset by copying `score` into two columns.

The table does contain the variables needed for the proposal's optional focused DRF analysis: 2,269 essays are marked `ell_status = Yes`, 14,798 are marked `No`, and 240 have missing ELL status. All seven prompts have both ELL and non-ELL essays, although their ELL counts are unequal. Eighteen rows lack the assignment text. The official source PDFs and rubric must be included in any subsequent prompt-specific scoring inputs.

Current decision: ELLIPSE is the prepared G1-passing pilot source; ASAP 2.0 remains a candidate for the separate RQ3/DRF extension using its single human score. No primary-estimand or study-design change is implied by this decision.
