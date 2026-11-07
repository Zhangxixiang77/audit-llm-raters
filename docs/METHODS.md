# Statistical methods and implementation notes

## Estimands

For rater types H and L, let each type's essay-level mean be its observed universe-score estimate. The implementation estimates within-type person variance with balanced p×r EMS and cross-type covariance with the sample covariance of essay means:

`ρ_HL = cov(H_bar, L_bar) / sqrt(var_p(H) var_p(L))`.

Because the cross-type means share essays but not random-rater residuals, their covariance estimates the universe-score covariance without a separate raw-score attenuation correction.

For p×(r:t), `var_p = cov(H_bar,L_bar)` and `var_pt = (var_p(H)+var_p(L))/2-var_p`. Human and LLM residual variances are estimated independently and averaged after dividing by their respective numbers of raters. `π_pt = var_pt / (var_p + var_pt + residual_mean)`. Essay bootstrap intervals preserve the paired human/LLM design.

The optional NumPyro model jointly estimates the two universe-score variances and their covariance, separate human/LLM residual standard deviations, and rater severities. `ρ_HL` and `π_pt` are deterministic posterior quantities; the analysis writes their posterior intervals and an EMS comparison when `--bayes-g` is enabled.

## Balanced EMS

`balanced_ems` accepts an arbitrary fully crossed array with one observation per full cell. It constructs every main effect and interaction by inclusion–exclusion of marginal means, forms balanced ANOVA mean squares, and solves the triangular expected-mean-square system. The full interaction is the residual. Negative estimates are not silently truncated. The full p×model×prompt×rubric×run analysis is only attempted when every cell is observed.

For p×r D studies:

- relative error: `var_pr / n_r`;
- absolute error: `(var_pr + var_r) / n_r`;
- `Eρ² = var_p / (var_p + relative error)`;
- `Φ = var_p / (var_p + absolute error)`.

The ensemble–human curve uses `var(L_ensemble)=var_p(L)+var_pr(L)/k`; its limit is `ρ_HL`, so additional models cannot remove shared universe-score disagreement.

## MFRM

For categories `0…K`, the adjacent-category partial-credit form is

`logit_k = Σ_{j=1…k} (θ_p - severity_r - difficulty_i - τ_j - γ_r ELL_p)`.

Severity, item difficulty, and thresholds use sum-to-zero parameterizations. The MAP objective uses weak normal regularizers (`θ~N(0,2²)`, other effects `N(0,1²)`) and an analytic gradient. `fit_mfrm_numpyro` exposes the corresponding NUTS model when the optional dependencies are installed.

Infit is squared residual sum divided by predicted variance sum; outfit is the mean squared standardized residual. Category proportions are reported by rater. If human person IDs cannot link across essays, each human position becomes an essay-nested rater; individual human severities are then not interpreted as crossed-rater effects.

DRF is `γ_r`, with positive values indicating greater severity for ELL essays under the model's sign convention. Only LLM raters enter the bootstrap DRF analysis. Essays are resampled with replacement as clusters; percentile intervals and two-sided sign-tail p-values receive Bonferroni adjustment across raters. Absolute effects of at least 0.5 logit are separately flagged as substantively important.

## Compression

For each model and prompt, the baseline slope is human1 regressed on human2 and the LLM slope is the model ensemble regressed on human2. Their difference is excess compression relative to the human attenuation baseline. The implied target from `LLM=a+b Human` is `a/(1-b)`, normalized by each prompt's original scale for cross-prompt comparisons with midpoint and human-mode targets.

## Robustness and limitations

- `missing`: invalid/non-integer/out-of-range/refusal responses are excluded before complete-case matrices are built.
- `floor`: every non-OK response is assigned the original scale minimum.
- Balanced subsamples repeat the primary estimates on a fixed essay count.
- Contamination analysis compares original and externally validated quality-preserving rewrites using paired absolute errors and Wilcoxon tests.

Simulation checks assess parameter recovery under the stated generating model; they do not demonstrate shared error in real models. Small pilot samples, negative variance estimates, crossed-rater assumptions, public-data contamination, and model-version drift remain empirical limitations.
