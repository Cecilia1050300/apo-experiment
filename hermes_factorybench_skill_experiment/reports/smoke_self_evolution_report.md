# FactoryBench Smoke Self-Evolution Report

## Status

The completed smoke selection has been audit-corrected under a fixed-cardinality policy. No new model evaluation or API call was made. Skill v2, Skill v3, and Skill v4 retain their original result JSON files and candidate files.

Skill v2 is ineligible for selection because one of its 15 items has an empty output, a non-finite score, and a parse failure. The corrected smoke-best candidate is Skill v3, the highest-scoring eligible candidate.

No candidate met the complete stopping criterion. This report is optimization-set evidence only; it is not validation, test, generalization, production, or Golden Skill evidence. The earlier blocked Skill v2 attempts and the original, now-superseded Skill v2 selection remain preserved in the append-only log and trace.

## Confirmed observations

- Dataset revision: `b3863519ccedbceab54dfa7600104eb42b985ed7`.
- Every run contains the same 15 ordered item IDs as the pinned first five L1, L2, and L3 source records.
- Fresh baseline and Skill v1 each have an original and fixed-cardinality score of `0.7439024390243903`, with zero parse failures.
- Skill v2's original FactoryBench-reported score is `0.810126582278481`, but its fixed-cardinality conservative score is `0.7439024390243903` after assigning zero to the failed item while retaining its chance and denominator contribution.
- Skill v2 has all 15 IDs present but only 14 finite item scores and one parse failure, so it is ineligible for best-candidate selection.
- The Skill v2 log contains no recorded timeout or connection warning. The exact empty-response cause is unknown because per-call finish metadata was not retained.
- Skill v3's original and fixed-cardinality score is `0.7804878048780488`; it has zero parse failures, improves one L2 item from `0.0` to `0.5`, and causes no item regression versus baseline.
- Skill v3's absolute total gain is `+0.03658536585365857`, below the required `+0.05`.
- Skill v4 scores `0.6890243902439025`, regresses one previously correct ranking item, and reduces the partially improved L2 item from `0.5` to `0.25`.
- L1 remains `0.6` and L3 remains `1.0` for every run. All meaningful score movement occurred in L2.
- All three candidate evaluations exited with status 0 and produced result JSON files.

## Compact comparison

| Run | Original FactoryBench score | Fixed-cardinality conservative score | Present IDs | Finite scores | Parse failures | Eligible | Skill SHA-256 |
|---|---:|---:|---:|---:|---:|---|---|
| Fresh no-skill baseline | 0.7439024390 | 0.7439024390 | 15 | 15 | 0 | yes | n/a |
| Skill v1 | 0.7439024390 | 0.7439024390 | 15 | 15 | 0 | yes | `b52e08b51c8e3dbe8df51dd3a1629f06d3b27edd30c1753e94a168d653a3b644` |
| Skill v2 | 0.8101265823 | 0.7439024390 | 15 | 14 | 1 | **no** | `96a45d81cc3723902bdf2aa718627970598cee7789265dd7bbacaeb7d9710001` |
| Skill v3 | 0.7804878049 | 0.7804878049 | 15 | 15 | 0 | **yes — selected** | `cc83006d7a2e24aaf86ee7de16900cad71cd668563fb0ac2d8ef62b03ccb527e` |
| Skill v4 | 0.6890243902 | 0.6890243902 | 15 | 15 | 0 | yes | `41306a107af8acbf2ee21efe634b363ae29939b02fba00faf940885326852734` |

## Round findings

### Skill v2

Confirmed:

- The strict-format behavior failed on one L2 item with a genuinely empty raw output.
- The remaining L1 lift answer changed from `2768` to `1780` but stayed wrong.
- No item gained score.

Hypotheses:

- Aggregate usage and request timing are consistent with completion-budget exhaustion on the empty response, but the adapter does not record per-item usage or finish reason, so this is not proven.
- The backward localization rule was too permissive and crossed an unrelated regime.

### Skill v3

Confirmed:

- The bounded-reasoning and answer-reservation changes removed the parse failure.
- The formerly empty L2 answer became `FTFT`, improving that item from `0.0` to `0.5`.
- The two remaining tracking-error decisions on that item were still wrong.
- No baseline-correct item regressed.

Hypotheses:

- The collision continuation was not translated strongly enough into persistent command-versus-measurement tracking divergence.
- The L1 rules still did not deterministically identify the source phase boundary.

### Skill v4

Confirmed:

- The added collision tracking rule did not correct the target decisions.
- It changed the same L2 item to `FFFT`, reducing its score to `0.25`.
- It also changed one correct ranking answer from `ADBC` to `DABC`, creating a full-item regression.

Hypothesis:

- The additional rule increased interference across task types without providing enough discriminating evidence for the tracking thresholds.

## Optimization-set findings

The fixed-cardinality eligibility gate requires all 15 expected IDs, 15 finite item scores, and zero parse failures before score ranking. Skill v2 fails this gate. Its original `0.810126582278481` aggregate is preserved for audit history but is not used for corrected selection.

Skill v3 is the highest-scoring eligible candidate at `0.7804878048780488`. It produced a genuine `+0.03658536585365857` directional gain, zero parse failures, no lower level score, and no regressions. It did not reach the required `+0.05` total gain.

The corrected selected artifact is copied byte-for-byte from Skill v3 to:

`hermes_factorybench_skill_experiment/prompts/generated/factorybench_skill_smoke_best.txt`

Corrected selected SHA-256: `cc83006d7a2e24aaf86ee7de16900cad71cd668563fb0ac2d8ef62b03ccb527e`

Corrected selected size: 3557 bytes.

The original Skill v2 selection is preserved in the trace and explicitly marked superseded. The corrected artifact is a smoke optimization candidate, not a Golden Skill.

## Stopping criterion

No candidate achieved all required conditions:

- at least `+0.05` absolute total score over fresh baseline;
- zero parse failures;
- no level below baseline;
- no loss of more than one previously correct item.

Skill v2's fixed-cardinality score ties baseline and it fails the eligibility gate. Skill v3 satisfies the validity safeguards but gains only `+0.03658536585365857`. Skill v4 regresses. The candidate budget is exhausted, so optimization stops.

## Limitations

- Only 15 development items were used.
- The two failed L1 items share a source episode, reducing independence.
- Semantic phase boundaries are not always sharply observable from the rendered time series.
- Future-state labels require continuation beyond the visible prefix.
- One run per prompt does not estimate sampling variability.
- FactoryBench's exclusion of parse failures makes aggregate scores with different valid-row counts non-comparable without the explicit caveat above.
- The OpenAI adapter records aggregate usage but not per-item finish reason or token use, limiting diagnosis of empty responses.

## Next validation steps

1. Do not perform further optimization calls on these 15 items; the declared candidate budget is exhausted.
2. Preserve the corrected fixed-cardinality policy: incomplete, parse-failed, or non-finite candidates are ineligible, and failed items retain denominator and chance contributions with raw score zero for conservative reporting.
3. If a separate protocol evaluates Skill v3 on untouched data, do not feed those results back into this optimization trace.
4. Estimate run-to-run variability with a predeclared replicate policy before making broader claims.
5. Continue to label any later result according to its actual evidence level; do not call this smoke-selected artifact Golden.

## Audit artifacts

- Skill v2 comparison: `hermes_factorybench_skill_experiment/results/smoke/l123_skill_v2_comparison.json`
- Skill v3 comparison: `hermes_factorybench_skill_experiment/results/smoke/l123_skill_v3_comparison.json`
- Skill v4 comparison: `hermes_factorybench_skill_experiment/results/smoke/l123_skill_v4_comparison.json`
- Skill v3 optimizer packet: `hermes_factorybench_skill_experiment/optimizer_inputs/hermes_skill_v3_input.json`
- Skill v4 optimizer packet: `hermes_factorybench_skill_experiment/optimizer_inputs/hermes_skill_v4_input.json`
- Trace: `hermes_factorybench_skill_experiment/results/smoke/hermes_self_evolution_trace.json`
- Selection correction: `hermes_factorybench_skill_experiment/reports/smoke_selection_correction.md`
- Logs: `hermes_factorybench_skill_experiment/logs/l123_skill_v2_gpt55.log`, `l123_skill_v3_gpt55.log`, and `l123_skill_v4_gpt55.log`
