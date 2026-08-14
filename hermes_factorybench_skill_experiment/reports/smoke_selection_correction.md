# FactoryBench Smoke Selection Audit Correction

## Scope

This correction changes candidate selection and audit reporting only. It made no model evaluation or API call and did not modify any candidate skill file or evaluation result JSON.

The evidence remains limited to the 15-item smoke optimization/development set. It does not establish validation, generalization, production readiness, Golden Skill status, or a 5–10 percentage-point improvement.

## Selection defect

The original selection ranked Skill v2 first using FactoryBench's reported score of `0.810126582278481`. That aggregate excluded item `8155904c-423f-4caa-95e8-b07102f065e7` because its score was non-finite.

The stored Skill v2 row has:

- raw output: empty string;
- parsed value: `null`;
- score: `NaN` in the original result JSON;
- parse error: `expected 4 T/F chars, got ''`;
- chance value: `0.5`.

The Skill v2 log contains no recorded timeout or connection warning. The exact cause of the empty response is unknown because the adapter did not retain per-call finish metadata. Completion-budget exhaustion remains only a hypothesis.

Excluding the failed row from both numerator and denominator produced the displayed Skill v2 L2 score of `0.8518518518518519`. This 14-finite-row aggregate is not comparable with complete 15-item runs and cannot be used to improve Skill v2's apparent standing.

## Corrected fixed-cardinality policy

Every candidate is assessed against the same 15 expected ordered item IDs.

A candidate is eligible for selection only when:

1. all 15 expected IDs are present;
2. all 15 item scores are finite;
3. parse failures equal zero.

For conservative scoring, each missing, parse-failed, or non-finite item receives raw score zero while retaining its original chance value and denominator contribution. Both the original FactoryBench score and fixed-cardinality conservative score are reported.

## Independently recomputed scores

| Run | Original FactoryBench score | Fixed-cardinality conservative score | IDs present | Finite scores | Parse failures | Eligible |
|---|---:|---:|---:|---:|---:|---|
| Fresh baseline | 0.7439024390243903 | 0.7439024390243903 | 15 | 15 | 0 | yes |
| Skill v1 | 0.7439024390243903 | 0.7439024390243903 | 15 | 15 | 0 | yes |
| Skill v2 | 0.810126582278481 | 0.7439024390243903 | 15 | 14 | 1 | **no** |
| Skill v3 | 0.7804878048780488 | 0.7804878048780488 | 15 | 15 | 0 | **yes** |
| Skill v4 | 0.6890243902439025 | 0.6890243902439024 | 15 | 15 | 0 | yes |

The final-digit difference between Skill v4's original and conservative display is floating-point representation only; no row substitution was required because all 15 Skill v4 scores are finite.

## Corrected selection

Skill v2 is ineligible. Among eligible skill candidates, Skill v3 has the highest fixed-cardinality conservative score.

Corrected source:

`hermes_factorybench_skill_experiment/prompts/generated/factorybench_skill_v3.txt`

Corrected smoke-best artifact:

`hermes_factorybench_skill_experiment/prompts/generated/factorybench_skill_smoke_best.txt`

SHA-256:

`cc83006d7a2e24aaf86ee7de16900cad71cd668563fb0ac2d8ef62b03ccb527e`

Size: 3557 bytes.

The corrected smoke-best file is byte-identical to Skill v3. The original Skill v2 selection remains in the machine-readable trace under `superseded_selection` and is marked `superseded_by_fixed_cardinality_audit_correction`.

## Outcome

Skill v3's fixed-cardinality improvement over baseline is `0.03658536585365857`. This is a directional smoke-set gain below `0.05`; the broader 5–10 percentage-point target was not achieved or claimed.
