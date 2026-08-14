# Manufacturing Meta-Prompt M1 Two-Task Frozen Smoke

- M1 SHA-256: `78187e3268294657d2398c9a79563a36f050c4189b2f6650cc569407512cb052`
- M1 bytes: 3588
- Core used: no

## Development

| Task | Baseline A | Baseline B | Adapter v1 A/B | Adapter v2 A/B | Selection |
|---|---:|---:|---|---|---|
| m1_factorybench_l123 | 0.5555555555555555 | 0.6 | n/a | n/a | NO_ADAPTER (baseline) |
| m1_factorybench_l4 | 0.5 | 0.0 | 0.0 / 0.5 | n/a | NO_ADAPTER (baseline) |

## Holdout

| Task | Baseline | Selected | Delta | Parse failures | Verdict |
|---|---:|---:|---:|---:|---|
| m1_factorybench_l123 | 0.5849056603773585 | NO_ADAPTER | 0.0 | 1 | TASK_NO_EFFECT |
| m1_factorybench_l4 | 0.0 | NO_ADAPTER | 0.0 | 0 | TASK_NO_EFFECT |

## Manufacturing verdict: **M1_MANUFACTURING_NO_EFFECT**

Calls including failed attempt: 50; known cost: $6.232495; summed known wall time: 659.584s.

No Core prompt was used. No holdout data entered M1 generation or refinement. M1 remains a smoke candidate, not Golden.
