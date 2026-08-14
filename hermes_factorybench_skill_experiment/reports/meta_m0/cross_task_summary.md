# Meta-Prompt M0 Cross-Task Smoke

M0 remains a candidate and is not Golden.

## Frozen artifacts

- M0: `e4bef66552518e4a1206aaa3b14b9d34335c046499291e70dfdc33a79f243257` (3012 bytes)
- Core v0: `463692cd0a201d916c3f0e39d10cda4c50d2a9a1ca3305cd9de804c295e482b3` (773 bytes)

## Data availability

| Task | Status | Development | Holdout | Manifest SHA-256 |
|---|---|---:|---:|---|
| factorybench_l123 | INSUFFICIENT_UNSEEN_DATA | 0 | 0 | `8d7a7d7b0b77ba2d946d4f5d3a448e55a574b7be935b63da1877d7690a58dc82` |
| factorybench_l4 | AVAILABLE | 2 | 3 | `08f0c809a3ff1df2511bcd409c084c099b7135510114457a24f54fae0fa88010` |
| causal_judgment | AVAILABLE | 3 | 5 | `5e85686501d8150ffa934eb51c82929d3cb35bfb66dd49949b03572fd8f759d5` |

## Development and selection

| Task | Baseline | Core-only | Adapter v1 | Adapter v2 | Selection |
|---|---:|---:|---:|---:|---|
| factorybench_l123 | n/a | n/a | n/a | n/a | INSUFFICIENT_UNSEEN_DATA |
| factorybench_l4 | -0.090909 | 0.454545 | None | None | NO_ADAPTER (core_only) |
| causal_judgment | 0.000000 | 0.000000 | 1.0 | 1.0 | ADAPTER (adapter_v2) |

## Holdout

| Task | Baseline | Core-only | Selected adapter | Selection |
|---|---:|---:|---:|---|
| factorybench_l123 | n/a | n/a | n/a | not run |
| factorybench_l4 | 0.200000 | -0.200000 | None | NO_ADAPTER |
| causal_judgment | 0.000000 | 0.000000 | 0.8 | ADAPTER |

## Verdicts

- Core Skill: **CORE_NEGATIVE** — Core regresses without compensating evidence on completed tasks.
- Task Adapter necessity: **NO_ADAPTER_EVIDENCE**
- Meta-Prompt: **M0_INVALID**

## Integrity and limitations

- No holdout result was sent to M0 or used for refinement.
- No M0-generated adapter was manually edited.
- Skill v3 and all Task A artifacts remain unchanged.
- This smoke does not establish statistical significance.

Total recorded API calls: 43; cost: $2.804425; summed condition wall time: 308.108 seconds.

Recommendation: do not promote M0 to Golden; resolve prospective L1-L3 data availability and repeat a complete cross-task holdout.
