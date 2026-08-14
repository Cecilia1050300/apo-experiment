# Meta-Prompt M0 Cross-Task Smoke — Completed Data-Availability Continuation

This report does not replace or alter the original incomplete report. It reuses frozen L4 and Causal results and adds only the recovered FactoryBench L1–L3 branch.

## Contamination correction

- Original registry: `b516d21adb2c7e3d2273325a5467b36d1984feb5da647d1b1ccb537d27ab9210`
- Corrected registry: `37afcbbd97b3489aa8e32eaf78692c2fe3570a806c09b412f84e79d8976bbd19`
- Audit report: `6c3d91b99ec3fc5c4d9496c18e9c505cb734ac8773c8fa8e8d51e514f08b337f`

## FactoryBench L1–L3 development

| Condition | Fixed-cardinality score | Parse failures |
|---|---:|---:|
| Baseline | 0.05555555555555555 | 0 |
| Core-only | 0.05555555555555555 | 0 |
| Adapter v1 | 0.3888888888888889 | 0 |
| Adapter v2 | 0.3888888888888889 | 0 |

Selection: **ADAPTER** (adapter_v2).

## Cross-task holdout

| Task | Baseline | Core-only | Selected adapter | Selection |
|---|---:|---:|---:|---|
| factorybench_l123 | 0.4 | 0.4 | 0.4 | ADAPTER |
| factorybench_l4 | 0.19999999999999998 | -0.19999999999999998 | None | NO_ADAPTER |
| causal_judgment | 0.0 | 0.0 | 0.8 | ADAPTER |

FactoryBench L1 uses test; L2/L3 use validation. Its aggregate is a mixed-split smoke score, not a pure validation score.

## Verdicts

- Core Skill: **CORE_NEGATIVE**
- Adapter necessity: **NO_ADAPTER_EVIDENCE**
- Meta-Prompt M0: **M0_INVALID** — Completed protocol contains 10 parse failures in frozen baseline/Core/selected conditions.

## Usage

- Additional calls: 32
- Additional cost: $2.668265
- Additional summed wall time: 565.317 seconds

No holdout result was sent to M0. No adapter was manually edited. L4 and Causal models were not called again.
