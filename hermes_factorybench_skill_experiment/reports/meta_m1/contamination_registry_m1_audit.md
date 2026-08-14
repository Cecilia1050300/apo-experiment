# M1 Contamination Registry Audit

- M1 SHA-256: `78187e3268294657d2398c9a79563a36f050c4189b2f6650cc569407512cb052`
- Base semantic registry: `37afcbbd97b3489aa8e32eaf78692c2fe3570a806c09b412f84e79d8976bbd19`
- M1 registry: `7c0d61e99b4a859151493b65eba7f16e316ee82ba033b49204fa64456ef6f9d0`

Only categories ACTUAL_MODEL_EVALUATION, ACTUAL_OPTIMIZER_EXPOSURE, ACTUAL_JUDGE_EXPOSURE, and EXECUTED_MANIFEST remain excluded. Item IDs and episode UUIDs are stored separately.

## Counts

- Excluded item IDs: 50
- Exposed episodes: 48
- New M0 exposure sources incorporated: 17

## Eligible source counts

- L1_test: 1228
- L2_validation: 3320
- L3_validation: 248
- L4_validation_free_form: 463

All M1 selected IDs and episodes are disjoint across development folds and holdout. L4 selection is restricted to free-form items so the existing deterministic diagnostic JSON validator and semantic judge contract remain applicable.
