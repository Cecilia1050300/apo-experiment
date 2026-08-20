# M1 L1-L3 Read-Only Audit

## Scope and conclusion

This audit resolves the FactoryBench L1-L3 branch of the prior M1 experiment without modifying it. FactoryBench Level 4 is excluded from the new experiment.

The decisive finding is that M1 L1-L3 did **not** use a Core prompt or an initial shared Adapter. Its target baseline called the model with `system=None`. M1 then declined to generate an Adapter in round 1 and reported insufficient refinement evidence in round 2. Therefore the byte-faithful v0 representation for this experiment is an empty Skill file (SHA-256 of zero bytes: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`).

## Resolved M1 implementation

Repository root: `/home/training/automatic_prompt_engineer`

| Purpose | Resolved path | SHA-256 |
|---|---|---|
| M1 runner | `hermes_factorybench_skill_experiment/scripts/meta_m1/run_m1.py` | `aca8337e54fc7d32c1f6c760f7f5aac7788d54a7871692573467d568b9499cd5` |
| M1 manifest/registry builder | `hermes_factorybench_skill_experiment/scripts/meta_m1/prepare_m1.py` | `4b714d200277ed811332ea1680347a847dad03722f363e17476eb6aeb9f54618` |
| M1 optimizer prompt | `hermes_factorybench_skill_experiment/prompts/meta/manufacturing_meta_prompt_m1_candidate.txt` | `78187e3268294657d2398c9a79563a36f050c4189b2f6650cc569407512cb052` |
| L123 combined manifest | `hermes_factorybench_skill_experiment/data_manifests/meta_m1/factorybench_l123_combined_manifest.json` | `187d208c6a432acffcbfd497c2af9051d4fd0256092597a89e36173449df8dd1` |
| M1 contamination registry | `hermes_factorybench_skill_experiment/data_manifests/meta_m1/contamination_registry_m1.json` | `7c0d61e99b4a859151493b65eba7f16e316ee82ba033b49204fa64456ef6f9d0` |
| L123 M1 selection | `hermes_factorybench_skill_experiment/prompts/adapters/m1_factorybench_l123/selection.json` | `02196562daf3e0bceff6f4e2f185e6f0ef9bae9f8fff40ad223a25453d07fd6f` |
| Development summary | `hermes_factorybench_skill_experiment/results/meta_m1/development/m1_factorybench_l123_summary.json` | `2f41e6f0c9bf0c4f63638397ccb98e527c9321098c0d177d622f9353c60ab960` |
| Holdout summary | `hermes_factorybench_skill_experiment/results/meta_m1/holdout/m1_factorybench_l123_summary.json` | `41902320c81843b9f7bfc17a51a378ce3f7161c4d67eea4da4f773f9c07cbc07` |

## Frozen manifests

The new manifest symlinks resolve to these M1 files:

| New path | M1 source | Count | SHA-256 |
|---|---|---:|---|
| `meta_coevoskills_experiment/manifests/dev_fold_a.json` | `hermes_factorybench_skill_experiment/data_manifests/meta_m1/factorybench_l123_dev_fold_a.json` | 3 | `fa34802b22cec32df90e59ac3db64b952c35706ef998d70f013d6298affbaec1` |
| `meta_coevoskills_experiment/manifests/dev_fold_b.json` | `hermes_factorybench_skill_experiment/data_manifests/meta_m1/factorybench_l123_dev_fold_b.json` | 3 | `95cec7a9285b7c079c8f6e243fcdfca23990c79e3c0606f606250c19e05e327c` |
| `meta_coevoskills_experiment/manifests/holdout.json` | `hermes_factorybench_skill_experiment/data_manifests/meta_m1/factorybench_l123_holdout.json` | 9 | `c1a6a29ad240a2585f2dcda7d3332924f43085a95d510dab4b9a2980cead2082` |

`prepare_m1.py` selected one item per level for each development fold and three per level for holdout. It used L1 `test` and L2/L3 `validation`. All selected IDs and provenance episodes were made disjoint across both folds and holdout. This is a **FactoryBench L1-L3 mixed-split evaluation**, not a pure validation score.

## Starting Skill resolution

Evidence that M1 used no starting Core/shared Skill:

1. `run_m1.py:evaluate_l123` sets `system=adapter if adapter else None`; baseline calls pass `adapter=None`.
2. `run_m1.py:m1_input` records `constraints.core_prompt=false`.
3. `m1_round_1_input.json` records `core_prompt: false` and `previous_adapter: null`.
4. `manufacturing_cross_task_summary.md` states `Core used: no` and `No Core prompt was used`.
5. `selection.json` records `NO_ADAPTER`, `selected_candidate: baseline`, and `selected_adapter_sha256: null`.

There is consequently no non-empty M1 Skill file or Skill hash to copy. The experiment represents the exact no-system-prompt baseline as two zero-byte files:

- `skills/control/skill_v0.txt`
- `skills/surrogate/skill_v0.txt`

Both hash to `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`; preflight fails loudly if they differ.

## Rendering, parsing, and scoring

M1 imports the pinned FactoryBench package and does not implement a separate L1-L3 parser.

| Component | Path | SHA-256 | Behavior |
|---|---|---|---|
| Item renderer | `FactoryBench/FactoryBench/factorybench/prompt.py` | `71832ea6f68019898b6405eafb14c90c0c669e06ed610d7956d9a7113d109b19` | Concatenates time-series description, acronym mapping, rows, question, then options. |
| Per-item evaluator | `FactoryBench/FactoryBench/factorybench/evaluate.py` | `49f18ba39301a5143dc59c47aeab27324d5dceff90b2efc9db2094fc2ce565ac` | `_score_one` uses deterministic parsing/scoring for L1-L3. |
| Parser | `FactoryBench/FactoryBench/factorybench/parse.py` | `204c5417caca5f48bd6a111f66f55fb68d05d87a0bce0328236fc79873b93ad8` | Parses MCQ letters, four-letter T/F, four-letter rankings, scalar formats, and tensors. |
| Scorer/chance rates | `FactoryBench/FactoryBench/factorybench/score.py` | `aa12643aab5ee33fb027873f4f25e404123c8f96f768ef1bd913419d095d7887` | Exact/range/margin or elementwise scoring; chance is 1/N for MCQ, 0.5 for T/F, 1/24 for ranking, 0 for scalar/tensor. |
| Canonical Result aggregation | `FactoryBench/FactoryBench/factorybench/result.py` | `7ecf78ff1b0059e30b41f9661ab465cb2220bc9cbeef051d4e01298c614fda7b` | Chance-corrected mean over clean, finite rows. |

M1's runner duplicates the aggregate formulas in small local helpers:

- `canonical(rows)` excludes parse-failed/non-finite rows, then computes `(mean_score - mean_chance) / (1 - mean_chance)`.
- `fixed(rows)` keeps expected cardinality and substitutes raw score zero for parse-failed/non-finite rows while retaining each row's chance.
- `grouped(rows, field)` applies fixed-cardinality scoring by level, answer format, and dataset.

The new harness reuses `render_prompt` and `_score_one` directly and preserves these M1 canonical/fixed formulas.

## Stable API and concurrency path

M1 used `OpenAI().chat.completions.create` for target inference with ordered `ThreadPoolExecutor` execution at concurrency 2. It checkpointed aggregate result artifacts but not every individual target call. The new experiment retains Chat Completions for `gpt-4o-mini`, uses conservative configurable concurrency 1 initially, and adds an atomic checkpoint per paid call.

The new verifier and rewriter use independent `responses.create` requests with strict JSON schemas. No prior response/conversation identifier is supplied, so every verifier request is context-isolated.

## Contamination/exposure policy

`prepare_m1.py` built `contamination_registry_m1.json` from the corrected M0 semantic registry. Its recorded rule is that only actual model, optimizer, judge, or executed-manifest exposure counts; inventory/prospective UUIDs and provenance episode UUIDs are not automatically item contamination. It excluded exposed item IDs and their provenance episodes before deterministic SHA-256 seeded selection. The combined L123 manifest records all 15 IDs and all 15 episodes as disjoint.

## What M1 attempted and why it selected NO_ADAPTER

Recorded development evidence:

- Fold A fixed-cardinality score: `0.5555555555555555`; one L1 scalar-margin miss, with L2 and L3 correct.
- Fold B canonical score: `1.0`, but fixed-cardinality score: `0.6`; one L3 empty-output parse failure, with L1 and L2 correct.
- Round 1 M1 output (`m1_round_1_parsed_output.json`, SHA-256 `499f4ae3e0694a1f7951c6fd969e99beb9dc4ec2170c7b0defba27ffcdfc0d53`) chose `NO_ADAPTER`. It classified the numeric miss and strict-format failure as isolated, non-recurring evidence and identified regression risk from broad rules.
- Round 2 output (`m1_round_2_parsed_output.json`, SHA-256 `a0d282e433b7b0aef2a508d5f13723e5c91a794b6e6afac27068189224fa44d5`) chose `INSUFFICIENT_DATA` because no Adapter v1 or Adapter-v1 fold results existed to refine.
- `selection.json` therefore had no candidates and selected baseline/`NO_ADAPTER`.

The conclusion above is based only on frozen traces and results. It does not infer an unrecorded reason.

A prior failed L123 attempt is also explicitly preserved in `m1_factorybench_l123_failed_attempt_1.json`: three candidate calls completed, then strict JSON serialization rejected an in-memory NaN. The corrected M1 runner normalized non-finite scores before serialization. The new harness preserves that correction.
