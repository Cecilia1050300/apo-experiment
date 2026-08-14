# FactoryBench L1–L3 Contamination Registry Audit

- Original registry preserved: `data_manifests/meta_m0/contamination_registry.json`
- Original SHA-256: `b516d21adb2c7e3d2273325a5467b36d1984feb5da647d1b1ccb537d27ab9210`
- Corrected registry: `data_manifests/meta_m0/contamination_registry_v2.json`

## Root cause

The original lexical UUID scan treated every UUID in broad inventories as an exposed item ID and did not distinguish provenance episode UUIDs from item UUIDs.

The broad `l123_all_unseen_generalization.json` file was an inventory, not an executed result. Its 7,572 UUID tokens included both prospective item IDs and UUID-valued provenance episodes. Treating all of them as actual exposure eliminated the required L1 test pool and substantially overexcluded the L2/L3 validation pools.

## Source classifications

| Source artifact | Classification | UUID tokens | Resolved pinned items | Remain excluded | Evidence |
|---|---|---:|---:|---|---|
| `apo_experiment/factorybench_experiment/data/factorybench_l4_subset.json` | SOURCE_DATASET_ONLY | 15 | 0 | no | Prepared source/subset file; model and judge execution are established only by separate result artifacts. |
| `apo_experiment/factorybench_experiment/data/factorybench_l4_subset_with_summary.json` | SOURCE_DATASET_ONLY | 15 | 0 | no | Prepared source/subset file; model and judge execution are established only by separate result artifacts. |
| `apo_experiment/factorybench_experiment/data/factorybench_l4_subset_with_temporal_features.json` | SOURCE_DATASET_ONLY | 15 | 0 | no | Prepared source/subset file; model and judge execution are established only by separate result artifacts. |
| `apo_experiment/factorybench_experiment/results/evaluator_smoke_test_gpt-5.5.json` | ACTUAL_MODEL_EVALUATION | 1 | 0 | yes | Result artifact contains candidate-model outputs, usage, scores, or a completed model-run status. |
| `apo_experiment/factorybench_experiment/results/failure_audit_source.json` | EXECUTED_MANIFEST | 4 | 0 | yes | Derivative result contains IDs and outputs/failures tied to a demonstrably executed model run. |
| `apo_experiment/factorybench_experiment/results/final_test_skill_comparison_gpt-4o-mini.json` | ACTUAL_MODEL_EVALUATION | 8 | 0 | yes | Result artifact contains candidate-model outputs, usage, scores, or a completed model-run status. |
| `apo_experiment/factorybench_experiment/results/final_test_skill_comparison_gpt-4o-mini_judged_gpt-5.5_x1.json` | ACTUAL_JUDGE_EXPOSURE | 8 | 0 | yes | Judged-result artifact contains judge records or scores for these IDs. |
| `apo_experiment/factorybench_experiment/results/gpt55_neutral_context_audit.json` | INVENTORY_ONLY | 5 | 0 | no | Deterministic context/contract audit only; no candidate-model or judge call is recorded in this artifact. |
| `apo_experiment/factorybench_experiment/results/gpt55_neutral_context_control.json` | ACTUAL_MODEL_EVALUATION | 8 | 0 | yes | Result artifact contains candidate-model outputs, usage, scores, or a completed model-run status. |
| `apo_experiment/factorybench_experiment/results/gpt55_neutral_context_control_judged_gpt-5.5.json` | ACTUAL_JUDGE_EXPOSURE | 8 | 0 | yes | Judged-result artifact contains judge records or scores for these IDs. |
| `apo_experiment/factorybench_experiment/results/json_contract_v2/contract_audit.json` | INVENTORY_ONLY | 5 | 0 | no | Deterministic context/contract audit only; no candidate-model or judge call is recorded in this artifact. |
| `apo_experiment/factorybench_experiment/results/json_contract_v2/gpt55_generation.json` | ACTUAL_MODEL_EVALUATION | 8 | 0 | yes | Result artifact contains candidate-model outputs, usage, scores, or a completed model-run status. |
| `apo_experiment/factorybench_experiment/results/json_contract_v2/gpt55_judged.json` | ACTUAL_JUDGE_EXPOSURE | 8 | 0 | yes | Judged-result artifact contains judge records or scores for these IDs. |
| `apo_experiment/factorybench_experiment/results/json_contract_v2/mini_generation.json` | ACTUAL_MODEL_EVALUATION | 8 | 0 | yes | Result artifact contains candidate-model outputs, usage, scores, or a completed model-run status. |
| `apo_experiment/factorybench_experiment/results/json_contract_v2/mini_judged.json` | ACTUAL_JUDGE_EXPOSURE | 8 | 0 | yes | Judged-result artifact contains judge records or scores for these IDs. |
| `apo_experiment/factorybench_experiment/results/json_contract_v2/mini_vs_gpt55_comparison.json` | EXECUTED_MANIFEST | 5 | 0 | yes | Derivative result contains IDs and outputs/failures tied to a demonstrably executed model run. |
| `apo_experiment/factorybench_experiment/results/l123_smoke/gpt55_l123_smoke.json` | ACTUAL_MODEL_EVALUATION | 15 | 15 | yes | Result artifact contains candidate-model outputs, usage, scores, or a completed model-run status. |
| `apo_experiment/factorybench_experiment/results/l123_smoke/gpt55_l123_smoke_retry.json` | ACTUAL_MODEL_EVALUATION | 15 | 15 | yes | Result artifact contains candidate-model outputs, usage, scores, or a completed model-run status. |
| `apo_experiment/factorybench_experiment/results/mini_neutral_context_audit.json` | INVENTORY_ONLY | 5 | 0 | no | Deterministic context/contract audit only; no candidate-model or judge call is recorded in this artifact. |
| `apo_experiment/factorybench_experiment/results/mini_neutral_context_control.json` | ACTUAL_MODEL_EVALUATION | 8 | 0 | yes | Result artifact contains candidate-model outputs, usage, scores, or a completed model-run status. |
| `apo_experiment/factorybench_experiment/results/mini_neutral_context_control_judged_gpt-5.5.json` | ACTUAL_JUDGE_EXPOSURE | 8 | 0 | yes | Judged-result artifact contains judge records or scores for these IDs. |
| `apo_experiment/factorybench_experiment/results/mini_vs_gpt55_comparison.json` | EXECUTED_MANIFEST | 5 | 0 | yes | Derivative result contains IDs and outputs/failures tied to a demonstrably executed model run. |
| `apo_experiment/factorybench_experiment/results/mini_vs_gpt55_summary.md` | EXECUTED_MANIFEST | 5 | 0 | yes | Result-side artifact is linked to a completed evaluation or judged comparison. |
| `apo_experiment/factorybench_experiment/results/rag_smoke/oracle_fixture_corpus.json` | INVENTORY_ONLY | 2 | 0 | no | Fixture corpus used for plumbing; it does not record candidate-model, optimizer, or judge exposure. |
| `apo_experiment/factorybench_experiment/results/skill_v1_optimization_failed_cases.json` | EXECUTED_MANIFEST | 7 | 0 | yes | Derivative result contains IDs and outputs/failures tied to a demonstrably executed model run. |
| `apo_experiment/factorybench_experiment/results/skill_v1_optimization_judged.json` | ACTUAL_JUDGE_EXPOSURE | 7 | 0 | yes | Judged-result artifact contains judge records or scores for these IDs. |
| `apo_experiment/factorybench_experiment/results/skill_v1_optimization_results.json` | ACTUAL_MODEL_EVALUATION | 7 | 0 | yes | Result artifact contains candidate-model outputs, usage, scores, or a completed model-run status. |
| `apo_experiment/factorybench_experiment/results/skill_v2_optimization_failed_cases.json` | EXECUTED_MANIFEST | 7 | 0 | yes | Derivative result contains IDs and outputs/failures tied to a demonstrably executed model run. |
| `apo_experiment/factorybench_experiment/results/skill_v2_optimization_results.json` | ACTUAL_MODEL_EVALUATION | 7 | 0 | yes | Result artifact contains candidate-model outputs, usage, scores, or a completed model-run status. |
| `apo_experiment/factorybench_experiment/results/skill_v2_temporal_optimization_failed_cases.json` | EXECUTED_MANIFEST | 7 | 0 | yes | Derivative result contains IDs and outputs/failures tied to a demonstrably executed model run. |
| `apo_experiment/factorybench_experiment/results/skill_v2_temporal_optimization_results.json` | ACTUAL_MODEL_EVALUATION | 7 | 0 | yes | Result artifact contains candidate-model outputs, usage, scores, or a completed model-run status. |
| `apo_experiment/factorybench_experiment/results/skill_v2_tool_optimization_failed_cases.json` | EXECUTED_MANIFEST | 7 | 0 | yes | Derivative result contains IDs and outputs/failures tied to a demonstrably executed model run. |
| `apo_experiment/factorybench_experiment/results/skill_v2_tool_optimization_results.json` | ACTUAL_MODEL_EVALUATION | 7 | 0 | yes | Result artifact contains candidate-model outputs, usage, scores, or a completed model-run status. |
| `hermes_factorybench_skill_experiment/data_manifests/l123_all_unseen_generalization.json` | INVENTORY_ONLY | 7572 | 5117 | no | Role is a broad all-unseen inventory; it lists thousands of available rows and has no associated result or execution log. |
| `hermes_factorybench_skill_experiment/data_manifests/l123_expanded_smoke_30.json` | UNEXECUTED_PROSPECTIVE_MANIFEST | 49 | 30 | no | Prospective expanded-smoke manifest; repository search found no result or log executing these selected IDs. |
| `hermes_factorybench_skill_experiment/data_manifests/l123_sealed_generalization_30.json` | UNEXECUTED_PROSPECTIVE_MANIFEST | 50 | 30 | no | Prospective sealed-generalization manifest; repository search found no result or log executing these selected IDs. |
| `hermes_factorybench_skill_experiment/data_manifests/task_a1_l23_validation7_manifest.json` | EXECUTED_MANIFEST | 28 | 14 | yes | Matching Task A result JSONs and exit-status-zero logs establish execution of the manifest IDs. |
| `hermes_factorybench_skill_experiment/data_manifests/task_a2_l1_episode_disjoint7_manifest.json` | EXECUTED_MANIFEST | 15 | 12 | yes | Matching Task A result JSONs and exit-status-zero logs establish execution of the manifest IDs. |
| `hermes_factorybench_skill_experiment/logs/l123_baseline_fresh_gpt55.log` | EXECUTED_MANIFEST | 1 | 0 | yes | Execution log records selected IDs and/or a successful evaluator exit status. |
| `hermes_factorybench_skill_experiment/logs/l123_skill_v1_gpt55.log` | EXECUTED_MANIFEST | 1 | 0 | yes | Execution log records selected IDs and/or a successful evaluator exit status. |
| `hermes_factorybench_skill_experiment/logs/l123_skill_v2_gpt55.log` | EXECUTED_MANIFEST | 1 | 0 | yes | Execution log records selected IDs and/or a successful evaluator exit status. |
| `hermes_factorybench_skill_experiment/logs/l123_skill_v3_gpt55.log` | EXECUTED_MANIFEST | 1 | 0 | yes | Execution log records selected IDs and/or a successful evaluator exit status. |
| `hermes_factorybench_skill_experiment/logs/l123_skill_v4_gpt55.log` | EXECUTED_MANIFEST | 1 | 0 | yes | Execution log records selected IDs and/or a successful evaluator exit status. |
| `hermes_factorybench_skill_experiment/logs/task_a1_l23_validation7_baseline_gpt55.log` | EXECUTED_MANIFEST | 14 | 14 | yes | Execution log records selected IDs and/or a successful evaluator exit status. |
| `hermes_factorybench_skill_experiment/logs/task_a1_l23_validation7_skill_v3_gpt55.log` | EXECUTED_MANIFEST | 14 | 14 | yes | Execution log records selected IDs and/or a successful evaluator exit status. |
| `hermes_factorybench_skill_experiment/logs/task_a2_l1_episode_disjoint7_baseline_gpt55.log` | EXECUTED_MANIFEST | 7 | 7 | yes | Execution log records selected IDs and/or a successful evaluator exit status. |
| `hermes_factorybench_skill_experiment/logs/task_a2_l1_episode_disjoint7_skill_v3_gpt55.log` | EXECUTED_MANIFEST | 7 | 7 | yes | Execution log records selected IDs and/or a successful evaluator exit status. |
| `hermes_factorybench_skill_experiment/optimizer_inputs/hermes_skill_v2_input.json` | ACTUAL_OPTIMIZER_EXPOSURE | 15 | 15 | yes | Optimizer packet contains development failures/results supplied during v2-v4 adapter refinement. |
| `hermes_factorybench_skill_experiment/optimizer_inputs/hermes_skill_v3_input.json` | ACTUAL_OPTIMIZER_EXPOSURE | 4 | 4 | yes | Optimizer packet contains development failures/results supplied during v2-v4 adapter refinement. |
| `hermes_factorybench_skill_experiment/optimizer_inputs/hermes_skill_v4_input.json` | ACTUAL_OPTIMIZER_EXPOSURE | 2 | 2 | yes | Optimizer packet contains development failures/results supplied during v2-v4 adapter refinement. |
| `hermes_factorybench_skill_experiment/reports/smoke_selection_correction.md` | EXECUTED_MANIFEST | 1 | 1 | yes | Audit/report trace records IDs from demonstrably executed runs and optimizer rounds. |
| `hermes_factorybench_skill_experiment/reports/task_a1_l23_validation7_report.md` | EXECUTED_MANIFEST | 2 | 2 | yes | Audit/report trace records IDs from demonstrably executed runs and optimizer rounds. |
| `hermes_factorybench_skill_experiment/reports/task_a2_l1_episode_disjoint7_report.md` | EXECUTED_MANIFEST | 5 | 5 | yes | Audit/report trace records IDs from demonstrably executed runs and optimizer rounds. |
| `hermes_factorybench_skill_experiment/results/generalization/task_a1_l23_validation7_baseline_gpt55.json` | ACTUAL_MODEL_EVALUATION | 14 | 14 | yes | Result artifact contains candidate-model outputs, usage, scores, or a completed model-run status. |
| `hermes_factorybench_skill_experiment/results/generalization/task_a1_l23_validation7_comparison.json` | EXECUTED_MANIFEST | 14 | 14 | yes | Derivative result contains IDs and outputs/failures tied to a demonstrably executed model run. |
| `hermes_factorybench_skill_experiment/results/generalization/task_a1_l23_validation7_skill_v3_gpt55.json` | ACTUAL_MODEL_EVALUATION | 14 | 14 | yes | Result artifact contains candidate-model outputs, usage, scores, or a completed model-run status. |
| `hermes_factorybench_skill_experiment/results/generalization/task_a2_l1_episode_disjoint7_baseline_gpt55.json` | ACTUAL_MODEL_EVALUATION | 7 | 7 | yes | Result artifact contains candidate-model outputs, usage, scores, or a completed model-run status. |
| `hermes_factorybench_skill_experiment/results/generalization/task_a2_l1_episode_disjoint7_comparison.json` | EXECUTED_MANIFEST | 10 | 7 | yes | Derivative result contains IDs and outputs/failures tied to a demonstrably executed model run. |
| `hermes_factorybench_skill_experiment/results/generalization/task_a2_l1_episode_disjoint7_skill_v3_gpt55.json` | ACTUAL_MODEL_EVALUATION | 7 | 7 | yes | Result artifact contains candidate-model outputs, usage, scores, or a completed model-run status. |
| `hermes_factorybench_skill_experiment/results/smoke/gpt55_l123_baseline.json` | ACTUAL_MODEL_EVALUATION | 15 | 15 | yes | FactoryBench result artifact contains model outputs or paired results from an executed run. |
| `hermes_factorybench_skill_experiment/results/smoke/hermes_self_evolution_trace.json` | EXECUTED_MANIFEST | 15 | 15 | yes | Audit/report trace records IDs from demonstrably executed runs and optimizer rounds. |
| `hermes_factorybench_skill_experiment/results/smoke/l123_baseline_fresh_gpt55.json` | ACTUAL_MODEL_EVALUATION | 15 | 15 | yes | FactoryBench result artifact contains model outputs or paired results from an executed run. |
| `hermes_factorybench_skill_experiment/results/smoke/l123_skill_v1_comparison.json` | EXECUTED_MANIFEST | 15 | 15 | yes | Derivative result contains IDs and outputs/failures tied to a demonstrably executed model run. |
| `hermes_factorybench_skill_experiment/results/smoke/l123_skill_v1_gpt55.json` | ACTUAL_MODEL_EVALUATION | 15 | 15 | yes | Result artifact contains candidate-model outputs, usage, scores, or a completed model-run status. |
| `hermes_factorybench_skill_experiment/results/smoke/l123_skill_v2_comparison.json` | EXECUTED_MANIFEST | 15 | 15 | yes | Derivative result contains IDs and outputs/failures tied to a demonstrably executed model run. |
| `hermes_factorybench_skill_experiment/results/smoke/l123_skill_v2_gpt55.json` | ACTUAL_MODEL_EVALUATION | 15 | 15 | yes | Result artifact contains candidate-model outputs, usage, scores, or a completed model-run status. |
| `hermes_factorybench_skill_experiment/results/smoke/l123_skill_v3_comparison.json` | EXECUTED_MANIFEST | 3 | 3 | yes | Derivative result contains IDs and outputs/failures tied to a demonstrably executed model run. |
| `hermes_factorybench_skill_experiment/results/smoke/l123_skill_v3_gpt55.json` | ACTUAL_MODEL_EVALUATION | 15 | 15 | yes | Result artifact contains candidate-model outputs, usage, scores, or a completed model-run status. |
| `hermes_factorybench_skill_experiment/results/smoke/l123_skill_v4_comparison.json` | EXECUTED_MANIFEST | 4 | 4 | yes | Derivative result contains IDs and outputs/failures tied to a demonstrably executed model run. |
| `hermes_factorybench_skill_experiment/results/smoke/l123_skill_v4_gpt55.json` | ACTUAL_MODEL_EVALUATION | 15 | 15 | yes | Result artifact contains candidate-model outputs, usage, scores, or a completed model-run status. |
| `FactoryBench/FactoryBench@b3863519ccedbceab54dfa7600104eb42b985ed7:factorybench_qa/level_1/test.jsonl` | SOURCE_DATASET_ONLY | 0 | 0 | no | Availability/source record only; it contributed no IDs to the original registry and is not evidence of model, optimizer, or judge exposure. |
| `FactoryBench/FactoryBench@b3863519ccedbceab54dfa7600104eb42b985ed7:factorybench_qa/level_2/validation.jsonl` | SOURCE_DATASET_ONLY | 0 | 0 | no | Availability/source record only; it contributed no IDs to the original registry and is not evidence of model, optimizer, or judge exposure. |
| `FactoryBench/FactoryBench@b3863519ccedbceab54dfa7600104eb42b985ed7:factorybench_qa/level_3/validation.jsonl` | SOURCE_DATASET_ONLY | 0 | 0 | no | Availability/source record only; it contributed no IDs to the original registry and is not evidence of model, optimizer, or judge exposure. |
| `hermes_factorybench_skill_experiment/data_manifests/meta_m0/preflight_summary.json` | INVENTORY_ONLY | 0 | 0 | no | Availability/source record only; it contributed no IDs to the original registry and is not evidence of model, optimizer, or judge exposure. |

## Eligible counts before and after correction

| Pool | Source | Original eligible | Corrected eligible |
|---|---:|---:|---:|
| L1_test | 1309 | 0 | 1238 |
| L2_validation | 3428 | 2403 | 3326 |
| L3_validation | 265 | 75 | 251 |

## Recovery decision

Recovered 3 development items and 6 holdout items with exact ID and episode disjointness.
L1 uses the pinned test split because L1 validation is unavailable. L2 and L3 use validation. The mixed-split aggregate is not a pure validation score.
