# Task A2 — L1 Episode-Disjoint Holdout

This is a frozen Skill v3 smoke evaluation. No refinement occurred and no result was fed back into Hermes.

## Configuration

- Revision: `b3863519ccedbceab54dfa7600104eb42b985ed7`
- Split: `test`
- Levels: `[1]`
- Items: 7
- Model: `gpt-5.5`
- Concurrency: 2
- Frozen Skill v3 SHA-256: `cc83006d7a2e24aaf86ee7de16900cad71cd668563fb0ac2d8ef62b03ccb527e`

## Frozen selection preflight

- Seed: `task-a-l1-episode-disjoint-v1`
- Rule: Require a non-empty provenance episode; exclude optimization IDs, all items sharing an original L1 optimization episode, and trace-established L1 IDs; sort remaining items by SHA-256(UTF-8(seed + item_id)) ascending with item_id tie-break; take first 7.
- Pinned L1 test items: 1309
- Union excluded: 51
- Eligible: 1258
- Selected: 7
- Selected IDs and provenance episodes are frozen in the manifest.

## Overall scores

| Run | FactoryBench score | Fixed-cardinality score | Parse failures | Cost | Wall time |
|---|---:|---:|---:|---:|---:|
| Baseline | 0.6842105263 | 0.6842105263 | 0 | $0.360545 | 64.309768 s |
| Frozen Skill v3 | 0.2105263158 | 0.2105263158 | 0 | $0.445455 | 89.653530 s |

Absolute delta: `-0.4736842105263158` (-47.368421 percentage points; -69.230769% relative).

## Group scores

### By level

| Group | Baseline | v3 | Delta |
|---|---:|---:|---:|
| L1 | 0.6842105263 | 0.2105263158 | -0.4736842105 |

### By dataset

| Group | Baseline | v3 | Delta |
|---|---:|---:|---:|
| aursad | 0.7272727273 | 0.1818181818 | -0.5454545455 |
| factorywave | 0.6250000000 | 0.2500000000 | -0.3750000000 |

### By answer format

| Group | Baseline | v3 | Delta |
|---|---:|---:|---:|
| scalar_exact | 1.0000000000 | 1.0000000000 | +0.0000000000 |
| scalar_range | 0.5000000000 | 0.2500000000 | -0.2500000000 |
| single_letter_mcq | 1.0000000000 | -0.5000000000 | -1.5000000000 |

## Paired outcomes

- Improved: 0
- Worse: 3
- Unchanged: 4
- Invalid: 0
- Raw-output changed cases: 5

### Changed cases

| ID | Level | Format | Baseline output | v3 output | Baseline score | v3 score | Status |
|---|---:|---|---|---|---:|---:|---|
| `27f70f84-afd1-47a5-b5c0-3fb1e97a2579` | L1 | scalar_range | `381` | `0` | 0.0 | 0.0 | unchanged |
| `5372d137-caed-4ca7-9e24-239182e3ace9` | L1 | single_letter_mcq | `A` | `B` | 1.0 | 0.0 | worse |
| `5255ae6d-57d1-4532-b25f-d2dab1557179` | L1 | single_letter_mcq | `C` | `A` | 1.0 | 0.0 | worse |
| `14b816fb-e301-4161-b5e4-282d5e5a835c` | L1 | scalar_range | `0` | `890` | 1.0 | 0.0 | worse |
| `70346496-3321-4592-9085-b247dad6b254` | L1 | scalar_range | `494` | `2877` | 0.0 | 0.0 | unchanged |

## Token usage

```json
{
  "baseline": {
    "candidate": {
      "model": "gpt-5.5",
      "input_tokens": 48481,
      "output_tokens": 7876,
      "calls": 7
    },
    "judges": {}
  },
  "v3": {
    "candidate": {
      "model": "gpt-5.5",
      "input_tokens": 53220,
      "output_tokens": 11957,
      "calls": 7
    },
    "judges": {}
  }
}
```

## Artifact hashes

- frozen_skill_v3: `cc83006d7a2e24aaf86ee7de16900cad71cd668563fb0ac2d8ef62b03ccb527e` — `prompts/generated/factorybench_skill_v3.txt`
- original_invalid_preflight: `e17a605c5dc8c644fd518c82f67f0b0041af8e2a3a4ae7aa7a8af1dd925cb259` — `data_manifests/task_a_validation7_invalid_preflight.json`
- original_invalid_preflight_report: `550b0c728f93bc3af28e13fc33c864c203f6518dc77bd9b6515514aa75a92316` — `reports/task_a_validation7_invalid_preflight.md`
- optimization_id_source: `10123b00423178db865f28b783b2949dff45449c828136245426b34bbd76d62a` — `results/smoke/l123_baseline_fresh_gpt55.json`
- manifest: `eb3be4837eb0d13a73a968607f9c3482dddf0d8bd51538b4aaccd4465eb10490` — `data_manifests/task_a2_l1_episode_disjoint7_manifest.json`
- baseline_result: `980256e276cf682846d47719126b88a2f17fc3713f802d54b54998725eea96e2` — `results/generalization/task_a2_l1_episode_disjoint7_baseline_gpt55.json`
- v3_result: `a06b3ce4d96c994df78dcc00ca907da4e5a0c05521989e6675f95796649fbc66` — `results/generalization/task_a2_l1_episode_disjoint7_skill_v3_gpt55.json`
- baseline_log: `500934646303f6eccf39abed5d332ad9a1324902531fc21415c2582f3d0f6779` — `logs/task_a2_l1_episode_disjoint7_baseline_gpt55.log`
- v3_log: `f18d49941a5e6179d816cdf7df7763b9eb8502c856a4734da37318310f3eb46d` — `logs/task_a2_l1_episode_disjoint7_skill_v3_gpt55.log`
- exact_id_runner: `a77f4fd2359ddd57dae4701a76d881010a32eaade15be9156fdd0e8cb2fd4ef8` — `scripts/task_a_exact_id_runner.py`
- comparison_builder: `58802c7b9485761eac3b316196c4f63a80300a591f89c54a9475802af0df3b95` — `scripts/task_a_build_comparison.py`
- refinement_trace: `0ced7e068caaf6db4f764f22cbac6cba8ad0acf925bc8035c9830041169bebc4` — `results/smoke/hermes_self_evolution_trace.json`
- Final comparison/report hashes: recorded in `results/generalization/task_a_stratified_integrity_hashes.json`.

## Verdict

**NEGATIVE_SIGNAL**

The earlier `INVALID_PREFLIGHT` record remains preserved as a separate artifact and was not overwritten or relabeled.

The frozen Skill v3 score is below baseline on the episode-disjoint L1 holdout, so this is a negative signal.

This small smoke result is not a test of a Golden Meta-Prompt and is not proof of broad generalization.
