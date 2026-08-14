# Task A1 — Split-Disjoint L2/L3 Validation

This is a frozen Skill v3 smoke evaluation. No refinement occurred and no result was fed back into Hermes.

## Configuration

- Revision: `b3863519ccedbceab54dfa7600104eb42b985ed7`
- Split: `validation`
- Levels: `[2, 3]`
- Items: 14
- Model: `gpt-5.5`
- Concurrency: 2
- Frozen Skill v3 SHA-256: `cc83006d7a2e24aaf86ee7de16900cad71cd668563fb0ac2d8ef62b03ccb527e`

## Overall scores

| Run | FactoryBench score | Fixed-cardinality score | Parse failures | Cost | Wall time |
|---|---:|---:|---:|---:|---:|
| Baseline | 0.8349834983 | 0.8349834983 | 0 | $1.255690 | 211.062842 s |
| Frozen Skill v3 | 0.8151815182 | 0.8151815182 | 0 | $1.229040 | 181.423916 s |

Absolute delta: `-0.01980198019801993` (-1.980198 percentage points; -2.371542% relative).

## Group scores

### By level

| Group | Baseline | v3 | Delta |
|---|---:|---:|---:|
| L2 | 0.6478873239 | 0.6056338028 | -0.0422535211 |
| L3 | 1.0000000000 | 1.0000000000 | +0.0000000000 |

### By dataset

| Group | Baseline | v3 | Delta |
|---|---:|---:|---:|
| factorywave | 0.8349834983 | 0.8151815182 | -0.0198019802 |

### By answer format

| Group | Baseline | v3 | Delta |
|---|---:|---:|---:|
| four_letter_ranking | 1.0000000000 | 1.0000000000 | +0.0000000000 |
| four_letter_tf | 0.2500000000 | 0.0000000000 | -0.2500000000 |
| scalar_margin | 1.0000000000 | 1.0000000000 | +0.0000000000 |
| tensor_margin | 0.3333333333 | 0.3333333333 | +0.0000000000 |

## Paired outcomes

- Improved: 0
- Worse: 1
- Unchanged: 13
- Invalid: 0
- Raw-output changed cases: 2

### Changed cases

| ID | Level | Format | Baseline output | v3 output | Baseline score | v3 score | Status |
|---|---:|---|---|---|---:|---:|---|
| `5f44af3e-26ee-40ce-a4e9-36c7f6a1fa9f` | L2 | four_letter_tf | `TFTT` | `TFFT` | 0.75 | 0.5 | worse |
| `485feba7-8c87-48b1-a027-5b25361204dc` | L2 | tensor_margin | `[-0.0129,-1.716,5.157,-3.442,0.000279,0.0344]` | `[-0.01292,-1.71644,5.15731,-3.44154,0.000278,0.03435]` | 0.6666666666666666 | 0.6666666666666666 | unchanged |

## Token usage

```json
{
  "baseline": {
    "candidate": {
      "model": "gpt-5.5",
      "input_tokens": 165476,
      "output_tokens": 28554,
      "calls": 14
    },
    "judges": {}
  },
  "v3": {
    "candidate": {
      "model": "gpt-5.5",
      "input_tokens": 174954,
      "output_tokens": 23618,
      "calls": 14
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
- manifest: `1d4169a4c10bac4602509e2bbe979d33d37f36ab2a20d85ea46b8222526b5404` — `data_manifests/task_a1_l23_validation7_manifest.json`
- baseline_result: `cd817920a9081afd182da188e3a1f7ed9702a1c771c3484ac1bc78fd3c18a3dd` — `results/generalization/task_a1_l23_validation7_baseline_gpt55.json`
- v3_result: `aac3f9a952f5c1b9e5803f504728950758ffd5fe888636c3f6478c4bb8581a4a` — `results/generalization/task_a1_l23_validation7_skill_v3_gpt55.json`
- baseline_log: `840301c7bcdb1ba3c2ba660b24e07b376513989e3a99a01d7e4bde4ed04a6d5b` — `logs/task_a1_l23_validation7_baseline_gpt55.log`
- v3_log: `3fd95cb43f8365314bd618b529941f1c6b12448dc5d4af7279f3761b51fd3154` — `logs/task_a1_l23_validation7_skill_v3_gpt55.log`
- exact_id_runner: `a77f4fd2359ddd57dae4701a76d881010a32eaade15be9156fdd0e8cb2fd4ef8` — `scripts/task_a_exact_id_runner.py`
- comparison_builder: `58802c7b9485761eac3b316196c4f63a80300a591f89c54a9475802af0df3b95` — `scripts/task_a_build_comparison.py`
- Final comparison/report hashes: recorded in `results/generalization/task_a_stratified_integrity_hashes.json`.

## Verdict

**FAIL_REGRESSION**

The earlier `INVALID_PREFLIGHT` record remains preserved as a separate artifact and was not overwritten or relabeled.

The overall delta is negative, so this frozen L2/L3 validation smoke is a regression under the predeclared rule.

This small smoke result is not a test of a Golden Meta-Prompt and is not proof of broad generalization.
