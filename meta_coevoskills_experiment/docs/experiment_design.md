# Controlled Meta-CoEvoSkills Experiment Design

## Research question

Does structured failure diagnosis from an independent Surrogate Verifier help `gpt-5.6-luna` with reasoning.mode=pro and reasoning.effort=xhigh produce a shared Skill that improves `gpt-4o-mini` on FactoryBench Levels 1-3 more reliably than coarse evaluator feedback alone?

FactoryBench Level 4 is out of scope and is rejected by preflight.

## Experimental arms

CONTROL:

`gpt-4o-mini output -> frozen deterministic evaluator -> coarse feedback -> gpt-5.6-luna with reasoning.mode=pro and reasoning.effort=xhigh`

SURROGATE:

`gpt-4o-mini output -> independent gpt-5.6-luna with reasoning.mode=standard and reasoning.effort=high verifier -> aggregate structured diagnosis -> gpt-5.6-luna with reasoning.mode=pro and reasoning.effort=xhigh`

Both arms use the same zero-byte v0, manifests, fold schedule, target parameters, rewriter parameters, generation budget, rendering, parser, evaluator, and holdout procedure. The feedback mode is the intended independent variable. Treatment rewriter input receives the aggregate diagnosis instead of GT-derived detailed feedback; evaluator outputs remain measurement-only except for CONTROL's explicitly permitted coarse fields.

No model may be silently substituted. Runtime accepts only:

- target: `gpt-4o-mini`
- verifier: `gpt-5.6-luna`, reasoning.mode=`standard`, reasoning.effort=`high`
- rewriter: `gpt-5.6-luna`, reasoning.mode=`pro`, reasoning.effort=`xhigh`

A normalized runtime guard rejects `gpt-5.6-luna-pro`, `gpt-5.6-sol`, accidental Sol aliases/fallbacks, and any returned-model alias before client creation and again before each model call.

## Data and schedule

Only the three symlinked M1 L123 manifests are accepted. Preflight checks frozen hashes, counts, one/three items per level, levels exactly 1/2/3, split provenance, ID disjointness, episode disjointness, v0 byte identity, schemas, and source-item metadata at pinned FactoryBench revision `b3863519ccedbceab54dfa7600104eb42b985ed7`.

Schedule:

| Round | Parent Skill optimized on | New Skill validated on |
|---:|---|---|
| 1 | A | B |
| 2 | B | A |
| 3 | A | B |

The immediately preceding validation checkpoint is reused as the next round's optimization result when the Skill hash, fold, and IDs are identical. This avoids a duplicate stochastic target call and makes resume cheaper without changing evidence.

L1 uses FactoryBench test samples; L2/L3 use validation samples. Reports must say **FactoryBench L1-L3 mixed-split evaluation**, never pure validation score.

## Starting Skill

M1 L123 had no Core and no Adapter. v0 is therefore the exact no-system-prompt baseline represented by a zero-byte file. CONTROL and SURROGATE copies are byte-identical with SHA-256:

`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`

## Feedback isolation

CONTROL rewriter receives only:

- current Skill;
- canonical score;
- fixed-cardinality score;
- parse-failure count;
- failed sample ID, level, and answer format.

It does not receive raw output, parsed output, reference answer, corrected answer, or detailed diagnosis.

Each SURROGATE verifier request is a fresh independent Responses API request. Its payload has exactly:

- sample ID and level;
- public rendered input (description, mapping, telemetry, question, options);
- public answer-format name;
- target answer.

It never receives the current Skill, prior Skills, rewrite traces, evaluator scores, reference answers, acceptance bounds beyond those publicly rendered, or holdout information.

Treatment rewriter receives only current Skill plus the aggregate diagnosis. Individual diagnosis prose is never copied into the rewriter prompt. Aggregation reduces diagnoses to the controlled failure taxonomy, counts, cross-level recurrence, and a fixed general-purpose failure-type-to-Skill-gap mapping. UNKNOWN or confidence below 0.5 is retained only as a generic uncertainty marker; item-level root-cause, Skill-gap, observed, and proposed-revision prose is withheld.

## Rewriter constraints

The rewriter must classify evidence into Skill-fixable, capability-limited, and ambiguous/insufficient categories before emitting a complete new Skill. Output is rejected if empty or if it includes any manifest sample ID, holdout/fold terminology, experiment-arm terminology, or a Sol model identifier. The prompt prohibits exact development answers, reference answers, GT, and sample-specific rules.

## Evaluation and holdout

After each rewrite, the child Skill is evaluated only on the opposite development fold. That validation result cannot modify the same generation. It may serve as the next round's optimization evidence according to the frozen alternating schedule.

Full mode evaluates v0-v3 on the frozen nine-item holdout. v0 target outputs are shared across arms because the Skill bytes and all target settings are identical. Intermediate holdout scores are descriptive only. No holdout artifact is passed to verifier, aggregation, or rewrite functions. No best generation is selected with holdout.

Primary endpoint: final v3 holdout fixed-cardinality and canonical performance, with v3 as predetermined endpoint.

## Metrics

Each generation artifact records optimization and validation canonical/fixed scores, parse failures, format validity, by-level and by-format scores, holdout metrics, paired improved/regressed/unchanged counts versus v0, Skill hash/characters/tokens, model IDs, and rewriter usage/cost. Per-call target, verifier, and rewriter checkpoints retain input/output token usage, estimated cost from the pinned local pricing snapshot, and wall time.

FactoryBench canonical score excludes parse-failed/non-finite rows. The fixed-cardinality score retains every expected item, substitutes zero raw score for invalid rows, and preserves chance contribution. Fixed-cardinality is the conservative comparison.

## Checkpointing and resume

Every paid response is atomically checkpointed immediately before parsing/evaluation continues. Every checkpoint is bound to a SHA-256 identity covering its exact request inputs, model settings, prompts/schemas where applicable, config, Skill/prompt content, and runner code. Returned model IDs must exactly match the requested model; a returned Sol model or any alias mismatch is checkpointed as invalid and execution stops. Parsed diagnoses, aggregate diagnoses, exact rewriter inputs, raw rewriter responses, parsed rewrite records, every Skill version, evaluations, and generation metrics are separate artifacts.

A fresh run refuses existing checkpoints. `--resume` reuses only request-identical, validated `complete` checkpoints. Error or invalid structured-output checkpoints are archived before an explicit resume retry, preserving the paid raw output; mismatched request identities fail instead of being reused. The OpenAI client is configured with `max_retries=0`, so the reviewed logical call inventory is also the automatic HTTP-attempt bound. Smoke and full runs use separate directories.

## Paid-call gate

Dry-run performs all local preflight and source verification, prints the complete call plan, writes a machine-readable plan, and exits successfully before creating an API client. Paid execution requires both `--execute-paid` and the exact plan SHA-256 printed by that dry-run. A mismatched plan hash refuses execution.

### Smoke plan and command

Dry-run:

`python meta_coevoskills_experiment/scripts/run_experiment.py --dry-run --smoke --arm surrogate --rounds 1`

Reviewed plan SHA-256:

`ccdacd63327de18aa5698fc3c1ceea68e4219cc7253957008cba2518550cdb0a`

Manual paid smoke command (not run during implementation):

`python meta_coevoskills_experiment/scripts/run_experiment.py --smoke --arm surrogate --rounds 1 --execute-paid --plan-sha256 ccdacd63327de18aa5698fc3c1ceea68e4219cc7253957008cba2518550cdb0a`

Smoke maximum: 6 target + 3 verifier + 1 rewrite = 10 paid calls. It uses one L1, one L2, and one L3 optimization sample and one of each on the opposite fold for validation. It makes zero holdout calls. Unit tests cover checkpoint resume; rerunning the completed smoke with `--resume` and the same authorization must reuse completed calls.

### Full plan and command

Dry-run:

`python meta_coevoskills_experiment/scripts/run_experiment.py --dry-run --arm both --rounds 3`

Reviewed plan SHA-256:

`adceb38605db61064972d5e4c6beabd1251e6312c8826d022b1dd9bde39a68f1`

Manual paid full command, only after smoke succeeds:

`python meta_coevoskills_experiment/scripts/run_experiment.py --arm both --rounds 3 --execute-paid --plan-sha256 adceb38605db61064972d5e4c6beabd1251e6312c8826d022b1dd9bde39a68f1`

Full maximum: 87 target + 9 verifier + 6 rewrite = 102 paid calls. Of the 87 target calls, 63 are holdout measurements: 9 shared v0 calls plus 27 post-rewrite calls per arm.

If interrupted, rerun the same command with `--resume`; the plan hash is unchanged because resume policy does not change the experimental call inventory.

## Analysis policy

The final report will compare CONTROL and SURROGATE by identical holdout IDs at every generation, emphasize L3 gains, L1/L2 regressions, parse/format regressions, and diagnosis-followed-by-validation-worsening cases, and state whether v3 treatment beats v3 control. It will not claim Surrogate benefit unless frozen v3 holdout measurements support it.
