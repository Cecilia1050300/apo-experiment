# Factory Troubleshooting Skill v1 Creation Record

## Why this skill was created

Skill v1 was created as the initial reusable troubleshooting procedure for the FactoryBench Level 4 optimization experiment. Its purpose is to make diagnosis systematic, evidence-grounded, causally coherent, and reusable across examples without encoding any example-specific answer.

## Reasoning principles included

The skill instructs an agent to:

- reconstruct the expected machine state from mode, commands, sequence, load, setpoints, permissives, and interlocks;
- identify anomalies as deviations from that state rather than treating isolated high/low or on/off values as intrinsically abnormal;
- separate direct observations from inferences and assumptions;
- organize timing, alarms, measurements, contextual changes, and negative evidence;
- distinguish root causes, contributing conditions, symptoms, and protective responses;
- construct a causal chain from initiating fault through local mechanism and downstream effects;
- compare grounded alternative hypotheses by coverage, temporal fit, state fit, corroboration, contradiction, and assumption cost;
- discriminate real process faults from sensor or signal-path faults using independent physical corroboration;
- calibrate confidence and avoid unsupported assumptions;
- recommend a targeted, safe corrective action that addresses the supported cause and includes a recovery check.

## Information available at creation

At creation time, the available information consisted only of:

- the experiment instructions supplied by the user;
- the requested FactoryBench Level 4 troubleshooting capabilities;
- the empty experiment directory structure (`data/`, `logs/`, `prompts/`, `results/`, and `skills/`), inspected by file names only;
- general reusable causal-reasoning and Hermes skill-authoring guidance.

No FactoryBench task examples, optimization answers, evaluation outputs, failure cases, result files, or held-out final-test answers were present or inspected.

## Evaluation status

This is the pre-evaluation v1 skill. No evaluation failures or external harness feedback were available or used when creating it. No Round 2 optimization was started.