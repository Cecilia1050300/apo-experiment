---
name: factory-troubleshooting
description: Diagnose FactoryBench Level 4 machine faults.
version: 0.1.0
author: Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [factorybench, industrial, troubleshooting, diagnosis]
    related_skills: []
---

# FactoryBench Level 4 Troubleshooting

Solve FactoryBench Level 4 industrial troubleshooting tasks with an evidence-first causal diagnosis. Interpret the supplied machine state, distinguish symptoms from causes, compare plausible fault hypotheses, and recommend the safest corrective action justified by the scenario. Do not import answers or facts from other FactoryBench examples.

## When to Use

Use this skill only when the task is a FactoryBench Level 4 industrial machine troubleshooting problem involving one or more of:

- machine state, mode, sequence, alarm, or interlock interpretation;
- anomalous sensor, actuator, process, quality, or maintenance evidence;
- diagnosis of an underlying equipment, control, material, utility, or operating fault;
- selection of a corrective action from the supplied evidence.

Do not use it for other FactoryBench levels, generic causal-judgment benchmarks, or questions whose primary purpose is machine design rather than fault diagnosis.

## Core Rules

1. Treat the scenario as the complete evidence set unless it explicitly authorizes outside information.
2. Separate observations, inferences, and assumptions. Never restate an inference as if it were measured.
3. Interpret every signal in context: operating mode, commanded state, sequence step, timing, load, setpoint, and related signals can change its meaning.
4. Diagnose the earliest supported abnormal condition that explains the downstream pattern, not merely the loudest alarm or latest symptom.
5. Prefer a cause that explains several independent observations over one that explains only one reading.
6. Compare alternatives and use contradictory or missing expected evidence to reduce confidence.
7. Do not invent sensor values, component failures, maintenance history, causal links, or actions absent from the task.
8. Match the requested response format exactly. If the task asks for one diagnosis or action, commit to the best-supported one rather than returning an unranked list.

## Procedure

### 1. Parse the task contract

Identify:

- the exact question: state interpretation, anomaly, root cause, corrective action, or a combination;
- required output form and level of detail;
- the machine or process boundary under diagnosis;
- the time window and operating phase described.

Completion criterion: the requested decision and response format are explicit before diagnosis begins.

### 2. Reconstruct the intended machine state

Build a compact state model from the prompt:

- current mode: stopped, idle, setup, manual, automatic, starting, running, stopping, faulted, or recovering;
- commanded actions and expected sequence step;
- setpoints, permissives, interlocks, and controller outputs;
- expected relationships among sensors, actuators, process flow, and product behavior.

A value is not anomalous merely because it is high, low, on, or off. Ask whether it is appropriate for the current command, sequence, and load.

Completion criterion: expected behavior for the relevant operating state is stated or marked unknown.

### 3. Extract observations without interpretation drift

Organize the evidence into these categories:

- direct measurements and statuses;
- alarms, trips, and controller messages;
- operator or maintenance observations;
- recent changes, disturbances, or interventions;
- explicit normal findings and negative evidence;
- timing and order of events.

Preserve qualifiers such as intermittent, delayed, only under load, after warm-up, or despite a command. These often localize the fault.

Completion criterion: every diagnosis-relevant fact can be traced to the scenario, and no inferred fact is labeled as observed.

### 4. Identify the anomaly pattern

Compare actual behavior with the reconstructed expected state. Look for:

- command-versus-response mismatch;
- sensors that disagree about the same physical condition;
- values inconsistent with operating phase or load;
- a broken sequence transition or unsatisfied permissive;
- process imbalance across upstream and downstream points;
- timing anomalies, oscillation, drift, saturation, or intermittency;
- quality symptoms that correlate with machine-state changes.

Distinguish a true process anomaly from a measurement anomaly. A surprising reading with normal corroborating process behavior may indicate sensing, scaling, wiring, or signal-path trouble; multiple independent physical effects usually favor a real process fault.

Completion criterion: the abnormal pattern is expressed as a specific deviation from expected behavior.

### 5. Build the causal chain

Order the evidence from initiating condition to propagated effects:

`candidate fault -> local mechanism -> state/sensor changes -> control response or alarm -> production symptom`

Use temporal order and mechanism together:

- a cause should occur before or at the onset of its effects;
- the mechanism should explain the direction and combination of changes;
- controller reactions and alarms may be downstream protective responses rather than causes;
- an upstream fault can produce many downstream symptoms, while a downstream symptom rarely explains upstream evidence.

Do not confuse these categories:

- **root cause:** the underlying fault that initiated the abnormal chain;
- **contributing condition:** a factor that worsened or enabled it;
- **symptom:** a resulting reading, alarm, quality defect, or machine behavior;
- **protective response:** an interlock, trip, shutdown, or controller correction.

Completion criterion: the leading hypothesis connects the suspected root cause to each major symptom through a plausible chain supported by the prompt.

### 6. Compare competing hypotheses

Generate only plausible alternatives grounded in the supplied system and evidence. For each candidate, test:

- explanatory coverage: how many key observations does it explain?
- temporal fit: does it precede the effects?
- state fit: is it compatible with commands, mode, and sequence?
- corroboration: do independent sensors or observations agree?
- contradiction: what supplied evidence should not occur if it were true?
- assumption cost: how many unstated conditions must be invented?

Prefer the hypothesis with the strongest combined fit, not the most familiar failure mode. Absence of a mentioned alarm is weak evidence unless the task says that alarm should be present or enumerates the alarm state.

Completion criterion: at least one credible alternative has been checked, or the evidence uniquely determines the diagnosis.

### 7. Calibrate the conclusion

Use decisive wording when the evidence is discriminating. When it is not, identify the leading diagnosis and the precise uncertainty rather than fabricating certainty.

A useful internal form is:

- most likely root cause;
- decisive evidence;
- why the closest alternative fits less well;
- confidence limitation, if material.

Do not expose a long internal chain of thought. Give a concise evidence-based explanation appropriate to the requested format.

Completion criterion: confidence matches the evidence, and the answer does not rely on unsupported assumptions.

### 8. Recommend the corrective action

Choose an action aimed at the diagnosed cause, not just suppression of its symptoms. The action should be:

- specific to the implicated component, control condition, material path, utility, or operating state;
- consistent with the evidence and no more invasive than justified;
- sequenced safely: stabilize or stop if required, isolate hazardous energy where applicable, correct the cause, then verify restoration;
- paired with a functional check that would confirm normal command-response and process behavior.

Do not recommend bypassing interlocks, defeating safeguards, repeatedly resetting a trip without diagnosis, or replacing unrelated parts. If evidence cannot distinguish between causes, recommend the safest discriminating inspection or test before replacement.

Completion criterion: the action addresses the root cause and includes a way to verify recovery without bypassing protection.

## Answer Pattern

Unless the prompt requires another format, answer compactly:

- **Machine state/anomaly:** the relevant state and specific mismatch.
- **Most likely root cause:** one best-supported underlying cause.
- **Why:** two or three decisive observations linked causally, including why a salient symptom is downstream when relevant.
- **Corrective action:** a targeted, safe correction and a brief verification step.

Omit fields the question does not request.

## Pitfalls

- Anchoring on the first alarm even when it is a consequence.
- Treating all out-of-range values as independent faults.
- Ignoring whether an actuator was actually commanded to move.
- Assuming a sensor is bad solely because its reading is abnormal.
- Recommending recalibration when physical corroboration indicates a real process fault.
- Naming a common failure mode without connecting it to the supplied evidence.
- Inferring that unmentioned equipment is normal, abnormal, present, or absent.
- Listing many possible causes when the task requires the most likely one.
- Giving a maintenance action that masks the symptom but leaves the initiating fault.
- Adding scenario-specific rules learned from another example.

## Verification Checklist

Before answering, verify:

- [ ] The task is FactoryBench Level 4.
- [ ] The operating mode and expected state were considered.
- [ ] Each claimed observation appears in the prompt.
- [ ] The key anomaly is a state-relative mismatch.
- [ ] The proposed cause precedes and explains the major symptoms.
- [ ] Symptoms, protective responses, and root cause are distinguished.
- [ ] A credible alternative was tested against contradictory evidence.
- [ ] The corrective action targets the supported cause and respects safeguards.
- [ ] No held-out knowledge, example-specific answer, or unsupported assumption was used.
- [ ] The response follows the requested format.