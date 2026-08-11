# Skill v2 Error Attribution and Modification Rationale

## Scope and evidence boundary

This analysis used only:

- the persistent FactoryBench Level 4 `factory-troubleshooting` Skill v1; and
- `results/skill_v1_optimization_failed_cases.json`.

The failed-case file contained five Round 1 failures. No aggregate optimization-results file, Round 2 output, final-test file, held-out answer, or original dataset file was opened. No case ID or example-specific option mapping was added to the skill.

## Concise error attribution

### F1. Sensor and control symptoms were promoted to root causes (3 of 5 failures)

Three responses described signed target torques, estimated forces, or alleged position discrepancies as the root cause. In each, the answer named the visible signal pattern rather than identifying the functional or physical condition producing it. The same responses defaulted to calibration, controller checks, or broad component inspection without evidence that a sensor or controller was faulty.

Attribution: Skill v1 distinguished symptoms from causes in general terms, but it did not explicitly classify target/control signals as intent or require a first-broken-boundary diagnosis before permitting a sensor fault.

### F2. Torque/force sign and magnitude were interpreted without phase context (3 of 5 failures)

The responses treated negative target torque as inherently excessive or abnormal. The failed cases show that sign can represent commanded direction and that a target value is controller intent, not measured proof of failure. No supplied baseline established that the magnitude itself was excessive.

Attribution: Skill v1 said to consider operating state, but lacked an operational gate requiring phase, direction, same-phase baseline, response tracking, and corroboration before labeling a magnitude abnormal.

### F3. Coherent task-phase behavior was missed (2 of 5 failures)

Two failures were coherent with a loosening/release-type phase, yet the responses diagnosed sensor, actuator, or control faults. They did not first ask whether reversal and load behavior matched the current sequence step, nor whether that step was expected at that point.

Attribution: Skill v1 mentioned sequence state but did not prioritize phase inference ahead of component-fault generation or distinguish a valid operation from an operation occurring out of order.

### F4. Multivariate evidence was not translated into a physical fault category (3 distinct categories represented)

The failed set required reasoning from relationships among commands, feedback tracking, load/contact, external motion, workpiece behavior, and device activation. The missed categories included an unexpected assembly/interference condition, external mounting/support instability despite coherent robot motion, and a commanded-function activation failure while the main trajectory remained coherent.

Attribution: Skill v1 asked for corroboration but supplied no cross-channel signature library or functional-boundary hierarchy to convert correlated evidence into physical categories.

### F5. Bare option letters concealed or omitted the diagnosis (2 of 5 failures)

Two responses returned only a letter. One bare letter matched the failed case's reference letter but was still marked incorrect against the semantic root-cause field; the other selected the wrong letter. In both cases, the output contained no explicit diagnosis that could be evaluated or inspected semantically.

Attribution: Skill v1's strict format rule allowed the model to suppress the diagnosis entirely when a prompt requested letter-only output.

### F6. Corrective actions followed the symptom, not the causal boundary (3 narrative failures)

The narrative failures recommended reset/calibration/general actuator inspection instead of phase validation and sequence correction, or safe retraction and removal of the physical interference. These actions inherited the earlier attribution error.

Attribution: Skill v1 required cause-targeted action in principle but did not map corrective action to the identified broken boundary or allow “no repair” when a phase is intended and coherent.

## Exact rationale for every Skill v2 modification

### M1. Version changed from 0.1.0 to 0.2.0

Rationale: records a behavior-changing revision while preserving Skill v1 separately. This is revision bookkeeping, not a diagnostic rule.

### M2. Core Rules 3–6 rewritten around phase, signal semantics, and causal level

Added requirements to determine task phase before interpreting signs; treat target torque/current, control output, and actuator commands as intent; require a contextual baseline or corroborating response before calling a magnitude abnormal; and prefer physical/sequence causes over sensor-centric paraphrases.

Trace: F1, F2, and F3. This directly blocks the three torque/sensor descriptions and the two missed phase interpretations.

### M3. Core output rule changed from strict letter-only compliance to explicit semantic diagnosis

Added a rule forbidding bare letters. Choice-based answers must pair the selected letter with the diagnosis; absent choices require an inferred diagnosis rather than a guessed letter.

Trace: F5. This makes the causal classification observable and semantically evaluable. No letter-to-label mapping was encoded.

### M4. Added the “Signal-role and phase gate”

Added four signal roles—intent, response, interaction/load, and context/phase—and a fixed question order ending with identification of the first broken boundary. Added a prohibition on calling signed values excessive without baseline/limit/corroboration. Added recognition that coherent reversal plus matching command and feedback often denotes a release/retract/unwind/loosening-type phase whose abnormality depends on sequence position.

Trace: F1, F2, and F3. This turns general state-awareness into an executable pre-diagnosis gate.

### M5. Added reusable multivariate physical signatures

Added general signatures for:

- command without expected device effect → activation/command-path failure;
- close joint tracking plus unstable external/workpiece evidence → mounting/support/workholding instability;
- tracked motion plus premature or misplaced contact/load → obstruction/interference/unexpected assembly condition;
- coherent reversal across intent, response, and load → task-phase behavior;
- one discordant reading amid otherwise coherent evidence → sensor/signal-path fault.

Trace: F4 and F1. These are category-level relations, not memorized case answers, IDs, values, or option labels.

### M6. Added a diagnostic priority in competing-hypothesis analysis

The new order is phase/intent, command path, mechanical interaction, actuator/control, then measurement. Sensor calibration now requires independent disagreement or implausible discontinuity; surprising values alone are insufficient.

Trace: F1–F4. This prevents premature sensor/controller hypotheses and forces sequence and mechanical explanations to be tested first when their predictions fit.

### M7. Corrective-action guidance now follows the broken boundary

Added boundary-specific action classes: sequence correction for unintended phase, command/device checks for activation failure, stabilization for external support/workholding faults, and safe retraction/removal for premature interference. Added “no repair indicated” when the phase is intended and completes coherently.

Trace: F3, F4, and F6. This replaces generic reset/calibration advice with actions causally matched to the diagnosis.

### M8. Answer Pattern strengthened

The answer must now state the relevant phase, one explicit diagnosis, and two or three decisive cross-channel observations including command-response and phase evidence. Choice answers use `LETTER — <explicit diagnosis>` rather than a bare letter.

Trace: F1, F3, F4, and F5. This makes the final response reveal the physical inference instead of stopping at a signal description or letter.

### M9. Pitfalls and Verification Checklist expanded

Added explicit checks against treating command targets as measured faults, treating negative values as inherently excessive, missing release/loosening phases, missing external instability or device activation faults when robot tracking is normal, choosing a sensor without independent disagreement, and returning bare letters.

Trace: F1–F5. These checks reinforce each newly observed failure mode at the final-answer gate.

## Preserved Skill v1 structure

Skill v2 retains Skill v1's evidence-first workflow: task parsing, intended-state reconstruction, observation extraction, anomaly comparison, causal-chain construction, competing hypotheses, confidence calibration, safe corrective action, pitfalls, and verification. Changes were limited to gaps demonstrated by the five Round 1 failures.

## Evaluation status

No external evaluation was run. Skill v2 is ready for the separately controlled Round 2 harness.