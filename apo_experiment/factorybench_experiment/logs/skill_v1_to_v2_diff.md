# Skill v1 to v2 Change Summary

## Metadata

- Version: `0.1.0` → `0.2.0`.
- Scope remains FactoryBench Level 4 troubleshooting only.

## Behavioral changes

1. **Phase-first interpretation:** task phase, event/sequence context, command direction, and setpoint transitions must be resolved before judging torque or force.
2. **Signal-role separation:** signals are classified as intent, response, interaction/load, or context/phase.
3. **No magnitude guessing:** signed or large torque/force is not called abnormal without a phase-relative baseline, limit evidence, or corroborating response failure.
4. **Functional-boundary diagnosis:** identify whether the first break is program-to-command, command-to-motion, motion-to-contact, support/workholding, or measurement.
5. **Multivariate fault categories:** added reusable signatures for activation failure, external mounting/workholding instability, obstruction/unexpected assembly condition, coherent task-phase behavior, and genuine sensor-path inconsistency.
6. **Hypothesis priority:** test phase/intent, command path, mechanical interaction, and actuator/control before measurement fault.
7. **Cause-matched actions:** corrective action now follows the broken boundary; coherent intended phases may require verification rather than repair.
8. **Explicit outputs:** bare option letters are prohibited; choice answers must include the semantic diagnosis.
9. **Final checks:** pitfalls and verification now explicitly cover the Round 1 torque, phase, multivariate-category, sensor-attribution, activation, mounting, and output failures.

## Preserved from v1

The evidence-first structure, observation/inference separation, causal-chain reasoning, alternative comparison, confidence calibration, safety guidance, and prohibition on unsupported assumptions remain intact.

## Data boundary

The changes derive only from the five optimization failed cases and Skill v1. No held-out or final-test material was used, and no evaluation was run.