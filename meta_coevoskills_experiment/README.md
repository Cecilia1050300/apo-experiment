# Meta-CoEvoSkills FactoryBench L1-L3

Controlled A/B shared-Skill optimization experiment. Previous experiment directories are read-only references. FactoryBench Level 4 is excluded.

Start with:

`python meta_coevoskills_experiment/scripts/run_experiment.py --dry-run --smoke --arm surrogate --rounds 1`

API-facing configuration is explicit: target `gpt-4o-mini`; verifier `gpt-5.6-luna` with `reasoning.mode=standard` and `reasoning.effort=high`; rewriter `gpt-5.6-luna` with `reasoning.mode=pro` and `reasoning.effort=xhigh`. The local preflight validates these exact IDs and settings without making an API call.
