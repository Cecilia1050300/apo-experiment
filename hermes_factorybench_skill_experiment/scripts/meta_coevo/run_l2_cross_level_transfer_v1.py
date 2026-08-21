#!/usr/bin/env python3
import argparse, hashlib, importlib.util, json, math, random, sys
from collections import Counter
from pathlib import Path

ROOT = Path("/home/training/automatic_prompt_engineer/hermes_factorybench_skill_experiment")
FB_ROOT = Path("/home/training/automatic_prompt_engineer/FactoryBench")
ARM_F_SCRIPT = ROOT / "scripts/meta_coevo/run_format_stratified_loocv.py"
MANIFEST_DIR = ROOT / "data_manifests/meta_m1"
L2_MANIFEST = MANIFEST_DIR / "factorybench_l2_cross_level_test_v1.json"
L2_LOCK = MANIFEST_DIR / "factorybench_l2_cross_level_test_v1_lock.json"
FROZEN_SKILL = ROOT / "prompts/adapters/meta_coevo_format_loocv_f1/m1_factorybench_l123/selected_adapter.json"
EXPECTED_SKILL_SHA256 = "253044ed73fa651337dcc86bbae797c227a65e37523b19195b74ddcf7efc2af1"
OLD_MANIFESTS = [
    MANIFEST_DIR / "factorybench_l123_dev_fold_a.json",
    MANIFEST_DIR / "factorybench_l123_dev_fold_b.json",
    MANIFEST_DIR / "factorybench_l123_holdout.json",
]
TARGETS = {"four_letter_tf": 20, "scalar_range": 30}
SELECTION_SEED = "factorybench-l2-cross-level-v1-50cases-20260821"
RESULT_DIR = ROOT / "results/meta_coevo/cross_level_l1_to_l2_v1"
SUMMARY_PATH = RESULT_DIR / "factorybench_l2_cross_level_test_v1_summary.json"
N_REPEATS = 5
BOOTSTRAP_SAMPLES = 10000
BOOTSTRAP_SEED = 20260821
EPS = 1e-12

def read_json(p):
    return json.loads(p.read_text(encoding="utf-8"))

def write_json_new(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        raise RuntimeError(f"refusing overwrite: {p}")
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8")

def sha256(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def safe(v):
    return "unknown" if v is None else str(v)

def load_arm_f():
    spec = importlib.util.spec_from_file_location("arm_f_l2_transfer", ARM_F_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    mod.ARM = "cross_level_l1_to_l2_v1"
    mod.RESULT_DIR = RESULT_DIR
    mod.TRACE_DIR = ROOT / "traces/meta_coevo/cross_level_l1_to_l2_v1"
    mod.N_REPEATS = N_REPEATS
    return mod

def get_base():
    sys.path.insert(0, str(FB_ROOT))
    sys.path.insert(0, str(ROOT))
    from scripts.meta_coevo import run_static_surrogate as base
    return base

def prior_used_ids():
    used = set()
    for p in OLD_MANIFESTS:
        d = read_json(p)
        used.update(r["id"] for r in d["items"])
    return used

def build():
    if L2_MANIFEST.exists() or L2_LOCK.exists():
        raise RuntimeError("L2 cross-level v1 already exists; do not overwrite a frozen test set.")
    if sha256(FROZEN_SKILL) != EXPECTED_SKILL_SHA256:
        raise RuntimeError("Frozen L1 Skill SHA changed.")

    base = get_base()
    from factorybench import load_split

    used = prior_used_ids()
    items = load_split(2, split="test", revision=base.REV, max_items=None)
    unseen = [x for x in items if x.id not in used]
    counts = Counter(x.answer_format.value for x in unseen)

    print("===== BUILD L1 -> L2 CROSS-LEVEL TEST V1 =====")
    print("revision:", base.REV)
    print("L2 total:", len(items))
    print("L2 unseen:", len(unseen))
    print("format counts:")
    for k,v in sorted(counts.items()):
        print(f"  {k}: {v}")

    for fmt,n in TARGETS.items():
        if counts.get(fmt,0) < n:
            raise RuntimeError(f"not enough {fmt}: {counts.get(fmt,0)} < {n}")

    def dataset_of(x): return safe(getattr(x, "dataset", None))
    def template_of(x): return getattr(x, "template_id", None)
    def episode_of(x): return (getattr(x, "provenance", None) or {}).get("episode")
    def tie_hash(x): return hashlib.sha256(f"{SELECTION_SEED}|{x.id}".encode()).hexdigest()

    def select_diverse(pool, n):
        chosen, rem = [], list(pool)
        ds_seen, tp_seen, ep_seen = set(), set(), set()
        while rem and len(chosen) < n:
            ranked = []
            for x in rem:
                ds,tp,ep = dataset_of(x),template_of(x),episode_of(x)
                novelty = (int(ds not in ds_seen), int(tp not in tp_seen), int(ep not in ep_seen))
                ranked.append((-novelty[0],-novelty[1],-novelty[2],tie_hash(x),x))
            ranked.sort(key=lambda z: z[:4])
            x = ranked[0][4]
            chosen.append(x)
            ds_seen.add(dataset_of(x)); tp_seen.add(template_of(x)); ep_seen.add(episode_of(x))
            rem = [y for y in rem if y.id != x.id]
        if len(chosen) != n:
            raise RuntimeError(f"selection failed for {n}")
        return chosen

    selected = []
    for fmt,n in TARGETS.items():
        pool = [x for x in unseen if x.answer_format.value == fmt]
        chosen = select_diverse(pool, n)
        selected.extend(chosen)
        print(f"selected {fmt}: {len(chosen)} / {len(pool)}")

    rows = []
    for x in selected:
        rows.append({
            "id": x.id, "level": 2, "split": "test",
            "episode": episode_of(x),
            "answer_format": x.answer_format.value,
            "dataset": getattr(x, "dataset", None),
            "template_id": template_of(x),
        })

    ext_ids = {r["id"] for r in rows}
    if ext_ids & used:
        raise RuntimeError("data leakage detected")

    manifest = {
        "name": "factorybench_l2_cross_level_test_v1",
        "purpose": "Test exact frozen L1 Skill on L2 cases of the same active formats.",
        "dataset_revision": base.REV,
        "frozen_after_creation": True,
        "selection_policy": {
            "source_skill_level": 1,
            "evaluation_level": 2,
            "split": "test",
            "same_active_formats_only": list(TARGETS),
            "target_counts": TARGETS,
            "selection_method": "deterministic metadata-diversity only; no model outputs or GT performance used",
            "tie_break_seed": SELECTION_SEED,
        },
        "items": rows,
    }
    write_json_new(L2_MANIFEST, manifest)
    msha = sha256(L2_MANIFEST)
    ssha = sha256(FROZEN_SKILL)
    lock = {
        "manifest_sha256": msha,
        "skill_sha256": ssha,
        "dataset_revision": base.REV,
        "case_count": len(rows),
        "format_counts": dict(sorted(Counter(r["answer_format"] for r in rows).items())),
        "no_l2_skill_evolution": True,
    }
    write_json_new(L2_LOCK, lock)

    print("\n===== FROZEN =====")
    print("cases:", len(rows))
    print("format counts:", lock["format_counts"])
    print("manifest sha256:", msha)
    print("skill sha256:", ssha)
    print("Do not modify Skill or L2 manifest after this point.")

def validate():
    if not L2_MANIFEST.exists() or not L2_LOCK.exists():
        raise RuntimeError("run --phase build first")
    lock = read_json(L2_LOCK)
    if sha256(L2_MANIFEST) != lock["manifest_sha256"]:
        raise RuntimeError("L2 manifest changed after freeze")
    if sha256(FROZEN_SKILL) != lock["skill_sha256"]:
        raise RuntimeError("Frozen L1 Skill changed after freeze")
    if lock["skill_sha256"] != EXPECTED_SKILL_SHA256:
        raise RuntimeError("Skill does not match original frozen Arm-F Skill")
    d = read_json(L2_MANIFEST)
    counts = Counter(r["answer_format"] for r in d["items"])
    if dict(counts) != TARGETS:
        raise RuntimeError(f"format composition changed: {dict(counts)}")
    skill = read_json(FROZEN_SKILL)
    rules = skill.get("rules") or []
    active = sorted({r["answer_format"] for r in rules})
    if set(active) != set(TARGETS):
        raise RuntimeError(f"Skill active formats changed: {active}")
    return {"lock": lock, "rules": rules, "active_formats": active}

def load_pool(base):
    manifest, items = base.source_items(L2_MANIFEST)
    split_lookup = {r["id"]: r["split"] for r in manifest["items"]}
    return [{"fold":"CROSS_LEVEL_L2_V1","split":split_lookup[x.id],"item":x} for x in items]

def effective_score(r):
    clean = r.get("parse_error") is None and isinstance(r.get("score"), (int,float)) and math.isfinite(float(r["score"]))
    return float(r["score"]) if clean else 0.0

def case_table(result):
    t = {}
    for run in result["runs"]:
        for r in run["items"]:
            rec = t.setdefault(r["id"], {
                "answer_format": r["answer_format"],
                "chance": float(r.get("chance",0.0)),
                "scores": [],
                "parse_failures": 0,
            })
            rec["scores"].append(effective_score(r))
            rec["parse_failures"] += int(r.get("parse_error") is not None)
    for cid,rec in t.items():
        if len(rec["scores"]) != N_REPEATS:
            raise RuntimeError(f"{cid}: repeat mismatch")
        rec["mean_effective_score"] = sum(rec["scores"])/len(rec["scores"])
    return t

def fixed(case_ids, table):
    scores = [table[c]["mean_effective_score"] for c in case_ids]
    chances = [table[c]["chance"] for c in case_ids]
    ms = sum(scores)/len(scores); mc = sum(chances)/len(chances)
    return (ms-mc)/(1-mc)

def percentile(vals,q):
    vals = sorted(vals)
    pos=(len(vals)-1)*q; lo=math.floor(pos); hi=math.ceil(pos)
    if lo==hi: return vals[lo]
    w=pos-lo
    return vals[lo]*(1-w)+vals[hi]*w

def bootstrap_delta(bt, ct, ids, seed):
    pb, pc = fixed(ids,bt), fixed(ids,ct)
    rng = random.Random(seed)
    ds=[]
    n=len(ids)
    for _ in range(BOOTSTRAP_SAMPLES):
        sample=[ids[rng.randrange(n)] for _ in range(n)]
        ds.append(fixed(sample,ct)-fixed(sample,bt))
    lo,hi=percentile(ds,0.025),percentile(ds,0.975)
    return {
        "baseline_score": pb,
        "candidate_score": pc,
        "mean_delta": pc-pb,
        "ci_95": [lo,hi],
        "ci_excludes_zero_positive": lo>0,
        "bootstrap_probability_delta_gt_0": sum(x>0 for x in ds)/len(ds),
        "case_count": len(ids),
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "pairing_unit": "case_id",
    }

def parse_total(result):
    return sum(int(r.get("parse_error") is not None) for run in result["runs"] for r in run["items"])

def preflight():
    v=validate()
    out={
        "status":"READY",
        "experiment":"l1_to_l2_cross_level_transfer_v1",
        "source_skill_level":1,
        "evaluation_level":2,
        "case_count":v["lock"]["case_count"],
        "format_counts":v["lock"]["format_counts"],
        "active_formats":v["active_formats"],
        "manifest_sha256":v["lock"]["manifest_sha256"],
        "skill_sha256":v["lock"]["skill_sha256"],
        "repeats_per_condition":N_REPEATS,
        "bootstrap_samples":BOOTSTRAP_SAMPLES,
        "l2_skill_evolution_allowed":False,
        "external_feedback_allowed":False,
    }
    print(json.dumps(out,indent=2,ensure_ascii=False))

def run():
    if SUMMARY_PATH.exists():
        raise RuntimeError(f"completed result already exists: {SUMMARY_PATH}")
    v=validate()
    arm_f=load_arm_f()
    base=arm_f.load_base()
    arm_f.RESULT_DIR=RESULT_DIR
    arm_f.N_REPEATS=N_REPEATS
    pool=load_pool(base)
    if len(pool)!=50:
        raise RuntimeError(f"expected 50 cases, got {len(pool)}")
    from openai import OpenAI
    client=OpenAI()

    print("Running L2 baseline x5 ...")
    baseline=arm_f.evaluate_repeated_records(base,client,records=pool,rules=[],condition="l2_cross_level_baseline")
    print("Running SAME frozen L1 Skill on L2 x5 ...")
    cand=arm_f.evaluate_repeated_records(base,client,records=pool,rules=v["rules"],condition="l2_cross_level_frozen_l1_skill")

    if baseline["ordered_ids"] != cand["ordered_ids"]:
        raise RuntimeError("baseline/candidate IDs differ")

    bt,ct=case_table(baseline),case_table(cand)
    ids=baseline["ordered_ids"]
    combined=bootstrap_delta(bt,ct,ids,BOOTSTRAP_SEED)
    byfmt={}
    for i,fmt in enumerate(TARGETS,1):
        fids=[cid for cid in ids if bt[cid]["answer_format"]==fmt]
        byfmt[fmt]=bootstrap_delta(bt,ct,fids,BOOTSTRAP_SEED+i)

    bp,cp=parse_total(baseline),parse_total(cand)
    combined_pos=combined["mean_delta"]>EPS
    all_nonreg=all(x["mean_delta"]>=-EPS for x in byfmt.values())
    any_gain=any(x["mean_delta"]>EPS for x in byfmt.values())
    parse_safe=cp<=bp
    passed=combined_pos and all_nonreg and any_gain and parse_safe
    ci_positive=combined["ci_excludes_zero_positive"]

    summary={
        "experiment":"l1_to_l2_cross_level_transfer_v1",
        "status":"COMPLETE",
        "research_question":"Does the exact frozen L1 Skill transfer to L2 cases with the same active formats?",
        "source_skill_level":1,
        "evaluation_level":2,
        "case_count":len(ids),
        "frozen_artifacts":{
            "l1_skill_sha256":v["lock"]["skill_sha256"],
            "l2_manifest_sha256":v["lock"]["manifest_sha256"],
            "active_formats":v["active_formats"],
            "rule_count":len(v["rules"]),
        },
        "protocol_integrity":{
            "l2_rule_generation":False,
            "l2_rule_selection":False,
            "l2_skill_revision":False,
            "surrogate_called_on_l2":False,
            "l2_feedback_used":False,
        },
        "baseline":baseline,
        "frozen_l1_skill_on_l2":cand,
        "case_level_bootstrap":{"combined":combined,"by_format":byfmt},
        "parse_failures":{"baseline_total":bp,"candidate_total":cp,"non_regression_pass":parse_safe},
        "frozen_success_criteria":{
            "combined_mean_delta_gt_0":combined_pos,
            "every_active_format_mean_delta_ge_0":all_nonreg,
            "at_least_one_active_format_mean_delta_gt_0":any_gain,
            "parse_failures_do_not_increase":parse_safe,
            "cross_level_transfer_pass":passed,
        },
        "statistical_support":{
            "combined_bootstrap_95ci_excludes_zero_positive":ci_positive
        }
    }
    write_json_new(SUMMARY_PATH,summary)

    print("\n===== RESULT =====")
    print("combined:", combined)
    print("by_format:", json.dumps(byfmt,indent=2))
    print("parse:", {"baseline":bp,"candidate":cp})
    print("cross_level_transfer_pass =", passed)
    print("bootstrap_95ci_positive =", ci_positive)
    print("summary:", SUMMARY_PATH)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--phase",choices=["build","preflight","run"],required=True)
    a=ap.parse_args()
    if a.phase=="build": build()
    elif a.phase=="preflight": preflight()
    else: run()

if __name__=="__main__":
    main()
