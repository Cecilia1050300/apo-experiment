
今天 上午10:06
我現在要直接用這篇paper提供的github 直接先復刻他的實驗
Zhang-Henry/CoEvoSkills.git

我的設備是linux 的遠端server 請根據這個條件給我指令。
第一步驟我要先離開我個專案資料夾 再開一個做實驗嗎，還是舊統一做再我apo experience的file 再開子file


已貼上文字 (1)(20260820-030933).txt
文件


已貼上文字 (1)(20260820-031301).txt
文件

a80d8f53-410c-470f-9481-2cefb4fd7051.png
你現在如果要開始實作，我建議下一步不是寫 prompt，而是先讓 Codex 只盤點 meta_coevoskills_experiment/ 現有 runner、split、GT evaluator、Skill update、results schema，然後我們再決定三個 Arm 最小需要插在哪些 function。這樣可以避免又開一套重複 harness。

可以線來做這個 但是我的file放在遠端server 並且codex目前用量已滿 我可以自己手工給

所以我現在要回去哪一個資料夾執行接下來的實驗



已貼上文字 (1)(20260820-071407).txt
文件


已貼上文字 (1)(20260820-071817).txt
文件

好，那下一步就先做 Arm B：Static Surrogate，先不要碰 verifier evolution。

你可以先建立第一版 verifier prompt：

cd ~/automatic_prompt_engineer/hermes_factorybench_skill_experiment

cat > prompts/meta_coevo/surrogate_verifier_v0.txt <<'EOF'
You are an independent surrogate verifier for FactoryBench Level 1-3 tasks.

Your job is to evaluate whether the agent answer is likely correct based only on:
- the task question
- the required answer format
- the agent answer
- the verifier rules below

You MUST NOT use or infer any hidden reference answer, GT label, evaluator score, or holdout information.

Shared verification rules:
1. Check whether the answer follows the required output format.
2. Check whether the requested temporal window is interpreted correctly.
3. Check whether the requested operator is applied correctly, including mean, max, min, sum, absolute value, ranking, comparison, and threshold operations.
4. Check whether threshold direction and boundary conditions are handled correctly.
5. Check whether the answer makes unsupported assumptions not justified by the task input.
6. Check for arithmetic or logical inconsistencies that can be verified from the supplied task.
7. For multi-element outputs, check each element independently when possible.

Return JSON only:

{
  "verdict": "PASS" | "FAIL",
  "confidence": 0.0,
  "diagnosis": [
    "brief reason"
  ],
  "failed_checks": [
    "name of failed verification rule"
  ]
}

Decision rule:
- PASS only if there is no identifiable correctness or format issue.
- FAIL if any identifiable issue could make the final answer incorrect.
- If uncertain, do not invent a GT answer. Explain the uncertainty in diagnosis.
EOF
然後確認：

cat prompts/meta_coevo/surrogate_verifier_v0.txt
第一版我們先二值化：

Surrogate:
PASS / FAIL

GT:
score == 1.0 且 parse_error == None → PASS
其他 → FAIL
這樣 smoke test 最單純。

接下來我們要改的是：

scripts/meta_coevo/run_coevo.py
但第一階段只新增 Static Surrogate，不修改 Skill optimizer、不修改 verifier 本身。

目標流程：

FactoryBench item
    ↓
Adapter
    ↓
Agent Answer
    ↓
Static Surrogate v0
    ↓
_score_one() GT
    ↓
保存：
surrogate_verdict
gt_verdict
false_pass
false_reject
先確認能正常跑，再進 Arm C。

你建好 surrogate_verifier_v0.txt 後，下一步我可以直接給你 run_coevo.py 要加的第一段 Python code。


好

好，接下來直接改 scripts/meta_coevo/run_coevo.py，先只做到 Arm B：Static Surrogate。

先備份：

cd ~/automatic_prompt_engineer/hermes_factorybench_skill_experiment

cp scripts/meta_coevo/run_coevo.py \
   scripts/meta_coevo/run_coevo.py.bak
然後先在檔案上方常數區加入：

SURROGATE_PATH = ROOT / "prompts/meta_coevo/surrogate_verifier_v0.txt"
再加入這個 function：

def run_surrogate(client, item, raw_output):
    system = SURROGATE_PATH.read_text(encoding="utf-8")

    payload = {
        "question": render_prompt(item),
        "answer_format": item.answer_format.value,
        "agent_answer": raw_output,
    }

    prompt = json.dumps(payload, ensure_ascii=False, indent=2)

    raw, u, lat, err = call_chat(
        client,
        system,
        prompt,
    )

    parsed = None
    parse_error = None

    if err is None:
        try:
            parsed = json.loads(raw)
        except Exception as exc:
            parse_error = f"surrogate_json_error: {type(exc).__name__}: {exc}"

    if parsed is not None:
        verdict = parsed.get("verdict")
        if verdict not in {"PASS", "FAIL"}:
            parse_error = "invalid_surrogate_verdict"
    else:
        verdict = None

    return {
        "raw_output": raw,
        "parsed": parsed,
        "verdict": verdict,
        "parse_error": parse_error,
        "transport_error": err,
        "usage": u,
        "latency_seconds": lat,
    }
接著新增 GT status function：

def gt_pass_from_score(score, parse_error):
    if parse_error is not None:
        return False

    if score is None:
        return False

    try:
        value = float(score)
    except Exception:
        return False

    return math.isfinite(value) and value == 1.0
然後修改 evaluate_l123() 裡這段。

原本：

for item,(raw,u,lat,err) in zip(items,calls):
    s=_score_one(item,raw)
    finite_score=s.score if isinstance(s.score,(int,float)) and math.isfinite(float(s.score)) else None
    rows.append({...})
先改成比較好讀的版本：

for item, (raw, u, lat, err) in zip(items, calls):

    # 1. Surrogate must run BEFORE seeing GT.
    surrogate = run_surrogate(
        client=client,
        item=item,
        raw_output=raw,
    )

    # 2. Hidden GT oracle.
    s = _score_one(item, raw)

    finite_score = (
        s.score
        if isinstance(s.score, (int, float))
        and math.isfinite(float(s.score))
        else None
    )

    gt_pass = gt_pass_from_score(
        finite_score,
        s.parse_error or (
            "non_finite_score"
            if finite_score is None
            else None
        ),
    )

    surrogate_verdict = surrogate.get("verdict")

    false_pass = (
        surrogate_verdict == "PASS"
        and not gt_pass
    )

    false_reject = (
        surrogate_verdict == "FAIL"
        and gt_pass
    )

    rows.append({
        "id": item.id,
        "level": item.level,
        "split": next(
            r["split"]
            for r in m["items"]
            if r["id"] == item.id
        ),
        "dataset": item.dataset,
        "episode": item.provenance.get("episode"),
        "answer_format": item.answer_format.value,

        "raw_output": raw,
        "parsed": s.parsed,
        "score": finite_score,
        "chance": s.chance,

        "parse_error": s.parse_error or (
            "non_finite_score"
            if finite_score is None
            else None
        ),

        "gt_pass": gt_pass,

        "surrogate": surrogate,
        "surrogate_verdict": surrogate_verdict,

        "false_pass": false_pass,
        "false_reject": false_reject,

        "transport_error": err,
        "usage": u,
        "latency_seconds": lat,

        "development_evidence": {
            "rendered_input": render_prompt(item),
            "reference_answer": item.answer,
        } if part == "development" else None,
    })
然後在 payload 裡增加 aggregate 統計。

原本有：

"parse_failures": sum(
    r["parse_error"] is not None
    for r in rows
),
旁邊加：

"surrogate_valid_count": sum(
    r["surrogate_verdict"] in {"PASS", "FAIL"}
    for r in rows
),

"surrogate_parse_failures": sum(
    r["surrogate"].get("parse_error") is not None
    or r["surrogate"].get("transport_error") is not None
    for r in rows
),

"gt_full_pass_count": sum(
    r["gt_pass"]
    for r in rows
),

"false_pass_count": sum(
    r["false_pass"]
    for r in rows
),

"false_reject_count": sum(
    r["false_reject"]
    for r in rows
),
再加 rate：

"false_pass_rate": (
    sum(r["false_pass"] for r in rows) / len(rows)
    if rows else None
),

"false_reject_rate": (
    sum(r["false_reject"] for r in rows) / len(rows)
    if rows else None
),
現在 Arm B 的輸出就會開始包含：

GT performance
+
Surrogate performance
+
False Pass
+
False Reject
但此時：

Verifier v0
完全固定不變，所以它仍然是 Static Surrogate。

改完先不要直接花 API 費用跑。

先做 syntax check：

cd ~/automatic_prompt_engineer/hermes_factorybench_skill_experiment

python -m py_compile \
  scripts/meta_coevo/run_coevo.py
如果沒輸出，代表 syntax OK。

再確認 prompt 存在：

test -f prompts/meta_coevo/surrogate_verifier_v0.txt \
  && echo "surrogate prompt OK"
最後再：

grep -nE \
"run_surrogate|gt_pass_from_score|false_pass|false_reject|SURROGATE_PATH" \
scripts/meta_coevo/run_coevo.py
把這三個結果貼給我。

下一步我會先幫你檢查 是否有 GT leakage，確認安全後才讓你真正跑第一個 Arm B smoke。



貼上的程式碼 (1)。py
檔案
是改run_coevo.py.bak對齁
要不你直接幫我改一個新的

不是改 .bak。.bak 是備份，應該保持不動。

你現在這個檔案其實已經有兩個明顯問題：SURROGATE_PATH 被放在 ROOT 定義之前，而且 run_surrogate() 放在 main() 呼叫之後；我幫你直接整理成一個新的 Arm B / Static Surrogate 版本了。你原本上傳的檔案確實有這些位置問題。


新的檔案在這裡：

下載 run_static_surrogate.py

我順便做了幾個重要修正：

1. 不動 meta_m1/run_m1.py
   → 保留 Arm A Control

2. 不動 run_coevo.py.bak
   → 純備份

3. 新檔：
   run_static_surrogate.py
   → Arm B Static Surrogate

4. 結果獨立存到：
   results/meta_coevo/static/

5. trace 獨立存到：
   traces/meta_coevo/static/

6. Adapter 獨立存到：
   prompts/adapters/meta_coevo_static/

7. Surrogate 在 _score_one() GT Oracle 之前執行
   → 避免 GT leakage

8. 保存：
   gt_pass
   surrogate_verdict
   false_pass
   false_reject
   surrogate diagnosis
   failed_checks
   agreement rate

9. Static Surrogate diagnosis 會進 M1 optimizer input
   → 所以這才真正比較接近 Arm B
   → 不是只有「旁邊多算一個 verifier 指標」

10. 此版本只允許：
我也已經做過 Python syntax check，檔案可以 compile。

你下載後放到遠端：

cd ~/automatic_prompt_engineer/hermes_factorybench_skill_experiment

# 把下載的 run_static_surrogate.py 放到這裡
scripts/meta_coevo/run_static_surrogate.py
然後先跑：

python -m py_compile \
  scripts/meta_coevo/run_static_surrogate.py
再確認：

grep -nE \
"SURROGATE_PATH|run_surrogate|gt_pass_from_score|false_pass|false_reject|meta_coevo/static" \
scripts/meta_coevo/run_static_surrogate.py
先不要真的打 API。

你確認這兩個 command 都正常後，我下一步會帶你做一個 preflight / dry inspection，確認：

沒有偷讀 reference_answer

Holdout 不會拿去更新 Skill

Static Verifier 真的固定不變

不會覆蓋 Arm A 的結果

確認完才跑第一個 Arm B development smoke。


run_static_surrogate.py
程式碼


檔案庫
/
run_static_surrogate.py


#!/usr/bin/env python3
"""Arm B smoke: FactoryBench L1-L3 with a fixed independent surrogate verifier."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from openai import OpenAI
from factorybench.cost import compute_cost_from_usage
from factorybench.data import load_split
from factorybench.evaluate import _score_one
from factorybench.prompt import render_prompt

REPO = Path('/home/training/automatic_prompt_engineer')
ROOT = REPO / 'hermes_factorybench_skill_experiment'
SURROGATE_PATH = ROOT / 'prompts/meta_coevo/surrogate_verifier_v0.txt'
ARM = 'surrogate_static'
MODEL = 'gpt-5.5'
REV = 'b3863519ccedbceab54dfa7600104eb42b985ed7'
M1_PATH = ROOT / 'prompts/meta/manufacturing_meta_prompt_m1_candidate.txt'
M1_SHA = '78187e3268294657d2398c9a79563a36f050c4189b2f6650cc569407512cb052'
MANIFEST_DIR = ROOT / 'data_manifests/meta_m1'
RESULT_DIR = ROOT / 'results/meta_coevo/static'
TRACE_DIR = ROOT / 'traces/meta_coevo/static'
ADAPTER_ROOT = ROOT / 'prompts/adapters/meta_coevo_static'
M1_FIELDS = {'meta_prompt_version','task_name','mode','decision','task_scope','supported_subtasks','supported_formats','applicability_conditions','failure_taxonomy','adapter_text','fallback_policy','changes_from_previous','predicted_regression_risks','evidence_limitations'}
L4_SCHEMA = '''Return JSON only, with exactly this structure:
{
  "root_cause": "the most likely underlying physical or operational cause",
  "evidence": ["specific signal or task-phase evidence from the input"],
  "corrective_actions": ["corrective action and a verification step"]
}
Do not return Markdown, headings, a bare option letter, or a dataset label.'''
L4_BASE_INSTRUCTIONS = f'''You are solving a FactoryBench Level 4 industrial troubleshooting case.

Use the machine description, signal mapping, time series, and task question in the user input. Distinguish an observed symptom from the underlying physical or operational root cause. Ground every evidence item in the supplied input. Do not invent signals, machine specifications, fault documentation, or SOPs. If the supplied telemetry does not uniquely identify a cause, say so in the evidence array and return the most defensible diagnosis supported by the input.

{L4_SCHEMA}

No reusable diagnostic Adapter, shared Core prompt, RAG passage, fault catalog, signal-analysis tool, gold root cause, or reference answer is supplied unless an Adapter condition explicitly follows.'''


def sha(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()
def write_new(path: Path, data: bytes) -> None:
    if path.exists():
        if path.read_bytes() != data: raise RuntimeError(f'refusing overwrite: {path}')
        return
    path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(data)
def write_json(path: Path, obj: Any) -> None: write_new(path,(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode())
def load(path: Path) -> Any: return json.loads(path.read_text(encoding='utf-8'))

def evaluator_module():
    path=REPO/'apo_experiment/factorybench_experiment/factorybench_evaluator_smoke_test.py'
    spec=importlib.util.spec_from_file_location('m1_l4_evaluator',path); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod
EVALUATOR=evaluator_module()

def usage(response: Any) -> dict[str,int]:
    u=getattr(response,'usage',None)
    return {'input_tokens':int(getattr(u,'input_tokens',getattr(u,'prompt_tokens',0)) or 0),'output_tokens':int(getattr(u,'output_tokens',getattr(u,'completion_tokens',0)) or 0)}

class ResponseProxy:
    def __init__(self, client: OpenAI): self._client=client; self.ledger=[]; self.responses=self
    def create(self, **kwargs):
        response=self._client.responses.create(**kwargs); self.ledger.append({'model':kwargs.get('model',MODEL),**usage(response)}); return response


def manifests(task:str):
    prefix='factorybench_l123' if task=='m1_factorybench_l123' else 'factorybench_l4'
    return {k:MANIFEST_DIR/f'{prefix}_{v}.json' for k,v in {'fold_a':'dev_fold_a','fold_b':'dev_fold_b','holdout':'holdout'}.items()}
def adapter_dir(task:str): return ADAPTER_ROOT/task
def result_path(task,part,condition): return RESULT_DIR/part/f'{task}_{condition}.json'

def source_items(path:Path):
    m=load(path); cache={}
    for row in m['items']:
        key=(row['level'],row['split'])
        if key not in cache: cache[key]={x.id:x for x in load_split(key[0],split=key[1],revision=REV,max_items=None)}
    items=[cache[(r['level'],r['split'])][r['id']] for r in m['items']]
    for row,item in zip(m['items'],items):
        if item.provenance.get('episode')!=row['episode'] or item.answer_format.value!=row['answer_format']: raise RuntimeError('manifest mismatch')
    return m,items

def fixed(rows):
    scores=[]; chances=[]
    for r in rows:
        clean=r.get('parse_error') is None and isinstance(r.get('score'),(int,float)) and math.isfinite(float(r['score']))
        scores.append(float(r['score']) if clean else 0.0); chances.append(float(r.get('chance',0)))
    mc=sum(chances)/len(chances); return (sum(scores)/len(scores)-mc)/(1-mc)
def canonical(rows):
    clean=[r for r in rows if r.get('parse_error') is None and isinstance(r.get('score'),(int,float)) and math.isfinite(float(r['score']))]
    if not clean:return None
    mc=sum(float(r.get('chance',0)) for r in clean)/len(clean); return (sum(float(r['score']) for r in clean)/len(clean)-mc)/(1-mc)
def grouped(rows,field):
    b={}
    for r in rows:b.setdefault(str(r.get(field) or 'unknown'),[]).append(r)
    return {k:fixed(v) for k,v in sorted(b.items())}

def call_chat(client,system,prompt):
    started=time.perf_counter()
    try:
        messages=[]
        if system:messages.append({'role':'system','content':system})
        messages.append({'role':'user','content':prompt})
        resp=client.chat.completions.create(model=MODEL,messages=messages,max_completion_tokens=8192)
        return resp.choices[0].message.content or '',usage(resp),time.perf_counter()-started,None
    except Exception as exc:return '',{'input_tokens':0,'output_tokens':0},time.perf_counter()-started,f'{type(exc).__name__}: {exc}'
def parallel(client,system,prompts):
    out=[None]*len(prompts)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futs={pool.submit(call_chat,client,system,p):i for i,p in enumerate(prompts)}
        for f,i in futs.items():out[i]=f.result()
    return out


def run_surrogate(client, item, raw_output):
    """Run the fixed v0 surrogate verifier without exposing any GT fields."""
    system = SURROGATE_PATH.read_text(encoding='utf-8')
    payload = {
        'question': render_prompt(item),
        'answer_format': item.answer_format.value,
        'agent_answer': raw_output,
    }
    raw, u, lat, err = call_chat(
        client,
        system,
        json.dumps(payload, ensure_ascii=False, indent=2),
    )

    parsed = None
    parse_error = None
    verdict = None
    if err is None:
        try:
            parsed = json.loads(raw.strip())
        except Exception as exc:
            parse_error = f'surrogate_json_error: {type(exc).__name__}: {exc}'

    if parsed is not None:
        if not isinstance(parsed, dict):
            parse_error = 'surrogate_output_not_object'
        else:
            verdict = parsed.get('verdict')
            if verdict not in {'PASS', 'FAIL'}:
                parse_error = 'invalid_surrogate_verdict'
                verdict = None

    return {
        'raw_output': raw,
        'parsed': parsed,
        'verdict': verdict,
        'diagnosis': parsed.get('diagnosis', []) if isinstance(parsed, dict) else [],
        'failed_checks': parsed.get('failed_checks', []) if isinstance(parsed, dict) else [],
        'confidence': parsed.get('confidence') if isinstance(parsed, dict) else None,
        'parse_error': parse_error,
        'transport_error': err,
        'usage': u,
        'latency_seconds': lat,
    }


def gt_pass_from_score(score, parse_error):
    """Binary smoke-test GT: only a fully correct deterministic score is PASS."""
    if parse_error is not None or score is None:
        return False
    try:
        value = float(score)
    except Exception:
        return False
    return math.isfinite(value) and math.isclose(value, 1.0, rel_tol=0.0, abs_tol=1e-12)


def surrogate_summary(rows):
    valid = [r for r in rows if r.get('surrogate_verdict') in {'PASS', 'FAIL'}]
    gt_pass_rows = [r for r in valid if r.get('gt_pass') is True]
    gt_fail_rows = [r for r in valid if r.get('gt_pass') is False]
    false_pass = sum(bool(r.get('false_pass')) for r in valid)
    false_reject = sum(bool(r.get('false_reject')) for r in valid)
    agree = sum(
        (r.get('surrogate_verdict') == 'PASS') == bool(r.get('gt_pass'))
        for r in valid
    )
    return {
        'valid_count': len(valid),
        'parse_or_transport_failures': len(rows) - len(valid),
        'agreement_rate': (agree / len(valid)) if valid else None,
        'false_pass_count': false_pass,
        'false_reject_count': false_reject,
        'false_pass_rate_on_gt_fail': (false_pass / len(gt_fail_rows)) if gt_fail_rows else None,
        'false_reject_rate_on_gt_pass': (false_reject / len(gt_pass_rows)) if gt_pass_rows else None,
    }


def cost_for_same_model_usage(candidate_usage, surrogate_usage):
    """Price candidate and surrogate calls separately with the existing cost helper."""
    c1 = compute_cost_from_usage({'candidate': {'model': MODEL, **candidate_usage}, 'judges': {}})
    c2 = compute_cost_from_usage({'candidate': {'model': MODEL, **surrogate_usage}, 'judges': {}})
    return c1 + c2

def evaluate_l123(client,manifest_path,part,condition,adapter):
    path=result_path('m1_factorybench_l123',part,condition)
    if path.exists():return load(path)
    if not SURROGATE_PATH.exists():
        raise RuntimeError(f'missing surrogate prompt: {SURROGATE_PATH}')

    m,items=source_items(manifest_path)
    system=adapter if adapter else None
    start=time.perf_counter()
    calls=parallel(client,system,[render_prompt(x) for x in items])
    rows=[]
    surrogate_usage={'input_tokens':0,'output_tokens':0,'calls':0}

    for item,(raw,u,lat,err) in zip(items,calls):
        # IMPORTANT: surrogate runs before the hidden GT scorer and receives no GT fields.
        surrogate=run_surrogate(client,item,raw)
        su=surrogate.get('usage') or {}
        surrogate_usage['input_tokens'] += int(su.get('input_tokens',0) or 0)
        surrogate_usage['output_tokens'] += int(su.get('output_tokens',0) or 0)
        if su.get('input_tokens') or su.get('output_tokens'):
            surrogate_usage['calls'] += 1

        # Hidden GT oracle: deterministic FactoryBench parser/scorer for L1-L3.
        s=_score_one(item,raw)
        finite_score=s.score if isinstance(s.score,(int,float)) and math.isfinite(float(s.score)) else None
        item_parse_error=s.parse_error or ('non_finite_score' if finite_score is None else None)
        gt_pass=gt_pass_from_score(finite_score,item_parse_error)
        surrogate_verdict=surrogate.get('verdict')
        false_pass=surrogate_verdict=='PASS' and not gt_pass
        false_reject=surrogate_verdict=='FAIL' and gt_pass

        rows.append({
            'id':item.id,
            'level':item.level,
            'split':next(r['split'] for r in m['items'] if r['id']==item.id),
            'dataset':item.dataset,
            'episode':item.provenance.get('episode'),
            'answer_format':item.answer_format.value,
            'raw_output':raw,
            'parsed':s.parsed,
            'score':finite_score,
            'chance':s.chance,
            'parse_error':item_parse_error,
            'gt_pass':gt_pass,
            'surrogate':surrogate,
            'surrogate_verdict':surrogate_verdict,
            'false_pass':false_pass,
            'false_reject':false_reject,
            'transport_error':err,
            'usage':u,
            'latency_seconds':lat,
            'development_evidence':{
                'rendered_input':render_prompt(item),
                'reference_answer':item.answer,
            } if part=='development' else None,
        })

    candidate_usage={
        'input_tokens':sum(c[1]['input_tokens'] for c in calls),
        'output_tokens':sum(c[1]['output_tokens'] for c in calls),
        'calls':sum(bool(c[1]['input_tokens'] or c[1]['output_tokens']) for c in calls),
    }
    tokens={'candidate':{'model':MODEL,**candidate_usage},'surrogate':{'model':MODEL,**surrogate_usage},'judges':{}}
    ssummary=surrogate_summary(rows)
    payload={
        'arm':ARM,
        'task_name':'m1_factorybench_l123',
        'partition':part,
        'condition':condition,
        'manifest_path':str(manifest_path.relative_to(ROOT)),
        'manifest_sha256':sha(manifest_path),
        'adapter_sha256':hashlib.sha256(adapter.encode()).hexdigest() if adapter else None,
        'surrogate_prompt_path':str(SURROGATE_PATH.relative_to(ROOT)),
        'surrogate_prompt_sha256':sha(SURROGATE_PATH),
        'surrogate_static':True,
        'core_prompt_used':False,
        'ordered_ids':[r['id'] for r in rows],
        'item_count':len(rows),
        'canonical_score':canonical(rows),
        'fixed_cardinality_score':fixed(rows),
        'parse_failures':sum(r['parse_error'] is not None for r in rows),
        'by_level':grouped(rows,'level'),
        'by_format':grouped(rows,'answer_format'),
        'by_dataset':grouped(rows,'dataset'),
        'surrogate_summary':ssummary,
        'tokens_used':tokens,
        'cost':cost_for_same_model_usage(candidate_usage,surrogate_usage),
        'wall_time_seconds':time.perf_counter()-start,
        'items':rows,
    }
    write_json(path,payload)
    return payload


def evaluate_l4(client,manifest_path,part,condition,adapter):
    path=result_path('m1_factorybench_l4',part,condition)
    if path.exists():return load(path)
    m,items=source_items(manifest_path); system=L4_BASE_INSTRUCTIONS+(f'\n\n--- BEGIN TASK ADAPTER ---\n{adapter}\n--- END TASK ADAPTER ---' if adapter else ''); start=time.perf_counter(); calls=parallel(client,system,[render_prompt(x) for x in items]); rows=[]; judge_proxy=ResponseProxy(client)
    for item,(raw,u,lat,err) in zip(items,calls):
        validation=EVALUATOR.validate_deterministically(raw); judge=None; score=0.0; perr=None
        if validation['valid']:
            payload={'question':render_prompt(item),'reference_answer':str(item.answer),'known_root_cause':str(item.root_cause or ''),'model_raw_answer':raw}
            try: judge=EVALUATOR.call_judge(judge_proxy,MODEL,payload); score=float(judge['parsed']['score'])
            except Exception as exc: perr=f'judge_error: {type(exc).__name__}: {exc}'
        else: perr='schema_failure: '+'; '.join(validation['errors'])
        rows.append({'id':item.id,'level':4,'split':'validation','dataset':item.dataset,'episode':item.provenance.get('episode'),'answer_format':'diagnostic_json','raw_output':raw,'parsed':validation.get('parsed'),'score':score,'chance':0.0,'parse_error':perr,'schema_valid':validation['valid'],'schema_errors':validation['errors'],'deterministic_validation':validation,'judge':judge,'judge_score':score if judge else None,'root_cause_correct':judge['parsed']['root_cause_correct'] if judge else None,'corrective_actions_correct':judge['parsed']['protocol_correct'] if judge else None,'evidence_count':len((validation.get('parsed') or {}).get('evidence',[])) if validation['valid'] else 0,'corrective_action_count':len((validation.get('parsed') or {}).get('corrective_actions',[])) if validation['valid'] else 0,'transport_error':err,'usage':u,'latency_seconds':lat,'development_evidence':{'rendered_input':render_prompt(item),'reference_answer':item.answer,'known_root_cause':item.root_cause} if part=='development' else None})
    judges={'gpt-5.5':{'model':MODEL,'input_tokens':sum(x['input_tokens'] for x in judge_proxy.ledger),'output_tokens':sum(x['output_tokens'] for x in judge_proxy.ledger),'calls':len(judge_proxy.ledger)}}
    tokens={'candidate':{'model':MODEL,'input_tokens':sum(c[1]['input_tokens'] for c in calls),'output_tokens':sum(c[1]['output_tokens'] for c in calls),'calls':sum(bool(c[1]['input_tokens'] or c[1]['output_tokens']) for c in calls)},'judges':judges}
    payload={'task_name':'m1_factorybench_l4','partition':part,'condition':condition,'manifest_path':str(manifest_path.relative_to(ROOT)),'manifest_sha256':sha(manifest_path),'adapter_sha256':hashlib.sha256(adapter.encode()).hexdigest() if adapter else None,'core_prompt_used':False,'output_contract':'factorybench_diagnostic_json_v2','judge_model':MODEL,'judge_count':1,'judge_agreement':None,'judge_agreement_note':'Unavailable with one judge.','ordered_ids':[r['id'] for r in rows],'item_count':len(rows),'canonical_score':canonical(rows),'fixed_cardinality_score':fixed(rows),'parse_failures':sum(r['parse_error'] is not None for r in rows),'schema_failures':sum(not r['schema_valid'] for r in rows),'root_cause_correct_count':sum(r['root_cause_correct'] is True for r in rows),'corrective_actions_correct_count':sum(r['corrective_actions_correct'] is True for r in rows),'by_level':{'4':fixed(rows)},'by_format':{'diagnostic_json':fixed(rows)},'by_dataset':grouped(rows,'dataset'),'tokens_used':tokens,'cost':compute_cost_from_usage(tokens),'wall_time_seconds':time.perf_counter()-start,'items':rows}
    write_json(path,payload);return payload

def evaluate(client,task,manifest_path,part,condition,adapter):return evaluate_l123(client,manifest_path,part,condition,adapter) if task=='m1_factorybench_l123' else evaluate_l4(client,manifest_path,part,condition,adapter)

def compact(r):return {'condition':r['condition'],'item_count':r['item_count'],'ordered_ids':r['ordered_ids'],'canonical_score':r['canonical_score'],'fixed_cardinality_score':r['fixed_cardinality_score'],'parse_failures':r['parse_failures'],'by_format':r['by_format'],'by_dataset':r['by_dataset'],'by_level':r['by_level'],'surrogate_summary':r.get('surrogate_summary'),'items':[{k:x.get(k) for k in ('id','level','dataset','answer_format','raw_output','parsed','score','chance','parse_error','gt_pass','surrogate_verdict','false_pass','false_reject','development_evidence') if k in x} | ({'surrogate_diagnosis':(x.get('surrogate') or {}).get('diagnosis',[]),'surrogate_failed_checks':(x.get('surrogate') or {}).get('failed_checks',[])} if 'surrogate' in x else {}) for x in r['items']]}

def m1_input(task,mode,basea,baseb,v1a,v1b,previous):
    if task=='m1_factorybench_l123':desc='FactoryBench L1-L3 manufacturing telemetry tasks with strict item-specific scalar, tensor, ranking, Boolean, and MCQ contracts.'; evaluator={'canonical':'FactoryBench chance-corrected score','fixed_cardinality':'parse failures retained as zero','parser':'FactoryBench deterministic parser/scorer'};contracts=['Follow each item-specific exact output contract.'];sub=['L1 predictive/identification','L2 predictive/identification','L3 predictive']
    else:desc='FactoryBench L4 industrial troubleshooting with diagnostic JSON output grounded in telemetry.';evaluator={'validator':'existing deterministic diagnostic JSON validator','judge':'existing gpt-5.5 semantic judge, count 1','score_values':[0,0.5,1]};contracts=['JSON object with exactly root_cause, evidence, corrective_actions'];sub=['anomaly diagnosis','root cause identification','evidence grounding','corrective action and verification']
    ms=manifests(task)
    return {'task_name':task,'mode':mode,'task_description':desc,'execution_model':MODEL,'evaluator_specification':evaluator,'output_contracts':contracts,'supported_subtasks':sub,'surrogate_verifier':{'enabled':task=='m1_factorybench_l123','mode':'static_v0' if task=='m1_factorybench_l123' else None,'prompt_sha256':sha(SURROGATE_PATH) if task=='m1_factorybench_l123' else None,'gt_visible_to_surrogate':False},'development_manifests':{'fold_a_sha256':sha(ms['fold_a']),'fold_b_sha256':sha(ms['fold_b'])},'development_results':{'baseline_fold_a':compact(basea),'baseline_fold_b':compact(baseb),'adapter_v1_fold_a':compact(v1a) if v1a else None,'adapter_v1_fold_b':compact(v1b) if v1b else None},'previous_adapter':previous,'constraints':{'maximum_adapter_rounds':2,'manual_adapter_editing':False,'core_prompt':False,'holdout_access':False,'case_memorization':False,'surrogate_verifier_static':task=='m1_factorybench_l123'}}

def validate_m1(parsed,task,mode,data,hold_ids):
    errors=[]
    if not isinstance(parsed,dict) or set(parsed)!=M1_FIELDS:return ['schema fields mismatch']
    if parsed.get('meta_prompt_version')!='m1' or parsed.get('task_name')!=task or parsed.get('mode')!=mode:errors.append('identity mismatch')
    if parsed.get('decision') not in {'ADAPTER','NO_ADAPTER','INSUFFICIENT_DATA'} or not isinstance(parsed.get('adapter_text'),str):errors.append('decision/text invalid')
    adapter=parsed.get('adapter_text','')
    if parsed.get('decision')=='ADAPTER' and not adapter.strip():errors.append('empty adapter')
    if parsed.get('decision')!='ADAPTER' and adapter!='':errors.append('nonempty null adapter')
    devitems=data['development_results']['baseline_fold_a']['items']+data['development_results']['baseline_fold_b']['items']; low=adapter.casefold()
    for x in devitems:
        if x['id'].casefold() in low:errors.append('development ID leak')
        ev=x.get('development_evidence') or {}; gold=str(ev.get('reference_answer') or '')
        if len(gold)>=20 and gold.casefold() in low:errors.append('copied gold answer')
        rendered=str(ev.get('rendered_input') or '')
        nums=set(re.findall(r'(?<![A-Za-z])(?:\d{4,}|-?\d+\.\d{3,})(?![A-Za-z])',rendered))
        if any(num in adapter for num in nums):errors.append('case-specific signal value');break
    if any(x.casefold() in low for x in hold_ids):errors.append('holdout ID leak')
    if re.search(r'holdout|held[- ]?out|final[-_ ]?test',low):errors.append('holdout reference')
    for k in ('task_scope','supported_subtasks','supported_formats','applicability_conditions','failure_taxonomy','changes_from_previous','predicted_regression_risks','evidence_limitations'):
        if not isinstance(parsed.get(k),list):errors.append(f'{k} invalid')
    return errors

def call_m1(client,task,roundn,data):
    td=TRACE_DIR/task; ip=td/f'm1_round_{roundn}_input.json'; rp=td/f'm1_round_{roundn}_raw_output.txt'; pp=td/f'm1_round_{roundn}_parsed_output.json';write_json(ip,data)
    if rp.exists() or pp.exists():raise RuntimeError('existing partial M1 trace')
    user='<MANUFACTURING_PROMPT_OPTIMIZATION_INPUT>\n'+json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False)+'\n</MANUFACTURING_PROMPT_OPTIMIZATION_INPUT>';start=time.perf_counter();resp=client.chat.completions.create(model=MODEL,messages=[{'role':'system','content':M1_PATH.read_text()},{'role':'user','content':user}],max_completion_tokens=8192);wall=time.perf_counter()-start;raw=resp.choices[0].message.content or '';write_new(rp,raw.encode())
    try:parsed=json.loads(raw.strip());jerr=None
    except Exception as exc:parsed={};jerr=f'{type(exc).__name__}: {exc}'
    holdids=[x['id'] for x in load(manifests(task)['holdout'])['items']];errors=([jerr] if jerr else [])+validate_m1(parsed,task,data['mode'],data,holdids);u=usage(resp);env={**parsed,'_trace_validation':{'valid':not errors,'errors':errors,'input_sha256':sha(ip),'raw_output_sha256':sha(rp),'m1_sha256':sha(M1_PATH),'usage':u,'cost':compute_cost_from_usage({'candidate':{'model':MODEL,**u,'calls':1},'judges':{}}),'wall_time_seconds':wall}};write_json(pp,env)
    if errors:raise RuntimeError(f'invalid M1 output {errors}')
    adapter=None;path=None
    if parsed['decision']=='ADAPTER':adapter=parsed['adapter_text'];path=adapter_dir(task)/f'adapter_v{roundn}.txt';write_new(path,adapter.encode())
    return env,adapter,path

def select_candidate(task,basea,baseb,candidates):
    path=adapter_dir(task)/'selection.json'
    if path.exists():return load(path)
    expecteda=basea['ordered_ids'];expectedb=baseb['ordered_ids'];records=[];eligible=[]
    base_formats={}
    for r in basea['items']+baseb['items']:base_formats.setdefault(r['answer_format'],[]).append(r)
    base_fmt={k:fixed(v) for k,v in base_formats.items()}
    for label,a,b,ap in candidates:
        reasons=[]
        if a['ordered_ids']!=expecteda or b['ordered_ids']!=expectedb:reasons.append('ID mismatch')
        if a['parse_failures']+b['parse_failures']>0:reasons.append('Adapter parse failure')
        if a['fixed_cardinality_score']<basea['fixed_cardinality_score']:reasons.append('fold A regression')
        if b['fixed_cardinality_score']<baseb['fixed_cardinality_score']:reasons.append('fold B regression')
        if not (a['fixed_cardinality_score']>basea['fixed_cardinality_score'] or b['fixed_cardinality_score']>baseb['fixed_cardinality_score']):reasons.append('no strict fold gain')
        combined=a['items']+b['items'];fmts={}
        for r in combined:fmts.setdefault(r['answer_format'],[]).append(r)
        fs={k:fixed(v) for k,v in fmts.items()}
        if any(fs.get(k,float('-inf'))<v for k,v in base_fmt.items()):reasons.append('critical format regression')
        rec={'candidate':label,'eligible':not reasons,'reasons':reasons,'fold_a_score':a['fixed_cardinality_score'],'fold_b_score':b['fixed_cardinality_score'],'macro_score':(a['fixed_cardinality_score']+b['fixed_cardinality_score'])/2,'minimum_fold_score':min(a['fixed_cardinality_score'],b['fixed_cardinality_score']),'worst_subgroup_score':min(fs.values()),'parse_failures':a['parse_failures']+b['parse_failures'],'adapter_sha256':sha(ap) if ap else None,'adapter_bytes':ap.stat().st_size if ap else 0};records.append(rec)
        if not reasons:eligible.append(rec)
    if eligible:
        order={'adapter_v1':1,'adapter_v2':2};chosen=sorted(eligible,key=lambda x:(-x['macro_score'],-x['minimum_fold_score'],-x['worst_subgroup_score'],x['adapter_bytes'],order[x['candidate']]))[0];decision='ADAPTER';selected=chosen['candidate'];src=adapter_dir(task)/f"{selected}.txt";dst=adapter_dir(task)/'selected_adapter.txt';write_new(dst,src.read_bytes());selected_hash=sha(dst)
    else:decision='NO_ADAPTER';selected='baseline';selected_hash=None
    payload={'task_name':task,'decision':decision,'selected_candidate':selected,'selected_adapter_sha256':selected_hash,'m1_sha256':M1_SHA,'candidates':records};write_json(path,payload);return payload

def development(client,task):
    ms=manifests(task);ba=evaluate(client,task,ms['fold_a'],'development','baseline_fold_a',None);bb=evaluate(client,task,ms['fold_b'],'development','baseline_fold_b',None);o1,a1,p1=call_m1(client,task,1,m1_input(task,'generate',ba,bb,None,None,None));cands=[];r1a=r1b=None
    if a1:r1a=evaluate(client,task,ms['fold_a'],'development','adapter_v1_fold_a',a1);r1b=evaluate(client,task,ms['fold_b'],'development','adapter_v1_fold_b',a1);cands.append(('adapter_v1',r1a,r1b,p1))
    prev={'sha256':sha(p1),'text':a1} if p1 else {'sha256':None,'text':''};o2,a2,p2=call_m1(client,task,2,m1_input(task,'refine',ba,bb,r1a,r1b,prev));r2a=r2b=None
    if a2:r2a=evaluate(client,task,ms['fold_a'],'development','adapter_v2_fold_a',a2);r2b=evaluate(client,task,ms['fold_b'],'development','adapter_v2_fold_b',a2);cands.append(('adapter_v2',r2a,r2b,p2))
    sel=select_candidate(task,ba,bb,cands);summary={'task_name':task,'status':'DEVELOPMENT_COMPLETE','baseline_fold_a':compact(ba),'baseline_fold_b':compact(bb),'adapter_v1_decision':o1['decision'],'adapter_v1_sha256':sha(p1) if p1 else None,'adapter_v1_fold_a':compact(r1a) if r1a else None,'adapter_v1_fold_b':compact(r1b) if r1b else None,'adapter_v2_decision':o2['decision'],'adapter_v2_sha256':sha(p2) if p2 else None,'adapter_v2_fold_a':compact(r2a) if r2a else None,'adapter_v2_fold_b':compact(r2b) if r2b else None,'selection':sel};write_json(RESULT_DIR/'development'/f'{task}_summary.json',summary);return summary

def holdout(client,task):
    ms=manifests(task);sel=load(adapter_dir(task)/'selection.json');base=evaluate(client,task,ms['holdout'],'holdout','baseline',None);selected=None
    if sel['decision']=='ADAPTER':ad=(adapter_dir(task)/'selected_adapter.txt').read_text();selected=evaluate(client,task,ms['holdout'],'holdout','selected_adapter',ad)
    summary={'task_name':task,'status':'HOLDOUT_COMPLETE','selection_sha256':sha(adapter_dir(task)/'selection.json'),'selection_decision':sel['decision'],'baseline':compact(base),'selected_adapter':compact(selected) if selected else None,'selected_label':'selected_adapter' if selected else 'NO_ADAPTER (baseline reused; no duplicate call)','no_holdout_feedback':True,'core_prompt_used':False};write_json(RESULT_DIR/'holdout'/f'{task}_summary.json',summary);return summary

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--task',choices=['m1_factorybench_l123'],required=True);ap.add_argument('--phase',choices=['development','holdout'],required=True);args=ap.parse_args()
    if not os.getenv('OPENAI_API_KEY'):raise SystemExit('OPENAI_API_KEY missing in model process')
    if sha(M1_PATH)!=M1_SHA:raise SystemExit('M1 hash mismatch')
    client=OpenAI();result=development(client,args.task) if args.phase=='development' else holdout(client,args.task);print(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False))

if __name__=='__main__':
    main()
