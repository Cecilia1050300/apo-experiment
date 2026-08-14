#!/usr/bin/env python3
"""Run the frozen manufacturing Meta-Prompt M1 two-task smoke."""
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
MODEL = 'gpt-5.5'
REV = 'b3863519ccedbceab54dfa7600104eb42b985ed7'
M1_PATH = ROOT / 'prompts/meta/manufacturing_meta_prompt_m1_candidate.txt'
M1_SHA = '78187e3268294657d2398c9a79563a36f050c4189b2f6650cc569407512cb052'
MANIFEST_DIR = ROOT / 'data_manifests/meta_m1'
RESULT_DIR = ROOT / 'results/meta_m1'
TRACE_DIR = ROOT / 'traces/meta_m1'
ADAPTER_ROOT = ROOT / 'prompts/adapters'
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
    clean=[r for r in rows if r.get('parse_error') is None and math.isfinite(float(r['score']))]
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

def evaluate_l123(client,manifest_path,part,condition,adapter):
    path=result_path('m1_factorybench_l123',part,condition)
    if path.exists():return load(path)
    m,items=source_items(manifest_path); system=adapter if adapter else None; start=time.perf_counter(); calls=parallel(client,system,[render_prompt(x) for x in items]); rows=[]
    for item,(raw,u,lat,err) in zip(items,calls):
        s=_score_one(item,raw); rows.append({'id':item.id,'level':item.level,'split':next(r['split'] for r in m['items'] if r['id']==item.id),'dataset':item.dataset,'episode':item.provenance.get('episode'),'answer_format':item.answer_format.value,'raw_output':raw,'parsed':s.parsed,'score':s.score,'chance':s.chance,'parse_error':s.parse_error,'transport_error':err,'usage':u,'latency_seconds':lat,'development_evidence':{'rendered_input':render_prompt(item),'reference_answer':item.answer} if part=='development' else None})
    tokens={'candidate':{'model':MODEL,'input_tokens':sum(c[1]['input_tokens'] for c in calls),'output_tokens':sum(c[1]['output_tokens'] for c in calls),'calls':sum(bool(c[1]['input_tokens'] or c[1]['output_tokens']) for c in calls)},'judges':{}}
    payload={'task_name':'m1_factorybench_l123','partition':part,'condition':condition,'manifest_path':str(manifest_path.relative_to(ROOT)),'manifest_sha256':sha(manifest_path),'adapter_sha256':hashlib.sha256(adapter.encode()).hexdigest() if adapter else None,'core_prompt_used':False,'ordered_ids':[r['id'] for r in rows],'item_count':len(rows),'canonical_score':canonical(rows),'fixed_cardinality_score':fixed(rows),'parse_failures':sum(r['parse_error'] is not None for r in rows),'by_level':grouped(rows,'level'),'by_format':grouped(rows,'answer_format'),'by_dataset':grouped(rows,'dataset'),'tokens_used':tokens,'cost':compute_cost_from_usage(tokens),'wall_time_seconds':time.perf_counter()-start,'items':rows}
    write_json(path,payload);return payload

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

def compact(r):return {'condition':r['condition'],'item_count':r['item_count'],'ordered_ids':r['ordered_ids'],'canonical_score':r['canonical_score'],'fixed_cardinality_score':r['fixed_cardinality_score'],'parse_failures':r['parse_failures'],'by_format':r['by_format'],'by_dataset':r['by_dataset'],'by_level':r['by_level'],'items':[{k:x.get(k) for k in ('id','level','dataset','answer_format','raw_output','parsed','score','chance','parse_error','schema_valid','root_cause_correct','corrective_actions_correct','development_evidence') if k in x} for x in r['items']]}

def m1_input(task,mode,basea,baseb,v1a,v1b,previous):
    if task=='m1_factorybench_l123':desc='FactoryBench L1-L3 manufacturing telemetry tasks with strict item-specific scalar, tensor, ranking, Boolean, and MCQ contracts.'; evaluator={'canonical':'FactoryBench chance-corrected score','fixed_cardinality':'parse failures retained as zero','parser':'FactoryBench deterministic parser/scorer'};contracts=['Follow each item-specific exact output contract.'];sub=['L1 predictive/identification','L2 predictive/identification','L3 predictive']
    else:desc='FactoryBench L4 industrial troubleshooting with diagnostic JSON output grounded in telemetry.';evaluator={'validator':'existing deterministic diagnostic JSON validator','judge':'existing gpt-5.5 semantic judge, count 1','score_values':[0,0.5,1]};contracts=['JSON object with exactly root_cause, evidence, corrective_actions'];sub=['anomaly diagnosis','root cause identification','evidence grounding','corrective action and verification']
    ms=manifests(task)
    return {'task_name':task,'mode':mode,'task_description':desc,'execution_model':MODEL,'evaluator_specification':evaluator,'output_contracts':contracts,'supported_subtasks':sub,'development_manifests':{'fold_a_sha256':sha(ms['fold_a']),'fold_b_sha256':sha(ms['fold_b'])},'development_results':{'baseline_fold_a':compact(basea),'baseline_fold_b':compact(baseb),'adapter_v1_fold_a':compact(v1a) if v1a else None,'adapter_v1_fold_b':compact(v1b) if v1b else None},'previous_adapter':previous,'constraints':{'maximum_adapter_rounds':2,'manual_adapter_editing':False,'core_prompt':False,'holdout_access':False,'case_memorization':False}}

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
    ap=argparse.ArgumentParser();ap.add_argument('--task',choices=['m1_factorybench_l123','m1_factorybench_l4'],required=True);ap.add_argument('--phase',choices=['development','holdout'],required=True);args=ap.parse_args()
    if not os.getenv('OPENAI_API_KEY'):raise SystemExit('OPENAI_API_KEY missing in model process')
    if sha(M1_PATH)!=M1_SHA:raise SystemExit('M1 hash mismatch')
    client=OpenAI();result=development(client,args.task) if args.phase=='development' else holdout(client,args.task);print(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False))
if __name__=='__main__':main()
