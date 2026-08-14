#!/usr/bin/env python3
"""Build Manufacturing M1 cross-task report and artifact registry."""
from __future__ import annotations
import hashlib,json,math
from pathlib import Path
from typing import Any
ROOT=Path('/home/training/automatic_prompt_engineer/hermes_factorybench_skill_experiment')
SUMMARY=ROOT/'results/meta_m1/manufacturing_cross_task_summary.json';REPORT=ROOT/'reports/meta_m1/manufacturing_cross_task_summary.md';REGISTRY=ROOT/'results/meta_m1/artifact_registry.json'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(rel):return json.loads((ROOT/rel).read_text())
def write_new(p,b):
    if p.exists():
        if p.read_bytes()!=b:raise RuntimeError(f'refusing overwrite {p}')
        return
    p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b)
def write_json(p,o):write_new(p,(json.dumps(o,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode())
def calls(r):
    u=r.get('tokens_used') or {};return int((u.get('candidate') or {}).get('calls',0))+sum(int(x.get('calls',0)) for x in (u.get('judges') or {}).values())
def paired(a,b):
    bm={x['id']:x for x in b['items']};counts={'improved':0,'worse':0,'unchanged':0,'invalid':0};changed=[]
    for x in a['items']:
        y=bm[x['id']]
        if x.get('parse_error') is not None or y.get('parse_error') is not None:s='invalid'
        else:
            d=float(y['score'])-float(x['score']);s='improved' if d>0 else 'worse' if d<0 else 'unchanged'
        counts[s]+=1
        if x.get('raw_output')!=y.get('raw_output'):changed.append({'id':x['id'],'baseline_output':x.get('raw_output'),'selected_output':y.get('raw_output'),'baseline_score':x.get('score'),'selected_score':y.get('score'),'status':s})
    return {'exact_ordered_id_match':a['ordered_ids']==b['ordered_ids'],'counts':counts,'raw_output_changed_count':len(changed),'changed_cases':changed}
def task_verdict(sel,base,selected):
    if sel['decision']=='NO_ADAPTER':return 'TASK_NO_EFFECT',0.0,[]
    regress=[]
    for k,v in base['by_format'].items():
        if selected['by_format'].get(k,float('-inf'))<v:regress.append(k)
    delta=selected['fixed_cardinality_score']-base['fixed_cardinality_score']
    if selected['parse_failures']>0:return 'TASK_REGRESSION',delta,regress+['parse_failure']
    if delta<0 or regress:return 'TASK_REGRESSION',delta,regress
    if delta>=.05:return 'TASK_PASS',delta,regress
    if delta>0:return 'TASK_WEAK_SIGNAL',delta,regress
    return 'TASK_NO_EFFECT',delta,regress
def main():
    if any(p.exists() for p in (SUMMARY,REPORT,REGISTRY)):raise RuntimeError('M1 final target exists')
    tasks={}
    for task in ('m1_factorybench_l123','m1_factorybench_l4'):
        dev=load(f'results/meta_m1/development/{task}_summary.json');hold=load(f'results/meta_m1/holdout/{task}_summary.json');sel=load(f'prompts/adapters/{task}/selection.json');base=load(f'results/meta_m1/holdout/{task}_baseline.json');selected=load(f'results/meta_m1/holdout/{task}_selected_adapter.json') if sel['decision']=='ADAPTER' else None;verdict,delta,regress=task_verdict(sel,base,selected)
        tasks[task]={'development':dev,'selection':sel,'holdout':{'baseline':base,'selected_adapter':selected,'paired':paired(base,selected) if selected else {'label':'NO_ADAPTER','exact_ordered_id_match':True,'counts':{'improved':0,'worse':0,'unchanged':base['item_count'],'invalid':0},'raw_output_changed_count':0,'changed_cases':[]}},'task_verdict':verdict,'holdout_delta':delta,'critical_format_regressions':regress}
    verdicts=[x['task_verdict'] for x in tasks.values()]
    if any(v=='TASK_INVALID' for v in verdicts):overall='M1_INVALID'
    elif all(v=='TASK_PASS' for v in verdicts):overall='M1_MANUFACTURING_PROMISING'
    elif sum(v in {'TASK_PASS','TASK_WEAK_SIGNAL'} for v in verdicts)==1 and sum(v=='TASK_NO_EFFECT' for v in verdicts)==1:overall='M1_MANUFACTURING_PARTIAL'
    elif any(v in {'TASK_PASS','TASK_WEAK_SIGNAL'} for v in verdicts) and any(v=='TASK_REGRESSION' for v in verdicts):overall='M1_MANUFACTURING_MIXED'
    elif all(v=='TASK_NO_EFFECT' for v in verdicts):overall='M1_MANUFACTURING_NO_EFFECT'
    elif any(v=='TASK_REGRESSION' for v in verdicts) and not any(v in {'TASK_PASS','TASK_WEAK_SIGNAL'} for v in verdicts):overall='M1_MANUFACTURING_NEGATIVE'
    else:overall='M1_INVALID'
    result_files=[p for p in (ROOT/'results/meta_m1/development').glob('*.json') if 'summary' not in p.name and 'failed_attempt' not in p.name]+[p for p in (ROOT/'results/meta_m1/holdout').glob('*.json') if 'summary' not in p.name]
    total_calls=3;known_cost=0.;wall=0.;tokens={'input_tokens':0,'output_tokens':0}
    for p in result_files:
        r=json.loads(p.read_text());total_calls+=calls(r);known_cost+=float(r.get('cost',0));wall+=float(r.get('wall_time_seconds',0));u=r.get('tokens_used') or {};c=u.get('candidate') or {};tokens['input_tokens']+=int(c.get('input_tokens',0));tokens['output_tokens']+=int(c.get('output_tokens',0));
        for j in (u.get('judges') or {}).values():tokens['input_tokens']+=int(j.get('input_tokens',0));tokens['output_tokens']+=int(j.get('output_tokens',0))
    trace_stats=[]
    for p in (ROOT/'traces/meta_m1').glob('*/*_parsed_output.json'):
        t=json.loads(p.read_text())['_trace_validation'];total_calls+=1;known_cost+=float(t['cost']);wall+=float(t['wall_time_seconds']);tokens['input_tokens']+=int(t['usage']['input_tokens']);tokens['output_tokens']+=int(t['usage']['output_tokens']);trace_stats.append({'path':str(p.relative_to(ROOT)),'sha256':sha(p),'usage':t['usage'],'cost':t['cost'],'wall_time_seconds':t['wall_time_seconds']})
    manifests={p.name:{'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted((ROOT/'data_manifests/meta_m1').glob('*.json'))}
    summary={'experiment':'Manufacturing Meta-Prompt M1 Candidate — Two-Task Frozen Smoke','qualification':'M1 is not Golden.','m1':{'path':'prompts/meta/manufacturing_meta_prompt_m1_candidate.txt','sha256':sha(ROOT/'prompts/meta/manufacturing_meta_prompt_m1_candidate.txt'),'bytes':(ROOT/'prompts/meta/manufacturing_meta_prompt_m1_candidate.txt').stat().st_size},'core_prompt_used':False,'contamination_registry':{'path':'data_manifests/meta_m1/contamination_registry_m1.json','sha256':sha(ROOT/'data_manifests/meta_m1/contamination_registry_m1.json'),'counts':load('data_manifests/meta_m1/contamination_registry_m1.json')['counts']},'manifests':manifests,'tasks':tasks,'manufacturing_verdict':overall,'usage':{'total_calls_including_failed_attempt':total_calls,'known_cost':round(known_cost,6),'unavailable_failed_attempt_cost':True,'summed_wall_time_seconds_excluding_failed_attempt':wall,'token_usage_excluding_failed_attempt':tokens,'failed_attempt_calls':3},'m1_traces':trace_stats,'integrity':{'core_prompt_used':False,'manual_adapter_editing':False,'holdout_feedback_to_m1':False,'post_holdout_modification':False},'limitations':['Small frozen smoke folds/holdouts.','One L4 judge; agreement unavailable.','A failed L1-L3 development attempt made three calls before strict serialization failed; its usage, cost, and wall time are unavailable.','No statistical significance is claimed.'],'promotion_requirement':'A third manufacturing task, larger fresh holdouts, repeated evaluation, and a predeclared 5–10 point target are still required.'}
    write_json(SUMMARY,summary)
    lines=['# Manufacturing Meta-Prompt M1 Two-Task Frozen Smoke','',f"- M1 SHA-256: `{summary['m1']['sha256']}`",f"- M1 bytes: {summary['m1']['bytes']}",'- Core used: no','', '## Development','', '| Task | Baseline A | Baseline B | Adapter v1 A/B | Adapter v2 A/B | Selection |','|---|---:|---:|---|---|---|']
    for task,d in tasks.items():
        dev=d['development'];v1=f"{dev['adapter_v1_fold_a']['fixed_cardinality_score']} / {dev['adapter_v1_fold_b']['fixed_cardinality_score']}" if dev['adapter_v1_fold_a'] else 'n/a';v2=f"{dev['adapter_v2_fold_a']['fixed_cardinality_score']} / {dev['adapter_v2_fold_b']['fixed_cardinality_score']}" if dev['adapter_v2_fold_a'] else 'n/a';lines.append(f"| {task} | {dev['baseline_fold_a']['fixed_cardinality_score']} | {dev['baseline_fold_b']['fixed_cardinality_score']} | {v1} | {v2} | {d['selection']['decision']} ({d['selection']['selected_candidate']}) |")
    lines+=['','## Holdout','','| Task | Baseline | Selected | Delta | Parse failures | Verdict |','|---|---:|---:|---:|---:|---|']
    for task,d in tasks.items():
        b=d['holdout']['baseline'];s=d['holdout']['selected_adapter'];lines.append(f"| {task} | {b['fixed_cardinality_score']} | {s['fixed_cardinality_score'] if s else 'NO_ADAPTER'} | {d['holdout_delta']} | {s['parse_failures'] if s else b['parse_failures']} | {d['task_verdict']} |")
    lines+=['',f'## Manufacturing verdict: **{overall}**','',f"Calls including failed attempt: {total_calls}; known cost: ${known_cost:.6f}; summed known wall time: {wall:.3f}s.",'','No Core prompt was used. No holdout data entered M1 generation or refinement. M1 remains a smoke candidate, not Golden.','']
    write_new(REPORT,'\n'.join(lines).encode())
    paths=[]
    for base in (ROOT/'data_manifests/meta_m1',ROOT/'reports/meta_m1',ROOT/'results/meta_m1',ROOT/'traces/meta_m1',ROOT/'logs/meta_m1'):
        if base.exists():paths.extend(p for p in base.rglob('*') if p.is_file() and p not in {REGISTRY} and '__pycache__' not in p.parts and p.suffix!='.pyc')
    for task in ('m1_factorybench_l123','m1_factorybench_l4'):
        base=ROOT/'prompts/adapters'/task
        if base.exists():paths.extend(p for p in base.rglob('*') if p.is_file())
    paths += [ROOT/'prompts/meta/manufacturing_meta_prompt_m1_candidate.txt',ROOT/'prompts/meta/manufacturing_meta_prompt_m1_candidate.sha256',ROOT/'scripts/meta_m1/prepare_m1.py',ROOT/'scripts/meta_m1/run_m1.py',ROOT/'scripts/meta_m1/build_m1_report.py']
    rec={str(p.relative_to(ROOT)):{'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(set(paths))};write_json(REGISTRY,{'registry_self_hash':None,'volatile_files_excluded':['__pycache__','*.pyc','temporary verification files'],'credential_files_excluded':True,'artifact_count':len(rec),'artifacts':rec})
    print(json.dumps({'status':'COMPLETE','verdict':overall,'task_verdicts':{k:v['task_verdict'] for k,v in tasks.items()},'summary_sha256':sha(SUMMARY),'report_sha256':sha(REPORT),'registry_sha256':sha(REGISTRY),'usage':summary['usage']},indent=2))
if __name__=='__main__':main()
