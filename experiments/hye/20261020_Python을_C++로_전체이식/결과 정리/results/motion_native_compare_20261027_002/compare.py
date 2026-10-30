"""One-time reference oracle: legacy Python vs independent native C++ feature API.

This audit is not imported or executed by the C++ product/session runtime.
"""
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import numpy as np

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
source=ROOT.parent/'codex_experiments_20261014/joint_yt_motion_logic_20261026/motion.py'
spec=importlib.util.spec_from_file_location('legacy_motion_oracle',source)
legacy=importlib.util.module_from_spec(spec);sys.modules[spec.name]=legacy;spec.loader.exec_module(legacy)
binary=ROOT/'build/Release/motion_session.exe'
J=np.array([[120.,20.],[-80.,300.],[50.,-90.],[200.,150.]])
goal=np.array([7.,-2.,5.,1.])
comparisons=[];calls=[];maximum=0.;singular_diagnostics=[]

def clean(x):
    if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean(v) for v in x]
    if isinstance(x,np.ndarray):return clean(x.tolist())
    if isinstance(x,(float,np.floating)):return float(x) if math.isfinite(x) else None
    return x

def compare(expected,actual,path):
    global maximum
    if isinstance(expected,dict):
        for k,v in expected.items():compare(v,actual[k],path+'/'+k)
    elif isinstance(expected,list):
        assert len(expected)==len(actual),(path,'length mismatch')
        for i,(a,b) in enumerate(zip(expected,actual)):compare(a,b,path+'/'+str(i))
    elif isinstance(expected,(float,int)) and not isinstance(expected,bool):
        assert isinstance(actual,(int,float)),(path,expected,actual)
        if path.endswith('/scaled_response_condition') and expected>1e12 and actual>1e12:
            singular_diagnostics.append(dict(path=path,python=expected,cpp=actual,reason='Effectively rank-deficient condition diagnostic; numeric magnitude is not reproducible across SVD implementations. Both exceed the fixed 1000 rejection gate. Status and actions are still compared exactly.'))
            return
        difference=abs(expected-actual);maximum=max(maximum,difference)
        assert difference<=1e-9,(path,expected,actual,difference)
    else:assert expected==actual,(path,expected,actual)

def invoke(operation,state,payload=None):
    args=[str(binary),operation,str(state)]
    if payload is not None:
        p=state.parent/f'input_{len(calls):03d}.json';p.write_text(json.dumps(clean(payload),allow_nan=False),encoding='utf-8');args.append(str(p))
    run=subprocess.run(args,cwd=ROOT,capture_output=True,text=True,encoding='utf-8')
    calls.append(dict(command=args,exit_code=run.returncode))
    if run.returncode:raise RuntimeError(run.stderr)
    return json.loads(run.stdout)

def snapshot(c):
    j=c.snapshot();action=c.next_action();j['suggested_relative_command']=None if action is None else action.tolist();return clean(j)

def scenario(name,cfg,q,goal,plant=J,prior=None,ack=None,limit=20,failed=False,response_factor=1.):
    folder=OUT/name;folder.mkdir(exist_ok=False);state=folder/'state.json'
    s=goal+plant@q;c=legacy.MotionController(cfg,prior).start(s,goal)
    actual=invoke('start-features',state,dict(settings=cfg.__dict__,prior=prior,observation=s,fixed_goal=goal))
    compare(snapshot(c),actual,name+'/start');steps=0
    for i in range(limit):
        u=c.next_action()
        if u is None:break
        executed=np.array(ack[i] if ack is not None and i<len(ack) else u)
        q=q+executed;s=s+response_factor*plant@executed
        measured=np.full_like(s,np.nan) if failed else s
        c.observe(measured,executed)
        actual=invoke('observe-features',state,dict(observation=measured,executed_relative=executed))
        compare(snapshot(c),actual,name+'/'+str(i+1));steps+=1
    inspection=invoke('inspect',state)
    compare(clean(c.events),inspection['events'],name+'/events')
    comparisons.append(dict(name=name,steps=steps,terminal_status=c.status,passed=True))

def main():
    cfg=legacy.Settings(probe_y=.2,probe_t=.04,probe_count=2)
    scenario('linear',cfg,np.array([.65,-.4]),goal)
    scenario('partial_ack',cfg,np.array([.2,.3]),np.zeros(4),ack=[[.13,.003],[-.01,.027]])
    scenario('refusal',legacy.Settings(),np.array([.2,.3]),np.zeros(4),ack=[[0.,0.]]*3)
    one=np.c_[J[:,0],2*J[:,0]]
    scenario('rank_one',cfg,np.array([.5,.3]),np.zeros(4),plant=one)
    scenario('failed_capture',cfg,np.array([.2,.3]),np.zeros(4),failed=True)
    scenario('far_turn_limit',cfg,np.array([100.,100.]),np.zeros(4))
    scenario('guarded_no_execution',cfg,np.array([.2,.3]),np.zeros(4),prior=J,ack=[[0.,0.]]*3,limit=3)
    scenario('guarded_response_change',cfg,np.array([.2,.3]),np.zeros(4),prior=J,response_factor=3.,limit=2)
    report=dict(scope='Synthetic feature-API parity only; no image/hardware claims',scenarios=comparisons,native_process_calls=len(calls),maximum_numeric_difference=maximum,tolerance=1e-9,nonidentifiable_rank_deficient_condition_diagnostics=singular_diagnostics,criterion_timing='Rank-deficient diagnostic equivalence defined after the preserved failed absolute-comparison run; all finite identifiable outputs remain at the original 1e-9 criterion.',passed=True)
    (OUT/'checks.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    (OUT/'commands.json').write_text(json.dumps(calls,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
