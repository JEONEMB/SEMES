"""Recompute outcomes from stored movement traces, without controller scores."""
from pathlib import Path
import gzip,json,sys,hashlib
import numpy as np,pandas as pd,joblib
from evaluate import H,OLD,CONDITIONS,summarize
from motion import MotionController,Settings
from simulate import SCENARIOS

def inspect_trace(t,row,m):
    states=np.array(t['latent_q']);commands=np.asarray(t['acknowledged_commands']).reshape(-1,2)
    assert len(states)==len(commands)+1==int(row.turns)+1
    assert len(t['events'])==int(row.turns)<=20
    assert np.allclose(states[0],m.iloc[int(row['index'])][['Y','T']].to_numpy(dtype=float))
    assert np.allclose(abs(states[-1]),[row.Y_abs,row.T_abs],atol=1e-10)
    passed=bool(not row.aborted and np.max(abs(states[-1]))<=.09+1e-12)
    assert passed==bool(row.passed)
    assert bool(row.false_visual_stop)==(row.status=='visual_goal_confirmed' and not passed)
    hits=np.flatnonzero(np.max(abs(states),axis=1)<=.09+1e-12)
    assert int(row.first_true_tolerance_turn)==(int(hits[0]) if len(hits) else -1)
    assert np.all(abs(commands)<=.25+1e-12)
    for k,event in enumerate(t['events']):
        assert event['turn']==k+1
        assert np.allclose(event['executed_relative'],commands[k])
        assert np.allclose(event['cumulative_relative'],commands[:k+1].sum(0))
    return states,commands

def replay(t,row):
    bundle=joblib.load(OLD/'control'/f'{row.condition}_fold{row.fold}.joblib');cal=bundle['calibration'];plant=bundle['plant']
    scenario=next(s for s in SCENARIOS if s['name']==row.scenario);sid=SCENARIOS.index(scenario);rng=np.random.default_rng(20261023+1000*sid+int(row['index']))
    noise=plant.residual[rng.integers(len(plant.residual),size=21)]*scenario['noise'];K=cal.jacobian.shape[0]
    drift=rng.normal(size=K);drift/=max(float(np.sqrt(np.mean(drift**2))),1e-12)
    initial=np.array(t['latent_q'][0]);s=plant.value(initial,scenario['offset'])+noise[0]
    guarded=row.policy.startswith('guarded');count=2 if row.policy.endswith('2') else 3
    first_probe=t['acknowledged_commands'][0][0];second_probe=t['acknowledged_commands'][1][1]
    c=MotionController(Settings(probe_y=abs(first_probe),probe_t=abs(second_probe),probe_count=count,noise_rms=max(cal.noise_rms,1.)),cal.jacobian if guarded else None).start(s,np.zeros(K))
    q=initial.copy();previous=np.zeros(2);corpus=pd.read_csv(OLD/'manifest.csv')[['Y','T']].to_numpy();low=corpus.min(0)-.25;high=corpus.max(0)+.25
    for k,event in enumerate(t['events']):
        u=c.next_action();assert np.allclose(u,event['suggested'],atol=1e-10)
        effective=np.asarray(scenario['actuator'])@u
        reversal=(np.sign(effective)!=np.sign(previous))&(np.sign(previous)!=0)&(abs(effective)>1e-12)
        effective=np.sign(effective)*np.maximum(abs(effective)-reversal*np.asarray(scenario['backlash']),0.)
        previous=np.where(abs(effective)>1e-12,effective,previous);q+=effective
        assert np.allclose(q,t['latent_q'][k+1],atol=1e-10)
        s=plant.value(q,scenario['offset'])+noise[k+1]+(k+1)*scenario['drift']*drift
        if np.any(q<low)|np.any(q>high):s[:]=np.nan
        assert np.allclose(s-c.s,event['observed_change'],atol=1e-9,equal_nan=True)
        c.observe(s,u)
    assert c.status==row.status and c.turn==row.turns

def main():
    m=pd.read_csv(OLD/'manifest.csv');checks={};sampled=set()
    for prefix,tracefile in [('simulation','simulation_traces.json.gz'),('guarded','guarded_traces.json.gz')]:
        df=pd.read_csv(H/(prefix+'_cases.csv'));summary=pd.read_csv(H/(prefix+'_summary.csv'));df.index=df.case_id
        if prefix=='simulation':
            with gzip.open(H/tracefile,'rt',encoding='utf-8') as f:traces=json.load(f)
        else:
            stream=gzip.open(H/tracefile,'rt',encoding='utf-8');traces=(json.loads(line) for line in stream)
        count=0;reproduced=0
        for t in traces:
            row=df.loc[t['case_id']];inspect_trace(t,row,m);key=(row.condition,row.scenario,row.policy)
            if key not in sampled:
                replay(t,row);sampled.add(key);reproduced+=1
            count+=1
        assert count==len(df)
        for key,g in df.groupby(['condition','scenario','policy']):
            assert len(g)==341 and g['index'].nunique()==341
            r=summary[(summary.condition==key[0])&(summary.scenario==key[1])&(summary.policy==key[2])&(summary.dataset=='all')].iloc[0]
            actual=summarize(g)
            for k in ['n','estimated','passed','Y_MAE','Y_max','T_MAE','T_max']:assert np.isclose(actual[k],r[k],equal_nan=True)
        checks[prefix]=dict(cases=count,all_terminal_states_and_turns_recomputed=True,summaries_recomputed=True,full_surrogate_and_controller_replays=reproduced)
    # The static benchmark never fits projection/noise with its own rectangle.
    audits=json.loads((H/'recorded_training_audit.json').read_text())
    for a in audits:
        assert not set(a['train'])&set(a['excluded'])
        assert not set(m.iloc[a['train']].cv_group)&set(m.iloc[a['excluded']].cv_group)
    checks['recorded']=dict(purged_training_fits=len(audits),predictions=len(pd.read_csv(H/'recorded_predictions.csv')),source_images=48,real_sequential_trajectories=0)
    frozen=json.loads((OLD/'frozen_manifest.json').read_text());changed=[p for p,d in frozen['files'].items() if hashlib.sha256((OLD/p).read_bytes()).hexdigest()!=d]
    # The old review prose was revised on Oct23 AFTER its frozen manifest,
    # before this Oct26 run. Disclose that drift; do not silently re-baseline it.
    preexisting=[]
    for p in changed:
        assert p=='control/independent_review.json',changed
        assert (OLD/p).stat().st_mtime<(H/'protocol.json').stat().st_mtime
        preexisting.append(dict(path=p,frozen_hash=frozen['files'][p],current_hash=hashlib.sha256((OLD/p).read_bytes()).hexdigest(),mtime_utc=__import__('datetime').datetime.fromtimestamp((OLD/p).stat().st_mtime,__import__('datetime').timezone.utc).isoformat()))
    checks['previous_experiment']=dict(frozen_files_checked=len(frozen['files']),unchanged_frozen_files=len(frozen['files'])-len(changed),preexisting_review_text_drift=preexisting,current_run_preservation=json.loads((H/'preservation.json').read_text()),numerical_models_predictions_and_code_match_previous_freeze=True)
    checks['scope']='Arithmetic/split/replay verification. Neither independent new-image validation nor physical robot trials.'
    (H/'verification.json').write_text(json.dumps(checks,indent=2),encoding='utf-8');print(json.dumps(checks,indent=2))
if __name__=='__main__':main()
