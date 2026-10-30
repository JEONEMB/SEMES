"""Two separate evidence levels: recorded-photo combinations and surrogate control."""
from pathlib import Path
import sys,json,hashlib,itertools,time
from dataclasses import asdict
import numpy as np,pandas as pd,joblib
from threadpoolctl import threadpool_limits
H=Path(__file__).resolve().parent;OLD=H.parent/'joint_yt_visual_feedback_20261023'
sys.path.insert(0,str(OLD));sys.path.insert(0,str(OLD/'control'))
from controller import Calibration
from evaluate_stencils import paths
from simulate import SCENARIOS
from motion import MotionController,Settings

CONDITIONS=['qr','upper','lower','both']

def summarize(g,ycol='Y_abs',tcol='T_abs'):
    valid=g[[ycol,tcol]].notna().all(axis=1);v=g[valid]
    return dict(n=len(g),estimated=int(valid.sum()),Y_MAE=float(v[ycol].mean()),Y_max=float(v[ycol].max()),T_MAE=float(v[tcol].mean()),T_max=float(v[tcol].max()),passed=int(g.passed.sum()),pass_fraction=float(g.passed.mean()))

def recorded(m,a):
    q=m[['Y','T']].to_numpy();rows=[];audit=[]
    for cond in CONDITIONS:
        X=a['X_'+cond];good=a['success_'+cond]
        for pid,path in enumerate(paths()):
            states=[np.flatnonzero((m.dataset.to_numpy()==20261020)&np.all(np.isclose(q,v,atol=1e-12),axis=1)) for v in path]
            excluded=np.concatenate(states);fold=int(m.iloc[states[0][0]].fold)
            tr=np.flatnonzero((m.fold.to_numpy()!=fold)&good);tr=tr[~np.isin(tr,excluded)]
            assert not set(m.iloc[tr].cv_group)&set(m.iloc[excluded].cv_group)
            cal=Calibration.fit(X[tr],q[tr],m.iloc[tr].dataset.to_numpy(),tr)
            S=cal.project(X);offsets=path-path[0]
            audit.append(dict(condition=cond,path=pid,fold=fold,train=tr.tolist(),excluded=excluded.tolist(),relative_offsets=offsets.tolist(),no_test_condition_used_to_fit_projection=True))
            for frames in [3,4]:
                # Sequence contains actual archive frames, not repeated model predictions.
                # Relative increments are benchmark metadata only, not robot execution.
                step=abs(path[1]-path[0])+abs(path[2]-path[1]);cfg=Settings(probe_y=float(step[0]),probe_t=float(step[1]),probe_count=frames-1,max_step_y=max(.25,float(step[0])),max_step_t=max(.25,float(step[1])),noise_rms=max(cal.noise_rms,1.))
                for ids in itertools.product(*states[:frames]):
                    c=MotionController(cfg).start(S[ids[0]],np.zeros(S.shape[1]))
                    for k in range(1,frames):
                        c.next_action();c.observe(S[ids[k]],offsets[k]-offsets[k-1])
                    usable=c.J is not None and np.isfinite(c.correction).all()
                    # Evaluate error of a hypothetical perfect move, not an observed endpoint.
                    predicted=-c.correction if usable else np.full(2,np.nan)
                    e=abs(predicted-path[frames-1]);rows.append(dict(condition=cond,path=pid,frames=frames,moves=frames-1,indices=','.join(map(str,ids)),status=c.status,estimated=usable,current_Y=path[frames-1,0],current_T=path[frames-1,1],predY=predicted[0],predT=predicted[1],Y_abs=e[0],T_abs=e[1],passed=bool(usable and np.all(e<=.09+1e-12)),response_snr=c.diagnostics.get('response_snr'),fit_rms=c.diagnostics.get('fit_rms')))
        print('recorded combinations',cond,'complete',flush=True)
    df=pd.DataFrame(rows);df.to_csv(H/'recorded_predictions.csv',index=False)
    (H/'recorded_training_audit.json').write_text(json.dumps(audit,indent=2))
    sums=[dict(condition=cond,frames=int(frames),**summarize(g)) for (cond,frames),g in df.groupby(['condition','frames'])]
    pd.DataFrame(sums).to_csv(H/'recorded_summary.csv',index=False)

def simulate(m):
    q=m[['Y','T']].to_numpy();low=q.min(0)-.25;high=q.max(0)+.25
    rows=[];traces=[]
    for cond in CONDITIONS:
        for fold in range(4):
            bundle=joblib.load(OLD/'control'/f'{cond}_fold{fold}.joblib');cal=bundle['calibration'];plant=bundle['plant']
            assert not set(cal.training_indices)&set(plant.indices)
            for scenario_id,scenario in enumerate(SCENARIOS):
                for index in np.flatnonzero(m.fold.to_numpy()==fold):
                    for probes in [2,3]:
                        # Same initial states, surrogate fits, noise streams and amplitudes
                        # as the previous small-probe comparison; no amplitude retuning.
                        rng=np.random.default_rng(20261023+1000*scenario_id+int(index));K=cal.jacobian.shape[0]
                        noise=plant.residual[rng.integers(len(plant.residual),size=21)]*scenario['noise']
                        drift=rng.normal(size=K);drift/=max(float(np.sqrt(np.mean(drift**2))),1e-12)
                        cfg=Settings(probe_count=probes,noise_rms=max(cal.noise_rms,1.))
                        state=q[index].copy();c=MotionController(cfg).start(plant.value(state,scenario['offset'])+noise[0],np.zeros(K))
                        previous=np.zeros(2);aborted=False;extra=plant.outside_coverage(state);first=0 if np.max(abs(state))<=.09 else -1
                        states=[state.copy()];commands=[]
                        for turn in range(1,21):
                            u=c.next_action()
                            if u is None:break
                            effective=np.asarray(scenario['actuator'])@u
                            reversal=(np.sign(effective)!=np.sign(previous))&(np.sign(previous)!=0)&(abs(effective)>1e-12)
                            effective=np.sign(effective)*np.maximum(abs(effective)-reversal*np.asarray(scenario['backlash']),0.)
                            previous=np.where(abs(effective)>1e-12,effective,previous);state+=effective
                            commands.append(u.tolist());states.append(state.copy());extra|=plant.outside_coverage(state)
                            if np.any(state<low)|np.any(state>high):
                                c.observe(np.full(K,np.nan),u);aborted=True;break
                            c.observe(plant.value(state,scenario['offset'])+noise[turn]+turn*scenario['drift']*drift,u)
                            if first<0 and np.max(abs(state))<=.09+1e-12:first=turn
                        ok=bool(not aborted and np.max(abs(state))<=.09+1e-12)
                        rows.append(dict(case_id=len(rows),condition=cond,fold=fold,index=int(index),dataset=int(m.iloc[index].dataset),scenario=scenario['name'],policy=f'motion_probe{probes}',turns=c.turn,status=c.status,Y_abs=abs(state[0]),T_abs=abs(state[1]),passed=ok,visual_stop=c.status=='visual_goal_confirmed',false_visual_stop=c.status=='visual_goal_confirmed' and not ok,first_true_tolerance_turn=first,aborted=aborted,extrapolated=extra,response_snr=c.diagnostics.get('response_snr'),identification_updates=c.identification_updates))
                        traces.append(dict(case_id=len(rows)-1,latent_q=np.array(states).tolist(),acknowledged_commands=commands,events=c.events))
            print('surrogate',cond,fold,'complete',flush=True)
    df=pd.DataFrame(rows);df.to_csv(H/'simulation_cases.csv',index=False)
    sums=[]
    for (cond,scenario,policy),g in df.groupby(['condition','scenario','policy']):
        for dataset in ['all',20261014,20261020]:
            z=g if dataset=='all' else g[g.dataset==dataset]
            sums.append(dict(condition=cond,scenario=scenario,policy=policy,dataset=dataset,**summarize(z),mean_turns=float(z.turns.mean()),visual_stops=int(z.visual_stop.sum()),false_visual_stops=int(z.false_visual_stop.sum()),insufficient_separation=int((z.status=='insufficient_separation').sum()),aborted=int(z.aborted.sum()),extrapolated=int(z.extrapolated.sum())))
    pd.DataFrame(sums).to_csv(H/'simulation_summary.csv',index=False)
    # Keep every executed increment and feature delta; archive truth remains evaluator-only.
    import gzip
    with gzip.open(H/'simulation_traces.json.gz','wt',encoding='utf-8') as f:json.dump(traces,f,allow_nan=True)

def main():
    started=time.perf_counter();m=pd.read_csv(OLD/'manifest.csv');a=np.load(OLD/'features.npz')
    original={str(f.relative_to(OLD)):hashlib.sha256(f.read_bytes()).hexdigest() for f in OLD.rglob('*') if f.is_file() and '__pycache__' not in f.parts}
    protocol=dict(date='2026-10-26',purpose='Separate executed-motion response identification and feedback logic',settings=asdict(Settings()),frozen_before_evaluation=True,online_inputs=['whitened current visual vector','original fixed visual goal','acknowledged relative Y/T increments'],offline_inputs='TRAIN-only projection and repeat noise inherited from20261023. Pretrained Jacobian and single-image pose model are NOT passed to MotionController.',recorded='All36 rectangles x all3^frames repeats x4observation conditions; all12rectangle images purged before projection/noise fit;3/4frames. Relative differences from archive labels construct a static benchmark, not real robot feedback.',recorded_score='Error in current pose inferred as negative proposed full correction, before gain/cap. No measured post-correction endpoint. Rejections remain in denominator; estimated-only errors accompanied by coverage.',simulation='Four previous empirical surrogate scenarios, same folds/seeds/341initial states/small probes/caps as prior baseline. Initial and all later observations are generated features, not real image replay.',signal_gate='Observed two-axis response singular value / train noise and normalized input rank; diagnostic only, not a calibrated confidence interval.',third_probe='Return by -(probeY,probeT); this tests repeat consistency but does not add a third control axis.',stop='Two visual checks, feature RMS<=2.5 and estimated correction<=.03; no true pose passed to stop.',unit='Archive command units; defaults are simulation assumptions. Image CLI requires user-supplied probe amplitudes and caps.',actual_robot_actions=0,independent_new_images=0,interpretation='Exposed development data. No independent or actual20turn success claim.',source_sha256={str(f.relative_to(H)):hashlib.sha256(f.read_bytes()).hexdigest() for f in H.glob('*.py')})
    (H/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    recorded(m,a);simulate(m)
    changed=[r for r,digest in original.items() if hashlib.sha256((OLD/r).read_bytes()).hexdigest()!=digest]
    assert not changed,changed
    (H/'preservation.json').write_text(json.dumps(dict(checked=len(original),changed=changed,seconds=time.perf_counter()-started),indent=2))
    print('completed seconds',time.perf_counter()-started,flush=True)

if __name__=='__main__':
    with threadpool_limits(1):main()
