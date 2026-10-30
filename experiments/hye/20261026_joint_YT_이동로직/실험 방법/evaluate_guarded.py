"""Post-primary guarded alternatives; preserve the online-only results."""
from pathlib import Path
import sys,json,hashlib,gzip,time
from dataclasses import asdict
import numpy as np,pandas as pd,joblib
from threadpoolctl import threadpool_limits
from evaluate import OLD,H,CONDITIONS,summarize
from simulate import SCENARIOS
from motion import MotionController,Settings,choose_probe_amplitudes

def main():
    protected={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in H.glob('*') if f.is_file() and (f.suffix in ('.csv','.json') or f.name=='simulation_traces.json.gz')}
    m=pd.read_csv(OLD/'manifest.csv');q=m[['Y','T']].to_numpy();low=q.min(0)-.25;high=q.max(0)+.25
    variants=[('guarded_small3',3,False),('guarded_adaptive2',2,True),('guarded_adaptive3',3,True)]
    protocol=dict(scope='Exploratory extension after inspecting online-only failure. Not independent evaluation or actual robot control.',variants=variants,settings=asdict(Settings()),adaptive='TRAIN-only perpendicular response sets amplitude for nominal3x distinguishing noise; clipped to [.05,.01] minima and [.25,.25] caps. These are assumed simulator caps, not commissioned robot limits.',prior='TRAIN-only Jacobian. A rejected online update retains the last accepted model/prior. It does not pretend the session alone identified the axes.',gate='Same observed two-axis response SNR>=1.5, condition<=1000; additional normalized relative-response change<=.5; ridge2 in normalized input units.',comparison='Same16 stored fold calibration/plant bundles,341starting states, four scenarios and camera-noise seeds as20261023. Three policies selected before this extension run.',ground_truth='Only plant/evaluator see true q. Initial truth never initializes controller.',source_hashes={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in [H/'motion.py',Path(__file__)]},robot_actions=0)
    (H/'guarded_protocol.json').write_text(json.dumps(protocol,indent=2))
    rows=[];probeinfo=[]
    with gzip.open(H/'guarded_traces.json.gz','wt',encoding='utf-8') as trace:
        for cond in CONDITIONS:
            for fold in range(4):
                bundle=joblib.load(OLD/'control'/f'{cond}_fold{fold}.joblib');cal=bundle['calibration'];plant=bundle['plant'];K=cal.jacobian.shape[0]
                assert not set(cal.training_indices)&set(plant.indices)
                noise_rms=max(cal.noise_rms,1.)
                adaptive=choose_probe_amplitudes(cal.jacobian,[.05,.01],[.25,.25],noise_rms)
                probeinfo.append(dict(condition=cond,fold=fold,adaptive_probe=adaptive.tolist(),noise_rms=noise_rms,train_indices=cal.training_indices.tolist(),plant_indices=plant.indices.tolist()))
                for sid,scenario in enumerate(SCENARIOS):
                    for index in np.flatnonzero(m.fold.to_numpy()==fold):
                        for policy,probes,adapt in variants:
                            rng=np.random.default_rng(20261023+1000*sid+int(index))
                            noise=plant.residual[rng.integers(len(plant.residual),size=21)]*scenario['noise']
                            drift=rng.normal(size=K);drift/=max(float(np.sqrt(np.mean(drift**2))),1e-12)
                            amp=adaptive if adapt else np.array([.05,.01]);cfg=Settings(probe_y=amp[0],probe_t=amp[1],probe_count=probes,noise_rms=noise_rms)
                            state=q[index].copy();c=MotionController(cfg,cal.jacobian).start(plant.value(state,scenario['offset'])+noise[0],np.zeros(K))
                            previous=np.zeros(2);aborted=False;extra=plant.outside_coverage(state);first=0 if np.max(abs(state))<=.09+1e-12 else -1
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
                            row=dict(case_id=len(rows),condition=cond,fold=fold,index=int(index),dataset=int(m.iloc[index].dataset),scenario=scenario['name'],policy=policy,turns=c.turn,status=c.status,Y_abs=abs(state[0]),T_abs=abs(state[1]),passed=ok,visual_stop=c.status=='visual_goal_confirmed',false_visual_stop=c.status=='visual_goal_confirmed' and not ok,first_true_tolerance_turn=first,aborted=aborted,extrapolated=extra,response_snr=c.diagnostics.get('response_snr'),identification_updates=c.identification_updates)
                            rows.append(row);json.dump(dict(case_id=row['case_id'],latent_q=np.array(states).tolist(),acknowledged_commands=commands,events=c.events),trace,allow_nan=True);trace.write('\n')
                print('guarded',cond,fold,'complete',flush=True)
    df=pd.DataFrame(rows);df.to_csv(H/'guarded_cases.csv',index=False);sums=[]
    for (cond,scenario,policy),g in df.groupby(['condition','scenario','policy']):
        for dataset in ['all',20261014,20261020]:
            z=g if dataset=='all' else g[g.dataset==dataset]
            sums.append(dict(condition=cond,scenario=scenario,policy=policy,dataset=dataset,**summarize(z),mean_turns=float(z.turns.mean()),visual_stops=int(z.visual_stop.sum()),false_visual_stops=int(z.false_visual_stop.sum()),aborted=int(z.aborted.sum()),extrapolated=int(z.extrapolated.sum()),mean_online_updates=float(z.identification_updates.mean())))
    pd.DataFrame(sums).to_csv(H/'guarded_summary.csv',index=False)
    (H/'guarded_probe_training.json').write_text(json.dumps(probeinfo,indent=2))
    changed=[r for r,d in protected.items() if hashlib.sha256((H/r).read_bytes()).hexdigest()!=d]
    assert not changed,changed
    (H/'guarded_preservation.json').write_text(json.dumps(dict(primary_files_checked=len(protected),changed=changed,cases=len(rows)),indent=2))
if __name__=='__main__':
    with threadpool_limits(1):main()
