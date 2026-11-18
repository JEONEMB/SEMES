"""One three-candidate exploratory run; primary simulations remain unchanged."""
from pathlib import Path
import hashlib,json,sys,time
import joblib,numpy as np,pandas as pd
from threadpoolctl import threadpool_limits
HERE=Path(__file__).resolve().parent;CONTROL=HERE.parent;ROOT=CONTROL.parent
sys.path.insert(0,str(CONTROL))
from simulate import trajectory,SCENARIOS,protocol as primary_protocol
from rls_controller import GuardedRLS,POLICIES,SETTINGS

class CalibrationView:
    def __init__(self,original):self.jacobian=original.jacobian
    def controller(self,policy):
        self.last_controller=GuardedRLS(self.jacobian,policy);return self.last_controller

def main():
    start=time.perf_counter();m=pd.read_csv(ROOT/'manifest.csv');q=m[['Y','T']].to_numpy();a=np.load(ROOT/'features.npz')
    preserved={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in CONTROL.glob('*') if p.is_file()}
    protocol=dict(scope='EXPLORATORY ASSUMPTION SIMULATION added after primary results, not independent validation',parent_protocol=primary_protocol(),policies=list(POLICIES),settings=SETTINGS,
                  baseline_source_sha256=preserved,physical_permission='Probe .20/.04 and step bounds are only simulated policy settings, not approved physical robot movements',
                  initial_observations='Surrogate-generated at341 archived initial command values, not replay of341 observed images',
                  noise_gate='Both command axes must be excited; minimum train-predicted and observed weak-direction SNR>=1.5; regularized secants must improve fit beyond assumed noise floor and remain within50% of prior relative response',
                  no_followup_tuning=True)
    (HERE/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    rows=[];states=[];actions=[];rms=[];inferred=[]
    low=q.min(0)-.25;high=q.max(0)+.25
    for condition in ['qr','upper','lower','both']:
        good=a['success_'+condition].astype(bool)&np.isfinite(a['X_'+condition]).all(1)
        for fold in range(4):
            bundle=joblib.load(CONTROL/f'{condition}_fold{fold}.joblib');view=CalibrationView(bundle['calibration']);plant=bundle['plant'];te=np.flatnonzero(m.fold.to_numpy()==fold)
            assert set(plant.indices)==set(te[good[te]])
            for scenario_id,scenario in enumerate(SCENARIOS):
                for index in te:
                    for policy in POLICIES:
                        base=dict(case_id=len(rows),condition=condition,fold=fold,index=int(index),dataset=int(m.iloc[index].dataset),scenario=scenario['name'],policy=policy,initialY=float(q[index,0]),initialT=float(q[index,1]))
                        if good[index]:
                            result,qs,us,rs,ins=trajectory(view,plant,q[index],scenario,policy,20261023+1000*scenario_id+int(index),low,high)
                            c=view.last_controller;result.update(updates_attempted=c.update_attempts,updates_accepted=c.updates_accepted,last_two_axis_SNR=c.last_two_axis_SNR,**{'gate_'+k:v for k,v in c.gate_counts.items()})
                        else:
                            result=dict(turns=0,probe_turns=0,confirmation_turns=0,status='input_measurement_failed',Y_final=None,T_final=None,Y_abs=None,T_abs=None,passed=False,first_true_tolerance_turn=-1,visual_stop=False,false_visual_stop=False,extrapolated=False,aborted=False,measurement_failed=True,total_command_travel=0.,final_visual_RMS=None,updates_attempted=0,updates_accepted=0,last_two_axis_SNR=0.)
                            qs=np.full((21,2),np.nan);us=np.full((20,2),np.nan);rs=np.full(21,np.nan);ins=np.full((21,2),np.nan)
                        rows.append(dict(**base,**result));states.append(qs);actions.append(us);rms.append(rs);inferred.append(ins)
            print(condition,fold,'done',round(time.perf_counter()-start,1),flush=True)
    result=pd.DataFrame(rows);result.to_csv(HERE/'simulation_cases.csv',index=False)
    np.savez_compressed(HERE/'simulation_trajectories.npz',case_id=result.case_id.to_numpy(),latent_q=np.array(states),executed_relative_commands=np.array(actions),visual_RMS=np.array(rms),inferred_local_correction=np.array(inferred))
    summaries=[]
    for keys,g in result.groupby(['condition','scenario','policy']):
        for dataset in ['all',20261014,20261020]:
            z=g if dataset=='all' else g[g.dataset==dataset];valid=z[['Y_abs','T_abs']].notna().all(1);v=z[valid]
            summaries.append(dict(condition=keys[0],scenario=keys[1],policy=keys[2],dataset=dataset,n=len(z),passed=int(z.passed.sum()),Y_MAE=float(v.Y_abs.mean()),Y_max=float(v.Y_abs.max()),T_MAE=float(v.T_abs.mean()),T_max=float(v.T_abs.max()),mean_turns=float(z.turns.mean()),mean_success_turns=float(z[z.passed].turns.mean()) if z.passed.any() else None,visual_stops=int(z.visual_stop.sum()),false_visual_stops=int(z.false_visual_stop.sum()),measurement_failures=int(z.measurement_failed.sum()),aborted=int(z.aborted.sum()),extrapolated=int(z.extrapolated.sum()),mean_command_travel=float(z.total_command_travel.mean()),mean_accepted_updates=float(z.updates_accepted.mean()),cases_with_accepted_updates=int(z.updates_accepted.gt(0).sum())))
    pd.DataFrame(summaries).to_csv(HERE/'simulation_summary.csv',index=False)
    assert all(hashlib.sha256((CONTROL/n).read_bytes()).hexdigest()==digest for n,digest in preserved.items()),'Primary artifacts modified'
    audit=dict(status='complete',scope=protocol['scope'],cases=len(result),primary_files_preserved=len(preserved),seconds=time.perf_counter()-start,all_primary_file_hashes_unchanged=True,real_robot_actions=0)
    (HERE/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8');print(json.dumps(audit),flush=True)

if __name__=='__main__':
    with threadpool_limits(1):main()
