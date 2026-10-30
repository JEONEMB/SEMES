"""ASSUMPTION SIMULATION ONLY, using disjoint held-out forward surrogates.

No actual robot trajectory exists in this archive. Heldout labels/residuals are
used to create a synthetic plant and score it, never passed into the controller.
Do not report these results as measured return-to-center performance.
"""
from pathlib import Path
import argparse,hashlib,json,sys,time
import joblib,numpy as np,pandas as pd
from threadpoolctl import threadpool_limits
from controller import Calibration,FeedbackController,POLICIES,PARAMETERS,polynomial,grouped_rows
from plant import HeldoutPlant

HERE=Path(__file__).resolve().parent
SCENARIOS=[dict(name='ideal_zero_offset',offset=False,noise=0.,actuator=[[1.,0.],[0.,1.]],drift=0.,backlash=[0.,0.]),
           dict(name='repeat_noise_zero_offset',offset=False,noise=1.,actuator=[[1.,0.],[0.,1.]],drift=0.,backlash=[0.,0.]),
           dict(name='offset_gain_crossaxis_noise',offset=True,noise=1.,actuator=[[.85,.08],[-.04,1.12]],drift=0.,backlash=[0.,0.]),
           dict(name='offset_gain_crossaxis_drift_backlash_noise',offset=True,noise=1.,actuator=[[.85,.08],[-.04,1.12]],drift=.05,backlash=[.02,.005])]

def protocol():
    return dict(scope='ASSUMPTION SIMULATION ONLY; all341 static images are adaptively exposed DEVELOPMENT data',
                no_real_robot_or_continuous_trajectory=True,policies=list(POLICIES),parameters=PARAMETERS,scenarios=SCENARIOS,
                fixed_before_results=True,current_input='Projected observed feature s, fixed zero reference s0, executed relative command u only',
                controller_fit='Outer TRAIN images/commands only: feature projection, repeat covariance, linear Jacobian',
                plant_fit='Disjoint outer HELDOUT images/commands: separate six-coefficient quadratic forward map, including intercept; heldout repeated-condition residuals are simulated camera noise',
                inverse_crime_control='Controller and plant never share fitted response coefficients or calibration rows. Plant is still only an empirical smooth surrogate, not a physical simulator.',
                reference='Nominal fixed-center feature zero. First2 scenarios force plant intercept to zero as an ideal assumption; last2 retain its independently fitted intercept.',
                turns='Initial observation0; each suggested increment and observation counts1, including2/3 probes and zero-motion confirmation captures; maximum20',
                stop='Two consecutive visual checks: whitened feature RMS<=2.5 and local inferred correction<=.03, after probes. Absolute latent q NEVER controls stopping.',
                score='External evaluator checks terminal |qY|<=.09 AND |qT|<=.09. First latent attainment is diagnostic only and does not stop controller.',
                executed_u='Acknowledged relative command; injected actuator gain/cross-coupling/backlash is unknown to controller, not measured mechanical displacement.',
                extrapolation='Any latent pose outside convex hull of heldout commands plus required0 center marks extrapolation. Beyond full corpus box expanded by.25 aborts simulation; neither bound is a verified robot safety limit.',
                units='Archive command units. Probe amplitudes, max-step, actuator dynamics and drift are experiment assumptions, not actual robot allowances.',
                time='No cycle-time claim: capture, settling and robot motion times have not been measured.')

def trajectory(calibration,plant,q_initial,scenario,policy,seed,global_low,global_high):
    rng=np.random.default_rng(seed);N=PARAMETERS['max_turns'];K=calibration.jacobian.shape[0]
    noise=plant.residual[rng.integers(len(plant.residual),size=N+1)]*scenario['noise']
    drift=rng.normal(size=K);drift/=max(float(np.sqrt(np.mean(drift**2))),1e-12)
    q=np.array(q_initial,float);states=np.full((N+1,2),np.nan);actions=np.full((N,2),np.nan);rms=np.full(N+1,np.nan);inferred=np.full((N+1,2),np.nan)
    controller=calibration.controller(policy);controller.start(plant.value(q,scenario['offset'])+noise[0],np.zeros(K))
    states[0]=q;rms[0]=controller.feature_rms;inferred[0]=controller.correction;previous=np.zeros(2)
    first_true=0 if np.max(abs(q))<=.09+1e-12 else -1;extrapolated=plant.outside_coverage(q);aborted=False
    probe_turns=0;confirm_turns=0
    for turn in range(1,N+1):
        u=controller.next_action()
        if u is None:break
        probe_turns+=controller.last_kind=='probe';confirm_turns+=controller.last_kind=='confirm';actions[turn-1]=u
        effective=np.asarray(scenario['actuator'])@u
        reversal=(np.sign(effective)!=np.sign(previous))&(np.sign(previous)!=0)&(abs(effective)>1e-12)
        effective=np.sign(effective)*np.maximum(abs(effective)-reversal*np.asarray(scenario['backlash']),0.)
        previous=np.where(abs(effective)>1e-12,effective,previous);q=q+effective
        states[turn]=q;extrapolated|=plant.outside_coverage(q)
        if np.any(q<global_low)|np.any(q>global_high):
            controller.observe(np.full(K,np.nan),u);aborted=True;break
        s=plant.value(q,scenario['offset'])+noise[turn]+turn*scenario['drift']*drift
        controller.observe(s,u);rms[turn]=controller.feature_rms;inferred[turn]=controller.correction
        if first_true<0 and np.max(abs(q))<=.09+1e-12:first_true=turn
    terminal=controller.turn;valid=np.isfinite(q).all() and not aborted
    success=bool(valid and np.max(abs(q))<=.09+1e-12)
    result=dict(turns=terminal,probe_turns=probe_turns,confirmation_turns=confirm_turns,status=controller.status,
                Y_final=float(q[0]),T_final=float(q[1]),Y_abs=float(abs(q[0])),T_abs=float(abs(q[1])),passed=success,
                first_true_tolerance_turn=first_true,visual_stop=controller.status=='visual_goal_confirmed',
                false_visual_stop=controller.status=='visual_goal_confirmed' and not success,
                extrapolated=extrapolated,aborted=aborted,measurement_failed=controller.status=='measurement_failed',
                total_command_travel=float(np.nansum(abs(actions))),final_visual_RMS=float(rms[terminal]) if np.isfinite(rms[terminal]) else None)
    return result,states,actions,rms,inferred

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--features',type=Path,required=True);parser.add_argument('--manifest',type=Path,required=True);args=parser.parse_args()
    started=time.perf_counter();a=np.load(args.features);m=pd.read_csv(args.manifest);q=m[['Y','T']].to_numpy();assert len(m)==341
    if 'index' in a:assert np.array_equal(a['index'],m['index'])
    (HERE/'protocol.json').write_text(json.dumps(protocol(),indent=2),encoding='utf-8')
    rows=[];diagnostics=[];trajectories=[];actions=[];rmss=[];inferreds=[]
    global_low=q.min(0)-.25;global_high=q.max(0)+.25
    for condition in ['qr','upper','lower','both']:
        X=a['X_'+condition];good=a['success_'+condition].astype(bool)&np.isfinite(X).all(1)
        for fold in range(4):
            tr=np.flatnonzero((m.fold.to_numpy()!=fold)&good);te=np.flatnonzero(m.fold.to_numpy()==fold);fit_te=te[good[te]]
            assert not set(m.iloc[tr].cv_group)&set(m.iloc[fit_te].cv_group)
            calibration=Calibration.fit(X[tr],q[tr],m.iloc[tr].dataset.to_numpy(),m.iloc[tr]['index'].to_numpy())
            projected=calibration.project(X[fit_te]);plant=HeldoutPlant(projected,q[fit_te],m.iloc[fit_te].dataset.to_numpy(),m.iloc[fit_te]['index'].to_numpy())
            assert not set(calibration.training_indices)&set(plant.indices)
            joblib.dump(dict(calibration=calibration,plant=plant,protocol=protocol()),HERE/f'{condition}_fold{fold}.joblib')
            diagnostics.append(dict(condition=condition,fold=fold,controller=calibration.diagnostics,plant=plant.diagnostic,controller_train_indices=calibration.training_indices.tolist(),plant_test_indices=plant.indices.tolist(),index_overlap=False))
            for scenario_id,scenario in enumerate(SCENARIOS):
                for index in te:
                    for policy in POLICIES:
                        base=dict(case_id=len(rows),condition=condition,fold=fold,index=int(index),dataset=int(m.iloc[index].dataset),scenario=scenario['name'],policy=policy,initialY=float(q[index,0]),initialT=float(q[index,1]))
                        if good[index]:
                            result,states,us,rms,estimated=trajectory(calibration,plant,q[index],scenario,policy,20261023+1000*scenario_id+int(index),global_low,global_high)
                        else:
                            result=dict(turns=0,probe_turns=0,confirmation_turns=0,status='input_measurement_failed',Y_final=None,T_final=None,Y_abs=None,T_abs=None,passed=False,first_true_tolerance_turn=-1,visual_stop=False,false_visual_stop=False,extrapolated=False,aborted=False,measurement_failed=True,total_command_travel=0.,final_visual_RMS=None)
                            states=np.full((21,2),np.nan);us=np.full((20,2),np.nan);rms=np.full(21,np.nan);estimated=np.full((21,2),np.nan)
                        rows.append(dict(**base,**result));trajectories.append(states);actions.append(us);rmss.append(rms);inferreds.append(estimated)
            print(condition,fold,'done',len(rows),'seconds',round(time.perf_counter()-started,1),flush=True)
    result=pd.DataFrame(rows);result.to_csv(HERE/'simulation_cases.csv',index=False)
    np.savez_compressed(HERE/'simulation_trajectories.npz',case_id=result.case_id.to_numpy(),latent_q=np.array(trajectories),executed_relative_commands=np.array(actions),visual_RMS=np.array(rmss),inferred_local_correction=np.array(inferreds))
    summaries=[]
    for keys,g in result.groupby(['condition','scenario','policy']):
        for dataset in ['all',20261014,20261020]:
            z=g if dataset=='all' else g[g.dataset==dataset];valid=z[['Y_abs','T_abs']].notna().all(1);v=z[valid]
            summaries.append(dict(condition=keys[0],scenario=keys[1],policy=keys[2],dataset=dataset,n=len(z),passed=int(z.passed.sum()),Y_MAE=float(v.Y_abs.mean()),Y_max=float(v.Y_abs.max()),T_MAE=float(v.T_abs.mean()),T_max=float(v.T_abs.max()),mean_turns=float(z.turns.mean()),mean_success_turns=float(z[z.passed].turns.mean()) if z.passed.any() else None,visual_stops=int(z.visual_stop.sum()),false_visual_stops=int(z.false_visual_stop.sum()),measurement_failures=int(z.measurement_failed.sum()),aborted=int(z.aborted.sum()),extrapolated=int(z.extrapolated.sum()),mean_command_travel=float(z.total_command_travel.mean())))
    pd.DataFrame(summaries).to_csv(HERE/'simulation_summary.csv',index=False)
    audit=dict(scope=protocol()['scope'],diagnostics=diagnostics,features_sha256=hashlib.sha256(args.features.read_bytes()).hexdigest(),manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),cases=len(result),all_conditions_same341starts=True,number_of_real_robot_actions=0,seconds=time.perf_counter()-started)
    (HERE/'simulation_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8');print('completed',len(result),flush=True)

if __name__=='__main__':
    with threadpool_limits(1):main()
