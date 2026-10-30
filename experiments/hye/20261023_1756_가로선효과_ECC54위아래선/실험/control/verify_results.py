"""Recompute saved simulation accounting without running or tuning controllers."""
from pathlib import Path
import hashlib,json
import joblib,numpy as np,pandas as pd
from threadpoolctl import threadpool_limits
from controller import Calibration
from plant import HeldoutPlant

def check_directory(directory):
    cases=pd.read_csv(directory/'simulation_cases.csv');a=dict(np.load(directory/'simulation_trajectories.npz'));s=pd.read_csv(directory/'simulation_summary.csv',dtype={'dataset':str})
    assert len(cases)==16368 and cases.case_id.is_unique and np.array_equal(cases.case_id,a['case_id'])
    assert cases.groupby(['condition','policy','scenario']).size().eq(341).all()
    assert cases.turns.between(0,20).all()
    for row in cases.itertuples():
        qs=a['latent_q'][row.case_id];us=a['executed_relative_commands'][row.case_id]
        if row.status=='input_measurement_failed':assert not row.passed;continue
        assert np.isfinite(qs[:row.turns+1]).all() and np.isnan(qs[row.turns+1:]).all()
        assert np.isfinite(us[:row.turns]).all() and np.isnan(us[row.turns:]).all()
        assert np.allclose(qs[row.turns],[row.Y_final,row.T_final],rtol=0,atol=1e-12)
        assert row.passed==bool(not row.aborted and np.max(abs(qs[row.turns]))<=.09+1e-12)
        assert row.false_visual_stop==bool(row.visual_stop and not row.passed)
        reached=np.flatnonzero(np.all(abs(qs[:row.turns+1])<=.09+1e-12,axis=1))
        expected=int(reached[0]) if len(reached) else -1
        assert row.first_true_tolerance_turn==expected
        assert np.isclose(np.nansum(abs(us)),row.total_command_travel,rtol=0,atol=1e-12)
        expected_probe=3 if 'probe3' in row.policy else (2 if 'probe2' in row.policy else 0)
        assert row.probe_turns==expected_probe
    for summary in s.to_dict('records'):
        z=cases[(cases.condition==summary['condition'])&(cases.scenario==summary['scenario'])&(cases.policy==summary['policy'])]
        if summary['dataset']!='all':z=z[z.dataset==int(summary['dataset'])]
        assert len(z)==summary['n'] and int(z.passed.sum())==summary['passed']
        for key,column,method in [('Y_MAE','Y_abs','mean'),('Y_max','Y_abs','max'),('T_MAE','T_abs','mean'),('T_max','T_abs','max'),('mean_turns','turns','mean')]:
            assert np.isclose(summary[key],getattr(z[column],method)(),atol=1e-12,rtol=1e-10)
    return dict(cases=len(cases),all_case_denominators341=True,all_turns_at_most20=True,probe_turns_included=True,all_terminal_labels_recomputed=True,false_visual_stops_recomputed=True,all_summary_arithmetic_recomputed=True)

def main():
    here=Path(__file__).resolve().parent;root=here.parent
    a=np.load(root/'features.npz');m=pd.read_csv(root/'manifest.csv');model_checks=[]
    for condition in ['qr','upper','lower','both']:
        for fold in range(4):
            bundle=joblib.load(here/f'{condition}_fold{fold}.joblib');cal=bundle['calibration'];plant=bundle['plant']
            tr=np.flatnonzero(m.fold.to_numpy()!=fold);te=np.flatnonzero(m.fold.to_numpy()==fold)
            assert np.array_equal(cal.training_indices,tr) and np.array_equal(plant.indices,te)
            assert not set(m.iloc[tr].cv_group)&set(m.iloc[te].cv_group)
            fresh=Calibration.fit(a['X_'+condition][tr],m.iloc[tr][['Y','T']].to_numpy(),m.iloc[tr].dataset.to_numpy(),m.iloc[tr]['index'].to_numpy())
            assert np.array_equal(cal.projection,fresh.projection) and np.array_equal(cal.jacobian,fresh.jacobian)
            fresh_plant=HeldoutPlant(fresh.project(a['X_'+condition][te]),m.iloc[te][['Y','T']].to_numpy(),m.iloc[te].dataset.to_numpy(),m.iloc[te]['index'].to_numpy())
            assert np.array_equal(plant.B,fresh_plant.B) and np.array_equal(plant.residual,fresh_plant.residual)
            model_checks.append(dict(condition=condition,fold=fold,train=len(tr),plant_heldout=len(te),sample_overlap=False,reloaded_and_refitted_exactly=True))
    prior=json.loads((here/'first_run_result_hashes.json').read_text());assert all(hashlib.sha256((here/name).read_bytes()).hexdigest()==digest for name,digest in prior.items())
    result=dict(scope='Simulation bookkeeping and split audit; not independent physical validation',primary=check_directory(here),model_checks=model_checks,primary_results_unchanged_by_serialization_fix=True,real_robot_actions=0)
    if (here/'exploratory/simulation_summary.csv').exists():result['exploratory']=check_directory(here/'exploratory')
    (here/'results_verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))

if __name__=='__main__':
    with threadpool_limits(1):main()
