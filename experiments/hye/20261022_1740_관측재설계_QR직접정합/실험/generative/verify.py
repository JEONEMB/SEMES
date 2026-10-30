"""Synthetic inverse-map checks and saved real-development prediction audit."""
from pathlib import Path
import json,sys
import joblib,numpy as np,pandas as pd
from threadpoolctl import threadpool_limits
H=Path(__file__).resolve().parent
sys.path.insert(0,str(H.parents[1]/'joint_yt_generalization_20261022'))
from data import load
from generative_model import GenerativeProjective, CONFIGS
def metrics(y,p):
    error=np.abs(np.asarray(y)-np.asarray(p));good=np.isfinite(error).all(1);valid=error[good]
    return dict(n=len(error),failures=int((~good).sum()),passed=int(np.sum(good&np.all(error<=.09+1e-12,axis=1))),Y_MAE=float(valid[:,0].mean()),Y_max=float(valid[:,0].max()),T_MAE=float(valid[:,1].mean()),T_max=float(valid[:,1].max()))

def main():
    m,a=load();rng=np.random.default_rng(18273);checks=[]
    for cfg in [CONFIGS[1],CONFIGS[2]]:
        source=GenerativeProjective(cfg,a['points'],a['regions']);source.coefs=[]
        for j in range(3):
            if cfg['family']=='rational':
                coef=np.zeros((4,8));coef[0]=np.eye(3).ravel()[:8]
                coef[0,2]=.001*(j+1);coef[1,[2,4,6]]=[.12,.025,.001];coef[2,[1,2,3,5]]=[-.04,.7,.04,.02]
                coef[3,[0,5,7]]=[.002,.003,.0003]
            else:
                coef=np.zeros((3,8));coef[0,2]=.001*(j+1)
                coef[1,[2,4,6]]=[.12,.025,.001];coef[2,[1,2,3,5]]=[-.04,.7,.04,.02]
            source.coefs.append(coef.ravel())
        y=rng.uniform(-1,1,(60,2));truth=rng.uniform(-1.1,1.1,(12,2))
        training=source.generate_homographies(y,global_pixels=True);test=source.generate_homographies(truth,global_pixels=True)
        metadata=pd.DataFrame(dict(index=np.arange(60),dataset=20261020))
        fit=GenerativeProjective(cfg,a['points'],a['regions']).fit_homographies(training,y,metadata)
        pred=fit.predict_homographies(test);error=float(np.max(abs(pred-truth)))
        assert np.isfinite(pred).all() and error<1e-4
        broken=test.copy();broken[0]=np.nan;out=fit.predict_homographies(broken)
        assert np.isnan(out[0]).all() and np.isfinite(out[1:]).all()
        assert metrics(truth,out)['failures']==1 and metrics(truth,out)['n']==12
        checks.append(dict(check='synthetic_known_forward_map_inverse',family=cfg['family'],maximum_error=error,failed_observation_retained_in_denominator=True,scope='numerical software check only; not robot evidence'))
    audits=[];maxdiff=0.
    for directory in [p for p in H.iterdir() if p.is_dir() and (p/'summary.csv').exists()]:
        pred=pd.read_csv(directory/'predictions.csv');summ=pd.read_csv(directory/'summary.csv',dtype={'dataset':str});observed=np.load(directory/'observed_homographies.npz')['H']
        for candidate in range(4):
            z=pred[pred.candidate==candidate];assert len(z)==341 and z['index'].is_unique
            for fold in range(4):
                tr=np.flatnonzero(m.fold.to_numpy()!=fold);te=np.flatnonzero(m.fold.to_numpy()==fold)
                bundle=joblib.load(directory/f'candidate{candidate}_fold{fold}.joblib');model=bundle['model']
                assert np.array_equal(bundle['train_indices'],tr) and np.array_equal(model.training_indices,m.iloc[tr]['index'])
                assert not set(m.iloc[tr].cv_group)&set(m.iloc[te].cv_group)
                saved=z.set_index('index').loc[te];current=model.predict_homographies(observed[te])
                assert np.array_equal(np.isfinite(current),np.isfinite(saved[['predY','predT']].to_numpy()))
                diff=float(np.nanmax(abs(current-saved[['predY','predT']].to_numpy())));assert diff<1e-8;maxdiff=max(maxdiff,diff)
            for ds in ['all','20261014','20261020']:
                rows=z if ds=='all' else z[z.dataset==int(ds)];values=metrics(rows[['Y','T']],rows[['predY','predT']]);expected=summ[(summ.candidate==candidate)&(summ.dataset==ds)].iloc[0]
                assert all(np.isclose(expected[k],v,rtol=1e-10,atol=1e-12) for k,v in values.items())
                if ds=='20261020':audits.append(dict(experiment=directory.name,candidate=candidate,**values))
    result=dict(status='passed',synthetic_checks=checks,all_saved_predictions_reproduced=True,maximum_prediction_difference=maxdiff,all341_accounted_for_each_candidate=True,all_saved_training_indices_match_outer_training=True,mixed_condition_repeats_grouped=True,metrics_recomputed=True,development_only=True,mixed_metrics=audits)
    (H/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))

if __name__=='__main__':
    with threadpool_limits(1):main()
