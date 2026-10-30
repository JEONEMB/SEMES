from pathlib import Path
import sys, json, inspect
import numpy as np
import pandas as pd
import joblib
from threadpoolctl import threadpool_limits
H=Path(__file__).resolve().parent
sys.path.insert(0,str(H.parents[1]/'joint_yt_generalization_20261022'))
from data import load,metric
from robust_model import RobustBaggedForward

def main():
    m,a=load();p=pd.read_csv(H/'exploratory_nested_predictions.csv')
    s=json.loads((H/'exploratory_nested_selection.json').read_text())
    previous=json.loads((H.parents[1]/'joint_yt_disentangle_20261022/selection.json').read_text())
    max_delta=0.
    for f in range(4):
        now=next(x for x in s if x.get('fold')==f)
        before=next(x for x in previous if x.get('fold')==f)
        assert now['inner_splits']==before['inner_splits'] and now['seed']==before['seed']
        chosen=min(now['scores'],key=lambda r:(r['failures'],r['maximum'],r['mean']))['candidate']
        assert chosen==now['selected']
        b=joblib.load(H/f'exploratory_nested_fold{f}.joblib')
        tr=b['train_indices'];te=np.flatnonzero(m.fold.to_numpy()==f)
        assert set(tr)==set(np.flatnonzero(m.fold.to_numpy()!=f))
        assert not set(m.iloc[tr].cv_group)&set(m.iloc[te].cv_group)
        saved=p[(p.protocol=='exploratory_nested16_oof')&(p.fold==f)].set_index('index').loc[te]
        rec=b['model'].predict(a['D'][te])
        delta=float(abs(rec-saved[['predY','predT']].to_numpy()).max())
        assert delta<1e-8;max_delta=max(max_delta,delta)
    full=next(x['full_selection'] for x in s if 'full_selection' in x)
    oldfull=next(x['full_selection'] for x in previous if 'full_selection' in x)
    assert full['inner_splits']==oldfull['inner_splits'] and full['seed']==oldfull['seed']
    assert list(inspect.signature(RobustBaggedForward.predict).parameters)==['self','D']
    summary=pd.read_csv(H/'exploratory_nested_summary.csv')
    for _,row in summary.iterrows():
        z=p[p.protocol==row.protocol]
        if str(row.dataset)!='all':z=z[z.dataset==int(row.dataset)]
        q=metric(z[['Y','T']],z[['predY','predT']])
        for key,value in q.items():assert np.isclose(row[key],value,atol=1e-10)
    out=dict(status='passed',exact_prior_inner_splits_and_seeds=True,
             selection_recomputed=True,all341_outer_predictions_recomputed=True,
             maximum_saved_model_prediction_difference=max_delta,
             no_group_overlap=True,predict_signature='predict(self,D)',
             summary_recomputed=True,independent_test=False)
    (H/'exploratory_nested_verification.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    print(json.dumps(out,indent=2))

if __name__=='__main__':
    with threadpool_limits(1):main()
