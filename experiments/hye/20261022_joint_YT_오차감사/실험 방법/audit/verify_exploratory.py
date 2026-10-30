"""Read-only independent check of the subsequently selected development candidate."""
from pathlib import Path
import json, sys
import joblib, numpy as np, pandas as pd
from threadpoolctl import threadpool_limits
from evaluation_audit import sha, metric

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ROBUST = ROOT/'robust'
OLD = ROOT.parent/'joint_yt_disentangle_20261022'
sys.path.insert(0,str(ROOT.parent/'joint_yt_generalization_20261022'))
sys.path.insert(0,str(ROBUST))
from data import load
from robust_model import RobustBaggedForward

def main():
    m,a=load(); selections=json.loads((ROBUST/'exploratory_nested_selection.json').read_text())
    old=json.loads((OLD/'selection.json').read_text())
    pred=pd.read_csv(ROBUST/'exploratory_nested_predictions.csv')
    summary=pd.read_csv(ROBUST/'exploratory_nested_summary.csv',dtype={'dataset':str})
    maximum=0.; checks=[]; metrics=[]
    for fold in range(5):
        selection=selections[fold] if fold<4 else selections[-1]['full_selection']
        previous=old[fold] if fold<4 else old[-1]['full_selection']
        tr=np.flatnonzero(m.fold.to_numpy()!=fold) if fold<4 else np.arange(341)
        te=np.flatnonzero(m.fold.to_numpy()==fold) if fold<4 else np.arange(341)
        assert selection['inner_splits']==previous['inner_splits']
        assert len(selection['scores'])==16
        best=min(selection['scores'],key=lambda r:(r['failures'],r['maximum'],r['mean']))
        assert selection['selected']==best['candidate']
        for part in selection['inner_splits']:
            it,iv=set(part['train']),set(part['validation'])
            assert it|iv==set(tr) and not it&iv
            assert not set(m.iloc[list(it)].cv_group)&set(m.iloc[list(iv)].cv_group)
            if fold<4:assert not (it|iv)&set(te)
        path=ROBUST/(f'exploratory_nested_fold{fold}.joblib' if fold<4 else 'exploratory_nested_model.joblib')
        bundle=joblib.load(path)
        assert np.array_equal(bundle['train_indices'],tr)
        assert bundle['model'].audit['training_rows']==len(tr)
        assert bundle['config']==best['config']
        assert np.array_equal(bundle['points'],a['points']) and np.array_equal(bundle['ids'],a['ids'])
        protocol='exploratory_nested16_oof' if fold<4 else 'resub341'
        saved=pred[pred.protocol==protocol]
        if fold<4:saved=saved[saved.fold==fold]
        saved=saved.set_index('index').loc[te]
        assert np.allclose(saved[['Y','T']],m.iloc[te][['Y','T']])
        current=bundle['model'].predict(a['D'][te]);difference=float(np.max(abs(current-saved[['predY','predT']].to_numpy())))
        assert difference<1e-10
        maximum=max(maximum,difference)
        checks.append(dict(fold=fold if fold<4 else 'full_train',training_rows=len(tr),evaluation_rows=len(te),selected=selection['selected'],sha256=sha(path),prediction_max_difference=difference))
    for protocol,g in pred.groupby('protocol'):
        assert len(g)==341 and g['index'].is_unique and set(g['index'])==set(range(341))
        for ds in ['all','20261014','20261020']:
            z=g if ds=='all' else g[g.dataset==int(ds)];values=metric(z)
            row=summary[(summary.protocol==protocol)&(summary.dataset==ds)].iloc[0]
            assert all(np.isclose(row[k],v,rtol=1e-10,atol=1e-12) for k,v in values.items())
            metrics.append(dict(protocol=protocol,dataset=ds,**values))
    result=dict(status='passed',scope='Exploratory 16-candidate development comparison, selected after prior screen inspection; NOT independent test.',all_341_development_exposed=True,direct_fold_overlap_found=False,inner_scores_selected_without_outer_rows=True,all_predictions_reproduced=True,maximum_prediction_difference=maximum,checks=checks,metrics=metrics,model_default_hash=sha(ROBUST/'exploratory_nested_model.joblib'),model_modified_or_refit=False)
    (HERE/'exploratory_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    with threadpool_limits(1):main()
