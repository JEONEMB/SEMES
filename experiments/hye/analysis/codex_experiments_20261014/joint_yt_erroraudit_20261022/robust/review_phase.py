"""Independent source/readback review of phase model; no phase files mutated."""
from pathlib import Path
import sys,json,hashlib,inspect
import numpy as np
import pandas as pd
import joblib
from threadpoolctl import threadpool_limits

H=Path(__file__).resolve().parent
ROOT=H.parent
PREV=ROOT.parent/'joint_yt_generalization_20261022'
OLD=ROOT.parent/'joint_yt_disentangle_20261022'
sys.path[:0]=[str(ROOT),str(PREV)]
from data import load,metric
from phase_model import PhaseNuisance,CONFIGS

def main():
    m,a=load();D=a['D'];y=m[['Y','T']].to_numpy()
    before=D.copy();D.flags.writeable=False
    p=pd.read_csv(ROOT/'phase_predictions.csv')
    summary=pd.read_csv(ROOT/'phase_summary.csv')
    selection=json.loads((ROOT/'phase_selection.json').read_text())
    original=json.loads((OLD/'selection.json').read_text())
    checks=[];prediction_delta=0.;mean_delta=0.;refit_delta=0.
    for f in range(5):
        record=selection[f]
        prior=original[f] if f<4 else original[-1]['full_selection']
        assert record['inner_splits']==prior['inner_splits']
        tr=np.flatnonzero(m.fold.to_numpy()!=f) if f<4 else np.arange(len(m))
        te=np.flatnonzero(m.fold.to_numpy()==f) if f<4 else np.arange(len(m))
        for part in record['inner_splits']:
            assert set(part['train'])<=set(tr)
            assert set(part['validation'])<=set(tr)
            assert not set(m.iloc[part['train']].cv_group)&set(m.iloc[part['validation']].cv_group)
        chosen=min(record['scores'],key=lambda r:(r['failures'],r['maximum'],r['mean']))['candidate']
        assert chosen==record['chosen']
        path=ROOT/(f'phase_fold{f}.joblib' if f<4 else 'phase_model.joblib')
        bundle=joblib.load(path);model=bundle['model']
        assert np.array_equal(bundle['train_indices'],tr)
        assert np.array_equal(model.training_indices,m.iloc[tr]['index'])
        assert np.array_equal(bundle['ids'],a['ids'])
        assert np.array_equal(bundle['points'],a['points'])
        assert bundle['config']==CONFIGS[chosen]
        if f<4:assert not set(m.iloc[tr].cv_group)&set(m.iloc[te].cv_group)
        protocol='nested_development_oof' if f<4 else 'training_reapplication'
        saved=p[p.protocol==protocol]
        if f<4:saved=saved[saved.fold==f]
        saved=saved.set_index('index').loc[m.iloc[te]['index']]
        predicted=model.predict(D[te])
        assert predicted.shape==(len(te),2)
        one=model.predict(D[te[0]])
        assert one.shape==(1,2) and np.allclose(one[0],predicted[0],atol=1e-9)
        changed=model.transform(D[te])
        assert changed.shape==(len(te),630,2)
        mean_delta=max(mean_delta,float(abs(changed.mean(1)-D[te].mean(1)).max()))
        prediction_delta=max(prediction_delta,float(abs(predicted-saved[['predY','predT']].to_numpy()).max()))
        # Refit only the recorded training data. This also exercises read-only D.
        restored=PhaseNuisance(CONFIGS[chosen]).fit(D[tr],y[tr],m.iloc[tr])
        refit_delta=max(refit_delta,float(abs(restored.predict(D[te])-predicted).max()))
        checks.append(dict(fold=f,selected=chosen,enabled=bool(bundle['config']['enabled']),
                           n_train=len(tr),n_pred=len(te),phase=bundle['config']['phase'],alpha=bundle['config']['alpha']))
    assert np.array_equal(D,before)
    assert prediction_delta<1e-8 and refit_delta<1e-8 and mean_delta<1e-10
    for _,row in summary.iterrows():
        q=p[p.protocol==row.protocol]
        if str(row.dataset)!='all':q=q[q.dataset==int(row.dataset)]
        numbers=metric(q[['Y','T']],q[['predY','predT']])
        for key,value in numbers.items():assert np.isclose(row[key],value,atol=1e-10)
    assert list(inspect.signature(PhaseNuisance.predict).parameters)==['self','D']
    out=dict(status='passed',files_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ['phase_model.py','phase_evaluate.py','phase_protocol.json','phase_selection.json','phase_predictions.csv']},
             fold_checks=checks,original_D_unchanged=True,fixed630_ids_and_points_equal=True,
             transformed_shape_preserved=True,mean_translation_max_change=mean_delta,
             pure_predict_signature='predict(self,D); (630,2)->(1,2), (N,630,2)->(N,2)',
             all_inner_splits_exact_prior=True,all_inner_rows_in_outer_training=True,
             no_new_condition_group_overlap=True,selection_rank_recomputed=True,
             saved_model_predictions_max_difference=prediction_delta,train_only_refit_prediction_max_difference=refit_delta,
             summaries_recomputed=True,
             source_review=[
                 'Periodic phase inputs are computed solely from image displacement mean; query date, filename, command and repeat ID never enter predict.',
                 'Nuisance coefficients are estimated from within-training-condition centered phase and shape; validation/test groups do not enter these means.',
                 'The learned correction has exactly zero mean across630 points per coefficient; center-fixed observation D is not modified.',
                 'Group membership and new-date tags are training-only calibration metadata; subsequent RepeatGLS forward response and residual covariance also use training only.',
                 'Fixed candidate outer predictions are computed for diagnostics but are not consulted in inner-score candidate selection.'],
             limitations=[
                 'Pixel-locking is a hypothesis, not an established cause. Periodic shape correlation among repeats can reflect tracking, image noise, actual pose changes or other acquisition effects.',
                 'This is adaptive development on the repeatedly used341 images, not untouched independent test evidence.',
                 'Full-data inner selection chooses enabled=false. The saved phase_model.joblib deploys the phase-disabled baseline; the53/60 result describes the fold-wise selection procedure, not a single final enabled phase correction.',
                 'Within-condition repeat shape variation can contain genuine pose variation; treating it as nuisance is a modeling assumption, not proven separation of sensor and mechanical noise.',
                 'The implementation assumes the agreed630-point input. Empty/nonrepeated training-group handling is not needed by the current fixed splits and is not covered as a general-purpose API.'])
    (H/'phase_review.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    print(json.dumps(out,indent=2))

if __name__=='__main__':
    with threadpool_limits(1):main()
