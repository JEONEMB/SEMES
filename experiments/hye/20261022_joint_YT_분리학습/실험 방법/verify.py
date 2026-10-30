"""Independent read-only verification of the saved mixed-pose training results.

Writes verification.json only. Does not fit or modify any model.
"""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,sys
import joblib,numpy as np,pandas as pd
from threadpoolctl import threadpool_limits

H=Path(__file__).resolve().parent
PREV=H.parent/'joint_yt_generalization_20261022'
for folder in [PREV,H]:sys.path.insert(0,str(folder))
from data import load
from learned_model import LearnedSeparation

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def metrics(g):
    e=np.abs(g[['predY','predT']].to_numpy()-g[['Y','T']].to_numpy());valid=np.isfinite(e).all(1);v=e[valid]
    return dict(n=len(g),failures=int((~valid).sum()),passed=int(np.all(v<=.09+1e-12,axis=1).sum()),Y_MAE=float(v[:,0].mean()),Y_max=float(v[:,0].max()),T_MAE=float(v[:,1].mean()),T_max=float(v[:,1].max()))

def check_selection(entry):
    scores=entry['scores']
    for key,priority in [('pass_first',['failures','maximum','mean']),('max_first',['maximum','failures','mean'])]:
        best=min(scores,key=lambda r:tuple(r[k] for k in priority))
        assert entry['selected'][key]==best['candidate']
    return True

def main():
    m,a=load();D=a['D'];y=m[['Y','T']].to_numpy()
    p=pd.read_csv(H/'predictions.csv');summary=pd.read_csv(H/'summary.csv',dtype={'dataset':str})
    selection=json.loads((H/'selection.json').read_text());protocol=json.loads((H/'protocol.json').read_text())
    original=json.loads((H.parent/'joint_yt_improved_20261022/splits.json').read_text())
    prior_audit=json.loads((PREV/'verification.json').read_text())
    assert len(m)==341 and m['index'].is_unique and np.array_equal(m['index'],a['index'])
    assert D.shape==(341,630,2) and np.isfinite(D).all() and np.array_equal(y,a['Y'])
    assert protocol['folds']==m.fold.tolist()
    assert m.groupby('cv_group').fold.nunique().eq(1).all()
    centers=m[(m.dataset==20261020)&(m.scope=='center')]
    assert len(centers)==3 and centers.cv_group.nunique()==1 and set(centers.fold)=={0}
    before={x.name:sha(x) for x in H.glob('*.joblib')};fold_audit=[];max_prediction_difference=0.
    for fold in range(4):
        entry=selection[fold];assert entry['fold']==fold
        tr=np.flatnonzero(m.fold.to_numpy()!=fold);te=np.flatnonzero(m.fold.to_numpy()==fold)
        assert not set(tr)&set(te)
        assert not set(m.iloc[tr].cv_group)&set(m.iloc[te].cv_group)
        assert set(m.loc[(m.dataset==20261014)&(m.scope=='single')&(m.fold==fold),'file'])==set(original[fold]['test'])
        inner_seen=[]
        for inner in entry['inner_splits']:
            it,iv=set(inner['train']),set(inner['validation'])
            assert not it&iv and (it|iv)==set(tr) and not (it|iv)&set(te)
            assert not set(m.iloc[list(it)].cv_group)&set(m.iloc[list(iv)].cv_group)
            inner_seen.extend(iv)
        assert sorted(inner_seen)==sorted(tr)
        check_selection(entry)
        model_records=[]
        for key in ['pass_first','max_first']:
            bundle=joblib.load(H/f'{key}_fold{fold}.joblib');model=bundle['model']
            assert np.array_equal(bundle['train_indices'],tr)
            assert np.array_equal(model.training_indices,m.iloc[tr]['index'])
            assert np.array_equal(bundle['ids'],a['ids']) and np.array_equal(bundle['points'],a['points'])
            selected=entry['scores'][entry['selected'][key]]['config']
            assert model.config==bundle['config']==selected
            saved=p[(p.method==key)&(p.protocol=='nested_group_oof')&(p.fold==fold)].set_index('index').loc[te]
            current=model.predict(D[te]);delta=float(np.max(abs(current-saved[['predY','predT']].to_numpy())))
            max_prediction_difference=max(max_prediction_difference,delta)
            assert np.allclose(current,saved[['predY','predT']],rtol=1e-10,atol=1e-10)
            if model.config['method']=='conditional':
                for ii,jj in model.model.crossfit_splits:
                    assert not set(ii)&set(jj)
                    assert not set(m.iloc[tr[ii]].cv_group)&set(m.iloc[tr[jj]].cv_group)
            model_records.append(dict(method=key,config=selected,train_indices_match=True,prediction_max_difference=delta))
        fold_audit.append(dict(fold=fold,train=len(tr),test=len(te),test_new=int((m.iloc[te].dataset==20261020).sum()),original_280_test_files_unchanged=True,outer_and_inner_groups_disjoint=True,inner_validation_partitions_training=True,models=model_records))
    full=selection[-1]['full_selection'];check_selection(full)
    for inner in full['inner_splits']:
        it,iv=set(inner['train']),set(inner['validation']);assert not it&iv and it|iv==set(range(341))
        assert not set(m.iloc[list(it)].cv_group)&set(m.iloc[list(iv)].cv_group)
    for key in ['pass_first','max_first']:
        bundle=joblib.load(H/f'{key}_model.joblib');model=bundle['model']
        assert np.array_equal(bundle['train_indices'],np.arange(341)) and np.array_equal(model.training_indices,m['index'])
        assert bundle['config']==full['scores'][full['selected'][key]]['config']
        saved=p[(p.method==key)&(p.protocol=='resub341')].set_index('index').loc[np.arange(341)]
        current=model.predict(D);delta=float(np.max(abs(current-saved[['predY','predT']].to_numpy())))
        max_prediction_difference=max(max_prediction_difference,delta)
        assert np.allclose(current,saved[['predY','predT']],rtol=1e-10,atol=1e-10)
    recomputed=[]
    for (key,validation),g in p.groupby(['method','protocol']):
        assert len(g)==341 and g['index'].is_unique and set(g['index'])==set(range(341))
        assert np.isfinite(g[['predY','predT']]).all().all()
        assert np.allclose(g.eY,abs(g.predY-g.Y)) and np.allclose(g.eT,abs(g.predT-g['T']))
        assert np.allclose(g.correctionY,-g.predY) and np.allclose(g.correctionT,-g.predT)
        assert np.array_equal(g.passed,(g.eY<=.09+1e-12)&(g.eT<=.09+1e-12))
        if validation=='nested_group_oof':assert np.array_equal(g.fold,g['index'].map(m.set_index('index').fold))
        for dataset in ['all','20261014','20261020']:
            sub=g if dataset=='all' else g[g.dataset==int(dataset)];computed=metrics(sub)
            row=summary[(summary.method==key)&(summary.protocol==validation)&(summary.dataset==dataset)]
            assert len(row)==1
            for name,value in computed.items():assert np.isclose(row.iloc[0][name],value,rtol=1e-10,atol=1e-12)
            recomputed.append(dict(method=key,protocol=validation,dataset=dataset,**computed))
    primary=p[(p.method=='pass_first')&(p.protocol=='nested_group_oof')&(p.dataset==20261020)]
    assert int(primary.passed.sum())==50
    resub=p[(p.method=='pass_first')&(p.protocol=='resub341')&(p.dataset==20261020)]
    assert int(resub.passed.sum())==60 and np.isclose(resub.eY.max(),.07955046347215715)
    original_audit=json.loads((H.parent/'evaluate_archives_20261022/audit.json').read_text());preserved=[]
    for name,digest in original_audit['model_hashes'].items():
        current=sha(H.parent/'joint_yt_improved_20261022'/name);assert current==digest
        preserved.append(dict(file=str(H.parent/'joint_yt_improved_20261022'/name),sha256=current,unchanged=True))
    for name,digest in prior_audit['newly_created_models_unchanged_by_verifier'].items():
        current=sha(PREV/name);assert current==digest
        preserved.append(dict(file=str(PREV/name),sha256=current,unchanged=True))
    assert before=={x.name:sha(x) for x in H.glob('*.joblib')}
    result=dict(status='passed',checked_at_utc=datetime.now(timezone.utc).isoformat(),verifier='independent conditional_decoder agent',models_refit=False,input_images=341,fixed630=True,new_center3_grouped_and_excluded_from_fold0_training=True,all_saved_predictions_reproduced=True,prediction_max_difference=max_prediction_difference,all_summary_arithmetic_recomputed=True,folds=fold_audit,metrics=recomputed,preserved_models=preserved,current_model_hashes=before,source_hashes={name:sha(H/name) for name in ['evaluate.py','learned_model.py','paired_model.py','neural_model.py']},code_review=dict(test_filename_or_label_in_predict=False,training_metadata='Command labels, date/condition group membership used only during training and split/metric calculation.',paired_graphs='Constructed solely from training rows.',repeat_covariance='Repeat noise and same-axis slopes estimated solely from training rows.',neural='PCA, standardization, base model and network parameters trained inside supplied training split. Selected neural candidate uses in-sample base outputs during training; this is a distribution-mismatch risk, not outer-test leakage.',conditional='Training residual head uses internal out-of-fold nuisance predictions.',assessment='No material data leakage or arithmetic defect found.'),limitations=['Primary nested result is50/60, not60/60. The60/60 result is full-data resubstitution.','Primary pass rate decreases versus prior52/60; worstY error improves0.3282 to0.2574.','Secondary worst-error objective produces44/60 and is reported separately.','The model family was developed on this corpus; nested grouped OOF is not an untouched future-date test.','The training response matrix being identity is enforced by GLS algebra, not evidence of perfect unseen-image disentanglement.','20261020 training conditions are included in calibration.','Filename commands are evaluation targets, not independently measured physical pose.','No robot correction was executed.'])
    (H/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(status='passed',primary_new60=metrics(primary),resub_new60=metrics(resub),prediction_max_difference=max_prediction_difference,preserved_models=len(preserved)),indent=2))

if __name__=='__main__':
    with threadpool_limits(1):main()
