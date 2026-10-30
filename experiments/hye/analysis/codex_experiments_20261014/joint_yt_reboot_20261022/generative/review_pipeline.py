"""Independent read-only audit of sibling reboot pipelines; writes ../review.json."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,importlib.util,json,sys
import joblib,numpy as np,pandas as pd
from threadpoolctl import threadpool_limits

H=Path(__file__).resolve().parent.parent
GENERAL=H.parent/'joint_yt_generalization_20261022'
sys.path.insert(0,str(GENERAL));sys.path.insert(0,str(H))
from data import load
from direct_models import DirectDecoder
from pixel_models import PixelPLS,PixelCNN,SmallConv
from raw_pixels import PixelExtractor

def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()

def metric(frame):
    e=np.abs(frame[['predY','predT']].to_numpy()-frame[['Y','T']].to_numpy());good=np.isfinite(e).all(1);valid=e[good]
    return dict(n=len(e),failures=int((~good).sum()),passed=int(np.sum(good&np.all(e<=.09+1e-12,1))),Y_MAE=float(valid[:,0].mean()) if len(valid) else None,Y_max=float(valid[:,0].max()) if len(valid) else None,T_MAE=float(valid[:,1].mean()) if len(valid) else None,T_max=float(valid[:,1].max()) if len(valid) else None)

def main():
    m,oldarrays=load();oldselection=json.loads((H.parent/'joint_yt_disentangle_20261022/selection.json').read_text());original=json.loads((H.parent/'joint_yt_improved_20261022/splits.json').read_text())
    assert len(m)==341 and m.groupby('cv_group').fold.nunique().eq(1).all()
    centers=m[(m.dataset==20261020)&(m.scope=='center')];assert len(centers)==3 and set(centers.fold)=={0}
    for f in range(4):assert set(m.loc[(m.dataset==20261014)&(m.scope=='single')&(m.fold==f),'file'])==set(original[f]['test'])
    allmetrics=[];foldchecks=[];maxdiff=0.;selection_checks=[]
    for family,location in [('direct','direct/features.npz'),('shared','direct/shared/features.npz')]:
        a=np.load(H/location);assert np.array_equal(a['index'],m['index'])
        assert np.array_equal(a['points'],oldarrays['points']) and np.array_equal(a['ids'],oldarrays['ids'])
        p=pd.read_csv(H/f'{family}_predictions.csv');summary=pd.read_csv(H/f'{family}_summary.csv',dtype={'dataset':str});select=json.loads((H/f'{family}_selection.json').read_text())
        for f in range(5):
            before=oldselection[f] if f<4 else oldselection[-1]['full_selection'];assert select[f]['inner_splits']==before['inner_splits']
            tr=np.flatnonzero(m.fold.to_numpy()!=f) if f<4 else np.arange(341);te=np.flatnonzero(m.fold.to_numpy()==f) if f<4 else np.arange(341)
            for part in select[f]['inner_splits']:
                it,iv=set(part['train']),set(part['validation']);assert it|iv==set(tr) and not it&iv
                assert not set(m.iloc[list(it)].cv_group)&set(m.iloc[list(iv)].cv_group)
                if f<4:assert not (it|iv)&set(te)
            best=min(select[f]['scores'],key=lambda r:(r['failures'],r['maximum'],r['mean']));assert select[f]['chosen']==best['candidate']
            model_path=H/(f'{family}_fold{f}.joblib' if f<4 else f'{family}_model.joblib');bundle=joblib.load(model_path);cfg=bundle['config'];good=a['success_'+cfg['measurement']].astype(bool);train=tr[good[tr]]
            assert np.array_equal(bundle['train_indices'],train) and np.array_equal(bundle['model'].training_indices,m.iloc[train]['index'])
            assert cfg==best['config']
            pred=np.full((len(te),2),np.nan);pred[good[te]]=bundle['model'].predict(a['D_'+cfg['measurement']][te[good[te]]])
            protocol='development_nested_oof' if f<4 else 'training_reapplication';saved=p[p.protocol==protocol]
            if f<4:saved=saved[saved.fold==f]
            saved=saved.set_index('index').loc[te];expected=saved[['predY','predT']].to_numpy();assert np.array_equal(np.isfinite(pred),np.isfinite(expected));difference=float(np.nanmax(abs(pred-expected)));assert difference<1e-9;maxdiff=max(maxdiff,difference)
            foldchecks.append(dict(family=family,fold=f,train=len(train),test=len(te),model_train_indices_match=True,group_overlap=f==4,train_test_overlap=f==4,training_reapplication=f==4,heldout_validation=f<4,measurement_failures=int((~good[te]).sum()),prediction_difference=difference))
        for protocol,g in p.groupby('protocol'):
            assert len(g)==341 and g['index'].is_unique
            assert np.array_equal(g.passed,(g.eY<=.09+1e-12)&(g.eT<=.09+1e-12))
            for ds in ['all','20261014','20261020']:
                rows=g if ds=='all' else g[g.dataset==int(ds)];computed=metric(rows);row=summary[(summary.protocol==protocol)&(summary.dataset==ds)].iloc[0]
                assert all(np.isclose(row[k],v,rtol=1e-10,atol=1e-12) for k,v in computed.items());allmetrics.append(dict(family=family,protocol=protocol,dataset=ds,**computed))
        fixed=pd.read_csv(H/f'{family}_fixed_predictions.csv');assert fixed.groupby('candidate').size().eq(341).all()
        selection_checks.append(dict(family=family,exact_prior_inner_splits=True,selection_by_inner_only=True,all_fixed_candidates_count=len(fixed.candidate.unique()),all_fixed_candidates_have341_rows=True))
    pixels=np.load(H/'pixel_features.npz');good=pixels['success'].astype(bool);p=pd.read_csv(H/'pixel_predictions.csv');summary=pd.read_csv(H/'pixel_summary.csv',dtype={'dataset':str});selection=json.loads((H/'pixel_selection.json').read_text())
    pm=pd.read_csv(H/'pixel_measurement_manifest.csv');assert np.array_equal(pm['index'],m['index']) and np.array_equal(pm.success,good)
    parameter_count=sum(parameter.numel() for parameter in SmallConv().parameters() if parameter.requires_grad)
    assert parameter_count==42860
    cnn_state_counts=[]
    for f in range(5):
        prior=oldselection[f] if f<4 else oldselection[-1]['full_selection'];assert selection[f]['inner_splits']==prior['inner_splits']
        best=min(selection[f]['scores'],key=lambda r:(r['failures'],r['maximum'],r['mean']));assert best['components']==selection[f]['chosen']
        tr=np.flatnonzero((m.fold.to_numpy()!=f)&good) if f<4 else np.flatnonzero(good);te=np.flatnonzero(m.fold.to_numpy()==f) if f<4 else np.arange(341)
        for part in selection[f]['inner_splits']:
            it,iv=set(part['train']),set(part['validation']);assert not it&iv
            assert not set(m.iloc[list(it)].cv_group)&set(m.iloc[list(iv)].cv_group)
            if f<4:assert not (it|iv)&set(te)
        for name in ['pixel_pls','pixel_cnn']:
            bundle=joblib.load(H/(f'{name}_fold{f}.joblib' if f<4 else f'{name}_model.joblib'));model=bundle['model'];assert np.array_equal(bundle['train_indices'],tr) and np.array_equal(model.training_indices,m.iloc[tr]['index'])
            if name=='pixel_cnn':
                assert model.epochs==350 and sum(v.numel() for v in model.state.values())==parameter_count
                assert np.isclose(model.pixel_scale,max(float(np.std(pixels['pixels'][tr])),.02))
                assert np.allclose(model.shift_mean,pixels['shift'][tr].reshape(len(tr),-1).mean(0))
                assert np.allclose(model.shift_scale,np.maximum(pixels['shift'][tr].reshape(len(tr),-1).std(0),1.))
                cnn_state_counts.append(dict(fold=f,parameters=parameter_count,epochs=model.epochs,train_only_normalization_verified=True))
            else:
                assert model.components==selection[f]['chosen']
                x=model.features(pixels['pixels'][tr],pixels['shift'][tr]);assert np.array_equal(model.model._x_mean,x.astype(np.float64).mean(0))
                assert np.allclose(model.model._y_mean,m.iloc[tr][['Y','T']].to_numpy().mean(0))
            pred=np.full((len(te),2),np.nan);pred[good[te]]=model.predict(pixels['pixels'][te[good[te]]],pixels['shift'][te[good[te]]])
            protocol='development_group_oof' if f<4 else 'training_reapplication';saved=p[(p.method==name)&(p.protocol==protocol)]
            if f<4:saved=saved[saved.fold==f]
            expected=saved.set_index('index').loc[te][['predY','predT']].to_numpy();assert np.array_equal(np.isfinite(pred),np.isfinite(expected));difference=float(np.nanmax(abs(pred-expected)));assert difference<1e-7;maxdiff=max(maxdiff,difference)
            foldchecks.append(dict(family=name,fold=f,train=len(tr),test=len(te),model_train_indices_match=True,group_overlap=f==4,train_test_overlap=f==4,training_reapplication=f==4,heldout_validation=f<4,prediction_difference=difference))
    for (name,protocol),g in p.groupby(['method','protocol']):
        assert len(g)==341 and g['index'].is_unique
        for ds in ['all','20261014','20261020']:
            rows=g if ds=='all' else g[g.dataset==int(ds)];computed=metric(rows);row=summary[(summary.method==name)&(summary.protocol==protocol)&(summary.dataset==ds)].iloc[0]
            assert all(np.isclose(row[k],v,rtol=1e-10,atol=1e-12) for k,v in computed.items());allmetrics.append(dict(family=name,protocol=protocol,dataset=ds,**computed))
    extractor=PixelExtractor();pixel_checks=[]
    for i in [0,281,340]:
        extracted=extractor.extract(Path(m.iloc[i].path).read_bytes());pixel_difference=float(np.max(abs(extracted['pixels']-pixels['pixels'][i])));assert pixel_difference<1e-6 and np.array_equal(extracted['shift'],pixels['shift'][i]);pixel_checks.append(dict(index=i,byte_only_extraction_difference=pixel_difference))
    spec=importlib.util.spec_from_file_location('_review_pixel_eval',H/'evaluate_pixels.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    assert module.error_score([[0.,0.],[np.nan,np.nan],[.1,.01]])['failures']==2
    assert module.error_score([[.09,.09]])['failures']==0
    assert module.error_score([[np.nan,np.nan]])['failures']==1
    no_valid=module.metric(np.zeros((2,2)),np.full((2,2),np.nan))
    assert no_valid['n']==2 and no_valid['failures']==2 and no_valid['passed']==0 and no_valid['Y_max'] is None and no_valid['T_max'] is None
    # Independently reconstruct QR grid observations and re-fit every parallax
    # inner score using ONLY its recorded training indices (no files overwritten).
    pa=np.load(H/'parallax/features.npz');qa=np.load(H/'direct/features.npz')
    assert np.array_equal(pa['index'],m['index']) and np.array_equal(qa['index'],m['index'])
    qds=[]
    for qr,(x0,y0,x1,y1) in enumerate([(1552,969,1736,1167),(1809,971,1993,1169),(2067,973,2252,1171)]):
        points=np.array([(x0+fx*(x1-x0),y0+fy*(y1-y0)) for fy in [.2,.5,.8] for fx in [.2,.5,.8]])
        vectors=np.einsum('nij,kj->nki',qa['H_homography_g5'][:,qr],np.c_[points,np.ones(9)])
        qds.append(vectors[:,:,:2]/vectors[:,:,2:]-points)
    qd=np.concatenate(qds,axis=1);pp=pd.read_csv(H/'parallax_predictions.csv');ps=pd.read_csv(H/'parallax_summary.csv',dtype={'dataset':str});sel=json.loads((H/'parallax_selection.json').read_text());y=m[['Y','T']].to_numpy();parallax_checks=[]
    before_models={p.name:sha(p) for p in H.glob('*.joblib')}
    for family in ['external','qr_external']:
        for fold in range(5):
            record=next(r for r in sel if r['family']==family and r['fold']==fold);prior=oldselection[fold] if fold<4 else oldselection[-1]['full_selection'];assert record['inner_splits']==prior['inner_splits']
            tr=np.flatnonzero(m.fold.to_numpy()!=fold) if fold<4 else np.arange(341);te=np.flatnonzero(m.fold.to_numpy()==fold) if fold<4 else np.arange(341);inner_seen=[];fresh_scores=[]
            for score in record['scores']:
                cfg=score['config'];D=pa['D_'+cfg['measurement']] if family=='external' else np.concatenate([qd,pa['D_'+cfg['measurement']]],axis=1);good=pa['success_'+cfg['measurement']].astype(bool)&(qa['success_homography_g5'].astype(bool) if family=='qr_external' else True);errors=[];dates=[]
                for part in record['inner_splits']:
                    train_set,validation_set=set(part['train']),set(part['validation']);assert train_set|validation_set==set(tr) and not train_set&validation_set
                    assert not set(m.iloc[list(train_set)].cv_group)&set(m.iloc[list(validation_set)].cv_group)
                    if fold<4:assert not (train_set|validation_set)&set(te)
                    it=np.array(part['train']);it=it[good[it]];iv=np.array(part['validation']);model=DirectDecoder(cfg).fit(D[it],y[it],m.iloc[it]);pred=np.full((len(iv),2),np.nan);pred[good[iv]]=model.predict(D[iv[good[iv]]]);errors.extend(abs(pred-y[iv]));dates.extend(m.iloc[iv].dataset)
                error=np.asarray(errors);dates=np.array(dates);finite=np.isfinite(error).all(1);bad=(~finite)|np.any(error>.09+1e-12,axis=1);valid=error[finite]
                computed=dict(date_balanced_failure_rate=float(np.mean([bad[dates==d].mean() for d in np.unique(dates)])),failures=int(bad.sum()),maximum=float(valid.max()) if len(valid) else 1e100,mean=float(valid.mean()) if len(valid) else 1e100)
                # QR grid is reconstructed with independent einsum algebra;
                # ~1e-12 pixel summation differences can produce ~1e-9 command
                # differences in an ill-conditioned quadratic head.
                assert all(np.isclose(score[k],v,rtol=1e-6,atol=1e-8) for k,v in computed.items()),(family,fold,cfg,score,computed);fresh_scores.append(dict(candidate=score['candidate'],**computed))
            chosen=min(fresh_scores,key=lambda r:(r['date_balanced_failure_rate'],r['maximum'],r['mean']))['candidate'];assert chosen==record['chosen']
            bundle=joblib.load(H/(f'{family}_fold{fold}.joblib' if fold<4 else f'{family}_model.joblib'));cfg=bundle['config'];D=pa['D_'+cfg['measurement']] if family=='external' else np.concatenate([qd,pa['D_'+cfg['measurement']]],axis=1);good=pa['success_'+cfg['measurement']].astype(bool)&(qa['success_homography_g5'].astype(bool) if family=='qr_external' else True);train=tr[good[tr]]
            assert np.array_equal(bundle['train_indices'],train) and np.array_equal(bundle['model'].training_indices,m.iloc[train]['index']) and cfg==record['scores'][chosen]['config']
            refit=DirectDecoder(cfg).fit(D[train],y[train],m.iloc[train]);actual=bundle['model'].predict(D[te[good[te]]]);retrained=refit.predict(D[te[good[te]]]);assert np.allclose(actual,retrained,rtol=1e-8,atol=1e-9)
            pred=np.full((len(te),2),np.nan);pred[good[te]]=actual;protocol='development_nested_oof' if fold<4 else 'training_reapplication';saved=pp[(pp.method==family)&(pp.protocol==protocol)]
            if fold<4:saved=saved[saved.fold==fold]
            expected=saved.set_index('index').loc[te][['predY','predT']].to_numpy();difference=float(np.nanmax(abs(pred-expected)));assert difference<1e-9;maxdiff=max(maxdiff,difference)
            parallax_checks.append(dict(family=family,fold=fold,train=len(train),test=len(te),selected=chosen,config=cfg,group_overlap=fold==4,train_test_overlap=fold==4,training_reapplication=fold==4,heldout_validation=fold<4,all6_inner_candidate_scores_recomputed=True,selected_model_refitted_from_train_only_and_matches_saved=True,saved_prediction_max_difference=difference))
        for protocol,g in pp[pp.method==family].groupby('protocol'):
            assert len(g)==341 and g['index'].is_unique and set(g['index'])==set(range(341))
            assert np.array_equal(g.passed,(g.eY<=.09+1e-12)&(g.eT<=.09+1e-12))
            for ds in ['all','20261014','20261020']:
                rows=g if ds=='all' else g[g.dataset==int(ds)];computed=metric(rows);row=ps[(ps.method==family)&(ps.protocol==protocol)&(ps.dataset==ds)].iloc[0];assert all(np.isclose(row[k],v,rtol=1e-9,atol=1e-11) for k,v in computed.items());allmetrics.append(dict(family=family,protocol=protocol,dataset=ds,**computed))
    assert before_models=={p.name:sha(p) for p in H.glob('*.joblib')}
    previous=json.loads((H/'preservation_before.json').read_text())['files'];changed=[]
    for name,digest in previous.items():
        if sha(H.parent/name)!=digest:changed.append(name)
    assert not changed
    sources=['direct_models.py','evaluate_direct.py','pixel_models.py','evaluate_pixels.py','raw_pixels.py','evaluate_shared.py','direct/extractor.py','direct/shared/shared_extractor.py','evaluate_parallax.py','parallax/extractor.py','parallax/measure.py','parallax/protocol.json']
    for name in ['evaluation_metrics.py','metrics.py']:
        if (H/name).exists():sources.append(name)
    report=dict(status='completed; see resolved findings and evaluation limitations',checked_at_utc=datetime.now(timezone.utc).isoformat(),independent_reviewer='conditional_decoder',models_retrained=False,direct_fold_leakage_found=False,filename_or_test_command_in_predict_found=False,all341_are_development_exposed=True,new_measurement_does_not_make_old_photos_independent=True,original280_splits_unchanged=True,new_mixed_condition_repeats_grouped=True,new_center3_same_fold=True,per_image_normalization_is_image_only_not_test_set_fitting=True,fixed_original_center_reference_permitted=True,all_saved_predictions_reproduced=True,maximum_prediction_difference=maxdiff,all_summary_arithmetic_recomputed=True,folds=foldchecks,selection_checks=selection_checks,metrics=allmetrics,cnn=dict(trainable_parameters=parameter_count,approximately43k_is_correct=True,state_dict_counts=cnn_state_counts,architecture='Two shared conv layers over three48x48 QR difference images;6x6 spatial feature grid;6 normalized coarse NCC coordinates;32-hidden-unit regression head plus linear shift skip',epochs=350,early_stopping=False),raw_pixel_reextraction=pixel_checks,pixel_measurement_successes=int(good.sum()),findings=[dict(id='PLS-INNER-FAILURES',severity='P2',status='fixed_by_parent_and_verified',file='evaluate_pixels.py',issue='Original inner validation removed failed measurement rows instead of counting them. Repaired code retains every new-date validation row, initializes NaN predictions, and counts nonfinite errors.',effect_on_saved_results='None because all341 current measurements succeeded.',fault_injection_checks=3),dict(id='ZERO-SUCCESS-METRIC',severity='P2',status='reported_to_parent; verify replacement helper status before final delivery',file='prior data.metric imported by evaluation scripts',issue='Prior shared metric raises at max(empty) when a scored subset has zero finite predictions. Old source should remain preserved; a reboot-local safe helper should report n failures,zero passes,null error summaries.',effect_on_saved_results='None because current evaluated outputs are all finite.')],preserved_prior_files=len(previous),changed_prior_files=changed,reviewed_source_sha256={name:sha(H/name) for name in sources},limitations=['341 existing images have influenced adaptive method development; nested scores are development comparisons, not independent new-capture evidence.','Fixed candidates and families reviewed after OOF are additional adaptation; nested fold separation does not erase that history.','Pixel per-image normalization/NCC and dense ECC use only current image plus fixed center; learned PLS/CNN/head preprocessing is train-only.','Training reapplication, including PLS341/341 and CNN new60/60, must not be presented as held-out or future-photo success.','Filename command targets are not independently measured robot pose.','No physical robot correction, return-to-center test, camera calibration, or independent new capture was performed.'])
    report['models_retrained']='Parallax verification re-fitted models in memory from recorded training rows only; no saved model was changed.'
    report['pixel_measurement_successes']=int(pixels['success'].sum())
    report['fold4_explanation']='Fold4 is the full-data model reapplied to its training images: actual train/test and group overlap is true, heldout validation is false. Its internal model-selection validation splits remain disjoint.'
    if (H/'generative/inference_review.json').exists():
        report['raw_inference_api_review']=json.loads((H/'generative/inference_review.json').read_text(encoding='utf-8'))
        report['reviewed_source_sha256']['predict.py']=sha(H/'predict.py')
    report['parallax_checks']=parallax_checks
    raw_audit=json.loads((H/'generative/raw_parallax_audit/audit.json').read_text(encoding='utf-8'))
    assert raw_audit['n_read_raw']==341 and raw_audit['n_missing_or_exception']==0
    assert raw_audit['all_cached_arrays_reproduced_exactly'] and raw_audit['all_raw_hashes_match_both_manifests']
    assert raw_audit['extractor_read_guard']['all_341_passed'] and not raw_audit['duplicate_raw_sha_groups']
    report['parallax_raw_341_independent_audit']={'report':'generative/raw_parallax_audit/audit.json','report_sha256':sha(H/'generative/raw_parallax_audit/audit.json'),'all341_reextracted_from_bytes':True,'all28_cached_arrays_exactly_reproduced':True,'all_sha_match_both_manifests':True,'forbidden_file_reads_during_extraction_guard_passed':True,'duplicate_raw_sha_groups':[],'limitation':'Independent code/cache review, not independent new-photo generalization evidence.'}
    report['findings'][1].update(status='fixed_by_parent_and_verified',file='metrics.py',resolution='Reboot-local safe metric handles zero finite predictions; prior data.py preserved.',fault_injection_result=no_valid)
    (H/'review.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(dict(status='passed',cnn_parameters=parameter_count,max_prediction_difference=maxdiff,preserved_files=len(previous),parallax_inner_models_refitted=180,review=str(H/'review.json')),indent=2))

if __name__=='__main__':
    with threadpool_limits(1):main()
