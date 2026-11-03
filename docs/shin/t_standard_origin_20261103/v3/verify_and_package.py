from pathlib import Path
import os,sys,json,hashlib,shutil,subprocess
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import numpy as np
import pandas as pd
import cv2
from scene_estimator import data,infer,restore,predict

P=Path(__file__).resolve().parent;ROOT=P.parents[2]
AUDIT=ROOT/'analysis/frame_link_audit_20261103'
REPO=ROOT/'qrwithbar-lanczos-src'
RAW=Path(r'C:\Users\SSAFY\Documents\GitHub\S15P31A507\data\SSAFY Test Image Set')

def main():
    manifest,raw,natural,valid,ids,xy=data();refs=manifest.cohort.to_numpy()=='reference'
    checks={'source_images':len(manifest),'reference_images':int(refs.sum()),'evaluation_images':int((~refs).sum())}
    assert checks=={'source_images':575,'reference_images':4,'evaluation_images':571}
    for rel,h in json.loads((P/'input_hashes.json').read_text()).items():assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==h
    checks['upstream_input_hashes_unchanged']=True
    # Origin and nested-origin training masks: all non-reference evaluation poses are absent.
    for info in (P/'crossfit').glob('outer*/origin_fit.json'):
        j=json.loads(info.read_text());train=np.asarray(j['train_row_ids'],int)
        parts=info.parent.name.replace('outer','').split('_inner');excluded=[int(v) for v in parts]
        for fold in excluded:assert not np.any((manifest.iloc[train].fold.to_numpy()==fold)&~refs[train]),info
        assert set(np.flatnonzero(refs))<=set(train)
    checks['origin_train_test_isolation_25_splits']=True
    for variant in ['original','corrected','fixed_v3','crossfit','prior_half','prior_double','bound_double']:
        for directory in ['results','scene_results']:
            q=pd.read_csv(P/variant/directory/'predictions.csv')
            assert len(q)==571 and q.path.nunique()==571 and np.isfinite(q.T_pred).all() and (q.status==0).all()
            d=q.merge(manifest[['path','fold']],on='path',validate='one_to_one',suffixes=('','_expected'))
            assert (d.fold==d.fold_expected).all()
            assert np.max(np.abs(q.error_ticks-(q.T_pred-q.T_target)*100))<1e-10
    checks['prediction_counts_folds_and_error_arithmetic_14_runs']=True
    # Exact repeats, including the corrected .32 typo, remain in a single evaluation fold.
    assert manifest.groupby(['Y','T']).fold.nunique().max()==1
    checks['audited_pose_grouping']=True
    unavailable=pd.read_csv(P/'crossfit/results/predictions.csv');unavailable=unavailable[unavailable.target_available==0]
    checks['unidentifiable_crossfit_targets']=unavailable.path.tolist();assert len(unavailable)==3
    # Freeze a deployable v3 experiment without replacing the previous recommended model.
    out=P/'deployment';(out/'runtime').mkdir(parents=True,exist_ok=True)
    source=P/'fixed_v3/results/models/deployment.json';j=json.loads(source.read_text())
    j.update({'coordinate_frame':'standard_origin_v3_assumed','physical_accuracy_certified':False,'experiment_status':'development_candidate_no_1tick_guarantee'})
    (out/'qr_model.json').write_text(json.dumps(j,indent=2),encoding='utf-8')
    shutil.copytree(source.parent/'deployment_members',out/'deployment_members',dirs_exist_ok=True)
    shutil.copy2(P/'fixed_v3/scene_results/deployment.json',out/'scene_model.json')
    shutil.copy2(P/'scene_estimator.py',out/'scene_estimator.py')
    shutil.copytree(REPO/'modules/qrwithbar_lanczos/assets',out/'qr_assets',dirs_exist_ok=True)
    reference=cv2.imdecode(np.fromfile(str(RAW/'center.bmp'),np.uint8),cv2.IMREAD_GRAYSCALE)
    cv2.imwrite(str(out/'reference_scene.png'),reference)
    for name in ['v3_estimate.exe','v3_feature_extract.dll','opencv_world3415.dll']:shutil.copy2(P/'build/Release'/name,out/'runtime'/name)
    for name in ['natural_lanczos.dll','natural_quality.dll']:shutil.copy2(ROOT/'analysis/full_scene_origin_20261102/build/Release'/name,out/'runtime'/name)
    smoke=[]
    chosen=['Theta/T+0.50.bmp','Y/Y+0.10.bmp','Y+1.00 T+2.00/Y+1.00 T+2.00 (3).bmp']
    native_cache=pd.read_csv(P/'fixed_v3/results/deployment_predictions.csv').set_index('path')
    scene_cache=pd.read_csv(P/'fixed_v3/scene_results/deployment_predictions.csv').set_index('path')
    for path in chosen:
        i=int(manifest.index[manifest.path==path][0]);original_hash=hashlib.sha256((RAW/path).read_bytes()).hexdigest()
        result,x,n,v=infer(RAW/path,out/'scene_model.json')
        dx=float(np.max(np.abs(x-raw[i])));dn=float(np.max(np.abs(n-natural[i])));assert dx<1e-9 and dn<1e-8
        assert np.array_equal(v,valid[i]);assert abs(result['t_deg']-scene_cache.loc[path,'T_pred'])<1e-9
        native=json.loads(subprocess.check_output([str(out/'runtime/v3_estimate.exe'),str(RAW/path),str(out/'qr_assets'),str(out/'qr_model.json')],text=True))
        assert abs(native['t_deg']-native_cache.loc[path,'T_pred'])<1e-9
        assert original_hash==hashlib.sha256((RAW/path).read_bytes()).hexdigest()
        smoke.append({'path':path,'native':native,'scene':result,'QR104_max_difference':dx,'natural_max_difference_px':dn,'image_hash_unchanged':True})
        print('RAW IMAGE SMOKE PASS',path,flush=True)
    checks['raw_image_smoke']=smoke
    # Verify predictions from saved extended models after independent JSON reload.
    obj=json.loads((out/'scene_model.json').read_text());models=[restore(m) for m in obj['models']]
    p=sum(w*predict(m,raw,natural,valid) for w,m in zip(obj['weights'],models))
    expected=scene_cache.loc[manifest.path,'T_pred'].to_numpy();assert np.max(np.abs(p-expected))<1e-10
    checks['scene_all575_serialization_max_difference_deg']=float(np.max(np.abs(p-expected)))
    checks['native_all575_roundtrip']='checked during each training run; threshold 1e-10 deg'
    (P/'verification.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2),encoding='utf-8')
    print('VERIFIED',json.dumps({k:v for k,v in checks.items() if k!='raw_image_smoke'},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
