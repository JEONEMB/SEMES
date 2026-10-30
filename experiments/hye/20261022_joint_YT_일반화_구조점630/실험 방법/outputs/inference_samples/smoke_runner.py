"""End-to-end verification; labels here are evaluation metadata only."""
from pathlib import Path
import hashlib,json,os,sys
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
sys.dont_write_bytecode=True
import numpy as np,pandas as pd
from concurrent.futures import ProcessPoolExecutor,as_completed
H=Path(__file__).resolve().parents[2];sys.path.insert(0,str(H))
OUT=H/'outputs/inference_samples'
from threadpoolctl import threadpool_limits
import predict

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def check(index,tag,path):
    captured=[]
    original=predict.StructuralExtractor
    class CapturingExtractor(original):
        def extract(self,image):
            result=super().extract(image);captured.append(result);return result
    predict.StructuralExtractor=CapturingExtractor
    try:result=predict.predict_image(path)
    finally:predict.StructuralExtractor=original
    z=np.load(H/'measurements/features.npz');cached=z['D'][index]
    measured=captured[0]['D'];bundle,model_digest=predict._load_bundle()
    with threadpool_limits(1):cache_prediction=bundle['model'].predict(cached[None])[0]
    actual=np.array([result['Y'],result['T']]);coord_delta=float(np.max(abs(measured-cached)));prediction_delta=float(np.max(abs(actual-cache_prediction)))
    assert coord_delta<1e-8,(tag,coord_delta)
    assert prediction_delta<1e-8,(tag,prediction_delta)
    assert result['correction_Y']==-result['Y'] and result['correction_T']==-result['T']
    assert result['model_sha256']==model_digest and not result['actuation_executed']
    assert 'pass' not in result and 'ground_truth' not in result
    (OUT/f'{tag}.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    return dict(index=index,case=tag,source_path=str(path),prediction=result,cache_coordinate_max_abs_delta=coord_delta,cache_prediction_max_abs_delta=prediction_delta)

if __name__=='__main__':
    m=pd.read_csv(H/'measurements/manifest.csv').set_index('index')
    previous=H.parent/'joint_yt_improved_20261022'
    hashes={n:digest(previous/n) for n in ['fixedfeature_model.joblib','coordinate_model.joblib']}
    model_hash=digest(H/'model.joblib')
    renamed=OUT/'unlabeled_current.bmp';renamed.write_bytes(Path(m.loc[332,'path']).read_bytes())
    specs=[(210,'old_single',str(m.loc[210,'path'])),(332,'new_mixed',str(m.loc[332,'path'])),(335,'expanded_search',str(m.loc[335,'path'])),(332,'renamed_same_pixels',str(renamed))]
    rows=[]
    with ProcessPoolExecutor(max_workers=4) as pool:
        for f in as_completed([pool.submit(check,*s) for s in specs]):
            row=f.result();rows.append(row);print(row['case'],row['prediction']['Y'],row['prediction']['T'],row['cache_coordinate_max_abs_delta'],flush=True)
    a=next(r for r in rows if r['case']=='new_mixed')['prediction'];b=next(r for r in rows if r['case']=='renamed_same_pixels')['prediction']
    for k in ['Y','T','correction_Y','correction_T','image_sha256','model_sha256','diagnostic_valid_count']:
        assert a[k]==b[k],k
    assert digest(H/'model.joblib')==model_hash
    assert hashes=={n:digest(previous/n) for n in hashes}
    result=dict(status='passed',samples=sorted(rows,key=lambda r:r['case']),current_model_sha256=model_hash,original_model_hashes_before_and_after=hashes,renamed_image_predictions_identical=True,no_robot_actuation=True,limits=['This checks execution consistency, not independent accuracy.','Samples overlap final calibration data; error or tolerance guarantees are not inferred from smoke predictions.'])
    (H/'smoke_test.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    print('All image-to-estimate, cache equivalence, rename invariance, and model preservation checks passed.',flush=True)
