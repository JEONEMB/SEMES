"""Measure every actual image; metadata/targets never enter the bytes extractor."""
from pathlib import Path
import argparse,json,time,sys,hashlib
import cv2,numpy as np,pandas as pd
from threadpoolctl import threadpool_limits
from extractor import DirectExtractor,CANDIDATES,CENTER_PATH,CENTER_SHA256,BOXES,MAX_ITERATIONS,EPSILON,INPUT_MARGIN,TEMPLATE_INSET,PADS,NCC_MIN,ECC_MIN

H=Path(__file__).resolve().parent;ROOT=H.parents[2];GENERAL=ROOT/'codex_experiments_20261014/joint_yt_generalization_20261022'
sys.path.insert(0,str(GENERAL))
from data import load

def jsonable(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    raise TypeError(type(x).__name__)

def run(smoke=False):
    cv2.setNumThreads(1);m,_=load();ex=DirectExtractor();results=[];rows=[];started=time.perf_counter()
    take=[0,281,336] if smoke else list(range(len(m)))
    for i in take:
        meta=m.iloc[i];image_bytes=Path(meta.path).read_bytes();r=ex.extract_bytes(image_bytes);results.append(r)
        record=meta.to_dict();record.update(index=i,measurement_seconds=r['seconds'],ncc_pad=r['ncc_pad'],ncc_min=float(r['ncc_scores'].min()),input_bytes_sha256=r['image_sha256'])
        for name,_,_ in CANDIDATES:
            c=r['candidates'][name];record.update({name+'_success':c['success'],name+'_ecc_min':float(c['ecc'].min()),name+'_rmse_mean':float(c['photometric_rmse'].mean()),name+'_seconds':c['seconds'],name+'_reasons':json.dumps(c['reasons'])})
        rows.append(record)
        if smoke or (i+1)%20==0:print(json.dumps({'processed':len(results),'index':i,'elapsed':time.perf_counter()-started,'pad':r['ncc_pad'],'success':{n:r['candidates'][n]['success'] for n,_,_ in CANDIDATES}},default=jsonable),flush=True)
    if smoke:
        out=[]
        for i,r in zip(take,results):
            s={k:v for k,v in r.items() if k!='candidates'};s['index']=i;s['candidates']={n:{k:v for k,v in c.items() if k!='D'} for n,c in r['candidates'].items()};s['mean_D']={n:c['D'].mean(0) for n,c in r['candidates'].items()};out.append(s)
        # Same image bytes have no path/name parameter in this API.
        repeated=ex.extract_bytes(Path(m.iloc[take[-1]].path).read_bytes())
        repeatdiff={n:float(np.nanmax(abs(repeated['candidates'][n]['D']-results[-1]['candidates'][n]['D']))) for n,_,_ in CANDIDATES}
        (H/'smoke_test.json').write_text(json.dumps({'images':out,'repeat_bytes_max_difference':repeatdiff,'status':'measurement_smoke_not_accuracy_or_independent_validation'},default=jsonable,indent=2),encoding='utf-8');return
    arrays=dict(points=ex.points,ids=ex.ids,regions=ex.regions,index=np.array(take),candidate_names=np.array([x[0] for x in CANDIDATES]),ncc_shifts=np.array([r['ncc_shifts'] for r in results]),ncc_scores=np.array([r['ncc_scores'] for r in results]))
    for name,_,_ in CANDIDATES:
        for field,target in [('D','D'),('H','H'),('ecc','ecc'),('photometric_rmse','rmse'),('brightness_gain','gain'),('brightness_offset','offset'),('qr_success','qr_success'),('success','success')]:arrays[target+'_'+name]=np.array([r['candidates'][name][field] for r in results])
    np.savez_compressed(H/'features.npz',**arrays);pd.DataFrame(rows).to_csv(H/'manifest.csv',index=False,encoding='utf-8-sig')
    audit=dict(n=len(m),center_sha256=CENTER_SHA256,cv2_version=cv2.__version__,actual_image_bytes_read=len(results),old_D_used_as_measurements=False,parameters={'candidates':[list(c) for c in CANDIDATES],'max_iterations':MAX_ITERATIONS,'epsilon':EPSILON,'template_inset':TEMPLATE_INSET,'input_margin':INPUT_MARGIN,'boxes':BOXES,'ncc_pads':PADS,'ncc_min':NCC_MIN,'ecc_min':ECC_MIN},failures={n:int(np.sum(~arrays['success_'+n])) for n,_,_ in CANDIDATES},total_seconds=time.perf_counter()-started,median_image_seconds=float(np.median([r['seconds'] for r in results])),max_image_seconds=float(max(r['seconds'] for r in results)),fallback='NCC search expands130/260/512. No ECC candidate replacement; failed candidate D remains NaN and failure remains in denominator.',limits=['A per-QR affine/projective warp assumes one coherent local planar image transformation.','No measured camera intrinsics, metric QR size or physical T unit are invented.','ECC handles global affine brightness changes; it does not guarantee robustness to arbitrary shadows or occlusion.','The projected630 coordinates are smooth representations of newly fitted dense pixel warps, not direct measurements of630 independent image boundaries.','All341 images are already exposed development data.'])
    (H/'measurement_audit.json').write_text(json.dumps(audit,default=jsonable,indent=2),encoding='utf-8');print(json.dumps(audit,default=jsonable,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');a=p.parse_args()
    with threadpool_limits(1):run(a.smoke)
