from pathlib import Path
import sys,json,time,argparse,hashlib
import cv2,numpy as np,pandas as pd
from threadpoolctl import threadpool_limits
from extractor import ParallaxExtractor,CANDIDATES,BOXES,NAMES,PROTOCOL

H=Path(__file__).resolve().parent;ROOT=H.parents[2]
sys.path.insert(0,str(ROOT/'codex_experiments_20261014/joint_yt_generalization_20261022'))
from data import load

def plain(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    raise TypeError(type(x).__name__)

def main(smoke):
    cv2.setNumThreads(1);m,_=load();ex=ParallaxExtractor();indices=[0,281,336] if smoke else range(len(m));allr=[];rows=[];began=time.perf_counter();protocol_hash=hashlib.sha256((H/'protocol.json').read_bytes()).hexdigest()
    for i in indices:
        meta=m.iloc[i];r=ex.extract_bytes(Path(meta.path).read_bytes());allr.append(r);row=meta.to_dict();row.update(index=i,input_bytes_sha256=r['image_sha256'],measurement_seconds=r['seconds'],ncc_pad=r['ncc_pad'],ncc_min=float(r['ncc_scores'].min()))
        for n,_ in CANDIDATES:
            c=r['candidates'][n];row.update({n+'_success':c['success'],n+'_roi_success':json.dumps(c['roi_success'],default=plain),n+'_ecc_min':float(np.nanmin(c['ecc'])),n+'_rmse_mean':float(np.nanmean(c['photometric_rmse'])),n+'_reasons':json.dumps(c['reasons'])})
        rows.append(row)
        if smoke or (i+1)%20==0:print(json.dumps({'processed':len(allr),'index':i,'elapsed':time.perf_counter()-began,'success':{n:r['candidates'][n]['roi_success'] for n,_ in CANDIDATES}},default=plain),flush=True)
    if smoke:
        (H/'smoke_test.json').write_text(json.dumps([{k:v for k,v in r.items()} for r in allr],default=plain,indent=2),encoding='utf-8');return
    arrays=dict(index=np.arange(len(m)),points=ex.points,regions=ex.regions,ids=ex.ids,boxes=np.array(BOXES),roi_names=np.array(NAMES),centroids=ex.centroids,candidate_names=np.array([n for n,_ in CANDIDATES]),ncc_shifts=np.array([r['ncc_shifts'] for r in allr]),ncc_scores=np.array([r['ncc_scores'] for r in allr]))
    for n,_ in CANDIDATES:
        for field,target in [('D','D'),('centroidD','centroidD'),('H','H'),('success','success'),('roi_success','roi_success'),('ecc','ecc'),('photometric_rmse','rmse'),('gain','gain'),('offset','offset')]:arrays[target+'_'+n]=np.array([r['candidates'][n][field] for r in allr])
    np.savez_compressed(H/'features.npz',**arrays);pd.DataFrame(rows).to_csv(H/'manifest.csv',index=False,encoding='utf-8-sig')
    audit=dict(n=len(m),actual_image_bytes_read=len(allr),candidates=[n for n,_ in CANDIDATES],protocol_sha256=protocol_hash,protocol_unchanged=protocol_hash==hashlib.sha256((H/'protocol.json').read_bytes()).hexdigest(),successes={n:int(arrays['success_'+n].sum()) for n,_ in CANDIDATES},failures={n:int((~arrays['success_'+n]).sum()) for n,_ in CANDIDATES},roi_failure_counts={n:dict(zip(NAMES,(~arrays['roi_success_'+n]).sum(0).tolist())) for n,_ in CANDIDATES},total_seconds=time.perf_counter()-began,median_seconds=float(np.median([r['seconds'] for r in allr])),cv2_version=cv2.__version__,old_QR_D_used=False,limits=PROTOCOL['limits'],observations='Non-QR external bolts/bracket/handle. No known metric depth, calibrated physical parallax or measured pose is claimed.')
    (H/'measurement_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8');print(json.dumps(audit,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');a=p.parse_args()
    with threadpool_limits(1):main(a.smoke)
