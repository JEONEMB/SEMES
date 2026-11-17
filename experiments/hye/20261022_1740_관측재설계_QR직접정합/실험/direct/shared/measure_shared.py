"""Separate shared-plane measurement batch. Never modifies per-QR outputs."""
from pathlib import Path
import sys,json,argparse,time
import cv2,numpy as np,pandas as pd
from threadpoolctl import threadpool_limits
from shared_extractor import SharedExtractor,CANDIDATES,BBOX,MARGIN,base

H=Path(__file__).resolve().parent;ROOT=H.parents[3]
sys.path.insert(0,str(ROOT/'codex_experiments_20261014/joint_yt_generalization_20261022'))
from data import load

def plain(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    raise TypeError(type(x).__name__)

def main(smoke):
    cv2.setNumThreads(1);m,_=load();ex=SharedExtractor();started=time.perf_counter();take=[0,281,336] if smoke else range(len(m));results=[];rows=[]
    for i in take:
        meta=m.iloc[i];r=ex.extract_bytes(Path(meta.path).read_bytes());results.append(r);row=meta.to_dict();row.update(index=i,measurement_seconds=r['seconds'],ncc_pad=r['ncc_pad'],ncc_min=float(r['ncc_scores'].min()),input_bytes_sha256=r['image_sha256'])
        for name,_,_ in CANDIDATES:
            c=r['candidates'][name];row.update({name+'_success':c['success'],name+'_ecc_min':float(c['ecc'].min()),name+'_rmse_mean':float(c['photometric_rmse'].mean()),name+'_seconds':c['seconds'],name+'_reasons':json.dumps(c['reasons'])})
        rows.append(row)
        if smoke or (i+1)%20==0:print(json.dumps(dict(processed=len(results),index=i,elapsed=time.perf_counter()-started,success={n:r['candidates'][n]['success'] for n,_,_ in CANDIDATES})),flush=True)
    if smoke:
        report=[]
        for i,r in zip(take,results):report.append(dict(index=i,seconds=r['seconds'],candidates={n:{k:v for k,v in c.items() if k!='D'} for n,c in r['candidates'].items()}))
        (H/'smoke_test.json').write_text(json.dumps(report,default=plain,indent=2),encoding='utf-8');return
    a=dict(index=np.arange(len(m)),points=ex.points,ids=ex.ids,regions=ex.regions,candidate_names=np.array([x[0] for x in CANDIDATES]),ncc_shifts=np.array([r['ncc_shifts'] for r in results]),ncc_scores=np.array([r['ncc_scores'] for r in results]))
    for name,_,_ in CANDIDATES:
        for field,target in [('D','D'),('H','H'),('H_shared','H_single'),('ecc','ecc'),('photometric_rmse','rmse'),('brightness_gain','gain'),('brightness_offset','offset'),('qr_success','qr_success'),('success','success')]:a[target+'_'+name]=np.array([r['candidates'][name][field] for r in results])
    np.savez_compressed(H/'features.npz',**a);pd.DataFrame(rows).to_csv(H/'manifest.csv',index=False,encoding='utf-8-sig')
    audit=dict(n=len(m),candidates=[list(x) for x in CANDIDATES],bbox=BBOX,margin=MARGIN,template_inset=base.TEMPLATE_INSET,mask_pixels=int((ex.joint_mask>0).sum()),actual_image_bytes_read=len(results),old_D_used_as_measurement=False,center_sha256=base.CENTER_SHA256,cv2_version=cv2.__version__,failures={n:int(np.sum(~a['success_'+n])) for n,_,_ in CANDIDATES},total_seconds=time.perf_counter()-started,median_seconds=float(np.median([r['seconds'] for r in results])),max_seconds=float(max(r['seconds'] for r in results)),new_assumption='All3 QR interiors share a single affine/projective image warp: common-plane/common-motion approximation. Different QR depths or local optical distortion can violate it.',inference_inputs='Fixed center and current image bytes only. Fixed QR masks; median of image-NCC translations initializes one shared warp.',fallback='NCC130/260/512 expansion only. ECC exceptions/invalid warps remain failures with NaN D; no old or perQR displacement replacement.',background_usage='Objective mask includes the fixed interiors of the3 QR boxes. Inter-QR background pixels never enter the objective.',compatibility_H='H_<candidate> repeats the same3x3 global center-to-current map3times. H_single_<candidate> stores341x3x3.',sources=['https://sites.google.com/site/georgeevangelidis/ecc','https://docs.opencv.org/4.13.0/dc/d6b/group__video__track.html'],limits=['Same341 development images, no independent performance claim.','Common-plane hypothesis is a new comparison, not a measured physical calibration.','Background exists in the rectangular buffer but masked pixels do not enter ECC; fixed6px inset isolates valid QR pixels from3/5px smoothing of background.'])
    (H/'measurement_audit.json').write_text(json.dumps(audit,default=plain,indent=2),encoding='utf-8');print(json.dumps(audit,default=plain,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');args=p.parse_args()
    with threadpool_limits(1):main(args.smoke)
