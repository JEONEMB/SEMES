from pathlib import Path
import sys,json,time,hashlib,argparse
import cv2,numpy as np,pandas as pd
from extractor import LineExtractor,PROTOCOL

H=Path(__file__).resolve().parent;ROOT=H.parents[2]

def plain(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    raise TypeError(type(x).__name__)

def main(smoke):
    cv2.setNumThreads(1);source=ROOT/'codex_experiments_20261014/joint_yt_generalization_20261022/measurements/manifest.csv';m=pd.read_csv(source);ex=LineExtractor();out=[];rows=[];start=time.perf_counter();ph=hashlib.sha256((H/'protocol.json').read_bytes()).hexdigest();indices=[0,281,336] if smoke else range(len(m))
    for i in indices:
        meta=m.iloc[i];r=ex.extract_bytes(Path(meta.path).read_bytes());out.append(r);row=meta.to_dict();row.update(index=i,line_image_sha256=r['image_sha256'],line_seconds=r['seconds'],success_upper=r['success_upper'],success_lower=r['success_lower'],invalid_upper=int((~r['valid_blocks'][:24]).sum()),invalid_lower=int((~r['valid_blocks'][24:]).sum()),line_failure_reasons=json.dumps(r['reasons']));rows.append(row)
        if smoke or (i+1)%30==0:print(json.dumps(dict(processed=len(out),index=i,elapsed=time.perf_counter()-start,success_upper=r['success_upper'],success_lower=r['success_lower'],shift_range=[float(r['measured_shifts'].min()),float(r['measured_shifts'].max())])),flush=True)
    if smoke:
        (H/'smoke_test.json').write_text(json.dumps(out,default=plain,indent=2),encoding='utf-8');return
    arrays={k:np.array([r[k] for r in out]) for k in ['X_upper','X_lower','success_upper','success_lower','measured_shifts','valid_blocks','rmse','correlation','gain','offset','contrast']};arrays.update(index=np.arange(len(m)),block_names=np.array([b['id'] for b in ex.blocks]),line_names=ex.line,boxes=np.array([b['read_box'] for b in ex.blocks]),reference_y=np.array([b['reference_y'] for b in ex.blocks]),x_centers=np.array([(b['x0']+b['x1'])/2 for b in ex.blocks]),reference_profiles=ex.reference_profiles)
    np.savez_compressed(H/'features.npz',**arrays);pd.DataFrame(rows).to_csv(H/'manifest.csv',index=False,encoding='utf-8-sig')
    audit=dict(n=len(m),current_raw_images_read=len(out),old_external_or_QR_displacements_used=False,protocol_sha256=ph,protocol_unchanged=ph==hashlib.sha256((H/'protocol.json').read_bytes()).hexdigest(),valid_read_pixels=int(ex.read_mask.sum()),block_count=48,features_per_line=24,successes={line:int(arrays['success_'+line].sum()) for line in ['upper','lower']},invalid_counts_per_block=dict(zip(arrays['block_names'],(~arrays['valid_blocks']).sum(0).tolist())),median_seconds=float(np.median([r['seconds'] for r in out])),total_seconds=time.perf_counter()-start,limits=PROTOCOL['limits'])
    (H/'measurement_audit.json').write_text(json.dumps(audit,default=plain,indent=2),encoding='utf-8');print(json.dumps(audit,default=plain,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');args=p.parse_args();main(args.smoke)
