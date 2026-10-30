from pathlib import Path
import re,json,time,hashlib
import cv2,numpy as np,pandas as pd
from threadpoolctl import threadpool_limits
from measurement import StructuralExtractor
HERE=Path(__file__).resolve().parent;DATA=HERE.parent/'hw-analysis/data/hw/bmp'

def label(path):
    stem=path.stem;year=path.relative_to(DATA).parts[0]
    if re.fullmatch('center[0-9]*',stem,re.I):return dict(Y=0.,T=0.,scope='center',group='center',condition=stem)
    clean=re.sub(r'\s*\(\d+\)$','',stem);values={'Y':0.,'T':0.}
    for a,sign,v in re.findall(r'([YT])([+-])(\d+(?:\.\d*)?)',clean):values[a]=float(v)*(1 if sign=='+' else -1)
    active=[a for a in values if values[a]!=0];scope='mixed' if len(active)==2 else 'single'
    if year=='20261014':
        axis=active[0];v=values[axis];group=('T' if axis=='T' else ('Y2' if abs(round(v/.05))%2 else 'Y1'))+('+' if v>0 else '-')
    else:group='date20261020' # No assumption that repetition 1/2/3 is an independently known trial.
    return dict(**values,scope=scope,group=group,condition=f"Y{values['Y']:+.4f}_T{values['T']:+.4f}")

if __name__=='__main__':
    paths=sorted(DATA.rglob('*.bmp'));assert len(paths)==341
    center=cv2.imdecode(np.fromfile(DATA/'20261014/center.bmp',np.uint8),0);extractor=StructuralExtractor(center);p=extractor.points;u,v=(p[:,0]-1906)/486,(p[:,1]-1070)/122;B=np.c_[np.ones(630),u,v,u*u,u*v,v*v];pinv=np.linalg.pinv(B)
    manifest=[];features=[];Ds=[];valid=[];quality=[];start=time.perf_counter()
    with threadpool_limits(1):
        for i,path in enumerate(paths):
            row=dict(index=i,path=str(path.resolve()),file=path.name,relative=str(path.relative_to(DATA)),**label(path));t=time.perf_counter()
            try:
                raw=path.read_bytes();row['sha256']=hashlib.sha256(raw).hexdigest();img=cv2.imdecode(np.frombuffer(raw,np.uint8),0);decoded=time.perf_counter();r=extractor.extract(img);measured=time.perf_counter();X=(pinv@r['D']).ravel();end=time.perf_counter()
                assert r['ids'].tolist()==extractor.ids.tolist() and len(r['D'])==630
                row.update(success=True,decode_s=decoded-t,measure_s=measured-decoded,regression_s=end-measured,total_s=end-t,strict_success=min(r['strict_scores'])>=.6,pad=r['used_pad'],valid=int(r['valid'].sum()),ncc_min=float(min(r['coarse_score'])))
                Ds.append(r['D']);valid.append(r['valid']);quality.append(r['quality']);features.append(X)
            except Exception as exc:
                row.update(success=False,error=str(exc),total_s=time.perf_counter()-t,strict_success=False);features.append(np.full(12,np.nan));Ds.append(np.full((630,2),np.nan));valid.append(np.zeros(630,bool));quality.append(np.full((630,5),np.nan))
            manifest.append(row)
            if i%30==0:print('measured',i+1,'/',len(paths),'seconds',round(time.perf_counter()-start,1),flush=True)
    pd.DataFrame(manifest).to_csv(HERE/'results/manifest.csv',index=False)
    np.savez_compressed(HERE/'results/measurements.npz',X=features,D=Ds,valid=valid,quality=quality,ids=extractor.ids,points=p,regions=extractor.regions)
    print('complete',len(paths),'success',sum(r['success'] for r in manifest),'strict failures',sum(not r['strict_success'] for r in manifest),flush=True)
