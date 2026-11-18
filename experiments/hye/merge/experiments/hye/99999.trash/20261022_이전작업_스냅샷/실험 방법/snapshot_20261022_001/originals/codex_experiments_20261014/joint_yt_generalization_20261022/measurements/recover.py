from pathlib import Path
import os,time,json,hashlib,sys
sys.dont_write_bytecode=True
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import numpy as np,pandas as pd,cv2
from concurrent.futures import ProcessPoolExecutor,as_completed
from extractor_scipy import StructuralExtractor
from prepare import H,ROOT,CACHE,SRC

def recover(row):
    t=time.perf_counter();center=cv2.imdecode(np.fromfile(ROOT/'hw-analysis/data/hw/bmp/20261014/center.bmp',np.uint8),0)
    raw=Path(row['extracted_path']).read_bytes();assert hashlib.sha256(raw).hexdigest()==row['image_sha256']
    image=cv2.imdecode(np.frombuffer(raw,np.uint8),0);ex=StructuralExtractor(center);r=ex.extract(image)
    np.savez_compressed(H/f"recovered_{row['index']}.npz",**{k:v for k,v in r.items() if k in ['D','quality','valid','points','ids','regions','current_points','coarse_score']})
    rec=dict(index=int(row['index']),file=row['name'],sha256=row['image_sha256'],seconds=time.perf_counter()-t,valid=int(r['valid'].sum()),ncc_min=float(r['coarse_score'].min()),source='expanded_scipy_same_objective')
    (H/f"recovered_{row['index']}.json").write_text(json.dumps(rec,indent=2),encoding='utf-8')
    return rec

if __name__=='__main__':
    m=pd.read_csv(SRC/'source_manifest.csv');todo=[r for r in m.to_dict('records') if not (CACHE/f"result_{r['index']}.json").exists()]
    assert len(todo)==12
    with ProcessPoolExecutor(max_workers=4) as ex:
        for f in as_completed([ex.submit(recover,r) for r in todo]):print(json.dumps(f.result()),flush=True)
