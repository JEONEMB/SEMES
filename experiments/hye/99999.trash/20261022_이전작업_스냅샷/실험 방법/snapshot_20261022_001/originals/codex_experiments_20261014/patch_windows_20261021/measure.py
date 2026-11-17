from pathlib import Path
import sys,time,concurrent.futures,json
import numpy as np
H=Path(__file__).resolve().parent;B=H.parent
sys.path[:0]=[str(B/'measurement'),str(B/'center_patch_v2_20261021')]
from spread import SpreadPairExtractor
from refine import Refiner
from measure import ROOT
from run import spatial
a=np.load(B/'center_patch_v2_20261021/tracks.npz');names=a['names'];pts=a['points'];featurefile=B/'center_patch_v2_20261021/features.npz'
settings=[(9,.8),(13,.8),(25,.8),(31,.8),(17,.4),(31,.4)]
audit=[]
for window,sigma in settings:
    key=f'gn{window}_sigma{sigma}';ext=SpreadPairExtractor(ROOT/'center.bmp',str(featurefile));start=time.time()
    for k,(x0,y0,x1,y1) in enumerate(ext.local_boxes):ext.models[k]=Refiner(ext.ref[y0:y1,x0:x1],ext.local_points[k],window,sigma)
    paths=[ROOT/('center.bmp' if n=='center.bmp' else ('Y/' if n[0]=='Y' else 'Theta/')+n) for n in names]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:results=list(ex.map(ext.extract,paths))
    D=np.stack([r['D'] for r in results]).astype(float);D-=D[0];valid=np.stack([r['valid'] for r in results]);assert np.isfinite(D).all()
    np.savez_compressed(H/(key+'_tracks.npz'),names=names,points=pts,D=D,valid=valid,quality=np.stack([r['quality'] for r in results]))
    np.save(H/(key+'_features.npy'),spatial(pts,D));audit.append(dict(key=key,window=window,sigma=sigma,valid_fraction=float(valid.mean()),seconds=time.time()-start))
    print(audit[-1],flush=True)
(H/'audit.json').write_text(json.dumps(dict(method='same center630 IDs, LK15 initialization; GN window/blur varied only. QR crop boundary extension, all finite tracks retained even diagnostic-invalid.',results=audit),indent=2))
