from pathlib import Path
import sys,concurrent.futures,time,json
import numpy as np,cv2
H=Path(__file__).resolve().parent;B=H.parent
sys.path[:0]=[str(B/'measurement'),str(B/'center_patch_v2_20261021')]
from measure import ROOT,read,coarse,BOXES,CROP
from run import spatial
cv2.setNumThreads(1)
a=np.load(B/'center_patch_v2_20261021/tracks.npz');names=a['names'];pts=a['points'];ref=read(ROOT/'center.bmp')
regs=np.argmin(abs(pts[:,0,None]-[1644,1901,2159]),axis=1)
templates=[];locals=[];boxes=[]
for k,(x0,y0,x1,y1) in enumerate(BOXES):
    boxes.append((x0-CROP[0],y0-CROP[1],x1-CROP[0],y1-CROP[1]))
    templates.append(ref[y0-CROP[1]:y1-CROP[1],x0-CROP[0]:x1-CROP[0]].astype(np.float32)/255)
    locals.append(pts[regs==k]-[x0,y0])
def job(name):
    path=ROOT/('center.bmp' if name=='center.bmp' else ('Y/' if name[0]=='Y' else 'Theta/')+name)
    cur=read(path);shift,_=coarse(ref,cur,False);ds={t:np.zeros_like(pts) for t in ['ecc_affine','ecc_homography','phase']};fail=[]
    for k,(x0,y0,x1,y1) in enumerate(boxes):
        dx,dy=shift[k].astype(int);target=cur[y0+dy:y1+dy,x0+dx:x1+dx].astype(np.float32)/255;t=templates[k];p=locals[k]
        for key,motion in [('ecc_affine',cv2.MOTION_AFFINE),('ecc_homography',cv2.MOTION_HOMOGRAPHY)]:
            warp=np.eye(3,dtype=np.float32) if motion==cv2.MOTION_HOMOGRAPHY else np.eye(2,3,dtype=np.float32)
            try:
                score,warp=cv2.findTransformECC(t,target,warp,motion,(cv2.TERM_CRITERIA_COUNT|cv2.TERM_CRITERIA_EPS,100,1e-7),None,3)
                q=np.c_[p,np.ones(len(p))]@warp.T
                if q.shape[1]==3:q=q[:,:2]/q[:,2:]
                ds[key][regs==k]=q-p+shift[k]
            except cv2.error:
                # Explicit coarse-only fallback, counted and disclosed.
                fail.append([key,k]);ds[key][regs==k]=shift[k]
        window=cv2.createHanningWindow((t.shape[1],t.shape[0]),cv2.CV_32F)
        delta,response=cv2.phaseCorrelate(t.copy(),target.copy(),window)
        ds['phase'][regs==k]=shift[k]+delta
    return ds,fail
start=time.time();results=[]
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
    for i,r in enumerate(ex.map(job,names)):
        results.append(r)
        if i%40==0:print('ECC/phase',i+1,round(time.time()-start,1),flush=True)
for key in results[0][0]:
    D=np.stack([r[0][key] for r in results]);D-=D[0]
    np.savez_compressed(H/(key+'_tracks.npz'),names=names,points=pts,D=D)
    np.save(H/(key+'_features.npy'),spatial(pts,D))
(H/'measurement_audit.json').write_text(json.dumps(dict(failures=[dict(file=str(n),fallback=f) for n,(_,f) in zip(names,results) if f],seconds=time.time()-start,notes='QR-crop-only area registration; affine/H8 projected onto same630 IDs; phase gives QR-wise translation only. ECC is OpenCV, phase is OpenCV Hanning phase correlation, not Guizar DFT-upsample implementation.'),indent=2))
