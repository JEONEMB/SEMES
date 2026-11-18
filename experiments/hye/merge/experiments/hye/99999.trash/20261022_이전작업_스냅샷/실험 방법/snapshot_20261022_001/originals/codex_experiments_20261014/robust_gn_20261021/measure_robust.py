"""QR-only robust photometric GN; same630 center structural IDs; no labels in tracking."""
from pathlib import Path
import sys,json,time,concurrent.futures
import numpy as np,cv2
from scipy import ndimage
H=Path(__file__).resolve().parent;B=H.parent
sys.path[:0]=[str(B/'measurement'),str(B/'center_patch_v2_20261021')]
from spread import SpreadPairExtractor
from refine import Refiner
from measure import ROOT
from run import spatial
class RobustRefiner(Refiner):
    def __init__(self,ref,points,kind='huber',delta=6):
        super().__init__(ref,points,17,.8);self.kind=kind;self.delta=delta
        smooth=cv2.GaussianBlur(ref.astype(float),(0,0),.8);gy,gx=np.gradient(smooth)
        tx=ndimage.map_coordinates(gx,self.coords,order=3,mode='nearest');ty=ndimage.map_coordinates(gy,self.coords,order=3,mode='nearest')
        A=np.stack([tx,ty,self.template,np.ones_like(tx)],axis=-1)
        self.scale=np.maximum(np.sqrt(np.mean(A*A,axis=1)),1e-8);self.A=A/self.scale[:,None,:]
    def run(self,cur,initial):
        cur=cv2.GaussianBlur(cur.astype(float),(0,0),.8);coeff=ndimage.spline_filter(cur,order=3);d=initial.astype(float).copy()
        for it in range(12):
            values=ndimage.map_coordinates(coeff,self.coords+d[:,::-1].T[:,:,None],order=3,prefilter=False,mode='nearest')
            pars=np.einsum('nkp,np->nk',self.pinverse,values,optimize=True)
            for _ in range(2):
                resid=values-np.einsum('npk,nk->np',self.A,pars*self.scale)
                rr=abs(resid)/self.delta
                w=np.minimum(1,1/np.maximum(rr,1e-12)) if self.kind=='huber' else 1/(1+rr*rr)
                G=np.einsum('npi,np,npj->nij',self.A,w,self.A,optimize=True)
                rhs=np.einsum('npi,np,np->ni',self.A,w,values,optimize=True)
                pars=np.linalg.solve(G+np.eye(4)[None]*1e-9,rhs[...,None])[...,0]/self.scale
            step=np.clip(pars[:,:2]/np.maximum(pars[:,2,None],.25),-.5,.5);d-=step
            if np.max(np.linalg.norm(step,axis=1))<2e-5:break
        residual=values-pars[:,2,None]*self.template-pars[:,3,None]
        q=np.c_[np.sqrt(np.mean(residual**2,axis=1)),np.linalg.norm(step,axis=1),pars[:,2],pars[:,3]]
        valid=np.isfinite(d).all(axis=1)&(np.linalg.norm(d-initial,axis=1)<1.5)&(q[:,1]<.02)
        return d.astype(np.float32),q.astype(np.float32),valid
def verify():
    rng=np.random.default_rng(17);ref=cv2.GaussianBlur(rng.uniform(20,220,(96,96)),(0,0),1.2)
    pts=np.array([[25,25],[50,25],[25,50],[50,50]],np.float32);yy,xx=np.mgrid[:96,:96]
    cur=ndimage.map_coordinates(ref,[yy-.12,xx-.23],order=3,mode='reflect')
    ordinary=Refiner(ref,pts,17,.8).run(cur,np.zeros_like(pts))[0]
    robust=RobustRefiner(ref,pts,'huber',1e9).run(cur,np.zeros_like(pts))[0]
    diff=float(abs(ordinary-robust).max());err=float(abs(robust-[.23,.12]).max())
    assert diff<1e-5 and err<.02,(diff,err)
    (H/'numerical_checks.json').write_text(json.dumps(dict(uniform_weight_matches_original_max_px=diff,known_translation_max_error_px=err),indent=2))
def main():
    verify();a=np.load(B/'center_patch_v2_20261021/tracks.npz');names=a['names'];pts=a['points'];audit=[]
    for kind,delta in [('huber',3),('huber',6),('huber',12),('cauchy',6)]:
        key=f'{kind}{delta}';ext=SpreadPairExtractor(ROOT/'center.bmp',str(B/'center_patch_v2_20261021/features.npz'));start=time.time()
        for k,(x0,y0,x1,y1) in enumerate(ext.local_boxes):ext.models[k]=RobustRefiner(ext.ref[y0:y1,x0:x1],ext.local_points[k],kind,delta)
        paths=[ROOT/('center.bmp' if n=='center.bmp' else ('Y/' if n[0]=='Y' else 'Theta/')+n) for n in names]
        results=[]
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
            for i,r in enumerate(ex.map(ext.extract,paths)):
                results.append(r)
                if i%70==0:print(key,i,round(time.time()-start,1),flush=True)
        D=np.stack([r['D'] for r in results]).astype(float);D-=D[0];valid=np.stack([r['valid'] for r in results]);assert np.isfinite(D).all()
        np.savez_compressed(H/(key+'_tracks.npz'),names=names,points=pts,D=D,valid=valid,quality=np.stack([r['quality'] for r in results]))
        np.save(H/(key+'_features.npy'),spatial(pts,D));audit.append(dict(key=key,valid_fraction=float(valid.mean()),seconds=time.time()-start));print(audit[-1],flush=True)
    (H/'audit.json').write_text(json.dumps(dict(method='GN17 sigma.8, translation gain offset, pixel IRLS2 per GN iteration; all630 retained; ROI crop extension; hyperparameters fixed before this run',results=audit),indent=2))
if __name__=='__main__':main()
