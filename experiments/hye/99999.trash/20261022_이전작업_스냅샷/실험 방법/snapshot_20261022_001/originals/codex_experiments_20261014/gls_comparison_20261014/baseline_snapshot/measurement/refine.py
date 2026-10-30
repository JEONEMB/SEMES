"""Independent cubic-interpolated, brightness-adjusted inverse GN refinement.

Initial positions come from full-QR coarse+LK. Each template models local
translation, gain, and offset. The fixed template gradient gives an inverse
compositional approximation; gain/offset are eliminated analytically each step.
The output photometric residual is a fit diagnostic, not motion uncertainty.
"""
from pathlib import Path
import argparse, concurrent.futures, csv, json, time
import numpy as np
import cv2
from scipy import ndimage
from measure import ROOT,OUT,CROP,read,feature_set,coarse,track

cv2.setNumThreads(1)

class Refiner:
    def __init__(self,ref,points,window=17,sigma=.8):
        self.points=points;self.window=window;self.sigma=sigma
        ref=cv2.GaussianBlur(ref.astype(np.float64),(0,0),sigma)
        yy,xx=np.mgrid[-window//2+1:window//2+1,-window//2+1:window//2+1]
        self.coords=np.stack([points[:,1,None]+yy.ravel(),points[:,0,None]+xx.ravel()])
        self.template=ndimage.map_coordinates(ref,self.coords,order=3,mode='nearest')
        gy,gx=np.gradient(ref)
        tx=ndimage.map_coordinates(gx,self.coords,order=3,mode='nearest')
        ty=ndimage.map_coordinates(gy,self.coords,order=3,mode='nearest')
        design=np.stack([tx,ty,self.template,np.ones_like(tx)],axis=-1)
        self.pinverse=np.linalg.pinv(design,rcond=1e-10)

    def run(self,cur,initial):
        cur=cv2.GaussianBlur(cur.astype(np.float64),(0,0),self.sigma)
        coeff=ndimage.spline_filter(cur,order=3)
        d=initial.astype(np.float64).copy()
        for it in range(12):
            coords=self.coords+d[:,::-1].T[:,:,None]
            values=ndimage.map_coordinates(coeff,coords,order=3,prefilter=False,mode='nearest')
            pars=np.einsum('nkp,np->nk',self.pinverse,values,optimize=True)
            step=pars[:,:2]/np.maximum(pars[:,2,None],.25)
            step=np.clip(step,-.5,.5)
            d-=step
            if np.max(np.linalg.norm(step,axis=1))<2e-5:break
        resid=values-pars[:,2,None]*self.template-pars[:,3,None]
        quality=np.c_[np.sqrt(np.mean(resid**2,axis=1)),np.linalg.norm(step,axis=1),pars[:,2],pars[:,3]]
        valid=np.isfinite(d).all(axis=1)&(np.linalg.norm(d-initial,axis=1)<1.5)&(quality[:,1]<.02)
        return d.astype(np.float32),quality.astype(np.float32),valid

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--window',type=int,default=17);args=ap.parse_args()
    if args.window not in [17,31]:raise ValueError('window must be 17 or 31')
    tag='gn'+str(args.window)
    start=time.time();base=np.load(OUT/'lk31_blur.npz');points=base['points']-np.array(CROP[:2]);regions=base['regions'];names=base['names']
    ref=read(ROOT/'center.bmp')
    model=Refiner(ref,points,args.window,.8)
    def job(i):
        name=names[i];path=ROOT/name if name=='center.bmp' else ROOT/('Y' if name[0]=='Y' else 'Theta')/name
        return model.run(read(path),base['D'][i])
    outputs=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
        for i,result in enumerate(ex.map(job,range(len(names)))):
            outputs.append(result)
            if i%20==0:print('GN',i+1,len(names),round(time.time()-start,1),flush=True)
    D=np.stack([x[0] for x in outputs]);quality=np.stack([x[1] for x in outputs]);valid=np.stack([x[2] for x in outputs])
    np.savez_compressed(OUT/(tag+'.npz'),names=names,points=base['points'],regions=regions,D=D,quality=quality,valid=valid,strengths=base['strengths'])
    meta={'method':'inverse compositional GN, cubic spline image sampling, sigma .8, fitted photometric gain/offset','window':args.window,'quality_columns':['photometric_rmse_gray','last_step_px','gain','offset_gray'],'QR_valid_fraction':float(valid[:,regions<3].mean()),'max_initial_difference_px':float(np.linalg.norm(D-base['D'],axis=2)[:,regions<3].max()),'runtime_seconds':time.time()-start}
    (OUT/(tag+'_metadata.json')).write_text(json.dumps(meta,indent=2),encoding='utf8');print(meta,flush=True)
    synthetic(ref,points,regions,model,tag)

def synthetic(ref,points,regions,model,tag='gn17'):
    rows=[]
    # A cubic-spline resampling has no OpenCV 1/32-pixel transform quantization.
    yy,xx=np.mgrid[:ref.shape[0],:ref.shape[1]]
    for dx,dy in [(0.1,.07),(.25,-.3),(.5,.5),(7.7,-1.2),(28.35,.08),(80.15,-.45),(-79.4,.3)]:
        cur=ndimage.map_coordinates(ref.astype(np.float64),[yy-dy,xx-dx],order=3,mode='reflect')
        cur=np.clip(np.rint(cur),0,255).astype(np.uint8)
        init,score=coarse(ref,cur)
        blurred_ref=cv2.GaussianBlur(ref,(0,0),.8);blurred_cur=cv2.GaussianBlur(cur,(0,0),.8)
        d,q,v=track(blurred_ref,blurred_cur,points.astype(np.float32),init[regions],31)
        gd,gq,gv=model.run(cur,d)
        for name,dd,vv in [('lk31_blur',d,v),(tag,gd,gv)]:
            sel=(regions<3)&vv;err=dd[sel]-[dx,dy]
            rows.append(dict(method=name,dx=dx,dy=dy,valid_rate=float(vv[regions<3].mean()),bias_dx=float(err[:,0].mean()),bias_dy=float(err[:,1].mean()),rmse=float(np.sqrt((err**2).mean())),p95_euclidean=float(np.quantile(np.linalg.norm(err,axis=1),.95))))
    outfile='synthetic_cubic_validation.csv' if tag=='gn17' else 'synthetic_cubic_'+tag+'.csv'
    with (OUT/outfile).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    print('cubic synthetic complete',flush=True)

if __name__=='__main__':main()
