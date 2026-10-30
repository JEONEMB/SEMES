"""Affine synthetic localization check, independent of motion labels."""
import csv
import cv2,numpy as np
from scipy import ndimage
from measure import ROOT,OUT,CROP,read,coarse,track
from refine import Refiner

def main():
    data=np.load(OUT/'features.npz');points=(data['points']-np.array(CROP[:2])).astype(np.float32);regions=data['regions'];qr=regions<3
    ref=read(ROOT/'center.bmp');blurref=cv2.GaussianBlur(ref,(0,0),.8)
    models={f'gn{w}':Refiner(ref,points,w,.8) for w in [17,31]}
    yy,xx=np.mgrid[:ref.shape[0],:ref.shape[1]];center=np.array([ref.shape[1]/2,ref.shape[0]/2]);rows=[]
    cases=[('rotation',80.15,-.45,.03,0,0),('rotation',-79.4,.3,-.03,0,0),('rotation',.25,-.3,.10,0,0),('scale',28.35,.08,0,.001,0),('shear',7.7,-1.2,0,0,.001),('mixed',-10.2,.6,-.04,.0006,.0008)]
    for kind,dx,dy,angle,scale,shear in cases:
        t=np.deg2rad(angle);A=np.array([[np.cos(t),-np.sin(t)],[np.sin(t),np.cos(t)]])*(1+scale);A[0,1]+=shear
        b=center-A@center+np.array([dx,dy]);inv=np.linalg.inv(A);src=inv@(np.array([xx.ravel(),yy.ravel()])-b[:,None])
        cur=ndimage.map_coordinates(ref.astype(float),[src[1].reshape(ref.shape),src[0].reshape(ref.shape)],order=3,mode='reflect');cur=np.clip(np.rint(cur),0,255).astype(np.uint8)
        true=points@A.T+b-points;init,_=coarse(ref,cur);d,q,v=track(blurref,cv2.GaussianBlur(cur,(0,0),.8),points,init[regions],31)
        results={'lk31_blur':(d,v)}
        for name,model in models.items():gd,gq,gv=model.run(cur,d);results[name]=(gd,gv)
        for name,(dd,vv) in results.items():
            err=(dd-true)[qr&vv];rows.append(dict(method=name,case=kind,dx=dx,dy=dy,angle_deg=angle,scale_delta=scale,shear=shear,valid_rate=float(vv[qr].mean()),bias_dx=float(err[:,0].mean()),bias_dy=float(err[:,1].mean()),rmse=float(np.sqrt((err**2).mean())),p95_euclidean=float(np.quantile(np.linalg.norm(err,axis=1),.95))))
    with (OUT/'synthetic_affine_validation.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    for name in results:
        rs=[r for r in rows if r['method']==name];print(name,'minRMSE',min(r['rmse'] for r in rs),'maxRMSE',max(r['rmse'] for r in rs),'maxBias',max(max(abs(r['bias_dx']),abs(r['bias_dy'])) for r in rs),flush=True)

if __name__=='__main__':main()
