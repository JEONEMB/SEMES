"""External structure direct-pixel comparison. Fixed ROIs, current bytes only."""
from pathlib import Path
import json,hashlib,time
import cv2,numpy as np

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
PROTOCOL=json.loads((HERE/'protocol.json').read_text(encoding='utf-8'))
BOXES=tuple(tuple(x['box']) for x in PROTOCOL['rois']);NAMES=tuple(x['name'] for x in PROTOCOL['rois'])
CENTER_PATH=ROOT/'hw-analysis/data/hw/bmp/20261014/center.bmp'
CANDIDATES=(('translation_g5',cv2.MOTION_TRANSLATION),('affine_g5',cv2.MOTION_AFFINE))

def translate(x,y):return np.array([[1.,0.,x],[0.,1.,y],[0.,0.,1.]])
def project(H,p):
    v=np.c_[p,np.ones(len(p))]@H.T;return v[:,:2]/v[:,2,None]

class ParallaxExtractor:
    def __init__(self):
        data=CENTER_PATH.read_bytes()
        if hashlib.sha256(data).hexdigest()!=PROTOCOL['center_sha256']:raise ValueError('Fixed center changed')
        self.center=cv2.imdecode(np.frombuffer(data,np.uint8),0);self.points=[];self.ids=[];self.regions=[];self.templates=[];self.masks=[]
        for k,((x0,y0,x1,y1),name) in enumerate(zip(BOXES,NAMES)):
            template=self.center[y0:y1,x0:x1].copy();mask=np.zeros(template.shape,np.uint8);mask[6:-6,6:-6]=255;self.templates.append(template);self.masks.append(mask)
            for row,fy in enumerate([.25,.5,.75]):
                for col,fx in enumerate([.25,.5,.75]):self.points.append([x0+fx*(x1-x0),y0+fy*(y1-y0)]);self.ids.append(f'{name}:grid:{row}:{col}');self.regions.append(k)
        self.points=np.array(self.points);self.ids=np.array(self.ids);self.regions=np.array(self.regions)
        self.centroids=np.array([[(a+c)/2,(b+d)/2] for a,b,c,d in BOXES])

    def coarse(self,image):
        attempts=[]
        for pad in (130,260,512):
            shifts=[];scores=[]
            for (x0,y0,x1,y1),template in zip(BOXES,self.templates):
                left=max(0,x0-pad);top=max(0,y0-pad);right=min(image.shape[1],x1+pad);bottom=min(image.shape[0],y1+pad)
                corr=cv2.matchTemplate(image[top:bottom,left:right],template,cv2.TM_CCOEFF_NORMED);_,cc,_,loc=cv2.minMaxLoc(corr);shifts.append([loc[0]+left-x0,loc[1]+top-y0]);scores.append(float(cc))
            attempts.append(dict(pad=pad,scores=scores))
            if min(scores)>=.6:break
        return np.array(shifts),np.array(scores),pad,attempts

    def extract_bytes(self,image_bytes):
        started=time.perf_counter();image=cv2.imdecode(np.frombuffer(image_bytes,np.uint8),0)
        if image is None:raise ValueError('Could not decode image')
        if image.shape!=self.center.shape:raise ValueError('Image size differs from fixed center')
        t=time.perf_counter();shifts,scores,pad,attempts=self.coarse(image);result=dict(image_sha256=hashlib.sha256(image_bytes).hexdigest(),ncc_shifts=shifts,ncc_scores=scores,ncc_pad=pad,ncc_attempts=attempts,ncc_seconds=time.perf_counter()-t,candidates={})
        for name,motion in CANDIDATES:
            t=time.perf_counter();D=np.full(self.points.shape,np.nan);centroidD=np.full((5,2),np.nan);Hs=[];success=[];reasons=[];ecc=[];rmse=[];gain=[];offset=[]
            for k,((x0,y0,x1,y1),(dx,dy),template,mask) in enumerate(zip(BOXES,shifts,self.templates,self.masks)):
                HH=np.full((3,3),np.nan);why=[];cc=err=g=b=np.nan
                try:
                    if scores[k]<.6:raise ValueError('NCC below0.6 after expanded search')
                    left=x0+int(dx)-12;top=y0+int(dy)-12;right=x1+int(dx)+12;bottom=y1+int(dy)+12
                    if min(left,top)<0 or right>image.shape[1] or bottom>image.shape[0]:raise ValueError('ROI crop outside image')
                    crop=image[top:bottom,left:right].copy();warp=translate(12,12)[:2].astype(np.float32)
                    cc,local=cv2.findTransformECCWithMask(template,crop,mask,np.full(crop.shape,255,np.uint8),warp,motion,(cv2.TERM_CRITERIA_COUNT|cv2.TERM_CRITERIA_EPS,200,1e-8),5)
                    localH=np.vstack([local,[0.,0.,1.]]);HH=translate(left,top)@localH@translate(-x0,-y0);rr=self.regions==k;measured=project(HH,self.points[rr])-self.points[rr]
                    if not np.isfinite(measured).all():why.append('nonfinite')
                    if cc<.8:why.append('ECC below0.8')
                    if np.max(abs(measured-[dx,dy]))>10:why.append('refinement more than10px from NCC')
                    if not(.5<np.linalg.det(localH[:2,:2])<2):why.append('affine determinant outside0.5..2')
                    yy,xx=np.indices(template.shape,dtype=np.float32);p=project(localH,np.c_[xx.ravel(),yy.ravel()]);aligned=cv2.remap(crop.astype(np.float32),p[:,0].reshape(template.shape).astype(np.float32),p[:,1].reshape(template.shape).astype(np.float32),cv2.INTER_LINEAR);v=mask>0;A=np.c_[aligned[v],np.ones(v.sum())];g,b=np.linalg.lstsq(A,template[v].astype(float),rcond=None)[0];err=float(np.sqrt(np.mean((A@np.array([g,b])-template[v])**2)))
                    if not why:D[rr]=measured;centroidD[k]=project(HH,self.centroids[k:k+1])[0]-self.centroids[k]
                except (cv2.error,ValueError,np.linalg.LinAlgError) as exc:why=[type(exc).__name__+': '+str(exc)]
                Hs.append(HH);success.append(not why);reasons.append(why);ecc.append(float(cc));rmse.append(float(err));gain.append(float(g));offset.append(float(b))
            result['candidates'][name]=dict(D=D,centroidD=centroidD,H=np.array(Hs),roi_success=np.array(success),success=bool(all(success)),reasons=reasons,ecc=np.array(ecc),photometric_rmse=np.array(rmse),gain=np.array(gain),offset=np.array(offset),seconds=time.perf_counter()-t)
        result['seconds']=time.perf_counter()-started;return result

def extract_bytes(image_bytes):return ParallaxExtractor().extract_bytes(image_bytes)
