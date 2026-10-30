"""Frozen decoded-grid IDs and original blurred-boundary measurement, image bytes only."""
from pathlib import Path
import json,time
import cv2,numpy as np
from scipy.optimize import least_squares
from scipy.special import erf
HERE=Path(__file__).resolve().parent
BOXES=[(1552,969,1736,1167),(1809,971,1993,1169),(2067,973,2252,1171)]
CORNERS=json.loads((HERE/'assets/corners.json').read_text())
REF=np.load(HERE/'assets/reference.npz')

def field(xy,center,axes,sigma,patch):
    local=(xy-center)@np.linalg.inv(axes).T
    U=.5*(1+erf(local[:,0]/(np.sqrt(2)*sigma[0])));V=.5*(1+erf(local[:,1]/(np.sqrt(2)*sigma[1])))
    return patch[0,0]*(1-U)*(1-V)+patch[0,1]*U*(1-V)+patch[1,0]*(1-U)*V+patch[1,1]*U*V

def fit(xy,I,start,axes,patch):
    low,high=np.percentile(I,[10,90]);contrast=max(high-low,20)
    p0=[*start,1.,1.,low,contrast]
    lo=[start[0]-3.8,start[1]-3.8,.35,.35,0,10];hi=[start[0]+3.8,start[1]+3.8,2.7,2.7,240,255]
    def residual(p):return p[4]+p[5]*(1-field(xy,p[:2],axes,p[2:4],patch))-I
    r=least_squares(residual,p0,bounds=(lo,hi),loss='soft_l1',f_scale=6,max_nfev=85,xtol=1e-7,ftol=1e-7,gtol=1e-7)
    return r.x,float(np.sqrt(np.mean(residual(r.x)**2))),r.success

class StructuralExtractor:
    def __init__(self,center):
        self.center=center;self.points=REF['points'];self.ids=REF['ids'];self.regions=REF['regions']
        assert len(self.points)==630 and list(self.ids)==[c['id'] for c in CORNERS]
    def extract(self,img):
        start_time=time.perf_counter();shifts=[];scores=[]
        for x0,y0,x1,y1 in BOXES:
            pad=130
            corr=cv2.matchTemplate(img[y0-pad:y1+pad,x0-pad:x1+pad],self.center[y0:y1,x0:x1],cv2.TM_CCOEFF_NORMED)
            _,score,_,loc=cv2.minMaxLoc(corr);shifts.append((loc[0]-pad,loc[1]-pad));scores.append(score)
        if min(scores)<.6:raise ValueError('QR 위치 초기화 실패: NCC가 0.6 미만입니다.')
        ncc_seconds=time.perf_counter()-start_time
        fit9_seconds=fit11_seconds=window_seconds=0.
        current=[];qualities=[];valid=[];reasons=[]
        pixels=img.astype(float)
        for c in CORNERS:
            dx,dy=shifts[c['qr']-1];initial=np.array([c['cx']+dx,c['cy']+dy]);axes=np.array(c['axes'],dtype=np.float32);patch=np.array(c['patch'])
            a,b,cc,d=BOXES[c['qr']-1];a,cc,b,d=a+dx,cc+dx,b+dy,d+dy
            def window(center,radius):
                x0,y0=np.rint(center).astype(int);yy,xx=np.mgrid[y0-radius:y0+radius+1,x0-radius:x0+radius+1]
                xy=np.c_[xx.ravel(),yy.ravel()].astype(float);I=pixels[yy,xx].ravel()
                inside=(xy[:,0]>=a)&(xy[:,0]<cc)&(xy[:,1]>=b)&(xy[:,1]<d)
                return xy[inside],I[inside]
            est=initial
            for _ in range(3):
                tw=time.perf_counter();xy,I=window(est,4);window_seconds+=time.perf_counter()-tw
                tf=time.perf_counter();f1=fit(xy,I,initial,axes,patch);fit9_seconds+=time.perf_counter()-tf
                moved=np.any(np.rint(f1[0][:2])!=np.rint(est));est=f1[0][:2]
                if not moved:break
            p1,rmse,ok=f1
            tw=time.perf_counter();xy,I=window(est,5);window_seconds+=time.perf_counter()-tw
            tf=time.perf_counter();p2,_,ok2=fit(xy,I,initial,axes,patch);fit11_seconds+=time.perf_counter()-tf
            stability=float(np.linalg.norm(p1[:2]-p2[:2]));why=[]
            if not(ok and ok2):why.append('optimizer')
            if rmse>15:why.append('photometric_rmse')
            if p1[5]<45:why.append('low_contrast')
            if stability>.65:why.append('window_instability')
            if np.max(abs(p1[:2]-initial))>3.6:why.append('search_limit')
            if max(p1[2:4])>2.6:why.append('blur_limit')
            current.append(p1[:2]);qualities.append([rmse,stability,p1[5],p1[2],p1[3]]);valid.append(not why);reasons.append(why)
        coords=np.array(current)
        return dict(points=self.points,current_points=coords,D=coords-self.points,valid=np.array(valid),quality=np.array(qualities),regions=self.regions,ids=self.ids,reasons=reasons,coarse_score=np.array(scores),runtime_seconds=time.perf_counter()-start_time,timings=dict(ncc=ncc_seconds,fit9=fit9_seconds,fit11=fit11_seconds,windows=window_seconds))
