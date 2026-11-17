"""Compiled per-corner bounded robust LM; same erf field and diagnostic windows."""
import sys,math,time,json
from pathlib import Path
import numpy as np,cv2
sys.path.append(str(Path(__file__).parent/'deps'))
from numba import njit,prange,set_num_threads
from baseline import BOXES,CORNERS,REF

@njit(cache=True)
def objective(p,xy,I,inv,patch):
    n=len(I);J=np.empty((n,6));res=np.empty(n);cost=0.
    a=patch[0];du=patch[1]-a;dv=patch[2]-a;cross=patch[3]-patch[1]-patch[2]+a
    for i in range(n):
        dx=xy[i,0]-p[0];dy=xy[i,1]-p[1]
        u=dx*inv[0,0]+dy*inv[0,1];v=dx*inv[1,0]+dy*inv[1,1]
        zu=u/p[2];zv=v/p[3];U=.5*(1+math.erf(zu/math.sqrt(2)));V=.5*(1+math.erf(zv/math.sqrt(2)))
        F=patch[0]*(1-U)*(1-V)+patch[1]*U*(1-V)+patch[2]*(1-U)*V+patch[3]*U*V
        r=p[4]+p[5]*(1-F)-I[i];res[i]=r;cost+=36*(math.sqrt(1+(r/6)**2)-1)
        gu=(du+cross*V)*math.exp(-.5*zu*zu)/math.sqrt(2*math.pi);gv=(dv+cross*U)*math.exp(-.5*zv*zv)/math.sqrt(2*math.pi)
        J[i,0]=p[5]*(gu/p[2]*inv[0,0]+gv/p[3]*inv[1,0]);J[i,1]=p[5]*(gu/p[2]*inv[0,1]+gv/p[3]*inv[1,1])
        J[i,2]=p[5]*gu*u/(p[2]*p[2]);J[i,3]=p[5]*gv*v/(p[3]*p[3]);J[i,4]=1.;J[i,5]=1-F
    return cost,res,J

@njit(cache=True)
def fit_compiled(xy,I,start,inv,patch):
    ordered=np.sort(I);q=.1*(len(I)-1);k=int(q);low=ordered[k]*(1-(q-k))+ordered[min(k+1,len(I)-1)]*(q-k)
    q=.9*(len(I)-1);k=int(q);high=ordered[k]*(1-(q-k))+ordered[min(k+1,len(I)-1)]*(q-k)
    p=np.array([start[0],start[1],1.,1.,low,max(high-low,20.)]);lower=np.array([start[0]-3.8,start[1]-3.8,.35,.35,0.,10.]);upper=np.array([start[0]+3.8,start[1]+3.8,2.7,2.7,240.,255.])
    scales=np.array([1.,1.,1.,1.,60.,120.]);mu=1e-3;converged=False
    cost,r,J=objective(p,xy,I,inv,patch);evaluations=1
    for iteration in range(85):
        H=np.zeros((6,6));g=np.zeros(6)
        for i in range(len(I)):
            root=math.sqrt(1+(r[i]/6)**2);w=1/(root**3)
            for j in range(6):
                jj=J[i,j]*scales[j];g[j]+=jj*r[i]/root
                for k in range(6):H[j,k]+=jj*J[i,k]*scales[k]*w
        pg=g.copy()
        for j in range(6):
            if (p[j]-lower[j]<1e-9 and g[j]>0) or (upper[j]-p[j]<1e-9 and g[j]<0):pg[j]=0
        if np.max(np.abs(pg))<1e-7:converged=True;break
        for j in range(6):
            if (p[j]-lower[j]<1e-8 and g[j]>0) or (upper[j]-p[j]<1e-8 and g[j]<0):
                H[j,:]=0.;H[:,j]=0.;H[j,j]=1.;g[j]=0.
        diagonal=np.maximum(np.diag(H),1e-6);accepted=False
        for retry in range(14):
            A=H.copy()
            for j in range(6):A[j,j]+=mu*diagonal[j]
            delta=np.linalg.solve(A,-g)*scales
            trial=np.minimum(np.maximum(p+delta,lower+1e-12),upper-1e-12)
            newcost,newr,newJ=objective(trial,xy,I,inv,patch);evaluations+=1
            if np.linalg.norm(trial-p)<1e-7*(1e-7+np.linalg.norm(p)):
                converged=True
                if newcost<cost:p=trial;cost=newcost;r=newr;J=newJ
                accepted=True;break
            if newcost<cost:
                improvement=cost-newcost;step=np.linalg.norm(trial-p);norm=np.linalg.norm(p)
                p=trial;cost=newcost;r=newr;J=newJ;mu=max(mu*.3,1e-12);accepted=True
                if improvement<1e-7*max(cost,1e-12) or step<1e-7*(1e-7+norm):converged=True
                break
            mu=min(mu*10,1e14)
            if evaluations>=85:break
        if converged or evaluations>=85 or not accepted:break
    return p,math.sqrt(np.mean(r*r)),converged

@njit(cache=True)
def window(img,est,radius,box):
    x0=int(np.rint(est[0]));y0=int(np.rint(est[1]));xy=np.empty(((2*radius+1)**2,2));I=np.empty((2*radius+1)**2);n=0
    for y in range(y0-radius,y0+radius+1):
        for x in range(x0-radius,x0+radius+1):
            if box[0]<=x<box[2] and box[1]<=y<box[3]:xy[n,0]=x;xy[n,1]=y;I[n]=img[y,x];n+=1
    return xy[:n],I[:n]

@njit(parallel=True,cache=True)
def run_corners(img,starts,inverses,patches,boxes):
    n=len(starts);coords=np.empty((n,2));quality=np.empty((n,5));valid=np.zeros(n,np.bool_);flags=np.zeros((n,6),np.bool_)
    for i in prange(n):
        start=starts[i];est=start.copy();p1=np.zeros(6);rmse=0.;ok=False
        for _ in range(3):
            xy,I=window(img,est,4,boxes[i]);p1,rmse,ok=fit_compiled(xy,I,start,inverses[i],patches[i]);moved=np.any(np.rint(p1[:2])!=np.rint(est));est=p1[:2].copy()
            if not moved:break
        xy,I=window(img,est,5,boxes[i]);p2,_,ok2=fit_compiled(xy,I,start,inverses[i],patches[i]);stability=np.linalg.norm(p1[:2]-p2[:2])
        flags[i,0]=not(ok and ok2);flags[i,1]=rmse>15;flags[i,2]=p1[5]<45;flags[i,3]=stability>.65;flags[i,4]=np.max(np.abs(p1[:2]-start))>3.6;flags[i,5]=max(p1[2],p1[3])>2.6
        valid[i]=not np.any(flags[i]);coords[i]=p1[:2];quality[i,0]=rmse;quality[i,1]=stability;quality[i,2]=p1[5];quality[i,3]=p1[2];quality[i,4]=p1[3]
    return coords,quality,valid,flags

class StructuralExtractor:
    def __init__(self,center):
        self.center=center;self.points=REF['points'];self.ids=REF['ids'];self.regions=REF['regions'];self.base=np.array([[c['cx'],c['cy']] for c in CORNERS]);self.inverses=np.array([np.linalg.inv(np.array(c['axes'],np.float32)) for c in CORNERS],np.float64);self.patches=np.array([c['patch'] for c in CORNERS]).reshape(-1,4);self.boxes=np.array([BOXES[k] for k in self.regions],np.int64)
        set_num_threads(4)
        # Compile at process startup, separate from per-image latency.
        starts=self.base[:1].copy();run_corners(center,starts,self.inverses[:1],self.patches[:1],self.boxes[:1])
    def extract(self,img):
        t=time.perf_counter();shifts=[];scores=[]
        for x0,y0,x1,y1 in BOXES:
            pad=130;c=cv2.matchTemplate(img[y0-pad:y1+pad,x0-pad:x1+pad],self.center[y0:y1,x0:x1],cv2.TM_CCOEFF_NORMED);_,score,_,loc=cv2.minMaxLoc(c);shifts.append((loc[0]-pad,loc[1]-pad));scores.append(score)
        if min(scores)<.6:raise ValueError('QR NCC below .6')
        ncc=time.perf_counter()-t;shift=np.array(shifts);starts=self.base+shift[self.regions];boxes=self.boxes+np.c_[shift[self.regions],shift[self.regions]]
        t2=time.perf_counter();coords,q,valid,flags=run_corners(img,starts,self.inverses,self.patches,boxes);fit=time.perf_counter()-t2
        names=['optimizer','photometric_rmse','low_contrast','window_instability','search_limit','blur_limit'];reasons=[[name for j,name in enumerate(names) if flag[j]] for flag in flags]
        return dict(points=self.points,current_points=coords,D=coords-self.points,valid=valid,quality=q,regions=self.regions,ids=self.ids,reasons=reasons,coarse_score=np.array(scores),runtime_seconds=time.perf_counter()-t,timings=dict(ncc=ncc,fit9_and_fit11=fit))
