"""v7 공통: 추적 결과 → 낮은 차원 요약, 잡음 바닥 계산."""
import sys; sys.path.insert(0,'../analysis_v6')
import numpy as np, pandas as pd
from common import *
SES=['Y1+','Y1-','Y2+','Y2-','Θ+','Θ-']
F8=pd.read_csv('../analysis_v6/features_qr_hp8.csv')
def load(fn):
    if fn=='v6': Z=np.load('../analysis_v6/tracks_qr_hp8.npz'); return Z['D_gn'].astype(float), F8
    if fn=='v6_hp4': Z=np.load('../analysis_v6/tracks_qr_hp4.npz'); return Z['D_gn'].astype(float), pd.read_csv('../analysis_v6/features_qr_hp4.csv')
    Z=np.load(fn); return Z['D'].astype(float), F8
def summarize(D,F,model='affine'):
    """QR마다 dx, dy를 위치의 다항식으로 맞춘 계수. affine: 1,x,y / quad: +x²,xy,y²"""
    cols=[]; X=[]
    for r,k in [('QR좌','L'),('QR중','C'),('QR우','R')]:
        m=F.region.values==r; x=(F.x.values[m]-F.x.values[m].mean())/100; y=(F.y.values[m]-F.y.values[m].mean())/100
        A=np.c_[np.ones(m.sum()),x,y] if model=='affine' else np.c_[np.ones(m.sum()),x,y,x*x,x*y,y*y]
        tn=['0','x','y','xx','xy','yy'][:A.shape[1]]
        for c,cn in [(0,'dx'),(1,'dy')]:
            co=np.linalg.lstsq(A,D[:,m,c].T,rcond=None)[0]; X+=list(co); cols+=[f'{k}_{cn}_{t}' for t in tn]
    for i,(sx,sy) in enumerate(SCREWS):
        m=(F.region.values=='판나사')&(np.hypot(F.x.values-sx,F.y.values-sy)<60)
        if m.sum(): X+=[D[:,m,0].mean(1),D[:,m,1].mean(1)]; cols+=[f'S{i}_dx',f'S{i}_dy']
    return np.array(X).T, cols
def session_fit(X,names):
    ses=np.array([session_of(n) for n in names]); v=np.array([VAL[n] for n in names])
    R=np.zeros_like(X); ints={}; sl={}
    for s in SES:
        m=ses==s; A=np.c_[np.ones(m.sum()),v[m],v[m]**2]; co=np.linalg.lstsq(A,X[m],rcond=None)[0]; R[m]=X[m]-A@co; ints[s]=co[0]; sl[s]=co[1]
    return R[ses!='C'],ints,sl
def floor(X,names):
    R,ints,sl=session_fit(X,names); Sw=np.cov(R.T); Si=np.linalg.pinv(Sw)
    sY=np.mean([sl[s] for s in SES[:4]],0); sT=np.mean([sl[s] for s in SES[4:]],0)
    a=(sT@Si@sY)/(sT@Si@sT); d=sY-a*sT; n=np.sqrt(d@Si@d)
    u=Si@d/n/n; z={s:float(u@ints[s]) for s in SES}
    return n,3/n,z
