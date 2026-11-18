"""사진 1장의 이동장을 낮은 차원 값으로 요약: QR마다 아핀 6개 (dx, dy, x크기, y크기, 기울기 2) + 판 나사 2개 이동.
출력: lowdim_hp{hp}.npz  (feats: 사진 x 값, cols)"""
import sys; sys.path.insert(0,'../analysis_v6')
import numpy as np, pandas as pd
from common import *

def lowdim(hp):
    F=pd.read_csv(f'../analysis_v6/features_qr_hp{hp}.csv'); Z=np.load(f'../analysis_v6/tracks_qr_hp{hp}.npz'); D=Z['D_gn'].astype(float); names=list(Z['names'])
    sc=Z['score']; good=sc.min(0)>=0.8
    cols=[]; X=[]
    for r,k in [('QR좌','L'),('QR중','C'),('QR우','R')]:
        m=(F.region.values==r)&good; x=(F.x.values[m]-F.x.values[m].mean())/100; y=(F.y.values[m]-F.y.values[m].mean())/100
        A=np.c_[np.ones(m.sum()),x,y]
        for c,cn in [(0,'dx'),(1,'dy')]:
            co=np.linalg.lstsq(A,D[:,m,c].T,rcond=None)[0]   # 3 x photos
            X+= [co[0],co[1],co[2]]; cols+=[f'{k}_{cn}0',f'{k}_{cn}_x',f'{k}_{cn}_y']
    for i,(sx,sy) in enumerate(SCREWS):
        m=(F.region.values=='판나사')&good&(np.hypot(F.x.values-sx,F.y.values-sy)<60)
        if m.sum()==0: continue
        X+=[D[:,m,0].mean(1),D[:,m,1].mean(1)]; cols+=[f'S{i}_dx',f'S{i}_dy']
    return np.array(X).T, cols, names

if __name__=='__main__':
    for hp in (8,4):
        X,cols,names=lowdim(hp); np.savez(f'lowdim_hp{hp}.npz',X=X,cols=np.array(cols),names=np.array(names)); print(hp,X.shape,cols)
