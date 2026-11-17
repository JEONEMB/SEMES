"""신호 대비 잡음 (낮은 차원). 성분마다 구간(6개)별로 v에 대해 2차 맞춤 → 구간 안 잔차 공분산 Σw.
구간 절편 차이 → 0점 공분산 Σz. Y 1mm 무늬와 같은 평균 이동의 Θ 무늬 차이 Δ → 마할라노비스 거리/mm.
(평균 이동 방향은 값 추정에 쓰이므로, Δ에서 무늬 차이만 남도록 Θ 크기를 최적으로 맞춤)"""
import sys; sys.path.insert(0,'../analysis_v6')
import numpy as np, pandas as pd
from common import *
SES=['Y1+','Y1-','Y2+','Y2-','Θ+','Θ-']
def analyze(hp, sel):
    Z=np.load(f'lowdim_hp{hp}.npz'); X=Z['X']; cols=list(Z['cols']); names=list(Z['names'])
    k=[i for i,c in enumerate(cols) if sel(c)]; X=X[:,k]; cols=[cols[i] for i in k]
    ses=np.array([session_of(n) for n in names]); v=np.array([VAL[n] for n in names])
    R=np.zeros_like(X); ints={}; slopes={}
    for s in SES:
        m=ses==s; A=np.c_[np.ones(m.sum()),v[m],v[m]**2]; co=np.linalg.lstsq(A,X[m],rcond=None)[0]; R[m]=X[m]-A@co; ints[s]=co[0]; slopes[s]=co[1]
    ok=ses!='C'; Sw=np.cov(R[ok].T)
    sY=(slopes['Y1+']+slopes['Y1-']+slopes['Y2+']+slopes['Y2-'])/4; sT=(slopes['Θ+']+slopes['Θ-'])/2
    I=np.array([ints[s] for s in SES]); Sz=np.cov(I.T)
    out={}
    for nm,S in [('구간안 잡음',Sw),('구간안+0점',Sw+Sz)]:
        Si=np.linalg.pinv(S); a=(sT@Si@sY)/(sT@Si@sT); d=sY-a*sT; n=np.sqrt(d@Si@d); out[nm]=(n,3/n,a)
    # 0점 차이를 무늬로: center 대비 각 구간 절편의 '무늬 거리' (Y mm 환산)
    Si=np.linalg.pinv(Sw); a=(sT@Si@sY)/(sT@Si@sT); d=sY-a*sT; u=Si@d/np.sqrt(d@Si@d)
    zshift={s:float(u@ints[s]/np.sqrt(d@Si@d)) for s in SES}   # 판정 축 방향 투영 (mm 환산)
    return out, zshift, len(cols)
for hp in (8,4):
    for setn,sel in [('QR만',lambda c:c[0] in 'LCR'),('QR+나사',lambda c:True),('QR만 이동0만',lambda c:c[0] in 'LCR' and c.endswith('0'))]:
        out,z,nc=analyze(hp,sel)
        print(f'{2*hp+1}px {setn} ({nc}값):', {k:f'{v[0]:.1f}σ/mm → 3σ 최소 {v[1]:.3f}mm' for k,v in out.items()})
        print('   구간 0점의 판정방향 치우침 (Y mm 환산, center=0):', {k:round(v,3) for k,v in z.items()})
