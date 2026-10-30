import sys; sys.path.insert(0,'../analysis_v6')
import numpy as np, pandas as pd
from common import *
SES=['Y1+','Y1-','Y2+','Y2-','Θ+','Θ-']
P=pd.read_csv('../analysis_v6/t5_per_image.csv').set_index('file')
Z=np.load('lowdim_hp8.npz'); X=Z['X'][:,:18]; cols=list(Z['cols'])[:18]; names=list(Z['names'])
ses=np.array([session_of(n) for n in names]); v=np.array([VAL[n] for n in names]); ok=ses!='C'
R=np.zeros_like(X)
for s in SES:
    m=ses==s; A=np.c_[np.ones(m.sum()),v[m],v[m]**2]; R[m]=X[m]-A@np.linalg.lstsq(A,X[m],rcond=None)[0]
sh=P.loc[names,'sharp'].values; br=P.loc[names,'bright'].values
Rk=R[ok]; print('성분별 잔차 sd (px):'); print(pd.Series(Rk.std(0),cols).round(4).to_string())
for nm,z in [('선명도',sh),('밝기',br)]:
    zz=z[ok]-z[ok].mean(); c=[np.corrcoef(zz,Rk[:,j])[0,1] for j in range(18)]
    print(nm,'상관 최대', np.round(sorted(c,key=abs)[-5:],2))
u,s,vt=np.linalg.svd(Rk/Rk.std(0),full_matrices=False); print('잔차 PCA 분산비',np.round(s**2/np.sum(s**2),3)[:6])
# 시간 이웃 상관 (구간 안 순서)
for s_ in SES:
    m=np.where(ses==s_)[0]; o=m[np.argsort(np.abs(v[m]))]; r=R[o]
    print(s_,'이웃 상관 중앙', np.round(np.median([np.corrcoef(r[:-1,j],r[1:,j])[0,1] for j in range(18)]),2))
