import sys; sys.path.insert(0,'../analysis_v6')
import numpy as np, pandas as pd
from common import *
F=pd.read_csv('../analysis_v6/features_qr_hp8.csv'); Z=np.load('../analysis_v6/tracks_qr_hp8.npz'); D=Z['D_gn']; names=list(Z['names'])
ax=np.array([AXIS[n] for n in names]); v=np.array([VAL[n] for n in names]); ses=np.array([session_of(n) for n in names])
print(F.region.value_counts())
reg=F.region.values
m={r:reg==r for r in ['QR좌','QR중','QR우']}
L,C,R=[D[:,m[r],0].mean(1) for r in ['QR좌','QR중','QR우']]
Ly,Cy,Ry=[D[:,m[r],1].mean(1) for r in ['QR좌','QR중','QR우']]
mean=(L+C+R)/3; curv=C-(L+R)/2; tilt=R-L; dyt=Ry-Ly
df=pd.DataFrame(dict(n=names,ax=ax,v=v,ses=ses,mean=mean,curv=curv,tilt=tilt,dyt=dyt))
for a in ['Y','Θ']:
    d=df[df.ax==a]
    for f in ['curv','tilt','dyt']:
        A=np.c_[d['mean'],d['mean']**2,np.ones(len(d))]; co,*_=np.linalg.lstsq(A,d[f],rcond=None); r=d[f]-A@co
        print(a,f,'slope per px %.5f quad %.2e int %.4f resid sd %.4f'%(co[0],co[1],co[2],r.std()))
print(df[df.ax=='C'])
# per-session intercept of curv
for s in ['Y1+','Y1-','Y2+','Y2-','Θ+','Θ-']:
    d=df[df.ses==s]; A=np.c_[d['mean'],np.ones(len(d))]; co,*_=np.linalg.lstsq(A,d['curv'],rcond=None); print(s,'curv slope %.5f int %.4f'%tuple(co), ' mean int', np.polyfit(d.v,d['mean'],1)[1].round(3))
df.to_csv('e0_probe.csv',index=False)
