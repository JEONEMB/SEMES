import numpy as np
exec(open('field.py').read().split("if __name__")[0])
bx=['1','u','u2','v','uv']; A=np.stack([BASIS[b] for b in bx],1)
idx=[i for i in range(len(names)) if S[i]==1 and names[i]!='center.bmp']
Rr=[];Ph=[];Ii=[];Kk=[]
for i in idx:
    m=OK[i]; d=Q[i,:,0]-P0[:,0]; c,*_=np.linalg.lstsq(A[m],d[m],rcond=None); r=d-A@c
    Rr.append(np.where(m,r,np.nan)); Ph.append(np.mod(Q[i,:,0],1.0))   # 현재 영상에서의 서브픽셀 위치
Rr=np.array(Rr); Ph=np.array(Ph)
ok=~np.isnan(Rr)
print('residual sd (all)',np.nanstd(Rr).round(4))
# (1) 점마다 고정된 성분 (자세와 무관) — 점별 평균
pm=np.nanmean(Rr,0); print('per-point constant part sd',np.nanstd(pm).round(4),' remaining',np.nanstd(Rr-pm).round(4))
R2=Rr-pm
# (2) 서브픽셀 위치 의존 (전역 pixel-locking)
X=np.c_[np.sin(2*np.pi*Ph[ok]),np.cos(2*np.pi*Ph[ok]),np.sin(4*np.pi*Ph[ok]),np.cos(4*np.pi*Ph[ok])]
c,*_=np.linalg.lstsq(X,R2[ok],rcond=None); print('global pixel-locking harmonics',c.round(4),' explained',(1-np.var(R2[ok]-X@c)/np.var(R2[ok])).round(3))
# (3) 점별 pixel-locking (점마다 다른 곡선) — 학습 절반으로 맞추고 나머지로 검증
tr=np.arange(len(idx))%2==0
expl=[];
pred=np.zeros_like(R2)
for k in range(R2.shape[1]):
    mk=ok[:,k]&tr; mt=ok[:,k]&~tr
    if mk.sum()<20: continue
    Xk=lambda p: np.c_[np.sin(2*np.pi*p),np.cos(2*np.pi*p),np.sin(4*np.pi*p),np.cos(4*np.pi*p)]
    ck,*_=np.linalg.lstsq(Xk(Ph[mk,k]),R2[mk,k],rcond=None); pred[:,k]=Xk(Ph[:,k])@ck
mt=ok&~tr[:,None]
print('per-point pixel-locking (cross-validated) explained',(1-np.var((R2-pred)[mt])/np.var(R2[mt])).round(3))
