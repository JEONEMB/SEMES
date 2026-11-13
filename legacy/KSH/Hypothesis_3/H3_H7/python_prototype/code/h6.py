import numpy as np
exec(open('field.py').read().split("if __name__")[0])
bx,by=['1','u','u2','v','uv'],['1','u','v','uv']
A=np.stack([BASIS[b] for b in bx],1)
Xk=lambda p: np.stack([np.sin(2*np.pi*p),np.cos(2*np.pi*p),np.sin(4*np.pi*p),np.cos(4*np.pi*p)],-1)
# 학습 영상(1차 짝수)의 잔차로 점별 pixel-locking 곡선 학습
Rr=[];Ph=[]
for i in train:
    m=OK[i]; d=Q[i,:,0]-P0[:,0]; c,*_=np.linalg.lstsq(A[m],d[m],rcond=None); Rr.append(np.where(m,d-A@c,np.nan)); Ph.append(np.mod(Q[i,:,0],1))
Rr=np.array(Rr); Ph=np.array(Ph); pm=np.nanmean(Rr,0); R2=Rr-pm
coef=np.zeros((len(P0),4))
for k in range(len(P0)):
    m=~np.isnan(R2[:,k])
    if m.sum()>20: coef[k],*_=np.linalg.lstsq(Xk(Ph[m,k]),R2[m,k],rcond=None)
Q0=Q.copy()
corr=np.einsum('ikj,kj->ik',Xk(np.mod(Q[:,:,0],1)),coef)
for lab,QQ in (('보정 전',Q0),('H6 pixel-locking 보정 후',Q0-np.stack([corr,np.zeros_like(corr)],-1))):
    Q[:]=QQ; K,M,C,E,E2=run(bx,by,True)
    print(lab); print('  ',report(E,'set1 test Y sweep',test1y)); print('  ',report(E,'set1 test Θ sweep',test1t)); print('  ',report(E2,'set2 (ref center1)',s2))
