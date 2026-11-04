# H3 (선형화 물리 모델) : 영상마다 변위장을 물리 기저로 분해 → 계수로 (Y, Θ) GLS 추정
import numpy as np, json, re, sys
D=np.load('corr.npz',allow_pickle=True); P0,Q,OK,names,info=D['P0'],D['Q'],D['OK'],list(D['names']),D['info']
S,Yl,Tl,SH=info[:,0].astype(int),info[:,1],info[:,2],info[:,3].astype(int)
CX,CY=1912.0,1080.0
u=(P0[:,0]-CX)/1000; v=(P0[:,1]-CY)/1000
BASIS={'1':np.ones_like(u),'u':u,'u2':u*u,'v':v,'uv':u*v,'v2':v*v,'u3':u**3}
def coeffs(i,bx,by,robust=True):
    m=OK[i].copy(); out=[]
    for comp,bl in ((0,bx),(1,by)):
        if not bl: continue
        A=np.stack([BASIS[b] for b in bl],1); d=Q[i,:,comp]-P0[:,comp]; w=m.astype(float)
        for _ in range(3 if robust else 1):
            c,*_=np.linalg.lstsq(A[w>0]*1,d[w>0],rcond=None); r=d-A@c
            s=1.4826*np.median(np.abs(r[m]))+1e-9; w=(m&(np.abs(r)<3*s)).astype(float)
        out.append(c)
    return np.concatenate(out)
# 학습/검증 분할 (1차 세트 값 순서 짝수 = 학습)
idx1=[i for i in range(len(names)) if S[i]==1 and names[i]!='center.bmp']
ysw=sorted([i for i in idx1 if Tl[i]==0],key=lambda i:Yl[i]); tsw=sorted([i for i in idx1 if Yl[i]==0],key=lambda i:Tl[i])
train=ysw[::2]+tsw[::2]; test1y=ysw[1::2]; test1t=tsw[1::2]
s2=[i for i in range(len(names)) if S[i]==2 and not names[i].startswith('center')]
c1=names.index('center1.bmp')
def run(bx,by,quad=True,label=''):
    K=np.array([coeffs(i,bx,by) for i in range(len(names))])
    A=lambda y,t: np.c_[y,t,t*t,np.ones_like(t)] if quad else np.c_[y,t,np.ones_like(t)]
    M,*_=np.linalg.lstsq(A(Yl[train],Tl[train]),K[train],rcond=None)
    R=K[train]-A(Yl[train],Tl[train])@M; C=np.cov(R.T)+np.eye(K.shape[1])*1e-10; W=np.linalg.inv(C)
    def est(k):
        y,t=0.,0.
        for _ in range(10):
            J=np.stack([M[0], M[1]+(2*M[2]*t if quad else 0)],1)
            pred=y*M[0]+t*M[1]+(t*t*M[2] if quad else 0)+M[-1]
            d=np.linalg.solve(J.T@W@J,J.T@W@(k-pred)); y+=d[0]; t+=d[1]
        return y,t
    E=np.array([est(K[i]) for i in range(len(names))])
    E2=np.array([est(K[i]-K[c1]+M[-1]) for i in range(len(names))])   # 2차 세트 : center1 로 기준 맞춤 (계수 공간)
    return K,M,C,E,E2
def report(E,label,idx,ref=None,truthT=None,truthY=None):
    e=E[idx].copy()
    if ref is not None: e=e-E[ref]  # 같은 날 기준 영상으로 다시 맞춤
    tT=Tl[idx] if truthT is None else truthT; tY=Yl[idx] if truthY is None else truthY
    dT=e[:,1]-tT; dY=(e[:,0]-tY)*1000
    return f"{label:22s} n={len(idx):3d} | Θ RMS {np.sqrt((dT**2).mean()):.4f}° ±0.01 {np.sum(np.abs(dT)<=0.01):3d} | Y RMS {np.sqrt((dY**2).mean()):6.1f}µm ±90 {np.sum(np.abs(dY)<=90):3d}"
if __name__=='__main__':
    for bx,by,q in ((['1','u','u2'],[],True),(['1','u','u2','v','uv'],['1','u','v','uv'],True),(['1','u','u2','v','uv','v2'],['1','u','u2','v','uv','v2'],True),(['1','u','u2','v','uv','v2','u3'],['1','u','u2','v','uv','v2'],True)):
        K,M,C,E,E2=run(bx,by,q)
        print('basis dx',bx,'dy',by,'quad',q)
        print('  ',report(E,'set1 test Y sweep',test1y)); print('  ',report(E,'set1 test Θ sweep',test1t))
        # set2: center1 로 기준 맞춤 → 학습 원점과 연결 : e - E[c1]
        print('  ',report(E2,'set2 (ref center1)',s2))
