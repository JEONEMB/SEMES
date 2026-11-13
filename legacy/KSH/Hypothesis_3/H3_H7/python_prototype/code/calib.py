import numpy as np, json, sys, time
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from legacy.KSH.Hypothesis_3.H3_H7.python_prototype.code.phys import *
D=np.load('corr.npz',allow_pickle=True); P0,Q,OK,names,info=D['P0'],D['Q'],D['OK'],list(D['names']),D['info']
S,Yl,Tl,SH=info[:,0].astype(int),info[:,1],info[:,2],info[:,3].astype(int)
# 1차 세트 학습/검증 분할 (값 순서 짝수 = 학습)
idx1=[i for i in range(len(names)) if S[i]==1 and names[i]!='center.bmp']
ysw=sorted([i for i in idx1 if Tl[i]==0],key=lambda i:Yl[i]); tsw=sorted([i for i in idx1 if Yl[i]==0],key=lambda i:Tl[i])
train=ysw[::2]+tsw[::2]; test=ysw[1::2]+tsw[1::2]
ORDER=sys.argv[1] if len(sys.argv)>1 else 'TonY'
import os
FREE=os.environ.get('FREE','f,cx,k1,na,nb,e1,e2,w1,w2,ax,az,bY,bT,Y0,T0').split(',')   # d 고정 (f/d 만 결정됨)
scale=dict(Y0=0.01,T0=0.005,bY=0.01,bT=0.005,f=50,cx=50,cy=50,k1=0.1,na=0.05,nb=0.05,e1=0.02,e2=0.02,w1=0.02,w2=0.02,ax=50,az=50)
tr=list(train)   # center.bmp 는 기준 영상(자세가 라벨 원점과 다름) → 학습 제외, 원점 차이는 Y0·T0 로 추정
def unpack(p):
    g=dict(G0); g.update(bY=0.0,bT=0.0,Y0=0.0,T0=0.0)
    for k,v in zip(FREE,p[:len(FREE)]): g[k]=v
    return g,p[len(FREE):]
def resid(p):
    g,nus=unpack(p); X=plate_points(P0,g); out=[]
    for j,i in enumerate(tr):
        pr=project(X,Yl[i]+g['Y0']+g['bY']*np.sign(Yl[i]),Tl[i]+g['T0']+g['bT']*np.sign(Tl[i]),nus[j],g,ORDER); r=(pr-Q[i]); r[~OK[i]]=0; out.append(r.ravel())
    return np.concatenate(out)
npt=len(P0); nG=len(FREE)
sp=lil_matrix((len(tr)*npt*2,nG+len(tr)),dtype=int); sp[:,:nG]=1
for j in range(len(tr)): sp[j*npt*2:(j+1)*npt*2,nG+j]=1
init=dict(G0); init.update(Y0=0.0,T0=0.0,bY=0.0,bT=0.0,cx=1912.,k1=-0.03)
p0=np.r_[[init[k] for k in FREE],np.zeros(len(tr))]
t0=time.time()
res=least_squares(resid,p0,jac_sparsity=sp,loss='soft_l1',f_scale=0.1,x_scale=np.r_[[scale[k] for k in FREE],np.ones(len(tr))],max_nfev=200,verbose=0)
g,nus=unpack(res.x)
r=res.fun.reshape(len(tr),npt,2)
print(f'order {ORDER}: cost {res.cost:.1f}  {time.time()-t0:.0f}s  resid RMS x {np.sqrt((r[...,0]**2).mean()):.4f} y {np.sqrt((r[...,1]**2).mean()):.4f} px')
print({k:round(float(g[k]),5) for k in GN+['bY','bT','Y0','T0']})
tag=os.environ.get('TAG','')
json.dump({'order':ORDER,'g':{k:float(g[k]) for k in GN+['bY','bT','Y0','T0']},'train':[names[i] for i in train]},open(f'calib_{ORDER}{tag}.json','w'),ensure_ascii=False,indent=1)
