# 물리 모델 : 카메라(초점 f, 주점 cx·cy, 방사 왜곡 k1) + 플레이트 평면(법선·거리) + Y 이동 방향 + Θ 회전축(방향·위치)
import numpy as np
GN=['f','cx','cy','k1','na','nb','d','e1','e2','w1','w2','ax','az']
G0=dict(f=4584.,cx=1920.,cy=1080.,k1=0.,na=0.,nb=0.,d=323.,e1=0.,e2=0.,w1=0.,w2=0.,ax=0.,az=0.)
def rot(axis,th):
    axis=axis/np.linalg.norm(axis); K=np.array([[0,-axis[2],axis[1]],[axis[2],0,-axis[0]],[-axis[1],axis[0],0]])
    return np.eye(3)+np.sin(th)*K+(1-np.cos(th))*K@K
def unit(a,b,base):   # 작은 각 a,b 로 기울인 단위벡터
    if base=='x': v=np.array([-np.cos(b)*np.cos(a),np.sin(b),np.cos(b)*np.sin(a)])   # 기본 −x (Y+ 이면 영상이 +x 로 이동)
    elif base=='y': v=np.array([np.sin(a),-np.cos(a)*np.cos(b),np.cos(a)*np.sin(b)])   # 기본 −y (위쪽, Θ+ 이면 영상 +x)
    else: v=np.array([np.sin(a)*np.cos(b),np.sin(b),np.cos(a)*np.cos(b)])
    return v
def undist(u,v,g,it=5):
    x=(u-g['cx'])/g['f']; y=(v-g['cy'])/g['f']; xd,yd=x.copy(),y.copy()
    for _ in range(it):
        r2=x*x+y*y; s=1+g['k1']*r2; x=xd/s; y=yd/s
    return x,y
def plate_points(P0,g):
    x,y=undist(P0[:,0],P0[:,1],g); r=np.stack([x,y,np.ones_like(x)],1)
    n=unit(g['na'],g['nb'],'z'); lam=g['d']/(r@n); return r*lam[:,None]
def project(X,Y,T,nu,g,order='TonY'):
    th=np.deg2rad(T); w=unit(g['w1'],g['w2'],'y'); R=rot(w,th); e=unit(g['e1'],g['e2'],'x'); A=np.array([g['ax'],0.,g['az']])
    if order=='TonY': c=Y*e+(np.eye(3)-R)@A
    else: c=(np.eye(3)-R)@A+R@(Y*e)
    Xc=(X-c)@R            # = R^T (X-c) 행벡터
    x=Xc[:,0]/Xc[:,2]; y=Xc[:,1]/Xc[:,2]; r2=x*x+y*y; s=1+g['k1']*r2
    return np.stack([g['f']*x*s+g['cx'], g['f']*y*s+g['cy']+nu],1)
