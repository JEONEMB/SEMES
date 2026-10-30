"""실제 사진마다 center 대비 흐림 차이 측정 (위치와 무관한 스펙트럼 크기 비교).
판 영역(내용을 정수 이동만큼 따라감) x 한 창 → |F|² 비 = exp(-4π² Δσ² f²) → 가로(fx), 세로(fy) 방향 따로 Δσ² 추정.
Δσ²>0 : 현재 사진이 center보다 흐림."""
import sys; sys.path.insert(0,'../analysis_v7'); sys.path.insert(0,'../analysis_v6')
import numpy as np, pandas as pd
from multiprocessing import Pool
from common import NAMES, read_raw, VAL, AXIS
from lib7 import load
D,F=load('v6'); qr=F.region.str.startswith('QR').values; MD=np.round(D[:,qr].mean(1)).astype(int)
X0,X1,Y0,Y1=1440,2368,960,1184; W=np.outer(np.hanning(Y1-Y0),np.hanning(X1-X0))
fy=np.fft.fftfreq(Y1-Y0)[:,None]; fx=np.fft.fftfreq(X1-X0)[None,:]
def spec(n):
    i=NAMES.index(n); dx,dy=MD[i]; im=read_raw(n)[Y0+dy:Y1+dy,X0+dx:X1+dx].astype(float); im=(im-im.mean())*W
    return np.abs(np.fft.fft2(im))**2
BX=(np.abs(fy)<0.02)&(np.abs(fx)>0.06)&(np.abs(fx)<0.3); BY=(np.abs(fx)<0.02)&(np.abs(fy)>0.06)&(np.abs(fy)<0.3)
def fit(P,P0,B,f):
    r=np.log(P[B]/P0[B]); f2=(f**2)[B] if f.ndim==2 else None
    A=np.c_[np.ones(B.sum()),-4*np.pi**2*f2]; c=np.linalg.lstsq(A,r,rcond=None)[0]; return c[1],c[0]
def job(n):
    P=spec(n); return P
if __name__=='__main__':
    with Pool(16) as p: S=p.map(spec,NAMES)
    P0=S[NAMES.index('center.bmp')]; FX=np.broadcast_to(fx,P0.shape); FY=np.broadcast_to(fy,P0.shape)
    rows=[]
    for n,P in zip(NAMES,S):
        sx,gx=fit(P,P0,BX,FX); sy,gy=fit(P,P0,BY,FY)
        rows.append(dict(file=n,axis=AXIS[n],value=VAL[n],dsig2_x=sx,dsig2_y=sy,gain_x=gx,gain_y=gy))
    T=pd.DataFrame(rows); T.to_csv('r2_blur.csv',index=False,encoding='utf-8-sig')
    t=T[T.file!='center.bmp']
    print('Δσ² 가로: 중앙 %.4f, 5~95%% %.4f ~ %.4f'%(t.dsig2_x.median(),*t.dsig2_x.quantile([.05,.95])))
    print('Δσ² 세로: 중앙 %.4f, 5~95%% %.4f ~ %.4f'%(t.dsig2_y.median(),*t.dsig2_y.quantile([.05,.95])))
    print('사진끼리 흐림 차이 표준편차: 가로 %.4f, 세로 %.4f (px²), 가로-세로 상관 %.2f'%(t.dsig2_x.std(),t.dsig2_y.std(),np.corrcoef(t.dsig2_x,t.dsig2_y)[0,1]))
    # 검증: center를 알려진 σ로 흐리면 Δσ²가 맞게 나오나
    import cv2
    im=read_raw('center.bmp')
    for s in [0.2,0.3,0.5]:
        b=cv2.GaussianBlur(im,(0,0),s)[Y0:Y1,X0:X1].astype(float); b=(b-b.mean())*W; Pb=np.abs(np.fft.fft2(b))**2
        print(f' 검증 σ={s}: 참 Δσ²={s*s:.3f} → 추정 가로 {fit(Pb,P0,BX,FX)[0]:.3f}, 세로 {fit(Pb,P0,BY,FY)[0]:.3f}')
