"""실제 잡음의 공간 구조 (원인 가르기).
(a) 강체 잔차(무작위 부분)의 점 쌍 상관을 가로 거리 / 세로 거리별로: 같은 줄(세로 거리 작음)끼리만 상관 → 롤링 셔터 + 진동, 거리 따라 줄어듦 → 일렁임, 0 → 점마다 독립(센서/매칭)
(b) 실제 센서 잡음: 평평한 벽 영역의 잡음 크기와 스펙트럼 (한 장 = 시간 잡음 + 고정 패턴, 두 장 차이 = 시간 잡음 x √2)"""
import sys; sys.path.insert(0,'../analysis_v7'); sys.path.insert(0,'../analysis_v6')
import numpy as np, pandas as pd, cv2
from common import NAMES, read_raw, VAL, session_of
from lib7 import load, SES
from r1_rigid import session_detrend
D,F=load('../analysis_v7/tr_fs8_hp16_s2.npz'); qr=F.region.str.startswith('QR').values
R=session_detrend(np.load('r1_resid_33px.npy')); P=np.c_[F.x.values[qr],F.y.values[qr]].astype(float)
for c,cn in [(0,'x'),(1,'y')]:
    Z=R[:,:,c]; Z=(Z-Z.mean(0))/Z.std(0); C=(Z.T@Z)/len(Z)
    dx=np.abs(P[:,0][:,None]-P[:,0][None]); dy=np.abs(P[:,1][:,None]-P[:,1][None]); iu=np.triu_indices(len(P),1)
    print(f'[{cn} 잔차] 점 쌍 상관')
    for lo,hi in [(0,5),(5,20),(20,60),(60,250)]:
        for xlo,xhi,xn in [(0,60,'가로 가까움(<60)'),(200,1000,'가로 멂(>200, 다른 QR)')]:
            m=(dy[iu]>=lo)&(dy[iu]<hi)&(dx[iu]>=xlo)&(dx[iu]<xhi)
            if m.sum()>20: print(f'   세로 거리 {lo}~{hi}px, {xn}: 상관 {C[iu][m].mean():+.3f} ({m.sum()}쌍)')
# 줄(행) 단위 공통 흔들림이 있나: 사진마다 같은 y대(10px)에 있는 점들의 평균 잔차가 y대 사이에서 어떻게 변하나 → 행 묶음 분산 vs 독립일 때 기대
yb=(P[:,1]//10).astype(int); u,inv=np.unique(yb,return_inverse=True)
for c,cn in [(0,'x'),(1,'y')]:
    Z=R[:,:,c]; G=np.stack([Z[:,inv==k].mean(1) for k in range(len(u))],1); n=np.bincount(inv)
    obs=(G**2).mean(0); exp_=(Z**2).mean()/n; print(f'{cn}: 같은 줄(10px) 묶음 평균의 분산 / 독립 기대 = {np.median(obs/exp_):.2f} (1이면 줄 공통 흔들림 없음)')
# (b) 센서 잡음: 벽(평평) 블록 찾기
cen=read_raw('center.bmp'); b=cv2.GaussianBlur(cen,(0,0),3)
cands=[]
for y in range(100,2000,64):
    for x in range(200,3600,64):
        blk=cen[y:y+64,x:x+64]; g=b[y:y+64,x:x+64]
        if g.std()<1.0 and blk.max()<250: cands.append((x,y))
print('평평한 블록',len(cands))
def hp(a): return a-cv2.GaussianBlur(a,(0,0),4)
i1,i2=NAMES.index('Y+1.00.bmp'),NAMES.index('Y+1.05.bmp')
A=read_raw(NAMES[i1]); B=read_raw(NAMES[i2])
s1=[];s2=[];ps1=0;ps2=0
for x,y in cands[:200]:
    a=hp(A[y:y+64,x:x+64].astype(float)); bb=hp(B[y:y+64,x:x+64].astype(float))
    s1.append(a[8:-8,8:-8].std()); s2.append((a-bb)[8:-8,8:-8].std()/np.sqrt(2))
    ps1=ps1+np.abs(np.fft.fft2(a))**2; ps2=ps2+np.abs(np.fft.fft2((a-bb)/np.sqrt(2)))**2
print(f'한 장 잡음 {np.median(s1):.3f}, 두 장 차이로 본 시간 잡음 {np.median(s2):.3f}  → 고정 패턴 몫 ≈ {np.sqrt(max(np.median(s1)**2-np.median(s2)**2,0)):.3f}')
f=np.fft.fftfreq(64); fr=np.hypot(f[:,None],f[None]); 
for lo,hi in [(0.02,0.1),(0.1,0.2),(0.2,0.3),(0.3,0.5)]:
    m=(fr>=lo)&(fr<hi); print(f'  주파수 {lo}~{hi}: 잡음 파워(한 장) {ps1[m].mean()/len(cands[:200])/4096:.3f}, (시간) {ps2[m].mean()/len(cands[:200])/4096:.3f}')
np.save('r4_flat.npy',np.array(cands))
