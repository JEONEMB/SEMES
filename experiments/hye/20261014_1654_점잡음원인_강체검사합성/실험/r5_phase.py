"""픽셀 위상(앨리어싱) 시험: 점마다 강체 잔차가 그 점 이동의 소수점 부분(φx, φy)의 주기 함수인가.
점마다 sin/cos(2πkφ) (k=1,2, x와 y, 교차항) 회귀 → 교차검증(짝/홀 사진)으로 설명 비율."""
import sys; sys.path.insert(0,'../analysis_v7'); sys.path.insert(0,'../analysis_v6')
import numpy as np, pandas as pd
from common import NAMES
from lib7 import load
from r1_rigid import session_detrend
ok=np.array([n!='center.bmp' for n in NAMES])
for fn,lab in [('v6','17px'),('../analysis_v7/tr_fs8_hp16_s2.npz','33px')]:
    D,F=load(fn); qr=F.region.str.startswith('QR').values; Dq=D[:,qr][ok]
    R=session_detrend(np.load(f'r1_resid_{lab}.npy'))
    px=(F.x.values[qr]+Dq[:,:,0]); py=(F.y.values[qr]+Dq[:,:,1])     # 현재 사진에서의 위치 → 위상
    fx=px%1; fy=py%1
    def basis(a,b,K):
        cols=[np.ones_like(a)]
        for k in range(1,K+1): cols+=[np.sin(2*np.pi*k*a),np.cos(2*np.pi*k*a),np.sin(2*np.pi*k*b),np.cos(2*np.pi*k*b)]
        if K>=1: cols+=[np.sin(2*np.pi*(a+b)),np.cos(2*np.pi*(a+b)),np.sin(2*np.pi*(a-b)),np.cos(2*np.pi*(a-b))]
        return np.stack(cols,-1)
    rng=np.random.default_rng(0); fold=rng.integers(0,2,len(R))
    for K in (1,2):
        ex=[]
        for c in (0,1):
            num=den=0
            for j in range(R.shape[1]):
                B=basis(fx[:,j],fy[:,j],K); y=R[:,j,c]
                for k in (0,1):
                    tr,te=fold!=k,fold==k; co=np.linalg.lstsq(B[tr],y[tr],rcond=None)[0]; num+=((y[te]-B[te]@co)**2).sum(); den+=((y[te]-y[tr].mean())**2).sum()
            ex.append(1-num/den)
        print(f'{lab} 차수 {K}: 위상으로 설명되는 비율 (교차검증) x {ex[0]*100:.1f}%, y {ex[1]*100:.1f}%')
