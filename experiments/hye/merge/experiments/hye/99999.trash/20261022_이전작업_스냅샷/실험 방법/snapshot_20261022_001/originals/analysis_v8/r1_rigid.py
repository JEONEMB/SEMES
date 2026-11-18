"""강체 검사 (라벨 없이 점 정밀도 재기): 판은 휘지 않는다 → 판 위 점은 사진마다 하나의 원근 변환(호모그래피)을 따라야 한다.
사진마다 QR 점 232개에 호모그래피를 맞추고 남은 양 = 점 측정 오차 (+ 렌즈 왜곡 등 사진마다 같은 계통 성분).
계통 성분은 점마다 구간 안에서 값 v에 대한 2차로 빼서 '무작위 부분'만 따로 본다."""
import sys; sys.path.insert(0,'../analysis_v7'); sys.path.insert(0,'../analysis_v6')
import numpy as np, pandas as pd, cv2
from scipy.optimize import least_squares
from lib7 import load, SES
from common import NAMES, VAL, session_of
def H_apply(h,P):
    H=np.r_[h,1].reshape(3,3); q=np.c_[P,np.ones(len(P))]@H.T; return q[:,:2]/q[:,2:]
def rigid_resid(D,F):
    qr=F.region.str.startswith('QR').values; P=np.c_[F.x.values[qr],F.y.values[qr]].astype(float); c=P.mean(0); Pn=(P-c)/500
    R=np.zeros((len(D),qr.sum(),2))
    for i in range(len(D)):
        Q=(P+D[i,qr]-c)/500; H0,_=cv2.findHomography(Pn,Q,0); h0=(H0/H0[2,2]).ravel()[:8]
        r=least_squares(lambda h:(H_apply(h,Pn)-Q).ravel(),h0,method='lm'); R[i]=(H_apply(r.x,Pn)-Q)*500
    return R
def session_detrend(R):
    ses=np.array([session_of(n) for n in NAMES]); v=np.array([VAL[n] for n in NAMES]); out=np.full_like(R,np.nan)
    for s in SES:
        m=ses==s; A=np.c_[np.ones(m.sum()),v[m],v[m]**2]; X=R[m].reshape(m.sum(),-1); out[m]=(X-A@np.linalg.lstsq(A,X,rcond=None)[0]).reshape(R[m].shape)
    return out[ses!='C']
if __name__=='__main__':
    rows=[]
    for fn,lab in [('v6','17px σ1'),('../analysis_v7/tr_fs8_hp16_s2.npz','33px σ2')]:
        D,F=load(fn); R=rigid_resid(D,F); Rr=session_detrend(R)
        np.save(f'r1_resid_{lab.split()[0]}.npy',R)
        for nm,X in [('강체 잔차 그대로',R[1:] if False else np.delete(R,NAMES.index('center.bmp'),0)),('강체 잔차 중 무작위 부분',Rr)]:
            sx,sy=np.median(X[:,:,0].std(0)),np.median(X[:,:,1].std(0))
            rows.append(dict(측정=lab,구분=nm,점흔들림_x=sx,점흔들림_y=sy)); print(rows[-1])
        # 계통 부분 크기: 점마다 평균 잔차
        mean_r=np.delete(R,NAMES.index('center.bmp'),0).mean(0); print('  계통(점마다 평균 잔차) RMS',np.sqrt((mean_r**2).mean(0)).round(4))
    pd.DataFrame(rows).round(4).to_csv('r1_rigid.csv',index=False,encoding='utf-8-sig')
