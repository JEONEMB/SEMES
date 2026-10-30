"""보간 없는 쌍 차이, 밝기 맞춤 버전. 차이에서 (1) 전체 밝기 a*A+b, (2) 느린 조명 변화(고역 통과 σ8)를 뺀 뒤 영역별 크기.
영역: 벽(판 밖, 센서 잡음 기준), 판 어두운 평평, QR 흰 평평, QR 경계."""
import sys; sys.path.insert(0,'../analysis_v7'); sys.path.insert(0,'../analysis_v6')
import numpy as np, pandas as pd, cv2
from common import NAMES, read_raw, AXIS
from lib7 import load
D,F=load('../analysis_v7/tr_fs8_hp16_s2.npz'); qr=F.region.str.startswith('QR').values; M=D[:,qr].mean(1)
idx=[i for i,n in enumerate(NAMES) if AXIS[n]=='Y']; best=[]
for a in idx:
    for b in idx:
        if a<b:
            d=M[b]-M[a]; best.append((abs(d[0]-round(d[0]))+abs(d[1]-round(d[1])),a,b,d))
best.sort()
X0,X1,Y0,Y1=1300,2520,700,1260
rows=[]
for e,a,b,d in best[:6]:
    ix,iy=int(round(d[0])),int(round(d[1])); A=read_raw(NAMES[a]).astype(float)[Y0:Y1,X0:X1]; B=read_raw(NAMES[b]).astype(float)[Y0+iy:Y1+iy,X0+ix:X1+ix]
    yy,xx=np.mgrid[Y0:Y1,X0:X1]; plate=(xx>1440)&(xx<2380)&(yy>960)&(yy<1180)
    co=np.polyfit(A[plate],B[plate],1); diff=B-(co[0]*A+co[1])
    hp=diff-cv2.GaussianBlur(diff,(0,0),8)
    gm=cv2.GaussianBlur(A,(0,0),1.0); g=np.hypot(cv2.Sobel(gm,cv2.CV_64F,1,0),cv2.Sobel(gm,cv2.CV_64F,0,1))/8
    m={'벽(판 밖 위쪽)':(yy<900)&(g<1.0)&(A<245),'판 어두운 평평':plate&(g<1.0)&(A<150),'QR 흰 평평':plate&(g<1.0)&(A>=150),'QR 경계(기울기 상위)':plate&(g>np.quantile(g[plate],0.85))}
    r=dict(쌍=f'{NAMES[a]}-{NAMES[b]}',밝기비=co[0]); r.update({k:hp[v].std()/np.sqrt(2) for k,v in m.items()}); rows.append(r)
T=pd.DataFrame(rows); print(T.round(3).to_string()); print('한 장 기준 (차이/√2). 센서 잡음만이면 모든 영역 ≈ 벽 값')
T.round(4).to_csv('r8_pairdiff2.csv',index=False,encoding='utf-8-sig')
