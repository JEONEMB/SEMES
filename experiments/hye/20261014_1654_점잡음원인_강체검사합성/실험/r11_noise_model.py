"""실제 잡음 모델: 보간 없는 사진 쌍(거의 정수 이동)에서 차이/√2 의 분산을 밝기별, 영역(판/QR vs 벽)별로.
창 안 미세 이동은 국소(9x9) 최소제곱으로 빼서 이동 영향 제거."""
import sys; sys.path.insert(0,'../analysis_v7'); sys.path.insert(0,'../analysis_v6')
import numpy as np, pandas as pd, cv2
from common import NAMES, read_raw, AXIS
from lib7 import load
D,F=load('../analysis_v7/tr_fs8_hp16_s2.npz'); qr=F.region.str.startswith('QR').values; M=D[:,qr].mean(1)
idx=[i for i,n in enumerate(NAMES) if AXIS[n]=='Y']; best=[]
for a in idx:
    for b in idx:
        if a<b: d=M[b]-M[a]; best.append((abs(d[0]-round(d[0]))+abs(d[1]-round(d[1])),a,b,d))
best.sort()
X0,X1,Y0,Y1=1440,2380,960,1180
BR=np.arange(0,256,16); acc=[]
for e,a,b,d in best[:8]:
    ix,iy=int(round(d[0])),int(round(d[1])); A=read_raw(NAMES[a]).astype(float)[Y0:Y1,X0:X1]; B=read_raw(NAMES[b]).astype(float)[Y0+iy:Y1+iy,X0+ix:X1+ix]
    co=np.polyfit(A.ravel(),B.ravel(),1); dd=B-(co[0]*A+co[1])
    # 국소 이동 보정
    g=cv2.GaussianBlur((A+B)/2,(0,0),0.7); gx=np.gradient(g,axis=1); gy=np.gradient(g,axis=0); k=(9,9)
    Sxx=cv2.blur(gx*gx,k); Syy=cv2.blur(gy*gy,k); Sxy=cv2.blur(gx*gy,k); Sxd=cv2.blur(gx*dd,k); Syd=cv2.blur(gy*dd,k); det=Sxx*Syy-Sxy**2+1e-6
    sx=(Syy*Sxd-Sxy*Syd)/det; sy=(Sxx*Syd-Sxy*Sxd)/det; r=dd-gx*sx-gy*sy
    acc.append(pd.DataFrame(dict(br=((A+B)/2).ravel(),r=r.ravel(),g=np.hypot(gx,gy).ravel(),sx=sx.ravel())))
T=pd.concat(acc); T['밝기']=pd.cut(T.br,BR); T['기울기']=pd.cut(T.g,[0,2,8,30,200])
v=T.groupby(['밝기'],observed=True).r.var()/2
print('판 영역: 밝기별 한 장 잡음 sd'); print(np.sqrt(v).round(2).to_string())
print('기울기별 (밝기 무관):'); print(np.sqrt(T.groupby('기울기',observed=True).r.var()/2).round(2).to_string())
print('국소 이동 보정량 sd (px):',T.sx.std().round(4))
# 벽 블록 밝기 확인
cen=read_raw('center.bmp'); fl=np.load('r4_flat.npy'); print('벽 평평 블록 평균 밝기 중앙',np.median([cen[y:y+64,x:x+64].mean() for x,y in fl]).round(1))
np.save('r11_noise_by_brightness.npy',np.c_[ (BR[:-1]+8), [np.sqrt(v.get(iv,np.nan)) for iv in pd.cut(BR[:-1]+8,BR)] ])
