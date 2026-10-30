"""카메라 내부 처리 흔적 찾기.
(1) 블록 압축(JPEG/H.264): 사진 쌍 차이(잡음+처리 오차)의 가로 이웃 차이가 x mod 8 / 16 위치에 따라 다른가
(2) 원본 영상 8x8 DCT 계수 분포의 양자화 주기
(3) 디모자이크: 평평한 곳 잡음의 이웃 상관 (lag 1, 2), 짝/홀 픽셀 위치별 잡음 크기
(4) 원본 픽셀 값: R=G=B 인지, 값 히스토그램에 빈 값(감마/스케일링 흔적)"""
import sys; sys.path.insert(0,'../analysis_v6')
import numpy as np, cv2
from common import NAMES, read_raw, path_of
A=read_raw('Y-0.20.bmp'); B=read_raw('Y+0.40.bmp')
ix,iy=9,0   # r6에서 거의 정수 이동 쌍 (확인용으로 다시 계산)
import pandas as pd
d=pd.read_csv('r6_pairdiff.csv')
print(d.iloc[0].to_dict())
X0,X1,Y0,Y1=1300,2520,700,1300
# 정수 이동 다시 구하기: 판 영역 위상상관
a=A[950:1190,1430:2390]; 
best=None
for dx in range(-20,21):
    b=B[950:1190,1430+dx:2390+dx]; e=np.abs(b-a).mean()
    if best is None or e<best[0]: best=(e,dx)
dx=best[1]; print('정수 이동',dx)
diff=B[Y0:Y1,X0+dx:X1+dx].astype(float)-A[Y0:Y1,X0:X1].astype(float)
hp=diff-cv2.GaussianBlur(diff,(0,0),4)
gx=np.abs(np.diff(hp,axis=1)); gyy=np.abs(np.diff(hp,axis=0))
for P in (8,16):
    colx=(np.arange(X0,X1-1)+dx)%P; coly=np.arange(Y0,Y1-1)%P
    print(f'[{P}블록] 가로 이웃차 평균: 경계(x%{P}=={P-1}) {gx[:,colx==P-1].mean():.3f} vs 나머지 {gx[:,colx!=P-1].mean():.3f};  세로 경계 {gyy[coly==P-1].mean():.3f} vs {gyy[coly!=P-1].mean():.3f}')
# (2) DCT 양자화: center 판 영역 8x8 블록 (영상 격자 기준)
im=read_raw('center.bmp')[952:1192,1432:2392].astype(float)-128
co=[]
for y in range(0,im.shape[0]-7,8):
    for x in range(0,im.shape[1]-7,8): co.append(cv2.dct(im[y:y+8,x:x+8].astype(np.float32)))
co=np.array(co)
for (u,v) in [(0,1),(1,1),(2,3),(4,4)]:
    c=co[:,u,v]; frac=np.abs(c-np.round(c)); print(f' DCT({u},{v}) 값의 정수 근처 비율 {np.mean(frac<0.1):.2f} (양자화 없으면 ≈0.2)', ' 고유값 수',len(np.unique(np.round(c,2))))
# (3) 디모자이크: 평평한 곳 (벽) 잡음 이웃 상관
cen=read_raw('center.bmp').astype(float); flat=np.load('r4_flat.npy')
c1=[];c2=[];eo=[]
for x,y in flat[:150]:
    a=A[y:y+64,x:x+64].astype(float); b=B[y:y+64,x:x+64].astype(float); n=(a-b); n=n-cv2.GaussianBlur(n,(0,0),4)
    c1.append(np.corrcoef(n[:,:-1].ravel(),n[:,1:].ravel())[0,1]); c2.append(np.corrcoef(n[:,:-2].ravel(),n[:,2:].ravel())[0,1])
    eo.append([n[0::2,0::2].std(),n[0::2,1::2].std(),n[1::2,0::2].std(),n[1::2,1::2].std()])
print(f'(3) 벽 잡음 이웃 상관 lag1 {np.median(c1):+.3f}, lag2 {np.median(c2):+.3f} (백색이면 0, 고역통과 탓에 약간 음수); 2x2 위치별 크기 {np.median(eo,0).round(3)}')
# (4) 원본 채널
raw=cv2.imdecode(np.fromfile(str(path_of('center.bmp')),np.uint8),cv2.IMREAD_UNCHANGED)
print('(4) 채널 같음:',bool((raw[...,0]==raw[...,1]).all() and (raw[...,1]==raw[...,2]).all()))
h=np.bincount(raw[...,0].ravel(),minlength=256); print('    0~255 중 안 쓰인 값 (100~240 사이):',[i for i in range(100,241) if h[i]==0][:20])
