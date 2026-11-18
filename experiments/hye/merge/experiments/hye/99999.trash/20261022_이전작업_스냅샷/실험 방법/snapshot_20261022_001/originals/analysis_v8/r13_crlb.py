"""이론 하한(크라메르-라오): 점 창 안 픽셀마다 잡음 sd(밝기별 실측), 기울기 g.
같은 무게 추정 분산 = Σg²σ²/(Σg²)², 최적(잡음 가중) = 1/Σ(g²/σ²). 흐림 없는 원본 기준 (상대 비교용)."""
import sys; sys.path.insert(0,'../analysis_v6')
import numpy as np, pandas as pd, cv2
from common import read_raw
F=pd.read_csv('../analysis_v6/features_qr_hp8.csv'); F=F[F.region.str.startswith('QR')]
NB=np.load('r11_noise_by_brightness.npy'); NB=NB[~np.isnan(NB[:,1])]
cen=read_raw('center.bmp').astype(float); s=np.interp(cen,NB[:,0],NB[:,1])
gx=np.gradient(cen,axis=1); gy=np.gradient(cen,axis=0)
for hp in (8,16,24):
    u=[];o=[]
    for x,y in zip(F.x,F.y):
        sl=(slice(y-hp,y+hp+1),slice(x-hp,x+hp+1)); g=gx[sl]; v=s[sl]**2
        u.append(np.sqrt((g*g*v).sum()/(g*g).sum()**2)); o.append(np.sqrt(1/(g*g/v).sum()))
    print(f'창 {2*hp+1}px: 같은 무게 {np.median(u):.4f}px, 잡음 가중(최적) {np.median(o):.4f}px → {np.median(np.array(u)/np.array(o)):.2f}배 개선 가능')
