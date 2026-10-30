"""보간 없는 사진 쌍 차이: 두 사진의 판 이동 차이가 정수 픽셀에 가까운 쌍을 골라 정수만큼 옮겨 바로 뺀다.
차이 = 센서 잡음 x √2 (≈0.94) + 그림 자체의 변화. 판(QR) 영역과 평평한 벽의 차이 크기 비교 + 차이 영상."""
import sys; sys.path.insert(0,'../analysis_v7'); sys.path.insert(0,'../analysis_v6')
import numpy as np, pandas as pd, cv2
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams['font.family']='Malgun Gothic'; plt.rcParams['axes.unicode_minus']=False
from common import NAMES, read_raw, AXIS, session_of
from lib7 import load
D,F=load('../analysis_v7/tr_fs8_hp16_s2.npz'); qr=F.region.str.startswith('QR').values
M=D[:,qr].mean(1); S=D[:,qr,0].std(1)
idx=[i for i,n in enumerate(NAMES) if AXIS[n]=='Y']
best=[]
for a in idx:
    for b in idx:
        if a>=b: continue
        d=M[b]-M[a]; fx=abs(d[0]-round(d[0])); fy=abs(d[1]-round(d[1]))
        best.append((fx+fy,a,b,d))
best.sort(); pairs=best[:4]
X0,X1,Y0,Y1=1430,2390,950,1190
rows=[]; fig,axs=plt.subplots(len(pairs),1,figsize=(14,3.2*len(pairs)))
for ax,(e,a,b,d) in zip(axs,pairs):
    ix,iy=int(round(d[0])),int(round(d[1])); A=read_raw(NAMES[a]).astype(float); B=read_raw(NAMES[b]).astype(float)
    # B의 (x+ix, y+iy) 가 A의 (x, y) 와 같은 장면
    ra=A[Y0:Y1,X0:X1]+0; rb=B[Y0+iy:Y1+iy,X0+ix:X1+ix]; diff=rb-ra; diff-=np.median(diff)
    gm=cv2.GaussianBlur(ra,(0,0),1); g=np.hypot(cv2.Sobel(gm,cv2.CV_64F,1,0),cv2.Sobel(gm,cv2.CV_64F,0,1))/8
    edge=g>np.quantile(g,0.85); flat=g<np.quantile(g,0.3)
    # 경계에서 남은 소수 이동이 만드는 차이 예상: 기울기 x 남은 이동
    res=(d[0]-ix,d[1]-iy); pred=np.sqrt(np.mean((g[edge]*np.hypot(*res))**2))
    r=dict(쌍=f'{NAMES[a]} vs {NAMES[b]}',남은소수이동=f'({res[0]:+.3f},{res[1]:+.3f})',평평영역_차이sd=diff[flat].std(),경계영역_차이sd=diff[edge].std(),소수이동_예상=pred,센서잡음_예상=0.665*np.sqrt(2))
    rows.append(r); print(r)
    im=ax.imshow(diff,cmap='RdBu_r',vmin=-8,vmax=8); ax.set_title(f"{r['쌍']}  남은 소수 이동 {r['남은소수이동']}px  |  평평 {r['평평영역_차이sd']:.2f}, 경계 {r['경계영역_차이sd']:.2f} (센서 잡음만이면 0.94)",loc='left',fontsize=10); ax.set_xticks([]); ax.set_yticks([])
fig.colorbar(im,ax=axs,shrink=0.6,label='밝기 차이'); fig.savefig('../66_v8_보간없는_사진쌍차이.png',dpi=110,bbox_inches='tight')
pd.DataFrame(rows).round(3).to_csv('r6_pairdiff.csv',index=False,encoding='utf-8-sig')
