import sys; sys.path.insert(0,'../analysis_v6')
import numpy as np, cv2
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams['font.family']='Malgun Gothic'; plt.rcParams['axes.unicode_minus']=False
from common import read_raw
A=read_raw('Y-0.20.bmp').astype(float); B=read_raw('Y+0.40.bmp').astype(float); dx=8
regs=[('QR 왼쪽 위 모서리',1560,975),('QR 가운데 무늬',1900,1060),('판 어두운 곳',1820,1150)]
fig,axs=plt.subplots(3,4,figsize=(16,12))
for r,(nm,x,y) in enumerate(regs):
    a=A[y:y+60,x:x+60]; b=B[y:y+60,x+dx:x+dx+60]; co=np.polyfit(a.ravel(),b.ravel(),1); d=b-(co[0]*a+co[1])
    gx=np.gradient(cv2.GaussianBlur(a,(0,0),0.7),axis=1)
    # 이 창 안에서 최소제곱으로 남은 이동 추정: d ≈ -gx*sx - gy*sy
    gy=np.gradient(cv2.GaussianBlur(a,(0,0),0.7),axis=0); M=np.c_[gx.ravel(),gy.ravel(),np.ones(a.size)]; s=np.linalg.lstsq(M,d.ravel(),rcond=None)[0]; d2=d-(M@s).reshape(d.shape)
    for c,(im,t,cm,lim) in enumerate([(a,'사진 A (Y-0.20)','gray',None),(b,'사진 B (Y+0.40), 정수 이동만 맞춤','gray',None),(d,f'B - A (밝기 맞춤) sd {d.std():.2f}','RdBu_r',10),(d2,f'창 안 이동 ({s[0]:+.3f},{s[1]:+.3f})px 뺀 뒤 sd {d2.std():.2f}','RdBu_r',10)]):
        ax=axs[r,c]; ax.imshow(im,cmap=cm,vmin=None if lim is None else -lim,vmax=lim,interpolation='nearest'); ax.set_title(f'{nm}: {t}',fontsize=9); ax.set_xticks([]); ax.set_yticks([])
fig.tight_layout(); fig.savefig('../67_v8_차이영상_확대.png',dpi=100)
