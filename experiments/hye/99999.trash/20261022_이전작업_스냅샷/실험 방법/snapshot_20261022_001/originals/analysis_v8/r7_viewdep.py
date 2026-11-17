"""그림 변화가 시점(카메라 위치 차이)에 따라 커지나? 같은 구간 Y 사진 쌍을 이동 차이별로, 판 위 평평한 곳 / QR 흰 곳 / 벽(판 밖, 비교용)의 차이 크기.
정렬: 판 평균 이동 차이만큼 푸리에 이동 (평평한 곳은 보간 영향 없음)."""
import sys; sys.path.insert(0,'../analysis_v7'); sys.path.insert(0,'../analysis_v6')
import numpy as np, pandas as pd, cv2
from multiprocessing import Pool
from common import NAMES, read_raw, AXIS, VAL, session_of
from lib7 import load
D,F=load('../analysis_v7/tr_fs8_hp16_s2.npz'); qr=F.region.str.startswith('QR').values; M=D[:,qr].mean(1)
X0,X1,Y0,Y1=1300,2520,880,1260
cen=read_raw('center.bmp')[Y0:Y1,X0:X1].astype(float); gm=cv2.GaussianBlur(cen,(0,0),1.5); g=np.hypot(cv2.Sobel(gm,cv2.CV_64F,1,0),cv2.Sobel(gm,cv2.CV_64F,0,1))/8
yy,xx=np.mgrid[Y0:Y1,X0:X1]; onplate=(xx>1440)&(xx<2380)&(yy>960)&(yy<1180)
MASK={'판 위 어두운 평평한 곳':onplate&(g<1.0)&(cen<150),'QR 흰 평평한 곳':onplate&(g<1.0)&(cen>=150)}
def crop(n,d):
    im=read_raw(n).astype(float); ix,iy=int(np.floor(d[0])),int(np.floor(d[1])); fx,fy=d[0]-ix,d[1]-iy
    c=im[Y0+iy-4:Y1+iy+4,X0+ix-4:X1+ix+4]; ky=np.fft.fftfreq(c.shape[0])[:,None]; kx=np.fft.fftfreq(c.shape[1])[None]
    c=np.real(np.fft.ifft2(np.fft.fft2(c)*np.exp(2j*np.pi*(kx*fx+ky*fy))))
    return c[4:-4,4:-4]
def job(p):
    a,b=p; A=crop(NAMES[a],M[a]); B=crop(NAMES[b],M[b]); d=B-A
    out={}
    for k,m in MASK.items(): z=d[m]; out[k]=z.std()/np.sqrt(2)
    return out
if __name__=='__main__':
    ses=np.array([session_of(n) for n in NAMES]); v=np.array([VAL[n] for n in NAMES])
    pairs=[]
    for s in ['Y1+','Y2+']:
        idx=np.where(ses==s)[0]; idx=idx[np.argsort(v[idx])]
        for gap in [1,2,4,8,16]:
            for i in range(0,len(idx)-gap,3): pairs.append((idx[i],idx[i+gap]))
    with Pool(16) as p: R=p.map(job,pairs)
    T=pd.DataFrame([dict(dY=abs(v[a]-v[b]),**r) for (a,b),r in zip(pairs,R)])
    T['구간']=pd.cut(T.dY,[0,0.06,0.12,0.25,0.5,1.0,2.1])
    print(T.groupby('구간',observed=True)[list(MASK)].median().round(3)); print('센서 잡음 (벽에서 잰 값) 0.66')
    T.to_csv('r7_viewdep.csv',index=False,encoding='utf-8-sig')
