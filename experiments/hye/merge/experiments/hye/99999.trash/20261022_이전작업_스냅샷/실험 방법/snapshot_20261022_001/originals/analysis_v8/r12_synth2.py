"""합성 2: 실측 '판 영역 잡음' (밝기별 sd, r11) 을 픽셀마다 독립으로 넣으면 실제 점 잡음이 재현되나.
+ 같은 크기의 변화가 '무늬 모양을 따라가는' 경우 (장면 무늬 x 무작위 이득 = 무늬 결 변화) 도 비교."""
import numpy as np, pandas as pd, time, cv2
from multiprocessing import Pool
import r3_synth as S
NB=np.load('r11_noise_by_brightness.npy'); NB=NB[~np.isnan(NB[:,1])]
def sd_map(img): return np.interp(img,NB[:,0],NB[:,1])
def job(args):
    cond,k=args; rng=np.random.default_rng(5000+k); tx=rng.uniform(-20,20); ty=rng.uniform(-1,1)
    reg=S.CEN[S.RY0:S.RY1,S.RX0:S.RX1]; Fr=np.fft.fft2(reg)*np.exp(-2j*np.pi*(S.gx*tx+S.gy*ty))*np.exp(-2*np.pi**2*S.B0*(S.gx**2+S.gy**2)); out=np.real(np.fft.ifft2(Fr))
    if cond=='판 잡음(밝기별, 픽셀 독립)': out=out+rng.normal(size=out.shape)*sd_map(out)
    if cond=='판 잡음 절반 크기': out=out+rng.normal(size=out.shape)*sd_map(out)*0.5
    if cond=='무늬 결 변화(고주파 무늬 x 무작위)':
        hf=out-cv2.GaussianBlur(out,(0,0),1.5); out=out+hf*rng.normal(0,0.15,out.shape)
    img=S.CEN.copy(); img[S.RY0:S.RY1,S.RX0:S.RX1]=out; img=(img+rng.normal(0,S.SN,img.shape)).astype(np.float32); res={}
    for nm,St in S.SET.items():
        c0,_=S.coarse_init(np.pad(img,S.HPC,mode='edge'),St['BIG'],St['FP']); tm,_=S.track_tm(img,St['Tr'],S.F,St['hp'],init=c0)
        res[nm]=S.track_gn(St['bl'](img),St['Tb'],S.F,St['hp'],tm)-[tx,ty]
    return res
if __name__=='__main__':
    rows=[]
    with Pool(16) as p:
        for cond in ['판 잡음(밝기별, 픽셀 독립)','판 잡음 절반 크기','무늬 결 변화(고주파 무늬 x 무작위)']:
            R=p.map(job,[(cond,k) for k in range(40)])
            for nm in S.PIPE:
                E=np.stack([r[nm] for r in R]); Ec=E-E.mean(1,keepdims=True)
                rows.append(dict(조건=cond,측정=nm,점흔들림_x=np.median(Ec[:,:,0].std(0)),점흔들림_y=np.median(Ec[:,:,1].std(0)))); print(rows[-1],flush=True)
    pd.DataFrame(rows).round(5).to_csv('r12_synth2.csv',index=False,encoding='utf-8-sig')
