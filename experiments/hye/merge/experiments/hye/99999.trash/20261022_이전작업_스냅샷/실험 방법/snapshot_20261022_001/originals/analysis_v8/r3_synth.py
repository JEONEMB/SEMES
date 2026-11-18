"""합성 사다리: 정답 이동을 정확히 아는 합성 사진에 의심 원인을 하나씩 넣어, 실제 점 잡음(강체 검사 0.015~0.022px)이 재현되는지.
장면 = center 사진. 기준(템플릿) 사진 = 장면 + 새 센서 잡음, 현재 사진 = 장면을 푸리에로 정확히 이동 + 새 센서 잡음 (+ 원인).
정답: 모든 점의 이동 = (tx, ty). 점 오차 = 측정 − 정답. 사진마다 공통 부분(평균)을 뺀 점 흔들림도 같이 (강체 검사와 같은 기준)."""
import sys; sys.path.insert(0,'../analysis_v6')
import numpy as np, pandas as pd, cv2, time
from multiprocessing import Pool
import common as C
from common import read_raw, cut, coarse_init, track_tm, track_gn, HPC
F=pd.read_csv('../analysis_v6/features_qr_hp8.csv'); F=F[F.region.str.startswith('QR')].reset_index(drop=True)
RX0,RX1,RY0,RY1=1150,2670,760,1380
CEN=read_raw('center.bmp').astype(np.float64)
BL=pd.read_csv('r2_blur.csv'); BL=BL[BL.file!='center.bmp']
gy=np.fft.fftfreq(RY1-RY0)[:,None]; gx=np.fft.fftfreq(RX1-RX0)[None,:]
SN=0.65; B0=0.04       # 센서 잡음, 기본 흐림 Δσ² (음의 흐림 차이를 표현하려고 모두에 깔아 둠)
def render(tx,ty,sx2,sy2,rng,gain=None,warp=None):
    reg=CEN[RY0:RY1,RX0:RX1]; Fr=np.fft.fft2(reg)
    Fr=Fr*np.exp(-2j*np.pi*(gx*tx+gy*ty))*np.exp(-2*np.pi**2*(sx2*gx**2+sy2*gy**2))
    out=np.real(np.fft.ifft2(Fr))
    if warp is not None:
        from scipy import ndimage
        yy,xx=np.mgrid[0:out.shape[0],0:out.shape[1]].astype(float); out=ndimage.map_coordinates(out,[yy-warp[1],xx-warp[0]],order=5,mode='nearest')
    if gain is not None: out=out*gain
    img=CEN.copy(); img[RY0:RY1,RX0:RX1]=out; img+=rng.normal(0,SN,img.shape) if True else 0
    return img.astype(np.float32)
def smooth_field(rng,amp,L):
    sh=(RY1-RY0,RX1-RX0); f=np.exp(-2*np.pi**2*L*L*(gx**2+gy**2))
    w=[np.real(np.fft.ifft2(np.fft.fft2(rng.normal(size=sh))*f)) for _ in range(2)]; return [x/x.std()*amp for x in w]
TMPL=render(0,0,B0,B0,np.random.default_rng(999))
def setup(hp,sg):
    bl=(lambda im: cv2.GaussianBlur(im,(0,0),sg))
    FP=F.assign(x=F.x+HPC,y=F.y+HPC); return dict(hp=hp,bl=bl,Tr=cut(TMPL,F,hp),Tb=cut(bl(TMPL),F,hp),BIG=cut(np.pad(TMPL,HPC,mode='edge'),FP,HPC),FP=FP)
PIPE={'17px σ1':(8,1.0),'33px σ2':(16,2.0)}
SET={k:setup(*v) for k,v in PIPE.items()}
def job(args):
    cond,k=args; rng=np.random.default_rng(1000+k); tx=rng.uniform(-20,20); ty=rng.uniform(-1,1)
    sx2=sy2=B0; gain=None; warp=None
    if '흐림' in cond:
        r=BL.iloc[rng.integers(len(BL))]; sx2=max(B0+r.dsig2_x,0); sy2=max(B0+r.dsig2_y,0)
    if '조명' in cond:
        yy,xx=np.mgrid[0:RY1-RY0,0:RX1-RX0]; g=rng.normal(0,0.01,3); gain=1+g[0]+g[1]*(xx-xx.mean())/xx.std()*0.3+g[2]*(yy-yy.mean())/yy.std()*0.3
    if '일렁임' in cond: warp=smooth_field(rng,0.015,40)
    img=render(tx,ty,sx2,sy2,rng,gain,warp); out={}
    for nm,S in SET.items():
        c0,_=coarse_init(np.pad(img,HPC,mode='edge'),S['BIG'],S['FP']); tm,_=track_tm(img,S['Tr'],F,S['hp'],init=c0)
        out[nm]=track_gn(S['bl'](img),S['Tb'],F,S['hp'],tm)-[tx,ty]
    return out
CONDS=['잡음만','잡음+흐림차이(실측 분포)','잡음+조명 기울기','잡음+흐림차이+조명','잡음+일렁임 0.015px(장면 변화)']
if __name__=='__main__':
    N=40; rows=[]; t0=time.time()
    with Pool(16) as p:
        for cond in CONDS:
            res=p.map(job,[(cond,k) for k in range(N)])
            for nm in PIPE:
                E=np.stack([r[nm] for r in res]); bias=np.abs(E.mean((0,1))); Ec=E-E.mean(1,keepdims=True)
                rows.append(dict(조건=cond,측정=nm,편향_x=E.mean((0,1))[0],점오차_x=np.median(E[:,:,0].std(0)),점오차_y=np.median(E[:,:,1].std(0)),
                                 점흔들림_x=np.median(Ec[:,:,0].std(0)),점흔들림_y=np.median(Ec[:,:,1].std(0))))
                print(rows[-1],f'{time.time()-t0:.0f}s',flush=True)
    rows+= [dict(조건='실제 사진 (강체 검사)',측정='17px σ1',점흔들림_x=0.0222,점흔들림_y=0.0136),dict(조건='실제 사진 (강체 검사)',측정='33px σ2',점흔들림_x=0.0149,점흔들림_y=0.0089)]
    pd.DataFrame(rows).round(5).to_csv('r3_synth.csv',index=False,encoding='utf-8-sig')
