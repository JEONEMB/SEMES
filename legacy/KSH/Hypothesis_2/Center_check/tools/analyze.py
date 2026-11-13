# 시제품 분석 : landmark_shifts_py.json 으로 묶음별(플레이트만 / +스테이션 / +벽 / 전체) 모델을 학습·검증
# 사용 : python analyze.py   (tools 폴더에서 실행)
import os
HERE=os.path.dirname(os.path.abspath(__file__))
import json,re,numpy as np
R=json.load(open(os.path.join(HERE,'..','results','center','landmark_shifts_py.json')))
def truth(n):
    s=n[:-4]
    if s.startswith('center'): return 0.0,0.0,(2 if s!='center' else 1),(int(s[6:]) if s!='center' else 0)
    y=re.search(r'Y([+-][\d.]+)',s); t=re.search(r'T([+-][\d.]+)',s); sh=re.search(r'\((\d)\)',s)
    return (float(y.group(1)) if y else 0.0),(float(t.group(1)) if t else 0.0),(2 if sh else 1),(int(sh.group(1)) if sh else 0)
names=sorted(R); LM=[k for k in R[names[0]]]
use=['QR_L','QR_C','QR_R','label','nut_L','nut_R','bracket_scr_L','bracket_scr_R','wall_U1','wall_U2','wall_W1','wall_W2','wall_W3','wall_W4']
info=np.array([truth(n) for n in names]); Y,T,S,SH=info.T
DX=np.array([[R[n][k][0] for k in use] for n in names]); PK=np.array([[R[n][k][2] for k in use] for n in names])
print('min peak per landmark',dict(zip(use,PK.min(0).round(2))))
s1=(S==1)&~((Y==0)&(T==0))
# train/test split set1: sort within sweep, even=train
idx=np.where(s1)[0]
ysw=idx[(T[idx]==0)]; tsw=idx[(Y[idx]==0)]
ysw=ysw[np.argsort(Y[ysw])]; tsw=tsw[np.argsort(T[tsw])]
train=np.r_[ysw[::2],tsw[::2]]; test=np.r_[ysw[1::2],tsw[1::2]]
i0=np.where((S==1)&(SH==0)&(Y==0)&(T==0))[0]  # center.bmp
def design(y,t): return np.c_[y,t,t**2,np.ones_like(t)]
def fit(tr,cols):
    A=design(Y[tr],T[tr]); P,*_=np.linalg.lstsq(A,DX[tr][:,cols],rcond=None)
    r=DX[tr][:,cols]-A@P; C=np.cov(r.T)+np.eye(len(cols))*1e-8
    return P,C,r
def est(P,C,d,ref=None):
    W=np.linalg.inv(C); out=[]
    for row in d:
        y,t=0.0,0.0
        for _ in range(8):
            J=np.c_[P[0],P[1]+2*P[2]*t]; pred=P[0]*y+P[1]*t+P[2]*t*t+P[3]
            H=J.T@W@J; g=J.T@W@(row-pred); dlt=np.linalg.solve(H,g); y+=dlt[0]; t+=dlt[1]
        out.append((y,t))
    return np.array(out)
def report(lbl,e,m):
    eY=(e[:,0]-Y[m])*1000; eT=e[:,1]-T[m]
    print(f'  {lbl:30s} n={len(eT):3d} | Θ RMS {np.sqrt((eT**2).mean()):.4f}° max {np.abs(eT).max():.4f} mean {eT.mean():+.4f} ±0.01 {np.sum(np.abs(eT)<=0.01)}/{len(eT)} | Y RMS {np.sqrt((eY**2).mean()):6.1f}µm max {np.abs(eY).max():6.1f} ±90 {np.sum(np.abs(eY)<=90)}/{len(eY)} | both {np.sum((np.abs(eT)<=0.01)&(np.abs(eY)<=90))}')
for setname,cols in (('plate only (QR+label)',[0,1,2,3]),('plate + station',list(range(8))),('plate + wall (parallax)',[0,1,2,3,8,9,10,11,12,13]),('all landmarks',list(range(14)))):
    P,C,r=fit(train,cols)
    print(f'== {setname}:  residual σ per lm',np.sqrt(np.diag(C)).round(3))
    e=est(P,C,DX[test][:,cols]); 
    mY=test[T[test]==0]; mT=test[Y[test]==0]
    report('set1 test Y sweep',est(P,C,DX[mY][:,cols]),mY)
    report('set1 test Θ sweep',est(P,C,DX[mT][:,cols]),mT)
    # set2 relative to center1 (shot 1)
    c1=np.where((S==2)&(Y==0)&(T==0)&(SH==1))[0][0]
    m2=np.where((S==2)&~((Y==0)&(T==0)))[0]
    d2=DX[m2][:,cols]-DX[c1][cols]+P[3]   # re-reference
    e2=est(P,C,d2); report('set2 all (ref center1)',e2,m2)
    for lab,mm in (('  Θ<=0.2',T[m2]<=0.2+1e-9),('  Θ=1',np.abs(T[m2]-1)<1e-9),('  Θ=2',np.abs(T[m2]-2)<1e-9)):
        report('set2'+lab,e2[mm],m2[mm])
    cc=np.where((S==2)&(Y==0)&(T==0))[0]
    print('   centers (vs center.bmp): ', {names[i]:tuple(np.round(est(P,C,DX[[i]][:,cols])[0]*[1000,1],4)) for i in cc})
