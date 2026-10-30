import judge7 as J
from judge7 import *
cen=cv2.GaussianBlur(read_raw('center.bmp'),(0,0),2.0); gx=cv2.Sobel(cen,cv2.CV_64F,1,0)/8; gy=cv2.Sobel(cen,cv2.CV_64F,0,1)/8
F=F8; qr=F.region.str.startswith('QR').values
def ratio(x,y,h=16):
    s=(slice(y-h,y+h+1),slice(x-h,x+h+1)); a,c,b=(gx[s]**2).sum(),(gy[s]**2).sum(),(gx[s]*gy[s]).sum()
    t=np.sqrt(((a-c)/2)**2+b*b); return ((a+c)/2-t)/((a+c)/2+t)
r=np.array([ratio(x,y) for x,y in zip(F.x,F.y)]); dscrew=np.min([np.hypot(F.x-sx,F.y-sy) for sx,sy in SCREWS],0)
o=np.load('e15_top.npy'); Fq=F[qr].reset_index(drop=True); rq=r[qr]
print('큰 화살표 점의 모서리성 (작은/큰 고유값, 1=모서리, 0=한 방향 경계):',[(int(Fq.x[i]),int(Fq.y[i]),round(rq[i],3)) for i in o[:6]])
print('QR 점 전체 모서리성 중앙',np.median(rq).round(3),' 0.1 미만',int((rq<0.1).sum()),'개')
D,_=load('tr_fs8_hp16_s2.npz')
rng=np.random.default_rng(0); splits=[]
for _ in range(40):
    f=np.full(len(NAMES),-1)
    for s in SES:
        idx=np.where(SESA==s)[0]; p=rng.permutation(idx); f[p[:len(p)//2]]=0; f[p[len(p)//2:]]=1
    splits.append(f)
base=J.FOLDA.copy()
for nm,keep in [('전부 (232)',qr),('경계점 제외 (모서리성<0.1)',qr&(r>=0.1)),('경계점 제외 (<0.2)',qr&(r>=0.2)),('나사 150px 안 제외',qr&(dscrew>150)),('둘 다 제외',qr&(r>=0.1)&(dscrew>150))]:
    Fk=F[keep].reset_index(drop=True); X,_=summarize(D[:,keep],Fk,'affine'); n=floor(X,NAMES)[0]
    J.FOLDA=base; e=evaluate(run(X,True,'center','raw','full'),'cmd')
    acc=[]
    for f in splits: J.FOLDA=f; acc.append(evaluate(run(X,True,'center','raw','full'),'cmd')['축정확도'])
    J.FOLDA=base
    print(f'{nm:28s} 점 {keep.sum():3d}  {n:.1f}σ/mm  짝홀 {e["축정확도"]:.2f}% (틀림 {e["축틀림_Y"]+e["축틀림_Θ"]})  반복40 평균 {np.mean(acc):.2f}%')
