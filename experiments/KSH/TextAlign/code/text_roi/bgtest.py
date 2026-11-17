import cv2,numpy as np,glob,sys,json
from absmeas import CFG, coarse_shift, shifted, bg_center
def radial_center(roi):
    # Parthasarathy (2012, Nature Methods 9:724) radial symmetry center
    I=roi.astype(np.float64); Ny,Nx=I.shape
    dIdu=I[:-1,1:]-I[1:,:-1]; dIdv=I[:-1,:-1]-I[1:,1:]
    k=np.ones((3,3))/9; dIdu=cv2.filter2D(dIdu,-1,k); dIdv=cv2.filter2D(dIdv,-1,k)
    dImag2=dIdu**2+dIdv**2
    xm,ym=np.meshgrid(np.arange(Nx-1)+0.5,np.arange(Ny-1)+0.5)
    m=-(dIdv+dIdu)/(dIdu-dIdv+1e-12); m[~np.isfinite(m)]=0
    b=ym-m*xm
    sdI2=dImag2.sum(); xcen=(dImag2*xm).sum()/sdI2; ycen=(dImag2*ym).sum()/sdI2
    w=dImag2/np.sqrt((xm-xcen)**2+(ym-ycen)**2+1e-9)
    wm2p1=w/(m**2+1)
    sw=wm2p1.sum(); smmw=(m**2*wm2p1).sum(); smw=(m*wm2p1).sum(); smbw=(m*b*wm2p1).sum(); sbw=(b*wm2p1).sum()
    det=smw**2-smmw*sw
    xc=(smbw*sw-smw*sbw)/det; yc=(smbw*smw-smmw*sbw)/det
    return xc,yc
def bg_radial(im,q,m=40,half=34):
    cx,cy,hw=q["cx"],q["cy"],q["hw"]; x0,y0=cx-hw-m,cy-hw-m
    roi=cv2.GaussianBlur(im[y0:cy+hw+m,x0:cx+hw+m].astype(np.float32),(0,0),1.0)
    # 대략 중심 : 가장 어두운 곳
    sm=cv2.GaussianBlur(roi,(0,0),6); yy,xx=np.unravel_index(np.argmin(sm),sm.shape)
    for _ in range(3):
        a,b=int(round(xx))-half,int(round(yy))-half
        sub=roi[b:b+2*half+1,a:a+2*half+1]
        xc,yc=radial_center(sub); xx,yy=a+xc,b+yc
    return x0+xx,y0+yy
def bg_wcent(im,q,m=40,R=30):
    cx,cy,hw=q["cx"],q["cy"],q["hw"]; x0,y0=cx-hw-m,cy-hw-m
    roi=cv2.GaussianBlur(im[y0:cy+hw+m,x0:cx+hw+m].astype(np.float32),(0,0),1.0)
    border=np.r_[roi[:4].ravel(),roi[-4:].ravel()]; bg=np.median(border)
    sm=cv2.GaussianBlur(roi,(0,0),6); yc,xc=np.unravel_index(np.argmin(sm),sm.shape); yc=float(yc);xc=float(xc)
    yy,xx=np.mgrid[0:roi.shape[0],0:roi.shape[1]]
    for _ in range(10):
        g=np.exp(-((xx-xc)**2+(yy-yc)**2)/(2*R**2)); w=np.clip(bg-roi,0,None)*g; s=w.sum(); xc,yc=(w*xx).sum()/s,(w*yy).sum()/s
    return x0+xc,y0+yc
fs=sorted(glob.glob(sys.argv[1]+'/**/*.bmp',recursive=True))
out=[]
for p in fs:
    im=cv2.imread(p,0); dx=coarse_shift(im,CFG)
    r=dict(path=p)
    for nm,fn in [('cc',bg_center),('rad',bg_radial),('wc',bg_wcent)]:
        r[nm]=[fn(im,shifted(q,dx),40)[0] for q in CFG['bg']]
    out.append(r)
json.dump(out,open(sys.argv[2],'w'))
print(len(out))
