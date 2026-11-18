import numpy as np, cv2, json
def _updft(data, ups, uf, off):
    r,c=data.shape
    kc=np.exp((-1j*2*np.pi/(c*uf))*(np.fft.ifftshift(np.arange(c))[:,None]-np.floor(c/2)).dot(np.arange(ups[1])[None,:]-off[1]))
    kr=np.exp((-1j*2*np.pi/(r*uf))*(np.arange(ups[0])[:,None]-off[0]).dot(np.fft.ifftshift(np.arange(r))[None,:]-np.floor(r/2)))
    return kr.dot(data).dot(kc)
def xcorr_shift(a,b,uf=100):
    """shift of b relative to a (b(x)=a(x-s)), returns (dx,dy,peak)"""
    w=np.outer(np.hanning(a.shape[0]),np.hanning(a.shape[1]))
    a=(a-a.mean())*w; b=(b-b.mean())*w
    A=np.fft.fft2(a); B=np.fft.fft2(b); X=B*A.conj(); X/=np.abs(X)+1e-9*np.abs(X).max()
    cc=np.fft.ifft2(X); idx=np.unravel_index(np.argmax(np.abs(cc)),cc.shape)
    mid=np.array(cc.shape)//2; s=np.array(idx,float); s[s>mid]-=np.array(cc.shape)[s>mid]
    s=np.round(s*uf)/uf; ur=np.ceil(uf*1.5); ds=np.fix(ur/2); off=ds-s*uf
    cu=_updft(X.conj(),(int(ur),int(ur)),uf,off).conj()
    m=np.unravel_index(np.argmax(np.abs(cu)),cu.shape); s=s+(np.array(m)-ds)/uf
    return s[1],s[0],np.abs(cu).max()/X.size
import os
L=json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'landmarks.json')))
def measure(ref,g,L=L,skip=('plate_hole_L','plate_hole_R')):
    rf=ref.astype(np.float64); gf=g.astype(np.float64)
    # coarse from plate band
    y0,y1,x0,x1=950,1180,1440,2380
    dx,dy,_=xcorr_shift(rf[y0:y1,x0-250:x1+250],gf[y0:y1,x0-250:x1+250],10)
    out={}
    for k,(cx,cy,hw,hh,grp) in L.items():
        if k in skip: continue
        sx,sy=dx,dy; pk=0.0
        for it in range(3):
            ix,iy=int(round(sx)),int(round(sy))
            a=rf[cy-hh:cy+hh,cx-hw:cx+hw]; b=gf[cy-hh+iy:cy+hh+iy,cx-hw+ix:cx+hw+ix]
            if b.shape!=a.shape or cy-hh+iy<0 or cx-hw+ix<0: pk=0.0; break
            rx,ry,pk=xcorr_shift(a,b); sx,sy=ix+rx,iy+ry
            if abs(rx)<0.5 and abs(ry)<0.5: break
        out[k]=(sx,sy,pk,grp)
    return out
