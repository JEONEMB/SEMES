# 기준 사진 없이 '절대 좌표 ROI(텍스트)'만으로 특징을 재는 시험 구현
import cv2, numpy as np, sys, json
CFG = {
  "qr": [{"name":"QR_L","cx":1650,"cy":1065,"hw":110},{"name":"QR_C","cx":1907,"cy":1065,"hw":110},{"name":"QR_R","cx":2165,"cy":1065,"hw":110}],
  "bg": [{"name":"W1","cx":966,"cy":1358,"hw":60},{"name":"W2","cx":2865,"cy":1359,"hw":60},{"name":"W3","cx":1009,"cy":1684,"hw":60},{"name":"W4","cx":2821,"cy":1685,"hw":60}],
  "bg_method": "radial",
  "qr_search_px": 40, "bg_search_px": 40,
}
def subpix_edge(profile, sign):
    g = np.diff(profile.astype(np.float64)) * sign
    i = int(np.argmax(g))
    if 0 < i < len(g)-1:
        a, b, c = g[i-1], g[i], g[i+1]; den = a - 2*b + c
        off = 0.5*(a-c)/den if den != 0 else 0.0
    else: off = 0.0
    return i + 0.5 + off, g[i]
def qr_center(im, q, m):
    cx, cy, hw = q["cx"], q["cy"], q["hw"]
    x0, x1 = cx-hw-m, cx+hw+m; y0, y1 = cy-hw-m, cy+hw+m
    roi = cv2.GaussianBlur(im[y0:y1, x0:x1].astype(np.float32), (0,0), 1.0)
    H, Wd = roi.shape
    # 1) 대략 경계 : 밝은 사각형 열/행 평균
    colm = roi[H//2-hw//2:H//2+hw//2].mean(0); rowm = roi[:, Wd//2-hw//2:Wd//2+hw//2].mean(1)
    L,_ = subpix_edge(colm[:Wd//2], +1); R,_ = subpix_edge(colm[Wd//2:], -1); R += Wd//2
    T,_ = subpix_edge(rowm[:H//2], +1); B,_ = subpix_edge(rowm[H//2:], -1); B += H//2
    # 2) 각 행마다 좌/우 경계 서브픽셀 → 직선 맞춤
    ys = np.arange(int(T)+8, int(B)-8); lx=[]; rx=[]
    for y in ys:
        p = roi[y]; a=int(L)-6; b=int(L)+7; xl,_=subpix_edge(p[a:b],+1); lx.append(a+xl)
        a=int(R)-6; b=int(R)+7; xr,_=subpix_edge(p[a:b],-1); rx.append(a+xr)
    xs = np.arange(int(L)+8, int(R)-8); ty=[]; by=[]
    for x in xs:
        p = roi[:, x]; a=int(T)-6; b=int(T)+7; yt,_=subpix_edge(p[a:b],+1); ty.append(a+yt)
        a=int(B)-6; b=int(B)+7; yb,_=subpix_edge(p[a:b],-1); by.append(a+yb)
    def robust_line(t, v):
        t=np.asarray(t,float); v=np.asarray(v,float); k=np.ones(len(t),bool)
        for _ in range(3):
            p=np.polyfit(t[k],v[k],1); r=v-np.polyval(p,t); s=1.4826*np.median(np.abs(r[k]))+1e-6; k=np.abs(r)<3*s
        return p
    pl=robust_line(ys,lx); pr=robust_line(ys,rx); pt=robust_line(xs,ty); pb=robust_line(xs,by)
    yc=(np.polyval(pt,(L+R)/2)+np.polyval(pb,(L+R)/2))/2
    xc=(np.polyval(pl,yc)+np.polyval(pr,yc))/2
    yc=(np.polyval(pt,xc)+np.polyval(pb,xc))/2
    return x0+xc, y0+yc
def bg_center(im, q, m):
    cx, cy, hw = q["cx"], q["cy"], q["hw"]
    x0, x1 = cx-hw-m, cx+hw+m; y0, y1 = cy-hw-m, cy+hw+m
    roi = cv2.GaussianBlur(im[y0:y1, x0:x1].astype(np.float32), (0,0), 1.0)
    border = np.r_[roi[:4].ravel(), roi[-4:].ravel(), roi[:,:4].ravel(), roi[:,-4:].ravel()]
    bg = np.median(border); sd = 1.4826*np.median(np.abs(border-bg))+1.0
    w = np.clip(bg - roi - 4*sd, 0, None)
    # 가장 큰 어두운 덩어리만
    n, lab, st, cen = cv2.connectedComponentsWithStats((w>0).astype(np.uint8))
    if n < 2: return np.nan, np.nan
    k = 1+int(np.argmax(st[1:, cv2.CC_STAT_AREA])); w = w*(lab==k)
    yy, xx = np.mgrid[0:roi.shape[0], 0:roi.shape[1]]
    s = w.sum(); return x0+(w*xx).sum()/s, y0+(w*yy).sum()/s
def coarse_shift(im, cfg):
    """① 대략 위치 : QR 사이의 어두운 틈(사각형 간격 약 37 px)을 열 평균 프로파일에서 찾아 설정 좌표 대비 가로 이동(px)을 구함.
       QR 안의 검은 모듈은 닫힘 연산(21 px)으로 메워 사각형을 균일하게 밝게 만든 뒤 찾는다."""
    qs=sorted(cfg["qr"],key=lambda q:q["cx"]); hw=qs[0]["hw"]; cy=int(np.mean([q["cy"] for q in qs])); R=cfg.get("coarse_range_px",350)
    x0=qs[0]["cx"]-hw-R; x1=qs[-1]["cx"]+hw+R
    band=im[cy-hw//2:cy+hw//2, x0:x1].astype(np.float32)
    band=cv2.morphologyEx(band,cv2.MORPH_CLOSE,np.ones((21,21),np.uint8))
    prof=cv2.GaussianBlur(band,(0,0),2).mean(0)
    dark=(prof<0.6*np.median(prof)).astype(np.int8); dd=np.diff(np.r_[0,dark,0]); st=np.where(dd==1)[0]; en=np.where(dd==-1)[0]
    gaps=np.array([x0+(s+e)/2 for s,e in zip(st,en) if 12<=(e-s)<=80])
    exp=np.array([(qs[i]["cx"]+qs[i+1]["cx"])/2 for i in range(len(qs)-1)])
    pitch=np.diff(exp).mean() if len(exp)>1 else 0
    best=None
    for i in range(len(gaps)-len(exp)+1):
        g=gaps[i:i+len(exp)]; off=np.mean(g-exp); err=np.abs(g-exp-off).max()
        if best is None or err<best[1]: best=(off,err)
    if best is None or best[1]>cfg.get("coarse_max_spread_px",15): raise RuntimeError("coarse: QR 사이 틈을 찾지 못함 (%d개)"%len(gaps))
    return float(best[0])
def shifted(q,dx): r=dict(q); r["cx"]=int(round(q["cx"]+dx)); return r
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
def measure(path, cfg=CFG):
    im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if im is None: raise RuntimeError(path)
    dx=coarse_shift(im,cfg)
    qx=[qr_center(im,shifted(q,dx),cfg["qr_search_px"]) for q in cfg["qr"]]; bfn=bg_radial if cfg.get("bg_method","radial")=="radial" else bg_center
    bx=[bfn(im,shifted(q,dx),cfg["bg_search_px"]) for q in cfg["bg"]]
    s=float(np.mean([p[0] for p in qx])); bgm=float(np.mean([p[0] for p in bx]))
    return dict(s_abs=s, d_abs=s-bgm, qx=[p[0] for p in qx], qy=[p[1] for p in qx], bx=[p[0] for p in bx], by=[p[1] for p in bx])
if __name__=="__main__":
    for p in sys.argv[1:]:
        r=measure(p); print(json.dumps(dict(file=p,**{k:(np.round(v,4).tolist() if isinstance(v,list) else round(v,4)) for k,v in r.items()}),ensure_ascii=False))
