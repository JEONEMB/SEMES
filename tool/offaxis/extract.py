"""Raw-pixel QR L-corner fitting and independent affine patch ECC.

The optical blur parameter is part of the image formation model; images are
never smoothed. ECC explicitly uses a one-pixel Gaussian kernel (identity).
"""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import time

import cv2
import numpy as np
import pandas as pd
import qrcode
from qrcode.util import QRData, MODE_8BIT_BYTE, MODE_ALPHA_NUM
from scipy.optimize import least_squares
from scipy.special import erf

CRITERIA = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 100, 1e-7)


def load(path):
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)


def regenerate(payload, observed):
    candidates = []
    for ec in [0, 1, 2, 3]:
        for mask in range(8):
            for mode in [MODE_ALPHA_NUM, MODE_8BIT_BYTE]:
                qr = qrcode.QRCode(version=1, error_correction=ec, border=0, mask_pattern=mask)
                qr.add_data(QRData(payload, mode=mode), optimize=0)
                qr.make(fit=False)
                bits = np.array(qr.get_matrix(), dtype=np.uint8)
                candidates.append((np.mean(bits != (observed < 128)), ec, mask, mode, bits))
    candidates.sort(key=lambda c: c[0])
    error, ec, mask, mode, bits = candidates[0]
    if error > .03:
        raise ValueError(f'{payload}: regenerated QR mismatch {error}')
    return bits, dict(mismatch=error, ec=ec, mask=mask, encoding=mode)


def reference(image):
    detector = cv2.QRCodeDetector()
    result = []
    for slot, (x, y) in zip('LCR', [(1644,1068), (1902,1069), (2160,1070)]):
        crop = image[y-120:y+120, x-120:x+120]
        payload, corners, observed = detector.detectAndDecode(crop)
        if not payload:
            raise ValueError(f'Raw baseline QR decode failed: {slot}')
        corners = corners[0] + [x-120, y-120]
        bits, info = regenerate(payload, observed)
        n = len(bits)
        H = cv2.getPerspectiveTransform(np.float32([[0,0],[n,0],[n,n],[0,n]]), corners.astype(np.float32))
        padded = np.pad(bits, 1)
        features = []
        diagonal = 0
        for row in range(n+1):
            for col in range(n+1):
                b = padded[row:row+2,col:col+2].astype(float)
                if b.sum() == 2 and b[0,0] == b[1,1] and b[0,1] == b[1,0]:
                    diagonal += 1
                if b.sum() not in [1,3]:
                    continue
                seed = cv2.perspectiveTransform(np.float32([[[col,row]]]),H)[0,0]
                nearby = cv2.perspectiveTransform(np.float32([[[col+1,row],[col,row+1]]]),H)[0]
                basis = (nearby-seed).T
                basis /= np.linalg.norm(basis, axis=0)
                fit = fit_corner(image, seed, b, basis, 5)
                features.append(dict(id=row*(n+1)+col, row=row, col=col, bits=b.tolist(),
                                     seed=seed.tolist(), basis=basis.tolist(),
                                     reference=[fit['x'],fit['y']], reference_rmse=fit['rmse']))
        # PPT label-centered patch (204 wide x 216 high); exact pixel bounds recorded.
        result.append(dict(slot=slot,payload=payload,features=features,diagonal=diagonal,
                           center=[x,y],bounds=[x-102,y-108,204,216],qr_model=info))
    return result


def fit_corner(image, seed, bits, basis, radius):
    sx, sy = seed
    ix, iy = int(round(sx)), int(round(sy))
    patch = image[iy-radius:iy+radius+1,ix-radius:ix+radius+1].astype(float)
    if patch.shape != (2*radius+1,2*radius+1):
        return dict(x=np.nan,y=np.nan,rmse=np.nan,contrast=0,sigma=np.nan,condition=np.inf,success=False,boundary=True)
    yy,xx = np.mgrid[-radius:radius+1,-radius:radius+1]
    xx=(xx+ix-sx).ravel(); yy=(yy+iy-sy).ravel(); target=patch.ravel()
    inv = np.linalg.inv(np.asarray(basis))
    a,b,c,d=np.asarray(bits).ravel()
    # Bilinear interpolation of quadrant probabilities, not image pixels.
    def evaluate(p, jac=False):
        dx,dy,sigma,gain,offset=p
        u=inv[0,0]*(xx-dx)+inv[0,1]*(yy-dy)
        v=inv[1,0]*(xx-dx)+inv[1,1]*(yy-dy)
        fu=.5*(1+erf(u/(sigma*np.sqrt(2))))
        fv=.5*(1+erf(v/(sigma*np.sqrt(2))))
        pattern=a+(b-a)*fu+(c-a)*fv+(a-b-c+d)*fu*fv
        if not jac:
            return offset-gain*pattern-target
        gu=np.exp(-.5*(u/sigma)**2)/(sigma*np.sqrt(2*np.pi))
        gv=np.exp(-.5*(v/sigma)**2)/(sigma*np.sqrt(2*np.pi))
        du=(b-a)+(a-b-c+d)*fv
        dv=(c-a)+(a-b-c+d)*fu
        return np.column_stack([gain*(du*gu*inv[0,0]+dv*gv*inv[1,0]),
                                gain*(du*gu*inv[0,1]+dv*gv*inv[1,1]),
                                gain*(du*gu*u+dv*gv*v)/sigma,-pattern,np.ones(len(pattern))])
    initial=[0,0,.8,max(45.,float(np.ptp(patch))),float(patch.max())]
    opt=least_squares(evaluate,initial,jac=lambda p:evaluate(p,True),
                      bounds=([-3.6,-3.6,.18,1,0],[3.6,3.6,2.6,300,300]),
                      max_nfev=35,ftol=1e-7,xtol=1e-7,gtol=1e-6)
    p=opt.x
    j=opt.jac
    condition=np.linalg.cond(j/np.maximum(np.linalg.norm(j,axis=0),1e-12))
    return dict(x=sx+p[0],y=sy+p[1],rmse=float(np.sqrt(np.mean(opt.fun**2))),
                contrast=p[3],sigma=p[2],condition=condition,success=bool(opt.success),
                boundary=bool(np.any(opt.active_mask[:3] != 0)))


def affine(image, base, qr):
    x,y,w,h=qr['bounds']
    template=base[y:y+h,x:x+w]
    # Coarse search broad enough for 2 degree motion (~160px), without label input.
    left,top=max(0,x-300),max(0,y-120)
    search=image[top:min(image.shape[0],y+h+120),left:min(image.shape[1],x+w+300)]
    scores=cv2.matchTemplate(search,template,cv2.TM_CCOEFF_NORMED)
    _,coarse,_,loc=cv2.minMaxLoc(scores)
    cx,cy=left+loc[0],top+loc[1]
    patch=image[cy:cy+h,cx:cx+w]
    warp=np.eye(2,3,dtype=np.float32)
    correlation,warp=cv2.findTransformECC(template,patch,warp,cv2.MOTION_AFFINE,CRITERIA,None,1)
    global_warp=warp.astype(float)
    global_warp[:,2]+=np.array([cx,cy])-global_warp[:,:2]@np.array([x,y])
    center=np.array(qr['center'])
    delta=global_warp[:,:2]@center+global_warp[:,2]-center
    registered=cv2.warpAffine(patch,warp,(w,h),flags=cv2.INTER_LINEAR|cv2.WARP_INVERSE_MAP)
    X=np.column_stack([registered[5:-5,5:-5].ravel(),np.ones((h-10)*(w-10))])
    target=template[5:-5,5:-5].ravel()
    coeff=np.linalg.lstsq(X,target,rcond=None)[0]
    rmse=np.sqrt(np.mean((X@coeff-target)**2))
    return global_warp,dict(correlation=correlation,coarse_correlation=coarse,dx=delta[0],dy=delta[1],photometric_rmse=rmse,
                            **{f'a{i}{j}':global_warp[i,j] for i in range(2) for j in range(3)})


def extract_one(args):
    path,rel,basepath,models=args
    cv2.setNumThreads(1)
    image,base=load(path),load(basepath)
    points=[]; patches=[]
    for qr in models:
        try:
            warp,summary=affine(image,base,qr)
            patches.append(dict(path=rel,qr=qr['slot'],status='ok',**summary))
        except cv2.error as exc:
            patches.append(dict(path=rel,qr=qr['slot'],status=str(exc)))
            continue
        for f in qr['features']:
            ref=np.array(f['reference'])
            seed=warp[:,:2]@ref+warp[:,2]
            basis=warp[:,:2]@np.array(f['basis'])
            basis/=np.linalg.norm(basis,axis=0)
            fits=[fit_corner(image,seed,f['bits'],basis,r) for r in [4,5]]
            fit,other=fits
            difference=np.hypot(fit['x']-other['x'],fit['y']-other['y'])
            accepted=fit['success'] and fit['rmse']<=15 and fit['contrast']>=45 and not fit['boundary'] and difference<=.65
            points.append(dict(path=rel,qr=qr['slot'],id=f['id'],row=f['row'],col=f['col'],
                               reference_x=ref[0],reference_y=ref[1],dx=fit['x']-ref[0],dy=fit['y']-ref[1],
                               window_difference=difference,accepted=accepted,**fit))
    return points,patches


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--limit',type=int)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    cv2.setNumThreads(1)
    basepath=args.root/'center.bmp'
    models=reference(load(basepath))
    (args.output/'reference.json').write_text(json.dumps(models,indent=2),encoding='utf-8')
    print('Reference counts:',[(q['slot'],len(q['features']),q['diagonal']) for q in models],flush=True)
    paths=sorted(args.root.rglob('*.bmp'))
    if args.limit:
        paths=paths[:args.limit]
    start=time.time(); points=[];patches=[]
    tasks=[(str(p),p.relative_to(args.root).as_posix(),str(basepath),models) for p in paths]
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i,(p,q) in enumerate(pool.map(extract_one,tasks)):
            points.extend(p);patches.extend(q)
            if (i+1)%10==0:
                print(f'{i+1}/{len(paths)} images, {time.time()-start:.1f}s',flush=True)
    pd.DataFrame(points).to_csv(args.output/'points.csv',index=False)
    pd.DataFrame(patches).to_csv(args.output/'ecc.csv',index=False)
    manifest=dict(images=len(paths),seconds=time.time()-start,opencv=cv2.__version__,workers=args.workers,
                  smoothing=False,ecc_gauss_filter_size=1,point_model='separable Gaussian quadrant probabilities, local axes fixed by affine QR geometry',
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (args.output/'extraction.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(manifest),flush=True)


if __name__=='__main__':
    main()
