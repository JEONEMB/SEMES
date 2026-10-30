"""Independent direct center-to-current QR tracking. Originals are read only.

Outputs: names, points in original pixels, regions (0..2 QR, 3..4 screws),
D [image, point, dx/dy], quality [image, point, fb_error/lk_L1], valid.
No labels are used for feature selection or tracking. Full-QR correlation
initializes a single-scale LK solve, avoiding pyramid pixels outside the QR.
"""
from pathlib import Path
import argparse, concurrent.futures, csv, json, time
import cv2
import numpy as np

ROOT = Path(r'C:\Users\SSAFY\Downloads\[기업연계_세메스] 교보재\[기업연계_세메스] 교보재\extracted\SSAFY Test Image Set')
OUT = Path(__file__).parent
BOXES = [(1552, 969, 1736, 1167), (1809, 971, 1993, 1169), (2067, 973, 2252, 1171)]
SCREWS = [(1492,1063),(2312,1073)]
CROP = (1340, 800, 2470, 1330)
VARIANTS = {'lk15': (15,0), 'lk31': (31,0), 'lk51': (51,0), 'lk31_blur': (31,0.8)}
cv2.setNumThreads(1)

def read(path):
    a = cv2.imdecode(np.fromfile(path, np.uint8), 0)
    if a is None: raise ValueError(str(path))
    return a[CROP[1]:CROP[3], CROP[0]:CROP[2]].copy()

def feature_set(ref):
    pts, regions, strengths = [], [], []
    sm = cv2.GaussianBlur(ref,(0,0),0.65)
    eig = cv2.cornerMinEigenVal(sm,5)
    offset = np.array(CROP[:2],np.float32)
    for k in range(5):
        mask = np.zeros(ref.shape,np.uint8)
        if k<3:
            x0,y0,x1,y1 = BOXES[k]
            margin=28 # 25px max support + cornerSubPix movement + blur support
            mask[y0+margin-CROP[1]:y1-margin-CROP[1],x0+margin-CROP[0]:x1-margin-CROP[0]]=255
            maxn, dist, qual = 600,4,0.005
        else:
            x,y = SCREWS[k-3]
            cv2.circle(mask,(x-CROP[0],y-CROP[1]),21,255,-1)
            maxn,dist,qual=60,4,0.01
        p=cv2.goodFeaturesToTrack(sm,maxn,qual,dist,mask=mask,blockSize=5)
        p=cv2.cornerSubPix(sm,p,(2,2),(-1,-1),(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,0.001)).reshape(-1,2)
        for q in p:
            if k<3:
                qg=q+offset
                if not (x0+27<=qg[0]<x1-27 and y0+27<=qg[1]<y1-27):continue
            pts.append(q);regions.append(k);strengths.append(eig[round(float(q[1])),round(float(q[0]))])
    return np.array(pts,np.float32),np.array(regions,np.int16),np.array(strengths,np.float32)

def coarse(ref,cur,include_screws=True):
    out=[]; scores=[]
    for k in range(5 if include_screws else 3):
        if k<3: x0,y0,x1,y1=BOXES[k]
        else:
            x,y=SCREWS[k-3];x0,y0,x1,y1=x-31,y-31,x+32,y+32
        x0-=CROP[0];x1-=CROP[0];y0-=CROP[1];y1-=CROP[1]
        sx,sy=110,22
        template=ref[y0:y1,x0:x1]
        search=cur[y0-sy:y1+sy,x0-sx:x1+sx]
        c=cv2.matchTemplate(search,template,cv2.TM_CCOEFF_NORMED)
        _,v,_,pos=cv2.minMaxLoc(c)
        out.append((pos[0]-sx,pos[1]-sy));scores.append(v)
    return np.array(out,np.float32),np.array(scores,np.float32)

def track(ref,cur,p,init,window):
    criteria=(cv2.TERM_CRITERIA_COUNT|cv2.TERM_CRITERIA_EPS,60,0.0001)
    p0=p.reshape(-1,1,2)
    p1,s1,e1=cv2.calcOpticalFlowPyrLK(ref,cur,p0,(p+init).reshape(-1,1,2).copy(),winSize=(window,window),maxLevel=0,criteria=criteria,flags=cv2.OPTFLOW_USE_INITIAL_FLOW,minEigThreshold=1e-5)
    p2,s2,e2=cv2.calcOpticalFlowPyrLK(cur,ref,p1,p0.copy(),winSize=(window,window),maxLevel=0,criteria=criteria,flags=cv2.OPTFLOW_USE_INITIAL_FLOW,minEigThreshold=1e-5)
    fb=np.linalg.norm(p2.reshape(-1,2)-p,axis=1)
    d=p1.reshape(-1,2)-p
    quality=np.c_[fb,e1.ravel()]
    valid=(s1.ravel()>0)&(s2.ravel()>0)&(fb<0.5)&np.isfinite(d).all(axis=1)
    return d,quality,valid

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--synthetic-only',action='store_true');args=parser.parse_args()
    ref=read(ROOT/'center.bmp')
    p,regions,strengths=feature_set(ref)
    np.savez_compressed(OUT/'features.npz',points=p+np.array(CROP[:2]),regions=regions,strengths=strengths)
    canvas=cv2.cvtColor(ref,cv2.COLOR_GRAY2BGR)
    for k,(x0,y0,x1,y1) in enumerate(BOXES):
        cv2.rectangle(canvas,(x0-CROP[0],y0-CROP[1]),(x1-CROP[0],y1-CROP[1]),(0,255,0),1)
    for q,r in zip(p,regions):cv2.circle(canvas,tuple(np.rint(q).astype(int)),1,[(0,0,255),(255,0,0),(0,150,0),(255,0,255),(0,255,255)][r],-1)
    cv2.imwrite(str(OUT/'feature_support.png'),canvas)
    meta={'qr_boxes_xyxy':BOXES,'max_patch_width':51,'feature_boundary_margin_px':27,'feature_count_by_region':np.bincount(regions).tolist(),'selection':'ShiTomasi qualityLevel .005, minDistance 4, center image only','quality_columns':['forward_backward_error_px','LK_mean_abs_pixel_error'],'valid':'LK forward/backward success and FB<.5px','coarse':'Whole-QR NCC ±110x22px','note':'Every test independently references center. No session label or motion magnitude used. Shared feature support for 15/31/51 patch comparisons; all QR patches stay inside white QR-panel rectangle. Screw templates separate, only regions>=3.'}
    (OUT/'measurement_metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf8')
    print(json.dumps(meta),flush=True)
    refs={name:cv2.GaussianBlur(ref,(0,0),sigma) if sigma else ref for name,(w,sigma) in VARIANTS.items()}
    if not args.synthetic_only:
        paths=[ROOT/'center.bmp']+sorted((ROOT/'Theta').glob('*.bmp'))+sorted((ROOT/'Y').glob('*.bmp'))
        def job(path):
            cur=read(path);init,score=coarse(ref,cur);data={}
            for name,(w,sigma) in VARIANTS.items():
                ci=cv2.GaussianBlur(cur,(0,0),sigma) if sigma else cur
                data[name]=track(refs[name],ci,p,init[regions],w)
            return data,score
        start=time.time();results=[]
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            for k,result in enumerate(executor.map(job,paths)):
                results.append(result)
                if k%20==0:print(f'tracked {k+1}/{len(paths)} in {time.time()-start:.1f}s',flush=True)
        names=np.array([x.name for x in paths])
        summaries={}
        for name in VARIANTS:
            ds=np.stack([r[0][name][0] for r in results]);q=np.stack([r[0][name][1] for r in results]);v=np.stack([r[0][name][2] for r in results])
            np.savez_compressed(OUT/(name+'.npz'),names=names,points=p+np.array(CROP[:2]),regions=regions,D=ds,quality=q,valid=v,strengths=strengths,coarse_score=np.stack([r[1] for r in results]))
            qr=regions<3
            summaries[name]={'valid_fraction_qr':float(v[:,qr].mean()),'minimum_valid_fraction_per_image':float(v[:,qr].mean(axis=1).min()),'always_valid_qr_points':int(v[:,qr].all(axis=0).sum()),'qr_total':int(qr.sum()),'fb_median_px':float(np.median(q[:,qr,0])),'fb_max_px':float(np.max(q[:,qr,0])),'photo_error_median_gray':float(np.median(q[:,qr,1]))}
        (OUT/'tracking_summary.json').write_text(json.dumps(summaries,indent=2),encoding='utf8')
        print(json.dumps(summaries),flush=True)
    synthetic(ref,p,regions,refs)

def synthetic(ref,p,regions,refs):
    rows=[]
    tests=[(dx,dy,ang) for dx,dy in [(0.1,0.07),(0.25,-0.3),(0.5,0.5),(7.7,-1.2),(28.35,0.08),(80.15,-.45),(-79.4,.3)] for ang in [0,.03]]
    for dx,dy,ang in tests:
        M=cv2.getRotationMatrix2D((ref.shape[1]/2,ref.shape[0]/2),ang,1)
        M[:,2]+=[dx,dy]
        cur=cv2.warpAffine(ref,M,(ref.shape[1],ref.shape[0]),flags=cv2.INTER_CUBIC,borderMode=cv2.BORDER_REFLECT)
        true=p@M[:,:2].T+M[:,2]-p
        init,sc=coarse(ref,cur)
        for name,(w,sigma) in VARIANTS.items():
            ci=cv2.GaussianBlur(cur,(0,0),sigma) if sigma else cur
            d,q,v=track(refs[name],ci,p,init[regions],w)
            sel=(regions<3)&v
            err=d[sel]-true[sel]
            row=dict(variant=name,dx=dx,dy=dy,angle=ang,n=int(sel.sum()),valid_rate=float(v[regions<3].mean()),bias_dx=float(err[:,0].mean()),bias_dy=float(err[:,1].mean()),rmse=float(np.sqrt((err**2).mean())),p95_euclidean=float(np.quantile(np.linalg.norm(err,axis=1),.95)),max_euclidean=float(np.linalg.norm(err,axis=1).max()))
            rows.append(row)
    with (OUT/'synthetic_validation.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    print('synthetic validation complete',flush=True)

if __name__=='__main__':main()
