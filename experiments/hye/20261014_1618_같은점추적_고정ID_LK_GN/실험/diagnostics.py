"""Tracking versus independent corner redetection, with no motion labels used."""
import concurrent.futures,csv,json
import cv2,numpy as np
from scipy.spatial import cKDTree
from measure import ROOT,OUT,CROP,BOXES,read,coarse

cv2.setNumThreads(1)

def main():
    data=np.load(OUT/'gn17.npz');points=data['points']-np.array(CROP[:2]);regions=data['regions'];names=data['names'];D=data['D'];qr=regions<3
    ref=read(ROOT/'center.bmp')
    def job(i):
        name=names[i];path=ROOT/name if name=='center.bmp' else ROOT/('Y' if name[0]=='Y' else 'Theta')/name
        cur=read(path);init,_=coarse(ref,cur,False);sm=cv2.GaussianBlur(cur,(0,0),.65)
        distances=np.full(len(points),np.nan)
        for k in range(3):
            x0,y0,x1,y1=BOXES[k];dx,dy=init[k].astype(int);mask=np.zeros(cur.shape,np.uint8);margin=28
            mask[y0+margin-CROP[1]+dy:y1-margin-CROP[1]+dy,x0+margin-CROP[0]+dx:x1-margin-CROP[0]+dx]=255
            found=cv2.goodFeaturesToTrack(sm,600,.005,4,mask=mask,blockSize=5)
            found=cv2.cornerSubPix(sm,found,(2,2),(-1,-1),(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,.001)).reshape(-1,2)
            distances[regions==k]=cKDTree(found).query(points[regions==k]+D[i,regions==k])[0]
        return distances
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:distances=np.stack(list(ex.map(job,range(len(names)))))
    qd=distances[:,qr]
    result={}
    for tol in [1,2,3]:
        valid=qd<tol;result[str(tol)+'px']={'mean_redetection_fraction':float(valid.mean()),'minimum_per_image_fraction':float(valid.mean(axis=1).min()),'points_redetected_in_all_images':int(valid.all(axis=0).sum()),'points_total':int(qr.sum())}
    result['caveat']='Repeatability is nearest redetected corner distance to tracked position; it is not one-to-one matching nor independent motion ground truth. CornerSubPix shifts and ROI edges can alter detection identities.'
    result['tracking']='GN17 seeded from the same 928 center QR points remained valid in every image by convergence and forward-backward criteria; independent redetection does not guarantee the same point identities.'
    (OUT/'corner_repeatability.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    np.savez_compressed(OUT/'corner_redetection.npz',names=names,regions=regions,distance=distances)
    print(json.dumps(result),flush=True)
    # Spatial correlation of local residuals after a global affine displacement fit.
    X=np.c_[np.ones(qr.sum()),(points[qr]-points[qr].mean(axis=0))/100]
    pinv=np.linalg.pinv(X)
    affine=np.einsum('kn,mnc->mkc',pinv,D[:,qr]);resid=D[:,qr]-np.einsum('nk,mkc->mnc',X,affine)
    perpoint=np.sqrt(np.mean(resid**2,axis=(0,2)))
    perimage=np.sqrt(np.mean(resid**2,axis=(1,2)))
    residual_summary={'point_global_affine_residual_rms_px_median':float(np.median(perpoint)),'point_global_affine_residual_rms_px_p95':float(np.quantile(perpoint,.95)),'per_image_affine_residual_rms_px_median':float(np.median(perimage)),'per_image_affine_residual_rms_px_max':float(perimage.max()),'caveat':'Residual includes perspective/parallax/model mismatch and localization noise, and is not absolute motion error.'}
    (OUT/'affine_residual_summary.json').write_text(json.dumps(residual_summary,indent=2),encoding='utf8')
    with (OUT/'point_quality.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['point_index','region','x','y','strength','affine_residual_rms_px','redetection_fraction_1px','redetection_fraction_2px'])
        for i,j in enumerate(np.flatnonzero(qr)):w.writerow([j,regions[j],data['points'][j,0],data['points'][j,1],data['strengths'][j],perpoint[i],float((qd[:,i]<1).mean()),float((qd[:,i]<2).mean())])
    print(json.dumps(residual_summary),flush=True)

if __name__=='__main__':main()
