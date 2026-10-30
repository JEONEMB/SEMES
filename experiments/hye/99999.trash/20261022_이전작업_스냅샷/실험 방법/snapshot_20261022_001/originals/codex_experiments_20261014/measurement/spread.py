"""Wide spatial QR features with strict per-QR local image support.

Both LK initialization and GN use a cropped QR-panel image. No screw, plate,
or other QR pixels are available to a QR's refiner. Coarse full-QR matching
sets the integer crop displacement. The supported margin is 16 pixels.
"""
import argparse,concurrent.futures,json,time
from pathlib import Path
import cv2,numpy as np
from measure import ROOT,OUT,CROP,BOXES,read,coarse,track
from refine import Refiner

cv2.setNumThreads(1)

def select(ref,min_distance):
    sm=cv2.GaussianBlur(ref,(0,0),.65);eig=cv2.cornerMinEigenVal(sm,5);points=[];regions=[];strengths=[]
    for k,(x0,y0,x1,y1) in enumerate(BOXES):
        mask=np.zeros(ref.shape,np.uint8);margin=16
        mask[y0+margin-CROP[1]:y1-margin-CROP[1],x0+margin-CROP[0]:x1-margin-CROP[0]]=255
        p=cv2.goodFeaturesToTrack(sm,600,.02,min_distance,mask=mask,blockSize=5)
        p=cv2.cornerSubPix(sm,p,(2,2),(-1,-1),(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,.001)).reshape(-1,2)
        for q in p:
            qg=q+np.array(CROP[:2]);
            if not (x0+15<=qg[0]<x1-15 and y0+15<=qg[1]<y1-15):continue
            points.append(q);regions.append(k);strengths.append(eig[round(float(q[1])),round(float(q[0]))])
    return np.array(points,np.float32),np.array(regions,np.int16),np.array(strengths,np.float32)

class SpreadPairExtractor:
    """Reusable pair extractor matching gn17_spread and gn17_spread14 batches."""
    def __init__(self,center,features=None,min_distance=17):
        self.ref=read(Path(center));self.min_distance=min_distance
        if features:
            data=np.load(features);self.points=(data['points']-np.array(CROP[:2])).astype(np.float32);self.regions=data['regions'];self.strengths=data['strengths']
        else:self.points,self.regions,self.strengths=select(self.ref,min_distance)
        self.models=[];self.templates=[];self.local_points=[];self.local_boxes=[]
        for k,(x0,y0,x1,y1) in enumerate(BOXES):
            x0-=CROP[0];x1-=CROP[0];y0-=CROP[1];y1-=CROP[1]
            template=self.ref[y0:y1,x0:x1].copy();pts=self.points[self.regions==k]-np.array([x0,y0],np.float32)
            self.models.append(Refiner(template,pts,17,.8));self.templates.append(cv2.GaussianBlur(template,(0,0),.8));self.local_points.append(pts);self.local_boxes.append((x0,y0,x1,y1))

    def extract(self,current):
        start=time.perf_counter();cur=read(Path(current));initial,scores=coarse(self.ref,cur,include_screws=False)
        D=np.zeros((len(self.points),2),np.float32);quality=np.zeros((len(self.points),5),np.float32);valid=np.zeros(len(self.points),bool)
        for k,(x0,y0,x1,y1) in enumerate(self.local_boxes):
            dx,dy=initial[k].astype(int);target=cur[y0+dy:y1+dy,x0+dx:x1+dx].copy();sel=self.regions==k;pts=self.local_points[k]
            d,lq,lv=track(self.templates[k],cv2.GaussianBlur(target,(0,0),.8),pts,np.zeros_like(pts),15)
            gd,gq,gv=self.models[k].run(target,d)
            D[sel]=gd+initial[k];quality[sel,:4]=gq;quality[sel,4]=lq[:,0];valid[sel]=lv&gv
        return dict(D=D,quality=quality,valid=valid,points=self.points+np.array(CROP[:2]),regions=self.regions,strengths=self.strengths,coarse_score=scores,runtime_seconds=np.array(time.perf_counter()-start))

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--center');ap.add_argument('--current');ap.add_argument('--features');ap.add_argument('--out');ap.add_argument('--min-distance',type=int,default=17);args=ap.parse_args()
    if args.current:
        model=SpreadPairExtractor(args.center,args.features,args.min_distance);res=model.extract(args.current);np.savez_compressed(args.out,**res);print('saved',args.out,'runtime',res['runtime_seconds']);return
    paths=[ROOT/'center.bmp']+sorted((ROOT/'Theta').glob('*.bmp'))+sorted((ROOT/'Y').glob('*.bmp'))
    for md,tag in [(17,'gn17_spread'),(14,'gn17_spread14')]:
        model=SpreadPairExtractor(ROOT/'center.bmp',min_distance=md);n=len(model.points);start=time.time()
        np.savez_compressed(OUT/(tag+'_features.npz'),points=model.points+np.array(CROP[:2]),regions=model.regions,strengths=model.strengths)
        results=[]
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
            for i,result in enumerate(ex.map(model.extract,paths)):
                results.append(result)
                if i%40==0:print(tag,i+1,round(time.time()-start,1),flush=True)
        np.savez_compressed(OUT/(tag+'.npz'),names=np.array([p.name for p in paths]),points=model.points+np.array(CROP[:2]),regions=model.regions,strengths=model.strengths,D=np.stack([r['D'] for r in results]),quality=np.stack([r['quality'] for r in results]),valid=np.stack([r['valid'] for r in results]),coarse_score=np.stack([r['coarse_score'] for r in results]))
        canvas=cv2.cvtColor(model.ref,cv2.COLOR_GRAY2BGR)
        for p,k in zip(model.points,model.regions):cv2.circle(canvas,tuple(np.rint(p).astype(int)),2,[(0,0,255),(255,0,0),(0,150,0)][k],-1)
        cv2.imwrite(str(OUT/(tag+'_features.png')),canvas)
        coverage=[]
        for k in range(3):
            p=model.points[model.regions==k]+np.array(CROP[:2]);coverage.append({'region':k,'count':len(p),'min_xy':p.min(axis=0).tolist(),'max_xy':p.max(axis=0).tolist()})
        meta={'min_distance':md,'margin_detection_px':16,'minimum_refined_point_margin_px':15,'feature_quality_level':.02,'features':n,'regions':coverage,'support':'Each QR separately cropped before blur, LK15 initialization, cubic spline filtering, gradients, and GN17. No pixels outside that QR-panel crop used for the final refinement. Full-QR NCC initialization.','quality_columns':['photometric_rmse_gray','last_step_px','gain','offset_gray','LK_forward_backward_error_px'],'valid_fraction':float(np.stack([r['valid'] for r in results]).mean()),'runtime_seconds':time.time()-start}
        (OUT/(tag+'_metadata.json')).write_text(json.dumps(meta,indent=2),encoding='utf8');print(meta,flush=True)

if __name__=='__main__':main()
