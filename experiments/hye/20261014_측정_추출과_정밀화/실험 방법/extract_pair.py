"""Extract paired displacements from two photos, without any motion labels.

Example (PowerShell with Python dependencies configured):
python extract_pair.py --center center.bmp --current T+0.01.bmp --out pair.npz

The QR coordinates are configured for this 3840x2160 fixed-camera dataset.
For a trained point-coordinate model, the center image must be the same
calibration reference; supply --features features.npz for exact feature order.
By default this API reads only the three QR templates and uses QR points.
"""
from pathlib import Path
import argparse,json,time
import cv2
import numpy as np
from measure import read,feature_set,coarse,track,CROP,VARIANTS
from refine import Refiner

class PairExtractor:
    def __init__(self,center,method='gn17',features=None,include_screws=False):
        self.center=Path(center);self.method=method;self.include_screws=include_screws
        self.reference=read(self.center)
        if features:
            f=np.load(features);self.points=(f['points']-np.array(CROP[:2])).astype(np.float32);self.regions=f['regions'];self.strengths=f['strengths']
        else:self.points,self.regions,self.strengths=feature_set(self.reference)
        sel=np.ones(len(self.points),bool) if include_screws else self.regions<3
        self.feature_indices=np.flatnonzero(sel)
        self.points=self.points[sel];self.regions=self.regions[sel];self.strengths=self.strengths[sel]
        if method in ['gn17','gn31']:
            self.window,self.sigma=31,.8
            self.refiner=Refiner(self.reference,self.points,int(method[2:]),.8)
        elif method in VARIANTS:self.window,self.sigma=VARIANTS[method]
        else:raise ValueError('Unknown method '+method)
        self.tracking_reference=cv2.GaussianBlur(self.reference,(0,0),self.sigma) if self.sigma else self.reference

    def extract(self,current):
        start=time.perf_counter();cur=read(Path(current))
        initial,scores=coarse(self.reference,cur,include_screws=self.include_screws)
        tracking_current=cv2.GaussianBlur(cur,(0,0),self.sigma) if self.sigma else cur
        D,quality,valid=track(self.tracking_reference,tracking_current,self.points,initial[self.regions],self.window)
        if self.method in ['gn17','gn31']:D,quality,valid_gn=self.refiner.run(cur,D);valid&=valid_gn
        return dict(D=D,quality=quality,valid=valid,points=self.points+np.array(CROP[:2]),regions=self.regions,strengths=self.strengths,feature_indices=self.feature_indices,coarse_score=scores,method=np.array(self.method),runtime_seconds=np.array(time.perf_counter()-start))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--center',required=True);p.add_argument('--current',required=True);p.add_argument('--out',required=True);p.add_argument('--method',default='gn17',choices=['gn17','gn31']+list(VARIANTS));p.add_argument('--features');p.add_argument('--include-screws',action='store_true');args=p.parse_args()
    extractor=PairExtractor(args.center,args.method,args.features,args.include_screws)
    result=extractor.extract(args.current);np.savez_compressed(args.out,**result)
    print(json.dumps({'output':str(Path(args.out).resolve()),'points':len(result['D']),'valid':int(result['valid'].sum()),'runtime_seconds':float(result['runtime_seconds']),'method':args.method}))

if __name__=='__main__':main()
