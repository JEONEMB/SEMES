"""Frozen candidate inference from two image paths; no filename labels are parsed."""
from pathlib import Path
import argparse,json,sys,time
import numpy as np
import cv2
from scipy import ndimage
import experiment as E

class PairExtractor:
 def __init__(self,center,params=None):
  self.center=Path(center)
  expected=json.loads((E.HERE/'baseline_snapshot/reference.json').read_text(encoding='utf8'))['sha256']
  if E.sha(self.center)!=expected:raise ValueError('A different center image requires recalibration.')
  self.params=params or json.loads((E.HERE/'weighted_model_full280.json').read_text(encoding='utf8'))
  self.ref=E.read(self.center)
  features=np.load(E.HERE/'baseline_snapshot/measurement/gn17_spread14_features.npz')
  self.points=features['points'];self.regions=features['regions'];self.boxes=[];self.templates=[];self.local_points=[];self.models=[]
  for k,(x0,y0,x1,y1) in enumerate(E.BOXES):
   box=(x0-E.CROP[0],y0-E.CROP[1],x1-E.CROP[0],y1-E.CROP[1]);self.boxes.append(box)
   lx0,ly0,lx1,ly1=box;template=self.ref[ly0:ly1,lx0:lx1].copy()
   points=(self.points[self.regions==k]-[x0,y0]).astype(np.float32)
   model=E.WeightedRefiner(template,points);model.configure(self.params['noise_model'])
   self.models.append(model);self.local_points.append(points);self.templates.append(cv2.GaussianBlur(template,(0,0),.8))
  self.zero=self.extract(self.center)['D']

 def extract(self,current):
  cur=E.read(Path(current));initial,scores=E.coarse(self.ref,cur,include_screws=False)
  D=np.zeros((205,2),np.float32);valid=np.zeros(205,bool)
  for k,(x0,y0,x1,y1) in enumerate(self.boxes):
   dx,dy=initial[k].astype(int);target=cur[y0+dy:y1+dy,x0+dx:x1+dx].copy();sel=self.regions==k
   d,lq,lv=E.track(self.templates[k],cv2.GaussianBlur(target,(0,0),.8),self.local_points[k],np.zeros_like(self.local_points[k]),15)
   old_d,old_q,old_valid=self.models[k].old.run(target,d)
   coeff=ndimage.spline_filter(cv2.GaussianBlur(target.astype(float),(0,0),.8),order=3)
   gd,gq,gv,_,_=self.models[k].run(coeff,d,old_q[:,2:4])
   D[sel]=gd+initial[k];valid[sel]=gv&lv
  return dict(D=D,valid=valid)

 def predict(self,current):
  t=time.perf_counter();r=self.extract(current);valid=r['valid']
  if valid.mean()<.95 or any((valid&(self.regions==k)).sum()<20 for k in range(3)):
   return dict(status='rejected',valid_points=int(valid.sum()),Y=0.,T=0.)
  p=self.points[valid];x,y=(p[:,0]-1906)/486,(p[:,1]-1070)/122
  B=np.array([x*0+1,x,y,x*x,x*y,y*y]).T
  X=np.linalg.lstsq(B,(r['D']-self.zero)[valid].astype(float),rcond=None)[0].ravel()
  # The solver consumes only the measured feature vector and frozen parameters.
  import infer
  result=infer.solve_calibrated(X,self.params)
  axis=result['axis'];value=float(result['value'])
  return dict(status='prediction',axis=axis,sign=int(np.sign(value)),value=value,Y=value if axis=='Y' else 0.,T=value if axis=='T' else 0.,
   branch=result['calibration_mode'],valid_points=int(valid.sum()),runtime_seconds=time.perf_counter()-t,
   confidence_probability=None,actual_error=None,
   scope='Frozen all-280 development candidate; same camera/reference and six calibrated origin modes; no independent new-acquisition validation')

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--center',required=True);p.add_argument('--current',required=True);args=p.parse_args()
 print(json.dumps(PairExtractor(args.center).predict(args.current),ensure_ascii=False,indent=2))
