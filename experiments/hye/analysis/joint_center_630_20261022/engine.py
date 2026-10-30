"""Image-only 630-corner / discrete-center / simultaneous YT inference."""
from pathlib import Path
import json,time,hashlib
import cv2,numpy as np
from measurement import StructuralExtractor
from features import FeatureMaps
from model import JointOrigins
HERE=Path(__file__).resolve().parent
class Engine:
 def __init__(self):
  start=time.perf_counter();path=HERE.parent/'hw-analysis/data/hw/bmp/20261014/center.bmp';raw=path.read_bytes();provenance=json.loads((HERE/'assets/provenance.json').read_text());assert hashlib.sha256(raw).hexdigest()==provenance['center_sha256']
  self.old_params=json.loads((HERE/'assets/model.json').read_text());self.oldW=np.array(self.old_params['W']);self.center=cv2.imdecode(np.frombuffer(raw,np.uint8),0);self.extractor=StructuralExtractor(self.center);self.maps=FeatureMaps(self.extractor.points,self.extractor.regions);self.model=JointOrigins.load(HERE/'model.npz');cv2.setNumThreads(1);self.startup_seconds=time.perf_counter()-start
 def old_predict(self,X):
  choices=[]
  for axis,b,s,c in self.old_params['branches'].values():
   z=self.oldW@(X-np.array(b));s=self.oldW@np.array(s);c=self.oldW@np.array(c);v=float(z@s/(s@s))
   for _ in range(8):
    j=s+2*v*c;v+=float(j@(z-v*s-v*v*c)/max(j@j,1e-20))
   choices.append((float(np.sum((z-v*s-v*v*c)**2)),axis,v))
  _,axis,v=min(choices);return [v,0.] if axis=='Y' else [0.,v]
 def predict(self,raw,details=False,compare_baseline=False):
  start=time.perf_counter();img=cv2.imdecode(np.frombuffer(raw,np.uint8),0);decoded=time.perf_counter()
  if img is None or img.shape!=self.center.shape:raise ValueError('기준과 동일한 3840x2160 원본 이미지가 필요합니다.')
  r=self.extractor.extract(img);measured=time.perf_counter()
  if not np.isfinite(r['D']).all():raise ValueError('630점 중 유한하지 않은 측정값이 있습니다.')
  X=self.maps.transform(r['D'],self.model.feature_name);aggregated=time.perf_counter();p=self.model.predict(X);finished=time.perf_counter();pose=p['YT'][0];alt=p['alternative_YT'][0];gap=float(p['margin'][0]);ambiguous=gap<.05 and np.max(abs(pose-alt))>.09
  out=dict(Y=float(pose[0]),T=float(pose[1]),center_id=p['center_id'][0],center_index=int(p['center_index'][0]),center_candidates=len(self.model.centers),cost=float(p['cost'][0]),second_cost=float(p['second_cost'][0]),cost_margin=gap,alternative_YT=alt.tolist(),ambiguous=bool(ambiguous),valid=int(r['valid'].sum()),points=630,pad=r['used_pad'],strict_initialization_ok=min(r['strict_scores'])>=.6,feature=self.model.feature_name,model='joint_center_630',T_unit='unconfirmed',result_cache=False,timings=dict(decode=decoded-start,measurement=measured-decoded,aggregation=aggregated-measured,center_and_joint_fit=finished-aggregated,total=finished-start),scope='single and mixed Y/T command estimation; performance is experimental')
  if compare_baseline:
   t=time.perf_counter();oldX=self.maps.transform(r['D'],'P12');out['baseline_YT']=self.old_predict(oldX);out['baseline_aggregation_and_model_seconds']=time.perf_counter()-t;out['baseline_shared_measurement_total_seconds']=(measured-start)+out['baseline_aggregation_and_model_seconds']
  if details:out['candidates']=[dict(id=self.model.origin_ids[k],Y=float(p['all_YT'][0,k,0]),T=float(p['all_YT'][0,k,1]),cost=float(p['all_costs'][0,k])) for k in np.argsort(p['all_costs'][0])]
  return out
if __name__=='__main__':
 import argparse
 from threadpoolctl import threadpool_limits
 parser=argparse.ArgumentParser();parser.add_argument('--image',required=True);parser.add_argument('--out');args=parser.parse_args();e=Engine()
 with threadpool_limits(1):out=e.predict(Path(args.image).read_bytes(),True)
 out['startup_seconds']=e.startup_seconds;text=json.dumps(out,ensure_ascii=False,indent=2)
 if args.out:Path(args.out).write_text(text,encoding='utf-8')
 print(text)
