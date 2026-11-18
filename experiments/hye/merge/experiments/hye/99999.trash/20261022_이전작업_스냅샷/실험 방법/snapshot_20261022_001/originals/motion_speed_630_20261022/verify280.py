"""Fresh images; frozen baseline measurements are comparison only, never inference input."""
from pathlib import Path
import json,time,platform,sys
import pandas as pd,numpy as np
from threadpoolctl import threadpool_limits,threadpool_info
from engine import Engine,ROOT,HERE

def evaluate(name,r):
 axis=name[0];value=float(name[1:-4]);Y=value if axis=='Y' else 0.;T=value if axis=='T' else 0.
 eY=abs(r['Y']-Y);eT=abs(r['T']-T);correct=axis==r['axis'] and np.sign(value)==np.sign(r['value'])
 return dict(file=name,axis=axis,value=value,pred_axis=r['axis'],pred=r['value'],correct=bool(correct),eY=eY,eT=eT,passed=bool(correct and eY<=.09 and eT<=.09),seconds=r['timings']['total'],valid=r['valid'])
if __name__=='__main__':
 engine=Engine('compiled');baseline=pd.read_csv(HERE.parent/'motion_web_20261022/verification_630/recomputed280.csv').set_index('file');corners=pd.read_csv(HERE.parent/'analysis_grid_corners_20261019/out/corners_280.csv');rows=[];differences=[];diagnostic_changes=0;start=time.perf_counter()
 with threadpool_limits(1):
  measured_threads=threadpool_info()
  for i,name in enumerate(sorted(baseline.index)):
   path=ROOT/('Y' if name[0]=='Y' else 'Theta')/name
   r=engine.predict(path.read_bytes(),True);m=r.pop('measurement');old=baseline.loc[name];expected=corners[corners.file==name].set_index('id').loc[m['ids']]
   delta=np.linalg.norm(m['current_points']-expected[['x','y']].to_numpy(),axis=1);differences.extend(delta.tolist());diagnostic_changes+=int(np.sum(m['valid']!=expected.valid.to_numpy()))
   row=evaluate(name,r);row.update(baseline_axis=old.pred_axis,baseline_pred=old.pred,prediction_change=abs(r['value']-old.pred) if r['axis']==old.pred_axis else None,coordinate_max=float(delta.max()),valid_mask_changes=int(np.sum(m['valid']!=expected.valid.to_numpy())),axis_changed=r['axis']!=old.pred_axis)
   rows.append(row)
   if i%20==0:print('images',i+1,'elapsed',round(time.perf_counter()-start,1),'axis changes',sum(a['axis_changed'] for a in rows),flush=True)
 frame=pd.DataFrame(rows);frame.to_csv(HERE/'results/compiled280.csv',index=False);frame[~frame.passed].to_csv(HERE/'results/compiled_failures.csv',index=False)
 summary=dict(n=len(frame),protocol='frozen full280 model resubstitution, fresh images; no refit and no new OOF',passed=int(frame.passed.sum()),axis_sign_correct=int(frame.correct.sum()),baseline_passed=int(baseline.passed.sum()),axis_changed=int(frame.axis_changed.sum()),Y_max=float(frame.eY.max()),T_max=float(frame.eT.max()),baseline_Y_max=float(baseline.eY.max()),baseline_T_max=float(baseline.eT.max()),coordinate_difference_px=dict(median=float(np.median(differences)),p95=float(np.quantile(differences,.95)),max=float(np.max(differences))),diagnostic_validity_changes=diagnostic_changes,prediction_difference_max=float(frame.prediction_change.max()),latency_seconds=dict(median=float(frame.seconds.median()),p95=float(frame.seconds.quantile(.95)),max=float(frame.seconds.max()),min=float(frame.seconds.min())),init_seconds=engine.init_seconds,total_wall=time.perf_counter()-start,cpu=platform.processor(),python=sys.version,threads=measured_threads)
 (HERE/'results/compiled280_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
