from pathlib import Path
import json,time,cv2
import pandas as pd,numpy as np
from threadpoolctl import threadpool_limits
from engine import Engine,HERE
from evaluate import frozen
if __name__=='__main__':
 e=Engine();m=pd.read_csv(HERE/'results/manifest.csv');a=np.load(HERE/'results/measurements.npz');expected=e.model.predict(e.maps.transform(a['D'],e.model.feature_name))['YT'];rows=[]
 with threadpool_limits(1):
  for i,row in m.iterrows():
   r=e.predict(Path(row.path).read_bytes(),compare_baseline=True);assert abs(r['Y']-expected[i,0])<1e-7 and abs(r['T']-expected[i,1])<1e-7
   r.update(index=i,file=row.file,Y_truth=row.Y,T_truth=row['T'],eY=abs(r['Y']-row.Y),eT=abs(r['T']-row['T']));rows.append(r)
   if i%60==0:print('end-to-end',i+1,'latency',round(r['timings']['total'],4),flush=True)
  for bad in [b'bad',cv2.imencode('.png',np.zeros((30,30),np.uint8))[1].tobytes()]:
   try:e.predict(bad)
   except ValueError:pass
   else:raise AssertionError('Bad input accepted')
 values=[r['timings']['total'] for r in rows];result=dict(n=len(rows),startup_seconds=e.startup_seconds,latency=dict(median=float(np.median(values)),p95=float(np.quantile(values,.95)),maximum=float(np.max(values))),stage_median={key:float(np.median([r['timings'][key] for r in rows])) for key in rows[0]['timings']},candidates=len(e.model.centers),feature=e.model.feature_name,strict_initialization_rescued=sum(not r['strict_initialization_ok'] for r in rows),result_cache=False,all_outputs_match_feature_evaluation=True,baseline_common_measurement_total_median=float(np.median([r['baseline_shared_measurement_total_seconds'] for r in rows if r['strict_initialization_ok']])),new_common_measurement_total_median=float(np.median([r['timings']['total'] for r in rows if r['strict_initialization_ok']])),baseline_aggregation_and_model_median=float(np.median([r['baseline_aggregation_and_model_seconds'] for r in rows])))
 (HERE/'results/local_surfaces/end_to_end.json').write_text(json.dumps(dict(summary=result,rows=rows),indent=2));print(json.dumps(result),flush=True)
