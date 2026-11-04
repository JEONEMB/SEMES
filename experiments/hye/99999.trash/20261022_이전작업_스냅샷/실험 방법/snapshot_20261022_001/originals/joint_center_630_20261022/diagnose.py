from pathlib import Path
import json,numpy as np,pandas as pd
from threadpoolctl import threadpool_limits
from model import JointOrigins
from features import FeatureMaps
from evaluate import make_rows,metrics
HERE=Path(__file__).resolve().parent;OUT=HERE/'results/local_surfaces';m=pd.read_csv(HERE/'results/manifest.csv');a=np.load(HERE/'results/measurements.npz');maps=FeatureMaps(a['points'],a['regions']);bank={n:maps.transform(a['D'],n) for n in ['P12','QR18','QR36','raw1260']};splits=json.loads((OUT/'splits.json').read_text());rows=[]
with threadpool_limits(1):
 for f,split in enumerate(splits):
  model=JointOrigins.load(OUT/f'fold{f}_model.npz');ix=np.array([i for i,r in enumerate(m.relative) if r in set(split['test'])]);pred=model.predict(bank[model.feature_name][ix]);oracle=[]
  for j,i in enumerate(ix):
   key='learned:'+m.group.iloc[i]
   if m.scope.iloc[i]!='center' and key in model.origin_ids:
    k=model.origin_ids.index(key);oracle.append(dict(index=i,Y=m.Y.iloc[i],T=m['T'].iloc[i],predY=pred['all_YT'][j,k,0],predT=pred['all_YT'][j,k,1]))
  rows+=oracle
 oracle=pd.DataFrame(rows);oracle.to_csv(OUT/'diagnostic_oracle_group.csv',index=False)
 final=JointOrigins.load(HERE/'model.npz');L=final.Aw[:2];cos=abs(float(L[0]@L[1]/(np.linalg.norm(L[0])*np.linalg.norm(L[1]))));origin=[]
 for i,name in enumerate(final.origin_ids):
  delta=final.cw[i]-final.cw[0];equiv=delta@final.linear_pinv;unexplained=np.linalg.norm(delta-equiv@L)/max(np.linalg.norm(delta),1e-12)
  origin.append(dict(center=name,equivalent_Y=float(equiv[0]*final.scale[0]),equivalent_T=float(equiv[1]*final.scale[1]),unexplained_fraction=float(unexplained)))
 p=pd.read_csv(OUT/'predictions.csv');p=p[(p.method=='multi_center_joint')&(p.protocol=='oof4')];posegap=np.maximum(abs(p.predY-p.alternativeY),abs(p.predT-p.alternativeT));ambiguous=(p.margin<.05)&(posegap>.09)
 diag=dict(origin_count=len(final.centers),origin_ids=final.origin_ids,linear_response_absolute_cosine=cos,linear_response_condition=float(np.linalg.cond(L.T)),origin_motion_overlap=origin,normalized_margin_median=float(p.margin.median()),ambiguous_rows=int(ambiguous.sum()),ambiguity_rule='best-vs-second normalized cost margin <0.05 and alternative pose component gap >0.09; diagnostic, not a probability',oracle_training_group_diagnostic=metrics(oracle),oracle_note='Uses heldout group ONLY for this diagnostic counterfactual. Not operational inference, not reported as achievable evaluation score.')
 (OUT/'identifiability.json').write_text(json.dumps(diag,indent=2));print(json.dumps(diag),flush=True)
