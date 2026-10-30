from pathlib import Path
import json,inspect
import numpy as np,pandas as pd
from model import JointOrigins
from features import FeatureMaps
HERE=Path(__file__).resolve().parent;OUT=HERE/'results/local_surfaces'
if __name__=='__main__':
 m=pd.read_csv(HERE/'results/manifest.csv');a=np.load(HERE/'results/measurements.npz');assert a['D'].shape==(341,630,2) and len(set(a['ids']))==630 and np.isfinite(a['D']).all();splits=json.loads((OUT/'splits.json').read_text());maps=FeatureMaps(a['points'],a['regions']);models=[]
 for f,s in enumerate(splits):
  tr=set(s['train']);te=set(s['test']);assert not tr&te;model=JointOrigins.load(OUT/f'fold{f}_model.npz');assert set(m.relative.iloc[model.training_indices])==tr
  for name in model.origin_ids:
   if name.startswith('actual:'):assert name.split(':',1)[1] in tr and name.split(':',1)[1] not in te
  for _,g in m[m.scope=='mixed'].groupby('condition'):assert len(set(g.relative)&te) in (0,len(g))
  ix=np.array([i for i,r in enumerate(m.relative) if r in te]);p=model.predict(maps.transform(a['D'][ix],model.feature_name));saved=pd.read_csv(OUT/'predictions.csv');saved=saved[(saved.method=='multi_center_joint')&(saved.protocol=='oof4')&(saved.fold==f)].sort_values('index');assert np.max(abs(saved[['predY','predT']].to_numpy()-p['YT']))<1e-7
  assert list(inspect.signature(model.predict).parameters)==['X'];models.append(dict(fold=f,training=len(tr),test=len(te),centers=len(model.origin_ids)))
 rng=np.random.default_rng(43);n=240;truth=rng.uniform(-1,1,(n,2));groups=np.where(np.arange(n)%2,'A','B');scopes=np.full(n,'mixed');offset=np.where(groups=='A',0.,1.)
 X=np.c_[truth[:,0],truth[:,1],offset*2,np.zeros(n),truth[:,0]*.1+truth[:,1]*.2];idx=np.arange(n);model=JointOrigins(1,1.,'learned').fit(X,truth,groups,scopes,idx,np.array([str(i) for i in idx]));target=np.array([[.4,-.2],[-.6,.7]]);test=np.c_[target[:,0],target[:,1],[0.,2.],np.zeros(2),target[:,0]*.1+target[:,1]*.2];pred=model.predict(test);assert np.max(abs(pred['YT']-target))<1e-5
 report=dict(passed=True,fixed_ids=630,images=341,measurement_failures=int((~m.success).sum()),expanded_search_rescues=int((~m.strict_success).sum()),fold_checks=models,heldout_actual_centers_excluded=True,mixed_repetitions_kept_together=True,no_runtime_labels_or_groups=True,saved_models_reproduce_oof=True,synthetic_identifiable_center_and_joint_pose_recovered=True)
 (OUT/'checks.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
