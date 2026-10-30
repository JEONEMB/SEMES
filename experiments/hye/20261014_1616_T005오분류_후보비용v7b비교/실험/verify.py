"""Independent scoring, source fidelity and candidate-cost consistency checks."""
import json
import numpy as np
import pandas as pd
import experiment as N

def main():
 H=N.HERE;pred=pd.read_csv(H/'predictions.csv');metrics=pd.read_csv(H/'metrics.csv')
 counts=[]
 for (method,protocol),g in pred.groupby(['method','protocol']):
  assert len(g)==280 and g.file.nunique()==280
  truth=np.c_[np.where(g.axis=='Y',g.value,0),np.where(g.axis=='T',g.value,0)]
  output=np.c_[np.where(g.pred_axis=='Y',g.pred,0),np.where(g.pred_axis=='T',g.pred,0)]
  errors=np.abs(output-truth);axis=g.axis.to_numpy()==g.pred_axis.to_numpy();sign=np.sign(g.value.to_numpy())==np.sign(g.pred.to_numpy());ok=axis&sign
  row=metrics.query('method==@method and protocol==@protocol').iloc[0]
  assert abs(row.axis_accuracy-axis.mean())<1e-12 and abs(row.axis_sign_accuracy-ok.mean())<1e-12
  assert abs(row.vector_Y_max-errors[:,0].max())<1e-12 and abs(row.vector_T_max-errors[:,1].max())<1e-12
  assert int(row.failed)==np.sum(~ok|(errors>.09).any(1))
  assert np.array_equal(g.pass09.to_numpy(),ok&(errors<=.09).all(1))
  for a in ['Y','T']:
   sel=g.axis.to_numpy()==a
   assert abs(row[a+'_axis_accuracy']-axis[sel].mean())<1e-12 and abs(row[a+'_axis_sign_accuracy']-ok[sel].mean())<1e-12
  counts.append((method,protocol))
 splits=json.loads((H/'splits.json').read_text());assert splits==json.loads((N.PREV/'splits.json').read_text())
 for s in splits:
  done=json.loads((H/'folds'/(s['protocol']+'__'+str(s['fold']))/'complete.json').read_text())
  assert done['training_files']==s['train'] and done['test_files']==s['test']
  curve=json.loads((N.PREV/'folds'/(s['protocol']+'__'+str(s['fold']))/'noise_model.json').read_text(encoding='utf8'))
  assert done['measurement_cache']==N.curve_key(curve)+'.npz'
  assert all(r['a'] in s['train'] and r['b'] in s['train'] for r in curve['pairs'])
 old=pd.read_csv(N.PREV/'predictions.csv');differences=[]
 for new,previous in [('gn_six','baseline'),('pixel_six','pixel_noise_gn')]:
  merged=pred[pred.method==new].merge(old[old.method==previous],on=['protocol','file'],suffixes=('_new','_old'))
  assert len(merged)==7000 and (merged.pred_axis_new==merged.pred_axis_old).all()
  delta=float(np.abs(merged.pred_new-merged.pred_old).max());assert delta<1e-7;differences.append(delta)
 target=pd.read_csv(H/'T005_predictions_and_costs.csv',dtype={'fold':str});candidate=pd.read_csv(H/'T005_candidate_costs.csv',dtype={'fold':str})
 maxcost=0.
 for _,r in target.iterrows():
  c=candidate[(candidate.protocol==r.protocol)&(candidate.fold==r.fold)&(candidate.method==r.method)]
  c=c[c.allowed];winner=c.loc[c.cost.idxmin()]
  assert winner.axis==r.pred_axis
  if r.method.endswith('six'):assert abs(winner.value-r.pred)<1e-7
  source=np.load(H/'folds'/(r.protocol+'__'+r.fold)/(r.method+'_target.npz'))
  for i in np.flatnonzero(source['allowed']):
   residual=source['W']@(source['X']-source['candidate_features'][i]);cost=float(residual@residual)
   diff=abs(cost-source['costs'][i]);maxcost=max(maxcost,diff);assert diff<1e-5
 fid=json.loads((H/'source_fidelity_checks.json').read_text());assert len(fid)==4
 assert all(r['value_max_abs_difference']<1e-7 and r['cost_max_abs_difference']<1e-5 for r in fid)
 ledger=json.loads((H/'preserved_files.json').read_text(encoding='utf8'));assert all(N.sha(N.BASE/n)==v for n,v in ledger.items())
 originals=json.loads((N.PREV/'cache/prepared.json').read_text())['image_hashes'];assert all(N.sha(N.E.path_for(n))==d for n,d in originals.items())
 result=dict(prediction_rows=len(pred),method_protocol_groups=len(counts),exact_previous_splits=True,training_only_noise_provenance_all_119=True,
  previous_methods_prediction_max_abs_difference=max(differences),target_candidate_groups=len(target),target_candidate_rows=len(candidate),
  candidate_cost_reconstruction_max_abs_difference=maxcost,source_class_comparisons=sum(r['comparisons'] for r in fid),
  preserved_existing_files=len(ledger),existing_files_unchanged=True,original_281_images_unchanged=True,corrected_labels_executed=False,
  original_v7b_source_isolated_class_only=True,new_acquisition_independent_test=False)
 N.dump(H/'audit_checks.json',result);print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
