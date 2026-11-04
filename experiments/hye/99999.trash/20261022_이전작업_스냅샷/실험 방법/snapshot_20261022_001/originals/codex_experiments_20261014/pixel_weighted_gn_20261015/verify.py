"""Independent all-row scoring, split/fit isolation and raw two-image checks."""
import json,shutil,sys
import numpy as np
import pandas as pd
import experiment as E
sys.path.insert(0,str(E.HERE))
from predict import PairExtractor

def main():
 here=E.HERE;z,m,_=E.load_data()
 # An independent null experiment: constant variances must recover unweighted GN.
 init=np.load(here/'cache/initialization.npz');ref=E.read(E.ROOT/'center.bmp');maximum=0.
 test_names=['center.bmp','Y+0.05.bmp','T-0.01.bmp','Y-2.00.bmp']
 for k,(x0,y0,x1,y1) in enumerate(E.BOXES):
  sel=z['regions']==k;template=ref[y0-E.CROP[1]:y1-E.CROP[1],x0-E.CROP[0]:x1-E.CROP[0]]
  mod=E.WeightedRefiner(template,(z['points'][sel]-[x0,y0]).astype(np.float32))
  mod.configure({'centers':list(range(8,256,16)),'sd':[1.]*16})
  coeff=np.load(here/f'cache/coeff{k}.npy',mmap_mode='r')
  for name in test_names:
   i=list(z['names']).index(name)
   d,q,v,lo,hi=mod.run(coeff[i],init['initial'][i,sel],z['quality'][i,sel,2:4])
   delta=float(np.max(np.abs(d+init['coarse'][i,k]-z['D'][i,sel])));maximum=max(maximum,delta)
   assert delta<2e-6 and abs(lo-1)<1e-10 and abs(hi-1)<1e-10 and v.all()
 E.dump(here/'constant_variance_check.json',dict(images=test_names,QRs=3,max_abs_coordinate_difference_px=maximum,weights_all_one=True))
 pred=pd.read_csv(here/'predictions.csv');metrics=pd.read_csv(here/'metrics.csv')
 recalculated=[]
 for (method,protocol),g in pred.groupby(['method','protocol']):
  assert len(g)==280 and g.file.nunique()==280
  t=np.c_[np.where(g.axis=='Y',g.value,0),np.where(g.axis=='T',g.value,0)]
  p=np.c_[np.where(g.pred_axis=='Y',g.pred,0),np.where(g.pred_axis=='T',g.pred,0)]
  error=np.abs(t-p);axis=g.axis.to_numpy()==g.pred_axis.to_numpy()
  correct=axis&(np.sign(g.value.to_numpy())==np.sign(g.pred.to_numpy()))
  row=metrics.query('method==@method and protocol==@protocol').iloc[0]
  assert abs(row.vector_Y_max-error[:,0].max())<1e-12
  assert abs(row.vector_T_max-error[:,1].max())<1e-12
  assert abs(row.axis_sign_accuracy-correct.mean())<1e-12
  assert row.failed==np.sum(~correct|(error>.09).any(1)|g.rejected.to_numpy())
  assert np.allclose(g.error_Y,error[:,0]) and np.allclose(g.error_T,error[:,1])
  for a in ['Y','T']:
   sel=g.axis.to_numpy()==a
   assert abs(row[a+'_axis_accuracy']-axis[sel].mean())<1e-12
   assert abs(row[a+'_axis_sign_accuracy']-correct[sel].mean())<1e-12
  recalculated.append(dict(method=method,protocol=protocol,n=len(g),correct=int(correct.sum()),failed=int(row.failed)))
 splits=json.loads((here/'splits.json').read_text())
 original=json.loads((E.BASE/'gls_comparison_20261014/splits.json').read_text())
 assert splits==original
 pair_checks=0
 for s in splits:
  path=here/'folds'/(s['protocol']+'__'+str(s['fold']))/'noise_model.json'
  curve=json.loads(path.read_text(encoding='utf8'))
  assert curve['training_files']==s['train']
  tr=set(s['train'])
  assert all(p['a'] in tr and p['b'] in tr for p in curve['pairs'])
  pair_checks+=len(curve['pairs'])
 # Guard the raw pixel reader: the calibration fit must not even access a test row.
 target=next(s for s in splits if s['protocol']=='stratified5')
 index={n:i for i,n in enumerate(m.file)};tr=np.array([index[n] for n in target['train']]);te=np.array([index[n] for n in target['test']])
 raw=[np.load(here/f'cache/raw{k}.npy',mmap_mode='r') for k in range(3)]
 class Guard:
  def __init__(self,array):self.array=array
  def __getitem__(self,i):
   assert i in set(tr), 'Held-out raw pixel access in noise estimation'
   return self.array[i]
 init=np.load(here/'cache/initialization.npz')
 curve=E.estimate_noise([Guard(a) for a in raw],m,z['D'],init['coarse'],tr)
 expected=json.loads((here/'folds/stratified5__f0/noise_model.json').read_text(encoding='utf8'))
 assert curve==expected
 # Frozen candidate must produce the same coordinates from anonymously named raw files.
 checks=here/'raw_checks';checks.mkdir(exist_ok=True)
 center=checks/'reference.bmp';shutil.copy2(E.ROOT/'center.bmp',center)
 extractor=PairExtractor(center);weighted=np.load(here/'weighted_measurements_full280.npz')
 frames=pred.query("method=='pixel_noise_gn' and protocol=='full280'").set_index('file')
 details=[]
 for k,name in enumerate(['Y+0.05.bmp','T-0.01.bmp','Y-2.00.bmp','T+1.00.bmp']):
  path=checks/f'image_{k}.bmp';shutil.copy2(E.path_for(name),path)
  r=extractor.extract(path);i=list(z['names']).index(name)
  delta=float(np.max(np.abs(r['D']-weighted['D'][i])));assert delta<2e-6
  out=extractor.predict(path);old=frames.loc[name]
  assert out['axis']==old.pred_axis and abs(out['value']-old.pred)<1e-7
  details.append(dict(source=name,anonymous=path.name,coordinate_difference_px=delta,prediction_difference=float(abs(out['value']-old.pred)),result=out))
 # Original snapshot hashes, original image hashes, and baseline reproduction are all checked again.
 records=json.loads((here/'baseline_snapshot/hashes.json').read_text())
 assert all(E.sha(E.BASE/n)==d and E.sha(here/'baseline_snapshot'/n)==d for n,d in records.items())
 image_hashes=json.loads((here/'cache/prepared.json').read_text())['image_hashes']
 assert all(E.sha(E.path_for(n))==d for n,d in image_hashes.items())
 result=dict(all_prediction_rows_independently_scored=len(pred),method_protocol_groups=len(recalculated),
  exact_previous_split_lists=True,training_only_noise_pair_records_checked=pair_checks,heldout_raw_pixel_access_guard_passed=True,
  frozen_raw_checks=details,original_files_and_images_unchanged=True,
  constant_variance_check=json.loads((here/'constant_variance_check.json').read_text()),new_acquisition_independent_test=False)
 E.dump(here/'audit_checks.json',result)
 print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
