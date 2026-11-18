"""Predeclared, train-only pixel-noise weighted GN comparison.

Only this directory is written. Original measurements, models, splits and results
are read-only. Each outer fold refits its own brightness-noise curve, extracts
both training and test measurements with that curve, then refits the original
P12 / six-origin / quadratic / covariance classifier on its training rows.
"""
from pathlib import Path
import argparse, concurrent.futures, hashlib, json, shutil, sys, time
import numpy as np
import pandas as pd
import cv2
from scipy import ndimage

HERE=Path(__file__).resolve().parent
BASE=HERE.parent
sys.path[:0]=[str(BASE),str(BASE/'measurement'),str(BASE/'gls_comparison_20261014')]
import compare as C
import zero_experiment as Z
from measure import ROOT,CROP,BOXES,read,coarse,track
from refine import Refiner

cv2.setNumThreads(1)
METHODS=['baseline','pixel_noise_gn']
WORKERS=4
PLAN={
 'method':'pixel_noise_gn', 'comparison':'only pixel weighting changes; original downstream algorithm refitted',
 'reference':'one original center.bmp', 'points':205,'GN_window':17,'blur_sigma':.8,
 'initialization':'unchanged whole-QR NCC and LK15, cached before fold fitting',
 'noise_model':'v8-style integer-near training Y image pairs, restricted to the three QR crops',
 'noise_pairs':8,'brightness_bin_width':16,'minimum_bin_samples':100,'noise_sd_floor_gray':.5,
 'noise_fit':'per-QR affine brightness correction, 9x9 local displacement projection, residual variance/2',
 'weight_model':'frozen diagonal pair-residual variance from predicted template texture, both images, blur kernel squared',
 'weight_freeze':'per-image gain/offset from original unweighted GN; no motion labels; same initialization for weighted GN',
 'gain_offset':'same four-parameter GN design; all four columns use the same pixel weights',
 'variance_assumptions':'independent equal raw-frame noise for calibration; diagonal approximation after blur/interpolation; not known sensor covariance',
 'downstream':'unchanged unweighted P12, six origin candidates, axis-shared quadratic curves, legacy full residual covariance',
 'CV':'exact train/test filename lists from previous GLS splits.json; fit noise and classifier inside each training fold',
 'hyperparameters':'fixed before results; no outcome-based search or selection',
 'truth':'original filename commands; no independent physical pose truth',
 'full280':'resubstitution, not independent test', 'repeated_cv':'20 x 5-fold, same 280 images; 5600 correlated repeated predictions',
 'session_out':'separate diagnostic: missing origin/sign branch; not comparable with balanced new-run validation',
 'failure':'axis/sign error OR either full-vector absolute error > .09 OR quality rejection',
 'invalid':'P12 uses valid points; requires >=95% and >=20 per QR; reject counted as failure; invalid training rows excluded and recorded',
 'scope':'current 280-image development dataset; no new acquisition performance claimed'
}

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()

def dump(path,value):
 Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')

def snapshot():
 ledger=HERE/'baseline_snapshot/hashes.json'
 if ledger.exists():
  records=json.loads(ledger.read_text(encoding='utf8'))
  assert all(sha(BASE/n)==d and sha(ledger.parent/n)==d for n,d in records.items())
  return records
 records=json.loads((BASE/'gls_comparison_20261014/baseline_snapshot/hashes.json').read_text(encoding='utf8'))
 extras=['gls_comparison_20261014/'+n for n in ['compare.py','splits.json','predictions.csv','metrics.csv','comparison_summary.csv','checks.json']]
 extras+=['measurement/gn17_spread14_metadata.json']
 for n in list(records)+extras:
  src=BASE/n;dst=ledger.parent/n
  dst.parent.mkdir(parents=True,exist_ok=True)
  shutil.copy2(src,dst);records[n]=sha(src)
  assert sha(dst)==records[n]
 ledger.parent.mkdir(exist_ok=True,parents=True)
 dump(ledger,records)
 return records

def path_for(n):
 return ROOT/n if n=='center.bmp' else ROOT/('Y' if n[0]=='Y' else 'Theta')/n

def load_data():
 z=np.load(BASE/'measurement/gn17_spread14.npz')
 m,X,_,_=Z.load('gn17_spread14')
 assert len(m)==281 and (z['regions']<3).all() and z['D'].shape==(281,205,2)
 return z,m,X

def prepare():
 snapshot()
 dump(HERE/'experiment_plan.json',PLAN)
 z,m,X=load_data();ref=read(ROOT/'center.bmp')
 points=(z['points']-np.array(CROP[:2])).astype(np.float32)
 regions=z['regions'];boxes=[];models=[];templates=[];local_points=[]
 for k,(x0,y0,x1,y1) in enumerate(BOXES):
  x0-=CROP[0];x1-=CROP[0];y0-=CROP[1];y1-=CROP[1]
  box=(x0,y0,x1,y1);boxes.append(box)
  temp=ref[y0:y1,x0:x1].copy();p=points[regions==k]-np.array([x0,y0],np.float32)
  models.append(Refiner(temp,p,17,.8));templates.append(cv2.GaussianBlur(temp,(0,0),.8));local_points.append(p)
 folder=HERE/'cache';folder.mkdir(exist_ok=True)
 ready=folder/'prepared.json'
 if ready.exists():
  print('using prepared image cache',flush=True);return
 t=time.perf_counter()
 # Four independent workers only read images; output assignment stays ordered.
 def job(i):
  n=str(z['names'][i]);cur=read(path_for(n));initial,scores=coarse(ref,cur,include_screws=False)
  raw=[];coeff=[];init=np.zeros((205,2),np.float64);lkvalid=np.zeros(205,bool);baseline=np.zeros((205,2),np.float32)
  for k,(x0,y0,x1,y1) in enumerate(boxes):
   dx,dy=initial[k].astype(int);target=cur[y0+dy:y1+dy,x0+dx:x1+dx].copy();sel=regions==k
   assert target.shape==templates[k].shape
   blurred=cv2.GaussianBlur(target.astype(np.float64),(0,0),.8)
   d,lq,lv=track(templates[k],cv2.GaussianBlur(target,(0,0),.8),local_points[k],np.zeros_like(local_points[k]),15)
   gd,gq,gv=models[k].run(target,d)
   raw.append(target);coeff.append(ndimage.spline_filter(blurred,order=3))
   init[sel]=d;lkvalid[sel]=lv;baseline[sel]=gd+initial[k]
  return raw,coeff,init,lkvalid,initial,scores,float(np.max(np.abs(baseline-z['D'][i])))
 rawmaps=[];coeffmaps=[]
 for k,temp in enumerate(templates):
  shape=(len(m),*temp.shape)
  rawmaps.append(np.lib.format.open_memmap(folder/f'raw{k}.npy',mode='w+',dtype='uint8',shape=shape))
  coeffmaps.append(np.lib.format.open_memmap(folder/f'coeff{k}.npy',mode='w+',dtype='float64',shape=shape))
 initials=[];valids=[];coarses=[];scores=[];diffs=[]
 with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as pool:
  for i,r in enumerate(pool.map(job,range(len(m)))):
   for k in range(3):rawmaps[k][i]=r[0][k];coeffmaps[k][i]=r[1][k]
   initials.append(r[2]);valids.append(r[3]);coarses.append(r[4]);scores.append(r[5]);diffs.append(r[6])
   if i%50==0:print('prepared',i+1,len(m),round(time.perf_counter()-t,1),flush=True)
 for mm in rawmaps+coeffmaps:mm.flush()
 np.savez_compressed(folder/'initialization.npz',initial=np.array(initials),valid=np.array(valids),coarse=np.array(coarses),scores=np.array(scores))
 assert max(diffs)==0.,max(diffs)
 dump(ready,{'n':281,'QR_points':205,'original_GN_reproduced_all_281_max_abs_difference':max(diffs),'seconds':time.perf_counter()-t,
  'plan_sha256':sha(HERE/'experiment_plan.json'),'image_hashes':{str(n):sha(path_for(str(n))) for n in z['names']}})
 print('PREPARE COMPLETE',round(time.perf_counter()-t,1),flush=True)

def estimate_noise(raw,m,base_D,coarses,tr):
 # Neither held-out pixels nor held-out labels/measurements enter this function's fit.
 ids=np.array([i for i in tr if m.axis.iloc[i]=='Y'],int)
 mean=base_D[ids].mean(1)
 candidates=[]
 for a in range(len(ids)):
  for b in range(a+1,len(ids)):
   d=mean[b]-mean[a]
   candidates.append((float(np.abs(d-np.rint(d)).sum()),int(ids[a]),int(ids[b])))
 pairs=sorted(candidates)[:PLAN['noise_pairs']]
 assert len(pairs)==8 and all(a in tr and b in tr for _,a,b in pairs)
 count=np.zeros(16,np.int64);sums=np.zeros(16);sumsq=np.zeros(16)
 regions=np.load(BASE/'measurement/gn17_spread14_features.npz')['regions']
 details=[]
 for integer_error,a,b in pairs:
  for k in range(3):
   local=(base_D[b,regions==k]-base_D[a,regions==k]).mean(0)-(coarses[b,k]-coarses[a,k])
   dx,dy=np.rint(local).astype(int)
   aa=raw[k][a].astype(float);bb=raw[k][b].astype(float);h,w=aa.shape
   x0=max(3,3-dx);x1=min(w-3,w-3-dx);y0=max(3,3-dy);y1=min(h-3,h-3-dy)
   A=aa[y0:y1,x0:x1];B=bb[y0+dy:y1+dy,x0+dx:x1+dx]
   ac=A-A.mean(); gain=float(np.sum(ac*(B-B.mean()))/np.sum(ac*ac));offset=float(B.mean()-gain*A.mean())
   dd=B-(gain*A+offset)
   g=cv2.GaussianBlur((A+B)/2,(0,0),.7);gy,gx=np.gradient(g);kernel=(9,9)
   xx=cv2.blur(gx*gx,kernel);yy=cv2.blur(gy*gy,kernel);xy=cv2.blur(gx*gy,kernel)
   xd=cv2.blur(gx*dd,kernel);yd=cv2.blur(gy*dd,kernel);det=xx*yy-xy*xy+1e-6
   sx=(yy*xd-xy*yd)/det;sy=(xx*yd-xy*xd)/det
   residual=(dd-gx*sx-gy*sy)[5:-5,5:-5].ravel()
   brightness=((A+B)/2)[5:-5,5:-5].ravel()
   bins=np.minimum((np.clip(brightness,0,255)/16).astype(int),15)
   count+=np.bincount(bins,minlength=16);sums+=np.bincount(bins,weights=residual,minlength=16);sumsq+=np.bincount(bins,weights=residual**2,minlength=16)
   details.append(dict(a=str(m.file.iloc[a]),b=str(m.file.iloc[b]),qr=k,integer_error=integer_error,gain=gain,offset=offset))
 var=np.maximum((sumsq-sums*sums/np.maximum(count,1))/np.maximum(count-1,1)/2,0)
 valid=count>=PLAN['minimum_bin_samples'];centers=np.arange(8,256,16,dtype=float)
 assert valid.sum()>=8
 sd=np.sqrt(np.maximum(np.interp(centers,centers[valid],var[valid]),PLAN['noise_sd_floor_gray']**2))
 return dict(centers=centers.tolist(),sd=sd.tolist(),counts=count.tolist(),pairs=details,training_files=m.file.iloc[tr].tolist(),
  note='QR-only v8-style residual model; equal independent raw-frame variance assumption; not an identified sensor-noise mechanism')

class WeightedRefiner:
 def __init__(self,template,points):
  self.old=Refiner(template,points,17,.8)
  temp=self.old.template
  smooth=cv2.GaussianBlur(template.astype(np.float64),(0,0),.8);gy,gx=np.gradient(smooth)
  tx=ndimage.map_coordinates(gx,self.old.coords,order=3,mode='nearest')
  ty=ndimage.map_coordinates(gy,self.old.coords,order=3,mode='nearest')
  self.design=np.stack([tx,ty,temp,np.ones_like(temp)],axis=-1)
  yy,xx=np.mgrid[-11:12,-11:12]
  coords=np.stack([points[:,1,None]+yy.ravel(),points[:,0,None]+xx.ravel()])
  # Expanded support permits the same sigma-.8 Gaussian kernel's variance propagation.
  self.raw_texture=ndimage.map_coordinates(template.astype(float),coords,order=3,mode='nearest').reshape(-1,23,23)
  g=cv2.getGaussianKernel(7,.8).ravel();self.kernel_squared=g*g

 def variance(self,raw_variance):
  v=ndimage.convolve1d(raw_variance,self.kernel_squared,axis=1,mode='nearest')
  v=ndimage.convolve1d(v,self.kernel_squared,axis=2,mode='nearest')
  return v[:,3:-3,3:-3].reshape(-1,289)

 def configure(self,curve):
  self.centers=np.array(curve['centers']);self.sd=np.array(curve['sd'])
  self.reference_variance=self.variance(np.interp(self.raw_texture,self.centers,self.sd)**2)

 def run(self,coeff,initial,gain_offset):
  # Weights frozen before weighted optimization, using only this image's unweighted
  # photometric gain/offset. No label or neighboring image is consulted here.
  a=np.maximum(gain_offset[:,0],.25);b=gain_offset[:,1]
  expected=a[:,None,None]*self.raw_texture+b[:,None,None]
  curvar=self.variance(np.interp(expected,self.centers,self.sd)**2)
  pairvar=np.maximum(curvar+a[:,None]**2*self.reference_variance,1e-6)
  weights=1/pairvar
  weights/=weights.mean(axis=1,keepdims=True)
  design=self.design
  weighted_design=design*weights[:,:,None]
  normal=np.einsum('npi,npj->nij',design,weighted_design,optimize=True)
  rhs=weighted_design.transpose(0,2,1)
  pinverse=np.linalg.solve(normal,rhs)
  d=initial.astype(float).copy()
  for it in range(12):
   coords=self.old.coords+d[:,::-1].T[:,:,None]
   values=ndimage.map_coordinates(coeff,coords,order=3,prefilter=False,mode='nearest')
   pars=np.einsum('nkp,np->nk',pinverse,values,optimize=True)
   step=np.clip(pars[:,:2]/np.maximum(pars[:,2,None],.25),-.5,.5)
   d-=step
   if np.max(np.linalg.norm(step,axis=1))<2e-5:break
  resid=values-pars[:,2,None]*self.old.template-pars[:,3,None]
  quality=np.c_[np.sqrt(np.mean(resid**2,axis=1)),np.linalg.norm(step,axis=1),pars[:,2],pars[:,3]]
  valid=np.isfinite(d).all(1)&(np.linalg.norm(d-initial,axis=1)<1.5)&(quality[:,1]<.02)
  return d.astype(np.float32),quality.astype(np.float32),valid, float(weights.min()),float(weights.max())

def project(D,valid,points,regions):
 d=D.astype(float)-D[0].astype(float)
 x,y=(points[:,0]-1906)/486,(points[:,1]-1070)/122
 B=np.array([x*0+1,x,y,x*x,x*y,y*y]).T
 good=(valid.mean(1)>=.95)&np.array([all((v&(regions==k)).sum()>=20 for k in range(3)) for v in valid])
 if valid.all():
  X=np.linalg.lstsq(B,d.transpose(1,0,2).reshape(len(points),-1),rcond=None)[0].reshape(6,len(d),2).transpose(1,0,2).reshape(len(d),12)
 else:
  X=np.array([np.linalg.lstsq(B[v],row[v],rcond=None)[0].ravel() if q else np.zeros(12) for row,v,q in zip(d,valid,good)])
 return X,good

def models_equal(a,b):
 return np.allclose(a[1],b[1],rtol=0,atol=1e-12) and a[0].keys()==b[0].keys() and all(np.allclose(np.array(a[0][k][1:]),np.array(b[0][k][1:]),rtol=0,atol=1e-12) for k in a[0])

def run():
 t=time.perf_counter();records=snapshot();z,m,Xbase=load_data()
 assert sha(HERE/'experiment_plan.json')==json.loads((HERE/'cache/prepared.json').read_text())['plan_sha256']
 folder=HERE/'cache';raw=[np.load(folder/f'raw{k}.npy',mmap_mode='r') for k in range(3)]
 coeff=[np.load(folder/f'coeff{k}.npy',mmap_mode='r') for k in range(3)]
 init=np.load(folder/'initialization.npz');regions=z['regions'];points=z['points']
 ref=read(ROOT/'center.bmp');models=[]
 for k,(x0,y0,x1,y1) in enumerate(BOXES):
  template=ref[y0-CROP[1]:y1-CROP[1],x0-CROP[0]:x1-CROP[0]]
  models.append(WeightedRefiner(template,(points[regions==k]-[x0,y0]).astype(np.float32)))
 splits=json.loads((BASE/'gls_comparison_20261014/splits.json').read_text())
 name_index={n:i for i,n in enumerate(m.file)}
 old=pd.read_csv(BASE/'gls_comparison_20261014/predictions.csv').query("method=='baseline'")
 foldsdir=HERE/'folds';foldsdir.mkdir(exist_ok=True)
 allframes=[];diagnostics=[];metricrows=[];audit=[]
 for count,s in enumerate(splits):
  protocol,fold=s['protocol'],str(s['fold']);key=protocol+'__'+fold
  tr=np.array([name_index[n] for n in s['train']]);te=np.array([name_index[n] for n in s['test']])
  assert protocol=='full280' or not np.intersect1d(tr,te).size
  folder=foldsdir/key;folder.mkdir(exist_ok=True)
  done=folder/'predictions.csv'
  if done.exists():
   frame=pd.read_csv(done);allframes.append(frame)
   diagnostics.append(json.loads((folder/'diagnostics.json').read_text()))
   print('resume',count+1,len(splits),key,flush=True);continue
  ft=time.perf_counter()
  curve=estimate_noise(raw,m,z['D'],init['coarse'],tr)
  dump(folder/'noise_model.json',curve)
  for model in models:model.configure(curve)
  D=np.zeros((281,205,2),np.float32);valid=np.zeros((281,205),bool);quality=np.zeros((281,205,4),np.float32)
  def job(i):
   dd=np.zeros((205,2),np.float32);vv=np.zeros(205,bool);qq=np.zeros((205,4),np.float32);wlo=1e9;whi=0
   for k in range(3):
    sel=regions==k
    gd,gq,gv,lo,hi=models[k].run(coeff[k][i],init['initial'][i,sel],z['quality'][i,sel,2:4])
    dd[sel]=gd+init['coarse'][i,k];vv[sel]=gv&init['valid'][i,sel];qq[sel]=gq
    wlo=min(wlo,lo);whi=max(whi,hi)
   return dd,vv,qq,wlo,whi
  wlo=1e9;whi=0
  with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as pool:
   for i,r in enumerate(pool.map(job,range(len(m)))):
    D[i],valid[i],quality[i]=r[:3];wlo=min(wlo,r[3]);whi=max(whi,r[4])
  X,good=project(D,valid,points,regions)
  fittr=tr[good[tr]]
  assert len(fittr)>20
  baseline=Z.fit(m,Xbase,tr,m.value.to_numpy(),'inferred_session',True)
  model=Z.fit(m,X,fittr,m.value.to_numpy(),'inferred_session',True)
  parts=[]
  for method,XX,mod in [('baseline',Xbase,baseline),('pixel_noise_gn',X,model)]:
   df=Z.predict(m,XX,te,mod,'inferred_session').assign(method=method,protocol=protocol,fold=fold)
   df['rejected']=False
   if method=='pixel_noise_gn':
    reject=~good[te];df.loc[reject,'pred_axis']='REJECT';df.loc[reject,'pred']=0.;df.loc[reject,'branch']='REJECT';df.loc[reject,'rejected']=True
   parts.append(df)
  joined=pd.concat(parts,ignore_index=True)
  check=joined.query("method=='baseline'").merge(old.query('protocol==@protocol and fold==@fold'),on='file',suffixes=('_new','_old'))
  # CSV may infer integer fold strings inconsistently; match protocol/file if needed.
  if len(check)!=len(te):
   check=joined.query("method=='baseline'").merge(old.query('protocol==@protocol'),on='file',suffixes=('_new','_old'))
  assert len(check)==len(te) and (check.pred_axis_new==check.pred_axis_old).all()
  diff=float(np.max(np.abs(check.pred_new-check.pred_old)));assert diff<1e-8,diff
  if count in [0,1,13]:
   # Perturb every held-out cached input and label; noise fit / model must stay fixed.
   if protocol!='full280':
    m2=m.copy();m2.loc[te,'value']+=1000;D2=z['D'].copy();D2[te]+=1000
    curve2=estimate_noise(raw,m2,D2,init['coarse'],tr)
    assert curve2['sd']==curve['sd'] and curve2['pairs']==curve['pairs']
    X2=X.copy();X2[te]+=1000
    assert models_equal(model,Z.fit(m2,X2,fittr,m2.value.to_numpy(),'inferred_session',True))
  delta=np.linalg.norm(D.astype(float)-z['D'].astype(float),axis=2)
  diag=dict(protocol=protocol,fold=fold,training_n=len(tr),training_used=len(fittr),test_n=len(te),
   training_rejected=int((~good[tr]).sum()),test_rejected=int((~good[te]).sum()),valid_point_fraction=float(valid.mean()),
   measurement_change_median_px=float(np.median(delta)),measurement_change_p95_px=float(np.quantile(delta,.95)),
   measurement_change_max_px=float(delta.max()),weight_min=wlo,weight_max=whi,
   baseline_prediction_max_abs_difference=diff,seconds=time.perf_counter()-ft)
  dump(folder/'diagnostics.json',diag)
  if protocol=='full280':
   np.savez_compressed(HERE/'weighted_measurements_full280.npz',names=z['names'],points=points,regions=regions,D=D,valid=valid,quality=quality,X=X)
   dump(HERE/'weighted_model_full280.json',dict(W=model[1].tolist(),branches={k:[a,b.tolist(),u.tolist(),q.tolist()] for k,(a,b,u,q) in model[0].items()},noise_model=curve,plan=PLAN))
  joined.to_csv(done,index=False,encoding='utf-8-sig')
  allframes.append(joined);diagnostics.append(diag)
  print('fold',count+1,len(splits),key,'seconds',round(diag['seconds'],2),'noise_sd',np.round(curve['sd'],2).tolist(),'rejected',diag['test_rejected'],flush=True)
  if count in [0,1]:
   print('BENCHMARK estimated remaining minutes',round(diag['seconds']*(len(splits)-count-1)/60,1),flush=True)
 pred=pd.concat(allframes,ignore_index=True)
 pred['correct_axis']=pred.axis==pred.pred_axis
 pred['correct']=pred.correct_axis&(np.sign(pred.value)==np.sign(pred.pred))
 for a in ['Y','T']:
  pred['error_'+a]=np.abs(np.where(pred.axis==a,pred.value,0)-np.where(pred.pred_axis==a,pred.pred,0))
 pred['pass09']=pred.correct&(pred.error_Y<=.09)&(pred.error_T<=.09)&~pred.rejected
 for (method,protocol),g in pred.groupby(['method','protocol'],sort=False):
  assert len(g)==280 and g.file.nunique()==280
  metricrows.append(dict(method=method,protocol=protocol,rejected=int(g.rejected.sum()),**C.score(g)))
 pred.to_csv(HERE/'predictions.csv',index=False,encoding='utf-8-sig')
 pred.loc[~pred.pass09].to_csv(HERE/'failures.csv',index=False,encoding='utf-8-sig')
 pd.DataFrame(metricrows).to_csv(HERE/'metrics.csv',index=False,encoding='utf-8-sig')
 pd.DataFrame(diagnostics).to_csv(HERE/'fold_diagnostics.csv',index=False,encoding='utf-8-sig')
 shutil.copy2(BASE/'gls_comparison_20261014/splits.json',HERE/'splits.json')
 original_unchanged=all(sha(BASE/n)==d and sha(HERE/'baseline_snapshot'/n)==d for n,d in records.items())
 assert original_unchanged
 image_hashes=json.loads((HERE/'cache/prepared.json').read_text())['image_hashes']
 assert all(sha(path_for(n))==d for n,d in image_hashes.items())
 dump(HERE/'checks.json',dict(original_files_unchanged=True,original_images_unchanged=True,
  exact_previous_splits_sha256=sha(HERE/'splits.json'),baseline_reproduced_all_protocols=True,
  baseline_max_prediction_difference=max(d['baseline_prediction_max_abs_difference'] for d in diagnostics),
  every_method_protocol_280_unique=True,protocol_count=len(metricrows)//2,fold_count=len(splits),
  noise_training_only=True,heldout_perturbation_checked=True,all_predictions_count=len(pred),independent_new_acquisition_test=False,
  runtime_seconds=time.perf_counter()-t,experiment_code_sha256=sha(__file__),plan_sha256=sha(HERE/'experiment_plan.json')))
 print('EXPERIMENT COMPLETE',round(time.perf_counter()-t,1),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','run']);args=p.parse_args()
 HERE.mkdir(exist_ok=True)
 if args.phase=='prepare':prepare()
 else:run()
