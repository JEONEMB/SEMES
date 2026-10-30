"""Preserved four-way measurement/classifier comparison and T-0.05 diagnosis."""
from pathlib import Path
import argparse,ast,concurrent.futures,hashlib,importlib.util,json,os,shutil,sys,time
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
BASE=HERE.parent
PREV=BASE/'pixel_weighted_gn_20261015'
sys.path.insert(0,str(PREV))
spec=importlib.util.spec_from_file_location('preserved_pixel_gn_experiment',PREV/'experiment.py')
E=importlib.util.module_from_spec(spec);sys.modules[spec.name]=E;spec.loader.exec_module(E)
sys.path.insert(0,str(HERE))
Z=E.Z
METHODS=['gn_six','pixel_six','gn_v7b','pixel_v7b']
SOURCE=BASE.parent/'analysis_v7b/lib.py'
PLAN=dict(target='T-0.05.bmp',methods=METHODS,measurement='same QR205, existing GN and train-only pixel-weighted GN',
 split_source='pixel_weighted_gn_20261015/splits.json',same_splits=True,truth='raw filename commands, no image-derived corrected labels',
 v7b=dict(classification='linear, center-fixed, local |Y|<=.6 and |T|<=.12',
  value='quadratic, free intercept, full training range',covariance='R.T R/(n-6) + 1e-10 I, separate stage covariances',
  nuisance='original v7b vertical translation and image-plane rotation around the current QR205 centroid',
  solve='8 GN iterations from v=0, nuisance optimized per hypothesis',threshold='cost_Y < cost_T; no tuned threshold',
  center='same observed center included once in each axis fit',adaptation='v7b architecture on common QR205 measurements and raw commands; not the original QR232/image-derived-label score'),
 no_test_label_fit=True,no_hyperparameter_search=True,
 cost_comparison='absolute costs compared within a classifier/fold only; whitenings differ across models and folds',
 diagnostics='six-origin branch costs, target raw/P12 motion, whitened residual modes, measurement/model crossing',
 full280='resubstitution',CV='same acquisition development OOF; not a new independent acquisition',
 session_out='separate missing-origin diagnostic')

def dump(p,x):E.dump(p,x)
def sha(p):return E.sha(p)

def preserve():
 HERE.mkdir(exist_ok=True)
 ledger=HERE/'preserved_files.json'
 if ledger.exists():
  records=json.loads(ledger.read_text(encoding='utf8'));assert all(sha(BASE/n)==h for n,h in records.items());return records
 records={}
 for folder in [PREV,BASE/'gls_comparison_20261014']:
  for p in folder.rglob('*'):
   if p.is_file() and not any(q in {'cache','raw_checks','__pycache__','baseline_snapshot'} for q in p.relative_to(folder).parts):
    records[p.relative_to(BASE).as_posix()]=sha(p)
 for n in ['lib.py','twostage.py','exp.py']:
  p=BASE.parent/'analysis_v7b'/n
  records[os.path.relpath(p,BASE).replace('\\','/')]=sha(p)
 selected=['metrics.csv','predictions.csv','comparison_summary.csv','experiment_plan.json','splits.json','checks.json','audit_checks.json','weighted_model_full280.json']
 for n in selected:
  dst=HERE/'baseline_snapshot'/n;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(PREV/n,dst)
 for n in ['lib.py','twostage.py','exp.py']:
  dst=HERE/'baseline_snapshot/v7b_source'/n;dst.parent.mkdir(exist_ok=True,parents=True);shutil.copy2(BASE.parent/'analysis_v7b'/n,dst)
 dump(ledger,records);return records

def geometry(points):
 x,y=(points[:,0]-1906)/486,(points[:,1]-1070)/122
 B=np.array([x*0+1,x,y,x*x,x*y,y*y]).T
 xc,yc=(points[:,0]-points[:,0].mean())/100,(points[:,1]-points[:,1].mean())/100
 fields=np.stack([np.c_[xc*0,xc*0+1],np.c_[-yc,xc]])
 # Identical to v7b exp.U_for: transform .05-scaled nuisance fields, then divide.
 nuisance=np.linalg.lstsq(B,(fields*.05).transpose(1,0,2).reshape(len(points),-1),rcond=None)[0].reshape(6,2,2).transpose(1,0,2).reshape(2,12).T/.05
 return B,nuisance

class Stage:
 def __init__(self,quad,free,U,local=None):self.quad=quad;self.free=free;self.U=U;self.local=local
 def fit(self,X,m,tr):
  self.poly={};res=[]
  for a in ['Y','T']:
   ids=tr[m.axis.to_numpy()[tr]==a]
   if self.local:ids=ids[np.abs(m.value.to_numpy()[ids])<=self.local[a]]
   ids=np.r_[ids,0];v=m.value.to_numpy()[ids]
   A=np.c_[v,v*v,np.ones(len(v))] if self.quad and self.free else (np.c_[v,v*v] if self.quad else v[:,None])
   co=np.linalg.lstsq(A,X[ids],rcond=None)[0]
   self.poly[a]=(co[0],co[1] if self.quad else np.zeros(12),co[-1] if self.free else np.zeros(12))
   residual=X[ids]-A@co;res.append(residual[ids!=0])
  R=np.vstack(res);self.cov=R.T@R/max(len(R)-6,1)+1e-10*np.eye(12)
  self.W=np.linalg.cholesky(np.linalg.inv(self.cov)).T
  A=self.W@self.U
  self.Q=self.W-A@np.linalg.pinv(A)@self.W
  self.training_n=len(tr)
  return self
 def solve(self,F,axis):
  F=np.atleast_2d(F);s,c,b=self.poly[axis]
  z=(F-b)@self.Q.T;sw=self.Q@s;cw=self.Q@c
  v=np.zeros(len(F))
  for _ in range(8):
   r=z-v[:,None]*sw-(v*v)[:,None]*cw;j=sw+2*v[:,None]*cw
   v+=np.sum(j*r,axis=1)/np.maximum(np.sum(j*j,axis=1),1e-30)
  residual=z-v[:,None]*sw-(v*v)[:,None]*cw
  return v,np.sum(residual**2,axis=1)
 def candidate(self,f,axis):
  v,cost=self.solve(f,axis);v=float(v[0]);s,c,b=self.poly[axis]
  base=b+s*v+c*v*v
  u=np.linalg.lstsq(self.W@self.U,self.W@(f-base),rcond=None)[0]
  return dict(axis=axis,value=v,cost=float(cost[0]),feature=base+self.U@u,nuisance=u)

class TwoStage:
 def __init__(self,U):self.A=Stage(False,False,U,{'Y':.6,'T':.12});self.V=Stage(True,True,U)
 def fit(self,X,m,tr):self.A.fit(X,m,tr);self.V.fit(X,m,tr);return self
 def predict(self,m,X,te):
  vy,cy=self.A.solve(X[te],'Y');vt,ct=self.A.solve(X[te],'T')
  valueY,_=self.V.solve(X[te],'Y');valueT,_=self.V.solve(X[te],'T')
  pa=np.where(cy<ct,'Y','T');v=np.where(pa=='Y',valueY,valueT)
  return pd.DataFrame(dict(file=m.file.iloc[te].to_numpy(),axis=m.axis.iloc[te].to_numpy(),value=m.value.iloc[te].to_numpy(),session=m.session.iloc[te].to_numpy(),
   pred_axis=pa,pred=v,branch=pa,cost=np.minimum(cy,ct),cost_Y=cy,cost_T=ct,disc_vY=vy,disc_vT=vt,
   value_Y=valueY,value_T=valueT,log_cost_ratio=np.log(np.maximum(cy,1e-300)/np.maximum(ct,1e-300))))
 def export(self):
  return {name:dict(W=stage.W.tolist(),U=stage.U.tolist(),poly={a:[s.tolist(),c.tolist(),b.tolist()] for a,(s,c,b) in stage.poly.items()}) for name,stage in [('classification',self.A),('value',self.V)]}

def original_v7b_check(X,m,tr,model):
 # Compile only the original class; do not execute lib.py's imports, global data
 # loading or corrected-label construction.
 tree=ast.parse(SOURCE.read_text(encoding='utf-8-sig'));node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Model')
 scope={'np':np};exec(compile(ast.Module(body=[node],type_ignores=[]),str(SOURCE),'exec'),scope)
 ids=np.r_[tr,0];axes=m.axis.to_numpy()[ids].copy();axes[axes=='T']='Θ';values=m.value.to_numpy()[ids]
 differences=[]
 for stage in [model.A,model.V]:
  source=scope['Model'](quad=stage.quad,noise='full',intercept='free' if stage.free else 'zero',nuis=stage.U,
   local={'Y':.6,'Θ':.12} if stage.local else None).fit(X[ids],values,axes)
  assert np.allclose(source.W,stage.W,rtol=1e-7,atol=1e-6)
  for i in [1,5,99,201,240]:
   for a in ['Y','T']:
    v,c=stage.solve(X[i],a);sv,sc=source.solve(X[i],'Θ' if a=='T' else a)
    differences.append([float(abs(v[0]-sv)),float(abs(c[0]-sc))])
 result=np.array(differences)
 assert result[:,0].max()<1e-7 and result[:,1].max()<1e-5,result.max(0)
 return dict(original_Model_class_only=True,value_max_abs_difference=float(result[:,0].max()),cost_max_abs_difference=float(result[:,1].max()),comparisons=len(result))

def six_candidates(f,model):
 branches,W=model;rows=[]
 for name,(axis,b,s,c) in branches.items():
  zz=W@(f-b);ss=W@s;cc=W@c;v=float(zz@ss/(ss@ss))
  for _ in range(8):
   j=ss+2*v*cc;r=zz-v*ss-v*v*cc;v+=float(j@r/max(j@j,1e-20))
  fit=b+s*v+c*v*v;cost=float(np.sum((W@(f-fit))**2));allowed=v*(1 if name.endswith('+') else -1)>=0
  rows.append(dict(branch=name,axis=axis,value=v,raw_cost=cost,cost=cost if allowed else None,allowed=bool(allowed),feature=fit))
 return rows

def choose_candidates(rows):
 good=[r for r in rows if r['allowed']]
 best=min(good,key=lambda r:r['cost'])
 choices={a:min([r for r in good if r['axis']==a],key=lambda r:r['cost'],default=None) for a in ['Y','T']}
 return best,choices

def curve_key(curve):return hashlib.sha256(json.dumps([curve['centers'],curve['sd']],separators=(',',':')).encode()).hexdigest()[:20]

def main():
 start=time.perf_counter();records=preserve();dump(HERE/'experiment_plan.json',PLAN)
 z,m,Xbase=E.load_data();points=z['points'];regions=z['regions'];B,U=geometry(points)
 target=list(m.file).index(PLAN['target']);splits=json.loads((PREV/'splits.json').read_text());index={n:i for i,n in enumerate(m.file)}
 oldpred=pd.read_csv(PREV/'predictions.csv');raw=[np.load(PREV/f'cache/raw{k}.npy',mmap_mode='r') for k in range(3)]
 coeff=[np.load(PREV/f'cache/coeff{k}.npy',mmap_mode='r') for k in range(3)];init=np.load(PREV/'cache/initialization.npz');ref=E.read(E.ROOT/'center.bmp')
 models=[]
 for k,(x0,y0,x1,y1) in enumerate(E.BOXES):
  temp=ref[y0-E.CROP[1]:y1-E.CROP[1],x0-E.CROP[0]:x1-E.CROP[0]]
  models.append(E.WeightedRefiner(temp,(points[regions==k]-[x0,y0]).astype(np.float32)))
 cache=HERE/'measurements';cache.mkdir(exist_ok=True)
 frames=[];cost_rows=[];target_rows=[];fchecks=[]
 fidelity=json.loads((HERE/'source_fidelity_checks.json').read_text(encoding='utf8')) if (HERE/'source_fidelity_checks.json').exists() else []
 for number,split in enumerate(splits):
  protocol,fold=split['protocol'],str(split['fold']);key=protocol+'__'+fold
  tr=np.array([index[n] for n in split['train']]);te=np.array([index[n] for n in split['test']])
  folder=HERE/'folds'/key;folder.mkdir(parents=True,exist_ok=True)
  if (folder/'complete.json').exists():
   frames.append(pd.read_csv(folder/'predictions.csv'));cost_rows.extend(json.loads((folder/'target_candidates.json').read_text()))
   target_rows.extend(json.loads((folder/'target_summary.json').read_text()))
   fchecks.append(json.loads((folder/'complete.json').read_text()));continue
  t=time.perf_counter();curve=json.loads((PREV/'folds'/key/'noise_model.json').read_text(encoding='utf8'))
  assert curve['training_files']==split['train'] and all(r['a'] in split['train'] and r['b'] in split['train'] for r in curve['pairs'])
  ck=curve_key(curve);file=cache/(ck+'.npz')
  if not file.exists():
   if protocol=='full280':
    data=np.load(PREV/'weighted_measurements_full280.npz');D=data['D'];valid=data['valid'];X=data['X']
   else:
    for mod in models:mod.configure(curve)
    D=np.zeros((281,205,2),np.float32);valid=np.zeros((281,205),bool)
    def job(i):
     dd=np.zeros((205,2),np.float32);vv=np.zeros(205,bool)
     for k in range(3):
      sel=regions==k;gd,gq,gv,lo,hi=models[k].run(coeff[k][i],init['initial'][i,sel],z['quality'][i,sel,2:4])
      dd[sel]=gd+init['coarse'][i,k];vv[sel]=gv&init['valid'][i,sel]
     return dd,vv
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
     for i,(dd,vv) in enumerate(pool.map(job,range(281))):D[i]=dd;valid[i]=vv
    X,good=E.project(D,valid,points,regions);assert good.all()
   assert valid.all()
   np.savez_compressed(file,D=D,valid=valid,X=X)
  data=np.load(file);D=data['D'];valid=data['valid'];X=data['X'];assert valid.all()
  families={'gn':(Xbase,z['D']),'pixel':(X,D)}
  parts=[];tc=[];ts=[];repro_diff=0.
  for measurement,(XX,DD) in families.items():
   six=Z.fit(m,XX,tr,m.value.to_numpy(),'inferred_session',True)
   stage=TwoStage(U).fit(XX,m,tr)
   if number in [0,3]:
    fidelity=[r for r in fidelity if not (r['protocol']==protocol and r['measurement']==measurement)]
    fidelity.append(dict(protocol=protocol,measurement=measurement,**original_v7b_check(XX,m,tr,stage)))
    dump(HERE/'source_fidelity_checks.json',fidelity)
   if number==3:
    # The two-stage fit must be invariant to all held-out measurements and labels.
    X2=XX.copy();X2[te]+=1e3;m2=m.copy();m2.loc[te,'value']+=1e3
    check=TwoStage(U).fit(X2,m2,tr)
    assert json.dumps(stage.export(),sort_keys=True)==json.dumps(check.export(),sort_keys=True)
   for classifier in ['six','v7b']:
    method=measurement+'_'+classifier
    out=Z.predict(m,XX,te,six,'inferred_session') if classifier=='six' else stage.predict(m,XX,te)
    out=out.assign(method=method,protocol=protocol,fold=fold,rejected=False)
    parts.append(out)
    if classifier=='six':
     prev_method='baseline' if measurement=='gn' else 'pixel_noise_gn'
     saved=oldpred[(oldpred.method==prev_method)&(oldpred.protocol==protocol)]
     merged=out.merge(saved,on='file',suffixes=('_new','_old'));assert len(merged)==len(te)
     assert (merged.pred_axis_new==merged.pred_axis_old).all()
     diff=float(np.abs(merged.pred_new-merged.pred_old).max());assert diff<1e-7
     repro_diff=max(repro_diff,diff)
    if target in te:
     f=XX[target]
     if classifier=='six':
      candidates=six_candidates(f,six);best,choices=choose_candidates(candidates);W=six[1]
      cy=choices['Y']['cost'] if choices['Y'] else None;ct=choices['T']['cost'] if choices['T'] else None
     else:
      cy0=stage.A.candidate(f,'Y');ct0=stage.A.candidate(f,'T');W=stage.A.W
      candidates=[dict(branch=a,axis=a,value=c['value'],raw_cost=c['cost'],cost=c['cost'],allowed=True,feature=c['feature']) for a,c in [('Y',cy0),('T',ct0)]]
      best,choices=choose_candidates(candidates);cy,ct=cy0['cost'],ct0['cost']
     selected=out[out.file==PLAN['target']].iloc[0]
     ts.append(dict(protocol=protocol,fold=fold,method=method,pred_axis=selected.pred_axis,pred=float(selected.pred),cost_Y=cy,cost_T=ct,
      log_cost_ratio=None if cy is None or ct is None else float(np.log(max(cy,1e-300)/max(ct,1e-300))),branch=selected.branch,
      measurement_cache=file.name,mean_dx=float((DD[target]-DD[0])[:,0].mean()),mean_dy=float((DD[target]-DD[0])[:,1].mean())))
     for c in candidates:
      field=B@c['feature'].reshape(6,2);actual=DD[target].astype(float)-DD[0].astype(float)
      c2={k:v for k,v in c.items() if k!='feature'}
      c2.update(method=method,protocol=protocol,fold=fold,point_residual_rms=float(np.sqrt(np.mean((actual-field)**2))),
       P12_residual_rms=float(np.sqrt(np.mean((B@(f-c['feature']).reshape(6,2))**2))))
      tc.append(c2)
     np.savez_compressed(folder/(method+'_target.npz'),points=points,regions=regions,B=B,U=U,X=f,raw_displacement=DD[target].astype(float)-DD[0].astype(float),
      W=W,candidate_features=np.array([c['feature'] for c in candidates]),candidate_names=np.array([c['branch'] for c in candidates]),
      candidate_values=np.array([c['value'] for c in candidates]),allowed=np.array([c['allowed'] for c in candidates]),
      costs=np.array([np.nan if c['cost'] is None else c['cost'] for c in candidates]))
   if protocol=='full280':
    dump(HERE/(measurement+'_v7b_model_full280.json'),dict(stages=stage.export(),plan=PLAN,measurement=measurement))
  joined=pd.concat(parts,ignore_index=True);joined.to_csv(folder/'predictions.csv',index=False,encoding='utf-8-sig')
  dump(folder/'target_candidates.json',tc);dump(folder/'target_summary.json',ts)
  record=dict(protocol=protocol,fold=fold,measurement_cache=file.name,baseline_reproduction_max_difference=repro_diff,seconds=time.perf_counter()-t,
   all_205_points_valid=True,training_files=split['train'],test_files=split['test'])
  dump(folder/'complete.json',record);fchecks.append(record);frames.append(joined);cost_rows+=tc;target_rows+=ts
  print('fold',number+1,len(splits),key,'sec',round(record['seconds'],2),'cached_curves',len(list(cache.glob('*.npz'))),flush=True)
 pred=pd.concat(frames,ignore_index=True)
 pred['correct_axis']=pred.axis==pred.pred_axis;pred['correct']=pred.correct_axis&(np.sign(pred.value)==np.sign(pred.pred))
 for a in ['Y','T']:pred['error_'+a]=np.abs(np.where(pred.axis==a,pred.value,0)-np.where(pred.pred_axis==a,pred.pred,0))
 pred['pass09']=pred.correct&(pred.error_Y<=.09)&(pred.error_T<=.09)&~pred.rejected
 metrics=[]
 for (method,protocol),g in pred.groupby(['method','protocol'],sort=False):
  assert len(g)==280 and g.file.nunique()==280
  metrics.append(dict(method=method,protocol=protocol,**E.C.score(g)))
 pred.to_csv(HERE/'predictions.csv',index=False,encoding='utf-8-sig');pred[~pred.pass09].to_csv(HERE/'failures.csv',index=False,encoding='utf-8-sig')
 pd.DataFrame(metrics).to_csv(HERE/'metrics.csv',index=False,encoding='utf-8-sig')
 pd.DataFrame(cost_rows).to_csv(HERE/'T005_candidate_costs.csv',index=False,encoding='utf-8-sig')
 pd.DataFrame(target_rows).to_csv(HERE/'T005_predictions_and_costs.csv',index=False,encoding='utf-8-sig')
 dump(HERE/'fold_checks.json',fchecks);dump(HERE/'source_fidelity_checks.json',fidelity)
 shutil.copy2(PREV/'splits.json',HERE/'splits.json')
 assert all(sha(BASE/n)==h for n,h in records.items())
 dump(HERE/'checks.json',dict(previous_files_unchanged=True,preserved_files=len(records),previous_two_methods_exactly_reproduced=True,
  max_previous_prediction_difference=max(r['baseline_reproduction_max_difference'] for r in fchecks),same_split_sha256=sha(HERE/'splits.json'),
  four_methods=METHODS,protocols=25,folds=119,prediction_rows=len(pred),each_protocol_280_unique=True,
  raw_command_truth=True,corrected_labels_used=False,original_v7b_class_numerically_checked=True,heldout_fit_perturbation_passed=True,
  independent_new_acquisition=False,elapsed_seconds=time.perf_counter()-start,source_sha256=sha(__file__)))
 print('COMPLETE',round(time.perf_counter()-start,1),flush=True)

if __name__=='__main__':main()
