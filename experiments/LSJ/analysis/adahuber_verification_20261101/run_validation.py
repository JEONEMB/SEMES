from pathlib import Path
from collections import defaultdict
import json, hashlib, subprocess, time, math, csv
import numpy as np
import cv2

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
BASE=HERE.parent/'reference_line_methods_108_20261030'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(name,obj): (HERE/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
profiles=HERE.parent/'gaussian_window_no_rejection_20261101/profiles.jsonl'
groups=defaultdict(list)
for line in profiles.read_text(encoding='utf-8').splitlines():
    r=json.loads(line); groups[(r['ref'],r['target'])].append(r)
manifest=json.loads((BASE/'configs/inputs_manifest.json').read_text(encoding='utf-8'))
targets={r['id']:r for r in json.loads((BASE/'configs/targets.json').read_text(encoding='utf-8'))['targets']}
provenance={'profile_file':str(profiles),'profile_sha256':sha(profiles),'images':[],
 'scope':'4 original reference images; 8 complete upper/lower point sets; every saved integer Sobel point, no Gaussian or rejection',
 'upstream_source_sha256':sha(HERE/'sources/adaHuber/src/adaHuber.cpp'),
 'numerical_settings':'Original defaults unchanged. Counterexample datasets and diagnostic variants are tests only.',
 'execution':'Original C++ core unchanged, genuine Armadillo header math; R list I/O replaced by JSON adapter. R package runtime not executed.'}
jobs=[]
def add(key,x,y,**extra):
    jobs.append(dict(id=key,x=np.asarray(x,float).reshape(len(y),-1).tolist(),y=np.asarray(y,float).tolist(),**extra))
for ref in manifest['references']:
    p=Path(ref['path']); assert sha(p)==ref['sha256']
    img=cv2.imdecode(np.frombuffer(p.read_bytes(),np.uint8),cv2.IMREAD_GRAYSCALE)
    gy=cv2.Sobel(img,cv2.CV_32F,0,1,ksize=3,scale=1,borderType=cv2.BORDER_REFLECT_101)
    detail={'id':ref['reference_id'],'path':str(p),'sha256':sha(p),'shape':list(img.shape),'checks':[]}
    for target in ['upper_structure','lower_structure']:
        rows=groups[(ref['reference_id'],target)]; t=targets[target]
        xs=np.array([r['x'] for r in rows]); ys=np.array([r['coarse_y'] for r in rows]);
        expected_x=np.array([x for x in range(t['x_range'][0],t['x_range'][1]+1) if not any(a<=x<=b for a,b in t['exclude_x'])])
        assert np.array_equal(xs,expected_x)
        band=gy[t['band_y'][0]:t['band_y'][1]+1,xs]
        ys2=np.argmax(band,axis=0)+t['band_y'][0]
        assert np.array_equal(ys,ys2), (ref['reference_id'],target)
        assert np.array_equal(gy[ys,xs],np.array([r['coarse_response'] for r in rows]))
        detail['checks'].append({'target':target,'n':len(rows),'cached_points_equal_fresh_sobel':True})
        add(ref['reference_id']+'_'+target,xs,ys,category='real',type='adaptive',diagnostic_patch=True)
    provenance['images'].append(detail)

# Deterministic mathematical checks. These are not training/augmentation data.
x=np.linspace(-2,2,101); y=3+0.5*x
add('exact_affine',x,y,category='synthetic',type='adaptive',true_coef=[3,0.5])
add('constant_y',x,np.full(len(x),7.0),category='synthetic',type='adaptive',true_coef=[7,0])
add('constant_x',np.ones(101),y,category='invalid_input',type='adaptive')
add('insufficient_n',np.array([1.]),np.array([2.]),category='invalid_input',type='adaptive')
add('quantized_line',np.arange(200),np.rint(700+0.01*np.arange(200)),category='synthetic',type='adaptive')
add('majority_tie_y',np.arange(101),np.r_[np.zeros(80),np.linspace(0.1,1,21)],category='synthetic',type='adaptive')
noise=np.sin(np.arange(101)*1.37)*0.1
add('noisy_line',x,y+noise,category='synthetic',type='adaptive')
for scale in [0.001,1000.]: add('noisy_line_scale_'+str(scale),x,(y+noise)*scale,category='invariance',type='adaptive',parent='noisy_line',y_scale=scale)
add('noisy_line_translate',x+4000,y+noise+1900,category='invariance',type='adaptive',parent='noisy_line',x_offset=4000,y_offset=1900)
add('noisy_line_reverse',x[::-1],(y+noise)[::-1],category='invariance',type='adaptive',parent='noisy_line')
for cap in [0,1]: add('iteration_cap_'+str(cap),x,y+noise,category='termination',type='adaptive',iteMax=cap)
for scale in [1,0.001,1000]:
    vals=np.r_[np.zeros(99),100.]*scale
    jobs.append(dict(id='mean_outlier_scale_'+str(scale),y=vals.tolist(),category='counterexample',type='mean',diagnostic_patch=True))
jobs.append(dict(id='mean_single_sample',y=[5.],category='invalid_input',type='mean',diagnostic_patch=True))
jobs.append(dict(id='mean_small_range',y=[0.,0.,0.001],category='counterexample',type='mean',diagnostic_patch=True))
# Non-degenerate continuous, skewed samples exercise the intercept solver loop.
rng=np.random.default_rng(20261101)
for distribution in ['exponential','lognormal','normal_contaminated']:
    for n in [20,100,1000]:
        if distribution=='exponential': vals=rng.exponential(1,size=n)
        elif distribution=='lognormal': vals=rng.lognormal(0,1.5,size=n)
        else:
            vals=rng.normal(size=n); vals[-max(1,n//50):]+=25
        jobs.append(dict(id=f'mean_{distribution}_{n}',y=vals.tolist(),category='counterexample',type='mean',diagnostic_patch=True))
        if n==100:
            add(f'regression_{distribution}',np.linspace(-2,2,n),3+0.5*np.linspace(-2,2,n)+vals,category='counterexample',type='adaptive',diagnostic_patch=True)
# Off-path sparse-regression bug probe, clearly separate from chosen low-dim method.
xx=np.column_stack([x,np.cos(np.arange(101)*0.7)])
add('lasso_residual_probe',xx,5+3*x+noise,category='off_path',type='lasso',diagnostic_patch=True)
add('cov_constant',np.column_stack([np.ones(10)*5,np.ones(10)*7]),np.zeros(10),category='off_path',type='cov')
add('cov_variable',np.column_stack([np.arange(10),np.arange(10)*2]),np.zeros(10),category='off_path',type='cov')
add('standard_reference_check',x,y+noise,category='off_path',type='standard')
with (HERE/'requests.jsonl').open('w',encoding='utf-8') as f:
    for q in jobs: f.write(json.dumps(q,ensure_ascii=True,allow_nan=False)+'\n')
exe=HERE/'build/Release/adahuber_check.exe'
start=time.perf_counter()
proc=subprocess.run([str(exe),'requests.jsonl','native_results.jsonl'],cwd=HERE,capture_output=True,text=True,timeout=120)
(HERE/'execution.log').write_text(proc.stdout+'\n'+proc.stderr,encoding='utf-8')
assert proc.returncode==0,proc.stderr
results=[json.loads(s) for s in (HERE/'native_results.jsonl').read_text().splitlines()]
byid={r['id']:r for r in results}
summary=[]
for q in jobs:
    r=byid[q['id']]
    if 'exception' in r:
        summary.append({'id':q['id'],'category':q['category'],'exception':r['exception']});continue
    assert r['result']==r['observed_result'],q['id']
    s={'id':q['id'],'category':q['category'],'n':len(q['y']),'result':r['result'],'observed_matches_original':True,'runtime_ms':r['runtime_ms']}
    events=r['events']; regs=[e for e in events if e['kind']=='regression_pre_intercept']; means=[e for e in events if e['kind']=='mean']
    if regs: s['regression_stage']=regs[-1]
    if means: s['mean_stage']=means[-1]
    roots=[e for e in events if e['kind']=='root']; s['root_calls']=len(roots)
    if roots:s['last_root']=roots[-1]
    if 'patch_result' in r:s['diagnostic_patch_result']=r['patch_result']
    if q['type'] in ['adaptive','standard']:
        b=np.array(r['result']['coef'],dtype=float); X=np.array(q['x']); Y=np.array(q['y'])
        resid=Y-b[0]-X@b[1:]; tau=r['result']['tau']
        if np.isfinite(b).all():
            s.update(coef=b.tolist(),angle_deg=float(np.degrees(np.arctan(b[1]))),rmse_px=float(np.sqrt(np.mean(resid**2))),max_abs_residual_px=float(np.max(np.abs(resid))))
            if tau and tau>0:
                weights=np.minimum(1,tau/np.maximum(np.abs(resid),np.finfo(float).tiny))
                s.update(downweighted_points=int((weights<1).sum()),minimum_weight=float(weights.min()))
            if q.get('true_coef'):s['coefficient_error']=(b-np.array(q['true_coef'])).tolist()
            if regs and np.array(regs[-1]['beta_standardized'],dtype=float).size==2:
                # Evaluate exactly the two estimating equations, before and after intercept reset.
                z=np.column_stack([np.ones(len(Y)),(X[:,0]-X[:,0].mean())/X[:,0].std(ddof=1)])
                rr=Y-Y.mean()-z@np.array(regs[-1]['beta_standardized'])
                tt=regs[-1]['tau']
                if tt and tt>0:
                    s['pre_reset_equation2']=float(np.mean(np.minimum((rr/tt)**2,1))-(1+math.log(len(Y)))/len(Y))
                    s['post_reset_gradient_inf']=float(np.max(np.abs(z.T@np.clip(resid,-tt,tt)/len(Y))))
                    s['post_reset_equation2']=float(np.mean(np.minimum((resid/tt)**2,1))-(1+math.log(len(Y)))/len(Y))
        else:s['nonfinite_result']=True
    summary.append(s)
provenance.update(opencv_version=cv2.__version__,runtime_seconds=time.perf_counter()-start,job_count=len(jobs),source_unchanged=sha(HERE/'sources/adaHuber/src/adaHuber.cpp')==provenance['upstream_source_sha256'])
dump('data_provenance.json',provenance)
dump('summary.json',summary)
real=[r for r in summary if r['category']=='real']
fields=['id','n','angle_deg','rmse_px','max_abs_residual_px','downweighted_points','minimum_weight','runtime_ms']
with (HERE/'real_data_results.csv').open('w',newline='',encoding='utf-8-sig') as f:
    w=csv.DictWriter(f,fieldnames=fields); w.writeheader();w.writerows({k:r.get(k) for k in fields} for r in real)
for s in summary:
    print(s['id'],s.get('coef',s.get('result',s.get('exception'))),'reggrad',s.get('regression_stage',{}).get('gradient_inf'),'meangrad',s.get('mean_stage',{}).get('gradient'),'meaniter',s.get('mean_stage',{}).get('iteration'),flush=True)
