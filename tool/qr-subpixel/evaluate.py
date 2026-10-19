"""Nested parameter selection followed by sealed contiguous holdout evaluation.

Nominal filenames are reference commands, not independently measured truth.
This evaluator reproduces the degree-1 mean-removed theta / mean-motion Y
model, including six session-candidate offsets, without requiring pandas.
"""
import argparse,csv,json,hashlib
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent

def read(path):
    with open(path,newline='',encoding='utf-8-sig') as handle:return list(csv.DictReader(handle))

def write(path,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with open(path,'w',newline='',encoding='utf-8') as h:
        w=csv.DictWriter(h,fieldnames=keys);w.writeheader();w.writerows(rows)

def split_frames(frames):
    f=[]
    for r in frames:
        name=Path(r['path']).stem
        if name=='center':continue
        axis=name[0];value=float(name[1:]);group=axis+('+' if value>0 else '-')
        if axis=='Y':group+=str(int(round(abs(value)*100))%10)
        f.append(dict(frame=int(r['frame']),name=name,axis=axis,value=value,group=group))
    train=[];validation=[];test=[]
    for group in sorted({r['group'] for r in f}):
        rows=sorted([r for r in f if r['group']==group],key=lambda r:r['value']);n=len(rows)
        dev=[]
        for j,r in enumerate(rows):
            if n//3<=j<2*n//3:test.append(r)
            else:dev.append(r)
        for j,r in enumerate(dev):
            (validation if j%3==0 else train).append(r)
    return train,validation,test

def fit_model(frames,maps,reference):
    groups=sorted({r['group'] for r in frames});fitted={}
    for key,ref in reference.items():
        X=[];Z=[]
        for r in frames:
            p=maps.get(r['frame'],{}).get(key)
            if p is None:continue
            y=r['value'] if r['axis']=='Y' else 0.;t=r['value'] if r['axis']=='T' else 0.
            X.append([float(r['group']==g) for g in groups]+[y,t])
            Z.append(p-ref)
        X=np.asarray(X);Z=np.asarray(Z)
        if len(X)<max(48,int(.7*len(frames))) or np.linalg.matrix_rank(X)<len(groups)+2:continue
        c=np.linalg.lstsq(X,Z,rcond=None)[0];fitted[key]=(ref,c)
    return groups,fitted

def infer(fitted,groups,observed,group):
    keys=sorted(set(fitted)&set(observed))
    if len(keys)<24 or len({k.split(':')[0] for k in keys})<3:raise ValueError('insufficient_points')
    d=np.array([observed[k]-fitted[k][0]-fitted[k][1][groups.index(group)] for k in keys])
    a=np.array([fitted[k][1][-2] for k in keys]);b=np.array([fitted[k][1][-1] for k in keys])
    dc=d-d.mean(axis=0);bc=b-b.mean(axis=0)
    denom=np.sum(bc*bc)
    if denom<1e-12:raise ValueError('unobservable_theta')
    theta=float(np.clip(np.sum(dc*bc)/denom,-1,1))
    direction=a.mean(axis=0);remain=(d-b*theta).mean(axis=0)
    y=float(remain@direction/(direction@direction))
    # Separate QR subsets: same calibrated model, independent point collections.
    qr_t=[]
    for qr in sorted({k.split(':')[0] for k in keys}):
        mask=np.array([k.startswith(qr+':') for k in keys]);bd=b[mask]-b[mask].mean(axis=0);dd=d[mask]-d[mask].mean(axis=0)
        if np.sum(bd*bd)>1e-12:qr_t.append(float(np.sum(bd*dd)/np.sum(bd*bd)))
    return y,theta,len(keys),float(np.std(qr_t)),float(np.sqrt(np.mean((dc-bc*theta)**2)))

def evaluate(train,test,maps):
    groups,fitted=fit_model(train,maps,maps.get(0,{}));rows=[]
    for r in test:
        out={**r,'status':'failed','y_error':None,'theta_error':None}
        try:
            y,t,n,qrstd,resid=infer(fitted,groups,maps.get(r['frame'],{}),r['group'])
            yt=r['value'] if r['axis']=='Y' else 0.;tt=r['value'] if r['axis']=='T' else 0.
            out.update(status='ok',y=y,theta=t,y_error=y-yt,theta_error=t-tt,points=n,qr_theta_std=qrstd,shape_rms=resid)
        except ValueError as e:out['status']=str(e)
        rows.append(out)
    return summarize(rows,len(fitted)),rows

def summarize(rows,nfeatures):
    out={'frames':len(rows),'features_fitted':nfeatures}
    for label,sub in [('all',rows),('Y',[r for r in rows if r['axis']=='Y']),('T',[r for r in rows if r['axis']=='T'])]:
        valid=[r for r in sub if r['status']=='ok']
        result={'n':len(sub),'success':len(valid)}
        if valid:
            ey=np.array([r['y_error'] for r in valid]);et=np.array([r['theta_error'] for r in valid]);
            result.update(y_rmse=float(np.sqrt(np.mean(ey**2))),theta_rmse=float(np.sqrt(np.mean(et**2))),joint_pass=int(np.sum((np.abs(ey)<=.09)&(np.abs(et)<=.09))),qr_theta_std_median=float(np.median([r['qr_theta_std'] for r in valid])),points_median=float(np.median([r['points'] for r in valid])))
        out[label]=result
    s=out['all'];out['selection_score']=(s.get('y_rmse',1e3)/.09)**2+(s.get('theta_rmse',1e3)/.09)**2+1e4*(s['n']-s['success'])/max(1,s['n'])
    return out

def main():
    from run import REPO
    parser=argparse.ArgumentParser();parser.add_argument('--repo',type=Path,default=REPO);args=parser.parse_args()
    root=HERE/'results';frames=read(root/'frames.csv');raw=read(root/'points.csv');candidates=read(root/'candidates.csv')
    extra=root/'baseline-extra.csv'
    if extra.exists():raw+=read(extra)
    train,val,test=split_frames(frames)
    split={'inner_train':[r['frame'] for r in train],'inner_validation':[r['frame'] for r in val],'sealed_holdout':[r['frame'] for r in test]}
    (root/'split.json').write_text(json.dumps(split,indent=2))
    baseline_ref={f"{r['payload']}:{r['id']}" for r in candidates if r['frame']=='0' and r['baseline_accepted']=='1'}
    masks={(int(r['frame']),f"{r['payload']}:{r['id']}"):r['baseline_accepted']=='1' for r in candidates}
    methods=sorted({r['method'] for r in raw});maps={m:{} for m in methods};diagnostics={}
    for r in raw:
        m=r['method'];frame=int(r['frame']);key=f"{r['payload']}:{r['id']}"
        if r['valid']=='1' and key in baseline_ref and masks.get((frame,key),False):
            maps[m].setdefault(frame,{})[key]=np.array([float(r['x']),float(r['y'])])
    for m in methods:
        rs=[r for r in raw if r['method']==m]
        diagnostics[m]={'candidates':len(rs),'valid':sum(r['valid']=='1' for r in rs),'nonconverged':sum(r['converged']=='0' for r in rs),'boundary':sum(r['boundary']=='1' for r in rs),'mean_ms':float(np.mean([float(r['ms']) for r in rs])),'condition_median':float(np.median([float(r['condition']) for r in rs]))}
    inner={};inner_rows=[]
    for m in methods:
        s,rows=evaluate(train,val,maps[m]);inner[m]=s;inner_rows.extend([{'method':m,**r} for r in rows])
    # Freeze one configuration per family before loading any outer outcomes.
    tuned=min([m for m in methods if m.startswith('cornerSubPix')],key=lambda m:inner[m]['selection_score'])
    selected=list(dict.fromkeys(['cornerSubPix',tuned]+[min([m for m in methods if m.startswith(prefix)],key=lambda m:inner[m]['selection_score']) for prefix in ['model_','fourier_']]))
    (root/'selection.json').write_text(json.dumps({'selected':selected,'inner_validation':inner,'criterion':'sum squared Y/theta RMSE normalized by .09, plus failure penalty; degree 1 fixed'},indent=2))
    write(root/'inner-validation.csv',inner_rows)
    outer={};outer_rows=[]
    for m in selected:
        s,rows=evaluate(train+val,test,maps[m]);outer[m]=s;outer_rows.extend([{'method':m,**r} for r in rows])
    write(root/'holdout.csv',outer_rows)
    # Identical surviving identities per frame for the three selected methods.
    common={}
    for f in range(len(frames)):
        common[f]=set.intersection(*(set(maps[m].get(f,{})) for m in selected))
    matched={}
    for m in selected:
        mm={f:{k:p for k,p in points.items() if k in common[f]} for f,points in maps[m].items()}
        s,rows=evaluate(train+val,test,mm);matched[m]=s
        write(root/(m+'-common-holdout.csv'),rows)
    synthetic=read(root/'synthetic.csv');syn={}
    for m in methods:
        syn[m]={}
        for tex in ['0','1']:
            rs=[r for r in synthetic if r['method']==m and r['texture']==tex];ok=[r for r in rs if r['valid']=='1'];errors=np.array([float(r['error']) for r in ok])
            syn[m][tex]={'n':len(rs),'valid':len(ok),'rmse':float(np.sqrt(np.mean(errors**2))) if len(ok) else None,'max_error':float(np.max(errors)) if len(ok) else None}
    result={'selection':selected,'split_counts':{'train':len(train),'validation':len(val),'holdout':len(test)},'holdout':outer,'matched_identity_holdout':matched,'diagnostics':diagnostics,'synthetic':syn,'limits':['nominal commands are not independent measured truth','six groups are session candidates','no mixed-motion or stationary repeats','model implementation is a numerical Gaussian-convolution adaptation, not verified reproduction of 2011 approximations','Fourier raw follows finite truncation of Eq 23; centered variant is explicitly an adaptation','holdout was inspected during an engineering run before constrained-optimizer repair; final results reuse it and are not a pristine blind test','same baseline acceptance mask is applied before each method convergence mask; operational pipeline acceptance is evaluated separately']}
    (root/'summary.json').write_text(json.dumps(result,indent=2))
    manifest={'sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [HERE/'compare.cpp',args.repo/'include/qr_refinement.hpp',HERE/'evaluate.py',root/'points.csv',root/'candidates.csv',root/'split.json']}}
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps({'selected':selected,'holdout':outer,'matched':matched},indent=2),flush=True)

if __name__=='__main__':main()
