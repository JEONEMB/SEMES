"""Fit original feature vectors; theta inference never consumes the observed mean."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np,pandas as pd
from sequential import estimate

def run(data,out,repo):
    out.mkdir(parents=True,exist_ok=True);frames=pd.read_csv(data/'frames.csv');pts=pd.read_csv(data/'points.csv');pts=pts[pts.accepted==1].copy();pts['key']=pts.payload+':'+pts.id.astype(str)
    maps={int(i):g.set_index('key')[['x','y']].to_dict('index') for i,g in pts.groupby('frame')}
    maps={i:{k:[p['x'],p['y']] for k,p in d.items()} for i,d in maps.items()}
    baseline=int(frames[frames.path.str.endswith('center.bmp')].frame.iloc[0]);q=maps[baseline]
    frames['name']=frames.path.map(lambda p:Path(p).stem);frames=frames[frames.name!='center'].copy();frames['axis']=frames.name.str[0];frames['nominal']=frames.name.str[1:].astype(float)
    frames['group']=[a+('+' if v>0 else '-')+(str(int(round(abs(v)*100))%10) if a=='Y' else '') for a,v in zip(frames.axis,frames.nominal)]
    groups=sorted(frames.group.unique());summary={}
    for split in ['interleaved','blocked']:
        hold={}
        for group,g in frames.groupby('group'):
            g=g.sort_values('nominal');n=len(g)
            for j,i in enumerate(g.frame):hold[i]=(j%3==0 if split=='interleaved' else n//3<=j<2*n//3)
        train=frames[~frames.frame.map(hold)];test=frames[frames.frame.map(hold)]
        for degree in [1,2]:
            model={'groups':groups,'points':{},'theta_range_deg':[-1,1],'theta_uses_observed_mean':False,'real_mixed_validated':False,'split':split,'theta_degree':degree}
            for key,ref in q.items():
                X=[];Z=[]
                for r in train.itertuples():
                    if key not in maps[r.frame]:continue
                    y=r.nominal if r.axis=='Y' else 0.;t=r.nominal if r.axis=='T' else 0.
                    X.append([float(g==r.group) for g in groups]+[y,t]+([t*t] if degree==2 else []));Z.append(np.array(maps[r.frame][key])-ref)
                X=np.array(X);Z=np.array(Z)
                if len(X)<48 or np.linalg.matrix_rank(X)<len(groups)+degree+1:continue
                c=np.linalg.lstsq(X,Z,rcond=None)[0]
                if degree==1:c=np.vstack([c,[0.,0.]])
                model['points'][key]={'reference':ref,'coefficients':c.tolist()}
            rows=[]
            for r in test.itertuples():
                truth=[r.nominal,0.] if r.axis=='Y' else [0.,r.nominal]
                try:
                    est=estimate(model,maps[r.frame],r.group);oracle=estimate(model,maps[r.frame],r.group,oracle_theta=truth[1])
                    rows.append(dict(name=r.name,axis=r.axis,group=r.group,nominal=r.nominal,status='ok',
                        y_error=est['y_mm']-truth[0],theta_error=est['theta_deg']-truth[1],oracle_y_error=oracle['y_mm']-truth[0],**est))
                except ValueError as e:rows.append(dict(name=r.name,axis=r.axis,group=r.group,nominal=r.nominal,status=str(e)))
            predictions=pd.DataFrame(rows);label=f'{split}_degree{degree}'
            predictions.to_csv(out/(label+'_holdout.csv'),index=False);(out/(label+'_model.json')).write_text(json.dumps(model,indent=2))
            result={'features_fitted':len(model['points']),'train':len(train),'test':len(test),'axes':{}}
            for axis,g in predictions.groupby('axis'):
                valid=g[g.status=='ok'];result['axes'][axis]={'n':len(g),'success':len(valid),'y_rmse':float(np.sqrt(np.mean(valid.y_error**2))),'theta_rmse':float(np.sqrt(np.mean(valid.theta_error**2))),'oracle_theta_y_rmse':float(np.sqrt(np.mean(valid.oracle_y_error**2))),'joint_pass':int(((abs(valid.y_error)<=.09)&(abs(valid.theta_error)<=.09)).sum()),'theta_boundary':int(valid.theta_at_boundary.sum())}
            summary[label]=result;print(label,json.dumps(result),flush=True)
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    manifest={'base_commit':'28f6759','extraction':json.loads((data/'parallel_run.json').read_text()),'sha256':{}}
    for rel in ['src/alignment.cpp','src/qr_features.cpp','src/qr_model.cpp','src/qr_decode.cpp','tool/sequential/sequential.py','tool/sequential/evaluate.py']:
        manifest['sha256'][rel]=hashlib.sha256((repo/rel).read_bytes()).hexdigest()
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--repo',type=Path,default=Path('.'));a=p.parse_args();run(a.data,a.output,a.repo)
