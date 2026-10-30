"""Training-only smooth multi-reference curves. No heldout image becomes an anchor."""
from pathlib import Path
import sys,json,itertools
import numpy as np,pandas as pd
from scipy.interpolate import BSpline
from sklearn.model_selection import StratifiedKFold
H=Path(__file__).resolve().parent;B=H.parent
sys.path[:0]=[str(B),str(B/'multiref_4fold_20261021')]
import zero_experiment as Z
from experiment import enrich,stats
names=np.load(B/'center_patch_v2_20261021/tracks.npz')['names'];m=Z.manifest(names);ix={n:i for i,n in enumerate(names)};ids=np.flatnonzero(m.axis!='C')
features={'patch':np.load(B/'center_patch_v2_20261021/v2_patch630_features.npy'),'quality':np.load(B/'center_patch_v2_20261021/v2_patch630_quality_features.npy'),'gn31':np.load(B/'patch_windows_20261021/gn31_sigma0.8_features.npy')}
configs=list(itertools.product(features,[0,2,4],[.01,1.,100.],[.05,.5]))
def fit(config,tr):
    feature,nknots,reg,alpha=config;X=features[feature];branches={};residual=[]
    for k in np.unique(m.session.iloc[tr]):
        ii=tr[m.session.to_numpy()[tr]==k];v=m.value.to_numpy()[ii];lo,hi=float(v.min()),float(v.max());scale=hi-lo
        t=(v-lo)/scale;knots=np.r_[np.zeros(4),np.linspace(0,1,nknots+2)[1:-1],np.ones(4)];n=len(knots)-4
        spline=BSpline(knots,np.eye(n),3,extrapolate=True);A=spline(t);pen=np.diff(np.eye(n),n=2,axis=0)
        coef=np.linalg.solve(A.T@A+reg*pen.T@pen+1e-10*np.eye(n),A.T@X[ii]);residual.append(X[ii]-A@coef)
        branches[k]=(k[0],lo,scale,BSpline(knots,coef,3,extrapolate=True))
    R=np.vstack(residual);S=R.T@R/len(R);std=np.sqrt(np.maximum(np.diag(S),1e-14));C=S/std[:,None]/std[None,:];C=(1-alpha)*C+alpha*np.eye(len(std));W=np.linalg.cholesky(np.linalg.inv(C)).T/std[None,:]
    return branches,W
def predict(config,tr,te):
    branches,W=fit(config,tr);X=features[config[0]][te];choices=[]
    for k,(a,lo,scale,spline) in branches.items():
        # Bound uses training range plus 20%, never heldout command extrema.
        grid=np.linspace(-.2,1.2,81);q=lo+grid*scale;ok=q*(1 if k.endswith('+') else -1)>=0;grid=grid[ok]
        field=spline(grid);cost=np.sum(((X[:,None,:]-field[None])@W.T)**2,axis=2);t=grid[np.argmin(cost,axis=1)]
        deriv=spline.derivative()
        for _ in range(12):
            r=(X-spline(t))@W.T;j=deriv(t)@W.T;step=np.sum(r*j,axis=1)/np.maximum(np.sum(j*j,axis=1),1e-20);t=np.clip(t+np.clip(step,-.1,.1),-.2,1.2)
        v=lo+t*scale;c=np.sum(((X-spline(t))@W.T)**2,axis=1);c=np.where(v*(1 if k.endswith('+') else -1)>=0,c,np.inf);choices.append((k,a,v,c))
    best=np.argmin(np.stack([r[3] for r in choices],axis=1),axis=1);rows=[]
    for t,i in enumerate(te):
        k,a,v,cost=choices[best[t]];rows.append(dict(file=m.file.iloc[i],axis=m.axis.iloc[i],value=m.value.iloc[i],session=m.session.iloc[i],pred_axis=a,pred=v[t],branch=k,cost=cost[t]))
    return enrich(pd.DataFrame(rows))
def main():
    folds=json.loads((B/'multiref_4fold_20261021/extended/splits.json').read_text());(H/'splits.json').write_text(json.dumps(folds,indent=2));mapping={f's{k:03d}':dict(feature=c[0],interior_knots=c[1],smoothness_penalty=c[2],covariance_shrinkage=c[3]) for k,c in enumerate(configs)};(H/'configs.json').write_text(json.dumps(mapping,indent=2))
    rows=[];nested=[];selections=[]
    for f,s in enumerate(folds):
        tr=np.array([ix[n] for n in s['train']]);te=np.array([ix[n] for n in s['test']]);inner=list(StratifiedKFold(3,shuffle=True,random_state=100+f).split(tr,m.session.iloc[tr]));rank=[];local=[]
        for k,c in enumerate(configs):
            ip=pd.concat([predict(c,tr[a],tr[b]) for a,b in inner]);ss=stats(ip);rank.append((ss['failures'],max(ss['Y_max'],ss['T_max']),k));local.append(predict(c,tr,te).assign(config=f's{k:03d}',protocol='oof4',fold=f))
        chosen=min(rank)[2];rows+=local;nested.append(local[chosen].assign(config='nested_selected'));selections.append(dict(fold=f,config=f's{chosen:03d}',rank=sorted(rank)));print('fold',f,'chosen',chosen,flush=True)
    for k,c in enumerate(configs):rows.append(predict(c,ids,ids).assign(config=f's{k:03d}',protocol='resub280',fold=-1))
    p=pd.concat(rows+nested,ignore_index=True);p.to_csv(H/'predictions.csv',index=False);p[~p['pass']].to_csv(H/'failures.csv',index=False)
    s=pd.DataFrame([dict(config=k,protocol=t,**mapping.get(k,{}),**stats(g)) for (k,t),g in p.groupby(['config','protocol'])]);s.to_csv(H/'summary.csv',index=False)
    pd.DataFrame([dict(config=k,fold=f,**stats(g)) for (k,f),g in p[p.protocol=='oof4'].groupby(['config','fold'])]).to_csv(H/'fold_results.csv',index=False)
    (H/'nested_selection.json').write_text(json.dumps(selections,indent=2));print(s[s.protocol=='oof4'].sort_values(['failures','Y_max']).head(10).to_string(index=False));print(s[s.config=='nested_selected'].to_string(index=False))
if __name__=='__main__':main()
