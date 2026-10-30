"""구조점630+위·아래 선 vs ECC(QR54)+위·아래 선, 같은 341장·같은 fold·같은 중첩 선택.

A틀: joint_yt_visual_feedback_20261023/evaluate_static.py (337/341을 낸 평가)를 그대로 따르고
     QR 블록만 바꾼다. 상호작용 항의 shift는 각 QR 블록의 x 변위 평균/100.
B틀: joint_yt_erroraudit_20261022/robust/exploratory_nested.py (구조점 244/341)를 그대로 따르고
     선 48값을 특징에 덧붙인다.
측정은 기존 캐시(features.npz)를 다시 쓴다. 예측 함수는 특징만 받는다.
"""
from pathlib import Path
import sys,json,time,importlib.util
import numpy as np,pandas as pd
from threadpoolctl import threadpool_limits
H=Path(__file__).resolve().parent;ROOT=H.parent
VF=ROOT/'joint_yt_visual_feedback_20261023';GEN=ROOT/'joint_yt_generalization_20261022';ROB=ROOT/'joint_yt_erroraudit_20261022/robust'
sys.path.insert(0,str(GEN))
from forward_model import ForwardGLS

def _mod(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
VFM=_mod('vf_models',VF/'models.py');RBM=_mod('rob_model',ROB/'robust_model.py')
CONFIGS=VFM.CONFIGS

def metric(y,p):
    e=abs(np.asarray(y)-np.asarray(p));finite=np.isfinite(e).all(axis=1);bad=(~finite)|np.any(e>.09+1e-12,axis=1);e2=e[finite]
    return dict(n=len(y),failures=int((~finite).sum()),passed=int((~bad).sum()),Y_MAE=float(e2[:,0].mean()),Y_max=float(e2[:,0].max()),T_MAE=float(e2[:,1].mean()),T_max=float(e2[:,1].max()))

def shift_map(qr_dims):
    """[X, X*shift], shift = 앞쪽 QR 블록(x,y 교대)의 x 평균/100. 원래 두 feature_map과 같은 꼴."""
    def f(X):
        X=np.atleast_2d(np.asarray(X,float));X=X.reshape(len(X),-1);s=X[:,:qr_dims:2].mean(axis=1,keepdims=True)/100;return np.c_[X,X*s]
    return f

class Obs:
    """ObservationModel와 같은 학습식."""
    def __init__(self,config,qr_dims):self.config=dict(config);self.fmap=shift_map(qr_dims)
    def fit(self,X,y,meta):
        X=self.fmap(X);y=np.asarray(y);dates=meta.dataset.to_numpy();sel=(dates==20261020) if self.config['scope']=='mixed' else np.ones(len(y),bool)
        dates=dates[sel];w=np.array([1/np.sum(dates==d) for d in dates])
        self.model=ForwardGLS(self.config['degree'],self.config['shrink']).fit(X[sel],y[sel],weights=w);return self
    def predict(self,X):return self.model.predict(self.fmap(X))

def split(m,fold):
    f=m.fold.to_numpy()
    return (np.flatnonzero(f!=fold),np.flatnonzero(f==fold)) if fold<4 else (np.arange(len(m)),np.arange(len(m)))

def frame_a(name,X,qr_dims,m,y,old,log):
    out=[];audit=[]
    for fold in range(5):
        t0=time.monotonic()
        inner=old[fold]['inner_splits'] if fold<4 else old[-1]['full_selection']['inner_splits']
        tr,te=split(m,fold);scores=[]
        for k,c in enumerate(CONFIGS):
            err=[];dates=[]
            for s in inner:
                it=np.array(s['train']);iv=np.array(s['validation']);assert not set(m.iloc[it].cv_group)&set(m.iloc[iv].cv_group)
                p=Obs(c,qr_dims).fit(X[it],y[it],m.iloc[it]).predict(X[iv]);err.extend(abs(p-y[iv]));dates.extend(m.iloc[iv].dataset)
            e=np.array(err);d=np.array(dates);bad=np.any(e>.09+1e-12,axis=1)
            scores.append(dict(candidate=k,config=c,balanced_failures=float(np.mean([bad[d==dd].mean() for dd in np.unique(d)])),maximum=float(e.max()),mean=float(e.mean())))
        ch=min(scores,key=lambda r:(r['balanced_failures'],r['maximum'],r['mean']))['candidate']
        p=Obs(CONFIGS[ch],qr_dims).fit(X[tr],y[tr],m.iloc[tr]).predict(X[te])
        out.append(rows(m,te,p,name,'development_oof' if fold<4 else 'training_reapplication'));audit.append(dict(condition=name,fold=fold,chosen=ch,scores=scores))
        log(f'{name} fold{fold} cfg{ch} {metric(y[te],p)} {time.monotonic()-t0:.0f}s')
    return out,audit

def frame_b(name,X,m,y,prev,log):
    """RobustBaggedForward는 모듈 전역 feature_map을 부른다. 구조점 1260 뒤에 선이 붙어도 shift는 구조점 x만."""
    RBM.feature_map=shift_map(1260)
    out=[];audit=[];new=m.dataset.to_numpy()==20261020
    def select(a):
        sc=[]
        for k,c in enumerate(RBM.CONFIGS):
            P=[];T=[]
            for s in a['inner_splits']:
                it=np.asarray(s['train']);iv=np.asarray(s['validation']);iv=iv[new[iv]]
                P.append(RBM.RobustBaggedForward(c).fit(X[it],y[it],m.iloc[it]).predict(X[iv]));T.append(y[iv])
            e=np.abs(np.vstack(P)-np.vstack(T));sc.append(dict(candidate=k,config=c,failures=int(np.any(e>.09+1e-12,axis=1).sum()),maximum=float(e.max()),mean=float(e.mean())))
        return min(sc,key=lambda r:(r['failures'],r['maximum'],r['mean']))['candidate'],sc
    for f in range(5):
        t0=time.monotonic()
        a=next(x for x in prev if x.get('fold')==f) if f<4 else next(x['full_selection'] for x in prev if 'full_selection' in x)
        tr,te=split(m,f)
        k,sc=select(a);p=RBM.RobustBaggedForward(RBM.CONFIGS[k]).fit(X[tr],y[tr],m.iloc[tr]).predict(X[te])
        out.append(rows(m,te,p,name,'development_oof' if f<4 else 'training_reapplication'));audit.append(dict(condition=name,fold=f,chosen=k,scores=sc))
        log(f'{name} fold{f} cfg{k} {metric(y[te],p)} {time.monotonic()-t0:.0f}s')
    return out,audit

def rows(m,ix,p,name,protocol):
    r=m.iloc[ix][['index','dataset','file','Y','T','scope','condition','fold','cv_group']].copy();r['condition_name']=name;r['protocol']=protocol
    r['predY'],r['predT']=p.T;r['eY']=abs(r.predY-r.Y);r['eT']=abs(r.predT-r['T']);r['passed']=(r.eY<=.09+1e-12)&(r.eT<=.09+1e-12);return r

def main():
    which=sys.argv[1:] or ['A','B'];tag=''.join(which)
    logf=open(H/f'evaluate_{tag}.log','a',encoding='utf-8')
    def log(s):print(s,flush=True);logf.write(s+'\n');logf.flush()
    m=pd.read_csv(VF/'manifest.csv');y=m[['Y','T']].to_numpy()
    g=np.load(GEN/'measurements/features.npz');l=np.load(VF/'lines/features.npz');v=np.load(VF/'features.npz')
    assert (g['index']==m['index'].values).all() and (l['index']==m['index'].values).all() and np.abs(g['Y']-y).max()==0
    assert v['success_both'].all() and l['success_upper'].all() and l['success_lower'].all()
    lines=np.c_[l['X_upper'],l['X_lower']];ecc=v['X_qr'];S=g['D'].reshape(341,-1)
    assert np.allclose(np.c_[ecc,lines],v['X_both'])
    old=json.loads((ROOT/'joint_yt_disentangle_20261022/selection.json').read_text())
    out=[];audit=[]
    if 'A' in which:
        for name,X,q in [('A_ecc54_lines48',np.c_[ecc,lines],54),('A_struct1260_lines48',np.c_[S,lines],1260),('A_ecc54_only',ecc,54),('A_struct1260_only',S,1260)]:
            o,a=frame_a(name,X,q,m,y,old,log);out+=o;audit+=a
    if 'B' in which:
        for name,X in [('B_struct1260_only',S),('B_struct1260_lines48',np.c_[S,lines])]:
            o,a=frame_b(name,X,m,y,old,log);out+=o;audit+=a
    p=pd.concat(out);p.to_csv(H/f'predictions_{tag}.csv',index=False,encoding='utf-8-sig')
    (H/f'selection_{tag}.json').write_text(json.dumps(audit,indent=1,default=str),encoding='utf-8')
    rs=[]
    for (c,pr),gg in p.groupby(['condition_name','protocol'],sort=False):
        for ds in ['all',20261014,20261020]:
            z=gg if ds=='all' else gg[gg.dataset==ds];rs.append(dict(condition_name=c,protocol=pr,dataset=ds,**metric(z[['Y','T']],z[['predY','predT']])))
    s=pd.DataFrame(rs);s.to_csv(H/f'summary_{tag}.csv',index=False,encoding='utf-8-sig');log(s.to_string(index=False))

if __name__=='__main__':
    with threadpool_limits(1):main()
