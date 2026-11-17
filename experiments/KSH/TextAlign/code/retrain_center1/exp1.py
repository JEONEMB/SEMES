from lib import *
import numpy as np, json, itertools
REF=1  # center1
M0=load_yml('/mnt/user-data/uploads/PJT/config/model_s1d1.yml')
S1=D[D.set=='S1']; S4=D[D.set=='S4']
S1_10=S1[(S1.Y==0)|(np.round(S1.Y*100)%10==0)]
EV={'S2':D[D.set=='S2'],'S3':D[D.set=='S3'],'S2+S3':D[D.set.isin(['S2','S3'])]}
def fe_fit(parts,ref,os_,od,anchor='zero',w_bal=False):
    # common slopes, per-set intercept (fixed effects)
    res={}
    for key,o in [('s',os_),('d',od)]:
        rows=[];v=[];W=[]
        for i,df in enumerate(parts):
            Y=df.Y.values;T=df['T'].values
            A=np.column_stack([ (np.arange(len(parts))==i).astype(float)[None,:].repeat(len(df),0), Y]+[T**k for k in range(1,o+1)])
            rows.append(A); v.append(df[f'{key}{ref}'].values); W.append(np.full(len(df),1/len(df)) if w_bal else np.ones(len(df)))
        A=np.vstack(rows); v=np.concatenate(v); W=np.concatenate(W); sw=np.sqrt(W)
        x,*_=np.linalg.lstsq(A*sw[:,None],v*sw,rcond=None)
        r=A@x-v; k=len(parts)
        cs=x[:k]
        c={'zero':0.0,'mean':cs.mean()}[anchor]
        res[key]=dict(c=c,a=x[k],b=list(x[k+1:]),sigma=max(1e-6,np.sqrt(np.mean(r**2))),cs=list(cs))
    return res
cands={}
cands['A 기존 배포 s1d1 (1차 0.1배수, center 기준)']=M0
for o in [(1,1),(1,2)]:
    cands[f'B 통합적합 1+4차 s{o[0]}d{o[1]} (상수 자유)']=fit(pd.concat([S1,S4]),REF,*o)
    for anc in ['zero','mean']:
        cands[f'C 세트별 상수 1+4차 s{o[0]}d{o[1]} 상수={anc}']=fe_fit([S1,S4],REF,*o,anchor=anc)
        cands[f'C\' 세트별상수+세트균형 s{o[0]}d{o[1]} 상수={anc}']=fe_fit([S1,S4],REF,*o,anchor=anc,w_bal=True)
    cands[f'D 1차(0.1배수)+4차 세트별상수 s{o[0]}d{o[1]} 상수=0']=fe_fit([S1_10,S4],REF,*o,anchor='zero')
out={}
for name,M in cands.items():
    out[name]={}
    for en,df in EV.items():
        p=predict(M,df,REF); out[name][en]=met(p)
    m=out[name]['S2+S3']
    print(f"{name:45s} n={m['n']} Ymean={m['ym']:+6.1f} Ysd={m['ysd']:5.1f} Yrms={m['yrms']:5.1f} Ymax={m['ymax']:5.1f} | Tmean={m['tm']:+.4f} Trms={m['trms']:.4f} Tmax={m['tmax']:.4f} | spec={m['spec']} half={m['half']}  S2spec={out[name]['S2']['spec']}/57 S3spec={out[name]['S3']['spec']}/234")
json.dump({k:{e:{kk:float(vv) for kk,vv in m.items()} for e,m in v.items()} for k,v in out.items()},open('exp1.json','w'),ensure_ascii=False,indent=1)
import pickle; pickle.dump(cands,open('cands.pkl','wb'))
