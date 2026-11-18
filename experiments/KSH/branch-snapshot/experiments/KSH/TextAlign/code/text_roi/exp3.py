from lib import *
import numpy as np, pandas as pd, json
A=pd.read_pickle('abs.pkl')
import sys
DK=sys.argv[1] if len(sys.argv)>1 else 'd_abs4'
A=A.rename(columns={'s_abs':'sA',DK:'dA'})
def fe(parts,os_,od,key_s='sA',key_d='dA',bal=True):
    res={}
    for key,o in [(key_s,os_),(key_d,od)]:
        As=[];v=[];W=[]
        for i,df in enumerate(parts):
            Y=df.Y.values;T=df['T'].values
            oh=np.zeros((len(df),len(parts)));oh[:,i]=1
            As.append(np.column_stack([oh,Y]+[T**k for k in range(1,o+1)])); v.append(df[key].values); W.append(np.full(len(df),1/len(df)) if bal else np.ones(len(df)))
        M=np.vstack(As);v=np.concatenate(v);w=np.sqrt(np.concatenate(W))
        x,*_=np.linalg.lstsq(M*w[:,None],v*w,rcond=None); r=M@x-v;k=len(parts)
        res[key[0]]=dict(c=None,a=x[k],b=list(x[k+1:]),sigma=np.sqrt(np.mean(r**2)),cs=list(x[:k]),rsd=[float(np.std(r[sum(len(p) for p in parts[:i]):sum(len(p) for p in parts[:i+1])])) for i in range(k)])
    return res
S1=A[A.set=='S1'];S4=A[A.set=='S4'];C0=A[A.set=='C0'];C2=A[A.set=='C2'];C1=C2[C2.file=='center1']
EV={'S2':A[A.set=='S2'],'S3':A[A.set=='S3']}; EV['S2+S3']=pd.concat(EV.values())
def run(M,label):
    r={}
    for en,df in EV.items():
        df=df.copy(); out=np.array([invert(M,s,d) for s,d in zip(df.sA,df.dA)])
        df['ey']=(out[:,0]-df.Y)*1000; df['et']=out[:,1]-df['T']; r[en]=met(df)
    m=r['S2+S3']; print(f"{label:62s} Ymean={m['ym']:+6.1f} Ysd={m['ysd']:5.1f} Yrms={m['yrms']:5.1f} Ymax={m['ymax']:5.1f} | Tmean={m['tm']:+.4f} Trms={m['trms']:.4f} Tmax={m['tmax']:.4f} | spec={m['spec']}/291 half={m['half']} S2={r['S2']['spec']} S3={r['S3']['spec']}")
    return r
out={}
for o in [(1,1),(1,2)]:
    M=fe([S1,S4],*o)
    print('fit sigma s,d:',round(M['s']['sigma'],4),round(M['d']['sigma'],4),' per-set resid sd s',np.round(M['s']['rsd'],3),'d',np.round(M['d']['rsd'],3),' a_s',round(M['s']['a'],4),'a_d',round(M['d']['a'],4),'b_s',np.round(M['s']['b'],4),'b_d',np.round(M['d']['b'],4))
    # 영점 오프셋(센터 사진 vs 회귀 영점) : 1차 같은날 center 로부터
    off_s=M['s']['cs'][0]-C0.sA.iloc[0]; off_d=M['d']['cs'][0]-C0.dA.iloc[0]
    print('  1차 센터사진 오프셋 (회귀영점-센터) s=%.3f d=%.3f px'%(off_s,off_d))
    for lab,cs,cd in [('a) 영점 텍스트 = center1 측정값 + 1차 오프셋',C1.sA.iloc[0]+off_s,C1.dA.iloc[0]+off_d),
                      ('a2) 영점 텍스트 = center1 측정값 (오프셋 없음)',C1.sA.iloc[0],C1.dA.iloc[0]),
                      ('a3) 영점 = center1·2·3 평균 + 1차 오프셋',C2.sA.mean()+off_s,C2.dA.mean()+off_d),
                      ('b) 영점 다시 안 잡음 (1차 날 값 그대로)',M['s']['cs'][0],M['d']['cs'][0]),
                      ('b2) 영점 다시 안 잡음 (4차 날 값)',M['s']['cs'][1],M['d']['cs'][1])]:
        MM={'s':dict(M['s'],c=cs),'d':dict(M['d'],c=cd)}
        out[f's{o[0]}d{o[1]} {lab}']=run(MM,f's{o[0]}d{o[1]} {lab}')
json.dump({k:{e:{kk:float(vv) for kk,vv in m.items()} for e,m in v.items()} for k,v in out.items()},open('exp3.json','w'),ensure_ascii=False,indent=1)
# day drift of plate: center absolute positions
print(pd.concat([C0,C2])[['file','sA','dA']])
