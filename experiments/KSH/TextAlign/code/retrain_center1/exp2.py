from lib import *
import numpy as np, pickle, json
exec(open('exp1.py').read().split("cands={}")[0])
def slopes_plus_c(parts_ref1, s1df, os_, od, w_bal=True, c_src_ref=0):
    M=fe_fit(parts_ref1,1,os_,od,anchor='zero',w_bal=w_bal)
    # intercept: S1 (same-day center ref0) residual mean with common slopes
    for key in ['s','d']:
        m=M[key]; Y=s1df.Y.values;T=s1df['T'].values
        pred=m['a']*Y+sum(b*T**(k+1) for k,b in enumerate(m['b']))
        m['c']=float(np.mean(s1df[f'{key}{c_src_ref}'].values-pred))
    return M
C2={}
for o in [(1,1),(1,2),(1,3),(2,2)]:
    C2[f'E 기울기=1+4차(세트별상수·균형) / 상수=1차-같은날센터 s{o[0]}d{o[1]}']=slopes_plus_c([S1,S4],S1,*o)
    C2[f'E2 같은방식, 상수만 1차 0.1배수에서 s{o[0]}d{o[1]}']=slopes_plus_c([S1,S4],S1_10,*o)
res={}
for name,M in C2.items():
    r={en:met(predict(M,df,REF)) for en,df in EV.items()}; res[name]=r; m=r['S2+S3']
    print(f"{name:55s} Ymean={m['ym']:+6.1f} Ysd={m['ysd']:5.1f} Yrms={m['yrms']:5.1f} Ymax={m['ymax']:5.1f} | Tmean={m['tm']:+.4f} Trms={m['trms']:.4f} Tmax={m['tmax']:.4f} | spec={m['spec']}/291 half={m['half']}  S2={r['S2']['spec']}/57 S3={r['S3']['spec']}/234")
    print('   coef s:',{k:np.round(v,5) for k,v in M['s'].items() if k!='cs'},' d:',{k:np.round(v,5) for k,v in M['d'].items() if k!='cs'})
pickle.dump(C2,open('cands2.pkl','wb'))
json.dump({k:{e:{kk:float(vv) for kk,vv in m.items()} for e,m in v.items()} for k,v in res.items()},open('exp2.json','w'),ensure_ascii=False,indent=1)
