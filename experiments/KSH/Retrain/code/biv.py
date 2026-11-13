"""참고 : Y·Θ 2변수 다항식 (Y², YΘ 등 포함) — C++ 코드 확장이 필요한 형태"""
from rt import *
import numpy as np, copy, json
def terms(n): return [(i,j) for t in range(n+1) for i in range(t+1) for j in [t-i]]
def X(Y,T,n): return np.column_stack([Y**i*T**j for i,j in terms(n)])
def fitb(S,n):
    Y=np.array([r['Y'] for r in S]);T=np.array([r['T'] for r in S])
    cs,*_=np.linalg.lstsq(X(Y,T,n),np.array([r['s'] for r in S]),rcond=None); cd,*_=np.linalg.lstsq(X(Y,T,n),np.array([r['d'] for r in S]),rcond=None)
    return dict(n=n,cs=cs,cd=cd)
def predb(M,S):
    n=M['n'];tm=terms(n)
    L=fit(S,1,1)  # 초기값용 1차식
    for r in S:
        y,t=invert(L,r['s'],r['d'])
        for _ in range(50):
            f=np.array([X(np.array([y]),np.array([t]),n)[0]@M['cs']-r['s'], X(np.array([y]),np.array([t]),n)[0]@M['cd']-r['d']])
            dY=np.array([i*y**(i-1)*t**j if i>0 else 0 for i,j in tm]); dT=np.array([j*y**i*t**(j-1) if j>0 else 0 for i,j in tm])
            J=np.array([[dY@M['cs'],dT@M['cs']],[dY@M['cd'],dT@M['cd']]]); st=np.linalg.solve(J,-f); y+=st[0]; t+=st[1]
            if abs(st[0])<1e-10 and abs(st[1])<1e-11: break
        r['ye'],r['te']=y,t; r['ey']=(y-r['Y'])*1000; r['et']=t-r['T']
    return S
ALL=samples(); out={}
for n in [1,2,3]:
    M=fitb(ALL,n); P=predb(M,copy.deepcopy(ALL)); pool=[]
    for s in ['S1','S2','S3','S4']:
        Mt=fitb([r for r in ALL if r['set']!=s],n); pool+=predb(Mt,copy.deepcopy([r for r in ALL if r['set']==s]))
    out[n]=dict(terms=[f'Y^{i}Θ^{j}' for i,j in terms(n)],cs=list(M['cs']),cd=list(M['cd']),self=metrics(P),loso=metrics(pool),loso_sets={s:metrics([r for r in pool if r['set']==s]) for s in ['S1','S2','S3','S4']})
    a=out[n]['self'];l=out[n]['loso']; print(n,'self',a['spec'],a['goal'],a['half'],round(a['yrms'],1),round(a['trms'],4),'| loso',l['spec'],l['goal'],l['half'],round(l['yrms'],1),round(l['trms'],4))
json.dump(out,open('/home/claude/rt/work/E1_biv.json','w'),default=float,indent=1)
