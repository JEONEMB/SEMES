from rt import *
import json, itertools, copy
SETS=['S1','S2','S3','S4']
ALL=samples(tuple(SETS))
res={'n':len(ALL),'combos':{}}
def evalsets(M, S):
    P=predict(M, copy.deepcopy(S)); out={'ALL':metrics(P)}
    for s in SETS: out[s]=metrics([r for r in P if r['set']==s])
    for q in ['Q1','Q2','Q3','Q4','AX']: out[q]=metrics([r for r in P if r['q']==q])
    return out
for os_,od in itertools.product([1,2,3],[1,2,3]):
    M=fit(ALL,os_,od); self_=evalsets(M,ALL)
    # LOSO : 세트 하나를 빼고 학습 → 뺀 세트 평가
    loso={}; pool=[]
    for s in SETS:
        Mt=fit([r for r in ALL if r['set']!=s],os_,od); P=predict(Mt,copy.deepcopy([r for r in ALL if r['set']==s])); loso[s]=metrics(P); pool+=P
    loso['ALL']=metrics(pool)
    res['combos'][f's{os_}d{od}']=dict(model=M,self=self_,loso=loso)
json.dump(res,open('/home/claude/rt/work/E1.json','w'),default=float,indent=1)
for k,v in res['combos'].items():
    a=v['self']['ALL']; l=v['loso']['ALL']
    print(k,'self spec %d/%d goal %d half %d yrms %.1f trms %.4f | LOSO spec %d goal %d half %d yrms %.1f trms %.4f'%(a['spec'],a['n'],a['goal'],a['half'],a['yrms'],a['trms'],l['spec'],l['goal'],l['half'],l['yrms'],l['trms']))
