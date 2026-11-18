from rt import *; from style import table; from fmt import *
import json, copy
C=[r for r in F if r['set'] in ('C0','C2')]; C.sort(key=lambda r:r['file'])
cm=[[ (r[f's{k}'],r[f'd{k}']) for k in range(4)] for r in C]   # 행 = 센터 j, 열 = 기준 k
cfeat={k:np.array([cm[j][k] for j in range(4)]) for k in range(4)}
meanref=np.array([[np.mean([cm[j][k][0] for k in range(4)]),np.mean([cm[j][k][1] for k in range(4)])] for j in range(4)])
c_mean=meanref.mean(0)
SETS=['S1','S2','S3','S4']
def run(ref,c_fixed,os_=1,od=1):
    ALL=samples(ref=ref); M=fit(ALL,os_,od,c_fixed=c_fixed); P=predict(M,copy.deepcopy(ALL))
    out={'model':M,'self':{g:metrics([x for x in P if g=='ALL' or x['set']==g]) for g in ['ALL']+SETS}}
    pool=[]
    for s in SETS:
        Mt=fit([x for x in ALL if x['set']!=s],os_,od,c_fixed=c_fixed); pool+=predict(Mt,copy.deepcopy([x for x in ALL if x['set']==s]))
    out['loso']={g:metrics([x for x in pool if g=='ALL' or x['set']==g]) for g in ['ALL']+SETS}
    return out
cases=[]
for k in range(4): cases.append((f'{REFN[k]} 단일 기준 · 상수항 = 센터 특징(0)',k,(0.0,0.0)))
cases.append(('센터 4장 평균 기준 · 상수항 = 4장 평균 특징',"mean",tuple(c_mean)))
for k in range(4): cases.append((f'{REFN[k]} 단일 기준 · 상수항 자유',k,None))
cases.append(('센터 4장 평균 기준 · 상수항 자유','mean',None))
cases.append(('(기존 관례) 같은 날 기준 · 상수항 자유','sameday',None))
res={}
for o in [(1,1),(3,2)]:
    for nm,ref,cf in cases: res[f'{nm}|s{o[0]}d{o[1]}']=run(ref,cf,*o)
json.dump(dict(res=res,cm=cm,c_mean=list(c_mean)),open('/home/claude/rt/work/E3.json','w'),default=float,indent=1)
for k,v in res.items(): print(k[:45].ljust(45),'self',v['self']['ALL']['spec'],v['self']['ALL']['half'],'loso',v['loso']['ALL']['spec'],v['loso']['ALL']['half'],' '.join(str(v['loso'][s]['spec']) for s in SETS), 'c',round(v['model']['s']['c'],3),round(v['model']['d']['c'],3))
print(np.round(np.array(cm),3)); print('c_mean',c_mean)
