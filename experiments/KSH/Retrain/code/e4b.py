from rt import *
import json, copy
ALL=samples(); out={}
for o in [(1,1),(3,2)]:
    tag=f's{o[0]}d{o[1]}'
    tr=[r for r in ALL if r['q'] in ('Q1','AX')]; te=[r for r in ALL if r['q'] in ('Q2','Q3','Q4')]
    M=fit(tr,*o); P=predict(M,copy.deepcopy(te))
    out[tag]={'n_train':len(tr),'D':{g:metrics([x for x in P if g=='ALL' or x['q']==g]) for g in ['ALL','Q2','Q3','Q4']},'model':M}
    # 같은 Q2~Q4 를 4차 Q1 까지 빼고(1·2·3차만) 학습한 경우
    tr2=[r for r in ALL if r['set']!='S4']; P2=predict(fit(tr2,*o),copy.deepcopy(te))
    out[tag]['D2']={g:metrics([x for x in P2 if g=='ALL' or x['q']==g]) for g in ['ALL','Q2','Q3','Q4']}
    for k in ['D','D2']: print(tag,k,' '.join('%s %d/%d(%.1f,%.4f)'%(g,out[tag][k][g]['spec'],out[tag][k][g]['n'],out[tag][k][g]['yrms'],out[tag][k][g]['trms']) for g in ['ALL','Q2','Q3','Q4']))
json.dump(out,open('/home/claude/rt/work/E4b.json','w'),default=float,indent=1)
