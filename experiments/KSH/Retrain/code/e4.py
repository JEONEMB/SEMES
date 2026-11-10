from rt import *
import json, copy
ALL=samples(); G=['Q1','Q2','Q3','Q4','AX']
cnt={g:sum(1 for r in ALL if r['q']==g) for g in G}; print(cnt)
def weights(S):
    c={g:sum(1 for r in S if r['q']==g) for g in G}; tgt=np.mean([c[g] for g in ['Q1','Q2','Q3','Q4']])  # 사분면당 같은 총 가중
    tgt=min(c[g] for g in ['Q1','Q2','Q3','Q4'])
    return [1.0 if r['q']=='AX' else tgt/c[r['q']] for r in S]
def subsample(S,rng):
    c={g:[r for r in S if r['q']==g] for g in G}; m=min(len(c[g]) for g in ['Q1','Q2','Q3','Q4'])
    out=list(c['AX'])
    for g in ['Q1','Q2','Q3','Q4']: idx=rng.choice(len(c[g]),m,replace=False); out+=[c[g][i] for i in idx]
    return out
def grp(P): return {g:metrics([x for x in P if g=='ALL' or x['q']==g or x['set']==g]) for g in ['ALL','Q1','Q2','Q3','Q4','AX','S1','S2','S3','S4']}
res={}
for o in [(1,1),(3,2)]:
    tag=f's{o[0]}d{o[1]}'; R={}
    # T1 셀프 테스트 (전체 714장 평가)
    MA=fit(ALL,*o); MB=fit(ALL,*o,w=weights(ALL))
    R['self_A']=grp(predict(MA,copy.deepcopy(ALL))); R['self_B']=grp(predict(MB,copy.deepcopy(ALL)))
    rng=np.random.default_rng(0); subs=[]; coefC=[]
    for i in range(200):
        S=subsample(ALL,rng); M=fit(S,*o); coefC.append(M); subs.append(predict(M,copy.deepcopy(ALL)))
    R['self_C']={g:{k:float(np.mean([metrics([x for x in P if g=='ALL' or x['q']==g or x['set']==g])[k] for P in subs])) for k in ['n','spec','goal','half','yrms','trms','ym','tm']} for g in ['ALL','Q1','Q2','Q3','Q4','AX','S1','S2','S3','S4']}
    def agg(fn):
        return {f:{'c':float(fn([m[f]['c'] for m in coefC])),'a':float(fn([m[f]['a'] for m in coefC])),'b':[float(x) for x in fn(np.array([m[f]['b'] for m in coefC]),axis=0)]} for f in ['s','d']}
    R['coef']={'A':MA,'B':MB,'C_mean':agg(np.mean),'C_sd':agg(np.std)}
    # T2 반복 층화 홀드아웃 : 그룹마다 30% 시험, 200회
    rng=np.random.default_rng(1); acc={'A':[], 'B':[], 'C':[]}
    for i in range(200):
        tr=[];te=[]
        for g in G:
            L=[r for r in ALL if r['q']==g]; idx=rng.permutation(len(L)); k=int(round(0.3*len(L)))
            te+=[L[j] for j in idx[:k]]; tr+=[L[j] for j in idx[k:]]
        acc['A'].append(grp(predict(fit(tr,*o),copy.deepcopy(te))))
        acc['B'].append(grp(predict(fit(tr,*o,w=weights(tr)),copy.deepcopy(te))))
        acc['C'].append(grp(predict(fit(subsample(tr,rng),*o),copy.deepcopy(te))))
    R['holdout']={v:{g:{k:float(np.mean([a[g][k] for a in acc[v]])) for k in ['n','spec','goal','half','yrms','trms','ym','tm']} for g in ['ALL','Q1','Q2','Q3','Q4','AX']} for v in acc}
    res[tag]=R
    for v in ['A','B','C']:
        h=R['holdout'][v]; print(tag,v,'holdout', ' '.join('%s %.1f/%.1f(%.1f,%.4f)'%(g,h[g]['spec'],h[g]['n'],h[g]['yrms'],h[g]['trms']) for g in ['ALL','Q1','Q2','Q3','Q4','AX']))
    for v in ['A','B']: s=R['self_'+v]; print(tag,v,'self',' '.join('%s %d/%d'%(g,s[g]['spec'],s[g]['n']) for g in ['ALL','Q1','Q2','Q3','Q4','AX','S1','S2','S3','S4']))
    s=R['self_C']; print(tag,'C self',' '.join('%s %.1f'%(g,s[g]['spec']) for g in ['ALL','Q1','Q2','Q3','Q4','AX','S1','S2','S3','S4']))
json.dump(dict(cnt=cnt,res=res),open('/home/claude/rt/work/E4.json','w'),default=float,indent=1)
