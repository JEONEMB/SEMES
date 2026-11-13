from rt import *
import json, copy, time
ALL=samples(); G=['Q1','Q2','Q3','Q4','AX']; N=len(ALL)
def vec(M): return np.r_[M['s']['c'],M['s']['a'],M['s']['b'],M['d']['c'],M['d']['a'],M['d']['b']]
def avg(Ms):
    out={'os':Ms[0]['os'],'od':Ms[0]['od']}
    for f in ['s','d']: out[f]={k:float(np.mean([m[f][k] for m in Ms])) for k in ['c','a','sigma']}; out[f]['b']=list(np.mean([m[f]['b'] for m in Ms],0))
    return out
res={}
for o in [(1,1),(3,2)]:
    tag=f's{o[0]}d{o[1]}'; R={}
    # 5-1 결정론 : 같은 데이터로 1·3·10번
    Ms=[fit(ALL,*o) for _ in range(10)]; V=np.array([vec(m) for m in Ms])
    R['det']={'max_abs_diff_3':float(np.abs(V[:3]-V[0]).max()),'max_abs_diff_10':float(np.abs(V-V[0]).max()),'coef':list(V[0])}
    # 5-2 부트스트랩 반복(배깅) : R회 재표본 학습 → 계수 평균. 시드 30개로 안정성
    full=vec(fit(ALL,*o)); R['boot']={}
    for B in [1,3,10,50,200]:
        vs=[];specs=[];halfs=[];yr=[];tr=[]
        for seed in range(30):
            rng=np.random.default_rng(1000+seed); Ms=[]
            for b in range(B):
                idx=rng.integers(0,N,N); Ms.append(fit([ALL[i] for i in idx],*o))
            Mb=avg(Ms); vs.append(vec(Mb)); m=metrics(predict(Mb,copy.deepcopy(ALL))); specs.append(m['spec']); halfs.append(m['half']); yr.append(m['yrms']); tr.append(m['trms'])
        vs=np.array(vs)
        R['boot'][B]={'mean':list(vs.mean(0)),'sd':list(vs.std(0)),'dev_from_full':list(np.abs(vs.mean(0)-full)),'spec_mean':float(np.mean(specs)),'spec_sd':float(np.std(specs)),'half_mean':float(np.mean(halfs)),'half_sd':float(np.std(halfs)),'yrms':float(np.mean(yr)),'trms':float(np.mean(tr))}
        print(tag,'boot',B,'coef sd',np.round(vs.std(0)[:4],4),'spec %.1f±%.2f half %.1f±%.2f'%(np.mean(specs),np.std(specs),np.mean(halfs),np.std(halfs)))
    R['full']=list(full)
    # 5-3 반복 층화 k-겹 교차검증 : 정확도 추정의 재현성
    def strat_folds(rng,k):
        f=np.empty(N,int)
        for g in G:
            idx=[i for i,r in enumerate(ALL) if r['q']==g]; p=rng.permutation(idx)
            for j,i in enumerate(p): f[i]=j%k
        return f
    R['cv']={}
    for name,k,rep in [('10겹×1',10,1),('10겹×3',10,3),('10겹×10 (Bouckaert & Frank 2004)',10,10),('5×2겹 (Dietterich 1998)',2,5)]:
        est=[]; t0=time.time()
        for seed in range(30):
            rng=np.random.default_rng(5000+seed); sp=[];hf=[];yy=[];tt=[]
            for r_ in range(rep):
                f=strat_folds(rng,k); P=[]
                for j in range(k):
                    tr_=[ALL[i] for i in range(N) if f[i]!=j]; te=[ALL[i] for i in range(N) if f[i]==j]; P+=predict(fit(tr_,*o),copy.deepcopy(te))
                m=metrics(P); sp.append(m['spec']/m['n']); hf.append(m['half']/m['n']); yy.append(m['yrms']); tt.append(m['trms'])
            est.append((np.mean(sp),np.mean(hf),np.mean(yy),np.mean(tt)))
        est=np.array(est)
        R['cv'][name]={'fits':k*rep,'spec_mean':float(est[:,0].mean()),'spec_sd':float(est[:,0].std()),'half_mean':float(est[:,1].mean()),'half_sd':float(est[:,1].std()),'yrms_mean':float(est[:,2].mean()),'yrms_sd':float(est[:,2].std()),'trms_mean':float(est[:,3].mean()),'trms_sd':float(est[:,3].std())}
        print(tag,name,'spec %.4f±%.4f half %.4f±%.4f yrms %.2f±%.3f trms %.5f±%.6f  %.0fs'%(est[:,0].mean(),est[:,0].std(),est[:,1].mean(),est[:,1].std(),est[:,2].mean(),est[:,2].std(),est[:,3].mean(),est[:,3].std(),time.time()-t0))
    res[tag]=R
json.dump(res,open('/home/claude/rt/work/E5.json','w'),default=float,indent=1)
