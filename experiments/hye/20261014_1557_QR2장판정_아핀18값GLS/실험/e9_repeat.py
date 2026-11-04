"""후보 파이프라인을 무작위 반반 나누기(구간별 층화) 40번으로 채점 → 평균, 분포. 라벨은 명령값 (보정본은 짝/홀 폴드에만 있음).
축 정확도, 틀린 수, 0.09 통과 (명령값 기준)."""
import judge7 as J
from judge7 import *
def summarize_cubic(D,F):
    cols=[]; X=[]
    for r,k in [('QR좌','L'),('QR중','C'),('QR우','R')]:
        m=F.region.values==r; x=(F.x.values[m]-F.x.values[m].mean())/100; y=(F.y.values[m]-F.y.values[m].mean())/100
        A=np.c_[np.ones(m.sum()),x,y,x*x,x*y,y*y,x**3,x*x*y,x*y*y,y**3]
        for c in (0,1): co=np.linalg.lstsq(A,D[:,m,c].T,rcond=None)[0]; X+=list(co); cols+=[f'{k}_{c}_{t}' for t in range(10)]
    for i,(sx,sy) in enumerate(SCREWS):
        m=(F.region.values=='판나사')&(np.hypot(F.x.values-sx,F.y.values-sy)<60); X+=[D[:,m,0].mean(1),D[:,m,1].mean(1)]; cols+=[f'S{i}_dx',f'S{i}_dy']
    return np.array(X).T,cols
TR={'17px σ1':load('v6'),'25px σ2':load('tr_fs8_hp12_s2.npz'),'33px σ2':load('tr_fs8_hp16_s2.npz'),'17px σ2':load('tr_fs8_hp8_s2.npz')}
FEAT={}
for mn,(D,F) in TR.items():
    for sm in ['affine','quad']: FEAT[(mn,sm)]=summarize(D,F,sm)
    FEAT[(mn,'cubic')]=summarize_cubic(D,F)
rng=np.random.default_rng(0); NREP=40; base=FOLDA.copy()
splits=[]
for r in range(NREP):
    f=np.full(len(NAMES),-1)
    for s in SES:
        idx=np.where(SESA==s)[0]; p=rng.permutation(idx); f[p[:len(p)//2]]=0; f[p[len(p)//2:]]=1
    splits.append(f)
ZEROS=['center','shared','oracle']
cands=[]
for mn in TR:
    for sm in ['affine','quad']:
        for setn in ['QR만','QR+나사']:
            for zero in ZEROS:
                for cov in (['full','shrink'] if sm!='cubic' else ['shrink']): cands.append((mn,sm,setn,zero,cov))
rows=[]
for c in cands:
    mn,sm,setn,zero,cov=c; X,cols=FEAT[(mn,sm)]; sel=np.array([cc[0] in 'LCR' or setn=='QR+나사' for cc in cols]); Xs=X[:,sel]
    acc=[]; nb=[]; pas=[]; bmax=[]
    for f in splits:
        J.FOLDA=f; df=run(Xs,True,zero,'raw',cov); e=evaluate(df,'cmd')
        acc.append(e['축정확도']); nb.append(e['축틀림_Y']+e['축틀림_Θ']); pas.append(e['통과09']); bmax.append(e['틀림최대_Yeq'])
    J.FOLDA=base
    rows.append(dict(측정=mn,요약=sm,세트=setn,영점=zero,공분산=cov,값수=Xs.shape[1],축정확도_평균=np.mean(acc),축정확도_최저=np.min(acc),축정확도_최고=np.max(acc),
                     틀림수_평균=np.mean(nb),전부맞음_횟수=int(np.sum(np.array(nb)==0)),통과09_평균=np.mean(pas),틀림최대Yeq_평균=np.mean(bmax)))
    print(rows[-1],flush=True)
T=pd.DataFrame(rows); T.round(3).to_csv('e9b_repeat.csv',index=False,encoding='utf-8-sig')
print(T.sort_values('축정확도_평균',ascending=False).head(25).round(2).to_string())
