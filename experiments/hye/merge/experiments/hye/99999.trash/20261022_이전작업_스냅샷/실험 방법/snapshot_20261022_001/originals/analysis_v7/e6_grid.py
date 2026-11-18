import itertools, time
from judge7 import *
MEAS={'17px σ1':'v6','17px σ2':'tr_fs8_hp8_s2.npz','25px σ2':'tr_fs8_hp12_s2.npz','33px σ2':'tr_fs8_hp16_s2.npz'}
rows=[]; t0=time.time()
for mn,fn in MEAS.items():
    D,F=load(fn)
    for summ in ['affine','quad']:
        X,cols=summarize(D,F,summ)
        for setn in ['QR만','QR+나사']:
            sel=[c[0] in 'LCR' or setn=='QR+나사' for c in cols]; Xs=X[:,sel]
            for zero,label,cov in itertools.product(['center','learn','oracle'],['raw','corr'],['full','shrink','diag']):
                df=run(Xs,True,zero,label,cov)
                r=dict(측정=mn,요약=summ,세트=setn,값수=sum(sel),영점=zero,라벨=label,공분산=cov,**both(df)); rows.append(r)
    print(mn,f'{time.time()-t0:.0f}s',flush=True)
T=pd.DataFrame(rows); T.round(3).to_csv('e6_grid.csv',index=False,encoding='utf-8-sig')
c=['측정','요약','세트','영점','라벨','공분산','명령값_축정확도','명령값_축틀림_Y','명령값_축틀림_Θ','명령값_틀림최대_Yeq','명령값_통과09','실제_통과09','명령값_Y_평균오차','실제_Y_평균오차']
print(T.sort_values('명령값_축정확도',ascending=False)[c].head(30).round(3).to_string())
print(T.groupby(['영점','라벨'])['명령값_축정확도'].agg(['mean','max']).round(2))
print(T.groupby(['공분산'])['명령값_축정확도'].agg(['mean','max']).round(2))
print(T.groupby(['측정','요약','세트'])['명령값_축정확도'].agg(['mean','max']).round(2))
