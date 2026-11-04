import judge7 as J
from judge7 import *
TR={'17px σ1':load('v6'),'25px σ2':load('tr_fs8_hp12_s2.npz'),'33px σ2':load('tr_fs8_hp16_s2.npz'),'17px σ3':load('tr_fs8_hp8_s3.npz')}
def feat(mn,sm,setn):
    X,cols=summarize(*TR[mn],sm); sel=np.array([c[0] in 'LCR' or setn=='QR+나사' for c in cols]); return X[:,sel]
rng=np.random.default_rng(0); splits=[]
for r in range(40):
    f=np.full(len(NAMES),-1)
    for s in SES:
        idx=np.where(SESA==s)[0]; p=rng.permutation(idx); f[p[:len(p)//2]]=0; f[p[len(p)//2:]]=1
    splits.append(f)
base=FOLDA.copy()
for setn in ['QR만','QR+나사']:
    for combo in [['33px σ2'],['17px σ1','33px σ2'],['17px σ1','25px σ2','33px σ2'],['17px σ1','17px σ3','33px σ2']]:
        for sm in ['affine']:
            X=np.hstack([feat(m,sm,setn) for m in combo])
            for cov in ['full','shrink']:
                acc=[]
                for f in splits: J.FOLDA=f; acc.append(evaluate(run(X,True,'center','raw',cov),'cmd')['축정확도'])
                J.FOLDA=base; print(setn,'+'.join(combo),sm,cov,X.shape[1],f'{np.mean(acc):.2f} ({np.min(acc):.2f}~{np.max(acc):.2f})',flush=True)
