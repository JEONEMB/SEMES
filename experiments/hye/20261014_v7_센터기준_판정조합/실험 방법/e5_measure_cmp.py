from lib7 import *
rows=[]
for fn,lab in [('v6','17px σ1 (v6)'),('tr_fs8_hp8_s0.npz','17px σ0'),('tr_fs8_hp8_s2.npz','17px σ2'),('tr_fs8_hp8_s3.npz','17px σ3'),
               ('tr_fs8_hp12_s1.npz','25px σ1'),('tr_fs8_hp12_s2.npz','25px σ2'),('tr_fs8_hp16_s2.npz','33px σ2'),('v6_hp4','9px σ1 (643점)')]:
    D,F=load(fn); names=NAMES
    qr=F.region.str.startswith('QR').values
    # 점마다 구간안 흔들림
    Xp=D[:,qr].reshape(len(names),-1); R,_,_=session_fit(Xp,names); R=R.reshape(len(R),-1,2); R=R-np.median(R,1,keepdims=True)
    sdx,sdy=np.median(R[:,:,0].std(0)),np.median(R[:,:,1].std(0))
    for model in ['affine','quad']:
        X,cols=summarize(D,F,model)
        for setn,sel in [('QR만',[c[0] in 'LCR' for c in cols]),('QR+나사',[True]*len(cols))]:
            n,ymin,z=floor(X[:,sel],names)
            rows.append(dict(측정=lab,요약=model,세트=setn,값수=sum(sel),점흔들림x=sdx,점흔들림y=sdy,sigma_per_mm=n,최소Y_3sigma=ymin,**{f'0점_{k}':v for k,v in z.items()}))
            print(rows[-1]['측정'],model,setn,f'{n:.1f}σ/mm  min {ymin:.3f}  pt {sdx:.4f}/{sdy:.4f}',flush=True)
pd.DataFrame(rows).round(4).to_csv('e5_measure_cmp.csv',index=False,encoding='utf-8-sig')
