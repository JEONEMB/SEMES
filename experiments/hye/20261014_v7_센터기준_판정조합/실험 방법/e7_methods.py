"""계산 방법 비교 + 0점 처리 비교 + 점 전부 vs 골라 쓰기. 측정은 미리 정한 대표 2개 (17px σ1 = v6 그대로, 33px σ2 = 잡음 바닥 최소)."""
from judge7 import *
rows=[]; PRED={}
def add(name,df,**kw):
    r=dict(방법=name,**kw,**both(df)); rows.append(r); PRED[(name,)+tuple(kw.values())]=df
    print(f"{name:28s} {kw}  축 {r['명령값_축정확도']:.2f}% (Y{r['명령값_축틀림_Y']}/Θ{r['명령값_축틀림_Θ']}, ≤{r['명령값_틀림최대_Yeq']:.2f})  통과 {r['명령값_통과09']:.1f}/{r['실제_통과09']:.1f}",flush=True)
for mn,fn in [('17px σ1','v6'),('33px σ2','tr_fs8_hp16_s2.npz')]:
    D,F=load(fn); X,cols=summarize(D,F,'quad')
    for setn in ['QR만','QR+나사']:
        sel=np.array([c[0] in 'LCR' or setn=='QR+나사' for c in cols]); Xs=X[:,sel]; cs=[c for c,s in zip(cols,sel) if s]
        kw=dict(측정=mn,세트=setn)
        for lab in ['raw','corr']:
            for z in ['center','learn','oracle']: add(f'GLS 가설비교 / 0점 {z}',run(Xs,True,z,lab,'full'),라벨=lab,**kw)
            add('GLS 가설비교 / 0점 learn+Σz',run_zcov(Xs,True,lab),라벨=lab,**kw)
        add('GLS 가설비교 1차 매핑 / learn',run(Xs,False,'learn','corr','full'),라벨='corr',**kw)
        add('동시추정 (Y, Θ 함께)',run_joint(Xs,'corr'),라벨='corr',**kw)
        for ml in ['로지스틱','SVM(RBF)','GBM','kNN']: add(f'ML {ml}',run_ml(Xs,ml,'corr'),라벨='corr',**kw)
        add('곡률 규칙 (값 2개)',run_curv(Xs,cs,'corr'),라벨='corr',**kw)
T=pd.DataFrame(rows); T.round(3).to_csv('e7_methods.csv',index=False,encoding='utf-8-sig')
import pickle; pickle.dump(PRED,open('e7_pred.pkl','wb'))
