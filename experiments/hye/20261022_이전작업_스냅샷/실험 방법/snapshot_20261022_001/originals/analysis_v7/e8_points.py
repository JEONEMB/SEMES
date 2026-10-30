"""점 전부 vs 골라 쓰기, 상쇄되나?
(1) 점별 이동장 그대로 (점 x 2성분) → 가중 없음 / 점마다 잡음 가중(대각) / 수축 공분산(점끼리 상관 반영)
(2) 상위 k개 (학습 폴드에서 점 하나의 단서로 순위)
(3) QR 조합 (요약값에서 QR 고르기)
(4) 움직임마다 단서 큰 점이 다른가: 단서 순위를 Y+/Θ+ 와 Y-/Θ- 로, 작은 값(|Y환산|≤0.5)/큰 값으로 따로 계산해 순위 상관과 상위 50개 겹침
(5) 상쇄 시험: 단서 상위 50개에 단서 하위 점을 섞을 때 가중 없음 vs 가중"""
from judge7 import *
from scipy.stats import spearmanr
D,F=load('v6'); qr=F.region.str.startswith('QR').values
rows=[]
def add(name,df,**kw):
    r=dict(실험=name,**kw,**both(df)); rows.append(r)
    print(f"{name:34s} {kw}  축 {r['명령값_축정확도']:.2f}% (Y{r['명령값_축틀림_Y']}/Θ{r['명령값_축틀림_Θ']}, ≤{r['명령값_틀림최대_Yeq']:.2f})  통과 {r['명령값_통과09']:.1f}",flush=True)

def point_clue(Dp,names_mask,lab):
    """점마다 단서 (Y 1mm당 σ²): 점 하나(dx,dy)로 Y 무늬와 Θ 무늬가 얼마나 다른가 (Θ 크기 최적)."""
    S={}; V=[]
    for a in ['Y','Θ']:
        m=names_mask&(AXA==a); v=lab[m]; A=np.c_[np.ones(m.sum()),v,v*v]
        Xm=Dp[m].reshape(m.sum(),-1); co=np.linalg.lstsq(A,Xm,rcond=None)[0]; S[a]=co[1]; V.append(Xm-A@co)
    var=np.vstack(V).var(0); sY=S['Y'].reshape(-1,2); sT=S['Θ'].reshape(-1,2); w=1/np.maximum(var.reshape(-1,2),1e-6)
    a=(w*sY*sT).sum(1)/(w*sT*sT).sum(1); r=sY-a[:,None]*sT
    return (w*r*r).sum(1)

# ---- (1) 점별 ----
Dq=D[:,qr]; Xp=Dq.reshape(len(NAMES),-1)
for cov,nm in [('eye','점 전부, 가중 없음'),('diag','점 전부, 잡음 가중(대각)'),('shrink','점 전부, 수축 공분산')]:
    if cov=='eye':
        import judge7 as J; old=J.cov_est; J.cov_est=lambda R,k: np.eye(R.shape[1])*R.var()
        df=run(Xp,True,'learn','corr','x'); J.cov_est=old
    else: df=run(Xp,True,'learn','corr',cov)
    add(nm,df,점수=int(qr.sum()))
# ---- (2) 상위 k ----
def run_topk(kk,cov):
    recs=[]
    import judge7 as J
    for k in (0,1):
        te=FOLDA==k; tr=(FOLDA>=0)&~te; lab=np.array([CORR[k][n] for n in NAMES])
        c=point_clue(Dq,tr,lab); keep=np.argsort(-c)[:kk]
        Xk=Dq[:,keep].reshape(len(NAMES),-1)
        df=run(Xk,True,'learn','corr',cov); recs.append(df[df.file.isin([NAMES[i] for i in np.where(te)[0]])])
    return pd.concat(recs)
for kk in [5,10,25,50,100]:
    for cov in ['diag','shrink']: add(f'상위 {kk}점 ({cov})',run_topk(kk,cov),점수=kk)
# ---- (3) QR 조합 (요약값, 2차) ----
X,cols=summarize(D,F,'quad')
for comb in ['L','C','R','LC','LR','CR','LCR']:
    sel=np.array([c[0] in comb for c in cols]); add(f'QR 조합 {comb} (요약 2차)',run(X[:,sel],True,'learn','corr','full'),점수=int(sum(F.region.str[2].isin([{'L':'좌','C':'중','R':'우'}[z] for z in comb]))))
# ---- (4) 움직임마다 단서 큰 점이 다른가 ----
lab=np.array([CORR[0][n] for n in NAMES]); base=(FOLDA>=0)
yeq=np.abs(CMD)*np.where(AXA=='Y',1,THETA_TO_Y)
masks={'+방향':base&(CMD>0),'-방향':base&(CMD<0),'작은값(≤0.5mm)':base&(yeq<=0.5),'큰값(>0.5mm)':base&(yeq>0.5),'짝 폴드':FOLDA==0,'홀 폴드':FOLDA==1}
C={k:point_clue(Dq,m,lab) for k,m in masks.items()}
info=[]
for a,b in [('+방향','-방향'),('작은값(≤0.5mm)','큰값(>0.5mm)'),('짝 폴드','홀 폴드')]:
    rho=spearmanr(C[a],C[b]).correlation; ov=len(set(np.argsort(-C[a])[:50])&set(np.argsort(-C[b])[:50]))
    info.append(dict(비교=f'{a} vs {b}',순위상관=rho,상위50겹침=ov)); print(info[-1])
# 단서 집중도
c=np.sort(point_clue(Dq,base,lab))[::-1]; print('상위 10%/25%/50% 점이 가진 단서 비율', [round(c[:int(len(c)*f)].sum()/c.sum()*100,1) for f in (0.1,0.25,0.5)])
# ---- (5) 상쇄 시험 ----
cc=point_clue(Dq,base,lab); order=np.argsort(-cc)
for nweak in [0,50,100,182]:
    keep=np.r_[order[:50],order[::-1][:nweak]] if nweak else order[:50]
    Xk=Dq[:,keep].reshape(len(NAMES),-1)
    import judge7 as J; old=J.cov_est; J.cov_est=lambda R,k: np.eye(R.shape[1])*R.var(); de=run(Xk,True,'learn','corr','x'); J.cov_est=old
    add(f'상쇄: 상위50 + 하위{nweak} 가중 없음',de,점수=len(keep)); add(f'상쇄: 상위50 + 하위{nweak} 잡음 가중',run(Xk,True,'learn','corr','diag'),점수=len(keep))
pd.DataFrame(rows).round(3).to_csv('e8_points.csv',index=False,encoding='utf-8-sig'); pd.DataFrame(info).round(3).to_csv('e8_point_rank.csv',index=False,encoding='utf-8-sig')
