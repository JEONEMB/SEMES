"""최종 파이프라인 채점 (평가 기준) + 0점 7개 표 + 크기별 축 틀림률 (실측 40번 반복 vs 이론) + 100%에 필요한 잡음.
최종: 33px 템플릿, 흐림 σ2, 가우스-뉴턴 추적 → QR마다 아핀 요약 (QR만 18값 / +나사 22값) → 축마다 2차 매핑 → center를 0점 → 전체 공분산 GLS 가설 비교."""
import judge7 as J
from judge7 import *
from scipy.stats import norm
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams['font.family']='Malgun Gothic'; plt.rcParams['axes.unicode_minus']=False
BL,OR,AQ,YE,GR,TX,SUB='#2a78d6','#eb6834','#1baf7a','#eda100','#e4e3dc','#0b0b0b','#52514e'
D,F=load('tr_fs8_hp16_s2.npz'); X,cols=summarize(D,F,'affine')
SETS={'QR만':np.array([c[0] in 'LCR' for c in cols]),'QR+나사':np.ones(len(cols),bool)}
# ---- 1. 평가 기준 표 (짝/홀) ----
rows=[]; per=[]
for setn,sel in SETS.items():
    for lab in ['raw','corr']:
        for zero in ['center','oracle']:
            df=run(X[:,sel],True,zero,lab,'full')
            for basis,bn in [('cmd','명령값 (보정 안 함)'),('act','실제 위치 (보정함)')]:
                e=evaluate(df,basis); rows.append(dict(세트=setn,학습라벨='명령값' if lab=='raw' else '보정본',영점='center (배포 가능)' if zero=='center' else '구간 0점 앎 (상한)',정답기준=bn,**e))
            if zero=='center': per.append(df.assign(세트=setn,학습라벨=lab))
T=pd.DataFrame(rows); T.round(3).to_csv('../58_v7_평가표_최종.csv',index=False,encoding='utf-8-sig')
pd.concat(per).round(5).to_csv('../59_v7_사진별예측_최종.csv',index=False,encoding='utf-8-sig')
print(T[['세트','학습라벨','영점','정답기준','축정확도','축틀림_Y','축틀림_Θ','틀림최대_Yeq','Y_평균오차','Y_최대오차','Θ_평균오차','Θ_최대오차','통과09']].round(2).to_string())
# ---- 2. 0점 7개 ----
names=NAMES; Xq=X[:,SETS['QR만']]; R,ints,sl=session_fit(Xq,names)
m_idx=[cols.index(c) for c in ['L_dx_0','C_dx_0','R_dx_0']]
n_floor,ymin,zdir=floor(Xq,names)
z=[dict(구간='center',평균이동_px=0.0,Y환산_mm=0.0,Θ환산=0.0,판정방향_치우침_Yeq_mm=0.0)]
for s in SES:
    mpx=np.mean(ints[s][m_idx]); z.append(dict(구간=s,평균이동_px=mpx,Y환산_mm=mpx/14.39,Θ환산=mpx/80.66,판정방향_치우침_Yeq_mm=zdir[s]))
Z=pd.DataFrame(z); Z.round(4).to_csv('../60_v7_0점7개.csv',index=False,encoding='utf-8-sig'); print(Z.round(4).to_string()); print('잡음 바닥',n_floor,ymin)
# ---- 3. 크기별 틀림률: 40번 반복 ----
rng=np.random.default_rng(0); splits=[]
for r in range(40):
    f=np.full(len(NAMES),-1)
    for s in SES:
        idx=np.where(SESA==s)[0]; p=rng.permutation(idx); f[p[:len(p)//2]]=0; f[p[len(p)//2:]]=1
    splits.append(f)
base=FOLDA.copy(); yeq=np.abs(CMD)*np.where(AXA=='Y',1,THETA_TO_Y)
ERR={}
for setn,sel in SETS.items():
    for zero in ['center','oracle']:
        cnt=np.zeros(len(NAMES)); 
        for f in splits:
            J.FOLDA=f; df=run(X[:,sel],True,zero,'raw','full').set_index('file'); bad=(df.pred_axis!=df.axis)
            cnt[[NAMES.index(n) for n in bad[bad].index]]+=1
        J.FOLDA=base; ERR[(setn,zero)]=cnt/40
bins=[0,0.06,0.12,0.18,0.24,0.3,0.4,0.6,1,2.1]; ok=AXA!='C'; bi=np.digitize(yeq,bins)-1
bt=[]
for k,v in ERR.items():
    for b in range(len(bins)-1):
        m=ok&(bi==b); bt.append(dict(세트=k[0],영점=k[1],구간_Yeq=f'{bins[b]}~{bins[b+1]}',사진수=int(m.sum()),틀림률=v[m].mean()*100))
pd.DataFrame(bt).round(2).to_csv('../61_v7_크기별_축틀림률.csv',index=False,encoding='utf-8-sig')
# 이론
nq=floor(X[:,SETS['QR만']],names)[0]; ns=floor(X,names)[0]
def expected(n): return norm.cdf(-n*yeq[ok]/2).sum()
need=next(n for n in np.arange(5,200,0.5) if expected(n)<0.5)
print(f'이론 σ/mm QR만 {nq:.1f}, +나사 {ns:.1f}; 기대 틀림 {expected(nq):.1f}, {expected(ns):.1f}; 전체 기대 틀림<0.5장 필요 {need:.0f}σ/mm → 같은 자세 사진 {(need/nq)**2:.0f}장 평균 (QR만)')
print({k:round(v[ok].sum()/1,2) for k,v in ERR.items()})
# ---- 그림 57: 크기별 틀림률 ----
fig,axs=plt.subplots(1,2,figsize=(13,4.6),sharey=True)
xc=np.array([(bins[b]+bins[b+1])/2 for b in range(len(bins)-1)]); xx=np.linspace(0.01,1,300)
for ax,setn,n in [(axs[0],'QR만',nq),(axs[1],'QR+나사',ns)]:
    for zero,c,lbl in [('center',BL,'실측: center 0점 (배포 가능)'),('oracle',OR,'실측: 구간 0점 앎 (상한)')]:
        v=ERR[(setn,zero)]; r=[v[ok&(bi==b)].mean()*100 for b in range(len(bins)-1)]
        ax.plot(xc,r,'-o',color=c,lw=2,ms=8,label=lbl,mec='#fcfcfb',mew=2)
    ax.plot(xx,norm.cdf(-n*xx/2)*100,'--',color=SUB,lw=1.5,label=f'이론 (잡음 {n:.1f}σ/mm)')
    ax.axvline(0.09,color=AQ,lw=1); ax.text(0.095,45,'0.09',color=SUB)
    ax.set_xscale('log'); ax.set_xticks([0.05,0.1,0.2,0.3,0.5,1]); ax.set_xticklabels(['0.05','0.1','0.2','0.3','0.5','1'])
    ax.set_xlabel('움직임 크기 (Y mm 환산, Θ는 x5.6)'); ax.set_title(setn,loc='left',color=TX)
    for s_ in ['top','right']: ax.spines[s_].set_visible(False)
    ax.grid(axis='y',color=GR,lw=0.8)
axs[0].set_ylabel('축을 틀린 비율 (%)'); axs[0].legend(frameon=False,fontsize=9)
fig.suptitle('크기별 축 틀림률: 무작위 반반 나누기 40번, 33px σ2 아핀 요약 + GLS',x=0.01,ha='left',color=TX); fig.tight_layout(); fig.savefig('../57_v7_크기별_축틀림률.png',dpi=130)
# ---- 그림 54: 0점 7개 ----
fig,axs=plt.subplots(1,2,figsize=(12,4))
lbl=Z.구간.tolist(); col=[SUB]+[OR]*4+[BL]*2
axs[0].bar(lbl,Z.Y환산_mm*1000,color=col,width=0.6); axs[0].set_title('값 쪽 0점: 구간 절편의 평균 이동 (Y µm 환산)',loc='left',fontsize=11)
axs[0].axhline(0,color=SUB,lw=0.8); axs[0].set_ylabel('µm (Y 환산)')
axs[1].bar(lbl,Z.판정방향_치우침_Yeq_mm*1000,color=col,width=0.6); axs[1].set_title('모양 쪽 0점: 판정 방향 치우침 (Y µm 환산)',loc='left',fontsize=11)
axs[1].axhline(0,color=SUB,lw=0.8)
for a in axs:
    for s_ in ['top','right']: a.spines[s_].set_visible(False)
    a.grid(axis='y',color=GR,lw=0.8)
fig.tight_layout(); fig.savefig('../54_v7_0점7개.png',dpi=130)
