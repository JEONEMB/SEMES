import pandas as pd, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams['font.family']='Malgun Gothic'; plt.rcParams['axes.unicode_minus']=False
BL,OR,GR,SUB,TX='#2a78d6','#eb6834','#e4e3dc','#52514e','#0b0b0b'
def style(a):
    for s in ['top','right']: a.spines[s].set_visible(False)
    a.grid(axis='x',color=GR,lw=0.8); a.set_axisbelow(True)
# 55: 점 실험
P=pd.read_csv('e8_points.csv')
pick=['점 전부, 가중 없음','점 전부, 잡음 가중(대각)','점 전부, 수축 공분산','상위 10점 (shrink)','상위 25점 (shrink)','상위 50점 (shrink)','상위 100점 (shrink)',
      'QR 조합 L (요약 2차)','QR 조합 C (요약 2차)','QR 조합 R (요약 2차)','QR 조합 LR (요약 2차)','QR 조합 LCR (요약 2차)']
lab={'점 전부, 가중 없음':'점 232개 전부, 가중 없음','점 전부, 잡음 가중(대각)':'점 232개 전부, 점마다 독립 잡음 가중','점 전부, 수축 공분산':'점 232개 전부, 같이 흔들림 반영(공분산)',
     'QR 조합 L (요약 2차)':'QR 좌만','QR 조합 C (요약 2차)':'QR 중만','QR 조합 R (요약 2차)':'QR 우만','QR 조합 LR (요약 2차)':'QR 좌+우','QR 조합 LCR (요약 2차)':'QR 3개 (요약값 + 공분산)'}
Q=P.set_index('실험').loc[pick]
fig,ax=plt.subplots(figsize=(10,5.2)); y=np.arange(len(pick))[::-1]
c=[OR if '독립' in lab.get(p,p) or '가중 없음' in lab.get(p,p) else BL for p in pick]
ax.barh(y,Q['명령값_축정확도'],color=c,height=0.6)
for yy,v,p in zip(y,Q['명령값_축정확도'],pick): ax.text(v+0.15,yy,f"{v:.1f}%  (틀림 {int(Q.loc[p,'명령값_축틀림_Y']+Q.loc[p,'명령값_축틀림_Θ'])}장)",va='center',fontsize=9,color=TX)
ax.set_yticks(y); ax.set_yticklabels([lab.get(p,p).replace('(shrink)','(공분산)') for p in pick]); ax.set_xlim(85,101); ax.set_xlabel('축 정확도 (%), 짝/홀 교차검증, 17px σ1')
style(ax); ax.set_title('점을 다 넣어도 상쇄되지 않음: 같이 흔들리는 성분을 공분산으로 반영하는지가 차이를 만듦',loc='left',fontsize=11,color=TX)
fig.tight_layout(); fig.savefig('../55_v7_점전부vs골라쓰기.png',dpi=130)
# 56: 계산 방법
M=pd.read_csv('e7_methods.csv'); M=M[(M.측정=='33px σ2')&(M.세트=='QR만')&(M.라벨=='corr')]
order=['GLS 가설비교 / 0점 center','GLS 가설비교 / 0점 learn','GLS 가설비교 / 0점 learn+Σz','GLS 가설비교 / 0점 oracle','GLS 가설비교 1차 매핑 / learn','동시추정 (Y, Θ 함께)','ML 로지스틱','ML SVM(RBF)','ML GBM','ML kNN','곡률 규칙 (값 2개)']
M=M.set_index('방법').loc[order]
fig,ax=plt.subplots(figsize=(10,5)); y=np.arange(len(order))[::-1]
ax.barh(y,M['명령값_축정확도'],color=[BL if 'GLS' in o or '동시' in o else OR for o in order],height=0.6)
for yy,o in zip(y,order): ax.text(M.loc[o,'명령값_축정확도']+0.1,yy,f"{M.loc[o,'명령값_축정확도']:.1f}%  (Y {int(M.loc[o,'명령값_축틀림_Y'])} / Θ {int(M.loc[o,'명령값_축틀림_Θ'])})",va='center',fontsize=9)
ax.set_yticks(y); ax.set_yticklabels([o.replace('oracle','구간 0점 앎(상한)').replace('learn+Σz','학습+0점흔들림 잡음').replace('learn','축마다 학습').replace('center','= center') for o in order])
ax.set_xlim(93,100.5); ax.set_xlabel('축 정확도 (%), 짝/홀 교차검증, 33px σ2, QR만'); style(ax)
ax.set_title('계산 방법 비교: 가설 비교 GLS가 가장 좋고, 머신러닝은 이기지 못함',loc='left',fontsize=11,color=TX)
fig.tight_layout(); fig.savefig('../56_v7_계산방법비교.png',dpi=130)
