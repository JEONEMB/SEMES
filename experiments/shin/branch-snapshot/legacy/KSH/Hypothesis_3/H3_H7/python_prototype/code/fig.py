import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
for f in fm.findSystemFonts():
    if 'NotoSansCJK' in f: fm.fontManager.addfont(f)
plt.rcParams['font.family']=['Noto Sans CJK JP']; plt.rcParams['axes.unicode_minus']=False
ink='#2b2b2a'; mute='#8a8a85'; grid='#e6e5e0'; blue='#2a78d6'; gray='#b9b8b1'; orange='#eb6834'
fig,ax=plt.subplots(1,3,figsize=(17,5.6),dpi=110)
for a in ax: a.spines[['top','right']].set_visible(False); a.spines[['left','bottom']].set_color(mute); a.tick_params(colors=ink); a.grid(axis='x',color=grid); a.set_axisbelow(True)
# (a) 같은 날 (1차 Θ 스윕 홀수 100장) ±0.01 통과
m1=[('배경 시차 (정답 확인용)',89,0.0058),('위상 Gabor (기준선)',42,0.0184),('H3 선형화 물리 모델',39,0.0175),('H3 + H6 치우침 보정',40,0.0187),('OpenCV ECC 호모그래피',28,0.0273),('심화 ECC',27,0.0271)]
a=ax[0]; y=np.arange(len(m1))[::-1]
a.barh(y,[v[1] for v in m1],color=[gray if i==0 else (blue if 'H3' in v[0] else '#9ec5f4') for i,v in enumerate(m1)],height=0.6)
for yi,v in zip(y,m1): a.text(v[1]+1,yi,f'{v[1]}/100  (RMS {v[2]:.4f}°)',va='center',fontsize=9,color=ink)
a.set_yticks(y); a.set_yticklabels([v[0] for v in m1],fontsize=9); a.set_xlim(0,115); a.set_xlabel('±0.01° 통과 수 (100장)')
a.set_title('(a) 같은 날: 1차 세트 Θ 스윕 검증',loc='left',fontsize=11,color=ink)
# (b) 다른 날 + H7 (2차 세트 평가 33장)
m2=[('배경 시차',13,28),('OpenCV ECC 호모그래피',10,15),('심화 ECC',5,12),('H3 선형화 물리 모델',3,7)]
a=ax[1]; y=np.arange(len(m2))[::-1]
a.barh(y+0.17,[v[1] for v in m2],height=0.32,color='#cfcec8',label='H7 보정 전')
a.barh(y-0.17,[v[2] for v in m2],height=0.32,color=blue,label='H7 세션 보정 후')
for yi,v in zip(y,m2): a.text(v[2]+0.5,yi-0.17,f'{v[2]}/33',va='center',fontsize=9,color=ink); a.text(v[1]+0.5,yi+0.17,f'{v[1]}',va='center',fontsize=9,color=mute)
a.set_yticks(y); a.set_yticklabels([v[0] for v in m2],fontsize=9); a.set_xlim(0,36); a.set_xlabel('±0.01° 통과 수 (33장)')
a.set_title('(b) 다른 날: 2차 세트 + H7 세션 보정',loc='left',fontsize=11,color=ink); a.legend(frameon=False,fontsize=9,loc='lower right')
# (c) 점 단위 오차 구성
a=ax[2]
parts=[('연속 촬영 간 잡음\n(시간 잡음)',0.013),('자세마다 바뀌는\n계통 오차',0.052),('점마다 고정된 오차\n(기준 템플릿)',0.044),('전체 잔차',0.068)]
y=np.arange(len(parts))[::-1]
a.barh(y,[p[1] for p in parts],color=[gray,orange,'#cfcec8',ink],height=0.55)
for yi,p in zip(y,parts): a.text(p[1]+0.001,yi,f'{p[1]:.3f} px',va='center',fontsize=9,color=ink)
a.set_yticks(y); a.set_yticklabels([p[0] for p in parts],fontsize=9); a.set_xlim(0,0.085); a.set_xlabel('점 하나의 가로 위치 오차 σ (px)')
a.set_title('(c) QR 대응점 오차 구성 — 자세 의존 오차가 지배',loc='left',fontsize=11,color=ink)
fig.tight_layout(); fig.savefig('h3_h7_summary.png',facecolor='white')
