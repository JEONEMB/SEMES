import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from a4 import mt
from rt import *
plt.rcParams['font.family']='Noto Sans CJK KR'; plt.rcParams['axes.unicode_minus']=False
old=load_yml_model('/home/claude/cmp/ksh/config/model_s1d1.yml')
P=predict(old,samples(('S4',),ref=0,drop_bad=False)); t0=min(mt.values())
for r in P: r['t']=(mt[r['file']]-t0)/60
P.sort(key=lambda r:r['t'])
fig,ax=plt.subplots(3,1,figsize=(14,11),dpi=140,sharex=True,gridspec_kw={'height_ratios':[1.2,1.2,0.8]})
G=[r for r in P if r['file']!='Y-0.8 T-0.96']; bad=[r for r in P if r['file']=='Y-0.8 T-0.96'][0]
col=lambda r:'#2F6FED' if r['T']>0 else '#E8833A'
ax[0].scatter([r['t'] for r in G],[r['ey'] for r in G],c=[col(r) for r in G],s=22)
ax[0].axhspan(-90,90,color='#DCFCE7',alpha=.4,zorder=0); ax[0].axhspan(-45,45,color='#BBF7D0',alpha=.4,zorder=0)
ax[0].axhline(np.mean([r['ey'] for r in G]),color='#C2410C',ls='--',lw=1); ax[0].text(60,-27,'평균 치우침 −31.5 µm (같은 날 기준 사진 없음)',color='#C2410C',fontsize=10)
ax[0].set_ylabel('Y 오차 (µm)'); ax[0].set_title('4차 144장 — 촬영 순서대로 본 오차 (파랑 Θ+, 주황 Θ−, 초록 띠 = ±45/±90 µm)',fontsize=13)
for x0,x1,l in [(0,47,'Y −0.05 → −0.8 → +0.8 → +0.05 → +0.8 (Θ ±0.03~0.48, 5장씩)'),(53,60,'Y ±1.6 일부'),(93,100,'Y −1.6 나머지'),(103,118,'Θ ±0.96 줄 → Y +1.6')]:
    ax[0].annotate(l,xy=((x0+x1)/2,100),ha='center',fontsize=8.5,color='#334')
ax[0].set_ylim(-110,115)
ax[1].scatter([r['t'] for r in G],[r['et'] for r in G],c=[col(r) for r in G],s=22)
ax[1].axhspan(-0.01,0.01,color='#DCFCE7',alpha=.4,zorder=0); ax[1].axhspan(-0.005,0.005,color='#BBF7D0',alpha=.4,zorder=0)
ax[1].set_ylabel('Θ 오차 (°)'); ax[1].set_ylim(-0.014,0.016)
ax[1].annotate('이상 사진 Y−0.8 T−0.96 : 오차 +1.68° (축 밖)\n(+0.96 줄을 찍는 도중 101분에 촬영됨)',xy=(bad['t'],0.0155),xytext=(70,0.012),fontsize=9,arrowprops=dict(arrowstyle='->'))
ax[2].plot([r['t'] for r in P],[r['Y'] for r in P],'-',color='#888',lw=0.8); ax[2].scatter([r['t'] for r in P],[r['Y'] for r in P],c=[col(r) for r in P],s=12)
ax[2].set_ylabel('지령 Y (mm)'); ax[2].set_xlabel('촬영 경과 (분)')
for a in ax: a.grid(alpha=.25)
fig.tight_layout(); fig.savefig('/home/claude/rt/diag/4차_촬영순서_오차.png',bbox_inches='tight',facecolor='white')
