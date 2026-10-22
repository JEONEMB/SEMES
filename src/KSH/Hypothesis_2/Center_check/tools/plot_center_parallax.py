# 시제품 그림 : results/center/center_parallax.png (4칸)
# 사용 : python plot_center_parallax.py   (tools 폴더에서 실행, matplotlib 필요)
import os
HERE=os.path.dirname(os.path.abspath(__file__))
exec(open(os.path.join(HERE,'analyze.py'),encoding='utf-8').read().split("for setname,cols")[0])
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
for f in fm.findSystemFonts():
    if 'NanumGothic' in f or 'NotoSansCJK' in f: fm.fontManager.addfont(f)
names_f=[f.name for f in fm.fontManager.ttflist]
plt.rcParams['font.family']=[n for n in ['NanumGothic','Noto Sans CJK KR','Noto Sans CJK JP'] if n in names_f][:1] or ['DejaVu Sans']
plt.rcParams['axes.unicode_minus']=False
C={'plate':'#2a78d6','station':'#eda100','wall_upper':'#1baf7a','wall_lower':'#eb6834'}
grp={'QR_L':'plate','QR_C':'plate','QR_R':'plate','label':'plate','nut_L':'station','nut_R':'station','bracket_scr_L':'station','bracket_scr_R':'station','wall_U1':'wall_upper','wall_U2':'wall_upper','wall_W1':'wall_lower','wall_W2':'wall_lower','wall_W3':'wall_lower','wall_W4':'wall_lower'}
ink='#2b2b2a'; mute='#8a8a85'; grid='#e6e5e0'
fig,ax=plt.subplots(2,2,figsize=(15,10.5),dpi=110)
for a in ax.ravel():
    a.spines[['top','right']].set_visible(False); a.spines[['left','bottom']].set_color(mute); a.tick_params(colors=ink); a.grid(color=grid,lw=0.8); a.set_axisbelow(True)
# (a) depth layering
P,Cv,_=fit(train,list(range(14)))
a=ax[0,0]
for g in C:
    ks=[i for i,k in enumerate(use) if grp[k]==g]
    a.scatter(P[0][ks],P[1][ks],s=70,color=C[g],edgecolor='white',linewidth=1.5,label=g,zorder=3)
for i,k in enumerate(use):
    if k in ('QR_C','wall_U1','wall_W1','nut_L','bracket_scr_L'): a.annotate(k,(P[0][i],P[1][i]),xytext=(6,6),textcoords='offset points',fontsize=9,color=ink)
a.set_xlabel('Y 1 mm당 이동 (px/mm)'); a.set_ylabel('Θ 1°당 이동 (px/°)')
a.set_title('(a) 랜드마크 깊이별 감도 — Y는 깊이에 따라 다르고 Θ는 거의 같음',loc='left',fontsize=11,color=ink)
a.legend(frameon=False,fontsize=9,loc='lower left')
# (b) centers
cols=[0,1,2,3,8,9,10,11,12,13]; P2,C2,_=fit(train,cols)
cs=[i for i in range(len(names)) if names[i].startswith('center')]
lab={'center.bmp':'center\n10/04','center1.bmp':'center1\n10/15 10:33','center2.bmp':'center2\n10:43','center3.bmp':'center3\n12:54'}
E=np.array([est(P2,C2,DX[[i]][:,cols])[0] for i in cs])
a=ax[0,1]; x=np.arange(len(cs))
a.plot(x,E[:,0]*1000,'-o',color='#2a78d6',lw=2,ms=8,label='Y')
a.set_xticks(x); a.set_xticklabels([lab[names[i]] for i in cs],fontsize=9)
a.set_ylabel('Y 추정 (µm)')
for xi,v in zip(x,E): a.annotate(f'Y {v[0]*1000:+.0f} µm\nΘ {v[1]:+.4f}°',(xi,v[0]*1000),xytext=(8,-4),textcoords='offset points',fontsize=9,color=ink)
a.set_title('(b) 기준(center) 4장의 위치 추정 — 시간에 따라 Y −, Θ + 로 이동',loc='left',fontsize=11,color=ink)
a.axhline(0,color=mute,lw=1)
# (c) set1 test errors: plate-only vs parallax
a=ax[1,0]
Pp,Cp,_=fit(train,[0,1,2,3]); mT=test[Y[test]==0]
eP=est(Pp,Cp,DX[mT][:,[0,1,2,3]])[:,1]-T[mT]; eX=est(P2,C2,DX[mT][:,cols])[:,1]-T[mT]
a.axhspan(-0.01,0.01,color='#d9efd9',zorder=0,label='±0.01° 허용')
a.scatter(T[mT],eP,s=22,color='#b9b8b1',label=f'QR 중심만 (RMS {np.sqrt((eP**2).mean()):.3f}°)',zorder=2)
a.scatter(T[mT],eX,s=26,color='#2a78d6',edgecolor='white',linewidth=0.6,label=f'플레이트+벽 시차 (RMS {np.sqrt((eX**2).mean()):.4f}°)',zorder=3)
a.set_ylim(-0.1,0.1); a.set_xlabel('정답 Θ (°)'); a.set_ylabel('Θ 오차 (°)')
a.set_title('(c) 1차 세트 검증(Θ 스윕 홀수 번째) — 시차를 쓰면 86/100 통과',loc='left',fontsize=11,color=ink)
a.legend(frameon=False,fontsize=9,loc='upper left')
# (d) set2 per-pose
a=ax[1,1]
c1=[i for i in cs if names[i]=='center1.bmp'][0]; m2=np.where((S==2)&~((Y==0)&(T==0)))[0]
e2=est(P2,C2,DX[m2][:,cols]-DX[c1][cols]+P2[3])
a.axhspan(-0.01,0.01,color='#d9efd9',zorder=0)
mk={0.05:'o',0.1:'s',0.5:'^',1.0:'D'}; cc={0.05:'#2a78d6',0.1:'#eb6834',0.5:'#1baf7a',1.0:'#eda100'}
for yv in mk:
    mm=np.abs(Y[m2]-yv)<1e-9
    a.scatter(T[m2][mm],e2[mm,1]-T[m2][mm],marker=mk[yv],s=34,color=cc[yv],edgecolor='white',linewidth=0.6,label=f'Y +{yv:g} mm',zorder=3)
tt=np.linspace(0,2,10); a.plot(tt,0.0205*tt,'--',color=mute,lw=1.2); a.annotate('Θ에 비례하는 오차 ≈ +2% (날짜 간 배율 차)',(0.12,0.036),fontsize=9,color=ink)
a.set_xscale('symlog',linthresh=0.05); a.set_xticks([0.01,0.03,0.1,0.2,1,2]); a.set_xticklabels(['0.01','0.03','0.1','0.2','1','2'])
a.set_xlabel('정답 Θ (°, 로그 눈금)'); a.set_ylabel('Θ 오차 (°)')
a.set_title('(d) 2차 세트(다른 날) — 1차 세트 모델 그대로, center1 기준',loc='left',fontsize=11,color=ink)
a.legend(frameon=False,fontsize=9,loc='upper left')
fig.tight_layout(); fig.savefig(os.path.join(HERE,'..','results','center','center_parallax.png'),facecolor='white')
