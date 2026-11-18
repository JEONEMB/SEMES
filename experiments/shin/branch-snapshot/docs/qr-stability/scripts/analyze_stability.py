import sys,json,hashlib
from pathlib import Path
import numpy as np,pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
if Path('C:/Windows/Fonts/malgun.ttf').exists():
    font_manager.fontManager.addfont('C:/Windows/Fonts/malgun.ttf')
plt.rcParams.update({'font.family':'Malgun Gothic','axes.unicode_minus':False,'font.size':10,'figure.dpi':140,'axes.spines.top':False,'axes.spines.right':False})
p=Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).parent
# Input: frames.csv, qr_status.csv, points.csv in the supplied output directory.
f=pd.read_csv(p/'frames.csv');q=pd.read_csv(p/'qr_status.csv');pt=pd.read_csv(p/'points.csv')
f['name']=f.path.map(lambda x:Path(x).stem);f['axis']=f.name.str[0].where(f.name.str[0].isin(['T','Y']),'center');f['label']=[float(n[1:]) if a!='center' else 0 for n,a in zip(f.name,f.axis)]
q=q.merge(f[['frame','axis','label','plate']],on='frame');q['slot_name']=q.slot.map({0:'L',1:'C',2:'R'})
# No null frames, duplicates or missing output rows may be silently discarded.
assert len(f)==281 and len(q)==843 and not f.frame.duplicated().any()
assert not pt.duplicated(['frame','payload','id']).any()
q['failure']=np.select([q.plate==0,q.roi_valid==0,q.decoded==0,q.model_matched==0,q.features_valid==0],['plate','roi','decode','model','quality'],default='ok')
summary={'frames':len(f),'all_three_success':int((f.valid_features==3).sum()),'all_three_rate':float((f.valid_features==3).mean()),'plate_success':int(f.plate.sum()),
 'failure_slots':q.failure.value_counts().to_dict(),'per_slot':{},'per_axis':{}}
for slot,g in q.groupby('slot_name'):
 good=g[g.features_valid==1];summary['per_slot'][slot]={'valid':int(g.features_valid.sum()),'total':len(g),'median_accepted':float(good.accepted.median()),'p05_accepted':float(good.accepted.quantile(.05)),'min_accepted':int(good.accepted.min())}
for axis,g in f.groupby('axis'):summary['per_axis'][axis]={'n':len(g),'all_three_success':int((g.valid_features==3).sum())}
# Fig 1: unconditional detection rates + spatial failure map, excluding center only from scan panels.
fig,axs=plt.subplots(2,2,figsize=(13,8),layout='constrained')
stages=[int(f.plate.sum()),int((f.decoded==3).sum()),int((f.valid_features==3).sum())]
axs[0,0].bar(['플레이트','QR 3개 디코딩','QR 3개 특징점'],np.array(stages)/len(f)*100,color=['#4477aa','#44aa99','#228833'])
for i,n in enumerate(stages):axs[0,0].text(i,n/len(f)*100+1,f'{n}/{len(f)} ({n/len(f):.1%})',ha='center')
axs[0,0].set(ylim=(0,110),ylabel='전체 영상 대비 성공률 (%)',title='전체 281장: 조건부 성공률로 누락을 숨기지 않음')
fail=q[q.failure!='ok'].groupby(['slot_name','failure']).size().unstack(fill_value=0).reindex(['L','C','R'],fill_value=0)
if len(fail.columns):
    fail.plot.bar(stacked=True,ax=axs[0,1],rot=0,color=['#cc3311','#ee7733','#aa4499','#bbbbbb','#332288'][:len(fail.columns)])
else:
    axs[0,1].text(.5,.5,'실패 슬롯 없음',ha='center',va='center',transform=axs[0,1].transAxes)
    axs[0,1].set_ylim(0,1)
axs[0,1].set(title='QR 위치별 실패 원인',xlabel='QR 위치',ylabel='실패 슬롯 수')
if len(fail.columns): axs[0,1].legend(fontsize=8)
for ax,axis in zip(axs[1],['T','Y']):
 g=q[q.axis==axis].sort_values('label')
 for slot,name in enumerate(['L','C','R']):
  part=g[g.slot==slot]; ax.scatter(part.label,np.full(len(part),slot),c=np.where(part.features_valid==1,'#228833','#cc3311'),s=15,marker='s')
 ax.set(yticks=[0,1,2],yticklabels=['L','C','R'],xlabel='파일명 명목값 ('+('°' if axis=='T' else 'mm')+')',title=f'{axis} 스캔: 초록 성공 / 빨강 실패')
fig.savefig(p/'01_detection_stability.png');plt.close(fig)
# Fig 2: feature count and retention, all frames; zero means unavailable.
fig,axs=plt.subplots(2,2,figsize=(13,8),layout='constrained')
refid=int(f[f.axis=='center'].frame.iloc[0]);ref=pt[(pt.frame==refid)&(pt.accepted==1)].set_index(['payload','id'])
refkeys=set(ref.index);accepted={i:g[g.accepted==1].set_index(['payload','id']) for i,g in pt.groupby('frame')}
ret=[]
for r in f.itertuples():
 cur=accepted.get(r.frame,pd.DataFrame());keys=set(cur.index) if len(cur) else set();common=keys&refkeys
 ret.append({'frame':r.frame,'axis':r.axis,'label':r.label,'common':len(common),'reference_retention':len(common)/len(refkeys)})
ret=pd.DataFrame(ret);ret.to_csv(p/'feature_retention.csv',index=False)
colors=['#4477aa','#ee7733','#228833']
for row,axis in enumerate(['T','Y']):
 for slot in range(3):
  g=q[(q.axis==axis)&(q.slot==slot)].sort_values('label');axs[row,0].plot(g.label,g.accepted,'.-',ms=2,lw=.6,label='LCR'[slot],color=colors[slot])
 axs[row,0].set(title=f'{axis}: 채택 특징점 수 (실패=0)',ylabel='특징점 수',xlabel='명목 라벨');axs[row,0].legend()
 g=ret[ret.axis==axis].sort_values('label');axs[row,1].plot(g.label,g.reference_retention*100,'.-',ms=2,color='#4477aa',lw=.7)
 axs[row,1].set(title=f'{axis}: 기준 영상의 111개 점 중 공통 채택점',ylabel='유지율 (%)',xlabel='명목 라벨',ylim=(0,105))
fig.savefig(p/'02_feature_retention.png');plt.close(fig)
summary['reference_features']=len(refkeys)
summary['common_features_all3']=ret.merge(f[['frame','valid_features']],on='frame').query('valid_features==3')['common'].describe().to_dict()
# Fig 3: acceptance and ID retention. Denominator includes decoding failures.
fig,axs=plt.subplots(2,2,figsize=(13,8),layout='constrained')
for slot,axis in enumerate(['T','Y']):
 ids=sorted(refkeys);frames=f[f.axis==axis].sort_values('label');heat=np.zeros((len(ids),len(frames)))
 for j,r in enumerate(frames.itertuples()):
  keys=set(accepted[r.frame].index) if r.frame in accepted else set()
  heat[:,j]=[int(k in keys) for k in ids]
 axs[slot,0].imshow(heat,aspect='auto',interpolation='nearest',cmap='Greens',vmin=0,vmax=1,extent=[frames.label.min(),frames.label.max(),len(ids),0])
 axs[slot,0].set(title=f'{axis}: 기준 특징점 ID별 채택 지도',ylabel='(QR 내용, ID) 순서',xlabel='명목 라벨')
 rates=heat.mean(axis=1);axs[slot,1].hist(rates*100,bins=np.arange(0,105,5),color='#4477aa',edgecolor='white')
 axs[slot,1].set(title=f'{axis}: 점별 관측 유지율 분포',xlabel='전체 해당 축 영상 대비 채택률 (%)',ylabel='기준 특징점 개수')
 summary.setdefault('id_retention',{})[axis]={'median':float(np.median(rates)),'at_least_90pct':int((rates>=.9).sum()),'at_least_80pct':int((rates>=.8).sum())}
fig.savefig(p/'03_point_identity.png');plt.close(fig)
# Fig 4: local photometric/geometric quality is distinct from absolute measurement accuracy.
fig,axs=plt.subplots(2,2,figsize=(13,8),layout='constrained');good=pt[pt.accepted==1]
for ax,col,title,unit in zip(axs.ravel(),['shift','heldout_residual','eigenvalue','contrast'],['초기점→정밀화 이동량','평가점을 제외한 재투영 잔차','국소 최소 고유값','모델 흑백 사분면 대비'],['px','px','gradient response','8-bit intensity']):
 vals=good[col].dropna();ax.hist(vals,bins=45,color='#44aa99',edgecolor='white');ax.axvline(vals.quantile(.95),color='#cc3311',ls='--',label=f'95% 분위={vals.quantile(.95):.3g}');ax.legend();ax.set(title=title,xlabel=unit,ylabel='채택점 수')
summary['accepted_quality']=good[['shift','heldout_residual','eigenvalue','contrast']].quantile([.5,.95,.99]).to_dict()
fig.suptitle('채택된 점에 대한 조건부 분포 — 성공률 및 절대 정확도와 구분',fontsize=13)
fig.savefig(p/'04_point_quality.png');plt.close(fig)
# Mean-removed displacement for each frame, using only common accepted IDs in ALL 3 QRs.
relative=[];vectors=[]
for r in f[f.valid_features==3].itertuples():
 cur=accepted.get(r.frame);keys=sorted(set(cur.index)&refkeys)
 if len(keys)<24 or len(set(k[0] for k in keys))<3:continue
 base=ref.loc[keys][['x','y']].to_numpy();now=cur.loc[keys][['x','y']].to_numpy();disp=now-base;mean=disp.mean(axis=0);res=disp-mean
 relative.append(dict(frame=r.frame,axis=r.axis,label=r.label,n=len(keys),relative_rms_px=np.sqrt(np.mean(np.sum(res**2,axis=1))),mean_dx=mean[0],mean_dy=mean[1]))
 for key,xy,dxy in zip(keys,base,res):vectors.append(dict(frame=r.frame,payload=key[0],id=key[1],base_x=xy[0],base_y=xy[1],rx=dxy[0],ry=dxy[1],axis=r.axis,label=r.label))
relative=pd.DataFrame(relative);vectors=pd.DataFrame(vectors);relative.to_csv(p/'mean_removed_summary.csv',index=False);vectors.to_csv(p/'mean_removed_vectors.csv',index=False)
# Triples use the SAME point set for all three frames and center; never average different IDs.
triples=[]
for axis,step in [('T',.01),('Y',.05)]:
 for sign in [-1,1]:
  scan=f[(f.axis==axis)&(np.sign(f.label)==sign)].sort_values('label');rows=list(scan.itertuples())
  for a,b,c in zip(rows,rows[1:],rows[2:]):
   if not(np.isclose(b.label-a.label,step) and np.isclose(c.label-b.label,step)):continue
   if min(a.valid_features,b.valid_features,c.valid_features)<3:continue
   keys=sorted(set(accepted[a.frame].index)&set(accepted[b.frame].index)&set(accepted[c.frame].index)&refkeys)
   if len(keys)<24 or len(set(k[0] for k in keys))<3:continue
   pa,pb,pc=[accepted[r.frame].loc[keys][['x','y']].to_numpy() for r in [a,b,c]]
   # Remove common translation from second difference; includes mechanical/image variation.
   second=pc-2*pb+pa;second-=second.mean(axis=0)
   proxy=np.sqrt(np.mean(np.sum(second**2,axis=1))/6)
   triples.append(dict(axis=axis,sign=sign,label=b.label,n=len(keys),relative_local_noise_proxy_px=proxy))
triples=pd.DataFrame(triples);triples.to_csv(p/'relative_local_variation.csv',index=False)
summary['relative_noise_proxy']=triples.groupby('axis').relative_local_noise_proxy_px.agg(['count','median','mean','max']).to_dict('index')
fig,axs=plt.subplots(2,2,figsize=(13,8),layout='constrained')
for row,axis in enumerate(['T','Y']):
 g=relative[relative.axis==axis].sort_values('label');axs[row,0].plot(g.label,g.relative_rms_px,'.',ms=3,color='#4477aa');axs[row,0].set(title=f'{axis}: 공통 이동 제거 후 잔여 변위 크기',xlabel='명목 라벨',ylabel='잔여 벡터 RMS (px)')
 g=triples[triples.axis==axis];axs[row,1].plot(g.label,g.relative_local_noise_proxy_px,'.',ms=3,color='#ee7733');axs[row,1].set(title=f'{axis}: 동일 점 3프레임의 국소 요동 지표',xlabel='가운데 영상 명목 라벨',ylabel='2차 차분 RMS / √6 (px)')
fig.suptitle('각 프레임·3프레임 비교마다 동일 공통점으로 평균 제거; 구동 오차와 영상 오차는 분리되지 않음',fontsize=11)
fig.savefig(p/'05_mean_removed_stability.png');plt.close(fig)
# Matched vector plots: same fixed IDs across center, T+/-1 and Y+/-2; arrows magnified equally.
chosen=[]
for axis,labels in [('T',[-1.,1.]),('Y',[-2.,2.])]:
 for label in labels:
  r=f[(f.axis==axis)&np.isclose(f.label,label)].iloc[0];chosen.append(r)
keys=sorted(refkeys.intersection(*(set(accepted[int(r.frame)].index) for r in chosen)))
fig,axs=plt.subplots(2,2,figsize=(13,6.5),layout='constrained');base=ref.loc[keys][['x','y']].to_numpy();summary['vector_comparison_common_points']=len(keys)
for ax,r in zip(axs.ravel(),chosen):
 now=accepted[int(r.frame)].loc[keys][['x','y']].to_numpy();dxy=now-base;dxy-=dxy.mean(axis=0)
 ax.quiver(base[:,0],base[:,1],dxy[:,0],dxy[:,1],angles='xy',scale_units='xy',scale=1/80,color='#4477aa',width=.004)
 ax.scatter(base[:,0],base[:,1],s=4,color='black');ax.set_aspect('equal');ax.set_xlim(1520,2270);ax.set_ylim(1190,950)
 ax.set(title=f'{r.axis}{r.label:+.2f}: 동일 {len(keys)}점, 화살표 80배',xlabel='원본 x (px)',ylabel='원본 y (px)')
fig.suptitle('평균 이동 제거 벡터 — 동일 확대율, 회전/수축·팽창 성분을 제거하지 않음',fontsize=12)
fig.savefig(p/'06_mean_removed_vectors.png');plt.close(fig)
# Output machine-readable audit results and failed frame list.
q.to_csv(p/'qr_status_labeled.csv',index=False);f[f.valid_features<3].to_csv(p/'failed_frames.csv',index=False)
# JSON quantile keys already simple numeric values; handle numpy scalars explicitly.
(p/'stability_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,default=lambda x:x.item() if hasattr(x,'item') else str(x)),encoding='utf-8')
print(json.dumps(summary,ensure_ascii=False,indent=2,default=str))
