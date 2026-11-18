from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import json,html
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as md

H=Path(__file__).resolve().parent;R=H.parent
plt.rcParams.update({'font.family':'Malgun Gothic','axes.unicode_minus':False,'font.size':10})
parts=[]
for ds in ['20261014','20261020']:
    s=pd.read_csv(R/f'hw-analysis/docs/validation-runs/hw/eda/{ds}/csv/shots.csv')
    s['dataset']=ds
    s['relative']=ds+'\\'+s.file
    s['recorded_time']=pd.to_datetime(s.taken_at)
    s['mtime_kst']=[datetime.fromtimestamp((R/'hw-analysis/data/hw/bmp'/ds/Path(f.replace('\\','/'))).stat().st_mtime,ZoneInfo('Asia/Seoul')).strftime('%Y-%m-%d %H:%M:%S') for f in s.file]
    parts.append(s)
s=pd.concat(parts,ignore_index=True).sort_values(['dataset','recorded_time','file']).reset_index(drop=True)
m=pd.read_csv(R/'joint_center_630_20261022/results/manifest.csv')
z=np.load(R/'joint_center_630_20261022/results/measurements.npz')
lookup={v:i for i,v in enumerate(m.relative)}
idx=np.array([lookup[v] for v in s.relative]);assert len(set(idx))==341
s['common_dx_630']=z['D'][idx,:,0].mean(1);s['common_dy_630']=z['D'][idx,:,1].mean(1)
s['valid_points']=z['valid'][idx].sum(1);s['sha256']=m.sha256.iloc[idx].to_numpy()
s['condition']=[f'Y{y:+.2f} T{t:+.2f}' for y,t in zip(s.y,s.t)]
s['order_in_set']=s.groupby('dataset').cumcount()+1
s['gap_s']=s.groupby('dataset').recorded_time.diff().dt.total_seconds()
s['elapsed_min']=s.groupby('dataset').recorded_time.transform(lambda x:(x-x.min()).dt.total_seconds()/60)
trial_map={};num=0
for g in s[s.dataset=='20261014'].group.unique():
    if g!='center':num+=1;trial_map[g]=f'S{num}'
s['trial']=[('center' if c else trial_map[g]) if ds=='20261014' else ('center' if c else 'M'+str(int(rp))) for ds,g,c,rp in zip(s.dataset,s.group,s.center,s.rep)]
# M1/M2/M3 denote repetition labels only, never independent sweep trials.
s.to_csv(H/'all_341_shots.csv',index=False,encoding='utf-8-sig')
tr=[]
for g,a in s[(s.dataset=='20261014')&(s.center==0)].groupby('trial',sort=False):
    a=a.sort_values('recorded_time');axis='Y' if a.y.ne(0).any() else 'T';values=a[axis.lower()].to_numpy();delta=np.diff(values)
    tr.append(dict(trial=g,group=a.group.iloc[0],axis=axis,count=len(a),start=str(a.recorded_time.iloc[0]),end=str(a.recorded_time.iloc[-1]),duration_min=round((a.recorded_time.iloc[-1]-a.recorded_time.iloc[0]).total_seconds()/60,3),start_command=float(values[0]),end_command=float(values[-1]),minimum=float(values.min()),maximum=float(values.max()),step_values=','.join(f'{v:+.2f}' for v in np.unique(np.round(delta,2))),monotonic=bool(np.all(delta>=0) or np.all(delta<=0)),median_gap_s=float(a.gap_s.iloc[1:].median())))
tr=pd.DataFrame(tr);tr.to_csv(H/'single_axis_trials.csv',index=False,encoding='utf-8-sig')
old=s[s.dataset=='20261014'];blocks=old.loc[old.group.ne(old.group.shift()),'group'].tolist()
assert len(blocks)==7 and len(set(blocks))==7
conditions=[]
for (ds,y,t),a in s.groupby(['dataset','y','t'],sort=False):
    a=a.sort_values('recorded_time')
    conditions.append(dict(dataset=ds,Y=y,T=t,count=len(a),first=str(a.recorded_time.iloc[0]),last=str(a.recorded_time.iloc[-1]),span_s=(a.recorded_time.iloc[-1]-a.recorded_time.iloc[0]).total_seconds(),repetitions=','.join(str(int(v)) for v in a.rep),times=' | '.join(a.recorded_time.dt.strftime('%H:%M:%S')),common_dx_sd_px=float(a.common_dx_630.std()) if len(a)>1 else None,common_dy_sd_px=float(a.common_dy_630.std()) if len(a)>1 else None,dx_range_px=float(a.common_dx_630.max()-a.common_dx_630.min()),dy_range_px=float(a.common_dy_630.max()-a.common_dy_630.min())))
c=pd.DataFrame(conditions);c.to_csv(H/'all_conditions.csv',index=False,encoding='utf-8-sig')
rep=c[c['count']>1].copy();rep.to_csv(H/'repeated_conditions.csv',index=False,encoding='utf-8-sig')
centers=s[s.center==1].copy();centers.to_csv(H/'center_chronology.csv',index=False,encoding='utf-8-sig')
gaps=s[s.gap_s>=120].copy();gaps.to_csv(H/'gaps_at_least_2min.csv',index=False,encoding='utf-8-sig')

fig,axs=plt.subplots(4,1,figsize=(15,12),layout='constrained')
colors=['#1976b5','#d47b16','#26916a','#a4489b','#8a6946','#797979']
for k,ds in enumerate(['20261014','20261020']):
    a=s[s.dataset==ds]
    for j,axis in enumerate(['y','t']):
        ax=axs[k*2+j]
        if k==0:
            for color,(trial,b) in zip(colors,a[a.center==0].groupby('trial',sort=False)):
                ax.plot(b.recorded_time,b[axis],'.-',ms=4,lw=.8,color=color,label=trial)
                if axis==('y' if b.y.ne(0).any() else 't'):ax.annotate(trial,(b.recorded_time.iloc[len(b)//2],b[axis].iloc[len(b)//2]),xytext=(0,10),textcoords='offset points')
        else:
            for rp,marker in [(1,'o'),(2,'s'),(3,'^')]:
                b=a[(a.center==0)&(a.rep==rp)];ax.scatter(b.recorded_time,b[axis],s=20,marker=marker,label=f'반복 번호 {rp}')
            ax.legend(loc='upper left',ncol=3,fontsize=8)
        for _,row in a[a.center==1].iterrows():
            ax.scatter(row.recorded_time,0,marker='*',s=95,color='#b82738',zorder=5)
            ax.annotate(Path(row.file.replace('\\','/')).stem,(row.recorded_time,0),xytext=(0,-20),textcoords='offset points',fontsize=8)
        ax.set_ylabel('Y 명령 (mm)' if axis=='y' else 'T 명령 (단위 미확정)');ax.set_title(f'{ds} 묶음 · 기록 날짜 {a.recorded_time.iloc[0]:%Y-%m-%d} · {len(a)}장')
        ax.xaxis.set_major_formatter(md.DateFormatter('%H:%M'));ax.grid(alpha=.2);ax.set_xlabel('기록된 파일 수정 시각 (KST 해석)')
fig.suptitle('전체 341장 시간 순서 · 선은 같은 스윕 내부의 촬영 기록 연결이며 실제 로봇 경로가 아님',fontsize=14)
fig.savefig(H/'01_timeline.png',dpi=150);plt.close(fig)

fig,axs=plt.subplots(1,2,figsize=(15,6),layout='constrained')
for ax,ds in zip(axs,['20261014','20261020']):
    a=c[c.dataset==ds];sc=ax.scatter(a.Y,a['T'],s=40+20*a['count'],c=a['count'],cmap='viridis',vmin=1,vmax=3,edgecolors='k',linewidths=.3)
    if ds=='20261020':
        for _,v in a.iterrows():ax.annotate(str(int(v['count'])),(v.Y,v['T']),xytext=(4,4),textcoords='offset points',fontsize=8)
    ax.set_title(f'{ds}: {len(a)}개 명령 조건 / {int(a["count"].sum())}장');ax.set_xlabel('Y 명령 (mm)');ax.set_ylabel('T 명령 (단위 미확정)');ax.grid(alpha=.2)
fig.colorbar(sc,ax=axs,label='동일 명령 조건의 사진 수',ticks=[1,2,3]);fig.suptitle('명령 공간의 관측 범위 · 같은 명령은 같은 실제 자세를 보장하지 않음')
fig.savefig(H/'02_coverage.png',dpi=150);plt.close(fig)

mixed=s[(s.dataset=='20261020')&(s.center==0)].copy();groups=mixed.condition.unique()
pivot=mixed.pivot(index='condition',columns='rep',values='common_dx_630')
repeat2_higher=int((pivot[2]>pivot[1]).sum());repeat2_shift=float((pivot[2]-pivot[1]).median())
coverage=pd.crosstab(mixed.y,mixed.t).reindex(index=[.05,.1,.5,1.],columns=[.01,.02,.03,.1,.2,1.,2.],fill_value=0)
coverage.to_csv(H/'mixed_coverage_counts.csv',encoding='utf-8-sig')
fig,axs=plt.subplots(3,1,figsize=(15,12),layout='constrained')
for j,(col,title) in enumerate([('common_dx_630','수평 공통 이동의 조건 내 평균 대비 차이 (px)'),('common_dy_630','수직 공통 이동의 조건 내 평균 대비 차이 (px)')]):
    for rp,marker in [(1,'o'),(2,'s'),(3,'^')]:
        a=mixed[mixed.rep==rp];xx=[list(groups).index(g) for g in a.condition];res=a[col]-a.condition.map(mixed.groupby('condition')[col].mean());axs[j].scatter(xx,res,marker=marker,label=f'반복 {rp}')
    axs[j].axhline(0,color='gray',lw=.7);axs[j].set_xticks(range(len(groups)),groups,rotation=55,ha='right',fontsize=8);axs[j].set_ylabel(title);axs[j].legend(ncol=3);axs[j].grid(alpha=.2)
a=centers[centers.dataset=='20261020'];origin=a.iloc[0]
axs[2].plot(a.recorded_time,a.common_dx_630-origin.common_dx_630,'o-',label='Δx: center1 대비')
axs[2].plot(a.recorded_time,a.common_dy_630-origin.common_dy_630,'s-',label='Δy: center1 대비')
for _,v in a.iterrows():axs[2].annotate(Path(v.file.replace('\\','/')).stem,(v.recorded_time,v.common_dx_630-origin.common_dx_630),xytext=(5,6),textcoords='offset points')
axs[2].xaxis.set_major_formatter(md.DateFormatter('%H:%M'));axs[2].set_xlabel('10월 15일 파일 수정 시각');axs[2].set_ylabel('630점 평균 이동 차이 (px)');axs[2].legend();axs[2].grid(alpha=.2)
fig.suptitle('같은 명령의 반복 변동과 center 재촬영 · 실제 자세·시간 변화·측정 오차가 섞인 관측값')
fig.savefig(H/'03_repeats_centers.png',dpi=150);plt.close(fig)

audit=dict(total=len(s),mtime_matches=int((s.taken_at==s.mtime_kst).sum()),unique_hashes=int(s.sha256.nunique()),timestamp_ties=int(s.duplicated(['dataset','recorded_time']).sum()),single_trials=len(tr),mixed_conditions=len(groups),repeated_conditions=len(rep),center_count=len(centers),notes=['taken_at is generated from filesystem last_write_time, not camera timestamp.','Folder labels are collection-set labels; recorded dates differ.','S1-S6 are existing command groups confirmed as contiguous time blocks, not externally logged trial IDs.','M1/M2/M3 are per-condition repetition suffixes, not three full independent trials.','Same command does not prove identical actual pose; unobserved return-to-zero movements cannot be recovered.','Existing 630-point measurements are descriptive only; no models retrained.'])
(H/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
def table(df):return '<div class="scroll">'+df.to_html(index=False,border=0,na_rep='—',float_format=lambda v:f'{v:.4f}')+'</div>'
parts=['<!doctype html><html lang="ko"><meta charset="utf-8"><title>341장 촬영 기록과 trial 전체 정리</title><style>body{font:15px/1.6 "Malgun Gothic",sans-serif;max-width:1500px;margin:32px auto;padding:0 24px;color:#202936}h1,h2{line-height:1.35}table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:8px;border-bottom:1px solid #d8dfe6;white-space:nowrap;text-align:left}th{background:#edf2f7;position:sticky;top:0}.scroll{overflow-x:auto}img{max-width:100%}input{padding:10px;min-width:320px}a{color:#075bb5}</style><h1>341장 촬영 기록 · 단일축 trial · 동일 명령 반복</h1>',
'<p><b>시간 출처:</b> 기존 shots.csv의 taken_at은 BMP 파일 수정 시각으로 생성되었습니다. 현재 파일을 Asia/Seoul로 읽은 초 단위 시각과 341/341장 일치합니다. 카메라 촬영 로그는 아니므로 아래는 보존된 파일 시각 기준의 순서입니다. 폴더명 20261014/20261020을 촬영 날짜로 해석하지 않습니다.</p>',
'<p><b>범위:</b> 원본 341장 전부. 단일축 280장 + 혼합 19조건×3장=57장 + center 4장. 같은 명령 반복과 동일 실제 자세 반복은 구분합니다. 630개 점은 한 사진 안의 측정점입니다.</p>',
'<h2>1. 전체 시간 순서</h2><img src="01_timeline.png">',
'<h2>2. 단일축 6개 trial</h2><p>S1~S6는 기존 group을 시간 순으로 표기한 ID입니다. 파일명 기반 구간이 실제로 연속된 시간 블록인지 확인했습니다. 로봇 로그의 공식 trial ID는 아닙니다. 시작/끝은 각 구간에서 관측된 첫/마지막 사진의 명령값이며, 그 사이 또는 앞뒤에 촬영되지 않은 원점 복귀가 있었는지는 알 수 없습니다.</p>'+table(tr),
'<h2>3. 명령 공간의 관측 범위</h2><img src="02_coverage.png"><p>단일축 Y 80개와 T 200개 명령은 각 1장입니다. Y의 0.1 간격 시행과 0.05 홀수 시행은 서로 다른 명령을 측정했으므로 같은 조건 반복이 아닙니다. 혼합 자료는 모두 Y&gt;0,T&gt;0이며 관측 Y 4수준 × T 7수준 = 28개 양수 조합 중 19개만 존재합니다. 표의 0은 미촬영입니다.</p>'+table(coverage.reset_index()),
'<h2>4. 동일 명령 반복: 전체 20조건</h2><p>혼합 19조건은 각 3장. 추가 묶음의 (0,0)은 center1~3의 3장입니다. 반복 번호 1/2/3을 서로 다른 전체 trial로 보지 않습니다. 표의 표준편차는 고정 center 기준 630점 평균 변위의 표본 표준편차입니다.</p>'+table(rep),
f'<h2>5. 반복 변동 및 center 재촬영</h2><p>혼합 {len(groups)}조건 중 {repeat2_higher}조건에서 반복 2의 수평 공통 이동이 반복 1보다 큽니다. 반복 2−1 차이의 조건 간 중앙값은 {repeat2_shift:.4f}px입니다. 독립적인 무작위 잡음으로 단정하지 않습니다. S6는 −1.95→−0.05로 원점 쪽으로 진행했고 S2는 −0.10→−2.00으로 원점에서 멀어졌습니다.</p><img src="03_repeats_centers.png">'+table(centers[['dataset','file','taken_at','order_in_set','elapsed_min','common_dx_630','common_dy_630','valid_points']]),
'<h2>6. 2분 이상 기록 공백</h2><p>공백은 직전 사진과의 시각 차이이며 원인이나 그 사이의 이동을 뜻하지 않습니다.</p>'+table(gaps[['dataset','order_in_set','file','taken_at','gap_s']]),
'<h2>7. 전체 341장 사진별 기록</h2><p>검색: 파일명, trial, 시각, 명령값</p><input id="search" aria-label="사진 기록 검색" placeholder="예: S1, center3, Y+0.05"><span id="count"></span><div id="shots">'+table(s[['dataset','order_in_set','trial','group','file','taken_at','y','t','rep','elapsed_min','gap_s','common_dx_630','common_dy_630','valid_points']])+'</div>',
'<h2>파일</h2><p>'+ ' | '.join(f'<a href="{f}">{f}</a>' for f in ['all_341_shots.csv','single_axis_trials.csv','all_conditions.csv','repeated_conditions.csv','center_chronology.csv','gaps_at_least_2min.csv','audit.json'])+'</p><p>원본 및 기존 결과는 변경하지 않았습니다. 기준: <a href="../ALGORITHM_RULES.md">ALGORITHM_RULES.md</a>. 출처: hw-analysis/docs/validation-runs/hw/eda/{20261014,20261020}/csv/shots.csv, hw-analysis/tool/hw/eda/src/core.cpp, joint_center_630_20261022/results/{manifest.csv,measurements.npz}.</p>',
'<script>const q=document.getElementById("search"),rr=[...document.querySelectorAll("#shots tbody tr")],n=document.getElementById("count");function filter(){let k=0;rr.forEach(r=>{r.hidden=!r.textContent.toLowerCase().includes(q.value.toLowerCase());if(!r.hidden)k++});n.textContent=" "+k+" / 341장"}q.addEventListener("input",filter);filter();</script></html>']
(H/'report.html').write_text('\n'.join(parts),encoding='utf-8')
assert len(s)==341 and len(tr)==6 and len(groups)==19 and len(centers)==4
assert audit['mtime_matches']==341
print(json.dumps(audit,ensure_ascii=False,indent=2));print(tr.to_string(index=False));print(rep[['dataset','Y','T','count','first','last','span_s']].to_string(index=False))
