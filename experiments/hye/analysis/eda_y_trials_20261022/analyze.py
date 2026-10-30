from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
H=Path(__file__).resolve().parent;R=H.parent
plt.rcParams.update({'font.family':'Malgun Gothic','axes.unicode_minus':False,'font.size':10})
c=pd.read_csv(R/'analysis_grid_corners_20261019/out/corners_280.csv')
ref=c[c.file=='center.bmp'].sort_values('id');ref=ref[ref.black_neighbors!=2];ids=ref.id.to_numpy();assert len(ids)==630
s=pd.read_csv(R/'hw-analysis/docs/validation-runs/hw/eda/20261014/csv/shots.csv');s=s[s.file.str.startswith('Y')].copy();s['name']=s.file.str.replace('\\','/',regex=False).str.split('/').str[-1];s=s.sort_values('y').reset_index(drop=True)
assert len(s)==80 and s.name.nunique()==80
groups=list(s.groupby('group',sort=True).groups);colors=['#2563eb','#d97706','#059669','#dc2626']
data={a:c[c.id.isin(ids)].pivot(index='file',columns='id',values=a).loc[s.name,ids].to_numpy()-ref[a].to_numpy()[None,:] for a in ['x','y']}
assert all(np.isfinite(v).all() for v in data.values())
valid=c[c.id.isin(ids)].pivot(index='file',columns='id',values='valid').loc[s.name,ids].to_numpy()
x=s.y.to_numpy();G=np.column_stack([(s.group==g).to_numpy().astype(float) for g in groups])
models={'pooled':np.column_stack([np.ones(len(s)),x]),'trial_intercept':np.column_stack([G,x]),'trial_line':np.column_stack([G,G*x[:,None]])}
metrics=[];coefs=[];shot=[];summary={}
def lines(ax,x,y,color,alpha=.13):
    order=np.argsort(x);xx=x[order];yy=y[order]
    segments=np.stack([np.broadcast_to(xx[:,None],yy.shape),yy],axis=-1).transpose(1,0,2)
    ax.add_collection(LineCollection(segments,colors=color,linewidths=.45,alpha=alpha));ax.update_datalim(np.column_stack([np.repeat(xx,yy.shape[1]),yy.ravel()]));ax.autoscale_view()
for component,D in data.items():
    residuals={};fits={}
    for name,A in models.items():
        b=np.linalg.lstsq(A,D,rcond=None)[0];e=D-A@b;fits[name]=b;residuals[name]=e
        metrics.append(dict(component=component,model=name,rms_px=float(np.sqrt(np.mean(e**2))),sse=float(np.sum(e**2)),parameters_per_point=A.shape[1],observations_per_point=len(s)))
    for gi,g in enumerate(groups):
        ix=np.flatnonzero(s.group==g);xx=x[ix];v=D[ix];b=np.linalg.lstsq(np.column_stack([np.ones(len(ix)),xx]),v,rcond=None)[0];e=v-np.column_stack([np.ones(len(ix)),xx])@b
        common=e.mean(axis=1);spatial=e-common[:,None]
        for j,pid in enumerate(ids):coefs.append(dict(component=component,trial=g,id=int(pid),qr=int(ref.qr.iloc[j]),slope=b[1,j],intercept=b[0,j],rms_px=float(np.sqrt(np.mean(e[:,j]**2)))))
        for i,k in enumerate(ix):shot.append(dict(component=component,file=s.name.iloc[k],trial=g,command=x[k],minutes=s.minutes.iloc[k],common_residual_px=common[i],spatial_residual_rms_px=float(np.sqrt(np.mean(spatial[i]**2))),invalid_diagnostics=int((~valid[k]).sum())))
    # Main diagnostic: same positive range as the previous zigzag figure, fitted on positive data only.
    pos=x>0;xp=x[pos];dp=D[pos];gp=G[pos][:,G[pos].sum(axis=0)>0];trialnames=[g for g in groups if (s.loc[pos,'group']==g).any()]
    ap=np.column_stack([np.ones(sum(pos)),xp]);bp=np.linalg.lstsq(ap,dp,rcond=None)[0];ep=dp-ap@bp
    ai=np.column_stack([gp,xp]);bi=np.linalg.lstsq(ai,dp,rcond=None)[0];ei=dp-ai@bi
    al=np.column_stack([gp,gp*xp[:,None]]);bl=np.linalg.lstsq(al,dp,rcond=None)[0];el=dp-al@bl
    rms=[float(np.sqrt(np.mean(e**2))) for e in [ep,ei,el]]
    summary[component]=dict(positive_pooled_rms=rms[0],positive_trial_intercept_rms=rms[1],positive_trial_line_rms=rms[2],positive_sse_reduction_intercept=float(1-np.sum(ei**2)/np.sum(ep**2)),positive_sse_reduction_line=float(1-np.sum(el**2)/np.sum(ep**2)),positive_trial_intercept_mean=bi[:-1].mean(axis=1).tolist(),positive_trial_names=trialnames,positive_shared_slope_mean=float(bi[-1].mean()))
    fig,axs=plt.subplots(2,2,figsize=(15,9))
    for j,g in enumerate(trialnames):
        mask=s.loc[pos,'group'].to_numpy()==g
        # pooled residual connects interleaved trials only in the first diagnostic panel
        lines(axs[0,1],xp[mask],ep[mask],colors[j]);lines(axs[1,0],xp[mask],ei[mask],colors[j]);lines(axs[1,1],xp[mask],el[mask],colors[j])
    lines(axs[0,0],xp,ep,'#475569',.09)
    axs[0,0].set_title(f'① 시행을 섞어 연결: 공통 직선 잔차 | RMS {rms[0]:.4f}px')
    axs[0,1].set_title('② 같은 잔차를 시행별로 나눠 연결 (보정 전)')
    axs[1,0].set_title(f'③ 점별 공통 기울기 + 시행별 절편 | RMS {rms[1]:.4f}px')
    axs[1,1].set_title(f'④ 점별 시행별 기울기·절편 | RMS {rms[2]:.4f}px')
    for ax in axs.flat:ax.axhline(0,color='k',lw=.7);ax.grid(alpha=.2);ax.set(xlabel='Y 명령값 (명령값 순 연결)',ylabel='수평' if component=='x' else '수직');ax.set_ylabel(('수평' if component=='x' else '수직')+' 직선 잔차 (px)')
    lim=max(abs(axs[0,0].get_ylim()[0]),abs(axs[0,0].get_ylim()[1]));
    for ax in axs.flat:ax.set_ylim(-lim,lim)
    axs[0,1].legend(handles=[Line2D([0],[0],color=colors[j],label=g) for j,g in enumerate(trialnames)])
    fig.suptitle(f'기존 Y 양수 구간: 지그재그는 시행 차이인가? | 고정 630점 전부\n각 선은 동일 특징점 하나, 모든 패널 동일 세로 범위 | 절편 제거는 사후 EDA',fontsize=16)
    fig.tight_layout();fig.savefig(H/f'01_positive_comparison_{component}.png',dpi=165);plt.close(fig)
    fig,axs=plt.subplots(3,4,figsize=(19,11))
    for gi,g in enumerate(groups):
        ix=np.flatnonzero(s.group==g);xx=x[ix];v=D[ix];e=residuals['trial_line'][ix];coef=fits['trial_line'];cc=colors[gi]
        lines(axs[0,gi],xx,v,cc);lines(axs[1,gi],xx,e,cc)
        time_order=np.argsort(s.minutes.to_numpy()[ix]);tt=s.minutes.to_numpy()[ix][time_order];ee=e[time_order];lines(axs[2,gi],tt,ee,cc)
        for ax in axs[:,gi]:ax.grid(alpha=.2)
        axs[0,gi].set_title(f'{g}\n{len(ix)}장 × 630점');axs[0,gi].set_xlabel('Y 명령값')
        axs[1,gi].set_title(f'시행별 직선 잔차 | RMS {np.sqrt(np.mean(e*e)):.4f}px');axs[1,gi].set_xlabel('Y 명령값');axs[1,gi].axhline(0,color='k',lw=.7)
        axs[2,gi].set_title('동일 잔차를 기록된 촬영 시각순 연결');axs[2,gi].set_xlabel('첫 기준 사진부터 경과 (분)');axs[2,gi].axhline(0,color='k',lw=.7)
    for row in [1,2]:
        bound=max(max(abs(v) for v in ax.get_ylim()) for ax in axs[row]);
        for ax in axs[row]:ax.set_ylim(-bound,bound)
    for row in range(3):axs[row,0].set_ylabel(('수평' if component=='x' else '수직')+(' 변위 (px)' if row==0 else ' 残差 (px)'.replace('残差','잔차')))
    fig.suptitle('시행별로 분리한 전체 Y 80장·고정 630점 | '+('수평 Δx' if component=='x' else '수직 Δy')+'\ncenter 사진과 점 ID는 고정, 기준을 시행마다 재설정하지 않음 | 진단 불확실 점도 포함',fontsize=16)
    fig.tight_layout();fig.savefig(H/f'02_four_trials_{component}.png',dpi=165);plt.close(fig)
pd.DataFrame(metrics).to_csv(H/'model_comparison.csv',index=False,encoding='utf-8-sig');pd.DataFrame(coefs).to_csv(H/'per_point_trial_fits.csv',index=False,encoding='utf-8-sig');pd.DataFrame(shot).to_csv(H/'shot_common_and_spatial_residuals.csv',index=False,encoding='utf-8-sig')
s.to_csv(H/'shot_trial_manifest.csv',index=False,encoding='utf-8-sig')
audit=dict(points=630,shots=80,diagnostic_failures=int((~valid).sum()),finite_observations=int(np.isfinite(data['x']).sum()),summary=summary)
(H/'summary.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
q=summary['x'];shift=abs(np.diff(q['positive_trial_intercept_mean'])[0]);rms0=q['positive_pooled_rms'];rms1=q['positive_trial_intercept_rms'];rms2=q['positive_trial_line_rms']
coef_table=pd.DataFrame(coefs).groupby(['component','trial']).agg(slope_mean=('slope','mean'),intercept_mean_px=('intercept','mean'),median_point_rms_px=('rms_px','median')).reset_index()
shot_table=pd.DataFrame(shot).groupby(['component','trial']).agg(common_residual_rms_px=('common_residual_px',lambda v:float(np.sqrt(np.mean(v*v)))),spatial_residual_rms_px=('spatial_residual_rms_px',lambda v:float(np.sqrt(np.mean(v*v))))).reset_index()
trial_table=coef_table.merge(shot_table,on=['component','trial'])
trial_table.to_csv(H/'trial_summary.csv',index=False,encoding='utf-8-sig')
html=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><title>기존 Y 시행별 지그재그 진단</title><style>body{{font:16px 'Malgun Gothic';max-width:1500px;margin:30px auto;line-height:1.7;color:#172238}}img{{width:100%}}pre{{white-space:pre-wrap;background:#eef4fa;padding:20px}}</style><h1>기존 Y 지그재그: 시행별 절편과 시행 안의 잔차</h1>
<p><b>양수 구간의 지그재그는 시행 간 차이로 상당 부분 설명된다.</b> 점마다 하나의 직선을 적합할 때 잔차 RMS는 {rms0:.4f}px다. 점별 기울기를 공통으로 두고 시행별 절편만 허용하면 {rms1:.4f}px로 줄어, 잔차 제곱합의 {q['positive_sse_reduction_intercept']*100:.1f}%가 줄었다. 두 양수 시행의 점별 절편 차이 평균은 {shift:.4f}px다. 시행별 기울기까지 허용하면 RMS는 {rms2:.4f}px다. 이는 동일 자료의 사후 기술적 설명이며 원점 차이가 물리적 원인이라는 증명은 아니다.</p>
<a href="01_positive_comparison_x.png"><img src="01_positive_comparison_x.png"></a><p>①은 앞선 그림처럼 서로 다른 시행을 명령값 순으로 섞어 연결했다. ②는 같은 측정·같은 회귀 잔차를 시행별로 나눠 연결했으며 아직 절편을 보정하지 않았다. ③은 점마다 공통 기울기와 시행별 절편을 적합했다. ④는 점마다 시행별 기울기·절편을 적합했다. 4개 패널의 세로 범위는 같다. 이 비교는 공통 center의 원측정 변위를 보존한 채 회귀 잔차를 비교한 것이며 시행별 center를 새로 정한 결과가 아니다.</p>
<h2>양·음수 시행 4개를 각각 확인</h2><a href="02_four_trials_x.png"><img src="02_four_trials_x.png"></a><p>시행별로 분리해도 잔차가 0이 되지는 않는다. 남은 잔차는 로봇 위치 반복 오차, 시간 드리프트, 영상 경계 맞춤 오차, 비선형 등이 섞인 값이다. 같은 명령의 독립 반복 촬영이나 실제 자세 계측이 없으므로 이를 순수 랜덤 노이즈라고 판정할 수 없다. 여러 점의 잔차가 함께 움직이면 공통 이동·드리프트 가능성이 있으며 630점을 독립 반복 표본으로 세면 안 된다. 각 시행은 다른 명령 수준을 갖기 때문에 절편과 기울기는 설명 모델에 의존하고, Y=0의 시행별 사진을 직접 관측한 것이 아니다.</p>
<h2>수직 성분도 동일하게 비교</h2><img src="01_positive_comparison_y.png"><img src="02_four_trials_y.png">
<h2>시행별 기울기·절편·잔차 요약</h2>{trial_table.round(6).to_html(index=False)}<p>공통 잔차는 한 사진에서 630점 잔차를 평균한 값이며 공간 잔차는 각 점에서 그 사진의 공통 잔차를 뺀 값이다. 공간 잔차도 순수 영상 센서 노이즈는 아니며 지점별 광학·기하·측정 오차를 포함한다. 이 분해는 잔차의 동반 움직임을 확인하기 위한 기술 통계다.</p><h2>출처와 수치</h2><p>원좌표: analysis_grid_corners_20261019/out/corners_280.csv. 시행 구분: hw-analysis/docs/validation-runs/hw/eda/20261014/csv/shots.csv의 group을 파일명으로 결합했다. 임의로 시행을 나누지 않았다. Y 80장·고정 630점(대각선 제외), center 좌표를 뺀 Δx/Δy를 그대로 사용하고 진단 불확실 점도 포함했다. 입력 50,400개 관측 중 진단 실패 {audit['diagnostic_failures']}개이며 좌표는 모두 유한하다. 원점은 center 하나로 고정했다. 명령값을 사용한 절편 적합은 사후 EDA이며 운영 보정·교차검증 예측 성적으로 제시하지 않는다. 기존 결과는 보존한다. <a href="../ALGORITHM_RULES.md">공통 규칙</a>.</p><p><a href="per_point_trial_fits.csv">모든 점의 시행별 기울기·절편·잔차</a> | <a href="model_comparison.csv">전체 모형 비교</a> | <a href="shot_common_and_spatial_residuals.csv">사진별 공통·공간 잔차</a> | <a href="shot_trial_manifest.csv">시행·촬영 시각 목록</a></p><pre>{json.dumps(audit,ensure_ascii=False,indent=2)}</pre></html>'''
(H/'결과.html').write_text(html,encoding='utf-8');print(json.dumps(audit,ensure_ascii=False,indent=2))
