"""Descriptive extrapolated trial origins, all 630 points, no operational model changes."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
H=Path(__file__).resolve().parent;R=H.parent
plt.rcParams.update({'font.family':'Malgun Gothic','axes.unicode_minus':False,'font.size':11})
s=pd.read_csv(R/'eda_capture_inventory_20261022/all_341_shots.csv')
m=pd.read_csv(R/'joint_center_630_20261022/results/manifest.csv');z=np.load(R/'joint_center_630_20261022/results/measurements.npz')
lookup={v:i for i,v in enumerate(m.relative)}
s=s[(s.dataset==20261014)&(s.center==0)].copy()
D=z['D'][[lookup[v] for v in s.relative]].reshape(len(s),-1)
fits={};sens={'Y':[],'T':[]}
for trial in s.trial.unique():
    ix=np.flatnonzero(s.trial.to_numpy()==trial);axis='Y' if s.iloc[ix].y.ne(0).any() else 'T';cmd=s.iloc[ix][axis.lower()].to_numpy()
    beta=np.linalg.lstsq(np.c_[np.ones(len(ix)),cmd],D[ix],rcond=None)[0]
    fits[trial]=(ix,axis,cmd,beta);sens[axis].append(beta[1])
J=np.column_stack([np.mean(sens['Y'],axis=0),np.mean(sens['T'],axis=0)])
pinv=np.linalg.pinv(J);sv=np.linalg.svd(J,compute_uv=False)
out=[];shots=[];origin_fields=[]
for trial,(ix,axis,cmd,beta) in fits.items():
    # Each shot is extrapolated to command zero with that trial's fitted slope.
    zero=D[ix]-cmd[:,None]*beta[1];origin=zero.mean(0);yt=pinv@origin;projected=J@yt
    shared_zero=D[ix]-cmd[:,None]*J[:,0 if axis=='Y' else 1];shared_yt=pinv@shared_zero.mean(0)
    meanpix=origin.reshape(630,2).mean(0);a=s.iloc[ix];nearest=int(np.argmin(abs(cmd)))
    row=dict(trial=trial,group=a.group.iloc[0],axis=axis,count=len(ix),start=a.taken_at.iloc[0],end=a.taken_at.iloc[-1],first_command=float(cmd[0]),last_command=float(cmd[-1]),nearest_zero_command=float(cmd[nearest]),slope_dx_px=float(beta[1].reshape(630,2)[:,0].mean()),slope_dy_px=float(beta[1].reshape(630,2)[:,1].mean()),origin_dx_px=float(meanpix[0]),origin_dy_px=float(meanpix[1]),origin_Y_mm=float(yt[0]),origin_T=float(yt[1]),shared_slope_origin_Y_mm=float(shared_yt[0]),shared_slope_origin_T=float(shared_yt[1]),unexplained_field_rms_px=float(np.sqrt(np.mean((origin-projected)**2))))
    out.append(row);origin_fields.append(origin)
    for local,i in enumerate(ix):
        v=pinv@zero[local];shots.append(dict(trial=trial,file=s.iloc[i].file,command=float(cmd[local]),taken_at=s.iloc[i].taken_at,zero_Y=float(v[0]),zero_T=float(v[1]),zero_dx_px=float(zero[local].reshape(630,2)[:,0].mean()),zero_dy_px=float(zero[local].reshape(630,2)[:,1].mean())))
o=pd.DataFrame(out);p=pd.DataFrame(shots);o.to_csv(H/'trial_origins.csv',index=False,encoding='utf-8-sig');p.to_csv(H/'per_shot_zero_estimates.csv',index=False,encoding='utf-8-sig')
np.savez_compressed(H/'origin_fields.npz',J=J,trial_origins=np.array(origin_fields),ids=z['ids'])
colors=plt.get_cmap('tab10').colors
def scatter(ax,cloud=False):
    for k,row in o.iterrows():
        a=p[p.trial==row.trial]
        if cloud:ax.scatter(a.zero_Y,a.zero_T,s=12,alpha=.28,color=colors[k])
        ax.scatter(row.origin_Y_mm,row.origin_T,s=95,color=colors[k],edgecolor='white',linewidth=.8,zorder=5)
        ax.annotate(f'{row.trial} ({row.origin_Y_mm:+.4f}, {row.origin_T:+.4f})',(row.origin_Y_mm,row.origin_T),xytext=(8,8 if k%2==0 else -18),textcoords='offset points',fontsize=10)
    ax.scatter(0,0,marker='*',s=150,c='black',zorder=6,label='고정 center (0,0)');ax.axhline(0,color='gray',lw=.7);ax.axvline(0,color='gray',lw=.7)
    ax.set_xlabel('Y 원점 편차 (mm)');ax.set_ylabel('T 원점 편차 (물리 단위 미확정)');ax.grid(alpha=.2);ax.margins(.25);ax.legend(loc='best')
fig,ax=plt.subplots(figsize=(12,8),layout='constrained');scatter(ax)
ax.set_title('6개 trial의 역추정 원점 · 각 trial의 명령 이동을 제거한 630점 변위로 환산\n고정 center 대비 편차 / 실제 원점 촬영값이 아닌 선형 외삽',pad=16)
fig.savefig(H/'01_origins_yt.png',dpi=160);plt.close(fig)
fig,axs=plt.subplots(1,2,figsize=(16,7),layout='constrained');scatter(axs[0],True);axs[0].set_title('사진별 역추정 원점과 trial 평균')
for k,row in o.iterrows():
    axs[1].plot([row.origin_Y_mm,row.shared_slope_origin_Y_mm],[row.origin_T,row.shared_slope_origin_T],'-',color=colors[k],alpha=.6)
    axs[1].scatter(row.origin_Y_mm,row.origin_T,c=[colors[k]],s=70,marker='o');axs[1].scatter(row.shared_slope_origin_Y_mm,row.shared_slope_origin_T,c=[colors[k]],s=70,marker='x');axs[1].annotate(row.trial,(row.origin_Y_mm,row.origin_T),xytext=(5,5),textcoords='offset points')
axs[1].axhline(0,color='gray',lw=.7);axs[1].axvline(0,color='gray',lw=.7);axs[1].set_xlabel('Y 원점 편차 (mm)');axs[1].set_ylabel('T 원점 편차 (단위 미확정)');axs[1].set_title('기울기 가정에 따른 변화: ○ trial별 / × 축별 공통');axs[1].grid(alpha=.2)
fig.savefig(H/'02_cloud_and_sensitivity.png',dpi=150);plt.close(fig)
audit=dict(images=len(s),points=630,condition_number=float(sv[0]/sv[1]),sensitivity_cosine=float(J[:,0]@J[:,1]/(np.linalg.norm(J[:,0])*np.linalg.norm(J[:,1]))),method='Each trial: D_i = b_trial + a_trial * command_i. Extrapolate b_trial = mean(D_i - a_trial*command_i). J columns are equally weighted mean per-trial slopes for Y and T. Solve J*[Y0,T0] = b_trial using all 1260 components, unweighted least squares.',interpretation='Apparent offset of command-zero pose relative to fixed reference, not independently measured physical origin; descriptive in-sample fit, not validation. Small field differences and slope assumptions influence Y/T separation. No trial IDs or labels added to deployed inference.')
(H/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
report='''<!doctype html><html lang="ko"><meta charset="utf-8"><title>trial별 Y/T 원점 추정</title><style>body{font:16px/1.7 "Malgun Gothic",sans-serif;max-width:1400px;margin:30px auto;padding:0 24px;color:#223}img{max-width:100%}table{border-collapse:collapse;font-size:13px}td,th{padding:8px;border-bottom:1px solid #ccd;white-space:nowrap}.scroll{overflow-x:auto}</style><h1>각 trial의 명령 이동을 제거했을 때 남는 원점</h1><p>원점 편차 = 현재−고정 center 변위 − (명령값 × trial별 픽셀 이동 기울기). 사진별로 계산한 값을 trial 안에서 평균합니다. 최소 |명령값| 사진만 사용하는 대신 전체 사진을 사용하며, 양·음수 명령의 부호를 유지합니다.</p><p>평균 수평 픽셀 하나만으로 Y와 T 두 값을 동시에 결정할 수 없으므로, 630점의 수평·수직 변위 전체를 축별 반응과 맞춰 Y/T로 환산했습니다. x축은 Y, y축은 T이며 고정 center를 (0,0)으로 표시합니다.</p><img src="01_origins_yt.png"><h2>수치와 trial 범위</h2><div class="scroll">'''+o.round(6).to_html(index=False,border=0)+'''</div><h2>사진별 산포와 기울기 가정의 영향</h2><img src="02_cloud_and_sensitivity.png"><p>작은 점은 사진별로 이동 명령을 제거한 결과이고 큰 점은 trial 평균입니다. 이는 서로 독립적으로 측정한 실제 원점 표본이나 신뢰구간이 아닙니다. 오른쪽은 trial별 기울기와 공통 기울기에 따른 추정 변화입니다.</p><p>추정점에는 실제 영점 차이 외에도 비선형 반응·시간 변화·측정 편향이 섞일 수 있습니다. 이 선형 외삽만으로 실제 Y/T 원점이 확정됐다고 해석하지 않습니다. 추론 모델이나 기존 결과는 변경하지 않았습니다.</p><p><a href="trial_origins.csv">trial별 수치 CSV</a> | <a href="per_shot_zero_estimates.csv">280장 사진별 역추정 CSV</a> | <a href="audit.json">방법 및 진단</a> | <a href="../eda_capture_inventory_20261022/report.html">시간순 촬영 이력</a> | <a href="../ALGORITHM_RULES.md">공통 규칙</a></p></html>'''
(H/'report.html').write_text(report,encoding='utf-8')
print(o[['trial','origin_dx_px','origin_dy_px','origin_Y_mm','origin_T','unexplained_field_rms_px']].to_string(index=False));print(json.dumps(audit,ensure_ascii=False))
