from pathlib import Path
import json, hashlib, zipfile
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import f as fdist

OUT=Path(__file__).resolve().parent
ROOT=OUT.parent
BASE=ROOT/'hw-analysis/docs/validation-runs/hw/eda'
plt.rcParams.update({'font.family':'Malgun Gothic','axes.unicode_minus':False,'font.size':10})
d=pd.read_csv(BASE/'20261020/csv/shots.csv')
old=pd.read_csv(BASE/'20261014/csv/shots.csv')
z=ROOT/'hw-analysis/data/hw/raw/zip/zip_20261020.zip'
with zipfile.ZipFile(z) as zz:
    names=[x for x in zz.namelist() if x.lower().endswith('.bmp')]
names=list((ROOT/'hw-analysis/data/hw/png/20261020').rglob('*.png'))
assert len(names)==len(d), (len(names),len(d))
d.to_csv(OUT/'source_shots.csv',index=False,encoding='utf-8-sig')
g=d[d.center==0].groupby(['y','t']).agg(mean=('corner_dx','mean'),sd=('corner_dx','std'),n=('corner_dx','size'),ecc=('ecc_dx','mean')).reset_index()
g.to_csv(OUT/'condition_statistics.csv',index=False,encoding='utf-8-sig')
metrics=[]
fig,axs=plt.subplots(2,2,figsize=(14,10))
for col,(axis,fixed) in enumerate([('y','t'),('t','y')]):
    for v,sub in g.groupby(fixed):
        if len(sub)<3: continue
        sub=sub.sort_values(axis); x=sub[axis].to_numpy(); y=sub['mean'].to_numpy()
        p=np.polyfit(x,y,1); pred=np.polyval(p,x); res=y-pred
        raw=d[(d[fixed]==v)&(d.center==0)]
        sse_pure=sum(((h.corner_dx-h.corner_dx.mean())**2).sum() for _,h in raw.groupby(axis))
        sse_lof=float(np.sum(sub.n*res**2)); df_lof=len(sub)-2; df_pure=len(raw)-len(sub)
        F=(sse_lof/df_lof)/(sse_pure/df_pure) if sse_pure>0 else np.nan
        rec=dict(axis=axis,fixed_axis=fixed,fixed_value=v,levels=len(sub),slope=p[0],intercept=p[1],r2=1-np.sum(res**2)/np.sum((y-y.mean())**2),mean_residual_rms_px=np.sqrt(np.mean(res**2)),max_residual_px=max(abs(res)),pooled_repeat_sd_px=np.sqrt(sse_pure/df_pure),lack_of_fit_p=fdist.sf(F,df_lof,df_pure))
        metrics.append(rec)
        label=f'{fixed.upper()}={v:g}'
        line=axs[0,col].errorbar(x,y,yerr=sub.sd,fmt='o',capsize=4,label=label+' 평균 ± 1 SD')
        axs[0,col].plot(x,pred,'--',color=line[0].get_color(),alpha=.8)
        axs[1,col].errorbar(x,res,yerr=sub.sd,fmt='o-',capsize=4,label=label)
    axs[0,col].set(title=f'{fixed.upper()} 고정 → {axis.upper()} 상승',ylabel='기준 사진 대비 수평 이동 (px)',xlabel=f'{axis.upper()} 명령값')
    axs[1,col].set(title='각 구간의 자유 절편 직선에서 벗어난 양',ylabel='조건 평균 - 직선 (px)',xlabel=f'{axis.upper()} 명령값')
    axs[1,col].axhline(0,color='black',lw=.8)
    for a in axs[:,col]: a.legend(fontsize=8); a.grid(alpha=.25)
fig.suptitle('20261020 | 축별 선형 추세와 반복 산포\n점: 조건별 3장 평균 / 오차막대: 반복 표준편차 / 점선: 회귀 직선',fontsize=15)
fig.tight_layout(); fig.savefig(OUT/'01_axis_linearity.png',dpi=170); plt.close(fig)
m=pd.DataFrame(metrics); m.to_csv(OUT/'linearity_metrics.csv',index=False,encoding='utf-8-sig')
fig,axs=plt.subplots(1,3,figsize=(16,5))
for _,h in d[d.center==0].groupby(['y','t']):
    axs[0].plot(h.rep,h.corner_dx-h.corner_dx.mean(),'o-',alpha=.65)
axs[0].set(title='동일 조건 3회 반복의 흔들림',xlabel='파일의 반복 번호 (조건 안에서만 연결)',ylabel='조건 평균 대비 수평 이동 (px)'); axs[0].axhline(0,color='k',lw=.8)
cent=d[d.center==1]
axs[1].plot(cent.minutes,cent.corner_dx,'o-',label='모서리'); axs[1].plot(cent.minutes,cent.ecc_dx,'s--',label='ECC')
axs[1].set(title='(0,0) 기준 위치 3장',xlabel='기록된 촬영 시각 차이 (분)',ylabel='첫 기준 사진 대비 수평 이동 (px)'); axs[1].legend()
axs[2].scatter(d.corner_dx,d.corner_dx-d.ecc_dx,c=d.minutes,cmap='viridis'); axs[2].axhline(0,color='k',lw=.8)
axs[2].set(title='두 측정 방법의 차이',xlabel='모서리 수평 이동 (px)',ylabel='모서리 - ECC (px)')
for a in axs: a.grid(alpha=.25)
fig.tight_layout(); fig.savefig(OUT/'02_repeat_noise.png',dpi=170); plt.close(fig)

# Origin-constrained versus free-intercept additive descriptive fits; no prediction accuracy claim.
raw=d[d.center==0].copy(); fitrows=[]
fig,axs=plt.subplots(1,2,figsize=(13,5))
for a,origin in zip(axs,[True,False]):
    X=np.column_stack([raw.y,raw.t] if origin else [np.ones(len(raw)),raw.y,raw.t])
    b=np.linalg.lstsq(X,raw.corner_dx,rcond=None)[0]; r=raw.corner_dx-X@b
    raw['residual_origin' if origin else 'residual_free']=r
    a.scatter(raw.minutes,r,c=raw.y,cmap='viridis'); a.axhline(0,color='k',lw=.8)
    a.set(title=('원점 통과' if origin else '자유 절편')+f' 가산 직선 | RMS {np.sqrt(np.mean(r*r)):.3f} px',xlabel='기록된 촬영 시각 차이 (분)',ylabel='측정 - 적합 (px)'); a.grid(alpha=.25)
    fitrows.append(dict(model='origin' if origin else 'free_intercept',coefficients=b.tolist(),rms_px=float(np.sqrt(np.mean(r*r)))))
fig.tight_layout(); fig.savefig(OUT/'03_origin_offset.png',dpi=170); plt.close(fig)
raw.to_csv(OUT/'shot_residuals.csv',index=False,encoding='utf-8-sig')

fig,axs=plt.subplots(2,2,figsize=(13,9)); oldmetrics=[]
for col,(axis,fixed) in enumerate([('y','t'),('t','y')]):
    h=old[(old[fixed]==0)&(old[axis]>=0)].sort_values(axis)
    x=h[axis].to_numpy(); y=h.corner_dx.to_numpy(); p=np.polyfit(x,y,1); r=y-np.polyval(p,x)
    axs[0,col].plot(x,y,'o',ms=3,label='기존 단일 축 사진'); axs[0,col].plot(x,np.polyval(p,x),'--',label='회귀 직선')
    axs[0,col].set(title=f'기존 {fixed.upper()}=0, {axis.upper()}≥0',xlabel=f'{axis.upper()} 명령값',ylabel='수평 이동 (px)'); axs[0,col].legend()
    axs[1,col].plot(x,r,'o-',ms=3); axs[1,col].axhline(0,color='k',lw=.8); axs[1,col].set(xlabel=f'{axis.upper()} 명령값',ylabel='직선 잔차 (px)',title='명령값 순 연결 (촬영 시간 궤적 아님)')
    oldmetrics.append(dict(axis=axis,n=len(h),slope=p[0],intercept=p[1],rms_px=np.sqrt(np.mean(r*r)),r2=1-np.sum(r*r)/np.sum((y-y.mean())**2)))
for a in axs.flat: a.grid(alpha=.25)
fig.suptitle('참고: 기존 20261014 단일 축 자료의 양수 구간\n새 ZIP과 촬영 조건·측정점이 다르므로 별도 분석',fontsize=14); fig.tight_layout(); fig.savefig(OUT/'04_old_zero_axis_reference.png',dpi=170); plt.close(fig)
pd.DataFrame(oldmetrics).to_csv(OUT/'old_zero_axis_metrics.csv',index=False,encoding='utf-8-sig')
# Direct comparison: remove each fitted intercept only for slope visualization.
fig,axs=plt.subplots(2,2,figsize=(14,10))
compare=[]
for col,(axis,fixed) in enumerate([('y','t'),('t','y')]):
    h=old[(old[fixed]==0)&(old[axis]>=0)].sort_values(axis)
    x=h[axis].to_numpy(); y=h.corner_dx.to_numpy(); p=np.polyfit(x,y,1)
    axs[0,col].plot(x,y-p[1],'.',color='black',label='기존: 반대 축 0, 적합 절편 제거')
    axs[0,col].plot(x,p[0]*x,'--',color='black')
    axs[1,col].plot(x,y-np.polyval(p,x),'.',color='black',label='기존 단일 축')
    for _,r in m[m.axis==axis].iterrows():
        h2=g[g[fixed]==r.fixed_value].sort_values(axis)
        x2=h2[axis].to_numpy(); y2=h2['mean'].to_numpy()
        label=f'새 ZIP: {fixed.upper()}={r.fixed_value:g}'
        axs[0,col].errorbar(x2,y2-r.intercept,yerr=h2.sd,fmt='o',capsize=3,label=label)
        axs[1,col].errorbar(x2,y2-(r.intercept+r.slope*x2),yerr=h2.sd,fmt='o-',capsize=3,label=label)
        compare.append(dict(axis=axis,fixed_value=r.fixed_value,old_slope=p[0],new_slope=r.slope,slope_difference_percent=100*(r.slope/p[0]-1),old_intercept=p[1],new_intercept=r.intercept))
    axs[0,col].set(title=f'{axis.upper()} 상승 기울기 직접 비교',xlabel=f'{axis.upper()} 명령값',ylabel='각 직선 절편을 제거한 수평 이동 (px)')
    axs[1,col].set(title='직선 잔차 비교',xlabel=f'{axis.upper()} 명령값',ylabel='수평 잔차 (px)'); axs[1,col].axhline(0,color='k',lw=.8)
    for a in axs[:,col]: a.grid(alpha=.25); a.legend(fontsize=8)
fig.suptitle('기존 단일 축 vs 20261020 혼합 축 | 동일한 수평 이동 지표 비교\n절편 제거는 기울기 비교용 표시이며 실제 원점 보정 결과가 아님',fontsize=14)
fig.tight_layout(); fig.savefig(OUT/'05_old_new_direct_comparison.png',dpi=170); plt.close(fig)
cmp=pd.DataFrame(compare)
cmp.to_csv(OUT/'old_new_comparison.csv',index=False,encoding='utf-8-sig')
meta=dict(zip_sha256=hashlib.sha256(z.read_bytes()).hexdigest(),extracted_png_count=len(names),shots=len(d),conditions=len(g),fits=fitrows,old=oldmetrics,corner_ecc_max_difference_px=float(abs(d.corner_dx-d.ecc_dx).max()),center_spread_px=float(cent.corner_dx.max()-cent.corner_dx.min()))
(OUT/'summary.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
table=m.round(6).to_html(index=False)
html=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><title>20261020 축별 선형성과 노이즈 EDA</title><style>body{{font:16px 'Malgun Gothic',sans-serif;max-width:1250px;margin:40px auto;line-height:1.7;color:#172238}}img{{width:100%}}table{{border-collapse:collapse;font-size:12px}}td,th{{padding:6px;border:1px solid #ccc}}.box{{background:#eef4fa;padding:20px}}h2{{margin-top:45px}}</style>
<h1>20261020 ZIP: Y·T 상승의 선형성과 노이즈</h1>
<div class="box">새 자료는 기준 (0,0) 3장 + 양수 혼합 조건 19개 × 3장 = 60장이다. T=0의 Y 상승과 Y=0의 T 상승은 새 ZIP에 없다. 따라서 다른 축을 고정한 상승 구간을 분석하고, 기존 단일 축 자료는 마지막에 별도로 표시했다.</div>
<p><b>관찰 결과:</b> Y 고정 구간별 R²는 0.999164~0.999972, T는 0.999999 이상으로 전체 추세는 직선에 가깝다. 조건 평균 직선 잔차 RMS는 Y 0.029~0.158 px, T 0.019~0.050 px로 각 구간의 반복 산포보다 작다. 반복 SD의 조건별 중앙값은 0.164 px, 최대 0.334 px다. 모든 구간의 lack-of-fit p는 0.05 이상이지만 비선형이 없다는 증거로 단정하지 않는다. 혼합 조건 자유 절편은 -0.799 px이며, 원점 통과 RMS 0.534 px가 자유 절편에서 0.169 px로 줄어 원점 오프셋 가능성이 두드러진다.</p><h2>1. 선형성과 반복 노이즈를 분리해서 보기</h2><img src="01_axis_linearity.png">
<p>각 고정 축 구간에 자유 절편 직선을 적합했다. 높은 R²만으로 미세한 비선형성을 배제하지 않는다. 아래 잔차가 반복 산포보다 큰지 함께 본다. 3회 반복 표준편차는 센서 잡음뿐 아니라 로봇 반복 위치 오차, 드리프트, 영상 측정 오차를 포함한다. 이 자료만으로 원인을 분리할 수 없다. 오차막대는 신뢰구간이 아니다.</p>
{table}<p>lack_of_fit_p는 반복 내 산포와 직선의 조건 평균 잔차를 비교한 기술적 진단이다. 독립·등분산 오차를 가정하며, 반복 수가 적고 촬영 순서가 조건과 연관되어 있어 확정적 비선형 판정에 사용하지 않는다. 다중 비교 보정 전 수치다.</p>
<h2>2. 반복 흔들림과 기준 위치</h2><img src="02_repeat_noise.png"><p>조건별 반복 SD 중앙값 {g.sd.median():.4f} px, 최대 {g.sd.max():.4f} px. 기준 사진의 수평 범위는 {meta['center_spread_px']:.4f} px. 모서리와 ECC의 사진별 차이 최대 {meta['corner_ecc_max_difference_px']:.4f} px. 방법 간 일치는 실제 위치 정확도 검증을 의미하지 않는다.</p>
<h2>3. 원점에서 곧바로 비례하는가?</h2><img src="03_origin_offset.png"><p>혼합 조건 57장에 Δx=aY+bT와 Δx=c+aY+bT를 각각 적합했다. 계수: {fitrows}. 자유 절편 개선은 원점 어긋남 가능성을 보여주지만 기계적 영점·드리프트·명령 기준 차이를 확정하지 않는다. 시간과 명령값이 함께 변하므로 시간 추세를 인과적으로 해석하지 않는다. 이 분석은 기술 EDA이며 시험 예측 성적이나 Y·T 동시 식별 성적이 아니다.</p>
<h2>4. 기존 단일 축과 새 ZIP 직접 비교</h2><img src="05_old_new_direct_comparison.png"><p>기존 자료의 Y 기울기는 14.335 px/명령단위, 새 ZIP은 14.360~14.544로 약 +0.17~+1.46% 차이다. 기존 T는 80.254, 새 ZIP은 80.382~80.589로 약 +0.16~+0.42% 차이다. 두 자료 모두 직선 경향이 강하다. 그림 위쪽은 각 자료의 적합 절편만 제거해 기울기를 비교하며, 기존 원점을 새 ZIP에 적용한 예측 결과가 아니다. 아래쪽은 직선 잔차로, 새 ZIP의 오차막대는 3회 반복 SD다. 기존 자료에는 동일 조건 반복이 없어 노이즈 크기의 우열을 판정하지 않는다. 축 범위와 명령 수준도 서로 다르다.</p><h2>5. 기존 0축 자료 상세</h2><img src="04_old_zero_axis_reference.png"><p>{oldmetrics}</p><p>기존 사진에는 조건별 반복이 부족하므로 직선 잔차를 순수 노이즈 표준편차로 해석하지 않는다. 새 자료와 기울기 차이는 촬영 날짜·기준 사진·측정점의 차이가 섞여 있다.</p>
<h2>범위·출처·재현</h2><p>입력 ZIP: hw-analysis/data/hw/raw/zip/zip_20261020.zip (분할 압축을 담은 외부 ZIP; 변환 PNG 60장과 CSV 행 수 일치, SHA256 {meta['zip_sha256']}). 기존 저장된 EDA CSV를 재분석했으며 이번 작업에서 영상을 재측정하지 않았다. 출처는 hw-analysis/docs/validation-runs/hw/eda/20261020/csv/shots.csv, 기존 비교는 20261014/csv/shots.csv. 새 ZIP의 이름은 수신 묶음 이름이며 CSV 촬영 기록은 2026-10-15다.</p>
<p>측정은 기존 EDA의 center1 고정 모서리 추적 평균(625점 중 대각선 56점 및 추적 실패 11점 제외, 558점)과 QR 3개 ECC 평균이다. 최신 고정 630점 경계 맞춤 결과와 구별한다. 전체점의 평균을 그린 그림이며 개별점 분포가 아니다. 명령값은 실제 계측 자세가 아니며 T 물리 단위는 미확인이다. 기준 사진과 양수 조건은 연속된 단일 축 원점 상승 실험이 아니다.</p>
<p>공통 규칙: <a href="../ALGORITHM_RULES.md">ALGORITHM_RULES.md</a>. 기존 실험을 보존하고 이 폴더에만 저장했다. 재현: python analyze.py. CSV·summary.json에 수치를 보존했다.</p></html>'''
(OUT/'결과.html').write_text(html,encoding='utf-8')
print(m.to_string(index=False)); print(json.dumps(meta,ensure_ascii=False,indent=2)); print('repeat SD median/max',g.sd.median(),g.sd.max())
