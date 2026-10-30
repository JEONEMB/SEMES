from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

H=Path(__file__).resolve().parent; R=H.parent
plt.rcParams.update({'font.family':'Malgun Gothic','axes.unicode_minus':False,'font.size':10,'path.simplify':False})
c=pd.read_csv(R/'analysis_grid_corners_20261019/out/corners_280.csv')
ref=c[c.file=='center.bmp'].sort_values('id'); ref=ref[ref.black_neighbors!=2]
ids=ref.id.to_numpy(); assert len(ids)==630
new=pd.read_csv(H/'new_points.csv'); assert new.id.nunique()==625 and new.file.nunique()==60
new['dx']=pd.to_numeric(new.dx,errors='coerce');new['dy']=pd.to_numeric(new.dy,errors='coerce')
metrics=[]; colors=['#2563eb','#d97706','#059669'];old_summary={}

def fit(x,y):
    mask=np.isfinite(x)&np.isfinite(y)
    if np.unique(x[mask]).size<3:return None
    b=np.polyfit(x[mask],y[mask],1); residual=y-np.polyval(b,x)
    ss=np.sum((y[mask]-y[mask].mean())**2)
    return b,residual,float(np.sqrt(np.mean(residual[mask]**2))),float(1-np.sum(residual[mask]**2)/ss) if ss>0 else np.nan

for component in ['dx','dy']:
    fig,axs=plt.subplots(4,2,figsize=(16,18))
    for col,axis in enumerate(['Y','T']):
        rows=c[c.id.isin(ids)&((c.file=='center.bmp')|c.file.str.startswith(axis+'+'))].copy()
        rows['command']=rows.file.str.extract(r'[YT]\+([0-9.]+)\.bmp',expand=False).astype(float).fillna(0)
        rows=rows.merge(ref[['id','x','y','qr']].rename(columns={'x':'ref_x','y':'ref_y','qr':'ref_qr'}),on='id')
        rows['displacement']=rows['x' if component=='dx' else 'y']-rows['ref_x' if component=='dx' else 'ref_y']
        invalid=0
        for pid,h in rows.groupby('id'):
            h=h.sort_values('command');x=h.command.to_numpy();v=h.displacement.to_numpy();k=int(h.ref_qr.iloc[0])-1
            # Motion-only fit keeps the observed center-to-motion discontinuity visible.
            result=fit(x[x>0],v[x>0])
            axs[0,col].plot(x,v,color=colors[k],lw=.45,alpha=.10)
            invalid+=int((~h.valid).sum())
            if result:
                b,res,rms,r2=result
                axs[1,col].plot(x[x>0],res,color=colors[k],lw=.45,alpha=.12)
                metrics.append(dict(dataset='old',axis=axis,component=component,id=int(pid),fixed=np.nan,diagonal=False,slope=b[0],intercept=b[1],residual_rms_px=rms,r2=r2,n=int(sum(x>0)),invalid_diagnostic=int((~h.valid).sum())))
        old_summary[axis]=dict(points=630,shots=rows.file.nunique(),diagnostic_failures=invalid)
        axs[0,col].set_title(f'기존 단일 축 | {axis} 상승, 반대 축=0 | 630점 전부')
        axs[1,col].set_title('기존: 점마다 양수 구간 직선 적합 후 잔차 (원점 별도)')
        axs[0,col].legend(handles=[Line2D([0],[0],color=co,label=lab) for co,lab in zip(colors,['왼쪽 QR','가운데 QR','오른쪽 QR'])],fontsize=9)

        varying=axis; fixed='T' if axis=='Y' else 'Y'
        vals=[]
        for fv,hg in new[new.center==0].groupby(fixed):
            if hg[varying].nunique()<3:continue
            vals.append(float(fv));color=plt.get_cmap('tab10')(len(vals)-1)
            for pid,h in hg.groupby('id'):
                # Display every available point and condition mean; no replacement of failures.
                s=h.groupby(varying)[component].mean().sort_index();x=s.index.to_numpy();y=s.to_numpy();diag=bool(h.diagonal.iloc[0])
                axs[2,col].plot(x,y,color='#777777' if diag else color,lw=.45,alpha=.07 if diag else .10,ls='--' if diag else '-')
                result=fit(h[varying].to_numpy(),h[component].to_numpy())
                if result:
                    b,res,rms,r2=result
                    residual=s-np.polyval(b,x)
                    axs[3,col].plot(x,residual,color='#777777' if diag else color,lw=.45,alpha=.07 if diag else .10,ls='--' if diag else '-')
                    # All three repetitions are shown as points in the residual panel.
                    axs[3,col].scatter(h[varying],res,s=1,color='#777777' if diag else color,alpha=.035,rasterized=True)
                    metrics.append(dict(dataset='new',axis=axis,component=component,id=int(pid),fixed=float(fv),diagonal=diag,slope=b[0],intercept=b[1],residual_rms_px=rms,r2=r2,n=int(np.isfinite(h[component]).sum()),invalid_diagnostic=int((h.finite==0).sum())))
        axs[2,col].set_title(f'새 ZIP | {axis} 상승 | 625점 × {len(vals)}개 고정 {fixed} 조건')
        axs[3,col].set_title('새 ZIP: 점·고정 조건별 직선 잔차 | 작은 점=각 반복 촬영')
        axs[2,col].legend(handles=[Line2D([0],[0],color=plt.get_cmap('tab10')(i),label=f'{fixed}={fv:g}') for i,fv in enumerate(vals)]+[Line2D([0],[0],color='#777777',ls='--',label='대각선 모서리 56점')],fontsize=8)
        for row in range(4):
            axs[row,col].set_xlabel(f'{axis} 명령값 (명령값 순 연결, 촬영 시간 궤적 아님)')
            axs[row,col].set_ylabel(('수평' if component=='dx' else '수직')+(' 이동량 (px)' if row%2==0 else ' 직선 잔차 (px)'))
            axs[row,col].grid(alpha=.20)
            if row%2:axs[row,col].axhline(0,color='black',lw=.7)
    direction='수평 Δx' if component=='dx' else '수직 Δy'
    fig.suptitle(f'모든 특징점의 {direction}: 명령값에 대한 선형성과 흔들림\n선 하나=동일한 물리적 점 하나 | 평균 대표선 아님 | 기준 사진 좌표를 뺀 변위',fontsize=18,y=.995)
    fig.text(.5,.006,'기존: 고정 630개 구조 모서리(대각선 125개 제외), 진단 실패도 유지. 새 ZIP: 검출 625점 전부(대각선 포함), 추적 실패는 NaN.\n새 ZIP의 선은 조건별 반복 평균, 잔차의 작은 점은 3회 반복 원측정. 잔차에는 비선형·위치 반복 오차·영상 측정 오차가 섞임.',ha='center',fontsize=10)
    fig.tight_layout(rect=[0,.035,1,.965]); fig.savefig(H/f'all_points_{component}.png',dpi=180); plt.close(fig)

pd.DataFrame(metrics).to_csv(H/'point_linearity_metrics.csv',index=False,encoding='utf-8-sig')
pd.DataFrame(metrics).groupby(['dataset','axis','component']).agg(points_or_curves=('id','size'),median_rms=('residual_rms_px','median'),p95_rms=('residual_rms_px',lambda a:a.quantile(.95)),max_rms=('residual_rms_px','max')).to_csv(H/'summary_metrics.csv',encoding='utf-8-sig')
summary=dict(old=old_summary,new_points=int(new.id.nunique()),new_diagonal=int(new.drop_duplicates('id').diagonal.sum()),new_failed_observations=int((new.finite==0).sum()),new_points_with_any_failure=int(new[new.finite==0].id.nunique()),scope='descriptive per-point EDA; motion-only free-intercept fits; no claim of pure noise or new-image prediction')
(H/'audit.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
html='''<!doctype html><html lang="ko"><meta charset="utf-8"><title>모든 점의 선형성과 노이즈</title><style>body{font:16px 'Malgun Gothic';margin:30px auto;max-width:1500px;line-height:1.7;color:#172238}img{width:100%}a{color:#2563eb}</style><h1>Y·T 명령값에 대한 모든 점의 픽셀 이동량</h1><p>한 장의 그림에 기존 단일 축과 새 ZIP을 함께 표시했다. 선 하나는 동일 특징점 하나다. 왼쪽은 Y, 오른쪽은 T이며 이동량 아래에 해당 점의 직선 잔차를 놓았다. 그림을 클릭하면 원본 크기로 볼 수 있다.</p><a href="all_points_dx.png"><img src="all_points_dx.png"></a><p>기존 자료: center에서 고정한 구조 모서리 630점(전체 755점 중 대각선 125점 제외). 기존 경계 직접 맞춤 좌표 CSV를 사용하며 진단 실패점도 제외하지 않았다. 모든 좌표에서 해당 점의 center 좌표를 뺐다. 양수 명령 구간만 회귀하고 center는 이동량 그림에 그대로 표시했다. 따라서 center 부근의 원점 어긋남은 숨기지 않으며, 잔차 패널은 양수 이동 구간의 직선성이다.</p><p>새 ZIP: 기존 C++ EDA 측정기를 수정 없이 호출해 점별 좌표를 재추출했다. 검출된 625점 전체이며 대각선 56점은 회색 점선으로 구분했다. 서로 다른 기준 사진·점 집합이므로 기존 630점과 동일 ID라고 가정하지 않는다. (0,0) 3장을 혼합 조건 상승선에 끼워 넣지 않았다. 고정한 다른 축이 다른 선들을 같은 원점 실험으로 해석하지 않는다. 상승값이 3수준 이상인 모든 고정 조건을 표시했다. 새 ZIP의 선은 각 점의 조건별 반복 평균, 잔차의 작은 점은 반복 3장의 원측정값이다.</p><p>추적 실패는 NaN으로 보존했으며 유한한 측정값만 그렸다. 결측 수준은 선이 끊기며 채워 넣지 않았다. 점별 직선 적합은 유한한 값에 대해서만 계산했다. 대각선과 실패 건수는 아래 감사 기록에 공개한다. 진단 실패가 곧 픽셀 측정 불가능을 뜻하지 않는 기존 구조 모서리 결과와 구분한다.</p><h2>수직 이동량도 동일하게 확인</h2><a href="all_points_dy.png"><img src="all_points_dy.png"></a><p>이동량은 signed Δx와 Δy로 나눴다. 거리 크기만 사용하면 방향과 원점 부근 정보가 사라지므로 주 그림은 수평 성분이다. 큰 이동 구간에서 직선처럼 보이는지와 미세 흔들림은 별도 잔차 축으로 확인한다. 잔차는 순수 센서 노이즈가 아니며 기계 반복 위치 오차·드리프트·영상 맞춤 오차와 비선형을 포함한다. 새 ZIP은 3회 반복이 있으나 기존 단일 축 자료에는 동일 조건 반복이 없어 원인과 노이즈 우열을 단정하지 않는다. 서로 다른 점의 기울기 차이도 곧 노이즈를 의미하지 않는다.</p><p>파일: <a href="point_linearity_metrics.csv">점별 기울기·절편·R²·잔차 CSV</a>, <a href="summary_metrics.csv">요약 CSV</a>, <a href="new_points.csv">새 ZIP 점별 원측정</a>, <a href="audit.json">건수 감사</a>. 기존 입력: analysis_grid_corners_20261019/out/corners_280.csv. 새 입력: hw-analysis/data/hw/bmp/20261020. 재현: export_points 20261020 출력CSV(실행 위치 hw-analysis), python plot_all.py. C++ 호출부와 CMake 설정도 이 폴더에 보존했다. <a href="../ALGORITHM_RULES.md">공통 규칙</a>. 명령값은 별도 계측 자세가 아니며 T 물리 단위는 미확인이다.</p></html>'''
html=html.replace('<a href="all_points_dx.png">','<p><b>읽는 방법:</b> 큰 이동량에서는 모든 점이 거의 직선으로 겹치지만, 잔차에서는 흔들림이 드러난다. 기존 Y에는 여러 점에서 함께 나타나는 지그재그가 있고 T에는 완만한 공통 곡률이 보인다. 이를 점마다 독립적으로 생기는 랜덤 노이즈로 단정할 수 없다. 기존 Y의 서로 다른 명령 간격 시행을 명령값 순으로 섞어 연결한 그림이므로 시행별 원점·촬영 조건 차이도 가능한 원인이다. 새 ZIP의 잔차에는 같은 조건 반복 3장의 산포가 함께 표시되어 있다.</p>'+'<a href="all_points_dx.png">',1)
html=html.replace('<h2>수직 이동량',f'<pre>{json.dumps(summary,ensure_ascii=False,indent=2)}</pre><h2>수직 이동량')
(H/'결과.html').write_text(html,encoding='utf-8')
print(json.dumps(summary,ensure_ascii=False,indent=2));print(pd.read_csv(H/'summary_metrics.csv').to_string(index=False))
