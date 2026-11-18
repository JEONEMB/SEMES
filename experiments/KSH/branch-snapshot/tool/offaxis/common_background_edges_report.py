"""Render the common background edge experiment from saved measurements."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'.analysis_packages'))
import json
import base64
import html
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common_background_edges import OUT,BASE,REGIONS
from plate_edges import predict


def main():
    c=pd.read_csv(OUT/'comparison.csv').set_index('model')
    held=pd.read_csv(OUT/'coordinate_holdout.csv')
    ef=pd.read_csv(OUT/'edge_features.csv').set_index('path')
    samples=pd.read_csv(OUT/'edge_samples.csv.gz')
    repeat=pd.read_csv(OUT/'repeatability.csv')
    sweep=pd.read_csv(OUT/'Y_sweep_holdout.csv').set_index('model')
    names={'QR6':'QR만 (6개 특징)', 'common_shared4':'공통 경계만 (4개)',
        'QR6_common_shared4':'QR + 공통 경계 4개 · 주 비교',
        'QR6_common_segments8':'QR + 좌우 별도 경계 8개 · 탐색',
        'QR6_common_quadratic6':'QR + 곡률 경계 6개 · 탐색'}
    reference_rows=[]
    qr=pd.read_csv(BASE/'ppt-method-v2/qr_displacements.csv').pivot(index='path',columns='qr',values=['dx','dy'])
    refs=[p for p in ef.index if p.startswith('기준 위치/')]
    qx=qr.loc[refs,[(d,s) for s in 'LCR' for d in ['dx','dy']]].to_numpy()
    models=json.loads((OUT/'models.json').read_text(encoding='utf-8'))
    for name,cols in [('QR6',[]),('QR6_common_shared4',[e+'_'+s for e in ['upper','lower'] for s in ['position','slope']]),
        ('QR6_common_segments8',[e+'_'+s for e in REGIONS for s in ['position','slope']])]:
        x=qx
        if cols:x=np.c_[qx,ef.loc[refs,cols].to_numpy()-ef.loc['기준 위치/center1.bmp',cols].to_numpy()]
        model={k:np.array(v) if isinstance(v,list) else v for k,v in models[name].items()}
        p=predict(model,np.asarray(x,dtype=float))
        for path,z in zip(refs,p):reference_rows.append(dict(model=name,path=path,Y_prediction_ticks=z[0],yaw_prediction_ticks=z[1]))
    ref=pd.DataFrame(reference_rows)
    ref.to_csv(OUT/'reference_predictions.csv',index=False)
    detail=[]
    for name,g in held.groupby('model',sort=False):
        err=g[['Y_error_ticks','yaw_error_ticks']].to_numpy()
        for j,axis in enumerate(['Y','yaw']):
            detail.append(dict(model=name,axis=axis,absolute_error_median_ticks=np.median(abs(err[:,j])),
                absolute_error_p95_ticks=np.percentile(abs(err[:,j]),95),absolute_error_max_ticks=abs(err[:,j]).max()))
    pd.DataFrame(detail).to_csv(OUT/'error_tails.csv',index=False)
    primary='QR6_common_shared4';secondary='QR6_common_segments8'
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for row,name in enumerate([primary,secondary]):
        a=held[held.model.eq('QR6')].sort_values('path');b=held[held.model.eq(name)].sort_values('path')
        for j,(col,title) in enumerate([('Y_error_ticks','Y'),('yaw_error_ticks','yaw')]):
            ax=axes[row,j]
            ax.scatter(a[col],b[col],c=b.yaw_deg,cmap='viridis',s=27)
            lim=max(abs(a[col]).max(),abs(b[col]).max())*1.08
            ax.plot([-lim,lim],[-lim,lim],c='#bbb',lw=1)
            ax.axvspan(-1,1,color='#888',alpha=.09);ax.axhspan(-1,1,color='#259a65',alpha=.14)
            ax.axhline(0,color='#aaa',lw=.7);ax.axvline(0,color='#aaa',lw=.7)
            ax.set(xlim=(-lim,lim),ylim=(-lim,lim),xlabel='QR-only error (ticks)',ylabel='QR + common edge error (ticks)',title=f'{title}: '+('shared4 primary' if row==0 else 'segments8 exploratory'))
            ax.grid(alpha=.2)
    fig.savefig(OUT/'05_error_scatter.png',dpi=160);plt.close(fig)
    # Paired coordinate bootstrap: descriptive uncertainty on these same held-out predictions.
    rng=np.random.default_rng(271027);groups=np.arange(19);boot=[]
    baseline=held[held.model.eq('QR6')].groupby('group')[['Y_error_ticks','yaw_error_ticks']].agg(lambda z:np.mean(z*z)).to_numpy()
    for name in [primary,secondary]:
        current=held[held.model.eq(name)].groupby('group')[['Y_error_ticks','yaw_error_ticks']].agg(lambda z:np.mean(z*z)).to_numpy()
        index=rng.integers(0,19,size=(10000,19))
        improvement=1-np.sqrt(current[index].mean(1)/baseline[index].mean(1))
        for j,axis in enumerate(['Y','yaw']):
            low,high=np.percentile(improvement[:,j]*100,[2.5,97.5])
            boot.append(dict(model=name,axis=axis,improvement_low_percent=low,improvement_high_percent=high))
    pd.DataFrame(boot).to_csv(OUT/'paired_coordinate_bootstrap.csv',index=False)
    def img(file,caption):
        payload=base64.b64encode((OUT/file).read_bytes()).decode()
        return f'<figure><img src="data:image/png;base64,{payload}" alt="{html.escape(caption)}"><figcaption>{caption}</figcaption></figure>'
    def table(frame):return frame.to_html(index=False,border=0,escape=True,float_format=lambda x:f'{x:.4f}')
    scores=c.reset_index();scores['방법']=scores.model.map(names)
    scores['동시 ±1틱 (%)']=scores.both_within_tick*100
    scores=scores[['방법','Y_rmse_mm','yaw_rmse_deg','동시 ±1틱 (%)']]
    counts=samples.groupby(['edge','side']).agg(전체=('valid','size'),유효=('valid','sum'),중앙_잔차_RMSE=('pixel_rmse','median')).reset_index()
    ranges=held[held.model.eq('QR6')][['y_mm','yaw_deg']].agg(['min','max'])
    improvements=(1-c.loc[primary,['Y_rmse_mm','yaw_rmse_deg']]/c.loc['QR6',['Y_rmse_mm','yaw_rmse_deg']])*100
    rpt=repeat.copy();rpt['설명']=np.where(rpt.feature.str.endswith('_slope'),'정규화된 x 범위의 높이 계수(px); 각도 아님','경계 위치(px)')
    reference_display=ref.copy();reference_display['model']=reference_display.model.map(names)
    sweep_display=sweep.reset_index();sweep_display['model']=sweep_display.model.map(names);sweep_display.both_within_tick*=100
    doc=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
    <title>공통 배경 수평 경계 · Y/yaw 검증</title><style>
    body{{font:16px/1.75 system-ui,sans-serif;color:#21303c;background:#f3f5f7;margin:0}}main{{max-width:1120px;margin:auto;padding:32px 24px 80px}}
    h1{{font-size:30px;line-height:1.4}}h2{{margin-top:42px}}.lead{{font-size:20px;background:white;border-left:5px solid #269667;padding:20px}}
    figure{{margin:24px 0;background:white;padding:12px}}img{{width:100%;height:auto}}figcaption{{font-size:14px;color:#526372}}table{{border-collapse:collapse;width:100%;font-size:14px;background:white;display:block;overflow:auto}}th,td{{padding:10px;border-bottom:1px solid #ddd;text-align:right}}th:first-child,td:first-child{{text-align:left}}code{{background:#e7edf1;padding:2px 4px}}.note{{background:#fff3d8;padding:16px}}a{{color:#17649d}}
    </style><main><h1>공통 배경 경계를 넣으면 Y/yaw 예측이 좋아지는가?</h1>
    <p class="lead"><strong>좋아진다.</strong> QR에 공통 수평 경계 4개 특징을 추가했을 때, 좌표 보류 검증의 Y RMSE는 {improvements.iloc[0]:.1f}%, yaw RMSE는 {improvements.iloc[1]:.1f}% 줄었다. 두 축 동시 ±1틱 성공률은 29.8% → 52.6%였다.<br><strong>현재는 정밀 정렬의 완성보다, 유용한 보조 관측량을 확인한 결과다.</strong></p>
    <p>사용자가 새로 표시한 위쪽 배경 홈과 아래쪽 패널 경계를 좌우 노출 영역에서 추출했다. 중앙 장비의 외곽선은 특징에서 제외했다. 기존 두 번째 데이터 60장(혼합 57장 + 원점 기준 3장)만 사용했다.</p>
    <h2>1. 실제로 어디를 추출했나?</h2>
    {img('01_common_edges.png','기준 이미지에 그린 실제 추출 위치. 위쪽 홈의 밝아지는 경계(주황), 아래쪽 패널의 어두워지는 경계(청록). 중앙 가림 영역은 제외했다.')}
    <p>상·하 경계별로 좌우 각 30열, 이미지당 120개 밝기 프로파일을 적합했다. 60장 모두 4개 주 특징과 8개 좌우 특징이 생성됐고, 회귀 입력 누락 이미지는 0장이었다. 개별 프로파일의 유효 개수는 아래와 같다.</p>{table(counts)}
    <h2>2. 고정 픽셀 탐색 구간 없이 찾는 방법</h2>
    <ol><li>각 이미지의 QR 점 범위에서 QR 폭·높이와 중심을 계산한다. 이 범위로 중앙 장비를 피할 좌우 열과 넓은 상·하 탐색 띠를 정한다.</li>
    <li>탐색 띠 전체에서 원본 밝기에 배경 + PSF 계단 함수를 최소제곱 적합한다. 여러 폭의 모델을 직접 비교하여 각 열의 경계 후보를 만든다. Canny·Sobel·밝기 차분 검출기는 사용하지 않았다.</li>
    <li>좌우 후보를 함께 지나는 수평선의 일관성으로 초기 위치를 고른다. 첫 기준 이미지에서 찾은 선과 QR 이동으로 이후 이미지에 부드러운 추적 선호도를 준다. 참 좌표·Actual 값·파일 시각은 이 과정에 쓰지 않는다.</li>
    <li>각 후보 주변 원본 15×7 픽셀에 <code>I(n,t)=b+cₙn+cₜt+A·Φ((n−u)/σ)</code>를 연속 최적화한다. 위치 u, 흐림 폭 σ, 대비와 배경을 동시에 적합하고 이상치는 soft-L1 손실로 완화한다. 최종 위치는 정수 픽셀에 묶이지 않는다.</li></ol>
    <p class="note">절대 y 픽셀 위치를 고정하지는 않았지만, QR 대비 탐색 범위·밝기 변화 방향·수평성 등 구조 가정은 있다. 다른 장비에서도 경계가 같다는 전제하에 설계했으며, 실제 장비 간 전이는 이번 데이터로 검증하지 못했다. 서브픽셀 출력 자릿수는 절대 위치 정확도를 뜻하지 않는다.</p>
    {img('03_broad_model_search.png','기준 이미지의 넓은 QR 상대 탐색 띠와 직접 모델 적합으로 나온 후보들. 주황 선은 선택된 경계다.')}
    {img('02_numeric_profiles.png','원본 밝기 표본과 연속 PSF 계단 모델. 점선이 최종 서브픽셀 위치다. 기준 이미지의 위·아래 예시이며 전 이미지의 정답 오차를 뜻하지 않는다.')}
    <h2>3. 예측 정확도: 같은 57장에서 공정하게 비교</h2>
    <p>두 번째 데이터의 장비 Actual 기록을 기준 원점 Actual 평균에 대해 상대화했다. 기존 명령 라벨과 차이는 10⁻⁵ 미만이었다. Y 범위 {ranges.loc['min','y_mm']:.2f}~{ranges.loc['max','y_mm']:.2f}mm, yaw 범위 {ranges.loc['min','yaw_deg']:.5f}~{ranges.loc['max','yaw_deg']:.5f}°다. 한 틱은 Y 0.05mm, yaw 0.01°다.</p>
    <p>19개 좌표 중 하나의 반복 3장을 통째로 보류하는 검증을 19번 수행했다. 회귀의 표준화·결측 보완은 학습 자료에서만 계산했고, ridge 강도는 그 안의 18개 좌표를 다시 보류해 선택했다. 반복 사진이 학습·평가 양쪽에 섞이지 않는다. 기준 3장은 회귀 학습에 포함하지 않았다.</p>
    {table(scores)}
    <p><strong>주 비교:</strong> QR6은 세 QR의 x/y 평균 이동 6개다. 공통 경계 4개는 상·하 경계 각각의 위치와 기울기 계수다. 경계만 쓰면 Y가 크게 나빠지므로, QR과 결합하는 것이 핵심이다. 경계 자체의 반복 흔들림이 작다는 것만으로 Y/yaw 분리가 보장되지는 않는다.</p>
    <p><strong>추가 탐색:</strong> 좌우를 따로 적합한 8개 경계 특징을 더하면 Y 0.0672mm, yaw 0.01228°, 동시 ±1틱 61.4%(35/57)였다. 같은 검증 결과를 보고 표현을 비교한 수치이므로, 이 8개 표현의 우월성이 독립적으로 확인된 것은 아니다. 곡률까지 추가한 표현은 주 모델보다 좋아지지 않았다.</p>
    {img('04_prediction_comparison.png','동일한 좌표 보류 검증의 RMSE. 경계 단독의 Y 오차가 크며, QR 결합 시 두 축 모두 개선된다.')}
    {img('05_error_scatter.png','각 점은 평가 이미지다. 가로축은 QR만 사용한 오차, 세로축은 공통 경계를 추가한 오차다. 녹색 띠는 추가 모델의 ±1틱 구간이다. 위: 주 비교, 아래: 좌우 별도 탐색 모델.')}
    <h2>4. 더 어려운 보류와 원점에서의 한계</h2>
    <p>Y 값 하나에 해당하는 모든 yaw 좌표를 통째로 보류하면 오차가 다시 커졌다. 단위는 틱이며 마지막 열은 두 축 동시 ±1틱 비율(%)이다.</p>{table(sweep_display)}
    <p>혼합 57장 전체로 학습한 모델에, 학습에서 제외한 기준 3장을 넣었을 때의 출력은 아래와 같다. 실제 기준 위치에 대해 기대하는 출력은 (0, 0)이다. 이 결과는 기준점 사용 조건이 있는 확인용이며 별도 독립 시험은 아니다.</p>{table(reference_display)}
    <p class="note">좌표 보류는 현재 촬영 환경 안에서의 검증이다. 시간 순서·세션·조명과 좌표 사이의 상관관계가 남아 있을 수 있다. 개선의 원인을 순수한 기하 분리로 단정하거나, 반복 제어 5~10회 수렴을 보장할 수 없다. 원점에서의 바이어스와 큰 잔차가 남으므로 현재 모델을 곧바로 정밀 정렬기로 부르지는 않는다.</p>
    <h2>5. 추출 흔들림과 수치 검증</h2>
    <p>같은 명령 좌표의 3회 반복에서 좌표 평균을 뺀 pooled 표준편차다(19좌표, 자유도38). 상·하 공통 위치는 각각 0.0102px, 0.0210px였다. 기울기는 정규화된 x 범위에 대한 높이 계수로 저장돼 있으며 각도가 아니다. 이 값에는 촬영 사이 실제 변동도 포함되므로 추출기 자체의 절대 오차로 해석하지 않는다.</p>{table(rpt)}
    <p>PSF 계단 모델의 해석 Jacobian과 수치 미분 상대 차이는 약 1.0×10⁻⁹였다. 원본 60장의 SHA-256을 기존 기록과 대조했다. 원본 사진과 이전 장비 외곽 경계 결과는 수정하지 않았다.</p>
    <h2>다음 알고리즘에 반영할 결론</h2>
    <p><strong>공통 배경 경계를 QR의 보조 특징으로 채택할 근거는 생겼다.</strong> 다음 단계는 이 특징을 사용해 Y/yaw 보정 방향과 보정량을 추정하되, 기존 데이터에서 학습 범위를 벗어나거나 경계 품질이 낮을 때를 구분하는 것이다. 원점 근처 정밀도·축 전용 전환·반복 수렴은 별도로 검증해야 한다.</p>
    <h2>재현 자료</h2><p><code>python tool/offaxis/common_background_edges.py</code><br><code>python tool/offaxis/common_background_edges_report.py</code></p>
    <p>추출: <a href="edge_features.csv">경계 특징</a> · <a href="edge_samples.csv.gz">전체 프로파일 진단</a> · <a href="dynamic_search_regions.csv">이미지별 동적 탐색 구간</a><br>
    평가: <a href="comparison.csv">성능 표</a> · <a href="coordinate_holdout.csv">57장별 보류 예측</a> · <a href="error_tails.csv">오차 꼬리</a> · <a href="reference_predictions.csv">기준점 출력</a> · <a href="paired_coordinate_bootstrap.csv">좌표 묶음 재표집 참고 범위</a><br>
    기록: <a href="manifest.json">평가 기록·해시</a> · <a href="extraction_manifest.json">추출 기록·원본 해시</a> · <a href="numerical_checks.json">수치 검사</a></p>
    <p>재표집 범위는 이미 계산한 19개 좌표의 보류 예측을 좌표 단위로 짝지어 재표집한 조건부 참고치다. 세션·장비 변화나 모델 선택 불확실성을 포함한 신뢰 구간으로 사용하지 않는다.</p></main></html>'''
    (OUT/'report.html').write_text(doc,encoding='utf-8')
    print('Reference predictions:');print(ref.to_string(index=False))
    print('Valid profiles:',int(samples.valid.sum()),'/',len(samples))
    print('Report:',OUT/'report.html')


if __name__=='__main__':main()
