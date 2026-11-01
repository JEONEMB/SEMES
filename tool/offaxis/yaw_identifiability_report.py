"""Render a conclusion-first report of the Y/yaw signal-vs-jitter experiment."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'.analysis_packages'))
import base64
import hashlib
import json
import numpy as np
import pandas as pd
from PIL import Image,ImageDraw,ImageFont

OUT=ROOT/'docs/offaxis-eda/yaw-separation/identifiability'


def main():
    models=json.loads((OUT/'models_and_diagnostics.json').read_text())
    primary=models['full_axis_range__L630_full']['diagnostics']
    ss=pd.read_csv(OUT/'summary.csv')
    mix=pd.read_csv(OUT/'mixed_by_range.csv')
    stability=pd.read_csv(OUT/'pattern_split_stability.csv')
    val=pd.read_csv(OUT/'axis_coordinate_holdout.csv')
    noise=pd.read_csv(OUT/'repeat_sigma_by_range.csv')
    cases=[]
    for calib,weight,label in [('full_axis_range','unweighted','全範囲・等重み'),
                              ('full_axis_range','shrinkage_50_crossfit','全範囲・雑音重み'),
                              ('central_axis_range','shrinkage_50_crossfit','中心範囲・雑音重み')]:
        r=ss[ss.calibration.eq(calib)&ss.weighting.eq(weight)&ss.feature.eq('L630_full')].iloc[0]
        subset='within_central_calibration' if calib.startswith('central') else 'theta_within_old_axis'
        e=mix[mix.calibration.eq(calib)&mix.weighting.eq(weight)&mix.feature.eq('L630_full')&mix.subset.eq(subset)].iloc[0]
        n=noise[noise.calibration.eq(calib)&noise.weighting.eq(weight)&noise.feature.eq('L630_full')&noise.subset.eq(subset)].iloc[0]
        label={'全範囲・等重み':'전체 범위 · 기본','全範囲・雑音重み':'전체 범위 · 잡음 가중','中心範囲・雑音重み':'중심 범위 · 잡음 가중'}[label]
        cases.append(dict(method=label,Y_repeat_sigma_all=r.Y_repeat_sigma_ticks,yaw_repeat_sigma_all=r.yaw_repeat_sigma_ticks,
            Y_repeat_sigma_in_range=n.Y_repeat_sigma_ticks,yaw_repeat_sigma_in_range=n.yaw_repeat_sigma_ticks,
            mixed_in_range_images=int(e.images),Y_mixed_rmse=e.Y_rmse_ticks,yaw_mixed_rmse=e.yaw_rmse_ticks))
    t=pd.DataFrame(cases)
    own=primary['known_other_axis_repeat_sigma_ticks'];joint=primary['repeat_sigma_ticks']
    # A concise Korean visual makes the distinction from single-axis accuracy clear.
    card=Image.new('RGB',(1450,1050),'#f4f7fa');d=ImageDraw.Draw(card)
    font=lambda n:ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',n)
    bold=lambda n:ImageFont.truetype('C:/Windows/Fonts/malgunbd.ttf',n)
    d.text((55,40),'Y와 yaw, 현재 영상에서 구분할 수 있나?',font=bold(38),fill='#183d51')
    d.rounded_rectangle((45,115,1405,225),radius=12,fill='#fff0dd')
    d.text((65,135),'한 틱 차이를 안정적으로 구분하기에 충분하다고 확인되지 않았습니다.',font=bold(29),fill='#97561c')
    d.text((65,181),'축마다 잘 맞는 것과, 두 축이 섞인 상태에서 나누는 것은 다른 문제입니다.',font=font(24),fill='#735b47')
    d.text((60,270),'같은 반복 흔들림을 좌표 오차로 환산한 결과 (1σ)',font=bold(27),fill='#254b62')
    for x,text in [(70,'추정 조건'),(820,'Y'),(1110,'yaw')]:d.text((x,330),text,font=bold(26),fill='#38586d')
    for i,(label,values) in enumerate([('다른 축이 고정되어 있다고 알 때',own),('두 축이 모두 미지일 때',joint)]):
        y=390+i*95;d.rounded_rectangle((55,y-8,1395,y+65),radius=10,fill='white')
        d.text((75,y),label,font=font(27),fill='#24495e')
        d.text((800,y),f'{values[0]:.2f} 틱',font=bold(30),fill='#28754e' if i==0 else '#a35d20')
        d.text((1090,y),f'{values[1]:.2f} 틱',font=bold(30),fill='#28754e' if i==0 else '#a35d20')
    d.text((65,610),'검증에 사용한 자료',font=bold(27),fill='#24495e')
    d.text((65,660),'패턴: 단독축 280장 / L형 630점의 dx·dy 전체',font=font(26),fill='#455f70')
    d.text((65,705),'흔들림: 혼합 19좌표 × 반복 3장 / 공통 이동도 제거하지 않음',font=font(26),fill='#455f70')
    d.text((65,750),'한 틱: Y 0.05 mm, yaw 0.01° / 특징점 추출은 변경하지 않음',font=font(26),fill='#455f70')
    d.rounded_rectangle((50,830,1400,1000),radius=12,fill='#e8eef6')
    d.text((70,850),'국소 잡음 가중에서는 반복 흔들림이 약 0.31틱까지 줄었지만,',font=bold(25),fill='#30526b')
    d.text((70,895),'보정 범위 안 실제 혼합 예측 오차는 Y 4.20틱 · yaw 3.42틱입니다.',font=bold(25),fill='#30526b')
    d.text((70,945),'현재 선형 패턴의 식별·재현성이 부족하다는 결과이며, 모든 방법의 불가능성은 아닙니다.',font=font(22),fill='#4d6577')
    card.save(OUT/'summary.png')

    methodtext='''1. L형 630점의 dx/dy를 모두 유지해 영상 한 장을 1,260차원으로 표현했다. 특징점 추출값은 기존 고정 계측을 재사용했다. 3.76%의 기존 진단 탈락점도 임의로 골라 빼지 않았다.
2. 기존 Y 단독 80장·yaw 단독 200장으로 1틱당 이동 패턴 두 개를 적합했다. Y±의 두 파일 계열과 yaw±의 6개 세션 후보마다 별도 절편을 둬, 그룹 사이 기준 차이를 운동 신호로 사용하지 않았다. 주 분석은 선형 근사다.
3. 혼합 19좌표의 반복 3장에서 좌표별 평균만 빼 반복 잔차를 구했다. 이미지 전체의 공통 흔들림은 남겼다. 분산의 분모는 57이 아닌 19×(3−1)=38이다. 이 잔차에는 계측 잡음뿐 아니라 기구 재현오차, 조명 변화 등이 섞일 수 있다.
4. Y 패턴과 겹치는 성분을 yaw 패턴에서 제거하고, 같은 방향으로 반복 잔차를 투영했다. 반대 방향도 동일하게 계산했다. 한 틱 신호 크기 / 그 방향 잡음 표준편차를 SNR로 정의했다. 이는 두 축 공동 최소제곱의 반복 오차와 동치다.
5. 점별 잡음의 독립성을 가정하지 않기 위해 전체 반복 잔차 공분산의 50% 축소 추정도 비교했다. 최대 rank가 38인 공분산을 그대로 역행렬로 만들지 않았다. 혼합 좌표 하나와 반복 3장 모두를 제외해 가중을 추정하고, 그 보류 좌표의 잡음과 예측을 평가했다.
6. 전체 단독축 범위와 중심 범위(Y±0.5mm, yaw±0.2°)를 비교했다. 중앙 범위는 60장이다. 단독축 좌표 보류 검증, 축 자료를 두 부분으로 나눈 패턴 재현성, 신규 기준 영상 예측도 별도로 검사했다.'''

    md=f'''# 현재 L630 특징에서 Y와 yaw를 구별할 수 있는가

2026-10-26. **주 분석에서 한 틱 구별 신호는 반복 흔들림보다 작았다. 잡음 가중을 사용한 국소 변형에서는 반복성이 좋아졌지만, 실제 혼합 예측과 패턴 재현성까지 통과하지 못했다. 따라서 현재 데이터/보정으로 한 틱 분리가 충분하다고 확인할 수 없다.**

## 방법을 먼저 정하고 실행한 내용

{methodtext}

동일 좌표 반복을 학습·평가로 쪼개지 않았다. 주 패턴에는 혼합 영상을 전혀 사용하지 않았다. 혼합 예측의 원점은 새 세션의 실제 기준 영상 center1이며, 혼합 정답으로 절편을 다시 맞추지 않았다. 신규 기준 center1/2/3은 잡음 공분산 학습에 넣지 않았다. 각 그림과 수치는 새 yaw 기구학 모델을 적합한 결과가 아니라, 명령 축 이름에 따른 실측 영상 변화다.

## 주 결과: 전체 범위, 630점 전체, 기본 가중

| 지표 | Y | yaw |
|---|---:|---:|
| 다른 축과 구분되는 한 틱 신호 크기 (전체 벡터 L2, px) | {primary['unique'][0]['signal_l2_px']:.6f} | {primary['unique'][1]['signal_l2_px']:.6f} |
| 같은 단위 방향으로 투영한 반복 잡음 σ (px) | {primary['unique'][0]['noise_sd_on_unit_direction_px']:.6f} | {primary['unique'][1]['noise_sd_on_unit_direction_px']:.6f} |
| 한 틱 SNR | {primary['one_tick_snr'][0]:.3f} | {primary['one_tick_snr'][1]:.3f} |
| 두 축 미지일 때 반복 오차 σ (틱) | {joint[0]:.3f} | {joint[1]:.3f} |
| 다른 축을 안다고 가정했을 때 σ (틱) | {own[0]:.3f} | {own[1]:.3f} |

두 패턴의 벡터 각도는 {primary['pattern_angle_degrees']:.6f}°이고, 틱 단위 Jacobian의 조건수는 {primary['condition_number']:.2f}다. 이 각도는 실제 yaw 각도나 각도 추정 오차가 아니다. 큰 공통 이동은 잘 측정해도 그 원인이 Y인지 yaw인지 구분할 때 필요한 차이는 매우 작다. 단독축에서는 잘 맞는다는 사용자 관측과 모순되지 않는다.

Yaw 구별 성분의 dx/dy 성분당 RMS는 약 0.001434px이나, 630점 평균으로 잡음이 독립적으로 sqrt(N)만큼 줄어든다고 가정하지 않았다. SNR은 전체 벡터의 구별 방향에서 계산했다. 단순 점별 거리 평균으로 신호/잡음을 비교하지 않았다.

## 개선 여지와 검증 실패를 함께 본 결과

전체 범위의 잡음 가중에서는 반복 σ가 Y 1.873틱, yaw 1.793틱으로 줄었지만 한 틱보다 컸다. 중심 범위 + 잡음 가중은 모든 혼합 좌표의 반복 σ가 Y 0.314틱, yaw 0.309틱으로 줄었다. 이 조건만 보면 한 틱 SNR은 약 3.2다.

하지만 중심 보정 범위 안의 혼합 9좌표/27장에서도 실제 예측 RMSE는 Y **4.199틱**, yaw **3.417틱**이었다. 반복 잡음이 작아진 것과 실제 명령 좌표를 맞힌 것은 다른 결과다. 각 조건의 동일 범위 반복 σ도 repeat_sigma_by_range.csv에 별도로 저장했다. 전체 57장 중 yaw=2° 12장은 기존 yaw 단독 범위 밖이어서 외삽으로 표시하고, 범위 안의 45장 결과를 분리했다.

더 중요하게, 축 자료를 각 그룹 안에서 서로 겹치지 않는 두 위치 집합으로 나눠 구별 패턴을 다시 구하면 full-range yaw 패턴의 cosine은 약 **0.201**, 중심 범위는 약 **−0.065**다. 같은 구별 패턴이 안정적으로 재현됐다고 보기 어렵다. 패턴 보정 오차, 비선형성, 세션/좌표별 변동이 포함될 수 있고 이를 특징점 추출 잡음 하나로 단정하지 않는다.

단독축 중앙 블록을 보류한 주 모델은 Y RMSE 약 4.646틱, yaw 약 4.147틱이었다. 알려진 세션 그룹 절편을 쓸 수 있는 유리한 조건에서도 완전한 선형 분리는 되지 않았다. 따라서 혼합 예측 악화를 세션 차이 하나의 탓으로만 돌리지 않는다.

## 해석 범위

- 이 결과는 **현재 특징 + 실측 선형 패턴 보정의 신뢰성이 부족함**을 보여준다. 실제 광학계에서 어떤 알고리즘도 Y/yaw를 구분할 수 없다는 정보이론적 한계 증명은 아니다. 구별 패턴 추정 자체의 불확실성이 크다.
- 혼합 좌표는 한 신규 세션의 양의 19좌표뿐이다. 단독축 자료의 잡음 패턴이 신규 세션과 같다고 인증할 수 없다. 반복 잔차를 독립 백색잡음으로 가정하지 않았다. 파일의 (1)/(2)/(3)은 실제 시간순이 아니다.
- 명령값을 좌표 정답으로 썼다. 실측 장비 좌표의 절대 정확도 평가가 아니다.
- Bootstrap은 좌표별 반복 묶음을 재표집한 이 자료 내 민감도이며, 미래 촬영 세션 성능의 90% 신뢰구간이 아니다. Jacobian 보정 불확실성도 포함하지 않는다.
- 조건수가 좋아져도 신호의 절대 크기가 작으면 구별이 어렵다. 평균 제거 특징은 full-range에서 조건수 약 1.32로 좋아졌지만 기본 반복 오차는 여전히 Y 약 2.74틱, yaw 약 2.07틱이었다.
- 잡음 가중에 혼합 반복의 명령값 자체를 회귀 정답으로 넣지 않았다. 다만 이미 수집된 다른 혼합 위치의 반복 자료를 가중 보정에 사용하는 조건이므로 실제 신규 세션의 무보정 성능보다 유리할 수 있다.

## 알고리즘 개발에 주는 결론

현재는 ‘평균 제거만 하면 yaw를 정밀하게 분리할 수 있다’고 전제할 근거가 부족하다. 특징점 추출은 유지하고, 다음 알고리즘 단계에서는 올바른 yaw 제약을 가진 순방향 모델이나 세션 차이를 다루는 보정으로 **구별 패턴의 재현성**부터 확보할 필요가 있다. 축 접근 제어, 전환 기준, 5~10회 수렴 시험은 이번 작업에 포함하지 않았다.

## 산출물과 재현

[시각 보고서](report.html), summary.png, 01_motion_patterns.png, 02_repeat_jitter.png, 03_tick_snr.png, 04_mixed_predictions.png.

- summary.csv: 모든 보정 범위/특징/가중 조건의 반복 σ와 혼합 오차.
- point_patterns.csv: 630점별 한 틱 패턴과 다른 축을 제거한 패턴.
- repeat_projection.csv: 57장의 좌표 평균 제거 후 두 축 추정 방향으로 투영한 흔들림.
- mixed_predictions.csv, mixed_by_range.csv, repeat_sigma_by_range.csv: 혼합 예측 및 범위별 실제 오차/반복 흔들림.
- axis_coordinate_holdout.csv, axis_holdout_predictions.csv: 단독축의 위치 보류 검증.
- pattern_split_stability.csv: 축 데이터 두 집합에서 얻은 구별 패턴 재현성.
- reference_checks.csv: 신규 기준 영상 예측. center1의 0은 기준 정의로, 정확도 검증값이 아니다.
- models_and_diagnostics.json, manifest.json: 패턴, 추정 계수, 수치 검산, 입력/코드 해시, 가정.

저장소 루트에서 numpy,pandas,matplotlib,Pillow를 사용할 수 있는 Python으로 `tool/offaxis/yaw_identifiability.py`, `tool/offaxis/yaw_identifiability_report.py` 실행. 원본/기존 계측은 변경하지 않았고, 장비 이동·커밋·push는 하지 않았다.
'''
    (OUT/'README.md').write_text(md,encoding='utf-8')
    def image(name,caption):
        encoded=base64.b64encode((OUT/name).read_bytes()).decode()
        return f'\n<figure><img src="data:image/png;base64,{encoded}"><figcaption>{caption}</figcaption></figure>\n'
    table=t.rename(columns={'method':'조건','Y_repeat_sigma_all':'Y 반복 σ · 전체','yaw_repeat_sigma_all':'yaw 반복 σ · 전체',
        'Y_repeat_sigma_in_range':'Y 반복 σ · 범위 안','yaw_repeat_sigma_in_range':'yaw 반복 σ · 범위 안',
        'mixed_in_range_images':'범위 안 혼합 장수','Y_mixed_rmse':'Y 실제 RMSE','yaw_mixed_rmse':'yaw 실제 RMSE'}).to_html(index=False,float_format=lambda x:f'{x:.3f}',border=0)
    page='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Y와 yaw 식별 가능성</title><style>body{max-width:1200px;margin:auto;padding:30px;background:#f4f7fa;color:#203f52;font-family:system-ui,'Malgun Gothic',sans-serif}h1{font-size:32px}h2{margin-top:36px}p,li{line-height:1.85}.lead{padding:20px;background:#fff0dd;border-radius:12px;font-weight:650}figure{margin:24px 0;background:white;padding:15px;border-radius:12px}img{max-width:100%}figcaption{line-height:1.7;color:#5a6c78;padding-top:10px}table{background:white;border-collapse:collapse;white-space:nowrap}th,td{padding:10px;border-bottom:1px solid #ddd;text-align:right}td:first-child,th:first-child{text-align:left}.scroll{overflow-x:auto}.note{background:#e5eef6;padding:18px;border-radius:10px}a{color:#126a9b}</style><body>
<h1>현재 특징점으로 Y와 yaw를 나눌 수 있나?</h1>
<p class="lead">한 틱 차이를 안정적으로 구분하기에 충분하다고 확인되지 않았습니다. 기본 방법에서는 반복 흔들림이 구별 신호보다 컸고, 잡음 가중으로 반복성을 높인 경우에도 실제 예측 정확도와 패턴 재현성을 확보하지 못했습니다.</p>
SUMMARY
<h2>방법</h2><ol><li>단독축 280장으로 Y 한 틱과 yaw 한 틱이 만드는 점별 이동 패턴을 구했습니다. 틱은 Y 0.05mm, yaw 0.01°입니다.</li><li>혼합 19좌표 × 반복 3장으로 관측 흔들림을 측정했습니다. 점 전체의 공통 흔들림도 포함했습니다.</li><li>두 패턴의 공통 부분을 제거하고 남는 구별 성분과, 같은 방향의 흔들림을 비교했습니다.</li><li>잡음 가중은 평가할 좌표의 반복 3장을 모두 제외하고 보정했습니다. 다른 위치에서 패턴이 재현되는지와 실제 좌표 예측도 별도로 검증했습니다.</li></ol>
<h2>한 틱 신호와 반복 흔들림</h2><p>기본 결과에서 Y 한 틱의 SNR은 0.415, yaw는 0.447입니다. 1보다 작다는 것은 한 틱의 구별 성분이 그 방향 반복 표준편차보다 작다는 뜻입니다. 두 패턴 사이 각도 0.144°는 실제 yaw 오차가 아니라, 1,260차원 이동 패턴의 방향 차이입니다.</p>
PATTERNS
JITTER
<h2>잡음 가중이 좋아 보여도 확인해야 할 것</h2><p>아래 표는 L630 전체 결과이며 단위는 틱입니다. 전체 범위 보정의 실제 예측은 yaw≤1°인 45장, 중심 범위 보정은 Y≤0.5mm·yaw≤0.2°인 27장을 사용했습니다. 조건마다 평가 범위가 달라 행 사이를 직접적인 개선 배수로 읽으면 안 됩니다.</p><div class="scroll">TABLE</div>
<p class="note">중심 보정 + 잡음 가중에서 반복 σ는 약 0.31틱이지만, 같은 보정 범위 안의 실제 혼합 RMSE는 Y 4.20틱·yaw 3.42틱입니다. 반복성이 좋아진 것만으로 좌표를 정확히 구분했다고 판정하지 않았습니다.</p>
SNR
<h2>구별 패턴 자체도 안정적인가?</h2><p>서로 다른 축 위치 절반씩으로 구별 패턴을 다시 추정하면, yaw 패턴의 cosine은 전체 범위 0.201, 중심 범위 −0.065였습니다. 같은 패턴이 재현됐다면 1에 가까워야 합니다. 지금은 보정 오차·비선형성·세션 변동을 함께 의심해야 합니다.</p>
PRED
<h2>이번 결론의 범위</h2><p>현재 특징과 실측 선형 보정의 조합이 한 틱 분리를 입증하지 못했다는 결론입니다. 모든 알고리즘이 불가능하다는 뜻은 아닙니다. 다른 축이 고정됐다는 조건에서는 큰 공통 이동을 쓸 수 있어 단독축의 좋은 정확도와 모순되지 않습니다. 특징점 추출 변경과 축 접근 제어는 이번에 수행하지 않았습니다.</p>
<p>새 이미지 계측 없이 기존 고정 L630 데이터를 재사용했습니다. 원본과 기존 계측은 보존했습니다. 반복 표준편차는 센서 잡음뿐 아니라 기구 재현오차·조명·정합 변화를 포함할 수 있습니다. 파일명 명령 좌표를 기준으로 했으며 독립 실측 정답은 아닙니다.</p>
<p><a href="README.md">상세 방법과 제한</a> · <a href="summary.csv">전체 조건 수치</a> · <a href="pattern_split_stability.csv">패턴 재현성</a> · <a href="mixed_by_range.csv">범위별 혼합 검증</a> · <a href="manifest.json">재현 정보</a></p></body></html>'''
    page=page.replace('\nSUMMARY\n',image('summary.png','단독축의 정밀도와 혼합 상태의 분리 성능을 구분해서 봤습니다.'))
    page=page.replace('\nPATTERNS\n',image('01_motion_patterns.png','위: 각 축 한 틱의 전체 이동. 아래: 상대 축을 제거한 작은 차이. 아래 화살표는 15,000배 확대이므로 실제 차이가 크게 보이지 않도록 주의해야 합니다.'))
    page=page.replace('\nJITTER\n',image('02_repeat_jitter.png','같은 좌표 반복의 흔들림을 고정된 두 축 추정기로 환산했습니다. 초록 구간은 ±1틱입니다.'))
    page=page.replace('\nSNR\n',image('03_tick_snr.png','반복 흔들림만을 본 지표입니다. 오른쪽 막대가 높아도 실제 혼합 예측과 패턴 재현성은 별도 검증이 필요합니다.'))
    page=page.replace('\nPRED\n',image('04_mixed_predictions.png','단독축에서 배운 패턴을 신규 세션 혼합 사진에 적용했습니다. 새 기준 center1 외에 혼합 정답으로 절편을 맞추지 않았습니다.'))
    page=page.replace('TABLE',table)
    (OUT/'report.html').write_text(page,encoding='utf-8')
    (OUT/'report_manifest.json').write_text(json.dumps(dict(source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        html_sha256=hashlib.sha256((OUT/'report.html').read_bytes()).hexdigest()),indent=2),encoding='utf-8')
    print(t.to_string(index=False));print('Written',OUT/'report.html')


if __name__=='__main__':main()
