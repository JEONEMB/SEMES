"""Conclusion-first report: sequence calibration benefit versus gate failures."""
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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=ROOT/'docs/offaxis-eda/yaw-separation/sequence-experts'


def main():
    s=pd.read_csv(OUT/'validation_summary.csv')
    inventory=pd.read_csv(OUT/'sequence_inventory.csv')
    p=s[s.gap_seconds.eq(120)&s.split.eq('interleaved')&s.feature.eq('L630_full')&s.family.eq('independent')&s.subset.eq('all_supported')].set_index('model')
    six=s[s.gap_seconds.eq(120)&s.split.eq('interleaved')&s.feature.eq('three_QR_means')&s.family.eq('independent')&s.subset.eq('all_supported')].set_index('model')
    names={'global_inverse_6ch':'전체 통합 회귀','global_parameters_known_sequence':'통합 계수 + 시퀀스 조건',
        'known_sequence':'시퀀스별 모델 + 시퀀스 제공','image_hard_gate':'시퀀스별 모델 + 이미지 선택',
        'image_soft_blend':'시퀀스별 모델 + 부드러운 혼합'}
    main6=six.loc[list(names),['images','Y_rmse_ticks','yaw_rmse_ticks','both_within_tick']].copy()
    main6.index=[names[i] for i in main6.index]
    main6.both_within_tick*=100
    main6.columns=['평가 장수','Y RMSE · 틱','yaw RMSE · 틱','두 축 ±1틱 · %']
    table=main6.to_html(float_format=lambda x:f'{x:.3f}',border=0)
    seq=inventory[inventory.gap_seconds.eq(120)]
    ranks=seq.pose_rank.value_counts().to_dict()
    # Split selection diagnostics: inspect supported test images only.
    pred=pd.read_csv(OUT/'heldout_predictions.csv')
    bad=pred[pred.gap_seconds.eq(120)&pred.split.eq('interleaved')&pred.feature.eq('L630_full')&pred.family.eq('independent')&pred.model.eq('image_hard_gate')&pred.supported].copy()
    bad['max_error_ticks']=np.maximum(abs(bad.Y_pred_ticks-bad.Y_true_ticks),abs(bad.yaw_pred_ticks-bad.yaw_true_ticks))
    bad.sort_values('max_error_ticks',ascending=False).to_csv(OUT/'primary_gate_errors.csv',index=False)
    sensitivity=s[s.family.eq('independent')&s.feature.eq('L630_full')&s.subset.eq('all_supported')&s.model.isin(['known_sequence','image_hard_gate'])]
    sensitivity.to_csv(OUT/'primary_sensitivity.csv',index=False)
    fig,ax=plt.subplots(figsize=(9,4),layout='constrained')
    block=s[s.gap_seconds.eq(120)&s.split.eq('middle_block')&s.feature.eq('L630_full')&s.family.eq('independent')&s.subset.eq('all_supported')].set_index('model')
    axes=['known_sequence','image_hard_gate'];x=np.arange(2);w=.32
    ax.bar(x-w/2,p.loc[axes,'both_within_tick']*100,w,label='Interleaved coordinate holdout')
    ax.bar(x+w/2,block.loc[axes,'both_within_tick']*100,w,label='Middle coordinate block holdout')
    ax.set_xticks(x,['Known sequence','Image-only model selection']);ax.set_ylim(0,110)
    ax.set_ylabel('Both axes within one tick (%)');ax.legend(fontsize=9);ax.grid(axis='y',alpha=.2)
    ax.set_title('Conditional sequence accuracy remains high; image selection is less stable')
    fig.savefig(OUT/'04_block_validation.png',dpi=170);plt.close(fig)
    card=Image.new('RGB',(1450,1050),'#f4f7fa');draw=ImageDraw.Draw(card)
    font=lambda n:ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',n)
    bold=lambda n:ImageFont.truetype('C:/Windows/Fonts/malgunbd.ttf',n)
    draw.text((55,40),'시퀀스별 모델, 정확도가 좋아질까?',font=bold(40),fill='#193d52')
    draw.rounded_rectangle((45,115,1405,230),radius=12,fill='#e0f1e8')
    draw.text((65,140),'시퀀스 정보를 사용할 수 있다면, 유효한 방향입니다.',font=bold(32),fill='#226244')
    draw.text((65,187),'지금 남은 핵심은 이미지에 맞는 모델을 틀리지 않게 고르는 것입니다.',font=font(25),fill='#376b53')
    draw.text((65,280),'동일한 보류 98장 / L630 전체 특징 / 두 축 모두 ±1틱',font=bold(27),fill='#284b62')
    for i,(label,value,color) in enumerate([
        ('정확한 시퀀스를 알려준 경우',100*p.loc['known_sequence','both_within_tick'],'#287b53'),
        ('이미지만으로 시퀀스를 고른 경우',100*p.loc['image_hard_gate','both_within_tick'],'#b17521')]):
        y=360+i*115;draw.rounded_rectangle((55,y-10,1390,y+80),radius=10,fill='white')
        draw.text((75,y+5),label,font=font(28),fill='#284b62')
        draw.text((1120,y),f'{value:.1f}%',font=bold(38),fill=color)
    draw.text((65,630),'함께 확인한 대조 결과',font=bold(29),fill='#284b62')
    draw.text((65,685),'시퀀스마다 계수를 바꾸지 않아도, 고정 축 조건을 알려주면 ±1틱 100%',font=font(25),fill='#496272')
    draw.text((65,732),'따라서 개선에는 시퀀스 보정뿐 아니라 “어느 축이 고정됐는가”도 기여',font=font(25),fill='#496272')
    draw.rounded_rectangle((50,825,1400,1005),radius=12,fill='#fff0dd')
    draw.text((70,845),'이 98장은 단독축 92장 + Y가 고정된 혼합 6장입니다.',font=bold(28),fill='#93581e')
    draw.text((70,895),'자료 부족 시퀀스의 보류 12장은 비교에서 제외했습니다.',font=font(25),fill='#785f46')
    draw.text((70,945),'임의의 혼합 자세나 새 촬영 세션까지 해결됐다는 결과는 아닙니다.',font=font(25),fill='#785f46')
    card.save(OUT/'summary.png')
    md=f'''# 파일 수정 시각 기준 시퀀스 모델 검증

2026-10-26. **시퀀스를 알려주었을 때의 조건부 정확도는 크게 좋았다. 이미지만으로 모델을 고르면 일부 큰 오차가 남았다. 다만 120초 분할의 시퀀스는 전부 1차원 스윕 또는 정지점이어서, 이 결과를 임의의 2축 혼합 자세 분리 성공으로 읽으면 안 된다.**

## 사용한 방법

사용자가 촬영 시퀀스 판단에 파일 수정 시각을 사용했다고 확인했다. 기존 frames.csv의 mtime_ns가 현재 원본의 수정 시각과 1μs 이내 일치하는지 341장 확인했다. 특징점 추출은 바꾸지 않고 기존 L형 630점 dx/dy를 재사용했다. 기준 영상은 경계 탐지에는 포함하고 모델 학습/평가에서는 제외했다.

연속 파일의 시간 차가 60/120/300초를 넘으면 새 구간으로 나누는 세 분할을 사전에 지정했다. 오류가 가장 작은 임계값을 최종 기준으로 골라 보고하지 않았다. 대표 설명은 120초다. 이것은 시험용 경계 규칙이며 모든 긴 간격이 실제 기구 재설정이었다는 확정은 아니다.

각 구간에서 명령 좌표 변화의 rank를 구했다. rank=0은 한 위치 반복, rank=1은 하나의 축/직선만 변화, rank=2는 두 축을 독립적으로 변화시킨 자료다. 120초 분할은 활성 시퀀스 {len(seq)}개, rank 0이 {ranks.get(0,0)}개, rank 1이 {ranks.get(1,0)}개, rank 2는 {ranks.get(2,0)}개다.

세 가지 순방향 모델을 비교했다: (1) 공통 기울기+구간별 중심, (2) 구간별 기울기+중심, (3) 구간별 기울기를 공통 기울기 쪽으로 25% 축소한 모델. 구간에 없는 운동 방향을 새로 학습했다고 가정하지 않고 실제 학습 좌표가 놓인 부분공간만 사용했다. 모델 생성에는 최소 3개의 서로 다른 학습 좌표를 요구했다.

## 검증 분할과 누출 방지

같은 명령 좌표의 반복은 전부 같은 쪽에 두었다. 교차 배치에서는 각 category의 좌표를 정렬해 매 세 번째 좌표를 보류했다. 추가로 정렬 좌표 중앙 1/3을 연속 보류했다. 좌표 분할은 시간 임계값을 보기 전에 동일하게 유지해서 임계값별 테스트를 바꾸지 않았다. 일부 시퀀스는 보류 후 학습 좌표가 부족해 모델을 만들 수 없었다.

대표 교차 배치에서 보류는 110장, 사용할 수 있는 실제 시퀀스 모델이 있는 공통 평가 자료는 98장이다. 단독축 92장+Y 고정 혼합 6장이고, 나머지 혼합 12장은 지원할 모델이 없어 공통 비교에서 제외했다. 두 방식의 RMSE를 비교할 때 정확히 같은 98장을 썼다. 중앙 블록 보류는 112장 중 97장을 공통 비교했다.

## 정확한 시퀀스를 알 때와 이미지로 고를 때

metadata 조건은 수정 시각으로 얻은 실제 구간 ID를 제공한다. 이 ID를 알면 그 구간에서 어느 축이 고정돼 있는지도 학습 좌표에서 알 수 있다. 시퀀스별 독립 모델로 98장 모두 두 축 ±1틱이었다. 주 L630 결과의 RMSE는 Y {p.loc['known_sequence','Y_rmse_ticks']:.6f}틱, yaw {p.loc['known_sequence','yaw_rmse_ticks']:.6f}틱이다.

이미지 gate는 정답 좌표·파일명·수정 시각을 입력으로 쓰지 않는다. 각 모델에 특징 벡터를 투영한 뒤 영상 특징 재구성 오차가 가장 작은 모델을 선택했다. 보정한 좌표 구간 안의 선분/영역에 한정하는 방식과 외삽을 허용하는 방식을 모두 저장했다. 선분 제한 gate는 98장 중 두 축 ±1틱 비율 {100*p.loc['image_hard_gate','both_within_tick']:.2f}%였으나, 잘못 고른 경우 큰 오차로 RMSE가 Y {p.loc['image_hard_gate','Y_rmse_ticks']:.3f}틱, yaw {p.loc['image_hard_gate','yaw_rmse_ticks']:.3f}틱이었다.

soft blend는 학습 영상 재구성 잔차에서 고정한 온도로 각 후보에 가중을 주어 좌표를 평균했다. 이 가중은 보정된 확률이 아니다. 이번 단순 soft blend는 hard gate보다 좋지 않았다. 서로 다른 시퀀스의 예측을 평균하면 물리적으로 맞지 않는 중간 좌표가 될 수 있다.

## 중요한 대조: 좋아진 이유를 분리

같은 전역 계수만 유지하고, 시퀀스의 고정축/좌표 직선 조건만 알려준 대조도 98장 모두 ±1틱이었다. 주 L630 대조 RMSE는 Y {p.loc['global_parameters_known_sequence','Y_rmse_ticks']:.3f}틱, yaw {p.loc['global_parameters_known_sequence','yaw_rmse_ticks']:.3f}틱이다. 시퀀스별 계수는 추가로 Y 오차를 줄였지만, 큰 개선을 전부 ‘여러 직선을 분리했기 때문’이라고 해석할 수 없다. 원래 2개의 미지수였던 것을 1개로 줄여준 정보의 효과가 매우 크다.

공통 기울기+구간별 중심에서도 ±1틱 100%였다. 독립 기울기까지 허용하는 추가 효과는 제한적이며 보류 영상 재구성 잔차가 더 좋아진다고 일관되게 확인되지는 않았다. 처음부터 복잡한 모델 접합을 만들 근거는 없다.

실용적인 대조를 위해 동일한 QR별 평균 dx/dy 6채널을 사용하는 통합 역회귀도 비교했다. 이는 단순한 순방향 2열 모델보다 강한 대조군이다. 아래 모든 결과는 같은 보류 98장이다.

{main6.to_markdown() if False else main6.to_csv()}

6채널 이미지 gate는 두 축 ±1틱 비율 87.76%로 대부분을 잘 맞혔지만, 일부 큰 오차 때문에 RMSE는 통합 역회귀보다 나빴다. ‘대부분 정확함’과 ‘큰 오차까지 줄어듦’을 구분한다.

## 대표 결과의 한계

- 120초 모델에서 rank=2 시퀀스는 없다. 혼합 평가 6장도 Y가 고정된 두 위치의 반복이므로, 알려진 시퀀스에서 Y 오차 0은 Y를 영상으로 추정한 성능이 아니다.
- 새로운 촬영 세션을 통째로 보류한 일반화 실험은 아니다. 미래의 파일 수정 시각만 보고 예전 구간 ID를 알 수는 없다. 실시간으로 run ID를 관리하거나 새 기준 영상으로 세션 보정을 하는 방법이 필요하다.
- 300초로 크게 묶으면 독립 2축을 포함하는 구간도 생기지만, 대표 교차 배치에서 6채널 알려진 시퀀스 모델의 혼합 12장 RMSE는 Y 약 11.0틱, yaw 약 9.72틱이었다. 단순히 큰 시퀀스별 2축 회귀로 나누는 것으로 임의 혼합 문제가 해결되지는 않았다.
- 60초로 더 잘게 나누면 알려진 구간 내 성능은 좋지만, 작은 구간은 학습 자료가 부족하고 중앙 블록 보류에서 이미지 gate가 불안정해졌다. 경계를 정확도에 맞춰 고르면 안 된다.
- 기존 identifiability 실험도 파일명 기반 6그룹 절편을 따로 사용했다. 그 실험의 모든 실패가 ‘세션을 완전히 무시했기 때문’은 아니다. 이번 차이는 시간 경계, 구간별 기울기뿐 아니라 시퀀스의 1차원 운동 조건을 추론에 직접 사용한 것이다.
- 명령값은 파일명 기준이며 독립 실측값이 아니다. 새로운 카메라 yaw 물리모델이나 장비 제어를 개발/실행한 결과가 아니다.

## 알고리즘 방향에 대한 판단

사용자 제안은 타당하며, 시퀀스 조건부 모델은 보존할 가치가 있다. 다음 개발의 핵심은 모델을 많이 만드는 것보다 **새 이미지/현재 run에 어떤 모델을 적용할지와 그 선택이 불확실할 때 어떻게 처리할지**다. 영상 한 장만의 단순 최소 잔차 선택이나 무조건적인 좌표 평균은 아직 충분하지 않다. 기존 run 정보와 알려진 명령 이력을 사용할 수 있다면 별도의 조건으로 평가할 수 있다.

이 구조는 [Jacobs et al. (1991), Adaptive Mixtures of Local Experts](https://www.cs.toronto.edu/~hinton/absps/jjnh91.pdf)의 전문가+gate 관점과 연결된다. 이번 구현은 논문의 신경망 학습 알고리즘을 재현한 것이 아니라 선형 시퀀스 모델과 단순 잔차 gate로 가설을 검증한 것이다.

## 산출물

[시각 보고서](report.html), sequence_assignments.csv, sequence_inventory.csv, validation_summary.csv, heldout_predictions.csv, linearity_validation.csv, image_gate_diagnostics.csv, primary_gate_errors.csv, primary_sensitivity.csv, primary_experts.json, manifest.json.

재현: `tool/offaxis/sequence_experts.py`, 이어서 `tool/offaxis/sequence_experts_report.py`. 원본·기존 계측은 변경하지 않았고 커밋/push는 하지 않았다.
'''
    (OUT/'README.md').write_text(md,encoding='utf-8')
    def embed(name,caption):
        b64=base64.b64encode((OUT/name).read_bytes()).decode()
        return f'<figure><img src="data:image/png;base64,{b64}"><figcaption>{caption}</figcaption></figure>'
    page='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>시퀀스 모델 검증</title><style>body{max-width:1200px;margin:auto;padding:30px;background:#f4f7fa;color:#203f52;font-family:system-ui,'Malgun Gothic',sans-serif}h1{font-size:32px}h2{margin-top:36px}p,li{line-height:1.85}.lead{padding:20px;background:#e0f1e8;border-radius:12px;font-weight:650}figure{margin:24px 0;background:white;padding:15px;border-radius:12px}img{max-width:100%}figcaption{line-height:1.7;color:#5a6c78;padding-top:10px}table{background:white;border-collapse:collapse;white-space:nowrap}th,td{padding:10px;border-bottom:1px solid #ddd;text-align:right}td:first-child,th:first-child{text-align:left}.scroll{overflow-x:auto}.note{background:#fff0dd;padding:18px;border-radius:10px}a{color:#126a9b}</style><body>
<h1>시퀀스 모델로 정확도가 좋아지는가?</h1><p class="lead">시퀀스를 알고 있을 때의 조건부 정확도는 좋았습니다. 이미지만으로 모델을 선택할 때는 일부 큰 오차가 남았습니다. 모델을 나누는 아이디어는 유효하지만, 임의의 혼합 자세까지 해결된 결과는 아닙니다.</p>
@@SUMMARY@@
<h2>확인한 시간 구간과 직선 관계</h2><p>사용자가 파일 수정 시각을 촬영 시퀀스 기준으로 확인했습니다. 60/120/300초 간격을 시험 규칙으로 미리 정했고, 오류가 작아지는 경계를 골라 결과로 제시하지 않았습니다. 대표 설명은 120초입니다. 이 경계가 실제 기구 재설정과 항상 일치한다는 뜻은 아닙니다.</p>
@@LINES@@
@@TIMES@@
<h2>같은 사진에서 공정하게 비교</h2><p>같은 좌표의 반복 3장은 함께 보류했습니다. 대표 보류 110장 중 학습 자료가 충분한 시퀀스 모델이 있는 98장만 공통 비교했고, 자료가 부족한 혼합 12장은 제외했습니다. 아래 표는 같은 QR별 평균 dx/dy 6채널을 쓰므로 특징 종류가 달라 생기는 차이를 줄였습니다.</p><div class="scroll">@@TABLE@@</div>
<p class="note">98장은 단독축 92장과 Y가 고정된 혼합 6장입니다. 시퀀스 정보를 알면 ‘고정된 축의 값’도 알게 됩니다. 이 정보를 이용한 성능을 임의의 혼합 자세에서 두 좌표를 동시에 맞힌 성능으로 읽으면 안 됩니다.</p>
<h2>모델 분리 효과와 조건 정보의 효과</h2><p>통합 계수를 그대로 두고 시퀀스의 고정축 조건만 알려준 대조도 두 축 ±1틱 100%였습니다. 구간별 보정은 추가로 오차를 줄였지만, 큰 개선을 전부 세션별 기울기의 효과로 해석할 수는 없습니다. 공통 기울기+시퀀스별 중심도 좋은 결과를 보였습니다.</p>
@@GATING@@
@@BLOCK@@
<h2>이미지 선택이 남은 병목</h2><p>이미지 gate는 정답·파일명·수정 시각 없이 각 모델의 영상 재구성 잔차만 비교했습니다. 대부분을 잘 고르는 경우에도 일부 오선택이 큰 좌표 오차를 만들었습니다. 단순 soft blend는 서로 다른 모델의 좌표를 평균하면서 더 나빠졌습니다.</p>
<p>시퀀스별 좋은 모델을 보관하고, 현재 run 정보·선택의 불확실성·새 기준 영상으로의 세션 보정을 다루는 방향은 검토할 가치가 있습니다. 새 세션 일반화나 장비 동작은 이번에 시험하지 않았습니다.</p>
<h2>놓치면 안 되는 제한</h2><ul><li>120초 분할의 활성 시퀀스에는 독립 2축을 함께 변화시킨 구간이 없습니다.</li><li>300초로 크게 묶은 혼합 2축 구간에서는 알려진 시퀀스 모델도 큰 오차가 남았습니다.</li><li>수정 시각으로 구분된 기존 run 안에서의 검증입니다. 미래 시각만으로 예전 모델 ID를 알 수는 없습니다.</li><li>이전 실험도 일부 그룹별 절편을 사용했습니다. 이번에는 시퀀스의 1차원 조건까지 추론에 사용했습니다.</li></ul>
<p><a href="README.md">상세 방법·해석</a> · <a href="sequence_assignments.csv">사진별 시퀀스</a> · <a href="validation_summary.csv">전체 조건 결과</a> · <a href="primary_gate_errors.csv">모델 선택 오류</a> · <a href="manifest.json">재현 정보</a></p>
<p>참고: <a href="https://www.cs.toronto.edu/~hinton/absps/jjnh91.pdf">Jacobs et al. (1991), Adaptive Mixtures of Local Experts</a>. 논문의 신경망 학습을 재현한 것이 아니라 선형 모델+단순 gate의 검증입니다.</p></body></html>'''
    values={'@@SUMMARY@@':embed('summary.png','시퀀스 조건부 정확도와 이미지 선택 정확도를 분리했습니다.'),
        '@@LINES@@':embed('01_sequence_lines.png','축별로 공통 기울기를 제거해 시퀀스의 절편·기울기 차이를 확대했습니다. 원시 이동량 자체의 그래프는 아닙니다.'),
        '@@TIMES@@':embed('02_time_sequences.png','한국 시간 기준 파일 수정 시각. 날짜 사이의 큰 공백도 그대로 표시했습니다.'),
        '@@TABLE@@':table,'@@GATING@@':embed('03_gating_comparison.png','L630 전체 특징에서의 별도 비교입니다. 앞의 표는 6채널이고 이 그림은 1,260채널입니다.'),
        '@@BLOCK@@':embed('04_block_validation.png','시퀀스 조건부 정확도는 유지됐지만, 좌표를 연속 구간으로 보류하면 이미지 gate가 더 불안정해졌습니다.')}
    for key,value in values.items():page=page.replace(key,value)
    (OUT/'report.html').write_text(page,encoding='utf-8')
    (OUT/'report_manifest.json').write_text(json.dumps(dict(source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        html_sha256=hashlib.sha256((OUT/'report.html').read_bytes()).hexdigest()),indent=2),encoding='utf-8')
    print('Wrote',OUT/'report.html');print(main6.to_string())


if __name__=='__main__':main()
