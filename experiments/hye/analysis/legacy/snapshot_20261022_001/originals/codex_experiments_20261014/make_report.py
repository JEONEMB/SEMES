"""Generate an evidence-linked Korean report and offline HTML summary."""
from pathlib import Path
import json,base64,html
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
FIG=HERE/'figures';FIG.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'Malgun Gothic','axes.unicode_minus':False,'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'#fbfcfe','axes.facecolor':'#fbfcfe'})

def readj(p):return json.loads((HERE/p).read_text(encoding='utf8'))
def table(headers,rows):
    return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+'\n'.join('| '+' | '.join(str(v) for v in r)+' |' for r in rows)
def percent(x):return f'{100*x:.2f}%'
def fnum(x):return f'{x:.6f}'

def main():
    stress=pd.read_csv(HERE/'calibrated_runs/stress_metrics.csv')
    named=stress.set_index('cv');repeat=stress[stress.cv.str.startswith('repeated')]
    gen=readj('modeling/unified_isolated_stratified_metrics.json');new=readj('modeling/unified_isolated_session_out_metrics.json')
    zero=pd.read_csv(HERE/'zero_gn17_spread14/session_zero_estimates.csv')
    za=pd.read_csv(HERE/'zero_gn17_spread14/zero_ablation.csv')
    repeat_pred=pd.read_csv(HERE/'calibrated_runs/stress_predictions.csv')
    wrong=repeat_pred[(repeat_pred.cv.str.startswith('repeated'))&(repeat_pred.axis!=repeat_pred.pred_axis)]
    wrong_count=wrong.groupby('file').agg(failures=('file','size'),true_axis=('axis','first'),true_value=('value','first')).reset_index().sort_values('failures',ascending=False)
    wrong_count.to_csv(HERE/'calibrated_runs/repeated_hard_cases.csv',index=False)
    summary_rows=[]
    for cv,label in [('interleaved','같은 시행 / 짝홀'),('stratified5','같은 시행 / 층화 5분할'),('blocked_within_session5','같은 시행 / 연속구간 5분할')]:
        r=named.loc[cv];summary_rows.append([label,percent(r.Y_axis_sign_accuracy),percent(r.T_axis_sign_accuracy),fnum(r.vector_Y_max),fnum(r.vector_T_max),'현재 데이터에서 통과'])
    summary_rows.append(['동일 모델 / 무작위 5분할 20회',f'{100*repeat.Y_axis_sign_accuracy.min():.2f}–100%',f'{100*repeat.T_axis_sign_accuracy.min():.2f}–100%',fnum(repeat.vector_Y_max.max()),fnum(repeat.vector_T_max.max()),'일부 분할 실패'])
    for r,label in [(gen,'추출기·모델 선택 포함 중첩 3분할'),(new,'새 시행 모사 / 시행 전체 제외')]:
        summary_rows.append([label,percent(r['Y_joint_accuracy']),percent(r['T_joint_accuracy']),fnum(r['max_vector_error_Y']),fnum(r['max_vector_error_T']),'미달'])
    headers=['평가 조건','Y 축+부호','T 축+부호','최대 Y 오차','최대 T 오차','판정']
    pd.DataFrame(summary_rows,columns=headers).to_csv(HERE/'summary_metrics.csv',index=False,encoding='utf-8-sig')

    # Accuracy and actual two-axis output error, including wrong-axis outputs.
    fig,ax=plt.subplots(1,2,figsize=(14,5.5),layout='constrained')
    labs=['6개 시행 보정\n구간 5분할','반복분할\n각 축 최저 성적','중첩 검증\n추출·모델 선택','시행 전체 제외\n새 0점 모사']
    ys=[1,repeat.Y_axis_sign_accuracy.min(),gen['Y_joint_accuracy'],new['Y_joint_accuracy']]
    ts=[1,repeat.T_axis_sign_accuracy.min(),gen['T_joint_accuracy'],new['T_joint_accuracy']]
    x=np.arange(4)
    ax[0].bar(x-.18,np.array(ys)*100,.34,label='Y',color='#2065b8');ax[0].bar(x+.18,np.array(ts)*100,.34,label='T',color='#df842f')
    ax[0].set(xticks=x,xticklabels=labs,ylim=(80,102),ylabel='축과 부호를 함께 맞힌 비율 (%)',title='조건부 100%와 일반화 성능을 구분')
    ax[0].axhline(100,color='#767e8b',ls='--',lw=1);ax[0].legend()
    yerr=[named.loc['blocked_within_session5','vector_Y_max'],repeat.vector_Y_max.max(),gen['max_vector_error_Y'],new['max_vector_error_Y']]
    terr=[named.loc['blocked_within_session5','vector_T_max'],repeat.vector_T_max.max(),gen['max_vector_error_T'],new['max_vector_error_T']]
    ax[1].bar(x-.18,yerr,.34,label='Y 오차 (mm)',color='#2065b8');ax[1].bar(x+.18,terr,.34,label='T 오차 (T 단위)',color='#df842f')
    ax[1].axhline(.09,color='#b32932',ls='--',label='각 축 허용치 0.09');ax[1].set(xticks=x,xticklabels=labs,ylabel='오분류까지 포함한 각 축 최대오차',title='축을 틀린 사진도 제외하지 않음');ax[1].legend(fontsize=9)
    fig.savefig(FIG/'evaluation.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(1,2,figsize=(12,4.6),layout='constrained')
    for a,axis in zip(ax,['Y','T']):
        q=zero[zero.session.str.startswith(axis)]
        a.bar(q.session,q.center_relative_offset,color='#2065b8' if axis=='Y' else '#df842f')
        a.errorbar(q.session,q.center_relative_offset,yerr=1.96*q.intercept_se_px/abs(q.slope_px_per_unit),fmt='none',color='#293346',capsize=4)
        a.axhline(0,color='#67717f',lw=1);a.set(title=f'{axis}: 각 시행의 0점 절편 / 기울기',ylabel='센터 기준 등가 이동량'+(' (mm)' if axis=='Y' else ' (T 단위)'))
    fig.suptitle('6개 회귀 외삽 절편과 센터 비교 — 실제 0점 7장의 실측값이 아님',fontsize=13)
    fig.savefig(FIG/'zero_offsets.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,4.5),layout='constrained')
    ax.bar(np.arange(20),repeat.axis_sign_accuracy.to_numpy()*100,color=np.where(repeat.axis_sign_accuracy==1,'#20735d','#cc8231'))
    ax.set(ylim=(98.5,100.2),xticks=np.arange(20),xlabel='무작위 분할 seed 번호 (같은 280장을 재사용)',ylabel='축+부호 정확도 (%)',title='205점·6개 0점 후보 모델: 분할 변경에 따른 변동')
    fig.savefig(FIG/'repeat_stability.png',dpi=160);plt.close(fig)

    zrows=[[r.session,int(r.n),f'{r.slope_px_per_unit:.3f}',f'{r.intercept_px:+.4f}',f'{r.center_relative_offset:+.6f}'] for r in zero.itertuples()]
    zcomp=za[(za.cv=='interleaved')&~za.nuisance&~za.quadratic]
    znames={'raw_center':'센터 0 고정 / 원래 라벨','raw_free':'축별 절편 학습 / 원래 라벨','center_relative_session':'시행 절편으로 학습 라벨 보정','center_relative_drift':'시간·방향 모델로 학습 라벨 보정','inferred_session':'6개 절편 후보를 영상에서 선택','oracle_zero':'정답 시행의 절편을 제공 (진단용)'}
    zerorows=[[znames[r.config],percent(r.Y_axis_sign_accuracy),percent(r.T_axis_sign_accuracy),fnum(r.Y_matched_max),fnum(r.T_matched_max)] for r in zcomp.itertuples() if r.config in znames]
    s=named.loc['blocked_within_session5']; rp100=int((repeat.axis_sign_accuracy==1).sum())
    checks=readj('verification/checks.json')
    deployment=readj('modeling/deployed_isolated_metadata.json')
    body=f'''# 센터·현재 사진 두 장으로 Y/T 추정 — 독립 실험 결과

2026-10-14 · 원본 281장(센터 1, Y 80, T 200) · 모든 신규 결과는 이 폴더에 저장

**현재 6개 시행의 0점 패턴을 학습한 QR 3개 모델에서 조건부 100% 결과를 얻었다. 다만 분할 변경 및 새 시행 평가까지 포함한 전체 목표는 아직 검증되지 않았다.**

QR별로 잘라낸 영상에서 205개 점을 추적하고, 판 변형 12계수와 6개 0점 후보를 비교하는 모델은 짝홀·층화 5분할·시행 내부 연속구간 5분할에서 모두 Y 80/80, T 200/200의 축과 부호를 맞혔다. 세 평가를 통틀어 최대오차는 Y {max(named.loc[c,'vector_Y_max'] for c in ['interleaved','stratified5','blocked_within_session5']):.6f} mm, T {max(named.loc[c,'vector_T_max'] for c in ['interleaved','stratified5','blocked_within_session5']):.6f}로 0.09 안이다.

그러나 같은 모델의 무작위 5분할 20회 중 전부 맞힌 것은 {rp100}회이고, 나머지에서는 1~2장을 틀렸다. 모델·추출기 선택 자체를 학습 안에 넣은 평가와 시행 전체 제외 평가에서도 100%가 유지되지 않는다. **“현재 특정 평가에서 100%”와 “새 현재 사진에도 100% 보장”은 다른 결론이다.** 새 독립 촬영은 없으므로 모든 수치는 기존 개발 데이터의 검증 성적이다.

## 1. 합격 기준과 정답

- 사용자 확정 허용오차: **Y와 T 각각 최대 절대오차 ≤ 0.09**. 최소 이동 간격 Y 0.05, T 0.01은 별도이며 더 엄격한 합격선으로 바꾸지 않았다.
- 판정: 축(Y/T)과 부호(+/−)를 함께 맞혀야 한다. Y·T 각각 100%가 목표다.
- 주 정답: 파일명의 명령값. 보정 라벨을 독립적으로 측정한 “실제 위치”로 취급하지 않았다.
- 오분류도 포함: 정답이 Y=0.30인데 T=0.05로 판단하면 출력 벡터는 (0,0.05), 오차는 (0.30,0.05)다. 축을 맞힌 사진만의 오차는 별도 보조 지표다.
- T는 파일명 T 단위를 유지했다. 도 단위·시계방향 여부는 확인되지 않았다. ±는 제공된 명령 라벨의 방향이다.
- 학습에는 알려진 명령 라벨을 사용한다. **실행 시 입력은 센터와 현재 사진 두 장뿐**이며 파일명·시각·정답 시행 ID를 전달하지 않는다. 같은 센터·카메라 구성을 전제로 하는 사전 보정 모델이 필요하다.
- 현재 사진은 한 축만 움직인 경우에 한정된다. Y/T 동시 이동, 새로운 카메라/거리, 실제 정지 상태의 반복 사진은 제공되지 않았다.

## 2. 핵심 성능표

{table(headers,summary_rows)}

첫 4행은 205점·6개 0점 후보 모델이다. 마지막 2행은 사진 추출 방법부터 선택하는 별도 일반 모델 절차다. 서로 다른 문제 설정을 한 모델의 성적으로 섞지 않았다. 마지막 2행의 중첩 평가에서는 안쪽 3분할만으로 추적법·모델·격자 사용 여부를 선택하고 바깥 사진을 한 번씩 평가했다. 시행 전체 제외는 T 한 부호만 학습에 남는 외삽 조건도 포함한다.

![평가 요약](figures/evaluation.png)

반복분할에서 틀린 사진과 빈도:

{table(['파일','20회 중 오분류 횟수','정답'],[[r.file,r.failures,f'{r.true_axis} {r.true_value:+.2f}'] for r in wrong_count.itertuples()])}

반복분할 20회는 독립 사진 5,600장이 아니다. 같은 280장을 다시 나눈 민감도 실험이다. 설정을 개발 중 선택했으므로 성공한 분할만 골라 최종 합격으로 보고하지 않았다.

![분할 민감도](figures/repeat_stability.png)

## 3. 단계별로 무엇을 실험했고 무엇을 배웠나

| 단계 | 실행한 비교 | 확인된 결과와 결정 |
| --- | --- | --- |
| 데이터·기존 결과 감사 | 원본 목록/시각/라벨, 기존 v1–v7 코드, 평가 분할 | 센터 1장+Y 80장+T 200장. 기존 보정본 세션 제외 검증에 일부 시험 세션이 들어갔다. 원시 라벨을 주 정답으로 재평가했다. |
| 특징점의 동일성 | 매 사진 재검출 vs 센터 고정 ID 추적 | 재검출 위치가 추적 위치 1px 안에 있는 비율 평균 59.71%, 모든 사진 공통 재검출점 70/928. 고정 ID로 추적하면 31/51px LK와 GN에서 전 사진 유효점 100%. |
| 점의 개수·공간 배치 | 928점, 넓은 배치 157/205점, 학습 안에서 top-k, QR 한 개/두 개/세 개 | 많이 뽑는 것만으로 최적이 되지 않는다. 점 개수와 위치 범위가 함께 바뀌므로 원인도 구분했다. 현재 조건부 최고 모델은 넓은 배치 205점이다. |
| 위치 정밀화 | LK15/31/51, blur LK, GN17/31, QR 전체 직접 정합 | 정밀 GN은 합성 이동 측정 편향을 줄였다. 하지만 합성 좌표 정확도의 개선이 축 분류 개선을 보장하지 않았다. |
| 합성 검증 | 소수점·±80px 이동, 회전·축척·전단, 밝기 변화 | GN17 평행이동 좌표 RMSE 0.00128–0.00351px. 직접 영상 정합은 별도 16조건에서 최대 0.000440px. 이는 합성 검증이며 로봇 실제 이동량 오차가 아니다. |
| 계산 모델 | 점별 직접 적합, affine 6, 다항 12/20, 호모그래피 8, QR별 affine 18, 선형/2차 축 곡선, 공분산/강건 가중 | 저차원 공간 변형과 공분산을 쓰는 방식이 점을 독립으로 취급하는 방식보다 유리했다. 일반 배포 모델은 LK51→QR별 affine 18→선형 축 곡선이다. |
| 세로 이동·회전 성분 | 유지 / dy 제거 / 회전 제거 / 둘 다 제거 | 무조건 제거할 근거가 없다. 설정과 검증 분할에 따라 순위가 바뀌므로 학습 안에서 선택했다. |
| 0점 | 보정 없음, 축별 절편, 보정 가정 라벨, 6개 절편 후보, 정답 시행 절편 진단 | 기존 6개 시행 내에서는 6개 후보가 유효했다. 새 0점에 자동으로 일반화된다는 증거는 없다. 아래 별도 표 참조. |
| 명령 격자 | 연속량 추정 vs Y .05/T .01 격자 비용 비교 | 양자화가 일부 조건을 개선하지만 100%를 안정적으로 만들지 못했다. 격자 가정은 정확한 명령 격자 데이터에서만 유효하다. |
| 최후 비교: 판 나사 2개 | 신규 QR-only와 같은 모델에 나사 39점을 추가 | LK31 한 조건의 3분할 오분류는 8→6으로 줄었지만 시행 제외는 17→18로 증가. GN17 선형은 4→7로 악화. 100% 조건은 없었다. |
| 검증 방식 | 짝홀, 층화, 연속구간, 반복 분할, 중첩 선택, 시행 제외 | 특정 분할 최고값과 일반화 성적의 차이가 컸다. 모든 사진별 예측과 실패를 보존했다. |

QR 위 모든 점을 넣어도 되는가: **동일 점의 대응이 확인되고 품질과 공간 배치를 관리하면 모두 활용할 수 있다. 서로 다른 점을 같은 순번으로 넣거나, 약한 점을 무조건 같은 독립 정보로 취급하면 안 된다.** 이 데이터에서는 대응 실패보다 공통 편향·공분산·0점 차이가 더 큰 문제가 됐다. 현재 모든 점이 유효했다는 사실은 가림/조명 변화/다른 주소에서도 모두 유효하다는 보장이 아니다.

기존 캐시를 같은 평가기로 비교한 진단 실험에서는 점별 독립 적합 57장 오분류 → P12 공간 요약 3장으로 줄었다. QR 1개는 15–27장, 양끝 2개와 3개는 3장; top10/25/50/100점은 35/18/9/6장, 전체232점은 3장이었다. **이 수치는 기존 캐시의 모델링 진단이며 엄격한 QR-only 최종 성과가 아니다.** 새 928점과 157/205점은 경계 침식·배치도 달라 단순 개수 효과로 해석할 수 없다.

![QR만 사용하는 205점 배치](measurement/gn17_spread14_features.png)

## 4. 실제 계산 함수

조건부 최고 모델의 실행 순서:

1. 센터에서 QR 3개 영역을 잘라, 저장된 205개 특징점 ID를 사용한다.
2. QR 템플릿으로 큰 이동을 찾고, LK로 초기 대응을 잡는다. 17px GN으로 밝기 gain/offset을 함께 맞추며 소수점 이동을 정밀화한다.
3. 점 좌표를 정규화하여 각 점의 이동을 다음 공간식으로 적합한다. x/y 이동에 각각 6개, 총 12개 계수가 사진의 특징 `f`다.

   `d(p) = a0 + a1*x + a2*y + a3*x² + a4*x*y + a5*y²`

4. 학습 데이터에서 각 축의 곡선과 6개 시행의 절편을 맞춘다. 현재 사진의 실제 시행 ID는 알려주지 않는다.

   `f = b_s + u_axis*v + q_axis*v² + residual`

5. 후보 s마다 이동량 v를 풀고, 잔차 공분산으로 표준화한 비용을 비교한다.

   `cost_s = min_v (f - b_s - u*v - q*v²)^T Σ^-1 (f - b_s - u*v - q*v²)`

   구현은 후보마다 가우스–뉴턴으로 연속량 해를 구하고, 해당 시행의 학습 부호(+/−)와 해의 부호가 다르면 후보에서 제외한다. 시험 사진의 정답 부호는 사용하지 않는다.

6. 비용이 가장 작은 후보의 축과 v의 부호/크기를 출력한다. 방향은 학습 명령 부호를 따른다. 명령값 기준 역이동량은 부호가 반대지만, 실제 센터 영상으로 복귀하는지와 장치 폐루프 제어는 검증하지 않았다.

모델 파일은 [calibrated_runs/model.json](calibrated_runs/model.json), 두 사진 함수는 [infer.py](infer.py)다. 일반 모델은 [modeling/deploy_model.pkl](modeling/deploy_model.pkl)에 별도 저장했다. 일반 모델은 6개 시행 절편에 의존하지 않지만 현재 작은 이동에서 오류가 더 많다.

## 5. 7개 0점 문제: 필요한 보정과 검증 가능한 범위

실제로 촬영된 0점 사진은 센터 1장뿐이다. 아래 여섯 값은 각 시행의 명령값–영상 이동 직선을 0으로 외삽해 얻은 절편이다. 따라서 “7개 실제 0점이 측정됐다”는 의미가 아니다.

{table(['시행','장수','기울기 px/단위','0점 절편 px','등가 이동량'],zrows)}

![0점 비교](figures/zero_offsets.png)

그림의 오차 막대는 회귀 절편 표준오차의 약 1.96배를 기울기로 나눈 근사 범위다. 기울기 불확실성은 포함하지 않았으며, 로봇의 실제 위치 오차 범위가 아니다.

205점, P12 선형, 세로·회전 성분 유지, 짝홀 평가에서 0점 처리만 비교:

{table(['처리','Y 축+부호','T 축+부호','축 맞힌 Y 최대오차','축 맞힌 T 최대오차'],zerorows)}

- **원시 명령값에 대한 0.09 이동량 허용치만 놓고 보면 라벨 보정이 필수라는 증거는 없다.** 보정 전에도 축을 맞힌 경우 오차가 허용치 안에 들어가는 설정이 많다.
- 센터에 대한 실제 기하 위치가 목표라면 0점 일관성이 필요하다. 그러나 영상으로 만든 보정 라벨을 실제 위치의 정답으로 삼아 보정 효과를 인증할 수는 없다.
- 기존 6개 시행에서는 라벨을 바꾸기보다 6개 절편을 가진 후보 모델이 축 구분에 도움이 됐다. 이 모델은 기존 시행의 0점 패턴을 기억하므로 새 시행의 0점 검증이 필요하다.
- 시행 하나를 뺐을 때 그 시행의 절편을 알 수 없다. 특히 T+를 전부 빼면 T+ 절편 후보 자체가 없으므로 이 6후보 모델의 시행 제외 결과는 구조적으로 실패한다. 일반화 평가에는 별도 전역 모델을 사용했다.
- `oracle_zero`는 시험 사진의 정답 시행 절편을 준 진단이다. 두 사진만의 성과로 집계하지 않았다.
- 명령값과 시간이 각 시행 안에서 강하게 상관되어 있어 백래시·시간 드리프트·실제 이동을 현재 자료만으로 분리해 확정하지 않았다.

## 6. 기존 결과에서 수정한 사항

1. 기존 보정 라벨은 외부 실측값이 아니다. 새 주 평가는 명령값 기준이다.
2. 기존 v7의 시행 제외 평가에서도 보정 모델에 시험 시행의 절반이 들어간다. 새 보정은 바깥 학습 인덱스만 사용한다.
3. 기존 QR-only 232점 중 정밀17px 템플릿 60개, 초기81px 템플릿 166개가 QR 라벨 밖으로 나간다. 일부 초기 창은 나사 주변에도 걸친다. 신규 최종205점과 LK 경로는 지원 영역을 분리했다.
4. “QR만으로 100%는 이론상 불가능”이라는 이전 결론은 증명되지 않았다. 특정 특징·잡음·방해 성분 제거 설정에서의 경험적 결과일 뿐이다.

상세 코드 위치와 공식 자료: [기존 분석 감사](audit/audit.md), [새 파이프라인 감사](audit/pipeline_audit.md). 관련 원리는 [OpenCV 호모그래피](https://docs.opencv.org/4.13.0/d9/dab/tutorial_homography.html)에 설명되어 있다. 절대 mm 환산에는 알려진 치수/카메라 보정 또는 라벨 기반 사전 학습이 필요하다.

## 7. 사진별 정확도와 오차를 출력하는 방법

정확도는 정답이 있는 검증 집합에서 계산하는 비율이다. 새 사진 한 장의 실제 오차는 독립 정답 없이 알 수 없다. 실행 결과에는 축·부호·이동량·두 축 후보 비용·정규화 비용차·유효점 수와 검증 결과 경로를 준다. **비용차는 정확도 확률이 아니다.** 신뢰 확률과 실제 오차는 정답 없이 임의로 채워 넣지 않고 `null`로 둔다. 독립 정답을 CLI에 주면 해당 사진의 오차를 계산한다.

PowerShell, 프로젝트 루트에서:

```powershell
.\\codex_experiments_20261014\\run.ps1 -Action predict -Center '센터사진.bmp' -Current '현재사진.bmp' -Mode calibrated-runs -Output '결과.json'
.\\codex_experiments_20261014\\run.ps1 -Action predict -Center '센터사진.bmp' -Current '현재사진.bmp' -Mode general
.\\codex_experiments_20261014\\run.ps1 -Action tests
```

센터 파일은 학습에 쓴 `center.bmp`와 내용이 같아야 한다. 복사·이름 변경은 가능하다. 다른 센터·카메라에는 재보정이 필요하다. `calibrated-runs`는 조건부 최고 모델, `general`은 별도 전역 모델이다. 둘 다 새 운용에서 100%로 인증된 모델은 아니다.

검증: 원본 파일 크기·수정시각 유지, 평가280장 각1회 포함, CSV에서 성적 재계산 일치, 파일명 변경 시 예측 불변, 센터 동일 입력 0 반환, 배치 측정과 실행 함수의 일치를 포함한 **{len(checks)}개 검사 통과**. [검사 기록](verification/checks.json), [실행 예시](verification/runtime_examples.csv).

## 8. 목표까지 가기 위한 다음 최적 순서

1. **0점과 정답부터 고정한다.** 각 시행 전후에 0점으로 복귀해 센터를 다시 찍고, 가능하면 엔코더 또는 외부 계측 위치를 함께 기록한다. 명령값과 실제 위치 중 무엇이 최종 정답인지 결정해야 한다.
2. **작은 이동을 집중 수집한다.** Y {{0, ±0.05, ±0.10, ±0.20}}, T {{0, ±0.01, ±0.02, ±0.05}}의 합계13개 서로 다른 자세를 양쪽 접근 방향, 20회 반복으로 촬영하면 520장이 된다. 순서를 무작위화해 시간과 이동량의 결합을 줄인다. 각 시행의 기준 사진도 저장한다.
3. **촬영 시행을 미리 학습/검증으로 나눈다.** 다른 날 또는 완전히 새 시행을 최종 시험으로 남기고, 보정·점 선택·문턱·모델 선택에 사용하지 않는다. 현재 어려운 T−0.05, T+0.01 부근을 충분히 포함한다.
4. **현재 205점·12계수 모델을 기준으로 검증한다.** 실제 반복 자료로 0점 변동과 영상 측정 잡음을 따로 추정하고, 6개 고정 절편 대신 측정된 센터 기준 보정을 학습한다.
5. 그래도 두 사진의 QR만으로 축 오분류가 남으면 판 나사를 포함한 동일 조건 시험을 한다. 이번 자료에서는 나사 추가만으로 100%가 되지 않았다. 추가 사진/다른 깊이 단서를 쓰는 방법은 입력 조건 변경이므로 별도 실험이다.

현재 자료로 할 수 있는 추적·모델·보정·분할 실험은 수행했다. 추가 모델 탐색으로 특정 분할 숫자를 올리는 것보다, **반복 가능한 0점과 독립 위치 정답을 갖춘 새 시행 검증**이 다음 결정에 더 큰 정보를 준다.

## 9. 산출물과 재현

- [최종 요약 CSV](summary_metrics.csv)
- [조건부 모델 사진별 예측](calibrated_runs/stress_predictions.csv), [반복분할 성적](calibrated_runs/stress_metrics.csv)
- [중첩 검증 사진별 예측](modeling/unified_isolated_stratified_predictions.csv), [새 시행 사진별 예측](modeling/unified_isolated_session_out_predictions.csv)
- [특징점·모델 후보 비교](modeling/broad_metrics.csv), [신규 측정 비교](modeling/new_broad_metrics.csv), [점 수 비교](modeling/point_selection_metrics.csv), [나사 비교](modeling/new_screw_fallback_metrics.csv)
- [0점 실험](zero_gn17_spread14/zero_ablation.csv), [보정 학습 진단](zero_gn17_spread14/train_only_correction_diagnostics.csv)
- [특징점 추적 설명](measurement/MEASUREMENT_NOTES.txt), [합성 이동 검증](measurement/synthetic_cubic_validation.csv)
- [데이터 목록](manifest.csv), [센터 해시·단위·조건](reference.json), [라이브러리 버전](requirements.txt)

`run.ps1 -Action reproduce`는 원본 BMP에서 주요 측정·보정·통합 검증·모델·보고서를 다시 만든다. 최초 실행 환경은 번들 Python 3.12와 이 폴더의 `deps`다. 다른 환경은 `requirements.txt`를 설치하고 Python 경로를 지정한다. 과거 캐시 진단과 부가 합성 실험의 스크립트는 각 하위 폴더에 별도로 보존했다. 원본 사진과 기존 분석 파일은 수정하지 않았다.
'''
    (HERE/'RESULTS.md').write_text(body,encoding='utf8')
    def img(path):return '<img src="data:image/png;base64,'+base64.b64encode((HERE/path).read_bytes()).decode()+'">'
    tbl=pd.DataFrame(summary_rows,columns=headers).to_html(index=False,border=0,escape=True)
    summary=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>QR 3개 Y/T 독립 실험</title><style>body{{font-family:'Malgun Gothic',sans-serif;background:#f4f6fa;color:#17253a;margin:0;line-height:1.75}}main{{max-width:1120px;margin:32px auto;background:white;padding:40px 48px;border-radius:16px}}h1{{font-size:30px;line-height:1.4}}h2{{font-size:23px;margin-top:38px}}p{{font-size:17px}}.tag{{color:#467093;letter-spacing:2px}}.note{{background:#fff6e8;border-left:4px solid #da9336;padding:16px 20px}}.cards{{display:grid;grid-template-columns:repeat(3,1fr);gap:15px}}.card{{padding:20px;background:#eef4fc;border-radius:10px}}.big{{display:block;font-size:30px;font-weight:700}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{padding:11px 8px;border-bottom:1px solid #dce3ed;text-align:left}}th{{background:#edf2f8}}img{{width:100%;height:auto}}a{{color:#1e61a5}}@media(max-width:700px){{main{{padding:20px}}.cards{{grid-template-columns:1fr}}table{{font-size:11px}}}}@media print{{body{{background:white}}main{{margin:0;padding:0}}h2{{break-after:avoid}}img{{max-height:620px;object-fit:contain}}}}</style><main><p class="tag">SE MES · INDEPENDENT EXPERIMENT · 2026.10.14</p><h1>QR 3개로 Y/T 방향·이동량 추정</h1><p><b>현재 6개 시행의 0점을 학습하면 조건부 100%를 얻었다.</b><br>새 시행까지 100%라는 결론은 아직 성립하지 않는다.</p><div class="cards"><div class="card">동일 시행 구간 검증<span class="big">280 / 280</span>QR 내부 205점 · 축+부호</div><div class="card">세 주요 분할 최대오차<span class="big">Y 0.0371</span>T 0.00368 · 둘 다 0.09 이내</div><div class="card">반복분할 안정성<span class="big">{rp100} / 20회</span>나머지는 1~2장 오분류</div></div><p class="note">허용오차는 Y·T 각각 0.09. 최소 이동간격 0.05·0.01과 구분했다. 성적은 원래 명령 라벨 기준이며, 영상에서 만든 보정값을 실측 정답으로 사용하지 않았다.</p><h2>검증 범위별 결과</h2>{tbl}{img('figures/evaluation.png')}<h2>점은 많이 뽑는 것보다 대응 관계가 먼저다</h2><p>매 사진 재검출의 1px 위치 재현율은 평균59.71%였지만, 센터 고정 ID 추적은 모든 사진의 QR928점을 유지했다. 공간 배치·점수·모델을 비교해 조건부 모델에는 QR별로 잘라낸205점을 사용했다.</p>{img('measurement/gn17_spread14_features.png')}<h2>0점 차이는 확인됐지만, 실제 위치 정답은 아직 없다</h2><p>여섯 시행의 회귀 절편을 센터와 비교했다. 반복 촬영된0점7장이 있는 것은 아니다. 기존 시행 절편을 후보로 쓰는 방식이 효과적이었고, 새 시행의0점은 별도 검증이 필요하다.</p>{img('figures/zero_offsets.png')}<h2>실행 코드와 근거</h2><p><a href="RESULTS.md">전체 보고서: 과정·공식·모든 평가·재현</a> · <a href="summary_metrics.csv">요약 CSV</a> · <a href="calibrated_runs/stress_predictions.csv">사진별 예측</a> · <a href="infer.py">두 사진 추론 코드</a> · <a href="verification/checks.json">검증 기록</a></p><p>다음 우선순위는 모델 수를 늘리는 일이 아니라, 양방향 접근·무작위 순서·반복0점·독립 위치 정답을 갖춘 새 촬영 시행이다.</p></main></html>'''
    (HERE/'REPORT.html').write_text(summary.replace('SE MES','SEMES'),encoding='utf8')
    (HERE/'README.md').write_text('# 독립 QR/Y/T 실험\n\n먼저 [전체 결과 보고서](RESULTS.md) 또는 [시각 요약](REPORT.html)을 읽으세요.\n\n현재6개 시행 조건의100%와 새 시행 일반화 성적을 구분했습니다. 원본과 기존 분석 파일은 변경하지 않았습니다. 실행: `run.ps1 -Action predict -Center <경로> -Current <경로>`.\n',encoding='utf8')
    print('Wrote RESULTS.md, REPORT.html, summary_metrics.csv and three figures.')

if __name__=='__main__':main()
