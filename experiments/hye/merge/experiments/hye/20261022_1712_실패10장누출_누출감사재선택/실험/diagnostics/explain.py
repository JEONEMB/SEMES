"""Attach cautious per-image explanations to the unchanged saved OOF predictions."""
from pathlib import Path
import json,sys
import numpy as np,pandas as pd,joblib

H=Path(__file__).resolve().parent
sys.path.insert(0,str(H.parents[1]/'joint_yt_disentangle_20261022'))
import learned_model

def main():
    d=pd.read_csv(H/'all60_diagnostics.csv');q=pd.read_csv(H/'all60_qr_diagnostics.csv')
    notes={
      291: dict(mechanism='같은 조건 중 이 반복만 Y가 크게 올라감. 조건 편향보다 개별 반복의 영향이 큼.',observations=['다른 두 반복의 Y는 0.0708, 0.0212로 모두 통과한다.','같은 조건 평균에서 벗어난 Y +0.1141 중 QR간 상대 이동 기여 +0.0738, QR 내부 형태 기여 +0.0369이다. 공통 이동 기여는 +0.0034이다.','경계 RMSE p95는 48.3백분위로 중간이지만 두 창 좌표 안정성 p95는 95백분위이다.'],hypotheses=['개별 측정의 작은 상대 좌표 변화가 축 분리 방향으로 증폭됐을 가능성이 크다. 창 크기 민감성도 있으나 실제 자세 반복 오차와 영상 측정 오차는 분리할 수 없다.','이 fold에서 같은 T=0.1인 혼합 학습 Y 최소값은 0.1이므로 Y=0.05 조건에서 외삽이 추가로 작용할 수 있다.']),
      295: dict(mechanism='해당 조건의 지속적인 Y 과대추정에 반복 차이가 더해져 기준을 조금 초과함.',observations=['세 반복 Y가 0.1066, 0.0823, 0.1475로 모두 명령 0.05보다 크다.','조건 평균 편향 +0.0621에 개별 편차 +0.0354가 더해졌다.','같은 T=1.0에서 혼합 학습 Y는 0.1, 0.5, 1.0만 존재한다. 형태와 원좌표 모두 가장 가까운 학습 사진은 Y=0.1,T=1.0이다.','RMSE p95는 20백분위로 낮다.'],hypotheses=['낮은 Y 경계의 보정 편향과 반복별 상대 QR 위치 변화가 결합했을 가능성. 경계 맞춤 실패가 주원인이라는 증거는 약하다.']),
      305: dict(mechanism='세 반복이 함께 Y를 작게 추정하는 조건 편향이 주된 관측.',observations=['세 반복 Y는 0.0021, 0.0504, 0.0108이다. 해당 사진이 조건 평균보다 낮은 양은 -0.0190이며 조건 평균 편향은 -0.0789이다.','이 fold 학습에는 T=0.2인 혼합 조건이 없다. T 단일축 사진의 존재와는 별개이다.','명령값에서의 표준화 순방향 잔차 1.3916은 예측값에서의 1.2437보다 크다. 즉 잘못된 예측도 학습한 영상-자세 관계상 더 그럴듯하다.'],hypotheses=['미관측 혼합 조건의 조건부 보정 곡면이 실제 관측과 맞지 않는 가능성. 개별 사진의 큰 추적 실패보다 조건 단위의 보정 문제를 우선 의심할 근거가 있다.']),
      311: dict(mechanism='음의 조건 편향과 개별 반복의 낮은 Y 추정이 함께 작용.',observations=['다른 두 반복 Y는 0.1022, 0.0314이며 이 사진은 -0.0161이다.','조건 편향 -0.0609, 개별 편차 -0.0553이다. QR간 상대 이동 기여 -0.0733을 QR내 형태 +0.0224가 일부 상쇄한다.','가장 가까운 학습 사진은 Y=0.05,T=2.0이며 원좌표 RMS 거리 0.0924px이다.','같은 Y=0.1의 혼합 학습 T 최대는 1.0이고 시험 T는 2.0이다.'],hypotheses=['높은 T 조건으로의 외삽과 Y를 구분하는 QR간 상대 위치 변동이 결합했을 가능성. 음수 출력 자체를 반올림·클리핑하면 원인을 숨길 수 있다.']),
      321: dict(mechanism='Y=0.5,T=1.0 조건의 양의 편향에 이 반복의 QR 내부 형태 변동이 더해짐.',observations=['같은 조건 세 반복 중 두 장이 실패한다. Y 추정은 0.5275, 0.6415, 0.5928이다.','조건 편향 +0.0873에 개별 편차 +0.0542가 더해졌다. 개별 편차의 QR 내부 형태 기여는 +0.0506이다.','평행이동을 뺀 좌표 RMS가 전체60 중 93.3백분위이다. RMSE p95는 16.7백분위다.','정규화 픽셀 Laplacian은 6.7백분위로 낮지만 이 지표는 실제 흐림 외에 픽셀 위상·QR 투영 형태도 반영한다.'],hypotheses=['조건 보정 편향이 이미 0.09에 가까워 영상의 작은 추가 형태 차이를 흡수하지 못한다. 낮은 focus 지표는 보조 관측이며 흐림이 원인이라고 확정할 수 없다.']),
      322: dict(mechanism='개별 사진의 큰 이상보다 조건 공통 편향 때문에 0.09를 근소하게 초과.',observations=['조건 편향 +0.0873, 개별 편차 +0.0055이다. 실제 절대오차 0.09284는 기준보다 0.00284 크다.','같은 조건의 두 번째 반복도 실패하며 첫 번째 반복만 통과한다.','평행이동 제거 후 좌표 RMS는 40백분위, 경계 RMSE p95는 23.3백분위이다.'],hypotheses=['이 사진만 재측정하기보다 Y=0.5,T=1.0 부근의 모델 조건 편향을 개선해야 할 가능성이 높다. 단일 사진이 유난히 나쁘다는 근거는 부족하다.']),
      324: dict(mechanism='양의 조건 편향에 반복별 상대 이동·형태 변화가 더해짐.',observations=['세 반복 Y는 0.5769, 0.6249, 0.5360으로 모두 명령 0.5보다 크다.','조건 편향 +0.0793, 개별 편차 +0.0456이다. 공통 이동 +0.0099, QR 상대 이동 +0.0227, QR내 형태 +0.0130이다.','같은 Y=0.5의 혼합 학습 T 최대는 1.0이며 시험 T=2.0이다.','측정 RMSE p95는 18.3백분위이나 평행이동 제거 좌표 RMS는 90백분위이다.'],hypotheses=['높은 T로의 조건부 외삽 편향과 반복별 관측 변동이 결합했을 가능성. 경계 적합 품질만 개선해서 해결된다고 보기 어렵다.']),
      327: dict(mechanism='Y 상단·T 하단 조건에서 생긴 공통 양의 편향이 대부분을 설명.',observations=['세 반복 Y는 1.0844, 1.0928, 1.0680이다. 조건 편향 +0.08175, 개별 편차 +0.01109이다.','절대오차 0.09284는 기준보다 0.00284 크다.','같은 T=0.01의 혼합 학습 Y 최대는 0.5이고 같은 Y=1.0의 혼합 학습 T 최소는 0.1이다.','원좌표가 가장 가까운 학습 사진은 Y=0,T=0.19이다. 원좌표 유사성만으로 축을 구분하기 어렵다는 한 사례다.'],hypotheses=['조건 범위 끝에서의 보정 편향이 주된 가능성. T가 약간 음수여도 절대오차 0.0112로 통과하며, 실패 기준은 Y이다.']),
      330: dict(mechanism='이 반복만 낮아진 Y에 QR간 상대 위치 변화가 강하게 연결됨.',observations=['다른 두 반복 Y는 0.9934, 0.9831로 명령 1.0에 가깝지만 이 사진은 0.8869이다.','개별 편차 -0.0676 중 QR간 상대 이동 기여는 -0.1086이며 QR내 형태 +0.0353과 공통 이동 +0.0058이 일부 상쇄한다.','경계 RMSE p95는 91.7백분위지만 추정값에서 순방향 잔차는 0.0897로 전체60 중 가장 작다. 작은 모델 잔차가 정답 보장이 되지 않는 사례다.','같은 T=0.1의 혼합 학습 Y 최대는 0.5이다.'],hypotheses=['작은 QR 상대 위치 차이가 Y/T 보상 방향으로 해석됐을 가능성. 외삽도 겹치지만 경계 측정 오차와 실제 반복 자세 편차를 현재 자료로 분리할 수 없다.']),
      336: dict(mechanism='가장 큰 개별 반복 형태 차이가 Y/T 보상 방향으로 증폭된 최악 사례.',observations=['다른 두 반복 Y는 1.0653, 0.9668로 모두 통과하며 이 사진은 0.7426이다.','Y 오차 -0.25738 = 조건 편향 -0.07510 + 개별 편차 -0.18228이다.','개별 편차의 일차 근사는 공통 이동 +0.00608, QR 상대 이동 -0.04223, QR내 형태 -0.14622이다.','QR내 형태의 Y 기여는 세 QR 모두 음수(-0.04538,-0.05757,-0.04327)이다. 특정 QR 하나만 망가진 모양은 아니다.','경계 RMSE p95는 8.3백분위로 좋고 NCC 최소 0.977이다. 같은 조건 대비 평행이동 제거 좌표 RMS는 86.7백분위이다.','학습에서 같은 T=2.0의 Y 최대는 0.5이며 같은 Y=1.0의 T 최대는 1.0이다.'],hypotheses=['두 축 조건의 모서리 영역 외삽과 전 QR에 걸친 미세 관측 변화가 결합했을 가능성. 원점 평행이동만 바꾸거나 나쁜 QR 한 개를 제거하면 해결된다는 증거는 없다.'])
    }
    entries=[]
    for _,r in d[~d.passed].iterrows():
        i=int(r['index']);peers=d[(d.condition==r.condition)&(d['index']!=i)]
        fields=['Y','T','predY','predT','signed_error_Y','signed_error_T','eY','eT','condition_bias_Y','individual_pred_deviation_Y','translation_component_Y','qr_differential_translation_Y','within_qr_shape_Y','valid_points','ncc_min','rmse_p95','rmse_p95_percentile_all60','window_stability_p95_percentile_all60','repeat_centered_D_rms','repeat_centered_D_rms_percentile_all60','pixel_laplacian_normalized_mean_percentile_all60']
        entries.append(dict(index=i,file=r['file'],fold=int(r.fold),diagnosis=notes[i],evidence={c:r[c] for c in fields},other_repeats=peers[['index','file','predY','predT','eY','eT','passed']].to_dict('records'),qr=q[q['index']==i].to_dict('records'),limit='명령값 기준 오차의 사후 분석이다. 독립 실측 자세가 없어 실제 기계 반복오차와 영상 측정·모델 오차의 물리적 인과를 확정할 수 없다.'))
    err=d[['signed_error_Y','signed_error_T']].to_numpy();slope=float(np.linalg.lstsq(np.c_[np.ones(len(err)),err[:,0]],err[:,1],rcond=None)[0][1]);center=d[(d['Y']==0)&(d['T']==0)]
    folds=[]
    for f in range(4):
        b=joblib.load(H.parents[1]/'joint_yt_disentangle_20261022'/f'pass_first_fold{f}.joblib');w=b['model'];a=dict(fold=f,config=w.config,zero_displacement_prediction=w.predict(np.zeros((1,630,2)))[0].tolist())
        if w.config['method']=='repeat':
            s=w.model.slope[:,:1260];u=s/np.linalg.norm(s,axis=1,keepdims=True);a.update(raw_axis_cosine=float(u[0]@u[1]),raw_normalized_axis_condition_number=float(np.linalg.cond(u.T)),feature_response=w.model.response.tolist())
        folds.append(a)
    summary=dict(all60_error_correlation=float(np.corrcoef(err.T)[0,1]),T_error_on_Y_error_regression_slope=slope,center_oof=center[['file','predY','predT','passed']].to_dict('records'),folds=folds,mechanistic_observations=['학습 순방향 모델의 raw Y/T 방향이 거의 평행하므로 큰 공통 이동을 상쇄한 미세 상대 좌표로 축을 구분한다.','학습 평균방향에서 모델 응답이 항등행렬이어도 보지 못한 조건과 반복 변동에서 정확한 축 분리가 보장되지 않는다.','조건 편향은 양·음이 섞여 있으며 원점3장은 모두 통과하므로 단일 전역 원점 이동만으로 실패10장을 설명할 수 없다.'],limits=['방향 cosine은 각 fold에서 명령으로 학습한 순방향 모델의 기술통계이며 실제 물리 Jacobian 계측이 아니다.','zero_displacement_prediction은 정확히 D=0인 합성 입력에 대한 저장된 모델 응답이며 실제 새 센터 사진 성적과 구분한다.','미분 기여는 저장된 추정기의 국소 민감도다. 실제 물리 원인의 기여도가 아니다.','전체60은 반복 사용한 개발 데이터다. 독립검증 성적이 아니다.'])
    (H/'per_image_analysis.json').write_text(json.dumps(entries,ensure_ascii=False,indent=2),encoding='utf-8')
    (H/'fold_axis_diagnostics.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    check=dict(images=len(d),failure_entries=len(entries),qr_rows=len(q),max_linearization_discrepancy_Y=float(abs(d.linearized_repeat_delta_Y-d.individual_pred_deviation_Y).max()),exact_error_decomposition_max_error=float(abs(d.signed_error_Y-d.condition_bias_Y-d.individual_pred_deviation_Y).max()),gradient_component_sum_max_error=float(abs(d.linearized_repeat_delta_Y-d.translation_component_Y-d.qr_differential_translation_Y-d.within_qr_shape_Y).max()))
    assert check['exact_error_decomposition_max_error']<1e-12 and check['gradient_component_sum_max_error']<1e-12
    (H/'verification.json').write_text(json.dumps(check,indent=2),encoding='utf-8');print(json.dumps(check,indent=2))

if __name__=='__main__':main()
