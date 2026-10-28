"""Refit empirical coordinate estimators on the new mixed session only.

No hypothesis tests. Leave an entire commanded coordinate out per fold.
Inputs are the existing fixed-630-point measurements relative to center1.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def design(x, kind):
    if kind == 'mean_quadratic':
        a, b = x.T
        return np.column_stack([np.ones(len(x)), a, b, a*a, a*b, b*b])
    return np.column_stack([np.ones(len(x)), x])


def fit(x, y, kind):
    mean, scale = x.mean(axis=0), x.std(axis=0)
    if np.any(scale < 1e-12):
        raise ValueError('Degenerate measurement channel')
    d = design((x-mean)/scale, kind)
    if np.linalg.matrix_rank(d) != d.shape[1]:
        raise ValueError('Rank-deficient calibration')
    return dict(mean=mean.tolist(), scale=scale.tolist(),
                coefficients=np.linalg.lstsq(d, y, rcond=None)[0].tolist(),
                condition_number=float(np.linalg.cond(d)), kind=kind)


def predict(model, x):
    return design((x-np.array(model['mean']))/np.array(model['scale']),
                  model['kind']) @ np.array(model['coefficients'])


def main(source, output):
    output.mkdir(parents=True, exist_ok=True)
    files = ['frames.csv', 'qr_displacements.csv']
    provenance = {f: hashlib.sha256((source/f).read_bytes()).hexdigest() for f in files}
    meta = pd.read_csv(source/'frames.csv').set_index('path')
    qr = pd.read_csv(source/'qr_displacements.csv')
    wide = qr.pivot(index='path', columns='qr', values=['dx', 'dy'])
    columns = [(dim, slot) for slot in ['L', 'C', 'R'] for dim in ['dx', 'dy']]
    wide = wide[columns]
    data = meta.join(wide.set_axis([f'{s}_{d}' for d, s in columns], axis=1))
    channels = [f'{s}_{d}' for s in ['L', 'C', 'R'] for d in ['dx', 'dy']]
    mixed = data[data.category.eq('mixed')].sort_index().copy()
    refs = data[data.category.eq('new-reference')].sort_index().copy()
    if len(mixed) != 57 or len(refs) != 3:
        raise ValueError('Expected the documented 57 mixed + 3 reference images')
    x = mixed[channels].to_numpy()
    y = mixed[['y_mm', 'theta_deg']].to_numpy()
    coords = np.unique(y, axis=0)
    if len(coords) != 19:
        raise ValueError('Expected 19 mixed coordinates')
    # Equal QR weighting is deliberate: retain spatial differences for 3QR model.
    meanx = x.reshape(-1, 3, 2).mean(axis=1)
    scores, residuals, models = [], [], {}
    for kind in ['mean_affine', 'three_qr_affine', 'mean_quadratic']:
        xx = x if kind == 'three_qr_affine' else meanx
        held = np.zeros_like(y)
        for coord in coords:
            test = np.all(y == coord, axis=1)
            held[test] = predict(fit(xx[~test], y[~test], kind), xx[test])
        error = held-y
        rmse = np.sqrt(np.mean(error**2, axis=0))
        scores.append(dict(model=kind, y_rmse_mm=float(rmse[0]),
                           theta_rmse_deg=float(rmse[1]),
                           y_max_abs_mm=float(abs(error[:, 0]).max()),
                           theta_max_abs_deg=float(abs(error[:, 1]).max()),
                           selection_score=float(np.mean(rmse/np.ptp(y, axis=0)))))
        for i, path in enumerate(mixed.index):
            residuals.append(dict(model=kind, path=path, y_mm=y[i,0], theta_deg=y[i,1],
                                  predicted_y_mm=held[i,0], predicted_theta_deg=held[i,1],
                                  error_y_mm=error[i,0], error_theta_deg=error[i,1]))
        models[kind] = fit(xx, y, kind)
        models[kind]['input_channels'] = channels if kind == 'three_qr_affine' else ['mean_qr_dx', 'mean_qr_dy']
    scores = pd.DataFrame(scores).sort_values('selection_score')
    selected = scores.iloc[0].model
    pd.DataFrame(residuals).to_csv(output/'coordinate_holdout.csv', index=False)
    scores.to_csv(output/'model_comparison.csv', index=False)
    forward = fit(y, x, 'forward_affine')
    models['forward_affine'] = forward
    pred = predict(models[selected], x if selected == 'three_qr_affine' else meanx)
    result = mixed[['y_mm','theta_deg']].copy()
    result[['predicted_y_mm','predicted_theta_deg']] = pred
    result.to_csv(output/'fitted_coordinates.csv')
    reference_predictions = {}
    for kind in scores.model:
        rx = refs[channels].to_numpy()
        if kind != 'three_qr_affine':
            rx = rx.reshape(-1,3,2).mean(axis=1)
        reference_predictions[kind] = predict(models[kind], rx).tolist()
    bundle = dict(selected_model=selected, models=models, source_sha256=provenance,
                  measurement_reference='기준 위치/center1.bmp',
                  measurement_method='fixed 630 points; per-QR means',
                  training_images=57, training_coordinates=19,
                  units=['mm','degree'], reference_predictions=reference_predictions,
                  reference_paths=refs.index.tolist(),
                  scope='empirical command-coordinate calibration; no hypothesis tests',
                  limits='positive mixed coordinates only; origin extrapolates; CV used for model selection, no independent test set')
    (output/'model.json').write_text(json.dumps(bundle, ensure_ascii=False, indent=2), encoding='utf-8')
    best = scores.iloc[0]
    note = f'''# 신규 혼합축 데이터 모델 재학습

가설 검증 없이, 절편을 포함한 경험적 좌표 추정 모델을 재학습했다. 원본 영상과 기존 분석은 수정하지 않았다.
기존 v2의 center1 기준 고정 630점 계측을 재사용하며, QR별 평균 dx/dy를 입력으로 사용한다.
신규 혼합 19좌표 × 3장 = 57장만 학습했다. 기준 3장은 학습에서 제외하고 예측값만 별도 기록했다.
정답은 파일명 명령 Y(mm), θ(°)이며 독립 실측 정확도는 아니다.

## 선택한 모델

`{selected}`. 절편을 자유롭게 학습하므로 입력 변위 0에서 출력 좌표 0을 강제하지 않는다.
이 절편을 실제 기구 영점 오차로 해석하지 않는다. 입력은 학습 평균과 표준편차로 정규화한다.
각 모델의 입력 순서, 정규화 계수, 절편 포함 계수는 model.json에 저장했다.
mean 모델은 L/C/R QR 평균에 동일 가중을 사용한다. 기존 630점 전체 평균과 가중 방식이 다르다.
forward_affine은 [Y, θ]에서 QR 6채널을 예측하는 별도 정방향 모델이다.

## 예측 평가

좌표 하나의 반복 3장 전부를 보류하는 19-fold 평가를 사용했다. 정규화도 매 fold 학습분에서만 계산한다.
각 축의 RMSE를 해당 명령 범위로 나눈 값의 평균이 가장 작은 모델을 선택했다.
선택 모델의 좌표 보류 RMSE: Y **{best.y_rmse_mm:.6f} mm**, θ **{best.theta_rmse_deg:.6f}°**.
최대 절대 오차: Y {best.y_max_abs_mm:.6f} mm, θ {best.theta_max_abs_deg:.6f}°.
동일 평가를 모델 선택에도 사용했으므로 독립 최종 시험 성능은 아니다.
선택 모델의 정규화 설계행렬 조건수는 {models[selected]['condition_number']:.1f}이다.
QR 입력 채널의 강한 상관 때문에 계수가 불안정할 수 있어, 현재 모델은 실험용 보정 모델로 취급한다.

적용 범위는 관측한 양의 혼합 좌표 주변이다(Y 0.05~1 mm, θ 0.01~2°; 불규칙 19좌표).
범위 안의 모든 조합을 촬영한 것은 아니다. 원점·음수 영역은 외삽이다.
기준 3장의 예측은 model.json의 reference_predictions에 기록하며 성능 수치에 섞지 않았다.

## 산출물

- model.json: 재사용 가능한 모델과 입력 출처 SHA256
- model_comparison.csv: 세 후보의 좌표 보류 성능
- coordinate_holdout.csv: 영상별 보류 예측과 오차
- fitted_coordinates.csv: 전체 학습 후 예측(평가 성능과 구분)

재현: `python tool/offaxis/refit_model.py --source docs/offaxis-eda/ppt-method-v2 --output docs/offaxis-eda/refit-model`
브랜치 생성과 커밋은 수행하지 않았다.
'''
    (output/'README.md').write_text(note, encoding='utf-8')
    print(scores.to_string(index=False))
    print('Selected:', selected)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    main(args.source, args.output)
