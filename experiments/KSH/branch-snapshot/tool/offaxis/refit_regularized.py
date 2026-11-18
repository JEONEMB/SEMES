"""Nested coordinate CV for centered affine/quadratic ridge calibration.

Only numpy/pandas required. No hypothesis tests or image modifications.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd


TICKS = np.array([.05, .01])
ALPHAS = [0., 1e-6, 1e-4, .01, 1., 100.]


def features(z, degree):
    if degree == 1:
        return z
    return np.column_stack([z] + [z[:, i]*z[:, j] for i in range(6) for j in range(i, 6)])


def fit(x, y, degree, alpha):
    # A virtual center in command space; never constrained to center1 or origin.
    xc, yc = x.mean(0), y.mean(0)
    xs = x.std(0)
    xs = np.where(xs > 1e-12, xs, 1.)
    if degree < 0:
        z = (x-xc)/xs
        length = {-1: .5, -2: 2., -3: 8.}[degree]
        dist = np.sum((z[:,None,:]-z[None,:,:])**2,axis=2)
        kernel = np.exp(-dist/(2*length*length))
        coef = np.linalg.solve(kernel+alpha*np.eye(len(x)), (y-yc)/TICKS)
        return dict(degree=degree,alpha=alpha,x_center=xc.tolist(),y_center=yc.tolist(),
                    x_scale=xs.tolist(),length=length,training_z=z.tolist(),coefficients=coef.tolist())
    f = features((x-xc)/xs, degree)
    fc, fs = f.mean(0), f.std(0)
    fs = np.where(fs > 1e-12, fs, 1.)
    a = (f-fc)/fs
    u, s, vt = np.linalg.svd(a, full_matrices=False)
    if alpha == 0:
        weight = np.divide(1., s, out=np.zeros_like(s), where=s > s.max()*1e-12)
    else:
        weight = s/(s*s+alpha)
    coef = (vt.T*weight) @ (u.T @ ((y-yc)/TICKS))
    return dict(degree=degree, alpha=alpha, x_center=xc.tolist(),
                y_center=yc.tolist(), x_scale=xs.tolist(), feature_center=fc.tolist(),
                feature_scale=fs.tolist(), coefficients=coef.tolist())


def predict(m, x):
    if m['degree'] < 0:
        z = (x-np.array(m['x_center']))/m['x_scale']
        dist = np.sum((z[:,None,:]-np.array(m['training_z'])[None,:,:])**2,axis=2)
        return np.exp(-dist/(2*m['length']**2)) @ np.array(m['coefficients'])*TICKS + m['y_center']
    f = features((x-np.array(m['x_center']))/m['x_scale'], m['degree'])
    return ((f-m['feature_center'])/m['feature_scale']) @ np.array(m['coefficients'])*TICKS + m['y_center']


def folds(y):
    for c in np.unique(y, axis=0):
        test = np.all(y == c, axis=1)
        yield ~test, test


def tune(x, y):
    rows = []
    for degree in [1, 2, -1, -2, -3]:
        for alpha in (ALPHAS if degree > 0 else [1e-6, 1e-4, .01, 1.]):
            # Unregularized quadratic is excluded: too flexible for 19 poses.
            if degree == 2 and alpha == 0:
                continue
            pred = np.zeros_like(y)
            for train, test in folds(y):
                pred[test] = predict(fit(x[train], y[train], degree, alpha), x[test])
            score = float(np.mean(((pred-y)/TICKS)**2))
            rows.append(dict(degree=degree, alpha=alpha, mean_squared_tick_error=score))
    best = min(rows, key=lambda r: r['mean_squared_tick_error'])
    return best, rows


def metrics(name, pred, y):
    e = pred-y
    rmse = np.sqrt(np.mean(e*e, axis=0))
    return dict(model=name, y_rmse_mm=float(rmse[0]), theta_rmse_deg=float(rmse[1]),
                y_rmse_ticks=float(rmse[0]/TICKS[0]), theta_rmse_ticks=float(rmse[1]/TICKS[1]),
                y_within_one_tick=float(np.mean(abs(e[:,0]) <= TICKS[0])),
                theta_within_one_tick=float(np.mean(abs(e[:,1]) <= TICKS[1])),
                both_within_one_tick=float(np.mean(np.all(abs(e) <= TICKS, axis=1))),
                y_max_abs_mm=float(abs(e[:,0]).max()), theta_max_abs_deg=float(abs(e[:,1]).max()))


def main(source, output):
    output.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(source/'frames.csv').set_index('path')
    raw = pd.read_csv(source/'qr_displacements.csv')
    columns = [(d,s) for s in ['L','C','R'] for d in ['dx','dy']]
    wide = raw.pivot(index='path', columns='qr', values=['dx','dy'])[columns]
    channels = [f'{s}_{d}' for d,s in columns]
    data = meta.join(wide.set_axis(channels, axis=1))
    mixed = data[data.category.eq('mixed')].sort_index()
    x, y = mixed[channels].to_numpy(), mixed[['y_mm','theta_deg']].to_numpy()
    assert x.shape == (57,6) and len(np.unique(y,axis=0)) == 19
    old, new = np.zeros_like(y), np.zeros_like(y)
    selections = []
    for train, test in folds(y):
        old[test] = predict(fit(x[train], y[train], 1, 0.), x[test])
        best, _ = tune(x[train], y[train])
        model = fit(x[train], y[train], best['degree'], best['alpha'])
        new[test] = predict(model, x[test])
        selections.append(dict(held_y=float(y[test][0,0]), held_theta=float(y[test][0,1]), **best))
    best, tuning = tune(x,y)
    model = fit(x,y,best['degree'],best['alpha'])
    # Translation invariance: changing the measurement origin must not alter results.
    shift = np.array([31.,-.2,30.,-.1,29.,.3])
    shifted = fit(x+shift,y,best['degree'],best['alpha'])
    invariant_error = float(np.max(abs(predict(model,x)-predict(shifted,x+shift))))
    assert invariant_error < 1e-7
    assert np.isfinite(new).all()
    summary = [metrics('previous_affine',old,y), metrics('nested_selected_ridge',new,y)]
    pd.DataFrame(summary).to_csv(output/'comparison.csv',index=False)
    pd.DataFrame(selections).to_csv(output/'outer_fold_selection.csv',index=False)
    pd.DataFrame(tuning).to_csv(output/'final_tuning.csv',index=False)
    rows = []
    for name,pred in [('previous_affine',old),('nested_selected_ridge',new)]:
        for i,path in enumerate(mixed.index):
            rows.append(dict(model=name,path=path,y_mm=y[i,0],theta_deg=y[i,1],
                             predicted_y_mm=pred[i,0],predicted_theta_deg=pred[i,1],
                             error_y_mm=pred[i,0]-y[i,0],error_theta_deg=pred[i,1]-y[i,1]))
    pd.DataFrame(rows).to_csv(output/'coordinate_holdout.csv',index=False)
    bundle = dict(model=model,input_channels=channels,tick_sizes=TICKS.tolist(),
                  coordinate_origin='virtual training centroid, not constrained to a photographed center',
                  source_measurement_reference='center1; arbitrary translation removed by fitted x_center',
                  source_sha256={f:hashlib.sha256((source/f).read_bytes()).hexdigest() for f in ['frames.csv','qr_displacements.csv']},
                  final_parameters=best,metrics=summary,
                  validation=dict(outer_folds=19,inner_folds=18,translation_invariance_max_error=invariant_error),
                  limitations='command labels; positive mixed poses only; no independent session test')
    (output/'model.json').write_text(json.dumps(bundle,ensure_ascii=False,indent=2),encoding='utf-8')
    a,b=summary
    note=f'''# 가상 기준점과 정규화 회귀 재학습

가설 검증 없이 신규 혼합 57장·19좌표만 사용했다. 원본과 이전 모델은 보존한다.

## 기준점

center1은 기존 계측 파일의 저장 기준으로만 남긴다. 모델은 학습 자료의 평균 QR 변위와
평균 명령 좌표를 가상 기준으로 정하며, center1에서 출력 0을 강제하지 않는다.
최종 Y–θ 기준은 ({model['y_center'][0]:.8f} mm, {model['y_center'][1]:.8f}°)이다.
이 기준은 여러 영상의 평균에 대응하는 계산상 기준이며 해당 좌표에서 촬영한 실제 영상은 아니다.

절편이 자유로운 회귀에서는 기준을 평행 이동해도 예측은 같다. 따라서 임의 기준점의 위치를
평가 오차에 맞춰 탐색하는 것은 추가 개선 변수가 아니다. 이번에는 학습 중심을 기준으로
수치적으로 중심화하고, 실제 개선은 정규화 강도와 모델 차수 선택으로 수행했다.
기준 변위 이동 불변성 확인 최대 차이: {invariant_error:.3g}.

## 알고리즘

QR 3개 × dx/dy 6채널. 1차 선형, 모든 제곱·교차항을 포함한 2차 다항 릿지 회귀,
Gaussian RBF 커널 릿지 회귀(길이 0.5, 2, 8)를 비교했다.
degree 음수는 커널 모델 식별자(-1=0.5, -2=2, -3=8)이며 다항 차수가 아니다.
정규화·가상 중심·계수는 매 fold 학습 자료에서만 계산한다. 절편은 벌점에서 제외한다.
다항 모델은 SVD, 커널 모델은 정규화한 커널 선형계를 풀며 반복 비선형 최적화를 강제하지 않는다.
최종 차수 {best['degree']}, alpha={best['alpha']:g}. 출력축은 1틱 Y=0.05mm, θ=0.01°로 환산해 선택한다.

## 평가

외부 19-fold: 한 좌표의 3장을 통째로 보류. 내부 18-fold에서 차수와 alpha를 선택한다.
즉 외부 평가 좌표는 기준점 계산·정규화·튜닝·계수 학습에 사용하지 않는다.
최종 전체 학습 모델은 전체 자료의 좌표 CV로 설정을 정하며 그 학습 오차를 평가로 보고하지 않는다.

| 모델 | Y RMSE mm | θ RMSE ° | Y 틱 | θ 틱 | 두 축 모두 ±1틱 |
|---|---:|---:|---:|---:|---:|
| 이전 선형 | {a['y_rmse_mm']:.6f} | {a['theta_rmse_deg']:.6f} | {a['y_rmse_ticks']:.3f} | {a['theta_rmse_ticks']:.3f} | {a['both_within_one_tick']:.1%} |
| 신규 중첩 선택 | {b['y_rmse_mm']:.6f} | {b['theta_rmse_deg']:.6f} | {b['y_rmse_ticks']:.3f} | {b['theta_rmse_ticks']:.3f} | {b['both_within_one_tick']:.1%} |

반복을 학습/평가로 분리하지 않았다. 다만 같은 촬영 세션의 명령 좌표에 대한 성능이며 독립 실측 정확도가 아니다.
원점과 음수 분면은 학습하지 않았다. ±1틱 비율은 반올림 판정이 아닌 절대오차 기준이다.

## 재사용

model.json에 입력 순서, 가상 중심, 스케일, 계수 저장. 추론은 이 스크립트의 predict(model, x)를 사용한다.
2차 특징 순서는 원래 6채널 이후 i=0..5, j=i..5의 z_i*z_j이다.
comparison.csv는 성능, coordinate_holdout.csv는 영상별 외부 예측,
outer_fold_selection.csv는 외부 fold별 내부 선택, final_tuning.csv는 전체 학습 설정 선택 기록이다.

재현: `python tool/offaxis/refit_regularized.py --source docs/offaxis-eda/ppt-method-v2 --output docs/offaxis-eda/refit-regularized`
브랜치 생성과 커밋은 수행하지 않았다.
'''
    (output/'README.md').write_text(note,encoding='utf-8')
    print(pd.DataFrame(summary).to_string(index=False))
    print('Final settings:',best)
    print('Virtual center:',model['y_center'])


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    main(args.source,args.output)
