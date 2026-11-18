"""평가 (00_평가기준.md): 사진마다 (판정 축, 값) → 축 정확도 + 오차, 정답 기준 2개 (명령값 / 실제 위치).
대상: v6 특징점 (t7_predictions.csv), v5 패치 (analysis_v5/predictions.csv). 같은 짝/홀 폴드, 같은 보정본.
출력: 42_평가표_특징점vs패치.csv (전체 조건), 43_사진별_예측_대표방법.csv"""
import numpy as np, pandas as pd
from common import HERE, PROJ

MAIN = {  # 결과를 보기 전에 정한 대표 설정: 보정본으로 학습, 2차 매핑, 짝/홀, 특징점 17px
    'v6 특징점 QR만': '17px / QR만 / 보정본 / 2차 / 짝홀',
    'v6 특징점 QR + 판밖': '17px / QR + 판밖 / 보정본 / 2차 / 짝홀',
    'v5 패치 QR만': '보정본 / QR / PC+GN / 2차',
    'v6 특징점 QR만 (명령값 학습)': '17px / QR만 / raw / 2차 / 짝홀',
    'v6 특징점 QR + 판밖 (명령값 학습)': '17px / QR + 판밖 / raw / 2차 / 짝홀',
    'v5 패치 QR만 (명령값 학습)': 'raw / QR / PC+GN / 2차',
}

def evaluate(df, basis):
    t = df[basis].values; p = df.pred.values; ok = (df.pred_axis == df.axis).values
    y, th = (df.axis == 'Y').values, (df.axis == 'Θ').values
    e = np.abs(p - t)
    full = np.where(ok, e, np.abs(t))                                     # 축 틀리면 정답 축 예측 = 0
    passed = np.where(ok, e <= 0.09, (np.abs(t) <= 0.09) & (np.abs(p) <= 0.09))
    r = dict(축정확도=ok.mean() * 100, 축정확도_Y=ok[y].mean() * 100, 축정확도_Θ=ok[th].mean() * 100,
             축틀림_Y=int((~ok & y).sum()), 축틀림_Θ=int((~ok & th).sum()))
    for m, nm, u in [(y, 'Y', 1000), (th, 'Θ', 1000)]:
        mm = m & ok
        r[f'{nm}_평균오차'] = e[mm].mean() * u; r[f'{nm}_RMS'] = np.sqrt((e[mm]**2).mean()) * u; r[f'{nm}_최대'] = e[mm].max() * u
        r[f'{nm}_평균오차_전체'] = full[m].mean() * u
    r['통과09'] = passed.mean() * 100
    return r

if __name__ == '__main__':
    v6 = pd.read_csv(HERE/'t7_predictions.csv'); v5 = pd.read_csv(PROJ/'analysis_v5/predictions.csv')
    v6['방법'] = 'v6 특징점'; v5['방법'] = 'v5 패치'
    rows = []
    for src in (v6, v5):
        for key, d in src.groupby('조건', sort=False):
            for basis, bn in [('cmd', '명령값 (보정 안 함)'), ('act', '실제 위치 (보정함)')]:
                rows.append(dict(방법=d.방법.iloc[0], 조건=key, 정답기준=bn, 사진수=len(d), **evaluate(d, basis)))
    T = pd.DataFrame(rows)
    T['대표'] = T.조건.map({v: k for k, v in MAIN.items()}).fillna('')
    T.round(3).to_csv(PROJ/'42_평가표_특징점vs패치.csv', index=False, encoding='utf-8-sig')
    per = []
    for nm, key in MAIN.items():
        src = v5 if nm.startswith('v5') else v6; d = src[src.조건 == key].copy()
        d['방법'] = nm; d['축맞음'] = d.pred_axis == d.axis
        d['오차_명령값'] = np.where(d.축맞음, d.pred - d.cmd, np.nan); d['오차_실제위치'] = np.where(d.축맞음, d.pred - d.act, np.nan)
        per.append(d[['방법', 'file', 'axis', 'cmd', 'act', 'pred_axis', 'pred', '축맞음', '오차_명령값', '오차_실제위치', 'cost_ratio']])
    pd.concat(per).round(5).to_csv(PROJ/'43_사진별_예측_대표방법.csv', index=False, encoding='utf-8-sig')
    cols = ['대표', '정답기준', '축정확도', '축틀림_Y', '축틀림_Θ', 'Y_평균오차', 'Y_RMS', 'Y_최대', 'Θ_평균오차', 'Θ_RMS', 'Θ_최대', 'Y_평균오차_전체', 'Θ_평균오차_전체', '통과09']
    print(T[T.대표 != ''][cols].round(2).to_string())
