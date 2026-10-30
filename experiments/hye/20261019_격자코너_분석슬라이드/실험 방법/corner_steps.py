# 통과한 모서리로 (1) 한 칸 이동량, (2) trial별 점별 흔들림 계산
#   이동량 = 그 사진 모서리 x - center 모서리 x (같은 ID)
#   (1) 한 칸: 같은 trial에서 |명령값| 순서로 이웃한 두 사진, 두 사진 모두 통과한 모서리의 x 차이 평균 (부호는 바깥쪽 = +)
#       첫 칸 = center -> 첫 사진 (center와 첫 사진 모두 통과한 모서리)
#   (2) 점별 흔들림: trial의 모든 사진에서 통과한 모서리마다, 이동량(x, y)을 명령값에 2차 곡선으로 맞추고 남은 값의 표준편차
import numpy as np, pandas as pd, pathlib
import os
HERE = pathlib.Path(__file__).parent; RULE = os.environ.get('RULE', 'valid')   # valid: 진단 모두 통과 / nondiag: 대각선 모서리(검정 이웃 2)만 뺌 / all: 전부
OUT = HERE / 'out' / ('' if RULE == 'valid' else RULE); OUT.mkdir(exist_ok=True)
D = pd.read_csv(HERE / 'out' / 'corners_280.csv')
def trial(f):
    if f == 'center.bmp': return 'center'
    v = float(f[1:-4]); s = '+' if v > 0 else '-'
    if f[0] == 'T': return 'Θ ' + s
    return ('Y 0.1 간격 ' if round(abs(v) * 100) % 10 == 0 else 'Y 0.05 홀수 ') + s
D['trial'] = D.file.map(trial); D['value'] = D.file.map(lambda f: 0.0 if f == 'center.bmp' else float(f[1:-4]))
C = D[D.trial == 'center'].set_index('id')
D['dx'] = D.x - D.id.map(C.x); D['dy'] = D.y - D.id.map(C.y); D['ok'] = {'valid': D.valid & D.id.map(C.valid), 'nondiag': D.black_neighbors != 2, 'all': pd.Series(True, index=D.index)}[RULE]   # all: 755점 전부
X = D.pivot(index='file', columns='id', values='x'); OK = D.pivot(index='file', columns='id', values='ok')
TRIALS = ['Y 0.1 간격 +', 'Y 0.1 간격 -', 'Y 0.05 홀수 +', 'Y 0.05 홀수 -', 'Θ +', 'Θ -']

# (1) 한 칸 이동량
rows = []
for t in TRIALS:
    files = D[D.trial == t].drop_duplicates('file').sort_values('value', key=abs)[['file', 'value']].values
    sgn = np.sign(files[0][1]); prev = 'center.bmp'
    for i, (f, v) in enumerate(files):
        m = OK.loc[f].astype(bool) & OK.loc[prev].astype(bool)   # center도 같은 사용 점 규칙
        diff = (X.loc[f] - X.loc[prev])[m] * sgn
        rows.append(dict(trial=t, axis='Θ' if t[0] == 'Θ' else 'Y', step_from=prev[:-4], step_to=f[:-4], px=diff.mean(),
                         spread_px=diff.std(), n_points=int(m.sum()), kind='first' if i == 0 else 'step'))
        prev = f
S = pd.DataFrame(rows); S.to_csv(OUT / 'corner_step_px.csv', index=False, encoding='utf-8-sig')
st = S[S.kind == 'step']; g = st.groupby('axis').px.agg(['count', 'mean', 'std', 'min', 'max'])
st = st.join(g[['mean', 'std']], on='axis'); out = st[(st.px - st['mean']).abs() >= 3 * st['std']]
print('== 한 칸 이동량 (모서리) =='); print(g.round(4)); print(st.groupby('trial').px.agg(['mean', 'std']).round(4))
print('±3σ 밖:'); print(out[['trial', 'step_from', 'step_to', 'px']].round(3).to_string(index=False))
print('첫 칸:'); print(S[S.kind == 'first'][['step_to', 'px', 'n_points']].round(3).to_string(index=False))
print('한 칸 안 점들끼리 차이(σ) 중앙값:', st.groupby('axis').spread_px.median().round(4).to_dict())
# ECC 결과와 비교
E = pd.read_csv(HERE.parent / 'presentation_eda_20261019' / 'img' / 'step_px.csv')
cmp = S.merge(E[['step_from', 'step_to', 'px']].assign(step_from=lambda e: e.step_from.replace('center', 'center')), on=['step_from', 'step_to'], suffixes=('', '_ecc'))
cmp['diff'] = cmp.px - cmp.px_ecc; print('모서리 - ECC 차이: 평균', cmp['diff'].mean().round(4), ' 표준편차', cmp['diff'].std().round(4), ' 최대', cmp['diff'].abs().max().round(3))

# (2) 점별 흔들림
#   남은 값 = 이동량 - 점마다 2차 곡선. 이걸 '사진 공통'(그 사진에서 모든 점의 남은 값 평균 = 사진 전체가 같이 흔들린 양)과
#   '점 고유'(남은 값 - 사진 공통)로 나눔.  sigma_total = 점의 남은 값 σ, sigma_point = 점 고유 σ
R = []; PH = []
for t in TRIALS:
    T = D[D.trial == t]; good = T.groupby('id').ok.all(); ids = good[good].index
    T = T[T.id.isin(ids)]
    for comp in ('dx', 'dy'):
        M = T.pivot(index='file', columns='id', values=comp); v = T.drop_duplicates('file').set_index('file').value.loc[M.index].values
        fit = np.polyval(np.polyfit(v, M.values, 2), v[:, None]) if False else np.stack([np.polyval(np.polyfit(v, M[c].values, 2), v) for c in M.columns], 1)
        res = M.values - fit; common = res.mean(1, keepdims=True); own = res - common
        for j, cid in enumerate(M.columns):
            R.append(dict(trial=t, axis='Θ' if t[0] == 'Θ' else 'Y', id=cid, comp=comp, qr=int(T[T.id == cid].qr.iloc[0]),
                          sigma_total=res[:, j].std(ddof=3), sigma_point=own[:, j].std(ddof=3), n=len(v)))
        PH.append(dict(trial=t, comp=comp, sigma_common=common.std(ddof=3), n_points=len(M.columns)))
R = pd.DataFrame(R); R.to_csv(OUT / 'corner_point_sigma.csv', index=False, encoding='utf-8-sig')
PH = pd.DataFrame(PH); PH.to_csv(OUT / 'photo_common_sigma.csv', index=False, encoding='utf-8-sig')
q = R.groupby(['comp', 'trial']).agg(points=('id', 'size'), total_med=('sigma_total', 'median'), point_med=('sigma_point', 'median'),
                                     point_p90=('sigma_point', lambda x: x.quantile(.9)), point_max=('sigma_point', 'max')).round(4)
print('== 점별 흔들림 σ (px) =='); print(q); print('== 사진 공통 흔들림 σ (px) =='); print(PH.round(4).to_string(index=False))
