# 같은 Y/Θ 판정 방법에 측정값만 바꿔 넣어 판정 정확도 비교
#   판정 방법: codex_experiments_20261014/zero_experiment.py 의 fit/predict (mode='center', 2차)
#     = center 고정, 원점 통과 Y/T 2차 곡선, 학습 잔차 공분산으로 'Y만'/'Θ만' 가설 비교
#       (ALGORITHM_RULES.md 기준, center_fixed_20261015/run.py 와 같은 코드)
#   측정값 (사진마다 특징 벡터 X):
#     ref205_P12   : 기존 205점 추적 + 판 전체 2차 변형 12계수 (center_fixed_20261015 와 같음, 같은 코드로 다시 계산)
#     ecc_affine18 : 기존 ECC, QR 3개 아핀 (이동 2 + 변형 4) x 3 = 18 (analysis_agent/qr_measurements.csv)
#     모서리 3가지 규칙 x 2가지 요약 (analysis_grid_corners_20261019/out/corners_280.csv)
#       규칙 all = 755점 전부, nondiag = 대각선 모서리 제외 630점, valid = 진단 통과 점만
#       요약 P12    : 점 이동을 판 전체 2차 변형 12계수로 (valid 는 사진마다 통과 점만으로 맞춤)
#       요약 direct : 점 이동(x, y)을 그대로 (valid 는 281장 모두 통과한 점만 = 고정 집합)
#     참고 _diagcov : direct 와 같지만 공분산을 대각(점끼리 독립)으로만 씀
#                     (점 수가 학습 사진 수보다 훨씬 많아 전체 공분산이 불안정할 때 비교용)
#   분할: gls_comparison_20261014/splits.json 의 저장 분할 그대로 (재대입 1, 짝홀 2겹, 반복 5분할 20회)
import sys, json, pathlib, time, numpy as np, pandas as pd
HERE = pathlib.Path(__file__).parent; PROJ = HERE.parent; BASE = PROJ / 'codex_experiments_20261014'
sys.path[:0] = [str(BASE), str(BASE / 'measurement')]
import zero_experiment as Z

C = pd.read_csv(PROJ / 'analysis_grid_corners_20261019' / 'out' / 'corners_280.csv')
names = np.array(sorted(C.file.unique(), key=lambda n: (n != 'center.bmp', n)))
m = Z.manifest(names); index = {n: i for i, n in enumerate(names)}


def p12(px, py, dx, dy, w):
    # 판 전체 2차 변형: 기저 [1, x, y, x^2, xy, y^2] (center_fixed_20261015/run.py 와 같은 정규화), x/y 이동 각각 6계수
    x = (px - 1906) / 486; y = (py - 1070) / 122; B = np.c_[x * 0 + 1, x, y, x * x, x * y, y * y]
    sw = np.sqrt(w)[:, None]
    cx = np.linalg.lstsq(B * sw, dx * sw[:, 0], rcond=None)[0]; cy = np.linalg.lstsq(B * sw, dy * sw[:, 0], rcond=None)[0]
    return np.c_[cx, cy].ravel()   # 계수마다 (x, y) 순서, run.py 와 같음


FEAT = {}
# 기존 205점 P12
z = np.load(BASE / 'center_fixed_20261015' / 'center_once_tracks.npz'); zn = list(z['names'].astype(str)); pts = z['points']; D205 = z['D'].astype(float)
FEAT['ref205_P12'] = np.stack([p12(pts[:, 0], pts[:, 1], D205[zn.index(n)][:, 0], D205[zn.index(n)][:, 1], np.ones(len(pts))) for n in names])
# ECC 아핀 18
E = pd.read_csv(PROJ / 'analysis_agent' / 'qr_measurements.csv').set_index('file').loc[names]
cols = [f'q{q}_{k}' for q in range(3) for k in ('dx', 'dy', 'a00', 'a01', 'a10', 'a11')]
Xe = E[cols].to_numpy(float).copy()
for q in range(3): Xe[:, 6 * q + 2] -= 1; Xe[:, 6 * q + 5] -= 1   # 단위 행렬에서 벗어난 양
FEAT['ecc_affine18'] = Xe
# 모서리
Xp = C.pivot(index='file', columns='id', values='x').loc[names]; Yp = C.pivot(index='file', columns='id', values='y').loc[names]
V = C.pivot(index='file', columns='id', values='valid').loc[names].astype(bool).to_numpy()
BN = C[C.file == 'center.bmp'].set_index('id').black_neighbors.loc[Xp.columns].to_numpy()
cx0, cy0 = Xp.loc['center.bmp'].to_numpy(), Yp.loc['center.bmp'].to_numpy()
DX, DY = Xp.to_numpy() - cx0, Yp.to_numpy() - cy0
ic = index['center.bmp']
rules = {'all': np.ones(V.shape, bool), 'nondiag': np.broadcast_to(BN != 2, V.shape).copy(), 'valid': V & V[ic]}
fixed = {'all': np.ones(len(BN), bool), 'nondiag': BN != 2, 'valid': V.all(0)}
for r, M in rules.items():
    FEAT[f'corner_{r}_P12'] = np.stack([p12(cx0, cy0, DX[i], DY[i], M[i].astype(float)) for i in range(len(names))])
    k = fixed[r]; FEAT[f'corner_{r}_direct'] = np.c_[DX[:, k], DY[:, k]]
print({k: v.shape for k, v in FEAT.items()}, flush=True)

folds = json.loads((BASE / 'gls_comparison_20261014' / 'splits.json').read_text(encoding='utf-8'))
ids = np.flatnonzero(m.axis != 'C')
SPLITS = [dict(protocol='full280_resubstitution', fold='all', train=names[ids].tolist(), test=names[ids].tolist())] + \
         [f for f in folds if f['protocol'] == 'interleaved' or f['protocol'].startswith('repeated5_')]

preds = []; t0 = time.time()
runs = [(k, False) for k in FEAT] + [(k, True) for k in FEAT if k.endswith('_direct')]
for key, diag in runs:
    X = FEAT[key]; name = key + ('_diagcov' if diag else '')
    for ss in SPLITS:
        tr = np.array([index[n] for n in ss['train']]); te = np.array([index[n] for n in ss['test']])
        model = Z.fit(m, X, tr, m.value.to_numpy(), 'center', True)
        if diag:   # 같은 학습 잔차에서 분산만 쓰고 점끼리 상관은 버림
            branches, _ = model; res = []
            for a in ['Y', 'T']:
                ii = tr[m.axis.to_numpy()[tr] == a]; v = m.value.to_numpy()[ii]; _, base, s, c = branches[a]
                res.append(X[ii] - base - v[:, None] * s - (v * v)[:, None] * c)
            sd = np.sqrt(np.maximum(np.vstack(res).var(0, ddof=1), 1e-14)); model = (branches, np.diag(1 / sd))
        df = Z.predict(m, X, te, model, 'center'); df['method'] = name; df['protocol'] = ss['protocol']; df['fold'] = ss['fold']; preds.append(df)
    print('done', name, round(time.time() - t0), 's', flush=True)
P = pd.concat(preds, ignore_index=True); P.to_csv(HERE / 'predictions.csv', index=False, encoding='utf-8-sig')

rows = []
for (meth, prot), df in P.groupby(['method', 'protocol']): rows.append(dict(method=meth, protocol=prot, **Z.score(df)))
S = pd.DataFrame(rows); S.to_csv(HERE / 'metrics_by_protocol.csv', index=False, encoding='utf-8-sig')
# 요약: 재대입 / 짝홀 / 반복 5분할 20회 (20회를 모은 정확도, 20회 중 축과 부호 전부 맞은 횟수, 회당 평균 틀린 장수, 최대오차)
out = []
for meth, g in S.groupby('method'):
    rs = g[g.protocol == 'full280_resubstitution'].iloc[0]; il = g[g.protocol == 'interleaved'].iloc[0]; rp = g[g.protocol.str.startswith('repeated5_')]
    rep = Z.score(P[(P.method == meth) & P.protocol.str.startswith('repeated5_')])
    out.append(dict(method=meth, dims=FEAT[meth.replace('_diagcov', '')].shape[1],
                    resub_Y=rs.Y_axis_sign_accuracy, resub_T=rs.T_axis_sign_accuracy, resub_Ymax=rs.vector_Y_max, resub_Tmax=rs.vector_T_max,
                    evenodd_Y=il.Y_axis_sign_accuracy, evenodd_T=il.T_axis_sign_accuracy, evenodd_Ymax=il.vector_Y_max, evenodd_Tmax=il.vector_T_max,
                    rep_Y=rep['Y_axis_sign_accuracy'], rep_T=rep['T_axis_sign_accuracy'], rep_Ymax=rep['vector_Y_max'], rep_Tmax=rep['vector_T_max'],
                    rep_all_correct=int(((rp.Y_axis_sign_accuracy == 1) & (rp.T_axis_sign_accuracy == 1)).sum()),
                    rep_wrong_per_round=float(((1 - rp.Y_axis_sign_accuracy) * 80 + (1 - rp.T_axis_sign_accuracy) * 200).mean())))
O = pd.DataFrame(out); O.to_csv(HERE / 'summary.csv', index=False, encoding='utf-8-sig')
pd.set_option('display.width', 250); print(O.round(4).to_string(index=False))
F = P[(P.protocol == 'interleaved') & ((P.axis != P.pred_axis) | (np.sign(P.value) != np.sign(P.pred)))]
F.to_csv(HERE / 'failures_evenodd.csv', index=False, encoding='utf-8-sig')
print(F.groupby('method').file.apply(lambda s: ' '.join(sorted(s))).to_string())
