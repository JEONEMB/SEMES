# 독립 재검증 : C++ 결과 CSV 를 다른 코드(numpy)로 다시 계산해 대조
import csv, re, math, json, sys
import numpy as np
D = sys.argv[1]
def rd(p): return list(csv.DictReader(open(p, encoding='utf-8-sig')))
out = {}
# --- 1. 라벨 독립 파싱
def label(stem):
    m = re.fullmatch(r'Y([+-]\d+\.?\d*) T([+-]\d+\.?\d*) \((\d)\)', stem)
    if m: return float(m[1]), float(m[2]), 2
    m = re.fullmatch(r'Y([+-]\d+\.\d+)', stem)
    if m: return float(m[1]), 0.0, 1
    m = re.fullmatch(r'T([+-]\d+\.\d+)', stem)
    if m: return 0.0, float(m[1]), 1
    return None
bad = 0
for f in ['pred_set2_selected.csv', 'pred_train_cv.csv', 'pred_second_batch.csv']:
    for r in rd(f'{D}/{f}'):
        y, t, st = label(r['stem'])
        if abs(y - float(r['y_mm'])) > 1e-9 or abs(t - float(r['t_deg'])) > 1e-9: bad += 1
out['label_mismatch'] = bad
# --- 2. 학습/시험 분리
tr = rd(f'{D}/pred_train_cv.csv')
first = lambda y, t: t != 0 or abs(y * 10 - round(y * 10)) < 1e-6
out['train_n'] = len(tr)
out['train_all_set1_first'] = all(label(r['stem'])[2] == 1 and first(*label(r['stem'])[:2]) for r in tr)
te = rd(f'{D}/pred_set2_selected.csv')
out['test_all_set2'] = all(label(r['stem'])[2] == 2 for r in te) and len(te) == 57
out['overlap_train_test'] = len({r['stem'] for r in tr} & {r['stem'] for r in te})
# --- 3. 독립 재학습 : 측정 CSV 에서 특징을 직접 만들어 numpy 최소제곱
M1 = {r['stem']: r for r in rd(f'{D}/roi_meas_set1.csv')}
sel = ['wall_W2', 'wall_W3']
rows = []
for stem, r in M1.items():
    l = label(stem)
    if not l or l[2] != 1 or not first(l[0], l[1]): continue
    s = float(r['s']); d = s - np.mean([float(r[k + '_dx']) for k in sel])
    rows.append((stem, l[0], l[1], s, d))
A = np.array([[1, y, t] for _, y, t, _, _ in rows]); F = np.array([[s, d] for *_, s, d in rows])
coef = np.linalg.lstsq(A, F, rcond=None)[0]           # 3 × 2
res = F - A @ coef; sig = res.std(axis=0, ddof=3)
out['coef_s'] = coef[:, 0].round(6).tolist(); out['coef_d'] = coef[:, 1].round(6).tolist(); out['sigma'] = sig.round(6).tolist()
# --- 4. 독립 역산 (선형 모델이면 가중 최소제곱 해가 닫힌 꼴) + 2차 예측 비교
def invert(f, coef, sig):
    J = np.array([[coef[1, k], coef[2, k]] for k in range(2)]); W = np.diag(1 / sig ** 2)
    r = f - coef[0]
    return np.linalg.solve(J.T @ W @ J, J.T @ W @ r)
M2 = rd(f'{D}/roi_meas_set2.csv'); P2 = {r['stem']: r for r in te}
maxdiff = 0; groups = {}
for r in M2:
    s = float(r['s']); d = s - np.mean([float(r[k + '_dx']) for k in sel])
    y, t = invert(np.array([s, d]), coef, sig)
    p = P2[r['stem']]
    maxdiff = max(maxdiff, abs(y - float(p['y_est'])) * 1000, abs(t - float(p['t_est'])) * 100)
    ly, lt, _ = label(r['stem']); ok = abs(y - ly) * 1000 <= 90 and abs(t - lt) <= 0.01
    g = 'near' if lt <= 0.2 + 1e-9 else ('t1' if abs(lt - 1) < 1e-6 else 't2')
    groups.setdefault(g, []).append((ok, (t - lt), (y - ly) * 1000))
out['set2_pred_maxdiff_um_or_centideg'] = float(maxdiff)
out['set2_groups'] = {g: {'pass': int(sum(o for o, _, _ in v)), 'n': len(v),
                          'theta_rms': float(np.sqrt(np.mean([e ** 2 for _, e, _ in v]))),
                          'y_rms_um': float(np.sqrt(np.mean([e ** 2 for _, _, e in v])))} for g, v in groups.items()}
# --- 5. 독립 블록 교차검증 (다른 코드, 같은 규칙 : Y 4 묶음, Θ 10 묶음)
idx_y = sorted([i for i, r in enumerate(rows) if r[2] == 0], key=lambda i: rows[i][1])
idx_t = sorted([i for i, r in enumerate(rows) if r[2] != 0], key=lambda i: rows[i][2])
fold = np.zeros(len(rows), int)
for j, i in enumerate(idx_y): fold[i] = j * 4 // len(idx_y)
for j, i in enumerate(idx_t): fold[i] = 4 + j * 10 // len(idx_t)
errs = []
for f in range(14):
    trm = fold != f
    c = np.linalg.lstsq(A[trm], F[trm], rcond=None)[0]; sg = (F[trm] - A[trm] @ c).std(axis=0, ddof=3)
    for i in np.where(~trm)[0]:
        y, t = invert(F[i], c, sg); errs.append(((y - rows[i][1]) * 1000, t - rows[i][2]))
E = np.array(errs)
out['cv_J'] = float(np.sqrt(np.mean((E[:, 0] / 90) ** 2 + (E[:, 1] / 0.01) ** 2)))
out['cv_theta_rms'] = float(np.sqrt(np.mean(E[:, 1] ** 2))); out['cv_y_rms'] = float(np.sqrt(np.mean(E[:, 0] ** 2)))
print(json.dumps(out, ensure_ascii=False, indent=1))
