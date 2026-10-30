"""나머지 방법 실행 + 공통 평가.
방법 (모두 보정본 라벨로 학습, PC+GN 측정, 같은 짝수/홀수 폴드, 영역은 QR만 / QR+나사 둘 다):
  A 기준       : 패치 이동장 + 2차 매핑 + 대각 가중 (16조건 1등)
  B GLS        : A와 같은 매핑, 가중을 패치 잡음 공분산 전체(Ledoit-Wolf 수축 추정)로
  C ML         : 이동장 → 주성분 20개 → 판정 = 로지스틱 회귀, 값 = 가우시안 프로세스 (축별)
  D 물리 모델  : QR 코너 특징점 + 핀홀 카메라 투영. Y = 3D 평행 이동, Θ = 회전 중심 C 둘레 3D 회전.
                 판은 평면(역깊이 평면), 나사는 판보다 앞/뒤 깊이 1개씩. 파라미터 13~15개.
  E 판정 보류  : A~D 각각에 확신도 문턱을 걸어 애매한 사진은 보류
평가 (사진마다 축 + 값 기록 → 평균): 판정 정확도, 값 오차 평균|e| / RMS (축 맞힌 사진), 0.09 통과율.
채점 기준 2개: 보정 기준(실제 위치) / 미보정 기준(명령값)."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
import step4_8 as S
from step1_2 import HERE, PROJ
plt.rcParams['font.family'] = 'Malgun Gothic'; plt.rcParams['axes.unicode_minus'] = False
TXT, SUB, GRID = '#1f1f1e', '#5f5e58', '#e4e3dc'
NAMES, AXIS, VAL, FOLDA, CORR, U = S.NAMES, S.AXIS, S.VAL, S.FOLDA, S.CORR, S.U
DGN = S.F[1]                                   # PC+GN 이동장 (기준 2, 사진, 패치, 2)
AXES = ['Y', 'Θ']

def train_mask(k, a=None):
    tr = FOLDA != k
    return tr if a is None else tr & ((AXIS == a) | (AXIS == 'C'))

def rec(i, k, pa, pv, conf):
    return dict(file=NAMES[i], axis=AXIS[i], cmd=VAL[i], act=CORR[k][i], pred_axis=pa, pred=pv, conf=conf)

# ---------- A 기준 / B GLS ----------
def solve_white(d, coef, Linv, rows, huber=3.0, iters=6):
    s, c, b = coef[0][rows], coef[1][rows], coef[2][rows]; dd = d[rows]; Uw = Linv @ U[rows]; v = 0.0; ww = None
    for _ in range(iters):
        J = Linv @ (s + 2 * c * v); z = Linv @ (dd - (s * v + c * v * v + b))
        A = np.c_[J, Uw]; sw = np.ones(len(z)) if ww is None else np.sqrt(ww)
        p, *_ = np.linalg.lstsq(A * sw[:, None], z * sw, rcond=None); v += p[0]
        r = Linv @ (dd - (s * v + c * v * v + b)) - Uw @ p[1:]
        ww = np.where(np.abs(r) <= huber, 1, huber / np.maximum(np.abs(r), 1e-12))
    z = Linv @ (dd - (s * v + c * v * v + b)); sw = np.sqrt(ww)
    u, *_ = np.linalg.lstsq(Uw * sw[:, None], z * sw, rcond=None); rr = z - Uw @ u
    return v, np.sum(np.where(np.abs(rr) <= huber, rr**2, 2 * huber * np.abs(rr) - huber**2))

def method_map(region, gls):
    rows = S.mask_rows(region); out = []
    for k in (0, 1):
        D = DGN[k].reshape(len(NAMES), -1); maps, res = {}, []
        for a in AXES:
            m = train_mask(k, a); coef, var = S.fit_map(D[m], CORR[k][m], rows, quad=True); maps[a] = (coef, var)
            v = CORR[k][m]; res.append(D[m][:, rows] - (np.outer(v, coef[0][rows]) + np.outer(v**2, coef[1][rows]) + coef[2][rows]))
        if gls:
            Sig = LedoitWolf().fit(np.vstack(res)).covariance_
            Linv = np.linalg.inv(np.linalg.cholesky(Sig + 1e-10 * np.eye(len(Sig))))
        else:
            var = np.maximum((maps['Y'][1] + maps['Θ'][1]) / 2, 0.003**2)[rows]; Linv = np.diag(1 / np.sqrt(var))
        for i in np.where(FOLDA == k)[0]:
            o = {a: solve_white(D[i], maps[a][0], Linv, rows) for a in AXES}
            pa = min(o, key=lambda a: o[a][1]); conf = abs(np.log(o['Y'][1] / o['Θ'][1]))   # 두 가설 비용의 로그 비
            out.append(rec(i, k, pa, o[pa][0], conf))
    return pd.DataFrame(out)

# ---------- C ML ----------
def method_ml(region):
    rows = S.mask_rows(region); out = []
    for k in (0, 1):
        D = DGN[k].reshape(len(NAMES), -1)[:, rows]; tr = train_mask(k) & (AXIS != 'C')
        pca = PCA(n_components=20).fit(D[train_mask(k)]); sc = StandardScaler().fit(pca.transform(D[tr]))
        Zf = lambda X: sc.transform(pca.transform(X))
        clf = LogisticRegression(C=10.0, max_iter=5000).fit(Zf(D[tr]), AXIS[tr])
        reg = {}
        for a in AXES:
            m = train_mask(k, a); ker = ConstantKernel(1.0) * RBF(length_scale=np.ones(20) * 3.0, length_scale_bounds=(1e-2, 1e3)) + WhiteKernel(1e-4, (1e-8, 1e-1))
            reg[a] = GaussianProcessRegressor(ker, normalize_y=True, n_restarts_optimizer=1, random_state=0).fit(Zf(D[m]), CORR[k][m])
        te = np.where(FOLDA == k)[0]; Zt = Zf(D[te]); pr = clf.predict_proba(Zt); cls = list(clf.classes_)
        for j, i in enumerate(te):
            pa = cls[int(np.argmax(pr[j]))]; out.append(rec(i, k, pa, float(reg[pa].predict(Zt[j:j + 1])[0]), abs(np.log(pr[j].max() / pr[j].min()))))
    return pd.DataFrame(out)

# ---------- D 물리 모델 ----------
FT = pd.read_csv(HERE/'features.csv'); FF = np.load(HERE/'feat_field.npz')['F'].astype(np.float64)   # 기준 2, 사진, 점, 2
assert list(np.load(HERE/'feat_field.npz')['names']) == NAMES
CXI, CYI = 1920.0, 1080.0
def model_disp(th, pts_u, pts_v, scr, axis, vs):
    """th = [log f, p1, p2, s_L, s_R, Tx, Ty, Tz, wx, wy, wz, Cx, Cy, Cz]. vs: 값 배열 (n). 반환: n x 점 x 2 화면 이동.
    판 = 역깊이 평면 1/Z = 1 + p1 x + p2 y (+ 나사 깊이 보정), Y = 평행 이동 v*T, Θ = 점 C 둘레 회전 (회전 벡터 v*w, 로드리게스)."""
    f = np.exp(th[0]); xn, yn = (pts_u - CXI) / f, (pts_v - CYI) / f
    invz = 1 + th[1] * xn + th[2] * yn + th[3] * (scr == 1) + th[4] * (scr == 2); Z = 1 / invz
    P = np.c_[xn * Z, yn * Z, Z]; vs = np.atleast_1d(np.asarray(vs, float))
    if axis == 'Y':
        Q = P[None] + vs[:, None, None] * th[5:8][None, None]
    else:
        C = th[11:14]; w = th[8:11]; ang = np.linalg.norm(w) * vs; kax = w / (np.linalg.norm(w) + 1e-15); R0 = P - C
        cs, sn = np.cos(ang)[:, None, None], np.sin(ang)[:, None, None]
        kxr = np.cross(kax, R0)[None]; kdr = (R0 @ kax)[None, :, None]
        Q = R0[None] * cs + kxr * sn + kax[None, None] * kdr * (1 - cs) + C
    return np.stack([f * Q[..., 0] / Q[..., 2] + CXI - pts_u, f * Q[..., 1] / Q[..., 2] + CYI - pts_v], -1)

def nuis_basis(u, v):
    uc, vc = u - u.mean(), v - v.mean()
    return np.stack([np.c_[np.zeros_like(u), np.ones_like(u)].ravel(), np.c_[-vc, uc].ravel() / 100], 1)

def method_phys(region):
    """0점 무늬 = center 사진을 같은 기준 영상에 대해 잰 이동장 (center가 0 자세라는 정의에서 나옴)."""
    sel = FT.region.str.startswith('QR').values if region == 'QR' else np.ones(len(FT), bool)
    u, v = FT.cx.values[sel], FT.cy.values[sel]; scr = np.where(FT.region.values[sel] == '나사좌', 1, np.where(FT.region.values[sel] == '나사우', 2, 0))
    Ub = nuis_basis(u, v); Pn = np.eye(len(Ub)) - Ub @ np.linalg.pinv(Ub)       # 방해 무늬(세로 이동, 화면 회전)를 빼는 투영
    ic = NAMES.index('center.bmp'); out, info = [], []
    for k in (0, 1):
        Fk = FF[k][:, sel]; b0 = Fk[ic]; lab = CORR[k]
        idx = {a: np.where(train_mask(k) & (AXIS == a))[0] for a in AXES}
        def resid(th):
            return np.concatenate([((Fk[idx[a]] - b0 - model_disp(th, u, v, scr, a, lab[idx[a]])).reshape(len(idx[a]), -1) @ Pn.T).ravel() for a in AXES])
        th0 = np.array([np.log(1e4), 0, 0, 0, 0, 14.4e-4, 0, 0, 0, -0.016, 0, 0, 0, 1.5])
        fit = least_squares(resid, th0, x_scale='jac', max_nfev=3000, method='trf', xtol=1e-12, ftol=1e-12)
        th = fit.x; rr = fit.fun.reshape(-1, Pn.shape[0]); var = np.maximum(rr.var(0), 0.003**2); w = 1 / np.sqrt(var)
        info.append(dict(fold=k, region=region, f=np.exp(th[0]), T=th[5:8].round(6).tolist(), w=th[8:11].round(6).tolist(), C=th[11:14].round(4).tolist(),
                         screw_invz=th[3:5].round(5).tolist(), rms_px=np.sqrt((rr**2).mean()), nfev=fit.nfev))
        for i in np.where(FOLDA == k)[0]:
            o = {}
            for a in AXES:
                g = np.linspace(-2.2, 2.2, 221) if a == 'Y' else np.linspace(-1.1, 1.1, 221)
                E = ((Fk[i] - b0)[None] - model_disp(th, u, v, scr, a, g)).reshape(len(g), -1) @ Pn.T
                v0 = g[np.argmin(((E * w)**2).sum(1))]
                r = least_squares(lambda x: (Pn @ ((Fk[i] - b0) - model_disp(th, u, v, scr, a, x)[0]).ravel()) * w, [v0], loss='huber', f_scale=3.0)
                o[a] = (r.x[0], 2 * r.cost)
            pa = min(o, key=lambda a: o[a][1]); out.append(rec(i, k, pa, o[pa][0], abs(np.log(o['Y'][1] / o['Θ'][1]))))
    return pd.DataFrame(out), info

# ---------- 평가 ----------
def evaluate(df, basis):
    t = df[basis].values; ok = (df.pred_axis == df.axis).values; e = df.pred.values - t
    y, th = (df.axis == 'Y').values, (df.axis == 'Θ').values
    passed = np.where(ok, np.abs(e) <= 0.09, (np.abs(t) <= 0.09) & (np.abs(df.pred.values) <= 0.09))
    f = lambda m, sc: (np.mean(np.abs(e[m & ok])) * sc, np.sqrt(np.mean(e[m & ok]**2)) * sc)
    (ya, yr), (ta, tr) = f(y, 1000), f(th, 1000)
    return dict(판정정확도=ok.mean() * 100, Y판정정확도=ok[y].mean() * 100, Θ판정정확도=ok[th].mean() * 100,
                Y평균오차_um=ya, Y_RMS_um=yr, Θ평균오차_mΘ=ta, Θ_RMS_mΘ=tr, 통과09=passed.mean() * 100)

if __name__ == '__main__':
    PRED, INFO = {}, []
    for reg in ['QR', 'QR+나사']:
        PRED[('A 기준 (2차, 대각 가중)', reg)] = method_map(reg, gls=False); print('A', reg, flush=True)
        PRED[('B GLS (잡음 공분산 가중)', reg)] = method_map(reg, gls=True); print('B', reg, flush=True)
        PRED[('C ML (주성분 + 로지스틱 + GP)', reg)] = method_ml(reg); print('C', reg, flush=True)
        df, info = method_phys(reg); PRED[('D 물리 모델 (특징점 + 투영)', reg)] = df; INFO += info; print('D', reg, info, flush=True)
    rows = []
    for (m, reg), df in PRED.items():
        for basis, bn in [('act', '보정 기준'), ('cmd', '미보정 기준')]:
            rows.append(dict(방법=m, 영역=reg, 채점=bn, 사진수=len(df), **evaluate(df, basis)))
    # E 판정 보류: 확신도 하위 q% 를 보류
    hold = []
    for (m, reg), df in PRED.items():
        for q in [0, 2, 5, 10, 15, 20]:
            thr = np.percentile(df.conf, q) if q else -1; dd = df[df.conf > thr]
            for basis, bn in [('act', '보정 기준'), ('cmd', '미보정 기준')]:
                hold.append(dict(방법=m, 영역=reg, 채점=bn, 보류율=100 * (1 - len(dd) / len(df)), **evaluate(dd, basis)))
    T = pd.DataFrame(rows); H = pd.DataFrame(hold)
    T.round(3).to_csv(PROJ/'42_결과표_나머지방법_평가.csv', index=False, encoding='utf-8-sig')
    H.round(3).to_csv(PROJ/'43_결과표_판정보류.csv', index=False, encoding='utf-8-sig')
    pd.concat([d.assign(방법=m, 영역=r) for (m, r), d in PRED.items()]).to_csv(PROJ/'44_사진별_판정기록.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(INFO).to_csv(HERE/'phys_info.csv', index=False, encoding='utf-8-sig')
    with pd.option_context('display.width', 250, 'display.max_columns', 30): print(T.round(2).to_string())

    # ---------- 그림 45: 방법 x 영역 x 채점 요약 ----------
    cols = [('판정정확도', '판정 정확도 %', '.1f', 1), ('Y평균오차_um', 'Y 평균 오차 µm', '.1f', -1), ('Y_RMS_um', 'Y RMS µm', '.1f', -1),
            ('Θ평균오차_mΘ', 'Θ 평균 오차 mΘ', '.2f', -1), ('Θ_RMS_mΘ', 'Θ RMS mΘ', '.2f', -1), ('통과09', '0.09 통과 %', '.1f', 1)]
    fig, axs = plt.subplots(1, 2, figsize=(24, 6.5))
    for ax, bn in zip(axs, ['보정 기준', '미보정 기준']):
        t = T[T.채점 == bn].reset_index(drop=True); M = t[[c for c, *_ in cols]].values.astype(float); Mn = np.zeros_like(M)
        for j, (_, _, _, sgn) in enumerate(cols):
            z = (M[:, j] - M[:, j].min()) / (np.ptp(M[:, j]) + 1e-12); Mn[:, j] = z if sgn < 0 else 1 - z
        ax.imshow(Mn, cmap='RdYlGn_r', vmin=0, vmax=1, aspect='auto')
        for i in range(len(t)):
            for j, (_, _, f, _) in enumerate(cols): ax.text(j, i, format(M[i, j], f), ha='center', va='center', fontsize=12)
        ax.set_xticks(range(len(cols))); ax.set_xticklabels([l for _, l, _, _ in cols], fontsize=11)
        ax.set_yticks(range(len(t))); ax.set_yticklabels([f'{a} / {b}' for a, b in zip(t.방법, t.영역)], fontsize=11)
        ax.set_title(f'{bn} ({"백래시, 드리프트를 뺀 실제 위치" if bn == "보정 기준" else "파일 이름의 명령값"})', loc='left', fontsize=14)
    fig.suptitle('나머지 방법 평가: 시험 사진마다 Y인지 Θ인지 + 값을 기록하고 평균 (짝수/홀수 교차검증, 열마다 초록 = 좋음, 오차는 축을 맞힌 사진만)',
                 x=0.01, ha='left', fontsize=15)
    plt.tight_layout(rect=(0, 0, 1, 0.94)); plt.savefig(PROJ/'45_나머지방법_평가.png', dpi=105, facecolor='white'); plt.close()

    # ---------- 그림 46: 판정 보류 곡선 + 어디서 틀리나 ----------
    fig, axs = plt.subplots(1, 3, figsize=(24, 6))
    mk = {'QR': '-', 'QR+나사': '--'}; colors = dict(zip(sorted({m for m, _ in PRED}), ['#4a72d0', '#d2703f', '#5aa776', '#8a63c4']))
    for (m, reg), _ in PRED.items():
        h = H[(H.방법 == m) & (H.영역 == reg) & (H.채점 == '보정 기준')]
        axs[0].plot(h.보류율, h.판정정확도, mk[reg], marker='o', color=colors[m], label=f'{m} / {reg}')
        axs[1].plot(h.보류율, h.통과09, mk[reg], marker='o', color=colors[m])
    for ax, ttl in [(axs[0], '판정 보류율에 따른 판정 정확도 (보정 기준)'), (axs[1], '판정 보류율에 따른 0.09 통과율 (보정 기준)')]:
        ax.set_xlabel('보류한 사진 비율 (%)'); ax.set_ylabel('%'); ax.grid(color=GRID); ax.set_title(ttl, loc='left', fontsize=13)
        for sp in ['top', 'right']: ax.spines[sp].set_visible(False)
    axs[0].legend(fontsize=9, frameon=False)
    ax = axs[2]; bins = [0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.01]; lab = ['~0.05', '~0.1', '~0.2', '~0.3', '~0.5', '~1']
    for (m, reg), df in PRED.items():
        if reg != 'QR': continue
        acc = [((df.pred_axis == df.axis)[(df.act.abs() >= lo) & (df.act.abs() < hi)]).mean() * 100 for lo, hi in zip(bins[:-1], bins[1:])]
        ax.plot(range(len(lab)), acc, marker='o', color=colors[m], label=m)
    ax.set_xticks(range(len(lab))); ax.set_xticklabels(lab); ax.set_xlabel('실제 이동량 |Y| (mm) 또는 |Θ| 구간'); ax.set_ylabel('판정 정확도 %')
    ax.set_title('이동량 구간별 판정 정확도 (QR만)', loc='left', fontsize=13); ax.grid(color=GRID); ax.legend(fontsize=9, frameon=False)
    for sp in ['top', 'right']: ax.spines[sp].set_visible(False)
    plt.tight_layout(); plt.savefig(PROJ/'46_판정보류_구간별정확도.png', dpi=105, facecolor='white'); plt.close()
    print('saved')
