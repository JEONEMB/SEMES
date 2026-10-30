"""v7 공통 틀: 데이터(QR 판 특징점 이동), 폴드, 라벨, 계산 방법 M1~M3, 판정 문턱, 평가.
런타임 가정: center 사진 + 현재 사진 + 미리 학습한 모델. 점 이동 D는 center 템플릿 기준 (center에서 0).
판밖 점은 쓰지 않는다."""
import sys, io, contextlib
from pathlib import Path
import numpy as np, pandas as pd, cv2
HERE = Path(__file__).parent; PROJ = HERE.parent; V6 = PROJ/'analysis_v6'; OUT = HERE/'out'   # 결과는 이 폴더 안 out/ 에만
sys.path.insert(0, str(V6)); sys.path.insert(0, str(PROJ/'analysis_v5'))
from common import NAMES, AXIS, VAL, session_of, X0, X1, Y0, Y1
with contextlib.redirect_stdout(io.StringIO()):
    import step4_8 as V5

AX = np.array([AXIS[n] for n in NAMES]); CMD = np.array([VAL[n] for n in NAMES]); SES = np.array([session_of(n) for n in NAMES])
ISC = AX == 'C'
CORR = {k: np.array([V5.CORR[k][V5.NAMES.index(n)] for n in NAMES]) for k in (0, 1)}   # 0점 보정본 (폴드 k 학습 사진으로만 맞춘 v4 ①)

def load(hp, screws=False):
    F = pd.read_csv(V6/f'features_qr_hp{hp}.csv'); Z = np.load(V6/f'tracks_qr_hp{hp}.npz')
    assert list(Z['names']) == NAMES
    m = np.ones(len(F), bool) if screws else F.region.str.startswith('QR').values
    D = Z['D_gn'][:, m].astype(np.float64); D = D - D[ISC]          # center 기준 (center에서 정확히 0)
    return F[m].reset_index(drop=True), D

def folds(kind):
    """(이름, 시험 마스크, 보정본 폴드 번호). center는 항상 학습 쪽."""
    if kind == '짝홀':
        return [(f'f{k}', np.array([V5.FOLD.get(n, -1) == k for n in NAMES]), k) for k in (0, 1)]
    return [(s, SES == s, 0) for s in ['Y1+', 'Y1-', 'Y2+', 'Y2-', 'Θ+', 'Θ-']]

# ---------------- 특징 만들기 ----------------
def warp_features(D, F, kind):
    """사진마다 점 이동장을 판 전체 변형 값으로 요약. kind 'H8' = 원근 변환 (판 네 귀퉁이 이동 8개), 'P12' = 2차 다항 변형 12개."""
    x = (F.x.values - 1906) / 486; y = (F.y.values - 1070) / 122               # 판 영역 정규화
    if kind == 'H8':
        src = np.c_[F.x.values, F.y.values].astype(np.float64); corners = np.array([[X0, Y0], [X1, Y0], [X1, Y1], [X0, Y1]], np.float64)
        out = []
        for d in D:
            H, _ = cv2.findHomography(src, src + d, 0); c = cv2.perspectiveTransform(corners[None], H)[0] - corners; out.append(c.ravel())
        return np.array(out)
    if kind == 'P12x3':                                                         # QR마다 따로 2차 변형 (36개)
        return np.hstack([warp_features(D[:, g], F[g].reset_index(drop=True), 'P12') for g in [F.region.values == r for r in ['QR좌', 'QR중', 'QR우']]])
    B = [np.ones_like(x), x, y, x * x, x * y, y * y]
    if kind == 'P20': B += [x**3, x * x * y, x * y * y, y**3]                   # 3차 (렌즈 왜곡 포함)
    B = np.stack(B, 1); nb = B.shape[1]
    co, *_ = np.linalg.lstsq(B, D.transpose(1, 0, 2).reshape(len(x), -1), rcond=None)       # (nb, 사진*2)
    return co.reshape(nb, len(D), 2).transpose(1, 0, 2).reshape(len(D), 2 * nb)

# ---------------- 모델 ----------------
class Model:
    """축마다 f = b + s v (+ c v²). f: 사진마다 특징 벡터 (점 이동 2N개 또는 판 변형 값).
    잡음: 'diag' 점마다 분산 / 'equal' 같은 가중 / 'full' 전체 공분산 (저차원 특징만).
    절편: 'free' 학습 / 'zero' center 고정 (center = 0점)."""
    def __init__(self, quad=True, noise='diag', intercept='free', huber=3.0, nuis=None, deg=None, local=None):
        self.deg = deg or (2 if quad else 1); self.quad = self.deg >= 2
        self.noise, self.intercept, self.huber, self.U, self.local = noise, intercept, huber, nuis, local
    def _A(self, v):
        cols = [v**k for k in range(1, self.deg + 1)] + ([np.ones_like(v)] if self.intercept == 'free' else [])
        return np.stack(cols, 1)
    def fit(self, Fe, lab, ax):
        self.co, self.poly, res = {}, {}, []
        for a in ['Y', 'Θ']:
            m = (ax == a) | (ax == 'C')
            if self.local: m = m & (np.abs(lab) <= self.local[a])          # 0 근처 사진만으로 학습
            A = self._A(lab[m]); co, *_ = np.linalg.lstsq(A, Fe[m], rcond=None)
            b = co[-1] if self.intercept == 'free' else np.zeros_like(co[0])
            self.poly[a] = (list(co[:self.deg]), b)
            self.co[a] = (co[0], co[1] if self.deg >= 2 else np.zeros_like(co[0]), b); res.append(Fe[m & (ax != 'C')] - A[ax[m] != 'C'] @ co)
        R = np.vstack(res); dof = max(len(R) - 6, 1)
        if self.noise == 'full':
            C = R.T @ R / dof + 1e-10 * np.eye(R.shape[1]); self.W = np.linalg.cholesky(np.linalg.inv(C)).T     # 화이트닝
        else:
            var = R.var(0) if self.noise == 'diag' else np.full(R.shape[1], R.var())
            self.W = np.diag(1 / np.sqrt(np.maximum(var, 0.003**2))) if R.shape[1] < 60 else 1 / np.sqrt(np.maximum(var, 0.003**2))
        return self
    def _w(self, z):
        return self.W @ z if self.W.ndim == 2 else self.W * z
    def solve(self, f, a, iters=8):
        P, b = self.poly[a]; v = 0.0; U = self.U if self.U is not None else np.zeros((len(f), 0))
        fv = lambda v: sum(p * v**(k + 1) for k, p in enumerate(P)) + b
        dv = lambda v: sum((k + 1) * p * v**k for k, p in enumerate(P))
        hw = np.ones(len(f)) if self.W.ndim == 1 else None
        for _ in range(iters):
            J = dv(v); z = f - fv(v); A = np.c_[J, U]
            if self.W.ndim == 1:
                sw = self.W * np.sqrt(hw); p, *_ = np.linalg.lstsq(A * sw[:, None], z * sw, rcond=None)
            else:
                p, *_ = np.linalg.lstsq(self.W @ A, self.W @ z, rcond=None)
            v += p[0]
            if self.W.ndim == 1 and self.huber:
                r = (f - fv(v) - U @ p[1:]) * self.W; hw = np.where(np.abs(r) <= self.huber, 1, self.huber / np.maximum(np.abs(r), 1e-12))
        z = f - fv(v)
        if U.shape[1]:
            if self.W.ndim == 1:
                sw = self.W * np.sqrt(hw); u, *_ = np.linalg.lstsq(U * sw[:, None], z * sw, rcond=None)
            else:
                u, *_ = np.linalg.lstsq(self.W @ U, self.W @ z, rcond=None)
            z = z - U @ u
        r = self._w(z)
        if self.W.ndim == 1 and self.huber:
            h = self.huber; cost = np.sum(np.where(np.abs(r) <= h, r**2, 2 * h * np.abs(r) - h * h))
        else: cost = np.sum(r**2)
        return v, cost
    def predict(self, f):
        (vy, cy), (vt, ct) = self.solve(f, 'Y'), self.solve(f, 'Θ')
        return dict(vY=vy, vT=vt, llr=np.log(cy / ct))      # llr < 문턱 → Y

class Joint(Model):
    """M2: Y, Θ를 동시에 풀고 (선형), 더 큰 쪽(QR 이동 환산)을 축으로."""
    def predict(self, f):
        sY, _, bY = self.co['Y']; sT, _, bT = self.co['Θ']; b = (bY + bT) / 2
        U = self.U if self.U is not None else np.zeros((len(f), 0)); A = np.c_[sY, sT, U]
        if self.W.ndim == 1: p, *_ = np.linalg.lstsq(A * self.W[:, None], (f - b) * self.W, rcond=None)
        else: p, *_ = np.linalg.lstsq(self.W @ A, self.W @ (f - b), rcond=None)
        # 축은 동시 추정에서 더 큰 쪽 (QR 이동 환산), 값은 그 축 하나로 다시 푼 값
        return dict(vY=self.solve(f, 'Y')[0], vT=self.solve(f, 'Θ')[0], llr=np.log(abs(p[1]) * 80.66 + 1e-9) - np.log(abs(p[0]) * 14.39 + 1e-9))

def nuis_rows(F):
    xc, yc = (F.x.values - F.x.mean()) / 100, (F.y.values - F.y.mean()) / 100; z = np.zeros(len(F)); o = np.ones(len(F))
    return np.stack([np.c_[z, o].ravel(), np.c_[-yc, xc].ravel()], 1)

# ---------------- 실행 ----------------
def run(Fe, make_model, label='보정본', cv='짝홀', thr='1', keep=None):
    """Fe: (사진, 특징). thr '1' = 비용 비 1, 'cal' = 학습 폴드 안에서 짝홀로 문턱 보정."""
    recs = []; Fe_fn = Fe if callable(Fe) else None
    for nm, te, k in folds(cv):
        tr = ~te; lab = CMD if label == '명령값' else CORR[k]
        if Fe_fn is not None: Fe = Fe_fn(tr, lab)                  # 폴드마다 학습 사진으로만 점을 고른 특징
        mdl = make_model().fit(Fe[tr], lab[tr], AX[tr])
        t = 0.0
        if thr == 'cal':
            idx = np.where(tr & ~ISC)[0]; half = np.arange(len(idx)) % 2; ll, aa = [], []
            for h in (0, 1):
                inner = make_model().fit(Fe[np.r_[idx[half != h], np.where(ISC)[0]]], lab[np.r_[idx[half != h], np.where(ISC)[0]]], AX[np.r_[idx[half != h], np.where(ISC)[0]]])
                for i in idx[half == h]: ll.append(inner.predict(Fe[i])['llr']); aa.append(AX[i])
            ll, aa = np.array(ll), np.array(aa); cands = np.r_[0.0, np.sort(ll)]
            errs = [((ll < c) & (aa == 'Θ')).sum() + ((ll >= c) & (aa == 'Y')).sum() for c in cands]
            best = np.flatnonzero(errs == np.min(errs)); t = cands[best[np.argmin(np.abs(cands[best]))]]
        for i in np.where(te & ~ISC)[0]:
            p = mdl.predict(Fe[i]); pa = 'Y' if p['llr'] < t else 'Θ'
            recs.append(dict(file=NAMES[i], axis=AX[i], cmd=CMD[i], act=CORR[k][i], pred_axis=pa, pred=p['vY'] if pa == 'Y' else p['vT'],
                             vY=p['vY'], vT=p['vT'], llr=p['llr'], thr=t))
    return pd.DataFrame(recs)

TOL = {'0.09': {'Y': 0.09, 'Θ': 0.09}, '한스텝': {'Y': 0.05, 'Θ': 0.01}}
def score(df, basis):
    t = df[basis].values; p = df.pred.values; ok = (df.pred_axis == df.axis).values; y = (df.axis == 'Y').values; th = ~y
    e = np.abs(p - t); r = dict(축정확도_Y=ok[y].mean() * 100, 축정확도_Θ=ok[th].mean() * 100, 틀림_Y=int((~ok & y).sum()), 틀림_Θ=int((~ok & th).sum()),
                                Y_평균오차_um=e[y & ok].mean() * 1000, Y_최대_um=e[y & ok].max() * 1000,
                                Θ_평균오차_m=e[th & ok].mean() * 1000, Θ_최대_m=e[th & ok].max() * 1000)
    for tn, tol in TOL.items():
        tt = np.where(y, tol['Y'], tol['Θ']); tp = np.where(df.pred_axis == 'Y', tol['Y'], tol['Θ'])
        strict = ok & (e <= tt)                                                  # 축 맞고 오차 안
        harmless = strict | (~ok & (np.abs(t) <= tt) & (np.abs(p) <= tp))      # 축 틀렸어도 보정 후 둘 다 합격선 안
        r[f'합격_{tn}_엄격'] = strict.mean() * 100; r[f'합격_{tn}_피해없음'] = harmless.mean() * 100
    # 틀린 사진이 사라지는 크기 (Y는 mm, Θ는 같은 QR 이동의 Y mm로 환산)
    ye = np.abs(df.cmd.values) * np.where(y, 1, 80.66 / 14.39); r['전부맞음_시작_Ymm'] = ye[~ok].max() if (~ok).any() else 0.0
    return r

def table(df, **meta):
    rows = []
    for b, bn in [('cmd', '명령값'), ('act', '실제위치')]:
        rows.append(dict(**meta, 정답=bn, **score(df, b)))
    return rows


