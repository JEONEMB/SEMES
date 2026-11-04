"""C6: 특징점 조합용 가우스-뉴턴. 앞 모델 f = b + J[Θ,Y] + Q[Θ², ΘY, Y²] 를 학습 사진으로 맞추고,
새 사진에서는 C5(GLS) 해에서 출발해 GLS 가중 가우스-뉴턴으로 Θ, Y를 반복해서 푼다.
오차(가중 잔차 제곱합)가 줄어드는 걸음만 받아들이는 감쇠(레벤버그-마쿼트 방식)로 발산을 막는다."""
import numpy as np
from common import *
def basis(p): t, y = p[..., 0], p[..., 1]; return np.stack([t, y, t*t, t*y, y*y, np.ones_like(t)], -1)
def dbasis(p):
    t, y = p
    return np.array([[1, 0], [0, 1], [2*t, 0], [y, t], [0, 2*y], [0, 0]])
def a_gn(F, tr, te, iters=20):
    if F.shape[1] < 2: return None
    W = np.linalg.lstsq(basis(X[tr]), F[tr], rcond=None)[0]           # 6 x k
    R = F[tr] - basis(X[tr]) @ W
    S = np.cov(R.T); S = S + np.eye(len(S)) * np.trace(S) / len(S) * 1e-2; Si = np.linalg.inv(S)
    P = a_jgls(F, tr, te); out = []
    cost = lambda p, f: float((f - basis(p) @ W) @ Si @ (f - basis(p) @ W))
    for f, p in zip(F[te], P):
        p = p.copy(); lam = 1e-6; c = cost(p, f)
        for _ in range(iters * 5):
            r = f - basis(p) @ W; Jm = (dbasis(p).T @ W).T                # k x 2
            A = Jm.T @ Si @ Jm; g = Jm.T @ Si @ r
            step = np.linalg.solve(A + lam * np.diag(np.diag(A)), g)
            c2 = cost(p + step, f)
            if c2 < c:                                                     # 오차가 줄 때만 이동 (발산 방지)
                p = p + step; c = c2; lam = max(lam / 10, 1e-9)
                if np.abs(step).max() < 1e-9: break
            else:
                lam *= 10
                if lam > 1e8: break
        out.append(p)
    return np.array(out)
ALG['C6 가우스-뉴턴 (곡선 모델, 반복)'] = a_gn
