"""0점 맞추기 시뮬레이션. 구간(접근 방향, 시간)마다 '0에 있을 때의 판 모양' z_s가 center와 다르다.
Z0  center = 0점 (지금, 2단계 방법)
Z1  같은 조건의 0점 사진이 있다 (상한): 구간마다 z_s를 학습 사진으로 정확히 구해 뺀다. 0점을 뺐으니 정답 = 명령값.
Z2  같은 조건 0점 사진 K장 (현실): z_s + (실제 사진 K장의 잔차 평균) 을 뺀다. 무작위 20번 반복 평균.
Z1, Z2는 구간마다 0점을 알아야 해서 짝/홀 교차검증만 (구간 빼기는 불가능).
마지막으로 Z1 뒤의 잡음 구조로 여러 장 시뮬레이션 (가정 B: 점마다 흔들림만 1/N)."""
import numpy as np, pandas as pd
from exp import *
from twostage import TwoStage
F, D = load(8); Fe0 = feats(F, D, 'P12'); SESS = ['Y1+', 'Y1-', 'Y2+', 'Y2-', 'Θ+', 'Θ-']; rng = np.random.default_rng(1)

def session_zero(tr):
    """학습 사진으로 축마다 f = z_구간 + s v + c v² (구간별 절편) → 구간 0점 z_s와 잔차."""
    Z, R = {}, {}
    for a in ['Y', 'Θ']:
        ss = [s for s in SESS if s[0] == a]; m = tr & (AX == a)
        A = np.c_[np.stack([(SES[m] == s).astype(float) for s in ss], 1), CMD[m], CMD[m]**2]
        co, *_ = np.linalg.lstsq(A, Fe0[m], rcond=None); res = Fe0[m] - A @ co
        for j, s in enumerate(ss): Z[s] = co[j]; R[s] = res[SES[m] == s]
    return Z, R

def corrected(tr, K=None):
    Z, R = session_zero(tr); out = Fe0.copy()
    for s in SESS:
        z = Z[s] if K is None else Z[s] + R[s][rng.integers(0, len(R[s]), K)].mean(0)
        out[SES == s] -= z
    return out

def go(name, Fe_fn, label, reps=1):
    acc = []
    for r in range(reps):
        df = run(Fe_fn, lambda: TwoStage(F, 'zero'), label=label, cv='짝홀')
        for b in ['cmd', 'act']: acc.append(dict(방법=name, 반복=r, 정답='명령값' if b == 'cmd' else '실제위치', **score(df, b)))
    T = pd.DataFrame(acc); g = T.groupby('정답').mean(numeric_only=True)
    b = '명령값' if label == '명령값' else '실제위치'; r = g.loc[b]
    print(f"{name:34s} [정답={b}] 틀림 Y {r['틀림_Y']:.2f} Θ {r['틀림_Θ']:.2f}, Y 평균/최대 {r['Y_평균오차_um']:.1f}/{r['Y_최대_um']:.1f}µm, "
          f"Θ 평균/최대 {r['Θ_평균오차_m']:.2f}/{r['Θ_최대_m']:.2f}m, 한스텝 합격 {r['합격_한스텝_엄격']:.1f}%", flush=True)
    return T

if __name__ == '__main__':
    out = []
    out.append(go('Z0 center = 0점 (보정본 라벨)', Fe0, '보정본'))
    out.append(go('Z0 center = 0점 (명령값 라벨)', Fe0, '명령값'))
    out.append(go('Z1 같은 조건 0점, 정확히 앎 (상한)', lambda tr, lab: corrected(tr), '명령값'))
    for K in (1, 4, 16):
        out.append(go(f'Z2 같은 조건 0점 사진 {K}장', lambda tr, lab, K=K: corrected(tr, K), '명령값', reps=20))
    T = pd.concat(out); T.to_csv(OUT/'65_v7_0점맞추기.csv', index=False, encoding='utf-8-sig')
    # 0점 모양 차이가 얼마나 큰지 (구분 방향, 잡음 σ 단위)
    Z, R = session_zero(np.ones(len(NAMES), bool) & ~ISC)
    m = M(kind='P12', quad=True, noise='full', F=F)().fit(Fe0, CORR[0], AX)
    print('구간 0점 모양의 크기 (center 대비, 잡음 σ):', {s: round(float(np.linalg.norm(m.W @ Z[s])), 1) for s in SESS})
