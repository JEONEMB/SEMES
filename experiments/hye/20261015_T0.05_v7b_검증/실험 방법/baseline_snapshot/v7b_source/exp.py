"""v7 단계별 실험. 사용: python exp.py S1 (S3, S4, S4b, S5, FINAL)
기본 설정 (다른 단계에서 고정): QR만, 템플릿 17px, M1 2차, 점마다 잡음 가중 + Huber, 방해 무늬(세로 이동, 화면 회전),
라벨 보정본, 절편 자유, 문턱 1, 짝/홀 교차검증."""
import sys
from lib import *

def U_for(F, kind):
    U = nuis_rows(F)
    if kind == 'pts': return U
    Dn = np.stack([U[:, j].reshape(-1, 2) * 0.05 for j in range(U.shape[1])])      # 방해 무늬를 판 변형 값으로 (작은 양으로 선형화)
    return (warp_features(Dn, F, kind) / 0.05).T

def feats(F, D, kind):
    return D.reshape(len(D), -1) if kind == 'pts' else warp_features(D, F, kind)

def M(kind='pts', quad=True, noise='diag', intercept='free', huber=3.0, joint=False, F=None, **kw):
    U = U_for(F, kind)
    if kind != 'pts' and noise == 'diag': huber = 0
    return (lambda: (Joint if joint else Model)(quad=quad, noise=noise, intercept=intercept, huber=huber, nuis=U, **kw))

def go(rows, F, D, kind='pts', label='보정본', cv='짝홀', thr='1', **kw):
    meta = kw.pop('meta'); df = run(feats(F, D, kind), M(kind=kind, F=F, **kw), label=label, cv=cv, thr=thr)
    rows += table(df, **meta); r = rows[-1]
    print(meta, f"[실제위치] 축 Y {r['축정확도_Y']:.1f} Θ {r['축정확도_Θ']:.1f} (틀림 {r['틀림_Y']}/{r['틀림_Θ']}), Y {r['Y_평균오차_um']:.1f}µm, Θ {r['Θ_평균오차_m']:.2f}m, "
          f"합격 0.09 {r['합격_0.09_엄격']:.1f}/{r['합격_0.09_피해없음']:.1f}, 한스텝 {r['합격_한스텝_엄격']:.1f}/{r['합격_한스텝_피해없음']:.1f}, 전부맞음 {r['전부맞음_시작_Ymm']:.2f}mm", flush=True)
    return df

if __name__ == '__main__':
    st = sys.argv[1]; rows = []
    if st == 'S1':
        for hp in (4, 6, 8, 12):
            for scr in (False, True):
                F, D = load(hp, scr); go(rows, F, D, meta=dict(단계='S1', 템플릿=2 * hp + 1, 세트='QR+나사' if scr else 'QR만', 점수=len(F)))
    if st == 'S3':
        F, D = load(8)
        for label in ['명령값', '보정본']:
            for ic in ['free', 'zero']:
                go(rows, F, D, label=label, intercept=ic, meta=dict(단계='S3', 학습라벨=label, 절편='자유' if ic == 'free' else 'center 고정'))
    if st == 'S4':
        F, D = load(8)
        for kind in ['pts', 'H8', 'P12']:
            for quad in (False, True):
                for noise in (['diag'] if kind == 'pts' else ['diag', 'full']):
                    go(rows, F, D, kind=kind, quad=quad, noise=noise, meta=dict(단계='S4', 특징={'pts': 'M1 점별', 'H8': 'M3 원근 8', 'P12': 'M3 2차변형 12'}[kind],
                                                                              매핑='2차' if quad else '1차', 잡음=noise))
        for kind in ['pts', 'P12']:
            go(rows, F, D, kind=kind, quad=False, joint=True, noise='diag' if kind == 'pts' else 'full', meta=dict(단계='S4', 특징=f'M2 동시 추정 ({kind})', 매핑='1차', 잡음='-'))
    if st == 'S4c':
        F, D = load(8)
        for kind in ['P12', 'P20', 'P12x3']:
            for quad in (False, True):
                go(rows, F, D, kind=kind, quad=quad, noise='full', meta=dict(단계='S4c', 특징=kind, 매핑='2차' if quad else '1차', 잡음='full'))
        for kind, noise in [('pts', 'diag'), ('P12', 'full')]:
            go(rows, F, D, kind=kind, quad=False, joint=True, noise=noise, meta=dict(단계='S4c', 특징=f'M2 동시 ({kind})', 매핑='1차', 잡음=noise))
    if st == 'S4b':
        F, D = load(8)
        for noise, hub, nm in [('equal', 0, '같은 가중'), ('diag', 0, '잡음 가중'), ('diag', 3.0, '잡음 가중 + Huber')]:
            go(rows, F, D, noise=noise, huber=hub, meta=dict(단계='S4b', 방법='M1 점별', 가중=nm, 사용점='전부'))
        # M3 (2차 변형 12, 1차, 전체 공분산): 점 전부 vs 단서 상위 k개 (학습 사진으로만 고름)
        from t6_F1 import leftover
        def contrib(Dtr, lab, ax):
            """학습 사진으로 점마다 구분 단서 (Y 무늬를 Θ로 흉내 내고 남은 차이 / 잡음)."""
            S, R = {}, []
            for a in ['Y', 'Θ']:
                m = (ax == a) | (ax == 'C'); A = np.c_[lab[m], np.ones(m.sum())]; Dm = Dtr[m].reshape(m.sum(), -1)
                co, *_ = np.linalg.lstsq(A, Dm, rcond=None); S[a] = co[0]; R.append(Dm - A @ co)
            var = np.vstack(R).var(0); _, _, r = leftover(S['Y'], S['Θ'], var, F.x.values.astype(float), F.y.values.astype(float), 1)
            return (r**2 / np.maximum(var, 0.003**2)).reshape(-1, 2).sum(1)
        for kk in [20, 50, 100, 150, None]:
            def Fe_fn(tr, lab, kk=kk):
                keep = np.ones(len(F), bool)
                if kk is not None:
                    c = contrib(D[tr], lab[tr], AX[tr]); keep = np.zeros(len(F), bool); keep[np.argsort(-c)[:kk]] = True
                return warp_features(D[:, keep], F[keep].reset_index(drop=True), 'P12')
            df = run(Fe_fn, M(kind='P12', quad=False, noise='full', F=F)); meta = dict(단계='S4b', 방법='M3 2차변형12', 가중='점 같은 가중', 사용점='전부' if kk is None else f'상위 {kk}')
            rows += table(df, **meta); r = rows[-1]; print(meta, f"틀림 {r['틀림_Y']}/{r['틀림_Θ']}, 합격 0.09 {r['합격_0.09_엄격']:.1f}, 전부맞음 {r['전부맞음_시작_Ymm']:.2f}", flush=True)
    if st == 'S5':
        for hp in (4, 6, 8, 12):
            for scr in (False, True):
                F, D = load(hp, scr)
                for thr in ['1', 'cal']:
                    go(rows, F, D, kind='P12', quad=False, noise='full', thr=thr, meta=dict(단계='S5', 템플릿=2 * hp + 1, 세트='QR+나사' if scr else 'QR만', 문턱=thr))
    pd.DataFrame(rows).round(3).to_csv(HERE/f'{st}.csv', index=False, encoding='utf-8-sig')


