"""2단계 방법에 33px 측정(v7 최종: 33px 템플릿, 흐림 σ2, 가우스-뉴턴) 붙이기.
같은 점 232개 (QR만), 특징 P12, 2단계 (판정: 0 근처 1차 + center 0점 / 값: 전체 2차). 17px(v7b 기본)과 비교.
교차검증: 짝홀, 구간빼기 (라벨 보정본/명령값) + 구간별 무작위 반반 40번 (라벨 명령값, 축 정확도).
결과: out/70_2단계_33px_평가.csv, out/71_2단계_33px_반복40.csv"""
import exp
from exp import *

K = 80.66 / 14.39
LOC = {'Y': 0.6, 'Θ': 0.12}

def load33():
    F, _ = load(8)
    Z = np.load(PROJ/'analysis_v7'/'tr_fs8_hp16_s2.npz', allow_pickle=True)
    assert list(Z['names']) == NAMES
    Fa = pd.read_csv(V6/'features_qr_hp8.csv'); m = Fa.region.str.startswith('QR').values
    D = Z['D'][:, m].astype(np.float64); D = D - D[ISC]
    return F, D

class TwoStage:
    def __init__(self, F, kind):
        self.A = M(kind=kind, quad=False, noise='full', F=F, intercept='zero', local=LOC)()
        self.V = M(kind=kind, quad=True, noise='full', F=F, intercept='free')()
    def fit(self, Fe, lab, ax):
        self.A.fit(Fe, lab, ax); self.V.fit(Fe, lab, ax); return self
    def predict(self, f):
        return dict(vY=self.V.solve(f, 'Y')[0], vT=self.V.solve(f, 'Θ')[0], llr=self.A.predict(f)['llr'])

def splits40():
    rng = np.random.default_rng(0); out = []
    for _ in range(40):
        te = np.zeros(len(NAMES), bool)
        for s in np.unique(SES[~ISC]):
            idx = rng.permutation(np.where(SES == s)[0]); te[idx[:len(idx) // 2]] = True
        out += [te, ~te & ~ISC]
    return out

def repeat40(Fe, make):
    recs = []
    for r, te in enumerate(splits40()):
        tr = ~te; mdl = make().fit(Fe[tr], CMD[tr], AX[tr])
        for i in np.where(te & ~ISC)[0]:
            p = mdl.predict(Fe[i]); pa = 'Y' if p['llr'] < 0 else 'Θ'
            recs.append(dict(rep=r // 2, file=NAMES[i], axis=AX[i], cmd=CMD[i], pred_axis=pa, pred=p['vY'] if pa == 'Y' else p['vT']))
    return pd.DataFrame(recs)

if __name__ == '__main__':
    MEAS = {'17px σ1 (v7b)': load(8), '33px σ2 (v7)': load33()}
    rows, reps = [], []
    for mn, (F, D) in MEAS.items():
        Fe = feats(F, D, 'P12'); make = lambda F=F: TwoStage(F, 'P12')
        for label in ['보정본', '명령값']:
            for cv in ['짝홀', '구간빼기']:
                df = run(Fe, make, label=label, cv=cv)
                rows += table(df, 측정=mn, 학습라벨=label, 교차검증=cv)
                bad = df[df.axis != df.pred_axis]
                for r in rows[-2:]:
                    if (label == '보정본') == (r['정답'] == '실제위치'):
                        print(f"{mn} | {label} {cv} [정답 {r['정답']}] 틀림 {r['틀림_Y']}/{r['틀림_Θ']}, Y 평균/최대 {r['Y_평균오차_um']:.1f}/{r['Y_최대_um']:.1f}µm, "
                              f"Θ 최대 {r['Θ_최대_m'] / 1000:.4f}, 0.09 엄격/피해없음 {r['합격_0.09_엄격']:.1f}/{r['합격_0.09_피해없음']:.1f} | 틀린 사진 "
                              + ', '.join(f'{a}{v:+.2f}' for a, v in zip(bad.axis, bad.cmd)), flush=True)
        rp = repeat40(Fe, make); rp['측정'] = mn; reps.append(rp)
        ok = rp.axis == rp.pred_axis; per = (~ok).groupby(rp.rep).sum()
        wrong = rp[~ok]; ye = np.abs(wrong.cmd) * np.where(wrong.axis == 'Y', 1, K)
        print(f"{mn} | 반복40 (명령값): 축 정확도 평균 {ok.mean() * 100:.2f}%, 반복당 틀림 평균 {per.mean():.2f} (최소 {per.min()}, 최대 {per.max()}), "
              f"전부 맞은 반복 {(per == 0).sum()}/40, 틀린 사진 크기(Y환산) 중앙 {np.median(ye):.3f} 최대 {ye.max():.3f}mm", flush=True)
        print('   자주 틀린 사진:', wrong.groupby(['axis', 'cmd']).size().sort_values(ascending=False).head(8).to_dict(), flush=True)
    pd.DataFrame(rows).round(4).to_csv(OUT/'70_2단계_33px_평가.csv', index=False, encoding='utf-8-sig')
    pd.concat(reps).round(5).to_csv(OUT/'71_2단계_33px_반복40.csv', index=False, encoding='utf-8-sig')
