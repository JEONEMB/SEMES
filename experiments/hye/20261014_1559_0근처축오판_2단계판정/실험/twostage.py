"""2단계 방법: 축 판정과 값 계산을 나눈다.
  축 판정: 0 근처 사진(Y ≤ 0.6mm, Θ ≤ 0.12)만으로 학습한 1차 모델, 절편 = 0 (center = 0점). 0 근처에서 휨에 의한 0점 어긋남이 없음.
  값 계산: 전체 사진으로 학습한 2차 모델 (Θ의 휨을 담음), 판정된 축으로 값을 푼다.
결과: 63_v7_2단계_평가.csv. 여러 장 시뮬레이션도 이 방법으로 다시 (64)."""
from exp import *
K = 80.66 / 14.39
LOC = {'Y': 0.6, 'Θ': 0.12}
F, D = load(8); Fe = feats(F, D, 'P12')

class TwoStage:
    def __init__(self, F, intercept_val='free'):
        self.A = M(kind='P12', quad=False, noise='full', F=F, intercept='zero', local=LOC)()
        self.V = M(kind='P12', quad=True, noise='full', F=F, intercept=intercept_val)()
    def fit(self, Fe, lab, ax):
        self.A.fit(Fe, lab, ax); self.V.fit(Fe, lab, ax); return self
    def predict(self, f):
        llr = self.A.predict(f)['llr']; return dict(vY=self.V.solve(f, 'Y')[0], vT=self.V.solve(f, 'Θ')[0], llr=llr)

if __name__ == '__main__':
    rows, PRED = [], {}
    for label in ['보정본', '명령값']:
        for cv in ['짝홀', '구간빼기']:
            for iv in ['free', 'zero']:
                df = run(Fe, lambda iv=iv: TwoStage(F, iv), label=label, cv=cv)
                meta = dict(방법='2단계', 학습라벨=label, 값모델_절편='자유' if iv == 'free' else 'center 고정', 교차검증=cv); rows += table(df, **meta); PRED[(label, cv, iv)] = df
                r = rows[-2] if label == '명령값' else rows[-1]
                bad = df[df.axis != df.pred_axis]
                print(f"{label} {cv} 값절편 {iv}: [정답={r['정답']}] 틀림 {r['틀림_Y']}/{r['틀림_Θ']}, Y 평균/최대 {r['Y_평균오차_um']:.1f}/{r['Y_최대_um']:.1f}µm, "
                      f"Θ 평균/최대 {r['Θ_평균오차_m']:.2f}/{r['Θ_최대_m']:.2f}m, 합격 0.09 {r['합격_0.09_엄격']:.1f}/{r['합격_0.09_피해없음']:.1f}, "
                      f"한스텝 {r['합격_한스텝_엄격']:.1f}/{r['합격_한스텝_피해없음']:.1f} | 틀린 사진 " + ', '.join(f'{a}{v:+.2f}' for a, v in zip(bad.axis, bad.cmd)), flush=True)
    pd.DataFrame(rows).round(3).to_csv(OUT/'63_v7_2단계_평가.csv', index=False, encoding='utf-8-sig')
    PRED[('보정본', '짝홀', 'free')].to_csv(OUT/'63_v7_2단계_사진별예측.csv', index=False, encoding='utf-8-sig')

