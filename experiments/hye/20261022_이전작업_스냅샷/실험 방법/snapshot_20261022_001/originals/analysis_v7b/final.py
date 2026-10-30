"""최종 평가: QR만, 17px 템플릿, 가우스-뉴턴 점 이동 → 2차 다항 판 변형 12개 → 축마다 1차 매핑, 전체 공분산 가설 비교.
(1) 0점 처리 4가지 x 교차검증 2가지 x 정답 2가지  (2) 틀린 사진 목록  (3) 이론 한계 (3σ 최소 |Y|)
출력: 54_v7_최종평가.csv, 55_v7_사진별예측.csv"""
from exp import *

F, D = load(8); Fe = feats(F, D, 'P12'); rows, PRED = [], {}
for label in ['명령값', '보정본']:
    for ic in ['free', 'zero']:
        for cv in ['짝홀', '구간빼기']:
            df = run(Fe, M(kind='P12', quad=False, noise='full', intercept=ic, F=F), label=label, cv=cv)
            meta = dict(학습라벨=label, 절편='자유' if ic == 'free' else 'center 고정(0점=center)', 교차검증=cv)
            rows += table(df, **meta); PRED[tuple(meta.values())] = df
T = pd.DataFrame(rows); T.round(3).to_csv(OUT/'54_v7_최종평가.csv', index=False, encoding='utf-8-sig')
cols = ['학습라벨', '절편', '교차검증', '정답', '축정확도_Y', '축정확도_Θ', '틀림_Y', '틀림_Θ', 'Y_평균오차_um', 'Y_최대_um', 'Θ_평균오차_m', 'Θ_최대_m',
        '합격_0.09_엄격', '합격_0.09_피해없음', '합격_한스텝_엄격', '합격_한스텝_피해없음', '전부맞음_시작_Ymm']
pd.set_option('display.width', 250); print(T[cols].round(2).to_string())
best = PRED[('보정본', '자유', '짝홀')]
best.to_csv(OUT/'55_v7_사진별예측.csv', index=False, encoding='utf-8-sig')
bad = best[best.axis != best.pred_axis].sort_values('cmd')
print('틀린 사진 (보정본, 절편 자유, 짝홀):', ', '.join(f'{r.axis}{r.cmd:+.2f}→{r.pred_axis}{r.pred:+.3f} (llr {r.llr:+.2f})' for r in bad.itertuples()))
for nm, k in [('보정본 자유 구간빼기', ('보정본', '자유', '구간빼기')), ('보정본 center고정 짝홀', ('보정본', 'center 고정(0점=center)', '짝홀'))]:
    b = PRED[k]; b = b[b.axis != b.pred_axis].sort_values('cmd'); print(f'틀린 사진 ({nm}):', ', '.join(f'{r.axis}{r.cmd:+.2f}' for r in b.itertuples()))
# (3) 이론 한계: 12개 값 공간에서 Y 1mm 무늬를 Θ + 방해 무늬로 흉내 내고 남은 차이 (전체 공분산으로 화이트닝)
mdl = M(kind='P12', quad=False, noise='full', F=F)().fit(Fe, CORR[0], AX)
sY, sT = mdl.co['Y'][0], mdl.co['Θ'][0]; U = U_for(F, 'P12'); A = mdl.W @ np.c_[sT, U]
p, *_ = np.linalg.lstsq(A, mdl.W @ sY, rcond=None); n = np.linalg.norm(mdl.W @ sY - A @ p)
print(f'이론 한계: Y 1mm당 {n:.1f}σ → 3σ 구분 최소 |Y| {3 / n:.3f}mm (Θ로 {3 / n * 14.39 / 80.66:.4f}), 2σ {2 / n:.3f}mm')

