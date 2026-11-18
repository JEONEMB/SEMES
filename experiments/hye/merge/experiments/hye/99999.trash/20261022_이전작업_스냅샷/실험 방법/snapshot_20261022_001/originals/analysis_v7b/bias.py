"""왜 틀리는 방향이 모델마다 바뀌나: 0 근처 편향 진단과 고치기.
가설: Θ 이동은 살짝 휘어 있음(2차 항). 직선 모델이면 휜 만큼이 절편(0점)에 섞여서 0 근처에서 Θ 모델이 어긋남 → Y가 이김.
(1) 0 근처 사진(|Y환산| ≤ 0.15mm)의 llr 중앙값 = 덩어리가 문턱 어느 쪽에 있나
(2) 절편 차이: 직선 모델 절편 - 2차 모델 절편 을 구분 방향으로 잰 크기 (σ 단위)
(3) 고치기: 다항 차수 1/2/3, center 고정(0점=center), 0 근처 사진만으로 학습"""
from exp import *
K = 80.66 / 14.39
F, D = load(8); Fe = feats(F, D, 'P12'); rows = []
CONF = [('1차, 절편 자유 (최종)', dict(quad=False)), ('2차, 절편 자유', dict(quad=True)), ('3차, 절편 자유', dict(deg=3)),
        ('1차, center 고정', dict(quad=False, intercept='zero')), ('2차, center 고정', dict(quad=True, intercept='zero')), ('3차, center 고정', dict(deg=3, intercept='zero')),
        ('1차, 0 근처만 학습 (Y≤0.6, Θ≤0.12)', dict(quad=False, local={'Y': 0.6, 'Θ': 0.12})),
        ('1차, 0 근처만 + center 고정', dict(quad=False, intercept='zero', local={'Y': 0.6, 'Θ': 0.12})),
        ('2차, 0 근처만 (Y≤1.0, Θ≤0.2) + center 고정', dict(quad=True, intercept='zero', local={'Y': 1.0, 'Θ': 0.2}))]
for nm, kw in CONF:
    for cv in ['짝홀', '구간빼기']:
        df = run(Fe, M(kind='P12', noise='full', F=F, **kw), cv=cv)
        ye = np.abs(np.where(df.axis == 'Y', df.cmd, df.cmd * K)); near = ye <= 0.15
        r = dict(설정=nm, 교차검증=cv, **score(df, 'act'), llr0_Y사진=np.median(df.llr[near & (df.axis == 'Y')]), llr0_Θ사진=np.median(df.llr[near & (df.axis == 'Θ')]))
        bad = df[df.axis != df.pred_axis].sort_values('cmd'); r['틀린사진'] = ', '.join(f'{a}{v:+.2f}' for a, v in zip(bad.axis, bad.cmd))
        rows.append(r); print(f"{nm:38s} {cv:4s} 틀림 {r['틀림_Y']}/{r['틀림_Θ']}  0근처 llr 중앙 Y사진 {r['llr0_Y사진']:+.2f} Θ사진 {r['llr0_Θ사진']:+.2f}  {r['틀린사진']}", flush=True)
# (2) 절편 차이 크기
m1 = M(kind='P12', quad=False, noise='full', F=F)().fit(Fe, CORR[0], AX); m2 = M(kind='P12', quad=True, noise='full', F=F)().fit(Fe, CORR[0], AX)
for a in ['Y', 'Θ']:
    gap = m1.poly[a][1] - m2.poly[a][1]; c = m2.co[a][1]
    print(f'{a}: 2차 항 크기 {np.linalg.norm(m2.W @ c):.1f}σ (v=1일 때), 직선 절편 - 2차 절편 = {np.linalg.norm(m2.W @ gap):.2f}σ')
pd.DataFrame(rows).round(3).to_csv(OUT/'62_v7_0근처편향.csv', index=False, encoding='utf-8-sig')

