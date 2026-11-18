"""제안 검증: 미세한 성분을 증폭하고 전체 일관성을 보면 Y/Θ 구분이 나아지나.
점 p마다 '구분 방향' g_p = (Y 가설 예측 이동) - (Θ 가설 예측 이동). 둘의 중간 m_p에서 실제 이동이 어느 쪽으로 치우쳤나 z_p = <d_p - m_p, g_p>.
  C0 증폭: 모든 이동을 1000배 → 판정 변화 없음을 확인 (신호와 잡음이 같이 커짐)
  C1 정합 필터 합: Σ w_p z_p (점마다 잡음 가중) → 지금 방법과 같은 원리
  C2 다수결: z_p > 0 인 점의 비율 (튀는 점에 강함)
  C3 QR 3개 합의: QR마다 C1으로 판정, 3개가 다 같으면 그 축, 아니면 '보류'
  C4 크게 차이 나는 점만: |g_p|가 상위 25%인 점만으로 C1 (미세 성분이 큰 곳만 증폭)
값 vY, vT는 최종 방법(M3)에서 가져온다. 모델은 학습 폴드로만, 짝/홀 교차검증."""
from exp import *

F, D = load(8); Fe = feats(F, D, 'P12'); NP = len(F)
QRI = {q: (F.region == q).values for q in ['QR좌', 'QR중', 'QR우']}
U = nuis_rows(F)
recs = []
for nm, te, k in folds('짝홀'):
    tr = ~te; lab = CORR[k]
    m3 = M(kind='P12', quad=False, noise='full', F=F)().fit(Fe[tr], lab[tr], AX[tr])
    S, R = {}, []
    for a in ['Y', 'Θ']:
        m = tr & ((AX == a) | ISC); A = np.c_[lab[m], np.ones(m.sum())]; Dm = D[m].reshape(m.sum(), -1)
        co, *_ = np.linalg.lstsq(A, Dm, rcond=None); S[a] = co; R.append(Dm[~ISC[m]] - A[~ISC[m]] @ co)
    w = 1 / np.maximum(np.vstack(R).var(0), 0.003**2)
    for i in np.where(te & ~ISC)[0]:
        p = m3.predict(Fe[i]); vY, vT = p['vY'], p['vT']
        pY = S['Y'][0] * vY + S['Y'][1]; pT = S['Θ'][0] * vT + S['Θ'][1]; mid = (pY + pT) / 2; g = pY - pT
        r = D[i].ravel() - mid
        u, *_ = np.linalg.lstsq(U * np.sqrt(w)[:, None], r * np.sqrt(w), rcond=None); r = r - U @ u        # 세로 이동, 화면 회전 빼기
        zc = (w * r * g).reshape(NP, 2).sum(1)                        # 점마다 치우침 (양수 = Y 쪽)
        gp = np.sqrt((w * g * g).reshape(NP, 2).sum(1))
        out = dict(file=NAMES[i], axis=AX[i], cmd=CMD[i], act=CORR[k][i], vY=vY, vT=vT, M3=p['llr'])
        out['C0'] = -((w * (r * 1000) * (g * 1000)).sum())            # 1000배 증폭
        out['C1'] = -zc.sum()
        out['C2'] = 0.5 - (zc > 0).mean()                             # 음수 = Y
        votes = [np.sign(-zc[QRI[q]].sum()) for q in QRI]
        out['C3'] = votes[0] if len(set(votes)) == 1 else np.nan      # nan = 보류
        top = gp >= np.quantile(gp, 0.75); out['C4'] = -zc[top].sum()
        recs.append(out)
P = pd.DataFrame(recs)
rows = []
ye = np.abs(np.where(P.axis == 'Y', P.cmd, P.cmd * 80.66 / 14.39))
for c, nm in [('M3', '지금 방법 (M3 비용 비)'), ('C0', 'C0 1000배 증폭'), ('C1', 'C1 정합 필터 합'), ('C2', 'C2 점 다수결'), ('C3', 'C3 QR 3개 합의 (불일치 = 보류)'), ('C4', 'C4 차이 큰 점 25%만')]:
    s = P[c].values; dec = np.where(np.isnan(s), '보류', np.where(s < 0, 'Y', 'Θ')); wrong = (dec != P.axis) & (dec != '보류')
    rows.append(dict(방법=nm, 틀림_Y=int((wrong & (P.axis == 'Y')).sum()), 틀림_Θ=int((wrong & (P.axis == 'Θ')).sum()), 보류=int((dec == '보류').sum()),
                     틀린사진=', '.join(f'{a}{v:+.2f}' for a, v in zip(P.axis[wrong], P.cmd[wrong])),
                     보류사진_최대크기_Ymm=ye[dec == '보류'].max() if (dec == '보류').any() else 0))
    print(rows[-1], flush=True)
same = (np.sign(P.C0) == np.sign(P.C1)).all(); print('C0 (1000배)와 C1 판정이 모두 같은가:', same)
pd.DataFrame(rows).to_csv(OUT/'60_v7_증폭_일관성.csv', index=False, encoding='utf-8-sig'); P.to_csv(HERE/'consistency_pred.csv', index=False, encoding='utf-8-sig')

