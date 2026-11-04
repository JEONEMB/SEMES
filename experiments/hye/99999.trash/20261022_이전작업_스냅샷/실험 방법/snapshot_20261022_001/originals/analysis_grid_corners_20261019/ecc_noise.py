# 비교용: ECC(QR 3개 단위, 14_데이터_QR마커위치.csv)로 모서리와 같은 방식의 흔들림 계산
#   '점' = QR 3개(L, C, R). trial 안에서 QR마다 이동량을 명령값 2차 곡선에 맞추고 남은 값을 사진 공통 + QR 고유로 나눔
import numpy as np, pandas as pd, pathlib
HERE = pathlib.Path(__file__).parent
Q = pd.read_csv(HERE.parent / '14_데이터_QR마커위치.csv'); Q = Q[Q.file != 'center.bmp']
def trial(f):
    v = float(f[1:-4]); s = '+' if v > 0 else '-'
    return ('Θ ' + s) if f[0] == 'T' else (('Y 0.1 간격 ' if round(abs(v) * 100) % 10 == 0 else 'Y 0.05 홀수 ') + s)
Q['trial'] = Q.file.map(trial)
rows = []
for t, g in Q.groupby('trial'):
    v = g.value.values
    for comp, cols in (('dx', ['dLx', 'dCx', 'dRx']), ('dy', ['dLy', 'dCy', 'dRy'])):
        M = g[cols].values
        res = M - np.stack([np.polyval(np.polyfit(v, M[:, j], 2), v) for j in range(3)], 1)
        common = res.mean(1, keepdims=True); own = res - common
        rows.append(dict(trial=t, comp=comp, own_med=np.median(own.std(0, ddof=3)), common=common.std(ddof=3)))
R = pd.DataFrame(rows); R.to_csv(HERE / 'out' / 'ecc_noise.csv', index=False, encoding='utf-8-sig'); print(R.round(4).to_string(index=False))
