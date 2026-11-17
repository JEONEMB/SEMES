# 한 단계 이동량: 같은 trial(촬영 구간) 안에서 명령값이 한 칸 바뀔 때 QR이 가로로 몇 px 움직였나
#   이동량 = QR 3개 가로 이동(dLx, dCx, dRx, ECC 아핀으로 잰 값) 평균. trial마다 |명령값| 순서로 정렬해 이웃끼리 차이
#   Y 두 trial(0.1 간격, 0.05 홀수)은 모두 한 칸 = 0.1mm, Θ는 한 칸 = 0.01
#   center에서 첫 사진으로 가는 단계(0 -> 첫 값)는 0점 문제가 섞여서 따로 표시
import numpy as np, pandas as pd, pathlib
HERE = pathlib.Path(__file__).parent
d = pd.read_csv(HERE.parent / '14_데이터_QR마커위치.csv')
d['dx'] = d[['dLx', 'dCx', 'dRx']].mean(1)
def trial(r):
    if r.axis == 'center': return None
    s = '+' if r.value > 0 else '-'
    if r.file[0] == 'T': return 'Θ ' + s
    return ('Y 0.1 간격 ' if round(abs(r.value) * 100) % 10 == 0 else 'Y 0.05 홀수 ') + s
d['trial'] = d.apply(trial, axis=1)
c0 = d[d.file == 'center.bmp'].dx.iloc[0]
rows = []
for t, g in d.dropna(subset=['trial']).groupby('trial'):
    g = g.assign(a=g.value.abs()).sort_values('a')
    v, x = g.value.values, g.dx.values * np.sign(g.value.values)     # 부호를 맞춰서 '바깥쪽으로 한 칸' 이동량
    unit = 0.01 if t.startswith('Θ') else 0.1
    first = dict(trial=t, step_from='center', step_to=g.file.iloc[0][:-4], px=x[0] - c0 * np.sign(v[0]), kind='first')
    rows.append(first)
    for i in range(1, len(g)):
        rows.append(dict(trial=t, step_from=g.file.iloc[i - 1][:-4], step_to=g.file.iloc[i][:-4], px=x[i] - x[i - 1], kind='step'))
S = pd.DataFrame(rows)
S['axis'] = np.where(S.trial.str.startswith('Θ'), 'Θ', 'Y')
st = S[S.kind == 'step']
summ = st.groupby('axis').px.agg(['count', 'mean', 'std', 'min', 'max'])
per_trial = st.groupby('trial').px.agg(['count', 'mean', 'std'])
# 크게 벗어난 단계: 축 평균에서 3σ (축별 표준편차) 이상
st = st.join(summ[['mean', 'std']], on='axis')
st['z'] = (st.px - st['mean']) / st['std']
out = st[st.z.abs() >= 3].sort_values('z', key=abs, ascending=False)
S.to_csv(HERE / 'img' / 'step_px.csv', index=False, encoding='utf-8-sig')
pd.set_option('display.width', 200)
print(summ.round(3)); print(per_trial.round(3))
print('first steps (center -> 첫 사진):'); print(S[S.kind == 'first'][['trial', 'step_to', 'px']].round(3).to_string(index=False))
print('outliers |z|>=3:'); print(out[['trial', 'step_from', 'step_to', 'px', 'z']].round(3).to_string(index=False))
print('|z|>=2 count by axis:', st[st.z.abs() >= 2].groupby('axis').size().to_dict())
