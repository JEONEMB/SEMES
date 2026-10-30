# 정리본 v2 에 쓸 숫자 계산 (모서리 630점 = 대각선 모서리 제외 기준)
#   입력: analysis_grid_corners_20261019/out/corners_280.csv (모서리 755점 x 281장 서브픽셀 위치)
#   출력 (v2/img/): photo_motion.csv  사진마다 630점 평균 이동 dx, dy, 촬영 시각(분), trial
#                   zero_offsets.csv  trial마다 0점 위치 (평균 이동을 명령값 직선으로 맞춘 절편)
#                   common_resid.csv  사진마다 '사진 공통 흔들림' (점마다 2차 곡선을 빼고 남은 값의 점 평균)
#                   trial_slopes.csv  trial마다 기울기(px/명령 1), 직선 잔차, 세로 이동 범위
#                   corner_ref.csv    center 모서리 기준 위치 요약 (QR별 점 수, 경계, 중심)
#                   brightness.csv    사진마다 QR 판 영역 평균 밝기
import pathlib, numpy as np, pandas as pd, cv2
HERE = pathlib.Path(__file__).parent; PROJ = HERE.parent.parent; OUT = HERE / 'img'
ROOT = pathlib.Path(r"C:\Users\SSAFY\Downloads\[기업연계_세메스] 교보재\[기업연계_세메스] 교보재\extracted\SSAFY Test Image Set")
C = pd.read_csv(PROJ / 'analysis_grid_corners_20261019' / 'out' / 'corners_280.csv')
def trial(f):
    if f == 'center.bmp': return 'center'
    v = float(f[1:-4]); s = '+' if v > 0 else '-'
    return ('Θ ' + s) if f[0] == 'T' else (('Y 0.1 간격 ' if round(abs(v) * 100) % 10 == 0 else 'Y 0.05 홀수 ') + s)
path = lambda f: ROOT / 'center.bmp' if f == 'center.bmp' else ROOT / ('Theta' if f[0] == 'T' else 'Y') / f
files = sorted(C.file.unique())
cen = C[C.file == 'center.bmp'].set_index('id'); use = cen.index[cen.black_neighbors != 2]       # 630점
X = C.pivot(index='file', columns='id', values='x')[use]; Yp = C.pivot(index='file', columns='id', values='y')[use]
DX, DY = X - X.loc['center.bmp'], Yp - Yp.loc['center.bmp']
M = pd.DataFrame({'file': DX.index, 'dx': DX.mean(1).values, 'dy': DY.mean(1).values})
M['trial'] = M.file.map(trial); M['value'] = M.file.map(lambda f: 0.0 if f == 'center.bmp' else float(f[1:-4]))
mt = {f: path(f).stat().st_mtime for f in M.file}; t0 = min(mt.values()); M['minutes'] = M.file.map(lambda f: (mt[f] - t0) / 60)
M.to_csv(OUT / 'photo_motion.csv', index=False, encoding='utf-8-sig')

# trial마다: 직선 기울기, 0점(절편), 직선 잔차, 세로 이동
rows = []
for t, g in M[M.trial != 'center'].groupby('trial'):
    v, x = g.value.values, g.dx.values; k, b = np.polyfit(v, x, 1)
    unit = 'Y' if t[0] == 'Y' else 'Θ'
    rows.append(dict(trial=t, axis=unit, n=len(g), slope_px=k, zero_px=b, resid_px=np.std(x - (k * v + b), ddof=2),
                     zero_cmd=b / k, dy_min=g.dy.min(), dy_max=g.dy.max(), dy_end=g.sort_values('value', key=abs).dy.iloc[-1],
                     start_min=g.minutes.min(), end_min=g.minutes.max()))
Z = pd.DataFrame(rows)
Z['zero_um'] = np.where(Z.axis == 'Y', Z.zero_cmd * 1000, np.nan)          # Y: mm -> µm
Z['zero_mtheta'] = np.where(Z.axis == 'Θ', Z.zero_cmd * 1000, np.nan)      # Θ: 0.001 단위
Z.to_csv(OUT / 'zero_offsets.csv', index=False, encoding='utf-8-sig')
print(Z.round(4).to_string(index=False))

# 사진 공통 흔들림 (corner_steps.py 와 같은 방식, x 방향)
R = []
for t in Z.trial:
    fs = M[M.trial == t].sort_values('value', key=abs); v = fs.value.values; D = DX.loc[fs.file].values
    res = D - np.stack([np.polyval(np.polyfit(v, D[:, j], 2), v) for j in range(D.shape[1])], 1)
    for f, c, own in zip(fs.file, res.mean(1), (res - res.mean(1, keepdims=True)).std(1)):
        R.append(dict(file=f, trial=t, common_px=c, spread_px=own))
pd.DataFrame(R).merge(M[['file', 'minutes', 'value']], on='file').to_csv(OUT / 'common_resid.csv', index=False, encoding='utf-8-sig')

# center 모서리 기준 위치 요약
ref = []
for q, g in cen.groupby('qr'):
    u = g[g.black_neighbors != 2]
    ref.append(dict(qr='LCR'[q - 1], points=len(g), used=len(u), diag=int((g.black_neighbors == 2).sum()), center_valid=int(g.valid.sum()),
                    x_min=u.x.min(), x_max=u.x.max(), y_min=u.y.min(), y_max=u.y.max(), cx=u.x.mean(), cy=u.y.mean()))
F = pd.DataFrame(ref); F.to_csv(OUT / 'corner_ref.csv', index=False, encoding='utf-8-sig'); print(F.round(2).to_string(index=False))

# 밝기: QR 판 영역 (x 1545~2262, y 962~1180) 평균
B = []
for f in M.file:
    g = cv2.imdecode(np.fromfile(str(path(f)), np.uint8), cv2.IMREAD_GRAYSCALE)
    B.append(dict(file=f, plate_mean=g[962:1180, 1545:2262].mean(), whole_mean=g.mean()))
pd.DataFrame(B).merge(M[['file', 'minutes', 'trial']], on='file').to_csv(OUT / 'brightness.csv', index=False, encoding='utf-8-sig')
print('done')
