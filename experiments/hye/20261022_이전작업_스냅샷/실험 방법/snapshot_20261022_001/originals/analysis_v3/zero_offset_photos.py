"""0점 어긋남을 실제 사진으로: center 옆 사진 6장(Y ±0.05, ±0.10, Θ ±0.01)을
'center를 명령한 만큼 옮긴 예상 사진'과 비교해서, QR 모서리마다 예상보다 몇 px 더(덜) 갔는지 색으로 칠한다.

- 예상 사진: center를 (기울기 x 명령값)만큼 가로로 정확히 옮김 (3차 스플라인, 소수 위치 그대로)
- 어긋남 = -(실제 - 예상) / (예상 사진의 가로 밝기 기울기). 밝기가 크게 바뀌는 모서리에서만 계산
- 파랑 = 예상보다 왼쪽(덜 감 또는 더 감), 빨강 = 예상보다 오른쪽"""
import numpy as np, pandas as pd, cv2, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy import ndimage
from pathlib import Path
HERE = Path(__file__).parent; PROJ = HERE.parent
ROOT = Path(r"C:\Users\SSAFY\Downloads\[기업연계_세메스] 교보재\[기업연계_세메스] 교보재\extracted\SSAFY Test Image Set")
plt.rcParams['font.family'] = 'Malgun Gothic'; plt.rcParams['axes.unicode_minus'] = False
q = pd.read_csv(PROJ/'14_데이터_QR마커위치.csv').set_index('file'); c0 = q.loc['center.bmp']
X0, X1 = int(c0.Lx - 100), int(c0.Rx + 100); Y0, Y1 = int(c0.Cy - 105), int(c0.Cy + 105)
SY = 14.37                      # Y 1mm당 px (구간마다 맞춘 기울기의 평균, 18번 그림과 같음)
ST = 80.5                       # Θ 1당 px (+쪽 80.3, -쪽 80.8의 평균)
def load(f):
    p = ROOT/'center.bmp' if f == 'center.bmp' else ROOT/('Theta' if f[0] == 'T' else 'Y')/f
    im = cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_GRAYSCALE).astype(np.float64)
    return cv2.GaussianBlur(im[Y0 - 20:Y1 + 20, X0 - 40:X1 + 40], (0, 0), 1.0)
cen = load('center.bmp')
PH = [('Y-0.10.bmp', 'Y -0.10mm', '1차 -쪽, - 방향'), ('Y+0.10.bmp', 'Y +0.10mm', '1차 +쪽, + 방향'),
      ('Y-0.05.bmp', 'Y -0.05mm', '2차 -쪽, + 방향 (-쪽에서 올라옴)'), ('Y+0.05.bmp', 'Y +0.05mm', '2차 +쪽, + 방향'),
      ('T-0.01.bmp', 'Θ -0.01', 'Θ -쪽, - 방향'), ('T+0.01.bmp', 'Θ +0.01', 'Θ +쪽, + 방향')]
rows = []
fig, axes = plt.subplots(3, 2, figsize=(15, 7.6))
norm = TwoSlopeNorm(vmin=-0.8, vcenter=0, vmax=0.8)
for ax, (f, lab, seg) in zip(axes.ravel(), PH):
    v = float(f[1:-4]); s = (SY if f[0] == 'Y' else ST) * v                      # 명령대로라면 움직였어야 할 px
    exp = ndimage.shift(cen, (0, s), order=3, mode='nearest')                   # center를 명령만큼 옮긴 예상 사진
    act = load(f); gx = np.gradient(exp, axis=1)
    sl = (slice(20, -20), slice(40, -40)); d, g, A = (act - exp)[sl], gx[sl], act[sl]
    m = np.abs(g) > 12                                                          # 밝기가 크게 바뀌는 모서리만
    sh = np.where(m, -d / np.where(m, g, 1), np.nan)                            # 그 자리에서 예상 대비 가로로 밀린 양 (px)
    meas = q.loc[f, ['dLx', 'dCx', 'dRx']].mean(); off = meas - s              # QR 네모로 잰 값 (확인용)
    med = np.nanmedian(sh)
    ax.imshow(A, cmap='gray', vmin=0, vmax=255, alpha=0.55)
    yy, xx = np.where(m); k = slice(None, None, 3)
    ax.scatter(xx[k], yy[k], c=sh[m][k], cmap='RdBu_r', norm=norm, s=1.2, linewidths=0)
    um = off / SY * 1000 if f[0] == 'Y' else None
    u = f'{um:+.0f}µm' if um is not None else f'Θ {off / ST:+.4f}'
    ax.set_title(f'{lab}  ({seg})\n예상 {s:+.2f}px, 실제 {meas:+.2f}px  →  예상보다 {off:+.2f}px ({u})', loc='left', fontsize=11)
    ax.axis('off')
    rows.append((lab, seg, s, meas, off, med, um))
sm = plt.cm.ScalarMappable(cmap='RdBu_r', norm=norm)
cb = fig.colorbar(sm, ax=axes, orientation='vertical', fraction=0.015, pad=0.01)
cb.set_label('예상 위치보다 가로로 밀린 양 (px)\n파랑 = 왼쪽, 빨강 = 오른쪽', fontsize=10)
plt.savefig(PROJ/'29_0점어긋남_실제사진.png', dpi=130, facecolor='white', bbox_inches='tight')
for r in rows: print(f'{r[0]:10s} 예상 {r[2]:+.3f} 실제 {r[3]:+.3f} 차이 {r[4]:+.3f}px (모서리 중앙값 {r[5]:+.3f}px) {"" if r[6] is None else f"{r[6]:+.0f}µm"}')
pd.DataFrame(rows, columns=['사진', '구간', '예상px', '실제px', '차이px', '모서리중앙값px', '차이um']).to_csv(HERE/'zero_offset_photos.csv', index=False, encoding='utf-8-sig')
