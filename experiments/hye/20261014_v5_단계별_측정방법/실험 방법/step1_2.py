"""패치 이동장 방법 1~2단계.
1단계: 기준 영상 2종 (center 1장 / 학습 사진을 center 좌표로 되돌린 평균, 폴드마다 따로)
2단계: 영역(QR 3개 + 나사 2개)을 32x32 패치(16px 겹침)로 나누고, 무늬가 약한 패치를 뺀다.
원본은 읽기만 한다. 결과는 analysis_v5/ 와 프로젝트 루트 36~37번 파일로 저장."""
import json
import cv2, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from scipy import ndimage
from pathlib import Path

HERE = Path(__file__).parent; PROJ = HERE.parent
ROOT = Path(r"C:\Users\SSAFY\Downloads\[기업연계_세메스] 교보재\[기업연계_세메스] 교보재\extracted\SSAFY Test Image Set")
plt.rcParams['font.family'] = 'Malgun Gothic'; plt.rcParams['axes.unicode_minus'] = False
TXT, SUB = '#1f1f1e', '#5f5e58'
q = pd.read_csv(PROJ/'14_데이터_QR마커위치.csv').set_index('file'); c0 = q.loc['center.bmp']
hom = pd.read_csv(PROJ/'analysis_v2/homography_features.csv').set_index('file')

# ---------- 영역 ----------
X0, X1, Y0, Y1 = 1420, 2392, 948, 1192           # 판 전체 (나사 포함), 전체 사진 좌표
MX, MY = 130, 40                                  # 읽을 때 여유 (Θ ±81px, Y ±29px 이동)
HH, WW = Y1 - Y0, X1 - X0
PATCH, STRIDE = 32, 16

def read_raw(n):
    p = ROOT/'center.bmp' if n == 'center.bmp' else ROOT/('Theta' if n[0] == 'T' else 'Y')/n
    return cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_GRAYSCALE).astype(np.float32)

def find_screws(im):
    """밝은 와셔 고리의 무게중심으로 나사 중심을 잡는다 (대략 위치 주변 140x140에서)."""
    out = []
    for gx, gy in [(1492, 1063), (2312, 1073)]:
        w = im[gy - 70:gy + 70, gx - 70:gx + 70]; m = w >= 250   # 와셔가 포화(255)되어 있어서 포화 영역의 무게중심을 쓴다
        yy, xx = np.nonzero(m); out.append((gx - 70 + xx.mean(), gy - 70 + yy.mean()))
    return out

CEN = read_raw('center.bmp'); SCREWS = find_screws(CEN); R_SCREW = 52
QRBOX = [(c0[f'{k}x'], c0[f'{k}y'], c0[f'{k}w'] / 2 + 14, c0[f'{k}h'] / 2 + 14) for k in 'LCR']  # 여백 14px

def region_of(cx, cy):
    for i, (x, y, hw, hh) in enumerate(QRBOX):
        if abs(cx - x) <= hw and abs(cy - y) <= hh: return 'QR' + '좌중우'[i]
    for i, (x, y) in enumerate(SCREWS):
        if np.hypot(cx - x, cy - y) <= R_SCREW: return '나사' + '좌우'[i]
    return '그 외'

# ---------- 1단계: 기준 영상 ----------
CORN = ['좌상', '우상', '우하', '좌하']
SRC = np.float32([[X0, Y0], [X1, Y0], [X1, Y1], [X0, Y1]])
gy_, gx_ = np.mgrid[0:HH, 0:WW].astype(np.float64)
P0 = np.stack([gx_ + X0, gy_ + Y0], -1).reshape(-1, 2)

def corner_shift(n):
    """v2에서 QR 판으로 추출한 원근 변환을 판 영역 귀퉁이 이동으로 바꾼다 (귀퉁이 4개 x (dx, dy))."""
    return np.array([[hom.loc[n, f'{c}_dx'], hom.loc[n, f'{c}_dy']] for c in CORN], np.float32)

def back_to_center(n, im):
    """사진 n을 center 좌표로 되돌린 판 영역 (3차 스플라인 보간)."""
    H = cv2.getPerspectiveTransform(SRC, SRC + corner_shift(n))
    m = cv2.perspectiveTransform(P0[None], H)[0]
    coef = ndimage.spline_filter(cv2.GaussianBlur(im, (0, 0), 1.0).astype(np.float64), order=3)
    return ndimage.map_coordinates(coef, [m[:, 1], m[:, 0]], order=3, prefilter=False, mode='nearest').reshape(HH, WW)

names = [n for n in q.index if n != 'center.bmp']
import os
def _mt(m): return os.path.getmtime(ROOT/('Theta' if m[0] == 'T' else 'Y')/m)
def fold_of(n):
    """짝수/홀수 폴드: 축마다 촬영 시각 순서로 번갈아. 모든 촬영 구간이 양쪽 폴드에 반씩 들어간다.
    (명령값 순서로 번갈아 나누면 Y는 0.1 간격 세션과 0.05 간격 세션이 폴드와 겹쳐서 '구간 빼기'가 되어 버림)"""
    sub = sorted([m for m in names if m[0] == n[0]], key=_mt)
    return sub.index(n) % 2
FOLD = {n: fold_of(n) for n in names}

if __name__ == '__main__':
    cen_ref = cv2.GaussianBlur(CEN, (0, 0), 1.0)[Y0:Y1, X0:X1].astype(np.float64)
    acc = {0: np.zeros((HH, WW)), 1: np.zeros((HH, WW))}; cnt = {0: 0, 1: 0}
    for i, n in enumerate(names):
        w = back_to_center(n, read_raw(n)); f = FOLD[n]
        acc[1 - f] += w; cnt[1 - f] += 1     # 사진 n은 폴드 (1-f)를 시험할 때의 학습 사진
        if i % 40 == 0: print(i, n)
    # acc[k] = 폴드 k가 '시험'일 때 쓰는 기준 영상 = 폴드 (1-k) 사진들의 평균
    refs = {k: acc[k] / cnt[k] for k in (0, 1)}
    np.savez_compressed(HERE/'refs.npz', center=cen_ref, mean_test0=refs[0], mean_test1=refs[1], X0=X0, Y0=Y0)

    # ---------- 2단계: 패치 ----------
    gx = cv2.Sobel(cen_ref, cv2.CV_64F, 1, 0, ksize=3) / 8; gy = cv2.Sobel(cen_ref, cv2.CV_64F, 0, 1, ksize=3) / 8
    rows = []
    for py in range(0, HH - PATCH + 1, STRIDE):
        for px in range(0, WW - PATCH + 1, STRIDE):
            sl = (slice(py, py + PATCH), slice(px, px + PATCH))
            a, b, c = (gx[sl]**2).mean(), (gx[sl] * gy[sl]).mean(), (gy[sl]**2).mean()
            lmin = (a + c) / 2 - np.sqrt(((a - c) / 2)**2 + b**2); lmax = (a + c) / 2 + np.sqrt(((a - c) / 2)**2 + b**2)
            cx, cy = X0 + px + PATCH / 2, Y0 + py + PATCH / 2
            rows.append(dict(px=px, py=py, cx=cx, cy=cy, region=region_of(cx, cy), lmin=lmin, lmax=lmax))
    P = pd.DataFrame(rows)
    thr = 0.1 * P[P.region.str.startswith('QR')].lmin.median()
    P['use'] = (P.lmin > thr) & (P.region != '그 외')
    P.to_csv(HERE/'patches.csv', index=False, encoding='utf-8-sig')

    # ---------- 산출물: 비교표 ----------
    def stats(img, mask):
        g = np.hypot(cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3), cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)) / 8
        return img[mask].mean(), g[mask].mean()
    M = {}
    for nm in ['QR', '나사']:
        mk = np.zeros((HH, WW), bool)
        for _, r in P[P.region.str.startswith(nm)].iterrows(): mk[r.py:r.py + PATCH, r.px:r.px + PATCH] = True
        M[nm] = mk
    tab = []
    for lab, img in [('center 1장', cen_ref), ('평균 (폴드0 시험용)', refs[0]), ('평균 (폴드1 시험용)', refs[1])]:
        for nm in M:
            b, s = stats(img, M[nm]); tab.append(dict(기준영상=lab, 영역=nm, 밝기=b, 선명도=s))
    tab = pd.DataFrame(tab); tab.round(3).to_csv(PROJ/'36_1단계_기준영상_비교.csv', index=False, encoding='utf-8-sig')
    summ = P.groupby('region').agg(전체=('use', 'size'), 사용=('use', 'sum')).reset_index()
    print(tab.round(3)); print(summ); print('나사 중심', [tuple(round(v, 1) for v in s) for s in SCREWS], '문턱', round(thr, 3))

    # ---------- 그림 36: 기준 영상 3장 + 차이 ----------
    fig, axs = plt.subplots(2, 2, figsize=(18, 9.5))
    for ax, img, ttl in [(axs[0, 0], cen_ref, 'center 1장'), (axs[0, 1], refs[0], '학습 사진 평균 (폴드 0을 시험할 때)')]:
        ax.imshow(img, cmap='gray', vmin=0, vmax=255); ax.set_title(ttl, loc='left', fontsize=13); ax.axis('off')
    for ax, d, ttl in [(axs[1, 0], refs[0] - cen_ref, '평균 - center (밝기 차이)'), (axs[1, 1], refs[0] - refs[1], '평균(폴드0용) - 평균(폴드1용)')]:
        im_ = ax.imshow(d, cmap='RdBu_r', vmin=-8, vmax=8); ax.set_title(ttl, loc='left', fontsize=13); ax.axis('off')
        plt.colorbar(im_, ax=ax, fraction=0.02)
    t = tab.pivot(index='기준영상', columns='영역', values=['밝기', '선명도']).round(2)
    fig.text(0.01, 0.01, '밝기 / 선명도 (QR, 나사): ' + ' | '.join(f'{i}: 밝기 {t.loc[i, ("밝기", "QR")]}, {t.loc[i, ("밝기", "나사")]} / 선명도 {t.loc[i, ("선명도", "QR")]}, {t.loc[i, ("선명도", "나사")]}' for i in t.index),
             fontsize=10, color=SUB)
    fig.suptitle('1단계: 기준 영상 (판 영역, center 좌표). 평균 영상은 원근 변환으로 되돌린 뒤 3차 스플라인 보간으로 평균', x=0.01, ha='left', fontsize=15)
    plt.tight_layout(rect=(0, 0.03, 1, 0.96)); plt.savefig(PROJ/'36_1단계_기준영상.png', dpi=110, facecolor='white'); plt.close()

    # ---------- 그림 37: 패치 지도 ----------
    fig, ax = plt.subplots(figsize=(18, 5.6))
    ax.imshow(cen_ref, cmap='gray', vmin=0, vmax=255, extent=(X0, X1, Y1, Y0))
    col = {'QR좌': '#4a72d0', 'QR중': '#4a72d0', 'QR우': '#4a72d0', '나사좌': '#d2703f', '나사우': '#d2703f', '그 외': '#999999'}
    for _, r in P.iterrows():
        if r.use: ax.add_patch(Rectangle((r.cx - 6, r.cy - 6), 12, 12, fc=col[r.region], ec='none', alpha=0.8))
        elif r.region != '그 외': ax.plot(r.cx, r.cy, 'x', color='#d23c3c', ms=5, mew=1.2)
    for x, y in SCREWS: ax.add_patch(Circle((x, y), R_SCREW, fill=False, ec='#d2703f', lw=1.5, ls='--'))
    for x, y, hw, hh in QRBOX: ax.add_patch(Rectangle((x - hw, y - hh), 2 * hw, 2 * hh, fill=False, ec='#4a72d0', lw=1.5, ls='--'))
    s = summ.set_index('region')
    ax.set_title('2단계: 패치 지도 (32x32px, 16px 겹침). 파랑 = QR 패치, 주황 = 나사 패치, 빨강 x = 무늬가 약해서 뺀 패치   '
                 + ', '.join(f'{k} {int(s.loc[k, "사용"])}/{int(s.loc[k, "전체"])}' for k in ['QR좌', 'QR중', 'QR우', '나사좌', '나사우']),
                 loc='left', fontsize=12)
    ax.set_xlim(X0, X1); ax.set_ylim(Y1, Y0)
    plt.tight_layout(); plt.savefig(PROJ/'37_2단계_패치지도.png', dpi=110, facecolor='white'); plt.close()
    print('saved')
