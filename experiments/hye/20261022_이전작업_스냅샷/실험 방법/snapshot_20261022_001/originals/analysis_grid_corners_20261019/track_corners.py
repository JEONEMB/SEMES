# QR 디코딩 격자 모서리 755점(center에서 ID 고정)을 280장 전부에서 같은 방식으로 서브픽셀 맞춤
#   방법: codex_experiments_20261014/subpixel_grid_corners_20261015/refine.py 와 같은 경계 모델, 같은 통과 기준
#   사진마다: QR별 템플릿 매칭으로 대략 이동(정수 px) -> 시작점 = center 맞춤 위치 + 대략 이동
#   9x9 창으로 맞춘 뒤 창을 맞춘 위치(가장 가까운 픽셀)로 옮겨 다시 맞춤(최대 3번, 창이 안 바뀌면 멈춤)
#     -> 창이 시작점에 따라 1px 밀려 결과가 0.1px까지 달라지던 문제를 없앰 (refine.py 와의 유일한 차이)
#   마지막 위치에서 11x11 창으로 한 번 더 맞춰 두 창의 차이를 안정성 진단에 씀
#   통과 기준 (하나라도 걸리면 '불확실'): 최적화 수렴, 밝기 모델 RMSE <= 15, 대비 >= 45,
#                                     두 창 결과 차이 <= 0.65px, 시작점에서 3.6px 이내, 흐림 σ <= 2.6
import json, pathlib, time, numpy as np, pandas as pd, cv2
from multiprocessing import Pool
from scipy.optimize import least_squares
from scipy.special import erf
HERE = pathlib.Path(__file__).parent
ROOT = pathlib.Path(r"C:\Users\SSAFY\Downloads\[기업연계_세메스] 교보재\[기업연계_세메스] 교보재\extracted\SSAFY Test Image Set")
SRC = HERE.parent / 'codex_experiments_20261014'
BOXES = [(1552, 969, 1736, 1167), (1809, 971, 1993, 1169), (2067, 973, 2252, 1171)]   # measurement/measure.py 와 같음
META = json.loads((SRC / 'decoded_grid_corners_20261015' / 'metadata.json').read_text(encoding='utf-8'))
REF = pd.read_csv(SRC / 'subpixel_grid_corners_20261015' / 'refined_corners.csv')      # center 맞춤 결과 (ID, 위치)
read = lambda p: cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_GRAYSCALE)

def field(xy, center, axes, sigma, patch):
    local = (xy - center) @ np.linalg.inv(axes).T
    U = .5 * (1 + erf(local[:, 0] / (np.sqrt(2) * sigma[0]))); V = .5 * (1 + erf(local[:, 1] / (np.sqrt(2) * sigma[1])))
    return patch[0, 0] * (1 - U) * (1 - V) + patch[0, 1] * U * (1 - V) + patch[1, 0] * (1 - U) * V + patch[1, 1] * U * V

def fit(xy, I, start, axes, patch):
    low, high = np.percentile(I, [10, 90]); contrast = max(high - low, 20)
    p0 = [*start, 1., 1., low, contrast]
    lo = [start[0] - 3.8, start[1] - 3.8, .35, .35, 0, 10]; hi = [start[0] + 3.8, start[1] + 3.8, 2.7, 2.7, 240, 255]
    res = lambda p: p[4] + p[5] * (1 - field(xy, p[:2], axes, p[2:4], patch)) - I
    r = least_squares(res, p0, bounds=(lo, hi), loss='soft_l1', f_scale=6, max_nfev=85, xtol=1e-7, ftol=1e-7, gtol=1e-7)
    return r.x, float(np.sqrt(np.mean(res(r.x) ** 2))), r.success

# 모서리마다 고정 정보: 국소 축(격자 방향), 2x2 검정 배치
from PIL import Image
CORNERS = []
for item in META['qr']:
    k, n = item['qr'], item['modules']; q = np.array(item['quad'], np.float32)
    H = cv2.getPerspectiveTransform(np.float32([[0, 0], [n, 0], [n, n], [0, n]]), q)
    grid = Image.open(SRC / 'decoded_grid_corners_20261015' / f'QR{k}_decoded_grid.png').convert('L').resize((n, n), Image.Resampling.NEAREST)
    black = np.pad((np.array(grid) < 128).astype(float), 1)
    for r in REF[REF.qr == k].itertuples():
        u, v = int(r.grid_x), int(r.grid_y)
        pr = cv2.perspectiveTransform(np.float32([[[u, v], [u + .1, v], [u, v + .1]]]), H)[0]
        axes = (pr[1:] - pr[0]).T; axes /= np.linalg.norm(axes, axis=0)
        CORNERS.append(dict(id=len(CORNERS), qr=k, gx=u, gy=v, cx=r.x, cy=r.y, axes=axes, patch=black[v:v + 2, u:u + 2]))

def coarse_shift(img, cen):
    out = []
    for a, b, c, d in BOXES:
        tpl = cen[b:d, a:c]; pad = 130
        m = cv2.matchTemplate(img[b - pad:d + pad, a - pad:c + pad], tpl, cv2.TM_CCOEFF_NORMED)
        _, _, _, loc = cv2.minMaxLoc(m); out.append((loc[0] - pad, loc[1] - pad))
    return out

def run(fname):
    p = ROOT / 'center.bmp' if fname == 'center.bmp' else ROOT / ('Theta' if fname[0] == 'T' else 'Y') / fname
    img = read(p).astype(float); cen = read(ROOT / 'center.bmp')
    sh = coarse_shift(img.astype(np.uint8), cen)
    rows = []
    for c in CORNERS:
        k = c['qr']; dx, dy = sh[k - 1]; start = np.array([c['cx'] + dx, c['cy'] + dy])
        a, b, cc, d = BOXES[k - 1]; a, cc, b, d = a + dx, cc + dx, b + dy, d + dy
        def window(center, radius):
            x0, y0 = np.rint(center).astype(int)
            yy, xx = np.mgrid[y0 - radius:y0 + radius + 1, x0 - radius:x0 + radius + 1]
            xy = np.c_[xx.ravel(), yy.ravel()].astype(float); I = img[yy, xx].ravel()
            ins = (xy[:, 0] >= a) & (xy[:, 0] < cc) & (xy[:, 1] >= b) & (xy[:, 1] < d)
            return xy[ins], I[ins]
        est = start
        for _ in range(3):   # 창 다시 맞추기
            f1 = fit(*window(est, 4), start, c['axes'], c['patch'])
            moved = np.any(np.rint(f1[0][:2]) != np.rint(est)); est = f1[0][:2]
            if not moved: break
        fits = [f1, fit(*window(est, 5), start, c['axes'], c['patch'])]
        (p1, rmse, ok), (p2, _, ok2) = fits
        stab = float(np.linalg.norm(p1[:2] - p2[:2]))
        reasons = []
        if not (ok and ok2): reasons.append('optimizer')
        if rmse > 15: reasons.append('photometric_rmse')
        if p1[5] < 45: reasons.append('low_contrast')
        if stab > .65: reasons.append('window_instability')
        if np.max(abs(p1[:2] - start)) > 3.6: reasons.append('search_limit')
        if max(p1[2:4]) > 2.6: reasons.append('blur_limit')
        rows.append((fname, c['id'], k, c['gx'], c['gy'], int(c['patch'].sum()), p1[0], p1[1], rmse, stab, p1[5], not reasons, ';'.join(reasons)))
    return rows

if __name__ == '__main__':
    files = ['center.bmp'] + sorted(f.name for f in (ROOT / 'Y').glob('*.bmp')) + sorted(f.name for f in (ROOT / 'Theta').glob('*.bmp'))
    t0 = time.time()
    with Pool(20) as pool:
        out = []
        for i, rows in enumerate(pool.imap_unordered(run, files)):
            out += rows
            if i % 20 == 0: print(i, len(files), round(time.time() - t0), 's', flush=True)
    df = pd.DataFrame(out, columns=['file', 'id', 'qr', 'grid_x', 'grid_y', 'black_neighbors', 'x', 'y', 'rmse', 'window_stability_px', 'contrast', 'valid', 'reason'])
    df.to_parquet(HERE / 'out' / 'corners_280.parquet') if False else df.to_csv(HERE / 'out' / 'corners_280.csv', index=False, encoding='utf-8-sig')
    print('done', len(files), 'photos', len(df), 'rows', round(time.time() - t0), 's')
