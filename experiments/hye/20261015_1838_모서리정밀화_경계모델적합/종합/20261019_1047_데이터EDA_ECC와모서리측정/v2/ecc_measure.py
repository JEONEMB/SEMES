# ECC 새로 측정: 원본 BMP 281장 -> img/ecc_motion.csv (사진마다 QR 3개의 이동)
#   ① center 사진에서 QR 라벨(밝은 사각형)을 직접 찾고, 라벨 전체 + 바깥 MARGIN px 를 기준 패치로 둠 (세 QR 같은 규칙)
#      -> img/ecc_boxes.csv
#   ② 다른 사진마다 ±130px 범위에서 템플릿 매칭(정규화 상관)으로 정수 위치를 찾음
#   ③ 그 자리에서 ECC 아핀으로 패치 전체를 center 패치에 맞춤 (가우스 흐림 5px 창, 반복 200회, 수렴 1e-8)
#   ④ 이동 = 패치 가운데 점이 변환으로 옮겨진 양 (가운데 기준이라 회전, 크기 성분이 이동에 섞이지 않음)
import pathlib, re, numpy as np, pandas as pd, cv2
HERE = pathlib.Path(__file__).parent; IMG = HERE / 'img'
ROOT = pathlib.Path(r"C:\Users\SSAFY\Downloads\[기업연계_세메스] 교보재\[기업연계_세메스] 교보재\extracted\SSAFY Test Image Set")
PAD, MARGIN = 130, 8
SEED = pd.read_csv(IMG / 'corner_ref.csv').set_index('qr')   # 모서리 분석의 QR 중심 (라벨 찾기 시작점)

def label_box(gray, cx, cy, half=150):
    x0, y0 = int(cx) - half, int(cy) - half
    roi = cv2.GaussianBlur(gray[y0:y0 + 2 * half, x0:x0 + 2 * half], (0, 0), 1)
    t, _ = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    n, lab, st, _ = cv2.connectedComponentsWithStats((roi > t).astype(np.uint8))
    k = 1 + np.argmax(st[1:, 4]); lx, ly, lw, lh = st[k, :4]          # 가장 큰 밝은 덩어리 = QR 라벨
    return x0 + lx, y0 + ly, x0 + lx + lw, y0 + ly + lh
CRIT = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-8)
read = lambda p: cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_GRAYSCALE).astype(np.float32)

files = [('center', 0.0, ROOT / 'center.bmp')]
for ax, folder in [('Y', 'Y'), ('Θ', 'Theta')]:
    for p in sorted((ROOT / folder).glob('*.bmp')):
        m = re.fullmatch(r'[YT]([+-]\d+\.\d+)\.bmp', p.name)
        if m: files.append((ax, float(m.group(1)), p))
cen = read(files[0][2]); cen8 = cv2.imdecode(np.fromfile(str(files[0][2]), np.uint8), cv2.IMREAD_GRAYSCALE)
LB = {q: label_box(cen8, SEED.cx[q], SEED.cy[q]) for q in 'LCR'}
BOXES = {q: (x0 - MARGIN, x1 + MARGIN, y0 - MARGIN, y1 + MARGIN) for q, (x0, y0, x1, y1) in LB.items()}
import sys
OLD = '--oldbox' in sys.argv   # 비교용: 예전 measure_qr.py 의 고정 창 (180 x 192px)
if OLD: BOXES = {'L': (1554, 1734, 972, 1164), 'C': (1815, 1995, 972, 1164), 'R': (2073, 2253, 972, 1164)}
else: pd.DataFrame([dict(qr=q, label_x0=LB[q][0], label_y0=LB[q][1], label_x1=LB[q][2], label_y1=LB[q][3], x0=b[0], x1=b[1], y0=b[2], y1=b[3]) for q, b in BOXES.items()]).to_csv(IMG / 'ecc_boxes.csv', index=False)
print(BOXES)
T = {q: cen[y0:y1, x0:x1].copy() for q, (x0, x1, y0, y1) in BOXES.items()}
rows = []
for i, (ax, v, p) in enumerate(files):
    im = cen if ax == 'center' else read(p); row = dict(axis=ax, value=v, file=p.name)
    for q, (x0, x1, y0, y1) in BOXES.items():
        tpl = T[q]; h, w = tpl.shape
        search = im[y0 - PAD:y1 + PAD, x0 - PAD:x1 + PAD]
        r = cv2.matchTemplate(search, tpl, cv2.TM_CCOEFF_NORMED); _, ncc, _, loc = cv2.minMaxLoc(r)
        ox, oy = loc[0] - PAD, loc[1] - PAD
        tgt = im[y0 + oy:y1 + oy, x0 + ox:x1 + ox]
        Wm = np.eye(2, 3, dtype=np.float32)
        cc, Wm = cv2.findTransformECC(tpl, tgt, Wm, cv2.MOTION_AFFINE, CRIT, None, 5)
        c = np.array([(w - 1) / 2, (h - 1) / 2, 1.0]); moved = Wm @ c
        row.update({f'd{q}x': ox + moved[0] - c[0], f'd{q}y': oy + moved[1] - c[1], f'{q}_cc': cc, f'{q}_ncc': ncc,
                    f'{q}_rot_deg': np.degrees(np.arctan2(Wm[1, 0], Wm[0, 0])), f'{q}_scale': np.sqrt(abs(np.linalg.det(Wm[:, :2])))})
    rows.append(row)
    if i % 40 == 0: print(i, len(files), p.name, flush=True)
D = pd.DataFrame(rows)
D['dx'] = D[['dLx', 'dCx', 'dRx']].mean(1); D['dy'] = D[['dLy', 'dCy', 'dRy']].mean(1)
D.to_csv(IMG / ('ecc_motion_oldbox.csv' if OLD else 'ecc_motion.csv'), index=False, encoding='utf-8-sig'); print('WROTE', len(D))
