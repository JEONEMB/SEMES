# 기초 정보 슬라이드용 QR 확대 이미지: 모든 사진을 같은 창(center의 가운데 QR 기준)으로 잘라서 이동이 위치 차이로 보이게 함
import cv2, numpy as np, pathlib
ROOT = pathlib.Path(r"C:\Users\SSAFY\Downloads\[기업연계_세메스] 교보재\[기업연계_세메스] 교보재\extracted\SSAFY Test Image Set")
OUT = pathlib.Path(__file__).parent / 'img'
X0, X1, Y0, Y1 = 1720, 2080, 970, 1170   # 가운데 QR 하나 + 좌우 최대 이동(약 81px) 여유
FILES = ['center.bmp', 'Y+0.10.bmp', 'Y+2.00.bmp', 'Y-0.10.bmp', 'Y-2.00.bmp',
         'Y+0.05.bmp', 'Y+1.95.bmp', 'Y-0.05.bmp', 'Y-1.95.bmp',
         'T+0.01.bmp', 'T+1.00.bmp', 'T-0.01.bmp', 'T-1.00.bmp']
for f in FILES:
    p = ROOT / f if f == 'center.bmp' else ROOT / ('Theta' if f[0] == 'T' else 'Y') / f
    im = cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_GRAYSCALE)[Y0:Y1, X0:X1]
    ok, buf = cv2.imencode('.png', im)
    buf.tofile(str(OUT / ('qr_' + f.replace('.bmp', '.png'))))
print("crop", X1 - X0, "x", Y1 - Y0, "center QR x in crop:", round(1900.49 - X0, 1))
