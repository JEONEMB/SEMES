# QR 네모를 사진에서 직접 잰다: QR 라벨(밝은 사각형) 안쪽의 어두운 모듈이 차지하는 범위 = QR 영역
# (14_데이터_QR마커위치.csv 의 폭, 높이는 OpenCV QR 검출기의 정수 꼭짓점이라 실제 경계와 1~4px 어긋남)
import cv2, numpy as np

def qr_bbox(gray, cx, cy, half=150):
    x0, y0 = int(cx) - half, int(cy) - half
    roi = cv2.GaussianBlur(gray[y0:y0 + 2 * half, x0:x0 + 2 * half], (0, 0), 1)
    t, _ = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    n, lab, st, _ = cv2.connectedComponentsWithStats((roi > t).astype(np.uint8))
    k = 1 + np.argmax(st[1:, 4]); lx, ly, lw, lh = st[k, :4]          # 가장 큰 밝은 덩어리 = QR 라벨
    m = 5
    ys, xs = np.nonzero(roi[ly + m:ly + lh - m, lx + m:lx + lw - m] < t)  # 라벨 안쪽 어두운 모듈
    return (x0 + lx + m + xs.min(), y0 + ly + m + ys.min(), x0 + lx + m + xs.max() + 1, y0 + ly + m + ys.max() + 1)
