"""정밀 위상 상관(업샘플링 DFT, Guizar-Sicairos 2008, skimage.registration.phase_cross_correlation)으로 나사 이동을 잰다.

cv2.phaseCorrelate 는 정수 봉우리 주변 무게중심으로 소수 위치를 잡아 치우침이 있다. 여기서는 봉우리 주변을 1/100 px 로 다시 계산한다.
조각은 measure_pc.py 가 캐시 폴더에 저장한 것을 쓴다(같은 위치, 같은 Hann 창).
- pcu_raw     : 141 조각(초기값 위치), 밝기
- pcu_grad    : 141 조각, 기울기 영상
- pcu_raw_ni  : 257 조각(초기값 없음), 밝기
- pcu_grad_ni : 257 조각, 기울기 영상
출력: 결과/pcu_meas.npz
실행: 저장소 루트에서  python "<이 폴더>/실험/measure_pc_up.py" <캐시 폴더>
"""
import sys, pathlib, time, numpy as np, cv2
from skimage.registration import phase_cross_correlation

EXP = pathlib.Path(__file__).resolve().parent.parent; OUT = EXP / "결과"; CACHE = pathlib.Path(sys.argv[1])
UP = 100
off = np.fromfile(CACHE / "off.i32", np.int32).reshape(-1, 4, 2); n = len(off)
C = {141: (np.fromfile(CACHE / "refs141.u8", np.uint8).reshape(4, 141, 141), np.fromfile(CACHE / "crops141.u8", np.uint8).reshape(n, 4, 141, 141)),
     257: (np.fromfile(CACHE / "refs257.u8", np.uint8).reshape(4, 257, 257), np.fromfile(CACHE / "crops257.u8", np.uint8).reshape(n, 4, 257, 257))}


def grad(a):
    a = cv2.GaussianBlur(a.astype(np.float32), (0, 0), 1.5)
    return cv2.magnitude(cv2.Sobel(a, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(a, cv2.CV_32F, 0, 1, ksize=3))


def prep(a, w, g):
    a = grad(a).astype(np.float64) if g else a.astype(np.float64)
    return (a - a.mean()) * w


out = {}
for key, size, g in (("pcu_raw", 141, False), ("pcu_grad", 141, True), ("pcu_raw_ni", 257, False), ("pcu_grad_ni", 257, True)):
    refs, crops = C[size]; w = cv2.createHanningWindow((size, size), cv2.CV_64F); base = off if size == 141 else np.zeros_like(off)
    F = [np.fft.fft2(prep(r, w, g)) for r in refs]; v = np.zeros((n, 4, 2)); t0 = time.perf_counter()
    for i in range(n):
        for k in range(4):
            s = phase_cross_correlation(F[k], np.fft.fft2(prep(crops[i, k], w, g)), upsample_factor=UP, space="fourier", normalization="phase")[0]
            v[i, k] = base[i, k] + (-s[1], -s[0])       # skimage: (행, 열) 순서, ref 를 맞추려면 움직일 양 → 부호 반대
    print(key, f"{(time.perf_counter() - t0) / n / 4 * 1000:.2f} ms/나사 (Python)", flush=True); out[key] = v.reshape(n, 8)
np.savez(OUT / "pcu_meas.npz", **out)
