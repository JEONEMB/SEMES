"""D2, D7: 합성 이동 시험. center 원본을 알고 있는 양만큼 옮긴 가짜 사진으로 특징점 위치 재기의 편향과 흩어짐을 잰다.
가짜 사진은 푸리에 이동(측정에 쓰는 스플라인과 다른 보간)으로 만들고, 잡음(σ 0.73, 평평한 벽에서 잰 값)을 더한 뒤
0~255 정수로 반올림한다 (실제 사진도 8비트). 이론 한계(크라메르-라오)도 특징점마다 계산한다."""
import sys, numpy as np, pandas as pd
from numpy.fft import fft2, ifft2
from scipy import ndimage
from common import *

NOISE = 0.73
DY = lambda s: 0.6 * (abs(s) % 1 if abs(s) > 1 else s)   # 큰 이동은 가로만 크게 (실제 세로 이동은 1px 이하)
cen = read_raw('center.bmp')
SH = [0.0, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 40.37, -80.63]     # 마지막은 큰 이동 (정수 + 소수)
FT = fft2(cen.astype(np.float64))

def shifted(s):
    """장면을 (+s, +0.6s) 옮긴 사진. 큰 이동도 푸리에로 (가장자리는 감기지만 특징점은 안쪽)."""
    return np.real(ifft2(ndimage.fourier_shift(FT, (DY(s), s))))

if __name__ == '__main__':
    rng = np.random.default_rng(0); rows = []
    gx = cv2.Sobel(cen, cv2.CV_64F, 1, 0, ksize=3) / 8; gy = cv2.Sobel(cen, cv2.CV_64F, 0, 1, ksize=3) / 8
    for hp in [8, 12]:
        F = pick_features(cen, hp)
        # 크라메르-라오: 원본 기울기, 잡음 σ 0.73 → x 방향 표준편차 하한
        crb = []
        for x, y in zip(F.x, F.y):
            sl = (slice(y - hp, y + hp + 1), slice(x - hp, x + hp + 1))
            G = np.array([[(gx[sl]**2).sum(), (gx[sl] * gy[sl]).sum()], [(gx[sl] * gy[sl]).sum(), (gy[sl]**2).sum()]])
            crb.append(NOISE * np.sqrt(np.linalg.inv(G)[0, 0]))
        F['crb_x'] = crb
        F.to_csv(HERE/f'features_hp{hp}.csv', index=False, encoding='utf-8-sig')
        Tr, Tb = cut(cen, F, hp), cut(blur(cen), F, hp)
        BIG = cut(np.pad(cen, HPC, mode='edge'), F.assign(x=F.x + HPC, y=F.y + HPC), HPC)
        print(f'hp {hp}: 특징점 {len(F)}', F.region.value_counts().to_dict(), '이론 한계 x 중앙값', round(F.crb_x.median(), 4), flush=True)
        for s in SH:
            for rep in range(3):
                img = np.clip(np.round(shifted(s) + rng.normal(0, NOISE, cen.shape)), 0, 255).astype(np.float32)
                c0, _ = coarse_init(np.pad(img, HPC, mode='edge'), BIG, F.assign(x=F.x + HPC, y=F.y + HPC))
                tm, sc = track_tm(img, Tr, F, hp, init=c0)
                gn = track_gn(blur(img), Tb, F, hp, tm)
                for nm, m in [('TM', tm), ('GN', gn)]:
                    for ax, true in [(0, s), (1, DY(s))]:
                        e = m[:, ax] - true
                        for reg in ['QR', '판', '판밖']:
                            msk = F.region.str.startswith(reg).values if reg == 'QR' else (F.region == reg).values
                            if not msk.any(): continue
                            rows.append(dict(hp=hp, 이동=s, 반복=rep, 측정법=nm, 방향='dx' if ax == 0 else 'dy', 영역=reg, n=msk.sum(),
                                             편향=e[msk].mean(), 흩어짐=e[msk].std(), 최대=np.abs(e[msk]).max(), 이론한계=F.crb_x[msk].median()))
            print(' ', s, 'done', flush=True)
    R = pd.DataFrame(rows); R.to_csv(HERE/'t1_synthetic_raw.csv', index=False, encoding='utf-8-sig')
    S = R.groupby(['hp', '측정법', '방향', '영역', '이동']).agg(n=('n', 'first'), 편향=('편향', 'mean'), 흩어짐=('흩어짐', 'mean'),
                                                             최대=('최대', 'max'), 이론한계=('이론한계', 'first')).reset_index()
    S.round(4).to_csv(HERE/'t1_synthetic.csv', index=False, encoding='utf-8-sig')
    print(S[S.방향 == 'dx'].round(4).to_string())


