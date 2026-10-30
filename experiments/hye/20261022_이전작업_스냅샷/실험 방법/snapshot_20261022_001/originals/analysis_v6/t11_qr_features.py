"""QR 판만 (QR 3개 + 판 나사 2개). 판밖 점은 쓰지 않는다.
특징점 세트: 템플릿 17px (간격 17), 9px (간격 9, 최대한 많이). 나사는 와셔 포화를 허용하고 가장자리 점을 잡는다.
출력: features_qr_hp{hp}.csv, tracks_qr_hp{hp}.npz"""
import sys, time
from multiprocessing import Pool
from common import *

def pick_qr(cen_raw, hp):
    b = blur(cen_raw); md = 2 * hp + 1
    m = np.zeros(b.shape, np.uint8); m[Y0:Y1, X0:X1] = 255
    pts = cv2.goodFeaturesToTrack(b, maxCorners=5000, qualityLevel=0.01, minDistance=md, blockSize=5 if hp < 6 else 7, mask=m).reshape(-1, 2)
    rows = []
    for x, y in pts:
        xi, yi = int(round(x)), int(round(y)); r = region_of(xi, yi)
        if not (r.startswith('QR') or r == '판나사'): continue
        sat = (cen_raw[yi - hp:yi + hp + 1, xi - hp:xi + hp + 1] >= 250).mean()
        if r.startswith('QR') and sat > 0: continue
        if sat > 0.6: continue                         # 나사: 템플릿 대부분이 포화면 위치 정보 없음
        rows.append(dict(x=xi, y=yi, region=r, sat=sat))
    return pd.DataFrame(rows)

HP = int(sys.argv[1]) if len(sys.argv) > 1 else 8
cen = read_raw('center.bmp')
F = pick_qr(cen, HP); F.to_csv(HERE/f'features_qr_hp{HP}.csv', index=False, encoding='utf-8-sig')
Tr, Tb = cut(cen, F, HP), cut(blur(cen), F, HP)
FP = F.assign(x=F.x + HPC, y=F.y + HPC); BIG = cut(np.pad(cen, HPC, mode='edge'), FP, HPC)

def job(n):
    img = read_raw(n); c0, _ = coarse_init(np.pad(img, HPC, mode='edge'), BIG, FP)
    tm, sc = track_tm(img, Tr, F, HP, init=c0); gn = track_gn(blur(img), Tb, F, HP, tm)
    return gn, sc

if __name__ == '__main__':
    print(f'hp{HP}', F.region.value_counts().to_dict(), flush=True); t0 = time.time()
    with Pool(16) as p: out = p.map(job, NAMES)
    np.savez_compressed(HERE/f'tracks_qr_hp{HP}.npz', D_gn=np.stack([o[0] for o in out]), score=np.stack([o[1] for o in out]), names=np.array(NAMES))
    print('done', f'{time.time() - t0:.0f}s')
