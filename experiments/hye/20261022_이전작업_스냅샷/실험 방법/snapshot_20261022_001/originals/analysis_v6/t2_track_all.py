"""실제 사진 281장에서 특징점 이동 재기 (TM: v1 방식, GN: 2단계 탐색 + 가우스-뉴턴). 기준 = center (흐림 σ1).
출력: tracks_hp{hp}.npz  D_tm, D_gn (281, N, 2), score (281, N), names"""
import sys, time
from multiprocessing import Pool
from common import *

HP = int(sys.argv[1]) if len(sys.argv) > 1 else 8
cen = read_raw('center.bmp')
F = pd.read_csv(HERE/f'features_hp{HP}.csv')
Tr, Tb = cut(cen, F, HP), cut(blur(cen), F, HP)
FP = F.assign(x=F.x + HPC, y=F.y + HPC)
BIG = cut(np.pad(cen, HPC, mode='edge'), FP, HPC)

def job(n):
    img = read_raw(n)
    c0, _ = coarse_init(np.pad(img, HPC, mode='edge'), BIG, FP)
    tm, sc = track_tm(img, Tr, F, HP, init=c0)
    gn = track_gn(blur(img), Tb, F, HP, tm)
    return tm, gn, sc

if __name__ == '__main__':
    t0 = time.time(); out = []
    with Pool(16) as pool:
        for i, r in enumerate(pool.imap(job, NAMES)):
            out.append(r)
            if i % 40 == 0: print(i, NAMES[i], f'{time.time() - t0:.0f}s', flush=True)
    np.savez_compressed(HERE/f'tracks_hp{HP}.npz', D_tm=np.stack([o[0] for o in out]), D_gn=np.stack([o[1] for o in out]),
                        score=np.stack([o[2] for o in out]), names=np.array(NAMES))
    print('done', f'{time.time() - t0:.0f}s')
