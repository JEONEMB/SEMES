import sys, glob, os, cv2, numpy as np
D20, D14 = sys.argv[1], sys.argv[2]
g = lambda p: cv2.imdecode(np.fromfile(p, np.uint8), 0).astype(np.float32)
roi = (slice(256, 1792), slice(1280, 2560))   # 판(plate) 영역
win = cv2.createHanningWindow((1280, 1536), cv2.CV_32F)
pc = lambda a, b: cv2.phaseCorrelate(a[roi], b[roi], win)[0]
c14 = g(D14 + '/center.bmp')
print('Y scale on plate (1014): Y+2.00 dx =', round(pc(c14, g(D14 + '/Y/Y+2.00.bmp'))[0], 2), ' T+1.00 dx =', round(pc(c14, g(D14 + '/Theta/T+1.00.bmp'))[0], 2))
dirs = sorted(d for d in glob.glob(D20 + '/*/') if '_metadata' not in d)
c1 = g(sorted(glob.glob(D20 + '/기준 위치/*.bmp'))[0])
print('vs 1014 center: 1020 center1 dx,dy =', [round(v, 3) for v in pc(c14, c1)])
sx, sy, sb = [], [], []
for d in dirs:
    bm = sorted(glob.glob(d + '*.bmp')); im = [g(p) for p in bm]
    s = [pc(im[0], im[k]) for k in (1, 2)]
    sx += [v[0] for v in s]; sy += [v[1] for v in s]; sb += [im[k].mean() - im[0].mean() for k in (1, 2)]
    m = pc(c1, im[0])
    print(f"{os.path.basename(d[:-1]):>16} vs center1 dx={m[0]:8.2f} dy={m[1]:6.2f} | repeat dx {s[0][0]:+.3f} {s[1][0]:+.3f} dy {s[0][1]:+.3f} {s[1][1]:+.3f}")
print(f"repeat shift: dx std {np.std(sx):.3f} max|{np.max(np.abs(sx)):.3f}|, dy std {np.std(sy):.3f} max|{np.max(np.abs(sy)):.3f}|, brightness diff std {np.std(sb):.3f} max|{np.max(np.abs(sb)):.3f}|")
