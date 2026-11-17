import sys, cv2, numpy as np
sys.path.insert(0, '.')
D = sys.argv[1]
g = lambda p: cv2.imread(p, 0).astype(np.float32)
ref = g(D + '/center.bmp')
P = 256; win = cv2.createHanningWindow((P, P), cv2.CV_32F)
for name in sys.argv[2:]:
    cur = g(D + '/' + name + '.bmp')
    print(name, 'global', cv2.phaseCorrelate(ref, cur))
    rows = []
    for y0 in range(0, 2160 - P + 1, 256):
        row = []
        for x0 in range(0, 3840 - P + 1, 256):
            (dx, dy), r = cv2.phaseCorrelate(ref[y0:y0+P, x0:x0+P], cur[y0:y0+P, x0:x0+P], win)
            row.append(f"{dx:6.1f}" if r > 0.15 else "   .  ")
        print(' '.join(row))
