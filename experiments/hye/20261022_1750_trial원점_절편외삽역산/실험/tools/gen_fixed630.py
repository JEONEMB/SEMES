# 630점 경계 측정(joint_center_630 measurement.py)의 고정 자산을 C++ 상수 헤더로 옮긴다. 값은 바꾸지 않는다.
import json, hashlib, sys, numpy as np
L = 'C:/Users/SSAFY/Desktop/hwlee/projects/2026-10_SEMES/00000.legacies/joint_center_630_20261022/assets/'
c = json.load(open(L + 'corners.json')); r = np.load(L + 'reference.npz')
assert (r['ids'] == np.array([x['id'] for x in c])).all()
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
inv = [np.linalg.inv(np.array(x['axes'], np.float32)).astype(np.float64).ravel() for x in c]  # 원래 코드와 같이 float32 역행렬
o = ['// 고정 자산: center.bmp 위 QR 격자 모서리 630개 (측정 결과가 아니라 측정 위치·모양 정의).',
     '// 생성: 실험/tools/gen_fixed630.py. 원본 00000.legacies/joint_center_630_20261022/assets/',
     '//   corners.json sha256 ' + sha(L + 'corners.json') + ' (cx, cy = 탐색 시작점, axes 의 float32 역행렬, patch)',
     '//   reference.npz sha256 ' + sha(L + 'reference.npz') + ' (points = 변위 기준점, regions)',
     '// 자산을 만든 격자 해독·부화소 모서리 경로(decoded/subpixel_grid_corners_20261015)는 옮기지 않았다.',
     '#pragma once', 'namespace fixed630 {']
o.append('const double kRef[630][2] = {' + ',\n'.join('{%.17g, %.17g}' % tuple(p) for p in r['points']) + '};')
o.append('const int kRegion[630] = {' + ','.join(str(int(x)) for x in r['regions']) + '};')
o.append('const double kBase[630][2] = {' + ',\n'.join('{%.17g, %.17g}' % (x['cx'], x['cy']) for x in c) + '};')
o.append('const double kInv[630][4] = {' + ',\n'.join('{' + ', '.join('%.17g' % v for v in a) + '}' for a in inv) + '};')
o.append('const double kPatch[630][4] = {' + ',\n'.join('{' + ', '.join('%.17g' % v for v in np.array(x['patch'], float).ravel()) + '}' for x in c) + '};')
o.append('}  // namespace fixed630')
open(sys.argv[1], 'w', encoding='utf-8', newline='\n').write('\n'.join(o) + '\n')
