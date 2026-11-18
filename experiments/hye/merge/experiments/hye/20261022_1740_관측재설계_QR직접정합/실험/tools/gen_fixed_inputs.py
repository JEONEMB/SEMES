import numpy as np, pandas as pd, json, hashlib, sys
L='C:/Users/SSAFY/Desktop/hwlee/projects/2026-10_SEMES/00000.legacies/'
ref=L+'joint_center_630_20261022/assets/reference.npz'
r=np.load(ref)
man=L+'codex_experiments_20261014/joint_yt_reboot_20261022/direct/manifest.csv'
m=pd.read_csv(man)
sel=L+'codex_experiments_20261014/joint_yt_disentangle_20261022/selection.json'
s=json.load(open(sel))
sha=lambda p: hashlib.sha256(open(p,'rb').read()).hexdigest()
o=[]
o.append('// 고정 입력: 630점 기준 좌표, 바깥 4겹 fold, 안쪽 3겹 분할. 측정 결과가 아니라 실험 설계 상수다.')
o.append('// 생성: 실험/tools/gen_fixed_inputs.py 로 아래 원본에서 그대로 옮겼다(값 변경 없음).')
o.append('//   points/regions: 00000.legacies/joint_center_630_20261022/assets/reference.npz sha256 '+sha(ref))
o.append('//     (center.bmp 위 QR 격자 경계점 630개. 만든 경로 decoded_grid_corners -> subpixel_grid_corners 는 옮기지 않았다)')
o.append('//   fold: joint_yt_reboot_20261022/direct/manifest.csv 의 fold 열 (joint_center_630 splits.json 에서 유래, 10-20 center 는 fold 0)')
o.append('//   inner: joint_yt_disentangle_20261022/selection.json sha256 '+sha(sel)+' (sklearn StratifiedGroupKFold shuffle, C++ 재현 불가라 그대로 둔다)')
o.append('#pragma once')
o.append('namespace fixed {')
o.append('const double kPoints[630][2] = {')
o.append(',\n'.join('{%.17g, %.17g}'%(a,b) for a,b in r['points'])+'};')
o.append('const int kRegion[630] = {'+','.join(str(int(x)) for x in r['regions'])+'};')
o.append('const int kFold[341] = {'+','.join(str(int(x)) for x in m.fold)+'};')
o.append('// kInner[outer fold 0..4][inner 0..2]: train, validation 인덱스 (fold 4 = 전체 학습 재적용용 full_selection)')
def arr(v): return '{'+','.join(str(int(x)) for x in v)+'}'
for f in range(5):
    src=s[f] if f<4 else s[-1]['full_selection']
    for k,sp in enumerate(src['inner_splits']):
        o.append('const int kTr%d_%d[] = %s;'%(f,k,arr(sp['train'])))
        o.append('const int kVa%d_%d[] = %s;'%(f,k,arr(sp['validation'])))
o.append('struct Split { const int* tr; int ntr; const int* va; int nva; };')
o.append('const Split kInner[5][3] = {')
rows=[]
for f in range(5):
    src=s[f] if f<4 else s[-1]['full_selection']
    rows.append('{'+','.join('{kTr%d_%d,%d,kVa%d_%d,%d}'%(f,k,len(sp['train']),f,k,len(sp['validation'])) for k,sp in enumerate(src['inner_splits']))+'}')
o.append(',\n'.join(rows)+'};')
o.append('}  // namespace fixed')
open(sys.argv[1],'w',encoding='utf-8',newline='\n').write('\n'.join(o)+'\n')
print(len(s), [len(x) for x in [s[-1]]], s[-1]['full_selection'].keys())
