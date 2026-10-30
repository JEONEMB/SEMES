# 정리본 v3 10번 '노이즈 – Trial별 노이즈 분석' 표의 '크기' 칸을 v2 분석 결과(img/*.csv)와 맞춤
#   11번 0점 어긋남 설명의 µm 도 같은 값으로
import pathlib, pandas as pd, numpy as np
from pptx import Presentation
HERE = pathlib.Path(__file__).parent; DECK = HERE.parent; IMG = HERE / 'img'
V3 = DECK / 'SEMES_01_데이터EDA_정리본_v3.pptx'; E = 914400
Z = pd.read_csv(IMG / 'zero_offsets.csv').set_index('trial'); y = Z[Z.axis == 'Y']
R = pd.read_csv(IMG / 'common_resid.csv').groupby('trial').common_px.std()
B = pd.read_csv(IMG / 'brightness.csv'); bc = B[B.file == 'center.bmp'].plate_mean.iloc[0]; bo = B[B.file != 'center.bmp'].plate_mean.mean()
P = pd.read_csv(IMG / 'photo_motion.csv'); dy_t = P[P.trial.str.startswith('Θ')].dy.abs().max()
zp, zu = y.zero_px.max() - y.zero_px.min(), y.zero_um.max() - y.zero_um.min()
SIZE = {'0점 어긋남': f'trial끼리 최대 {zp:.2f}px ({zu:.0f}µm), Θ +쪽 {Z.zero_px["Θ +"]:.2f}px',
        '사진 공통 흔들림': f'가로 {R.min():.3f} ~ {R.max():.3f}px',
        '점 하나 흔들림': '가로 0.047px, 세로 0.033px',            # n7.png 중앙값 (12번 그림과 같음)
        '대각선 모서리': f'0.16px (다른 점의 약 {0.159 / 0.047:.1f}배)',  # n8.png
        '밝기 변화': f'center만 +{100 * (bc / bo - 1):.1f}%, trial 안에서 서서히 변함',
        '세로 밀림': f'Θ trial에서 최대 {dy_t:.2f}px'}
prs = Presentation(V3); S = list(prs.slides)
tbl = next(s for s in S if any(sh.has_text_frame and sh.text_frame.text == '노이즈 – Trial별 노이즈 분석' for sh in s.shapes))
def put(sh, t):
    rs = [r for p in sh.text_frame.paragraphs for r in p.runs]; rs[0].text = t
    for r in rs[1:]: r.text = ''
    for p in sh.text_frame.paragraphs[1:]: p._p.getparent().remove(p._p)
boxes = [sh for sh in tbl.shapes if sh.has_text_frame]
for sh in boxes:
    if sh.text_frame.text in SIZE:
        put(next(b for b in boxes if b.top == sh.top and 2.5 < b.left / E < 3.0), SIZE[sh.text_frame.text])
for s in S:
    for sh in s.shapes:
        if sh.has_text_frame and sh.text_frame.text.startswith('trial마다 0점이 다름'):
            put(sh, f'trial마다 0점이 다름. Y는 최대 {zp:.2f}px({zu:.0f}µm), Θ +쪽은 {Z.zero_px["Θ +"]:.2f}px. 6쪽 참고.')
prs.save(V3)
for k, v in SIZE.items(): print(k, '|', v)
