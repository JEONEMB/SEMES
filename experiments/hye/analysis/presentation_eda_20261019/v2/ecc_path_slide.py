# 정리본 v3 8번 '이동 관계 – 이동 경로 (ECC)': 새 ECC 측정(ecc_measure.py -> ecc_path.py)으로 그림, 표, 메모 교체
import pathlib, numpy as np, pandas as pd
from PIL import Image
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE_TYPE
from pptx.oxml.ns import qn
HERE = pathlib.Path(__file__).parent; DECK = HERE.parent; IMG = HERE / 'img'
import sys
V3 = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else DECK / 'SEMES_01_데이터EDA_정리본_v3.pptx'   # 인자로 다른 파일을 주면 그 파일을 고침
SUB, LINE, FAINT = (RGBColor.from_string(c) for c in ['4B5563', 'E5E7EB', '9CA3AF'])
FONT, E = 'Noto Sans KR', 914400
KX, KW, VX, VW, LW = 7.21, 1.3, 8.51, 4.03, 5.33
Z = pd.read_csv(IMG / 'path_ecc_stats.csv').set_index('trial'); y = Z[Z.axis == 'Y']; t = Z[Z.axis == 'Θ']
M = pd.read_csv(IMG / 'ecc_motion.csv').set_index('file'); C = pd.read_csv(IMG / 'photo_motion.csv').set_index('file')
ddx, ddy = (M.dx - C.dx).abs().max(), (M.dy - C.dy).abs().max(); k = Z.dy_slope.iloc[0]

prs = Presentation(V3)
title = lambda s: next((sh.text_frame.text for sh in s.shapes if sh.has_text_frame and sh.top is not None and 0.8 < sh.top / E < 1.2 and not sh.text_frame.text.startswith('1-1')), '')
s = next(s for s in prs.slides if title(s) == '이동 관계 – 이동 경로 (ECC)')
def text(x, y_, w, h, lines, size, color=SUB):
    tb = s.shapes.add_textbox(Inches(x), Inches(y_), Inches(w), Inches(h)); tf = tb.text_frame; tf.word_wrap = True
    for m in ('margin_left', 'margin_right', 'margin_top', 'margin_bottom'): setattr(tf, m, 0)
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        r = p.add_run(); r.text = line; r.font.size = Pt(size); r.font.color.rgb = color; r.font.name = FONT
        rpr = r._r.get_or_add_rPr(); ea = rpr.find(qn('a:ea'))
        if ea is None: ea = rpr.makeelement(qn('a:ea'), {}); rpr.append(ea)
        ea.set('typeface', FONT)

for sh in list(s.shapes):   # 그림, 그림 설명, 오른쪽 표 지우고 다시 그림
    if sh.is_placeholder or sh.top is None: continue
    if sh.shape_type == MSO_SHAPE_TYPE.PICTURE or (sh.top / E > 1.8 and (sh.left / E >= 6.95 or (sh.has_text_frame and sh.text_frame.text.startswith('ECC')))):
        sh._element.getparent().remove(sh._element)
iw, ih = Image.open(IMG / 'path_ecc.png').size; h = 6.0 * ih / iw
s.shapes.add_picture(str(IMG / 'path_ecc.png'), Inches(0.8), Inches(1.95), Inches(6.0), Inches(h))
text(0.8, 1.95 + h + 0.05, 6.0, 0.4, ['ECC: QR 3개 패치를 center에 맞춘 가운데 점 이동의 평균. 가로 ×1, 세로 ×100 (특징점 장과 같은 축).'], 9, FAINT)
rows = [('Y 기울기', [f'trial별 {y.slope_px.min():.2f} ~ {y.slope_px.max():.2f}px / 1mm']),
        ('Θ 기울기', [f'+쪽 {Z.slope_px["Θ +"]:.2f}px, -쪽 {Z.slope_px["Θ -"]:.2f}px / Θ 1', f'(+쪽과 -쪽이 {100 * abs(Z.slope_px["Θ -"] / Z.slope_px["Θ +"] - 1):.1f}% 다름)']),
        ('직선 잔차', [f'Y {y.resid_px.min():.3f} ~ {y.resid_px.max():.3f}px,  Θ {t.resid_px.min():.3f} ~ {t.resid_px.max():.3f}px (σ)']),
        ('세로 이동', [f'가로 이동의 {100 * k:.2f}% (이동 방향 {np.degrees(np.arctan(k)):.2f}° 기울어짐)']),
        ('특징점과 차이', [f'사진마다 가로 최대 {ddx:.3f}px, 세로 최대 {ddy:.3f}px', '두 방법의 경로와 숫자가 거의 같음'])]
top, bottom, lh = 2.03, 6.43, 0.2
pad = (bottom - top - sum(len(v) for _, v in rows) * lh) / len(rows); yy = top
for kk, vals in rows:
    text(KX, yy + pad / 2 - 0.02, KW, 0.3, [kk], 12); text(VX, yy + pad / 2, VW, len(vals) * lh, vals, 11)
    yy += len(vals) * lh + pad
    ln = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(KX), Inches(yy), Inches(KX + LW), Inches(yy)); ln.line.color.rgb = LINE; ln.line.width = Pt(0.75)
s.notes_slide.notes_text_frame.text = f'''[측정]
presentation_eda_20261019/v2/ecc_measure.py -> v2/img/ecc_motion.csv (원본 281장을 새로 측정)

[숫자, 그림]
v2/ecc_path.py -> v2/img/path_ecc_stats.csv, v2/img/path_ecc.png (특징점 장 img/path_corner.png 와 같은 축, 같은 배율)

[방법]
① center 사진에서 QR 라벨을 찾아 라벨 전체 + 바깥 8px 를 기준 패치로 둠 (v2/img/ecc_boxes.csv)
② 다른 사진마다 템플릿 매칭으로 대략 위치를 찾고, ECC 아핀으로 QR 패치 전체를 center 패치에 맞춤
③ 패치 가운데 점이 변환으로 옮겨진 양 = 그 QR의 이동
④ QR 3개 평균을 사진의 이동으로 보고, trial마다 명령값 크기 순으로 이어 그림
- 기울기, 직선 잔차, 세로 이동 비율은 특징점 장과 같은 방법으로 계산
- 특징점(모서리 630점 평균)과 사진별 차이: 가로 최대 {ddx:.3f}px, 세로 최대 {ddy:.3f}px'''
prs.save(V3)
for kk, v in rows: print(kk, '|', ' / '.join(v))
