# 사용자가 손본 덱(SEMES_01_데이터EDA_기초정보추가.pptx)의 '기초 정보' 슬라이드만 직접 고친다 (다른 슬라이드와 위치는 그대로)
#  1) QR 확대 이미지: 사진 테두리 대신 QR에만 색 네모 (img/qrbox_*.png), 아래 글자에 가로 이동 px 추가
#  2) 오른쪽 표를 카드 4개(데이터, 촬영 대상, 이미지, 라벨)로 바꿈
import sys, pathlib, pandas as pd
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE, MSO_SHAPE_TYPE
from pptx.oxml.ns import qn

HERE = pathlib.Path(__file__).parent
F = HERE / (sys.argv[1] if len(sys.argv) > 1 else 'SEMES_01_데이터EDA_기초정보추가.pptx')   # 입력
OUT = HERE / (sys.argv[2] if len(sys.argv) > 2 else F.name)                               # 출력 (기본: 덮어쓰기)
SHIFT = pd.read_csv(HERE / 'img' / 'qrbox_shift_px.csv', index_col=0)['dCx_px']
INK, SUB, FAINT, ACC, BG = (RGBColor.from_string(c) for c in ['111827', '4B5563', '9CA3AF', '0071E3', 'FFFFFF'])
FONT = 'Noto Sans KR'
E = 914400

prs = Presentation(F)
s = next(sl for sl in prs.slides if any(sh.has_text_frame and sh.text_frame.text == '기초 정보' for sh in sl.shapes))
grp = next(sh for sh in s.shapes if sh.shape_type == MSO_SHAPE_TYPE.GROUP)

# ---- 1) QR 이미지 교체, 테두리와 점선 지우기, 이동 px 표시
kids = list(grp.shapes)
pics = [k for k in kids if k.shape_type == MSO_SHAPE_TYPE.PICTURE]
caps = [k for k in kids if k.has_text_frame]
for pic in pics:
    cap = min(caps, key=lambda c: abs(c.left - pic.left) + abs(c.top - (pic.top + pic.height)))
    name = cap.text_frame.text.strip()
    f = 'center' if name.startswith('center') else name.replace('Θ ', 'T').replace('Y ', 'Y').split()[0]
    _, rid = s.part.get_or_add_image_part(str(HERE / 'img' / f'qrbox_{f}.png'))
    pic._element.blipFill.blip.rEmbed = rid
    if f != 'center' and len(cap.text_frame.paragraphs[0].runs) == 1:
        r = cap.text_frame.paragraphs[0].add_run()
        r.text = f'   {SHIFT[f]:+.1f}px'
        r.font.size = Pt(8); r.font.color.rgb = FAINT
    for k in kids:   # 사진과 같은 자리의 테두리 네모, 사진 위 세로 점선
        if k.shape_type == MSO_SHAPE_TYPE.AUTO_SHAPE and k.left >= pic.left - 10 and k.left <= pic.left + pic.width and abs(k.top - pic.top) < 10 and (k.width == 0 or abs(k.width - pic.width) < 10):
            k._element.getparent().remove(k._element)
    kids = list(grp.shapes)
for sh in s.shapes:
    if sh.has_text_frame and sh.text_frame.text.startswith('모든 사진을 같은 위치로'):
        sh.text_frame.paragraphs[0].runs[0].text = '파란 점선 = center의 QR 위치,  초록 = 첫 이동,  빨강 = 최대 이동,  px = 가로 이동량 (모든 사진을 같은 위치로 자름)'
        sh.left, sh.width = Inches(0.8), Inches(11.7)
        for r in sh.text_frame.paragraphs[0].runs[1:]: r.text = ''

# ---- 2) 표를 도형으로 다시 그림 (머리줄 + 줄무늬 행)
tbl = next(sh for sh in s.shapes if sh.has_table)
tbl._element.getparent().remove(tbl._element)
from pptx.enum.text import MSO_ANCHOR

def text(x, y, w, h, lines, size, color=INK, bold=False, sub=None):
    tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h)); tf = tb.text_frame
    tf.word_wrap = True; tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    for m in ('margin_left', 'margin_right', 'margin_top', 'margin_bottom'): setattr(tf, m, 0)
    items = [(l, size, color, bold) for l in (lines if isinstance(lines, list) else [lines])] + [(l, size - 1.5, SUB, False) for l in (sub or [])]
    for i, (line, sz, col, bd) in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        r = p.add_run(); r.text = line
        r.font.size = Pt(sz); r.font.bold = bd; r.font.color.rgb = col; r.font.name = FONT
        rpr = r._r.get_or_add_rPr(); ea = rpr.find(qn('a:ea'))
        if ea is None: ea = rpr.makeelement(qn('a:ea'), {}); rpr.append(ea)
        ea.set('typeface', FONT)
    return tb

def rect(x, y, w, h, fill, kind=MSO_SHAPE.RECTANGLE, line=None):
    r = s.shapes.add_shape(kind, Inches(x), Inches(y), Inches(w), Inches(h))
    if fill is None: r.fill.background()
    else: r.fill.solid(); r.fill.fore_color.rgb = fill
    if line is None: r.line.fill.background()
    else: r.line.color.rgb = line; r.line.width = Pt(0.75)
    r.shadow.inherit = False
    return r

# (항목, 값, 작은 설명, 높이 비율)
ROWS = [
    ('총 장수', '281장', None, 1),
    ('구성', 'center 1장, Y 80장, Θ 200장', None, 1),
    ('trial 수', '6구간', ['Y 0.1 간격 +쪽 20장, -쪽 20장', 'Y 0.05 홀수 +쪽 20장, -쪽 20장', 'Θ +쪽 100장, -쪽 100장'], 2.8),
    ('ADDRESS', '240', None, 1),
    ('QR', 'FP11L, FP11C, FP11R', ['왼쪽, 가운데, 오른쪽'], 1.45),
    ('해상도', '3840×2160', None, 1),
    ('형식', 'BMP 24bit, 무압축', ['R=G=B라서 실제로는 흑백'], 1.45),
    ('라벨 범위', 'Y ±2.00mm (0.05 간격)', ['Θ ±1.00 (0.01 간격)'], 1.45),
    ('라벨 구조', '십자 구조', ['두 축이 동시에 움직인 사진 없음'], 1.45),
    ('단위', 'Y는 mm, Θ는 도(°)로 추정', ['0.01 = 1 센치 디그리'], 1.45),
]
LIGHT = RGBColor.from_string('F3F4F6')
x0, y0, w = 7.2, grp.top / E, 13.333 - 0.8 - 7.2
H_all, hh, kw = grp.height / E, 0.42, 1.45
unit = (H_all - hh) / sum(r[3] for r in ROWS)
rect(x0, y0, w, H_all, BG, MSO_SHAPE.ROUNDED_RECTANGLE).adjustments[0] = 0.03
head = rect(x0, y0, w, hh, INK, MSO_SHAPE.ROUND_2_SAME_RECTANGLE); head.adjustments[0] = 0.03 * min(w, H_all) / hh  # 바깥 테두리와 같은 모서리 반경
text(x0 + 0.25, y0, kw, hh, '항목', 10.5, BG, True)
text(x0 + 0.25 + kw, y0, w - kw - 0.5, hh, '20261012 값', 10.5, BG, True)
y = y0 + hh
for i, (k, v, sub, f) in enumerate(ROWS):
    h = unit * f
    if i % 2: rect(x0, y, w, h, LIGHT)
    text(x0 + 0.25, y, kw - 0.1, h, k, 10.5, SUB, True)
    text(x0 + 0.25 + kw, y, w - kw - 0.45, h, v, 11.5, INK, False, sub)
    y += h
rect(x0, y0, w, H_all, None, MSO_SHAPE.ROUNDED_RECTANGLE, line=RGBColor.from_string('E5E7EB')).adjustments[0] = 0.03

prs.save(OUT)
print('saved', OUT.name)
