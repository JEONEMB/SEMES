# 사용자 시안(기초 정보 슬라이드 오른쪽: 항목 / 값 / 아래 가는 선) 스타일 그대로 10개 행을 채운다
# 시안의 자리표시 행(찍은 날 ...)은 지우고, 같은 글자 크기와 색, 같은 x 위치와 선으로 다시 그린다
import re, sys, pathlib
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE_TYPE
from pptx.oxml.ns import qn

HERE = pathlib.Path(__file__).parent
F = HERE / (sys.argv[1] if len(sys.argv) > 1 else 'SEMES_01_데이터EDA_기초정보추가_표도형.pptx')   # 입력
OUT = HERE / (sys.argv[2] if len(sys.argv) > 2 else F.name)                                         # 출력 (기본: 덮어쓰기)
SUB, LINE = RGBColor.from_string('4B5563'), RGBColor.from_string('E5E7EB')
INK, ACC = RGBColor.from_string('111827'), RGBColor.from_string('0071E3')
FONT, E = 'Noto Sans KR', 914400
KX, KW, VX, VW, LX, LW = 7.21, 1.3, 8.51, 4.03, 7.21, 5.33   # 시안의 위치
TOP, BOTTOM = 2.03, 6.43                                    # 왼쪽 QR 묶음과 같은 높이
LH = 0.2                                                     # 11pt 한 줄 높이

# 강조 표시: **굵게(진한 글씨)**, [[파랑 굵게]] (핵심 값 3개만 파랑)
ROWS = [
    ('총 장수', ['[[281장]]']),
    ('구성', ['**center 1장, Y 80장, Θ 200장**']),
    ('trial 수', ['[[6구간]]', 'Y 0.1 간격 +쪽 20장, -쪽 20장', 'Y 0.05 홀수 +쪽 20장, -쪽 20장', 'Θ +쪽 100장, -쪽 100장']),
    ('ADDRESS', ['**240**']),
    ('QR', ['**FP11L, FP11C, FP11R** (왼쪽, 가운데, 오른쪽)']),
    ('해상도', ['**3840×2160**']),
    ('형식', ['**BMP 24bit, 무압축.** R=G=B라서 실제로는 흑백']),
    ('라벨 범위', ['**Y ±2.00mm** (0.05 간격)', '**Θ ±1.00** (0.01 간격)']),
    ('라벨 구조', ['[[십자 구조]] (두 축이 동시에 움직인 사진 없음)']),
    ('단위', ['**Y는 mm, Θ는 도(°)로 추정**', '(0.01 = 1 센치 디그리)']),
]

prs = Presentation(F)
s = next(sl for sl in prs.slides if any(sh.has_text_frame and sh.text_frame.text == '기초 정보' for sh in sl.shapes))
# 오른쪽 영역(x >= 7.0in)의 시안 행, 이전에 그린 표, 카드 지우기 (그룹, 아래 설명, 쪽 번호는 남김)
for sh in list(s.shapes):
    if sh.left / E >= 7.0 and sh.top / E < 6.9 and sh.shape_type != MSO_SHAPE_TYPE.GROUP and not sh.is_placeholder:
        sh._element.getparent().remove(sh._element)

def parse(line):
    # '**굵게**', '[[파랑]]', 나머지 보통 글씨로 나눔
    out = []
    for m in re.finditer(r'\*\*(.+?)\*\*|\[\[(.+?)\]\]|([^*\[]+)', line):
        if m.group(1): out.append((m.group(1), 'bold'))
        elif m.group(2): out.append((m.group(2), 'blue'))
        else: out.append((m.group(3), 'plain'))
    return out

def text(x, y, w, h, lines, size):
    tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h)); tf = tb.text_frame
    tf.word_wrap = True
    for m in ('margin_left', 'margin_right', 'margin_top', 'margin_bottom'): setattr(tf, m, 0)
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        for seg, kind in parse(line):
            r = p.add_run(); r.text = seg
            r.font.size = Pt(size); r.font.name = FONT
            r.font.bold = kind != 'plain'; r.font.color.rgb = {'plain': SUB, 'bold': INK, 'blue': ACC}[kind]
            rpr = r._r.get_or_add_rPr(); ea = rpr.find(qn('a:ea'))
            if ea is None: ea = rpr.makeelement(qn('a:ea'), {}); rpr.append(ea)
            ea.set('typeface', FONT)

n_lines = sum(len(v) for _, v in ROWS)
pad = (BOTTOM - TOP - n_lines * LH) / len(ROWS)   # 행마다 위아래 여백 (선 포함)
y = TOP
for k, vals in ROWS:
    h = len(vals) * LH + pad
    text(KX, y + pad / 2 - 0.02, KW, 0.3, [k], 12)
    text(VX, y + pad / 2, VW, len(vals) * LH, vals, 11)
    y += h
    ln = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(LX), Inches(y), Inches(LX + LW), Inches(y))
    ln.line.color.rgb = LINE; ln.line.width = Pt(0.75)

prs.save(OUT)
print('saved', OUT.name, 'pad', round(pad, 3))
