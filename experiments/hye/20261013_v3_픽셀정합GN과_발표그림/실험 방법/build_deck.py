"""발표 자료: 팀원 공유 + SEMES 보고, 피드백 요청용. 모든 슬라이드에 그림, 도표, 차트 중 하나 이상.
출력: 발표_QR보정_문제이해와_다음단계.pptx (프로젝트 폴더)"""
import pandas as pd, numpy as np, cv2
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.chart.data import CategoryChartData
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION
HERE = Path(__file__).parent; PROJ = HERE.parent; DI = HERE/'deck_img'; SI = HERE/'slide_img'
DST = PROJ/'발표_QR보정_문제이해와_다음단계.pptx'
FONT = '맑은 고딕'
INK, SUBC, NAVY = RGBColor(0x1F, 0x1F, 0x1E), RGBColor(0x5F, 0x5E, 0x58), RGBColor(0x1F, 0x4E, 0x78)
BOX, WHITE = RGBColor(0xF2, 0xF2, 0xF2), RGBColor(255, 255, 255)
ORG, GRN, RED, BLUE = RGBColor(0xC5, 0x5A, 0x11), RGBColor(0x2E, 0x7D, 0x32), RGBColor(0xC0, 0x00, 0x00), RGBColor(0x2A, 0x78, 0xD6)
PEACH, MINT, SKY = RGBColor(0xFC, 0xE4, 0xD6), RGBColor(0xE2, 0xEF, 0xDA), RGBColor(0xDD, 0xEB, 0xF7)
prs = Presentation(); prs.slide_width = Inches(13.333); prs.slide_height = Inches(7.5)
BL = prs.slide_layouts[6]; W, HT = 13.333, 7.5

def text(sl, x, y, w, h, runs, size=14, bold=False, color=INK, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, fill=None, shape=MSO_SHAPE.ROUNDED_RECTANGLE, line=None, after=4):
    if fill is not None or line is not None:
        bx = sl.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
        if shape == MSO_SHAPE.ROUNDED_RECTANGLE: bx.adjustments[0] = 0.08
        if fill is not None: bx.fill.solid(); bx.fill.fore_color.rgb = fill
        else: bx.fill.background()
        if line is not None: bx.line.color.rgb = line; bx.line.width = Pt(1.5)
        else: bx.line.fill.background()
        bx.shadow.inherit = False; tf = bx.text_frame; m = 0.12
    else:
        bx = sl.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h)); tf = bx.text_frame; m = 0
    tf.word_wrap = True; tf.vertical_anchor = anchor
    for sd in ('left', 'right', 'top', 'bottom'): setattr(tf, f'margin_{sd}', Inches(m))
    lines = runs if isinstance(runs, list) else [runs]
    for i, ln in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph(); p.alignment = align; p.space_after = Pt(after)
        for t, o in ([(ln, {})] if isinstance(ln, str) else ln):
            r = p.add_run(); r.text = t; f = r.font; f.name = FONT; f.size = Pt(o.get('size', size)); f.bold = o.get('bold', bold); f.color.rgb = o.get('color', color)
    return bx

def H(t, b=True, c=NAVY, s=None):  # 강조 조각
    o = {'bold': b, 'color': c}
    if s: o['size'] = s
    return (t, o)

def head(sl, sec, t, sub=None):
    if sec: text(sl, 0.5, 0.22, 8, 0.3, sec, size=11, bold=True, color=BLUE)
    text(sl, 0.5, 0.48, W - 1.0, 0.65, t, size=26, bold=True)
    if sub: text(sl, 0.5, 1.12, W - 1.0, 0.4, sub, size=13, color=SUBC)

def pic(sl, path, x, y, w=None, h=None):
    kw = {}
    if w: kw['width'] = Inches(w)
    if h: kw['height'] = Inches(h)
    return sl.shapes.add_picture(str(path), Inches(x), Inches(y), **kw)

def table(sl, x, y, w, rows, widths=None, size=11, rh=0.32, hl=None, hdr=NAVY):
    t = sl.shapes.add_table(len(rows), len(rows[0]), Inches(x), Inches(y), Inches(w), Inches(rh * len(rows))).table
    if widths:
        for j, wv in enumerate(widths): t.columns[j].width = Inches(w * wv / sum(widths))
    for i, r in enumerate(rows):
        t.rows[i].height = Inches(rh)
        for j, v in enumerate(r):
            c = t.cell(i, j); c.text = str(v) or " "; p = c.text_frame.paragraphs[0]; p.alignment = PP_ALIGN.CENTER if j else PP_ALIGN.LEFT
            f = p.runs[0].font; f.name = FONT; f.size = Pt(size); f.bold = i == 0 or (hl and hl(i, r)); f.color.rgb = WHITE if i == 0 else INK
            c.fill.solid(); c.fill.fore_color.rgb = hdr if i == 0 else (PEACH if hl and hl(i, r) else WHITE)
            c.margin_top = c.margin_bottom = Inches(0.02); c.margin_left = c.margin_right = Inches(0.06); c.vertical_anchor = MSO_ANCHOR.MIDDLE
    return t

def arrow(sl, x1, y1, x2, y2, color=SUBC, w=2):
    c = sl.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    c.line.color.rgb = color; c.line.width = Pt(w)
    ln = c.line._get_or_add_ln(); from pptx.oxml.ns import qn
    tail = ln.makeelement(qn('a:tailEnd'), {'type': 'triangle', 'w': 'med', 'len': 'med'}); ln.append(tail)
    return c

def bar_chart(sl, x, y, w, h, cats, vals, title, unit, colors=None, maxv=None, horizontal=True, fmt='0'):
    cd = CategoryChartData(); cd.categories = cats; cd.add_series(title, vals)
    gf = sl.shapes.add_chart(XL_CHART_TYPE.BAR_CLUSTERED if horizontal else XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(x), Inches(y), Inches(w), Inches(h), cd)
    ch = gf.chart; ch.has_legend = False; ch.has_title = True
    ch.chart_title.text_frame.text = title; tp = ch.chart_title.text_frame.paragraphs[0]; tp.runs[0].font.size = Pt(13); tp.runs[0].font.bold = True; tp.runs[0].font.name = FONT
    pl = ch.plots[0]; pl.gap_width = 60; pl.has_data_labels = True
    dl = pl.data_labels; dl.number_format = fmt + f'"{unit}"'; dl.number_format_is_linked = False; dl.position = XL_LABEL_POSITION.OUTSIDE_END
    dl.font.size = Pt(11); dl.font.name = FONT
    va = ch.value_axis; va.has_major_gridlines = True; va.major_gridlines.format.line.color.rgb = RGBColor(0xE4, 0xE3, 0xDC)
    va.tick_labels.font.size = Pt(10); va.format.line.fill.background()
    if maxv: va.maximum_scale = maxv
    va.minimum_scale = 0
    ca = ch.category_axis; ca.tick_labels.font.size = Pt(11); ca.tick_labels.font.name = FONT; ca.format.line.color.rgb = RGBColor(0xBF, 0xBF, 0xBF)
    if horizontal: ca.reverse_order = True
    s = pl.series[0]
    for i, v in enumerate(vals):
        pt = s.points[i]; pt.format.fill.solid(); pt.format.fill.fore_color.rgb = (colors[i] if colors else BLUE)
    return ch

def notes(sl, t): sl.notes_slide.notes_text_frame.text = t

# ======================= 1. 표지 =======================
sl = prs.slides.add_slide(BL)
im = cv2.imdecode(np.fromfile(r"C:\Users\SSAFY\Downloads\[기업연계_세메스] 교보재\[기업연계_세메스] 교보재\extracted\SSAFY Test Image Set\center.bmp", np.uint8), cv2.IMREAD_GRAYSCALE)
cv2.imwrite(str(DI/'cover.png'), cv2.resize(im[180:2000, 1150:2650], (750, 910), interpolation=cv2.INTER_AREA))
bg = sl.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(W), Inches(HT)); bg.fill.solid(); bg.fill.fore_color.rgb = RGBColor(0x1B, 0x2A, 0x41); bg.line.fill.background()
pic(sl, DI/'cover.png', 8.3, 0.6, h=6.3)
text(sl, 0.7, 1.7, 7.3, 0.5, 'SEMES x SSAFY  비전 기반 Y, Θ 자동 보정', size=15, bold=True, color=RGBColor(0x9E, 0xC5, 0xF4))
text(sl, 0.7, 2.25, 7.3, 1.9, ['QR 사진으로 Y와 Θ를', '나눠 읽을 수 있을까'], size=36, bold=True, color=WHITE)
text(sl, 0.7, 4.25, 7.3, 1.2, ['우리가 문제를 이해한 방식, 지금까지 해본 것,', '앞으로 할 것과 그에 필요한 것'], size=16, color=RGBColor(0xD9, 0xD9, 0xD9))
text(sl, 0.7, 6.3, 7.3, 0.4, '2026-10-13  |  데이터: SSAFY Test Image Set 281장', size=12, color=RGBColor(0xBF, 0xBF, 0xBF))
notes(sl, '발표 목적: 팀원에게 지금까지의 분석을 공유하고, SEMES에 우리 이해가 맞는지 확인받고, 다음 단계에 필요한 데이터와 정보를 요청한다. 오른쪽 사진은 받은 데이터의 center 사진.')

# ======================= 2. 한 장 요약 =======================
sl = prs.slides.add_slide(BL)
head(sl, '한 장 요약', '세 가지를 알게 됐고, 세 가지가 필요합니다')
cards = [('QR 위치만으로는', '45장', 'Y 사진 80장 중 Θ 사진과 구분이 안 되는 사진 (Y ±0.8mm 이내 전부)', ORG),
         ('QR 판 전체를 픽셀로 풀면', '12장', '구분 안 되는 사진이 줄어듦. Y 오차 약 0.09mm (Y ±0.3mm 이내는 여전히 어려움)', BLUE),
         ('파일 이름(명령값) 자체가', '±40µm', '이동 방향에 따라 실제 위치가 다름 (백래시 추정). 정답값에도 오차가 있음', RED)]
for i, (a, b, c, col) in enumerate(cards):
    x = 0.5 + i * 4.15
    text(sl, x, 1.65, 3.95, 2.35, [[H(a, c=SUBC, s=13)], [H(b, c=col, s=40)], c], size=12.5, fill=BOX)
text(sl, 0.5, 4.2, 4.0, 0.4, '그래서 필요한 것', size=15, bold=True, color=NAVY)
asks = [('1', 'QR 밖 영역(브래킷 등) 사용 가능 여부', '쓸 수 있으면 Y 오차 약 0.02mm, 거의 전부 구분'),
        ('2', '보정 후 다시 찍은 사진 60회', '"보정하면 0.09mm 안에 든다"를 직접 검증'),
        ('3', 'center를 방향별로 반복 촬영 (각 10회 이상)', '목표 위치(center)가 얼마나 흔들리는지 확인')]
for i, (n, a, b) in enumerate(asks):
    y = 4.7 + i * 0.85
    c = sl.shapes.add_shape(MSO_SHAPE.OVAL, Inches(0.5), Inches(y + 0.08), Inches(0.5), Inches(0.5)); c.fill.solid(); c.fill.fore_color.rgb = NAVY; c.line.fill.background()
    c.text_frame.text = n; p = c.text_frame.paragraphs[0]; p.alignment = PP_ALIGN.CENTER; p.runs[0].font.size = Pt(16); p.runs[0].font.bold = True; p.runs[0].font.color.rgb = WHITE
    text(sl, 1.15, y, 6.4, 0.7, [[H(a, c=INK, s=14)], [(b, {'size': 12, 'color': SUBC})]], after=1)
pic(sl, PROJ/'27_구분안되는짝_연결도.png', 7.75, 4.2, h=2.85)
text(sl, 7.75, 7.08, 5.1, 0.3, '구분 안 되는 Y, Θ 사진 짝 (위: QR 위치만, 아래: 판 원근 변환)', size=9.5, color=SUBC)
notes(sl, '핵심 세 가지와 요청 세 가지. 오른쪽 아래 그림은 뒤에서 자세히 설명하는 "구분 안 되는 짝" 연결도.')

# ======================= 3. 목차 =======================
sl = prs.slides.add_slide(BL)
head(sl, None, '순서')
secs = [('1', '문제를 이해한 방식', '인풋 → 계산 → 아웃풋 → 검증'), ('2', '측정 방법', 'b - a의 a, b는 어디서 쟀나'),
        ('3', '주어진 값은 정확한가', 'center, Y 값, Θ 값'), ('4', '정확하게 하려면', '촬영 주의점, 측정 개선, 잡음'),
        ('5', '계산 방법별 결과', '해본 것, 믿을 만한가, 해볼 것'), ('6', '필요한 것과 질문', '데이터 요청 근거, QnA, 피드백')]
for i, (n, a, b) in enumerate(secs):
    x = 0.5 + (i % 3) * 4.15; y = 1.7 + (i // 3) * 2.6
    text(sl, x, y, 3.95, 2.3, [[H(n, c=BLUE, s=36)], [H(a, c=INK, s=18)], [(b, {'size': 13, 'color': SUBC})]], fill=BOX)
notes(sl, '6개 묶음 순서로 진행.')

# ======================= 4. 문제 =======================
sl = prs.slides.add_slide(BL)
head(sl, '1. 문제를 이해한 방식', 'Y 이동과 Θ 회전이 QR에서는 거의 똑같이 보인다', '같은 QR 위치인데 실제로는 전혀 다른 두 상황: 로봇이 어느 쪽으로 보정해야 할지 알 수 없다')
pic(sl, DI/'fig_twin.png', 0.5, 1.7, w=W - 1.0)
y0 = 1.7 + (W - 1.0) * 3.3 / 15 + 0.15
text(sl, 0.5, y0, 6.0, 7.1 - y0, [[H('왜 문제인가')],
     'Y 1mm = QR 14.3px, Θ 1 = QR 80px. 그래서 Y +0.50mm와 Θ +0.09가 거의 같은 위치(약 6.5px)로 보인다.',
     '잘못 나눠서 보정하면 QR은 가운데로 돌아오지만 로봇은 옆으로 0.5mm 밀린 채 비뚤게 서 있게 된다.'], size=12.5, fill=BOX)
text(sl, 6.85, y0, 5.98, 7.1 - y0, [[H('지금 현장 방식 (명세서)')],
     '사람이 Y를 먼저 수동으로 맞춘 뒤, 남은 틀어짐을 Θ로 자동 보정 (반자동).',
     [H('과제: ', c=INK), ('사진만 보고 Y 몫과 Θ 몫을 나눠 한 번에 자동 보정', {})]], size=12.5, fill=SKY)
notes(sl, '왼쪽 두 사진은 Y만 0.5mm 옮긴 것과 Θ만 0.09 돌린 것. QR 평균 위치가 6.61px vs 6.40px로 거의 같고, 차이 영상도 거의 회색. 아래 문구: 뒤쪽 볼트처럼 거리가 다른 물체는 다르게 움직인다(시차) = 구분 단서.')

# ======================= 5. 인풋 → 계산 → 아웃풋 → 검증 =======================
sl = prs.slides.add_slide(BL)
head(sl, '1. 문제를 이해한 방식', '우리가 찾는 것: 인풋 → 계산 방법 → 아웃풋 → 검증 방법')
steps = [('인풋', ['로봇이 도착한 뒤 찍은 사진 1장', '+ 그 위치의 center 사진', '(가능하면 여러 장)'], '받은 데이터로 확인', MINT),
         ('계산 방법', ['QR 위치 측정 (서브픽셀)', '→ Y 몫과 Θ 몫 분리', '해본 것 7가지, 해볼 것 3가지'], '일부 해결', PEACH),
         ('아웃풋', ['보정값: Y (mm), Θ', '+ 믿을 수 있는지 표시', '(구분 가능 / 불확실)'], '형식 제안 (확인 필요)', SKY),
         ('검증 방법', ['보정값대로 이동 후 재촬영', '→ center 사진과 비교', '허용 오차 0.09mm 안인가'], '데이터 없음 (요청 필요)', PEACH)]
for i, (a, lines, st, col) in enumerate(steps):
    x = 0.5 + i * 3.17
    text(sl, x, 1.75, 2.75, 0.6, a, size=18, bold=True, color=WHITE, fill=NAVY, anchor=MSO_ANCHOR.MIDDLE, align=PP_ALIGN.CENTER)
    text(sl, x, 2.45, 2.75, 2.2, [ (l, {}) and l for l in lines], size=13, fill=BOX)
    text(sl, x, 4.8, 2.75, 0.75, [[H('지금 상태', c=SUBC, s=11)], [H(st, c=INK, s=12.5)]], fill=col, after=1)
    if i < 3: arrow(sl, x + 2.78, 2.05, x + 3.14, 2.05, color=NAVY, w=2.5)
text(sl, 0.5, 5.8, W - 1.0, 1.25, [[H('핵심: ', c=INK), ('정답은 "몇 mm"라는 숫자가 아니라 "center 사진과 똑같이 보이는 상태"다. 그래서 검증도 보정 후 사진을 center와 비교해서 한다. '
     '받은 데이터의 파일 이름(명령값)은 계수 학습에 쓰지만, 그 자체에도 오차가 있다 (3장).', {})]], size=13, fill=SKY)
notes(sl, '네 단계 중 인풋과 계산은 받은 데이터로 다뤘고, 아웃풋 형식은 우리 제안, 검증은 보정 후 재촬영 데이터가 없어서 아직 못 함.')

# ======================= 6. 인풋 통계 =======================
sl = prs.slides.add_slide(BL)
head(sl, '1. 문제를 이해한 방식', '인풋: 받은 데이터 281장', '촬영 1회(약 58분), 장비 1대, 위치 1곳(ADDRESS 240). 같은 자리 반복 사진, Y와 Θ가 같이 틀어진 사진은 없음')
table(sl, 0.5, 1.75, 5.6, [('항목', '값'), ('center', '1장 (Θ 0, Y 0)'), ('Y 사진', '80장, -2.00 ~ +2.00mm, 0.05 간격, Θ = 0'),
      ('Θ 사진', '200장, -1.00 ~ +1.00, 0.01 간격, Y = 0'), ('Θ 단위', '미표기 (화면 이동량으로 보면 도°로 추정)'),
      ('해상도', '3840 x 2160, 흑백 (24bit로 저장, R=G=B)'), ('사진 잡음', '평평한 벽 기준 밝기 약 0.7단계 (매우 깨끗)'),
      ('촬영', '58분, 한 장 평균 4.5초, 중간 15분 공백'), ('0 사진', 'center 1장뿐 (Y, Θ 폴더엔 없음)')], widths=[1, 3.2], size=11, rh=0.4)
_p = pic(sl, PROJ/'16_데이터분석_촬영순서_밝기_드리프트.png', 6.35, 1.75, h=4.55); _p.left = Inches(6.35 + (6.48 - _p.width / 914400) / 2)
text(sl, 6.35, 6.4, 6.48, 0.6, '촬영 순서와 밝기 변화 (파일 수정 시각 기준). 처음 3분 동안 밝기가 내려가다 안정 → center만 밝기 +1.2% (조명 예열 추정)', size=10.5, color=SUBC)
notes(sl, '십자 모양 데이터: Θ와 Y가 둘 다 0이 아닌 사진이 없다. 촬영 순서는 파일 수정 시각에서 추정: center → Y+ → Y- → Θ+ → Θ- → (15분 공백) → Y 0.05 간격 추가.')

# ======================= 7. 아웃풋 형식 =======================
sl = prs.slides.add_slide(BL)
head(sl, '1. 문제를 이해한 방식', '아웃풋: 보정값 + "믿을 수 있는지" 표시를 제안', '틀린 보정값을 내는 것보다 "확실하지 않다"고 말하는 게 장비에는 더 안전하다')
text(sl, 0.5, 1.8, 5.2, 3.3, [[H('출력 예시', c=SUBC, s=12)], [H('Y  -0.32 mm', c=INK, s=30)], [H('Θ  +0.118', c=INK, s=30)],
     [H('판정: 구분 가능 (확실)', c=GRN, s=16)], [('예상 오차: Y 약 0.09mm (현재 최고 방법 기준)', {'size': 12, 'color': SUBC})]], fill=BOX)
text(sl, 0.5, 5.3, 5.2, 1.75, [[H('불확실할 때', c=SUBC, s=12)], [H('판정: 구분 불확실', c=RED, s=16)],
     [('→ 한 장 더 찍기, 로봇을 조금 움직여 다시 찍기, 사람 확인 중 하나로 넘김', {'size': 12})]], fill=PEACH)
text(sl, 6.1, 1.8, 6.73, 0.4, '어느 범위에서 "확실"할 수 있나 (QR만 쓰는 최고 방법, 사진 1장 기준)', size=14, bold=True, color=NAVY)
def band(y, lab, segs):
    text(sl, 6.1, y, 1.5, 0.5, lab, size=12, bold=True, anchor=MSO_ANCHOR.MIDDLE)
    x = 7.6
    for w_, t_, col in segs:
        text(sl, x, y, w_, 0.5, t_, size=11, bold=True, color=WHITE, fill=col, shape=MSO_SHAPE.RECTANGLE, anchor=MSO_ANCHOR.MIDDLE, align=PP_ALIGN.CENTER); x += w_
band(2.4, 'Y 틀어짐', [(1.75, '-2 ~ -0.35mm 확실', GRN), (1.7, '±0.30mm 불확실', RED), (1.75, '+0.35 ~ +2mm 확실', GRN)])
band(3.1, 'Θ 틀어짐', [(1.75, '-1 ~ -0.06 확실', GRN), (1.7, '±0.03 불확실', RED), (1.75, '+0.06 ~ +1 확실', GRN)])
text(sl, 6.1, 3.85, 6.73, 3.2, [[H('정해야 할 것 (SEMES 확인 필요)')], '허용 오차: Y 0.09mm로 들었음. ±인지, Θ는 얼마인지?',
     'Θ 단위가 도°가 맞는지?', '"불확실"일 때 여러 장 촬영이나 반복 보정이 허용되는지?', '사이 구간(0.30~0.35mm)은 구분되는 사진과 안 되는 사진이 섞여 있음'], size=12.5, fill=SKY)
notes(sl, '출력 예시 값은 설명용 가상 값. 범위 막대는 G1(픽셀 직접 가우스-뉴턴) 방법으로 Y ±0.30mm 이내 전부 구분 불가, ±0.35mm 이상 전부 구분 가능이었던 결과.')

# ======================= 8. 측정 방법 =======================
sl = prs.slides.add_slide(BL)
head(sl, '2. 측정 방법', 'b - a: a는 center 사진, b는 각 사진에서 같은 방법으로 잰다')
pic(sl, DI/'fig_measure.png', 0.5, 1.3, w=W - 1.0)
yb = 1.3 + (W - 1.0) * 4.9 / 15 + 0.1
table(sl, 0.5, yb, W - 1.0, [('측정 방법', '무엇을 기준으로', '어떻게', '쓰는 곳'),
      ('QR 네모 중심', 'center 사진의 QR 3개 위치', '템플릿 정합 후 서브픽셀 위치, 좌중우 평균', '짝 찾기, 기본 이동량'),
      ('QR 내부 점 263개', 'center 사진 QR 안의 모서리 점', '점마다 추적', '어제 방법 (Q7)'),
      ('QR 판 원근 변환', 'center 사진의 QR 판 영역 전체', '판 전체를 겹쳐 맞춤 (호모그래피)', '판 기울기 단서'),
      ('픽셀 직접', 'center 근처 사진들의 평균 영상', 'Θ, Y를 픽셀 밝기로 바로 풂', '현재 최고 (G1)')], widths=[1.3, 2.2, 2.6, 1.6], size=11, rh=0.33)
notes(sl, 'a는 항상 center 사진에서, b는 각 사진에서 같은 방법으로 잰다. 확대 그림은 중 QR 중심 주변 실제 픽셀. Y +0.50mm 사진은 중 QR이 6.62px 오른쪽으로 갔다.')

# ======================= 9. 측정을 믿을 수 있나 =======================
sl = prs.slides.add_slide(BL)
head(sl, '2. 측정 방법', '측정 자체는 믿을 만하다: 잡음 약 0.02px, 거의 완벽한 직선')
pic(sl, PROJ/'19_선형성_명령값vs픽셀.png', 0.5, 1.3, h=5.75)
xw = 0.5 + 5.75 * 1.34 + 0.25
text(sl, xw, 1.3, W - 0.5 - xw, 5.75, [[H('측정 신뢰도')], [H('사진 한 장 위치 잡음: ', c=INK), ('약 0.017px (Y 약 1.2µm)', {})],
     [H('명령값과 직선 관계: ', c=INK), ('R² 0.99998 (Y), 0.999985 (Θ)', {})], [H('서로 다른 측정 방법끼리: ', c=INK), ('픽셀 단위로 잰 값과 QR 네모로 잰 값이 0.02~0.06px 안에서 일치', {})],
     [H('주의', c=RED)], '오른쪽 확대 그래프의 띠는 곡선이 아니라 구간마다 높이만 다른 모양 = 측정 문제가 아니라 로봇 0점이 구간마다 다름 (3장)',
     'OpenCV 영상 변환은 위치를 1/32px 단위로 잘라 우리가 찾는 신호(0.02~0.06px)와 크기가 같음. 정밀 계산에는 다른 보간 필요 (C++ 구현 때 주의)'], size=12, fill=BOX)
notes(sl, '왼쪽 그래프: 큰 그림은 직선, 확대하면 구간별 띠. 측정 잡음보다 구간 사이 차이가 훨씬 크다는 게 다음 장의 주제.')

# ======================= 10. Y 값 정확한가 =======================
sl = prs.slides.add_slide(BL)
head(sl, '3. 주어진 값은 정확한가', '주어진 Y 값: 같은 "Y=0"인데 구간마다 최대 44µm 다른 자리에 섰다', '파일 이름은 로봇에 내린 명령값. 실제 위치는 이동 방향에 따라 다르다')
pic(sl, PROJ/'18_0점어긋남_시각화.png', 0.5, 1.65, h=4.75)
xr = 0.5 + 4.75 * 2175 / 1080 + 0.2
text(sl, xr, 1.65, W - 0.5 - xr, 4.75, [[H('읽는 법')], '구간 안에서는 평평한 띠 = 매 칸 정확히 같은 양씩 움직임', '띠 높이가 다름 = 구간마다 0점이 통째로 밀림',
     [H('+ 방향 3구간: ', c=INK), ('-30 ~ -37µm', {})], [H('- 방향 1구간: ', c=INK), ('+7µm', {})], '위치가 아니라 움직인 방향이 0점을 정함'], size=12, fill=BOX)
text(sl, 0.5, 6.5, W - 1.0, 0.55, [[H('결론: ', c=INK), ('주어진 Y 값(명령값)에는 방향에 따라 약 ±20~40µm 오차가 섞여 있다. 허용 오차 90µm의 약 절반 크기라, 그보다 작은 정확도는 이 데이터로 검증할 수 없다.', {})]], size=12.5, fill=PEACH)
notes(sl, '같은 세션 안의 Y 이동 구간 4개. 각 구간에 직선을 맞춰 Y=0까지 이어 본 값이 0점. center는 - 방향 구간과 같은 쪽.')

# ======================= 11. 한 칸 이동 =======================
sl = prs.slides.add_slide(BL)
head(sl, '3. 주어진 값은 정확한가', '그래도 구간 안에서는 매 칸 같은 양씩 움직인다', 'Y 0.1mm 명령 = 1.437px. 4구간 모두 공식과 0.15% 안쪽, 구간 끝까지 쌓인 오차 1~6µm')
pic(sl, PROJ/'30_한칸이동량_공식비교.png', 0.5, 1.65, h=4.1)
xr = 0.5 + 4.1 * 2080 / 975 + 0.2
text(sl, xr, 1.65, W - 0.5 - xr, 4.1, [[H('읽는 법')], '칸마다 흔들림(±0.024px)은 사진 두 장의 측정 잡음과 같은 크기', '첫 칸(주황)만 덜 감 = 방향을 바꾼 직후 기구 틈을 메움',
     'Θ는 + 방향 80.28, - 방향 80.76px로 0.6% 다름'], size=11.5, fill=BOX)
table(sl, 0.5, 5.9, W - 1.0, [('', 'Y 1차 +', 'Y 1차 -', 'Y 2차 +', 'Y 2차 -', 'Θ +', 'Θ -'),
      ('기울기 (공식 대비)', '-0.10%', '+0.13%', '+0.09%', '-0.06%', '-0.27%', '+0.32%'),
      ('끝까지 쌓인 오차', '-4.1µm', '+2.9µm', '-0.9µm', '-6.4µm', 'Θ -0.0047', 'Θ +0.0027')], size=11, rh=0.36)
notes(sl, '공식의 기준값(1.437px)은 이 데이터에서 구한 값이라, "정확히 0.1mm"인지는 카메라 캘리브레이션이 있어야 확인 가능. 확인된 건 "매 칸 일정하다"까지.')

# ======================= 12. Θ 값 =======================
sl = prs.slides.add_slide(BL)
head(sl, '3. 주어진 값은 정확한가', '주어진 Θ 값: 방향마다 0점과 비율이 다르다', 'Θ + 구간 0점은 center보다 약 -0.009~-0.010, Θ - 구간은 -0.0025. 출발 직후 몇 칸은 덜 감')
_p = pic(sl, DI/'fig_theta0.png', 0.5, 1.6, h=3.85); _p.left = Inches((W - _p.width / 914400) / 2)
yb = 1.6 + 3.85 + 0.12
text(sl, 0.5, yb, 6.0, 7.1 - yb, [[H('보이는 것')], '+ 구간과 - 구간이 0을 사이에 두고 크게 떨어져 있음', '+ 구간 첫 몇 장(0.01~0.05)이 덜 갔다가 안정됨 = 출발 시 기구 틈',
     '두 구간 기울기가 다름 (80.28 vs 80.76px)'], size=12, fill=BOX)
text(sl, 6.85, yb, 5.98, 7.1 - yb, [[H('Y와 다른 점')], 'Θ - 구간도 center와 0.2px 차이. Y처럼 "- 방향 = center"로 깔끔하지 않음',
     'Θ를 찍기 전 Y축도 움직였기 때문에 Y축 상태가 섞였을 수 있음. 축별, 방향별 반복 촬영으로만 분리 가능'], size=12, fill=PEACH)
notes(sl, '공통 기울기 80.52px/Θ를 뺀 나머지를 그렸기 때문에 두 띠가 기울어져 보임 (방향별 기울기 차이 0.6%).')

# ======================= 13. center =======================
sl = prs.slides.add_slide(BL)
head(sl, '3. 주어진 값은 정확한가', '주어진 center: 사진은 정상, "어느 방향으로 와서 섰는지"가 다르다', 'center를 명령만큼 옮긴 예상 사진과 실제 옆 사진 비교: + 방향으로 온 사진은 모두 예상보다 왼쪽(파랑)')
pic(sl, PROJ/'29_0점어긋남_실제사진.png', 0.5, 1.65, h=4.55)
xr = 0.5 + 4.55 * 1.98 + 0.25
text(sl, xr, 1.65, W - 0.5 - xr, 4.55, [[H('center에 대해 알게 된 것')], [H('판, 카메라: ', c=INK), ('정상. 세로 위치와 판 기울기는 다른 사진과 같음', {})],
     [H('가로 위치: ', c=INK), ('+ 방향 구간들의 0점보다 0.4~0.7px 오른쪽. - 방향 구간과 같은 자리', {})], [H('밝기: ', c=INK), ('다른 사진보다 +1.2% (조명 예열 중 촬영 추정)', {})],
     [H('해석', c=RED)], 'center가 "틀린" 게 아니라, - 방향으로 와서 멈춘 자리에서 찍힌 것. center는 목표의 정의라 맞고 틀림이 없지만, 매번 같은 자리에 만들어지는지는 확인 필요'], size=11.5, fill=BOX)
text(sl, 0.5, 6.35, W - 1.0, 0.7, [[H('예상 사진: ', c=INK), ('우리가 계산한 것. center를 (명령값 x 14.37px/mm)만큼 옮긴 사진. 어긋난 건 사진이 아니라 명령값과 실제 로봇 위치 사이', {})]], size=11.5, fill=SKY)
notes(sl, '6개 사진은 center 바로 옆 사진(Y ±0.05, ±0.10, Θ ±0.01). Y -0.10만 - 방향으로 와서 빨강(+6µm), 나머지는 + 방향으로 와서 파랑(-28~-35µm).')

# ======================= 14. 원인: 백래시 =======================
sl = prs.slides.add_slide(BL)
head(sl, '3. 주어진 값은 정확한가', '원인 추정: 기구의 틈(백래시) 약 40µm', '엔코더는 모터 축만 재므로, 방향을 바꿀 때 틈만큼 헛돈 것을 모른다')
# 기구 그림
text(sl, 0.7, 2.0, 1.8, 1.1, '모터\n(엔코더)', size=15, bold=True, color=WHITE, fill=NAVY, anchor=MSO_ANCHOR.MIDDLE, align=PP_ALIGN.CENTER)
text(sl, 2.8, 2.25, 1.9, 0.6, '기어, 나사, 벨트', size=13, bold=True, fill=BOX, anchor=MSO_ANCHOR.MIDDLE, align=PP_ALIGN.CENTER)
gap = sl.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(4.75), Inches(2.25), Inches(0.45), Inches(0.6)); gap.fill.solid(); gap.fill.fore_color.rgb = PEACH; gap.line.color.rgb = RED
text(sl, 4.45, 2.95, 1.1, 0.4, '틈', size=13, bold=True, color=RED, align=PP_ALIGN.CENTER)
text(sl, 5.25, 2.0, 2.2, 1.1, '실제 움직이는 부분\n(카메라)', size=14, bold=True, color=WHITE, fill=BLUE, anchor=MSO_ANCHOR.MIDDLE, align=PP_ALIGN.CENTER)
arrow(sl, 2.5, 2.55, 2.8, 2.55, NAVY); arrow(sl, 4.7, 2.55, 4.78, 2.55, NAVY)
text(sl, 0.7, 3.55, 6.75, 2.0, ['한 방향으로 계속 가면: 틈이 한쪽에 붙어 모터만큼 정확히 따라감', '방향을 바꾸면: 모터가 먼저 틈만큼 헛돌고 나서야 움직임',
     '로봇은 "갔다"고 믿지만 실제로는 틈만큼 덜 간 상태'], size=13, fill=BOX)
table(sl, 7.8, 1.75, 5.03, [('근거', '데이터'), ('방향으로 갈림', '+ 방향 -30~-37µm, - 방향 +7µm'), ('위치와 무관', 'Y -0.05(-쪽)도 + 방향이면 + 무리'),
      ('일정한 크기', '두 무리 차이 약 38~44µm'), ('출발 직후만 덜 감', 'Θ + 첫 사진 0.20→0.12→0.08px'), ('Θ도 같은 모양', 'Θ + -0.009, Θ - -0.0025')], widths=[1.3, 2.4], size=11, rh=0.42)
text(sl, 0.7, 5.75, W - 1.4, 1.3, [[H('다른 원인일 가능성: ', c=INK), ('카메라 문제라면 세로, 원근도 틀어져야 하는데 그대로 / 측정 잡음이라면 흩어져야 하는데 20장이 통째로 밀림. ', {}),
     H('남는 작은 차이: ', c=INK), ('같은 + 방향끼리 약 7µm (촬영 시작 vs 50분 뒤). 온도나 느린 드리프트 가능성', {})]], size=12, fill=SKY)
notes(sl, '핸들 유격 비유. 확정하려면 같은 자리를 방향별로 반복 촬영해야 한다(6장).')

# ======================= 15. 잡음 크기 비교 =======================
sl = prs.slides.add_slide(BL)
head(sl, '4. 정확하게 하려면', 'Y, Θ에 잡음이 있나: 흔들림은 아주 작고, 문제는 방향에 따른 어긋남', '모든 값을 Y(µm)로 환산해 허용 오차와 비교')
bar_chart(sl, 0.5, 1.6, 7.6, 5.4, ['사진 한 장 측정 잡음', '한 칸 이동 흔들림', '구간 끝까지 쌓인 오차 (최대)', '같은 방향끼리 0점 차이', '방향별 0점 차이 (백래시)', '허용 오차 (들은 값)'],
          [1.2, 1.7, 6.4, 7, 44, 90], '크기 비교 (µm, Y 환산)', 'µm', colors=[GRN, GRN, GRN, ORG, RED, SUBC], maxv=100)
text(sl, 8.4, 1.6, 4.43, 5.4, [[H('읽는 법')], [H('초록: 무작위 흔들림 ', c=GRN), ('= 사진, 로봇 모두 1~2µm 수준. 여러 장 평균하면 더 줄어듦', {})],
     [H('주황: 느린 변화 ', c=ORG), ('= 시간이 지나며 7µm 정도 (온도, 드리프트 추정)', {})], [H('빨강: 방향 ', c=RED), ('= 약 44µm. 허용 오차의 절반. 알고리즘으로는 못 줄이고 움직이는 방식으로 줄여야 함', {})],
     [H('그래서', c=NAVY)], '촬영 잡음은 걱정 안 해도 됨. 보정할 때 "같은 방향으로 접근"하는 게 가장 효과가 큼'], size=12.5, fill=BOX)
notes(sl, '사진 측정 잡음 0.017px / 14.37 = 1.2µm. 한 칸 흔들림 0.024px = 1.7µm. 방향별 0점 차이는 Y 구간 0점 -37 vs +7µm.')

# ======================= 16. center 고정 =======================
sl = prs.slides.add_slide(BL)
head(sl, '4. 정확하게 하려면', 'center를 미리 맞출 수 있나: 만드는 절차를 고정하면 된다 (제안)', 'center는 "가야 할 위치의 정의". 맞고 틀림은 없지만 매번 같은 자리, 같은 조건에서 만들어져야 한다')
proc = [('1', '조명 켜고\n기다리기', '처음 3분 밝기 변화'), ('2', '한쪽에서\n접근', '예: -0.5mm 갔다가\n+ 방향으로 0'), ('3', '멈춘 뒤\n잠시 대기', '잔류 진동 제거'),
        ('4', '여러 장 촬영\n평균', '예: 10장'), ('5', '엔코더 값\n함께 저장', '나중에 비교 가능')]
for i, (n, a, b) in enumerate(proc):
    x = 0.5 + i * 2.5
    sh = sl.shapes.add_shape(MSO_SHAPE.CHEVRON if i else MSO_SHAPE.PENTAGON, Inches(x), Inches(1.8), Inches(2.4), Inches(1.15))
    sh.fill.solid(); sh.fill.fore_color.rgb = NAVY; sh.line.fill.background()
    tf = sh.text_frame; tf.text = f'{n}. {a}';
    for p in tf.paragraphs:
        p.alignment = PP_ALIGN.CENTER
        for r in p.runs: r.font.size = Pt(13); r.font.bold = True; r.font.color.rgb = WHITE; r.font.name = FONT
    text(sl, x + 0.2, 3.05, 2.0, 0.8, b, size=11, color=SUBC, align=PP_ALIGN.CENTER)
text(sl, 0.5, 4.05, 6.0, 3.0, [[H('왜 이렇게?', c=NAVY)], '받은 center는 - 방향으로 접근, 조명 예열 중에 찍힌 것으로 보임',
     '접근 방향을 정하면 백래시가 항상 같은 쪽에 붙어 center 위치가 재현됨', '여러 장 평균은 사진 잡음을 1/√N로 줄임'], size=12.5, fill=BOX)
text(sl, 6.85, 4.05, 5.98, 3.0, [[H('확인이 필요한 것 (SEMES)', c=RED)], '지금 center는 언제, 어떤 절차로 만드나요?', '한 번 만든 center를 계속 쓰나요? 위치마다 있나요?',
     '찍을 때 어느 방향으로 접근하나요?', [H('데이터: ', c=INK), ('center 반복 촬영 + 방향 10회, - 방향 10회', {})]], size=12.5, fill=PEACH)
notes(sl, '절차 다섯 단계는 우리 제안. 실제 현장 절차를 듣고 맞춰야 함.')

# ======================= 17. Y, Θ 정확하게 + 촬영 주의 =======================
sl = prs.slides.add_slide(BL)
head(sl, '4. 정확하게 하려면', 'Y, Θ 값을 정확하게: 같은 방향으로 접근하고, 한 번 더 확인한다')
# 접근 방향 그림
ln = sl.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(0.8), Inches(2.6), Inches(6.2), Inches(2.6)); ln.line.color.rgb = SUBC; ln.line.width = Pt(1.5)
for xv, lab in [(1.4, '-0.5'), (3.5, '목표'), (5.6, '+0.5')]:
    tk = sl.shapes.add_shape(MSO_SHAPE.OVAL, Inches(xv - 0.08), Inches(2.52), Inches(0.16), Inches(0.16)); tk.fill.solid(); tk.fill.fore_color.rgb = INK if lab == '목표' else SUBC; tk.line.fill.background()
    text(sl, xv - 0.6, 2.75, 1.2, 0.35, lab, size=11, color=SUBC, align=PP_ALIGN.CENTER)
arrow(sl, 5.2, 1.95, 1.5, 1.95, RED, 2.5); text(sl, 1.5, 1.55, 3.7, 0.35, '① 목표를 조금 지나쳐 - 쪽으로', size=11, color=RED, align=PP_ALIGN.CENTER)
arrow(sl, 1.5, 2.3, 3.45, 2.3, GRN, 3); text(sl, 0.8, 3.1, 5.4, 0.35, '② 항상 + 방향으로 목표에 도착 → 틈이 늘 같은 쪽에 붙음', size=12, bold=True, color=GRN, align=PP_ALIGN.CENTER)
text(sl, 0.5, 3.65, 6.0, 3.4, [[H('측정 방법 개선')], '단방향 접근: 방향별 0점 차이 44µm → 같은 방향끼리 약 7µm', '보정 후 재촬영: 이동할 때 생긴 오차까지 한 번 더 잡음',
     'Θ는 방향별 계수 사용 (80.28 / 80.76px)', 'QR 판 전체를 픽셀로 비교 + 정확한 소수 위치 보간', '작은 틀어짐은 여러 장(로봇을 조금씩 움직이며) 촬영'], size=12.5, fill=BOX)
table(sl, 6.85, 1.55, 5.98, [('촬영할 때 주의점', '이유 (이번 데이터에서 본 것)'), ('조명 켜고 몇 분 뒤 촬영', '처음 3분 밝기 변화, center만 +1.2%'),
      ('구간마다 0 위치에서도 1장', 'Y, Θ 폴더에 0 사진이 없어 0점을 추정만 함'), ('이동 방향과 순서 기록', '백래시 분석에 필수'),
      ('엔코더 값 기록', '명령값과 실제 모터 위치 분리'), ('촬영 시각 기록 (ms)', '지금은 파일 수정 시각으로 추정'),
      ('무작위 순서로 촬영', '시간 변화와 위치 변화가 섞이지 않게'), ('8bit 흑백 BMP/PNG, JPG 금지', '24bit는 용량 3배, JPG는 정밀도 손실')],
      widths=[1.7, 2.3], size=10.5, rh=0.6)
notes(sl, '왼쪽 그림: 단방향 접근. 오른쪽 표: 다음 촬영 때 지켜주면 분석이 쉬워지는 것들.')

# ======================= 18. 계산 방법 지도 =======================
sl = prs.slides.add_slide(BL)
head(sl, '5. 계산 방법별 결과', '해본 계산 방법: 특징점 조합 15개 x 알고리즘 7개', '어제 결과표 49행을 같은 데이터, 같은 분할로 소수점까지 재현한 뒤 새 방법을 추가해 비교')
pic(sl, PROJ/'26_v3_조합x알고리즘_히트맵.png', 0.5, 1.6, h=5.45)
xr = 0.5 + 5.45 * 1950 / 1120 + 0.2
text(sl, xr, 1.6, W - 0.5 - xr, 5.45, [[H('무엇을 바꿔봤나')], [H('특징점: ', c=INK), ('QR 평균, 파인더, 네모 크기, 원근 변환, 내부 점 263개 / QR 밖 볼트, 브래킷', {})],
     [H('알고리즘: ', c=INK), ('비율 나누기, 회귀, 야코비안 최소제곱, 가중(GLS), 가우스-뉴턴(곡선, 반복), 픽셀 직접 가우스-뉴턴', {})],
     [H('알게 된 것', c=NAVY)], '가우스-뉴턴을 특징점에 쓰면 거의 그대로 (이미 직선이라)', '픽셀 직접 방식(G1)이 QR만 쓰는 방법 중 최고', '풀이법보다 쓰는 정보가 중요: QR 밖을 쓰면 12~19µm'],
     size=11.5, fill=BOX)
notes(sl, '칸마다 Y 오차 RMS와 Y 사진 80장 중 구분 불가 수. 짙을수록 좋음.')

# ======================= 19. 구분 안 되는 짝 =======================
sl = prs.slides.add_slide(BL)
head(sl, '5. 계산 방법별 결과', '구분 안 되는 짝이 방법에 따라 얼마나 줄어드나', '위 줄 Y 사진, 아래 줄 Θ 사진(±0.40). 굵은 테두리와 선 = QR 위치는 같은데 구분이 안 되는 사진')
yy = 1.65
for img, lab, nums in [('basis_최적.png', 'QR 네모 위치만 (가장 닮은 짝 기준)', 'Y 45 / 80장, Θ 41 / 71장  |  Y ±0.80mm 이내 전부 불가'),
                       ('method_v2-a2.png', 'QR 판 원근 변환', 'Y 15장, Θ 16장  |  Y ±0.30mm 이내 전부 불가'),
                       ('method_G1.png', '픽셀 직접 가우스-뉴턴 (QR만 최고)', 'Y 12장, Θ 8장  |  Y ±0.30mm 이내 전부 불가, ±0.35mm부터 전부 구분'),
                       ('method_O3+C5.png', '참고: QR 밖 브래킷 점 2개', 'Y 1장, Θ 1장  |  거의 전부 구분')]:
    text(sl, 0.5, yy, W - 1.0, 0.3, [[H(lab + '   ', c=INK, s=13), (nums, {'size': 11.5, 'color': SUBC})]])
    p_ = pic(sl, SI/img, 0.6, yy + 0.32, w=W - 1.2); yy += 0.32 + (W - 1.2) * 332 / 3948 + 0.12
notes(sl, '짝 기준: QR 3개 가로 평균 px가 같은 상대 축 지점. 판정 규칙은 방법마다 다름(첫 줄 값마다 3σ, 둘째 통계 거리, 셋째와 넷째 추정 Θ 3σ).')

# ======================= 20. 방법별 결과 =======================
sl = prs.slides.add_slide(BL)
head(sl, '5. 계산 방법별 결과', '방법별 Y 오차: QR만 쓰면 최고 91µm, QR 밖을 쓰면 12~19µm')
bar_chart(sl, 0.5, 1.35, 6.7, 5.7, ['QR 네모 크기, 간격 (Q4)', 'QR 내부 점 263개 (Q7)', 'QR 판 원근 변환 (Q5b)', '픽셀 직접 가우스-뉴턴 (G1)', '허용 오차 (참고)', 'QR 밖 브래킷 2점 (O3)', 'QR 밖 특징점 74개 (O5)'],
          [191, 137, 119, 91, 90, 19, 12], 'Y 오차 RMS (µm, 교차검증)', 'µm', colors=[BLUE, BLUE, BLUE, NAVY, SUBC, GRN, GRN], maxv=230)
table(sl, 7.45, 1.35, 5.38, [('방법', 'Y 0.09mm 안', '구분 불가 Y/Θ'), ('QR 평균만 (기준선)', '24%', '80 / 68'), ('Q4 네모 크기', '37%', '28 / 20'), ('Q7 내부 점', '51%', '16 / 15'),
      ('Q5b 판 원근 변환', '51%', '13 / 15'), ('G1 픽셀 직접', '69%', '12 / 8'), ('O3 QR 밖 2점', '100%', '1 / 1'), ('O5 QR 밖 74점', '100%', '0 / 0')],
      widths=[2.1, 1.3, 1.5], size=11, rh=0.38, hl=lambda i, r: r[0].startswith('G1'))
text(sl, 7.45, 1.35 + 0.38 * 8 + 0.2, 5.38, 7.05 - (1.35 + 0.38 * 8 + 0.2), [[H('읽는 법')], 'QR만 쓰는 최고(G1)도 사진 중 31%는 0.09mm를 넘음',
     'QR 평균만 쓰는 기준선은 Y 오차 612µm (그래프 밖)', 'QR 밖을 쓸 수 있는지가 가장 큰 갈림길'], size=11.5, fill=BOX)
notes(sl, '교차검증: 짝/홀 2겹(어제 결과표와 같은 분할). 구분 불가: 추정 Θ가 짝의 Θ에서 3σ 안.')

# ======================= 21. 믿을 만한가 =======================
sl = prs.slides.add_slide(BL)
head(sl, '5. 계산 방법별 결과', '믿을 수 있나: 이 데이터 안에서는 그렇다, 현장은 아직 모른다')
table(sl, 0.5, 1.4, 7.0, [('점검 항목', '상태', '설명'), ('학습과 채점 분리', '△', '짝/홀 교차검증. 이웃 사진이 섞여 실제보다 좋게 나올 수 있음'),
      ('정답값 정확도', '△', '명령값 자체에 약 ±40µm 오차'), ('같은 조건 반복', '✕', '촬영 1회뿐. 다른 날, 다른 위치 결과 모름'),
      ('실제 사용 조건', '✕', 'Θ와 Y가 같이 틀어진 사진 없음'), ('최종 목표 검증', '✕', '보정 후 다시 찍은 사진 없음'), ('측정 자체', '✓', '잡음 0.02px, 방법끼리 일치')],
      widths=[1.6, 0.6, 4.0], size=11, rh=0.55, hl=lambda i, r: r[1] == '✕')
bar_chart(sl, 7.8, 1.4, 5.03, 3.9, ['1차 +', '1차 -', '2차 +', '2차 -'], [12, 36, 7, 29],
          '구간 하나를 빼고 채점한 Y 오차', 'µm', colors=[BLUE, RED, BLUE, RED], maxv=45, horizontal=False)
text(sl, 7.8, 5.45, 5.03, 1.6, [[H('왜 중요한가', c=RED)], '빼는 구간에 따라 오차가 5배 달라짐. 독립 표본이 사진 280장이 아니라 사실상 촬영 구간 몇 개. 새로 찍으면 결과가 달라질 수 있음'], size=11.5, fill=PEACH)
notes(sl, '구간 빼고 채점은 Y 데이터, 단순 선형 방법 기준. "항상 0.09mm 이하"라고 말하려면 독립 시행이 최소 60회 필요(실패 0이면 95% 성공).')

# ======================= 22. 해볼 것 =======================
sl = prs.slides.add_slide(BL)
head(sl, '5. 계산 방법별 결과', '앞으로 해볼 것')
road = [('지금 데이터로', ['G1을 기본 방법으로 정리', '여러 장 촬영 시 작은 틀어짐 구간이 얼마나 줄어드는지 시뮬레이션', '촬영 구간 단위 교차검증으로 보수적 수치']),
        ('답변을 받으면', ['QR 밖 사용 가능 → 브래킷 기준 방법으로', '여러 장 허용 → 로봇 미세 이동 다중 촬영', '캘리브레이션 → 물리 공식 기반(PnP) 비교']),
        ('새 데이터를 받으면', ['보정 후 재촬영으로 최종 성공률', 'center 반복 촬영으로 백래시 확정', 'Θ, Y 혼합 사진으로 실제 조건 검증']),
        ('구현', ['C++ (OpenCV, MIL) 이식', '보간 정밀도(1/32px) 대책', '처리 시간 측정'])]
for i, (a, items) in enumerate(road):
    x = 0.5 + i * 3.17
    text(sl, x, 1.55, 2.95, 0.6, a, size=15, bold=True, color=WHITE, fill=[NAVY, BLUE, ORG, GRN][i], anchor=MSO_ANCHOR.MIDDLE, align=PP_ALIGN.CENTER)
    text(sl, x, 2.3, 2.95, 3.2, items, size=12.5, fill=BOX)
    if i < 3: arrow(sl, x + 2.97, 1.85, x + 3.15, 1.85, NAVY, 2.5)
text(sl, 0.5, 5.75, W - 1.0, 1.3, [[H('목표 그림: ', c=INK), ('사진 1장 → G1로 Y, Θ 계산 → 확실하면 바로 보정, 불확실하면 추가 촬영 → 같은 방향으로 접근해 이동 → 재촬영해 center와 비교 → 허용 오차 안이면 끝', {})]], size=13, fill=SKY)
notes(sl, '왼쪽부터 순서대로. 두 번째, 세 번째 칸은 SEMES 답변과 데이터에 달려 있음.')

# ======================= 23. 데이터가 더 필요한 이유 =======================
sl = prs.slides.add_slide(BL)
head(sl, '6. 필요한 것과 질문', '데이터가 더 필요한 이유: 분석에서 나온 근거')
table(sl, 0.5, 1.35, W - 1.0, [('분석에서 나온 것', '그래서 필요한 데이터', '수량'),
      ('보정 후 사진이 없어 "보정하면 0.09mm 안" 검증 불가', '보정 후 재촬영 (틀어놓고 → 우리 보정값 → 이동 → 재촬영)', '60회 (실패 0이면 95% 성공 주장)'),
      ('같은 "0"인데 방향별 최대 44µm 차이 (간접 추정)', '같은 자리 반복: center로 + 방향, - 방향 각각 돌아와 촬영', '방향별 30회'),
      ('모든 사진이 한 축씩만 틀어짐', 'Θ, Y 동시에 틀어진 사진 (무작위 순서)', '격자 49장 + 임의 50장'),
      ('촬영 1회, 독립 표본은 사실상 구간 몇 개', '다른 날, 다른 위치에서 같은 방식', '1회 + 2곳, 각 20장'),
      ('잡음이 카메라 때문인지 로봇 때문인지 미분리', '움직이지 않고 연속 촬영', '30장'),
      ('파일 이름이 명령값으로 추정, 엔코더 값 없음', '사진마다 시각, 이동 방향, 명령값, 엔코더 값', '모든 사진')],
      widths=[3.0, 3.4, 1.8], size=11.5, rh=0.62)
bar_chart(sl, 0.5, 5.85, 5.6, 1.35, ['지금', '요청 후'], [4, 60], '독립 시행 수', '', colors=[RED, GRN], maxv=70)
text(sl, 6.4, 5.85, W - 6.9, 1.3, [[H('3의 법칙: ', c=INK), ('n번 시험에서 실패가 0이면 실패율 상한(95%)은 약 3/n. 지금(사실상 4)으로는 아무 말도 못 하고, 60회면 "95% 이상 성공"이라고 말할 수 있음. 총 요청량 약 370장.', {})]], size=12, fill=BOX)
notes(sl, '각 요청은 왼쪽 칸의 분석 결과에서 나온 것. 지금 독립 시행 4는 Y 촬영 구간 수.')

# ======================= 24. 질문 =======================
sl = prs.slides.add_slide(BL)
head(sl, '6. 필요한 것과 질문', 'SEMES에 묻고 싶은 것', '답에 따라 알고리즘 방향이 달라지는 질문부터')
qs = [('허용 오차', 'Y 0.09mm가 맞나요? ±인가요? Θ 허용 오차와 단위(도°)는?', '성공 기준'),
      ('QR 밖 영역', '브래킷처럼 위치마다 같은 구조물을 써도 되나요?', '되면 Y 약 0.02mm'),
      ('여러 장, 반복 보정', '한 번 보정에 여러 장 찍거나 다시 찍어 확인해도 되나요? 허용 시간은?', '작은 틀어짐 해결'),
      ('사람의 Y 보정', '지금 사람은 Y를 무엇으로 맞추나요? 그 측정값을 정답으로 받을 수 있나요?', '독립 정답값'),
      ('center 만드는 법', '언제, 어떤 절차로, 어느 방향으로 접근해 찍나요? 위치마다 있나요?', '목표 위치 재현성'),
      ('받은 데이터', '파일 이름은 명령값인가요? 매번 0으로 돌아갔나요? "보정"은 무엇을 했나요?', '정답값 해석')]
for i, (a, b, c) in enumerate(qs):
    x = 0.5 + (i % 3) * 4.15; y = 1.75 + (i // 3) * 2.65
    text(sl, x, y, 3.95, 2.45, [[H(f'Q{i+1}. {a}', s=15)], b, [(f'→ {c}', {'size': 11.5, 'color': SUBC, 'bold': True})]], size=12.5, fill=BOX if i % 2 == 0 else SKY)
notes(sl, '전체 질문 26개는 00_SEMES질문세트.md에 있음. 이 6개가 가장 영향이 큼.')

# ======================= 25. 피드백 요청 =======================
sl = prs.slides.add_slide(BL)
head(sl, '6. 필요한 것과 질문', '피드백을 부탁드립니다')
fb = [('우리 이해가 맞나요?', ['정답 = center 사진과 같아지는 상태', 'Y와 Θ를 나누는 게 핵심 문제', '0점 차이 원인 = 백래시']),
      ('방향이 맞나요?', ['QR 판 전체 픽셀로 푸는 G1을 기본으로', '불확실하면 추가 촬영 또는 사람 확인', '보정은 같은 방향으로 접근']),
      ('현실적인가요?', ['요청 데이터 약 370장 (로봇 시간)', 'center 절차 제안 (예열, 단방향, 평균)', '출력에 "불확실" 표시 추가'])]
for i, (a, items) in enumerate(fb):
    x = 0.5 + i * 4.15
    c = sl.shapes.add_shape(MSO_SHAPE.OVAL, Inches(x + 1.55), Inches(1.5), Inches(0.85), Inches(0.85)); c.fill.solid(); c.fill.fore_color.rgb = [NAVY, BLUE, GRN][i]; c.line.fill.background()
    c.text_frame.text = '?'; p = c.text_frame.paragraphs[0]; p.alignment = PP_ALIGN.CENTER; p.runs[0].font.size = Pt(28); p.runs[0].font.bold = True; p.runs[0].font.color.rgb = WHITE
    text(sl, x, 2.5, 3.95, 3.2, [[H(a, c=INK, s=17)]] + items, size=13, fill=BOX)
text(sl, 0.5, 5.95, W - 1.0, 1.1, [[H('자료: ', c=INK), ('분석 기록 visualization_2026101301.pptx (17장), 결과표 25_YΘ구분표_v3.xlsx, 질문 전체 00_SEMES질문세트.md, 리포트 00_v3결과리포트.md', {})]], size=12, fill=SKY)
notes(sl, '피드백을 받고 싶은 세 가지. 답에 따라 다음 단계(22장)의 두 번째, 세 번째 칸이 정해짐.')

prs.save(DST); print('저장:', DST, len(prs.slides), '장')
