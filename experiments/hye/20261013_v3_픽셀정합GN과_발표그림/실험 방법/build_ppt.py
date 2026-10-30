"""visualization_2026101301.pptx 뒤에 분석 슬라이드를 추가한다. 기존 슬라이드 1~3은 건드리지 않는다.
다시 실행하면 원본 백업에서 시작해 같은 결과를 만든다."""
import pandas as pd, numpy as np
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from slide_data import *
IMG = HERE/'slide_img'
SRC = HERE/'visualization_2026101301_원본백업.pptx'; DST = PROJ/'visualization_2026101301.pptx'
FONT = '맑은 고딕'
INK, SUBC, NAVY, ORGC, GRNC, REDC, BOX = (RGBColor(0x1F, 0x1F, 0x1E), RGBColor(0x5F, 0x5E, 0x58), RGBColor(0x1F, 0x4E, 0x78),
                                          RGBColor(0xC5, 0x5A, 0x11), RGBColor(0x4E, 0x7A, 0x1E), RGBColor(0xC0, 0x00, 0x00), RGBColor(0xF2, 0xF2, 0xF2))
prs = Presentation(SRC)
BLANK = [l for l in prs.slide_layouts if l.name == '빈 화면'][0]
W = prs.slide_width / 914400

def text(sl, x, y, w, h, runs, size=14, bold=False, color=INK, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, fill=None, name=None):
    """runs: 문자열, 또는 줄 목록. 줄은 문자열이거나 [(글자, {옵션})] 목록."""
    if fill is not None:
        bx = sl.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
        bx.adjustments[0] = 0.08; bx.fill.solid(); bx.fill.fore_color.rgb = fill; bx.line.fill.background(); bx.shadow.inherit = False
        tb = bx; tf = bx.text_frame
    else:
        tb = sl.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h)); tf = tb.text_frame
    if name: tb.name = name
    tf.word_wrap = True; tf.vertical_anchor = anchor
    for side in ('left', 'right', 'top', 'bottom'): setattr(tf, f'margin_{side}', Inches(0.12 if fill is not None else 0))
    lines = runs if isinstance(runs, list) else [runs]
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph(); p.alignment = align
        segs = [(line, {})] if isinstance(line, str) else line
        for t, o in segs:
            r = p.add_run(); r.text = t; f = r.font; f.name = FONT; f.size = Pt(o.get('size', size)); f.bold = o.get('bold', bold)
            f.color.rgb = o.get('color', color)
        if fill is not None or len(lines) > 1: p.space_after = Pt(o.get('after', 4) if segs else 4)
    return tb

def title(sl, t, sub=None):
    text(sl, 0.5, 0.32, W - 1.0, 0.6, t, size=26, bold=True, name='제목')
    if sub: text(sl, 0.5, 0.92, W - 1.0, 0.35, sub, size=13, color=SUBC, name='부제')

def fixed_box(sl, y, items, h=0.62):
    """고정 조건 상자: [(라벨, 내용)]"""
    line = []
    for i, (k, v) in enumerate(items):
        if i: line.append(('     ', {}))
        line += [(f'{k}  ', {'bold': True, 'color': NAVY}), (v, {})]
    return text(sl, 0.5, y, W - 1.0, h, [line], size=11.5, fill=BOX, anchor=MSO_ANCHOR.MIDDLE, name='고정 조건')

def fmt_sum(s):
    y, t = s['Y'], s['Θ']
    ag = lambda v, u: '없음' if v is None else f'±{v:.2f}{u}'
    ab = lambda v, u: '전부 불가 구간 없음' if v == 0 else f'±{v:.2f}{u} 이내 전부 불가'
    return y, t, (f'구분 불가 Y {y["n_bad"]}/{y["n"]}장, Θ {t["n_bad"]}/{t["n"]}장   |   '
                  f'Y: {ab(y["allbad"], "mm")}, {ag(y["allgood"], "mm")} 이상 전부 구분   |   '
                  f'Θ: {ab(t["allbad"], "")}, {ag(t["allgood"], "")} 이상 전부 구분')

def panel(sl, y, label, img, numbers, w=None, lab_h=0.3):
    w = w or (W - 1.0); ih = w * 332 / 3948
    text(sl, 0.5, y, W - 1.0, lab_h, [[(label + '   ', {'bold': True, 'size': 13}), (numbers, {'size': 11, 'color': SUBC})]], size=12)
    sl.shapes.add_picture(str(img), Inches(0.5 + (W - 1.0 - w) / 2), Inches(y + lab_h), Inches(w), Inches(ih))
    return y + lab_h + ih

LEGEND = '굵은 테두리 = 구분 안 됨, 흐린 사진 = 구분됨, 파란 선 = Y 사진에서 QR 위치가 같은 Θ 지점, 빨간 점선 = Θ 사진에서 같은 Y 지점. 위 줄 Y 80장, 아래 줄 Θ ±0.40 80장 (가운데 center)'
RULE1 = ('판정', '좌, 중, 우 QR 가로 px 차이가 모두 3σ 안이면 구분 불가 (임계값 좌 0.045, 중 0.027, 우 0.048px)')
VALS1 = ('비교한 값', 'QR 네모 중심 가로 px 3개 (좌, 중, 우)')
REC = []   # 요약표용 (구분, 짝 기준, 비교한 값, 판정, Y 불가, Θ 불가, Y 전부 불가, Y 전부 구분, Y 오차 RMS)

# ---------------- 1. 공통 조건 ----------------
sl = prs.slides.add_slide(BLANK)
title(sl, 'QR 위치로 Y와 Θ를 구분할 수 있나: 분석 조건', '같은 QR 위치로 보이는 Y 사진과 Θ 사진 짝을 찾고, 그 짝을 구분할 수 있는지 판정')
text(sl, 0.5, 1.45, 6.0, 3.6, [
    [('데이터', {'bold': True, 'color': NAVY})],
    'Y 사진 80장 (Y −2.00 ~ +2.00mm, 0.05 간격, Θ = 0)',
    'Θ 사진 200장 (Θ −1.00 ~ +1.00, 0.01 간격, Y = 0), center 1장',
    'Θ 단위는 미확인 (화면 이동량으로 보면 도°로 추정)',
    [('짝 (매핑)', {'bold': True, 'color': NAVY})],
    '어떤 사진의 QR 위치(선택한 값)가 상대 축 사진들의 곡선에서 같아지는 지점. 상대 축 사진이 띄엄띄엄 있어서 보간으로 찾음',
    'Y 1mm = QR 14.3px, Θ 1 = 80px라서 Y 전체(±2mm)는 Θ ±0.36과만 겹침. 그 밖의 Θ 사진 129장은 짝이 없음',
    [('구분 판정', {'bold': True, 'color': NAVY})],
    '짝과 비교했을 때 차이가 잡음 범위(3σ) 안이면 "구분 불가". 잡음은 같은 축 사진끼리 비교해서 구함'], size=13)
text(sl, 6.9, 1.45, 5.9, 3.6, [
    [('슬라이드마다 고정한 것', {'bold': True, 'color': NAVY})],
    '짝 기준: 어떤 값이 같으면 짝으로 볼지 (좌, 중, 우, 평균 등)',
    '비교한 값: 짝과 무엇을 비교하는지',
    '판정 규칙: 어떤 계산으로 구분 가능, 불가를 정하는지',
    [('결과로 보는 것', {'bold': True, 'color': NAVY})],
    '구분 불가 세트 수 (Y 80장, Θ 71장 중. Θ는 짝이 생길 수 있는 사진만)',
    '가운데부터 "전부 구분 불가"인 범위, 바깥에서 "전부 구분"되기 시작하는 지점',
    '그 사이는 구분되는 사진과 안 되는 사진이 섞여 있음'], size=13, fill=BOX)
y = panel(sl, 5.25, '예시: 짝 기준 좌중우 평균 (v1)', IMG/'basis_좌중우.png', '')
text(sl, 0.5, y + 0.08, W - 1.0, 0.3, LEGEND, size=10, color=SUBC)

# ---------------- 2. 짝 기준 좌 / 중 / 우 ----------------
sl = prs.slides.add_slide(BLANK)
title(sl, '짝 기준 1: QR 좌 / 중 / 우 하나만 맞출 때')
fixed_box(sl, 0.98, [('짝 기준', 'QR 하나의 가로 px가 같은 상대 축 지점'), VALS1, RULE1])
y = 1.75
for b, lab in [('좌', '좌 QR 기준'), ('중', '중 QR 기준'), ('우', '우 QR 기준')]:
    s = summary(pairing(b)); yy, tt, nums = fmt_sum(s)
    y = panel(sl, y, lab, IMG/f'basis_{b}.png', nums) + 0.22
    REC.append(('짝 기준', f'{b} QR', 'QR 네모 가로 3개', '규칙1 (값마다 3σ)', s, None))
text(sl, 0.5, 7.0, W - 1.0, 0.3, LEGEND, size=10, color=SUBC)

# ---------------- 3. 짝 기준 평균 4가지 ----------------
sl = prs.slides.add_slide(BLANK)
title(sl, '짝 기준 2: QR 좌중 / 좌우 / 중우 / 좌중우 평균')
fixed_box(sl, 0.98, [('짝 기준', '선택한 QR들의 가로 px 평균이 같은 상대 축 지점'), VALS1, RULE1], h=0.55)
y = 1.62
for b in ['좌중', '좌우', '중우', '좌중우']:
    s = summary(pairing(b)); yy, tt, nums = fmt_sum(s)
    y = panel(sl, y, f'{b} 평균 기준' + (' (v1)' if b == '좌중우' else ''), IMG/f'basis_{b}.png', nums, w=11.6, lab_h=0.28) + 0.1
    REC.append(('짝 기준', f'{b} 평균', 'QR 네모 가로 3개', '규칙1 (값마다 3σ)', s, None))

# ---------------- 4. 최적 짝 ----------------
sl = prs.slides.add_slide(BLANK)
title(sl, '짝 기준 3: 가장 닮은 지점을 짝으로 (최적 짝)', '짝 기준에 따라 결과가 달라진다. 공정하게 세려면 가장 헷갈리는 짝을 기준으로 해야 한다')
fixed_box(sl, 1.4, [('짝 기준', '상대 축 곡선에서 좌, 중, 우 차이가 임계값 대비 가장 작아지는 지점'), VALS1, RULE1])
s = summary(pairing_best()); _, _, nums = fmt_sum(s)
y = panel(sl, 2.2, '최적 짝 기준', IMG/'basis_최적.png', nums)
REC.append(('짝 기준', '최적 짝', 'QR 네모 가로 3개', '규칙1 (값마다 3σ)', s, None))
rows = [('짝 기준', 'Y 구분 불가', 'Θ 구분 불가', 'Y 전부 불가', 'Y 전부 구분')]
for b in list(BASES) + ['최적']:
    ss = summary(pairing(b) if b != '최적' else pairing_best())
    rows.append((b if b != '최적' else '최적 짝', f'{ss["Y"]["n_bad"]}/80', f'{ss["Θ"]["n_bad"]}/71', f'±{ss["Y"]["allbad"]:.2f}mm 이내', f'±{ss["Y"]["allgood"]:.2f}mm 이상'))
tb = sl.shapes.add_table(len(rows), 5, Inches(0.5), Inches(y + 0.3), Inches(7.4), Inches(0.3 * len(rows))).table
for i, r in enumerate(rows):
    for j, v in enumerate(r):
        cell = tb.cell(i, j); cell.text = v; p = cell.text_frame.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
        f = p.runs[0].font; f.name = FONT; f.size = Pt(11); f.bold = (i == 0 or r[0] == '최적 짝'); f.color.rgb = RGBColor(255, 255, 255) if i == 0 else INK
        cell.fill.solid(); cell.fill.fore_color.rgb = NAVY if i == 0 else (RGBColor(0xFC, 0xE4, 0xD6) if r[0] == '최적 짝' else RGBColor(255, 255, 255))
        cell.margin_top = cell.margin_bottom = Inches(0.02)
text(sl, 8.2, y + 0.3, 4.6, 3.0, [
    [('읽는 법', {'bold': True, 'color': NAVY})],
    'QR 하나만 맞춘 짝은 나머지 QR이 어긋나 보여서 구분되는 것처럼 보인다 (좌 기준 18장).',
    '가장 닮은 지점을 짝으로 삼으면 Y 45장, Θ 41장이 구분 안 된다. QR 네모 가로 위치만으로는 Y ±0.80mm 이내가 전부 구분 불가.',
    '다음 슬라이드부터는 짝 기준을 좌중우 평균으로 고정하고, 비교하는 값과 판정 방법을 바꾼다.'], size=12.5, fill=BOX)

# ---------------- 5. 다른 처리 방법 ----------------
V3 = pd.read_pickle(HERE/'v3.pkl')['R']; est2 = pd.read_csv(PROJ/'analysis_v2/estimate_result.csv')
def rms3(fn, an): return float(V3[(V3.특징점 == fn) & (V3.알고리즘 == an)].Y_RMS.iloc[0]) * 1000
def rms2(k): return float(est2[est2.버전.str.startswith(k)].Y_RMS_um.iloc[0])
def sig3(fn, an): return float(V3[(V3.특징점 == fn) & (V3.알고리즘 == an)].σΘ.iloc[0])
MD = methods()
METHODS = [
 ('v2-b', '처리 1: QR 네모에서 값을 더 뽑기 (세로 위치, 폭, 높이)',
  ['QR 네모 3개마다 가로 위치뿐 아니라 세로 위치, 폭, 높이까지 12개 값을 비교한다.',
   'Θ로 돌면 판이 비스듬해져 가까운 쪽 QR이 커 보인다(높이, 폭 차이). 이 원근 단서를 직접 쓴다.',
   '값이 늘어서 잡음이 섞이는 만큼 판정은 통계 거리(규칙2)로 한다.'],
  [('짝 기준', '좌중우 평균 (고정)'), ('비교한 값', 'QR 네모 가로, 세로, 폭, 높이 12개'), ('판정', '규칙2: 12개 값의 통계 거리(마할라노비스, 잡음 공분산 반영)가 99.73% 기준 안이면 불가')],
  '규칙2 (통계 거리)', 'QR 네모 12개', rms2('v2-b'), 'v2 추정'),
 ('v2-a1', '처리 2: QR마다 내부 픽셀 전체로 맞추기 (ECC affine)',
  ['QR 하나하나를 기준 사진의 같은 QR에 픽셀 단위로 겹쳐 맞춘다 (ECC 반복 정렬).',
   '결과로 QR마다 이동, 기울기, 크기 6개 값, 3개 QR이면 18개 값을 얻는다. 네모 외곽보다 QR 안 무늬 전체를 쓰므로 잡음이 적다.'],
  [('짝 기준', '좌중우 평균 (고정)'), ('비교한 값', 'QR별 affine 변환 6개 x 3 = 18개'), ('판정', '규칙2: 통계 거리 99.73% 기준')],
  '규칙2 (통계 거리)', 'QR별 affine 18개', rms2('v2-a1'), 'v2 추정'),
 ('v2-a2', '처리 3: QR 3개가 붙은 판 전체를 한 번에 맞추기 (원근 변환)',
  ['QR 3개가 붙은 판은 평평해서, 사진 사이 변형은 원근 변환(호모그래피) 하나로 정확히 표현된다.',
   '판 전체를 한 번에 겹쳐 맞추고, 판 귀퉁이 4개와 QR 중심 3개의 이동(14개 값)을 비교한다.',
   '판 양끝(약 36mm)을 다 써서 "판이 얼마나 비스듬한가"를 QR 하나씩보다 잘 잡는다.'],
  [('짝 기준', '좌중우 평균 (고정)'), ('비교한 값', '판 귀퉁이 4개 + QR 중심 3개 이동 14개'), ('판정', '규칙2: 통계 거리 99.73% 기준')],
  '규칙2 (통계 거리)', '판 원근 변환 14개', rms3('Q5b 판 전체 원근 변환, 귀퉁이+QR중심 (14개)', 'C6 가우스-뉴턴 (곡선 모델, 반복)'), 'v3 Q5b+C6'),
 ('Q7+C5', '처리 4: QR 안쪽 특징점 263개 + 가중 최소제곱 (어제 QR만 최고)',
  ['QR 내부의 모서리 점 263개를 사진마다 추적한다.',
   '"Θ 1, Y 1mm일 때 각 점이 얼마나 움직이는지"(야코비안)를 학습하고, 점마다 잡음 크기와 서로 얽힌 정도로 가중해 Θ, Y를 푼다 (GLS).',
   '직선 모델을 한 번에 푸는 것이라 가우스-뉴턴 1회 반복과 같다.'],
  [('짝 기준', '좌중우 평균 (고정)'), ('비교한 값', '추정한 Θ (점 263개로 계산)'), ('판정', f'규칙3: 추정 Θ가 짝의 Θ에서 3σ 안이면 불가 (σ = {sig3("Q7 QR 내부 특징점 (263개)", "C5 야코비안 가중(GLS)"):.4f})')],
  '규칙3 (추정 Θ)', 'QR 내부 점 263개', rms3('Q7 QR 내부 특징점 (263개)', 'C5 야코비안 가중(GLS)'), 'v3'),
 ('G1', '처리 5: 픽셀 직접 가우스-뉴턴 (QR만 쓰는 방법 중 최고)',
  ['특징점을 뽑지 않고 QR 판 영역 픽셀 전부로 Θ, Y를 직접 푼다. 기준 영상을 (Θ, Y)만큼 변형해 새 사진과 밝기 차이가 최소가 되도록 반복.',
   '모델에 없는 작은 변형(세로 이동, 화면 내 회전, 크기)은 따로 풀어서 Θ, Y로 새어 들어가지 않게 한다. 이게 없으면 크게 틀린다.',
   'OpenCV 보간은 위치를 1/32px 단위로 잘라 신호(0.02~0.06px)가 깨진다. 정확한 소수 위치 보간으로 바꿈 (C++ 구현 때도 주의).'],
  [('짝 기준', '좌중우 평균 (고정)'), ('비교한 값', '추정한 Θ (QR 판 픽셀 전체로 계산)'), ('판정', f'규칙3: 추정 Θ가 짝의 Θ에서 3σ 안이면 불가 (σ = {sig3("G1 QR 판 픽셀 전체 (특징점 없음)", "G1 픽셀 직접 가우스-뉴턴 (Θ, Y + 방해 3)"):.4f})')],
  '규칙3 (추정 Θ)', 'QR 판 픽셀 전체', rms3('G1 QR 판 픽셀 전체 (특징점 없음)', 'G1 픽셀 직접 가우스-뉴턴 (Θ, Y + 방해 3)'), 'v3'),
 ('O3+C5', '참고: QR 밖 고정 구조물을 쓸 때 (브래킷 윗판 + 아래)',
  ['QR 판과 거리가 다른 브래킷 위, 아래 점 2개를 쓴다. 옆으로 이동(Y)하면 가까운 것이 더 많이 움직이고, 회전(Θ)하면 거리와 상관없이 비슷하게 움직인다 (시차).',
   '시차 신호가 원근 신호보다 수십 배 커서 거의 전부 구분된다. QR 밖 영역을 써도 되는지는 SEMES 확인 필요.'],
  [('짝 기준', '좌중우 평균 (고정)'), ('비교한 값', '추정한 Θ (브래킷 점 2개로 계산)'), ('판정', f'규칙3: 추정 Θ가 짝의 Θ에서 3σ 안이면 불가 (σ = {sig3("O3 조합B2 브래킷 윗판+아래 (2개)", "C5 야코비안 가중(GLS)"):.4f})')],
  '규칙3 (추정 Θ)', 'QR 밖 브래킷 점 2개', rms3('O3 조합B2 브래킷 윗판+아래 (2개)', 'C5 야코비안 가중(GLS)'), 'v3'),
]
for key_, ttl, expl, fixed, rule, vals, rms, src in METHODS:
    sl = prs.slides.add_slide(BLANK); title(sl, ttl)
    s = summary(MD[key_]); y_, t_, nums = fmt_sum(s)
    text(sl, 0.5, 1.25, 8.3, 2.55, [[('무엇을 했나', {'bold': True, 'color': NAVY})]] + expl, size=13)
    stats = [(f'{y_["n_bad"]}', '/80', 'Y 사진 구분 불가'), (f'{t_["n_bad"]}', '/71', 'Θ 사진 구분 불가'), (f'{rms:.0f}', 'µm', f'Y 오차 RMS ({src})')]
    for k, (big, unit, lab) in enumerate(stats):
        text(sl, 9.1, 1.2 + k * 0.9, 3.73, 0.8, [[(big, {'size': 30, 'bold': True, 'color': NAVY}), (' ' + unit, {'size': 14, 'color': SUBC}),
                                                    ('   ' + lab, {'size': 12, 'color': INK})]], fill=BOX, anchor=MSO_ANCHOR.MIDDLE)
    fixed_box(sl, 4.0, fixed, h=0.6)
    y = panel(sl, 4.95, '연결도', IMG/f'method_{key_}.png', nums)
    text(sl, 0.5, y + 0.12, W - 1.0, 0.3, LEGEND, size=10, color=SUBC)
    REC.append(('처리 방법' if key_ != 'O3+C5' else '참고 (QR 밖)', key_, vals, rule, s, rms))

# ---------------- 6. 요약표 ----------------
sl = prs.slides.add_slide(BLANK)
title(sl, '한눈에 보기: 짝 기준과 처리 방법별 구분 불가 수', '아래로 갈수록 구분 안 되는 범위가 좁아진다. QR만 쓰는 최고는 G1, QR 밖까지 쓰면 거의 전부 구분')
hd = ('구분', '짝 기준 / 방법', '비교한 값', '판정 규칙', 'Y 불가/80', 'Θ 불가/71', 'Y 전부 불가', 'Y 전부 구분', 'Y 오차 RMS')
rows = [hd] + [(g, k, v, r, str(s['Y']['n_bad']), str(s['Θ']['n_bad']), ('없음' if s["Y"]["allbad"] == 0 else f'±{s["Y"]["allbad"]:.2f}mm'), f'±{s["Y"]["allgood"]:.2f}mm', '-' if m is None else f'{m:.0f}µm')
               for g, k, v, r, s, m in REC]
widths = [1.3, 1.45, 2.05, 1.95, 1.0, 1.0, 1.15, 1.15, 1.08]
tb = sl.shapes.add_table(len(rows), len(hd), Inches(0.5), Inches(1.45), Inches(sum(widths)), Inches(0.33 * len(rows))).table
for j, wv in enumerate(widths): tb.columns[j].width = Inches(wv)
for i, r in enumerate(rows):
    tb.rows[i].height = Inches(0.33)
    hl = r[1] in ('G1', '최적 짝')
    for j, v in enumerate(r):
        cell = tb.cell(i, j); cell.text = v; p = cell.text_frame.paragraphs[0]; p.alignment = PP_ALIGN.CENTER if j >= 4 else PP_ALIGN.LEFT
        f = p.runs[0].font; f.name = FONT; f.size = Pt(10.5); f.bold = (i == 0 or hl); f.color.rgb = RGBColor(255, 255, 255) if i == 0 else INK
        cell.fill.solid(); cell.fill.fore_color.rgb = NAVY if i == 0 else (RGBColor(0xE2, 0xEF, 0xDA) if r[1] == 'G1' else (RGBColor(0xFC, 0xE4, 0xD6) if r[1] == '최적 짝' else RGBColor(255, 255, 255)))
        cell.margin_top = cell.margin_bottom = Inches(0.02); cell.vertical_anchor = MSO_ANCHOR.MIDDLE
text(sl, 0.5, 1.45 + 0.33 * len(rows) + 0.15, W - 1.0, 0.6,
     ['규칙1 = 값마다 3σ 임계값, 규칙2 = 통계 거리(마할라노비스) 99.73%, 규칙3 = 추정 Θ가 짝의 Θ에서 3σ 안. 짝 기준 행은 판정만 하고 Θ, Y를 추정하지 않아 오차 RMS가 없음.',
      'Y 오차 RMS: v2 방법은 v2 추정(짝/홀 교차검증), v3 방법은 어제 결과표와 같은 분할. 정답값 자체에 약 40µm 오차가 있어 그보다 작은 차이는 검증 불가.'], size=10, color=SUBC)

# ---------------- 7. 조합 x 알고리즘 히트맵 ----------------
sl = prs.slides.add_slide(BLANK)
title(sl, '특징점 조합 x 알고리즘 전체 비교 (가우스-뉴턴 포함)', '어제 결과표 13개 조합을 같은 분할로 재현하고, 판 원근 변환(Q5b), 가우스-뉴턴(C6), 픽셀 직접 가우스-뉴턴(G1)을 추가')
pic = sl.shapes.add_picture(str(PROJ/'26_v3_조합x알고리즘_히트맵.png'), Inches(0.5), Inches(1.4), height=Inches(5.25))
text(sl, 0.5 + pic.width / 914400 + 0.25, 1.4, W - 1.25 - pic.width / 914400, 5.25, [
    [('읽는 법', {'bold': True, 'color': NAVY})],
    '칸마다 Y 오차 RMS와 Y 사진 80장 중 구분 불가 수 (짙을수록 좋음)',
    [('가우스-뉴턴 효과', {'bold': True, 'color': NAVY})],
    'C4, C5(직선 모델)는 가우스-뉴턴 1회와 같다. 곡선 모델로 반복(C6)해도 QR만 쓰는 조합은 거의 그대로.',
    '반복이 효과를 본 건 먼 벽을 쓰는 O2뿐 (47 → 26µm).',
    '픽셀 직접 가우스-뉴턴(G1)은 QR만 쓰는 방법 중 최고 (91µm, 불가 12장).',
    [('결론', {'bold': True, 'color': NAVY})],
    '풀이법보다 쓰는 정보가 중요하다. QR 밖 구조물을 쓰면 14~19µm, 불가 0~1장.'], size=12, fill=BOX)

# ---------------- 8. 0점 어긋남을 실제 사진으로 ----------------
sl = prs.slides.add_slide(BLANK)
title(sl, '0점 어긋남을 실제 사진으로: center 바로 옆 사진 6장', '+ 방향으로 온 사진은 모두 예상보다 왼쪽(파랑), - 방향으로 온 Y -0.10만 오른쪽(빨강). 백래시 패턴과 같다')
fixed_box(sl, 1.4, [('예상 사진', 'center 사진을 (명령값 x 환산 계수)만큼 옮긴 것. 계수는 데이터에서 구간마다 맞춘 기울기 평균 (Y 14.37px/mm, Θ 80.5px)'),
                    ('색', 'QR 모서리마다 실제가 예상보다 가로로 밀린 양 (파랑 = 왼쪽, 빨강 = 오른쪽)')], h=0.62)
pic = sl.shapes.add_picture(str(PROJ/'29_0점어긋남_실제사진.png'), Inches(0.5), Inches(2.2), height=Inches(4.3))
zo = pd.read_csv(HERE/'zero_offset_photos.csv')
rows = [('사진', '방향', '예상보다')] + [(r.사진.replace('mm', ''), ('-' if r.구간.split(', ')[1].startswith('-') else '+'),
        (f'{r.차이um:+.0f}µm' if r.사진.startswith('Y') else f'{r.차이px:+.2f}px')) for r in zo.itertuples()]
x0 = 0.5 + pic.width / 914400 + 0.3; tw = W - 0.5 - x0
tb = sl.shapes.add_table(len(rows), 3, Inches(x0), Inches(2.2), Inches(tw), Inches(0.3 * len(rows))).table
for j, wv in enumerate([0.4, 0.22, 0.38]): tb.columns[j].width = Inches(tw * wv)
for i, r in enumerate(rows):
    for j, v in enumerate(r):
        cell = tb.cell(i, j); cell.text = v; p = cell.text_frame.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
        f = p.runs[0].font; f.name = FONT; f.size = Pt(10.5); f.bold = (i == 0 or r[1] == '-'); f.color.rgb = RGBColor(255, 255, 255) if i == 0 else INK
        cell.fill.solid(); cell.fill.fore_color.rgb = NAVY if i == 0 else (RGBColor(0xFC, 0xE4, 0xD6) if r[1] == '-' else RGBColor(255, 255, 255))
        cell.margin_top = cell.margin_bottom = Inches(0.02); cell.margin_left = cell.margin_right = Inches(0.04)
text(sl, x0, 2.2 + 0.3 * len(rows) + 0.15, tw, 6.95 - (2.2 + 0.3 * len(rows) + 0.15), [
    [('읽는 법', {'bold': True, 'color': NAVY})],
    'Y -0.05도 + 방향(아래에서 0쪽)으로 와서 파랑. 위치가 아니라 방향이 0점을 정함.',
    '두 무리 차이 약 40µm = 백래시 크기 추정.',
    '예상 사진은 우리가 계산한 것. 어긋난 건 사진이 아니라 명령값과 실제 로봇 위치 사이.'], size=10.5, fill=BOX)

# ---------------- 9. 한 칸 이동량과 공식 비교 ----------------
sl = prs.slides.add_slide(BLANK)
title(sl, '구간 안에서는 매 칸 같은 양씩 움직인다', 'Y는 ±2mm 전체에서 공식(0.1mm = 1.437px)과 0.15% 안쪽. Θ는 방향마다 비율이 0.6% 달라 공식 하나로는 끝에서 약 0.005 오차')
pic = sl.shapes.add_picture(str(PROJ/'30_한칸이동량_공식비교.png'), Inches(0.5), Inches(1.4), height=Inches(4.0))
_q = pd.read_csv(PROJ/'14_데이터_QR마커위치.csv'); _q['ax'] = _q.file.str[0]; _q['x'] = _q[['dLx', 'dCx', 'dRx']].mean(1)
_k = (_q.value * 100).round().astype(int)
_segs = [('Y 1차 +', _q[(_q.ax == 'Y') & (_k % 10 == 0) & (_q.value > 0)], True, 14.37), ('Y 1차 -', _q[(_q.ax == 'Y') & (_k % 10 == 0) & (_q.value < 0)], True, 14.37),
         ('Y 2차 +', _q[(_q.ax == 'Y') & (_k % 10 != 0) & (_q.value > 0)], True, 14.37), ('Y 2차 -', _q[(_q.ax == 'Y') & (_k % 10 != 0) & (_q.value < 0)], False, 14.37),
         ('Θ +', _q[(_q.ax == 'T') & (_q.value > 0)], True, 80.5), ('Θ -', _q[(_q.ax == 'T') & (_q.value < 0)], True, 80.5)]
rows = [('구간', '한 칸 공식', '한 칸 실제 평균', '흔들림', '기울기 (공식 대비)', '끝까지 쌓인 오차')]
for n, g, asc, S in _segs:
    g = g.sort_values('value', key=abs, ascending=asc); v = g.value.values; x = g.x.values; u = 0.1 if n[0] == 'Y' else 0.01
    stp = np.diff(x) / np.diff(v) * u; sp = abs(v[-1] - v[0]); tot = (x[-1] - x[0]) * np.sign(v[-1] - v[0]); e = (tot - S * sp) / S
    rows.append((n, f'{S*u:.3f}px', f'{stp.mean():.3f}px', f'±{stp.std():.3f}px', f'{(np.polyfit(v, x, 1)[0]/S-1)*100:+.2f}%',
                 f'{e*1000:+.1f}µm' if n[0] == 'Y' else f'Θ {e:+.4f}'))
tb = sl.shapes.add_table(len(rows), 6, Inches(0.5), Inches(5.5), Inches(W - 1.0), Inches(0.24 * len(rows))).table
for i, r in enumerate(rows):
    tb.rows[i].height = Inches(0.24)
    for j, v in enumerate(r):
        cell = tb.cell(i, j); cell.text = v; p = cell.text_frame.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
        f = p.runs[0].font; f.name = FONT; f.size = Pt(9.5); f.bold = (i == 0); f.color.rgb = RGBColor(255, 255, 255) if i == 0 else INK
        cell.fill.solid(); cell.fill.fore_color.rgb = NAVY if i == 0 else (RGBColor(0xFC, 0xE4, 0xD6) if r[0].startswith('Θ') else RGBColor(255, 255, 255))
        cell.margin_top = cell.margin_bottom = Inches(0.01); cell.margin_left = cell.margin_right = Inches(0.03)
text(sl, 0.5 + pic.width / 914400 + 0.3, 1.4, W - 1.3 - pic.width / 914400, 4.0, [
    [('읽는 법', {'bold': True, 'color': NAVY})],
    '칸마다 흔들림(약 ±0.024px)은 사진 두 장의 측정 잡음과 같은 크기. 로봇은 이보다 고르게 움직인다고 볼 수 있음',
    '첫 칸(주황)만 덜 감: 방향 바꾼 직후 백래시',
    'Θ는 + 방향 80.28px, - 방향 80.76px. 방향별 계수를 쓰는 게 안전.',
    '공식의 기준값은 이 데이터에서 구한 것이라, "정확히 0.1mm"인지는 카메라 캘리브레이션이 있어야 확인 가능.'], size=11, fill=BOX)

prs.save(DST); print('저장:', DST, '슬라이드', len(prs.slides))
