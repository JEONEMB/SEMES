# 덱 맨 뒤에 '4열 비교' 1장 추가: 모서리 전부 755점 / 대각선 제외 630점 / 통과 점만 / ECC
#   표는 PowerPoint 표 개체가 아니라 글상자와 가는 선으로 그림 (기초 정보 표와 같은 스타일)
#   숫자는 out/compare_4.csv (corner_steps.py 세 가지 규칙 + ecc_noise.py 결과를 모은 표)
#   사용: python add_compare_slide.py 입력.pptx 출력.pptx   (presentation_eda_20261019 폴더 기준)
import copy, re, sys, pathlib, pandas as pd
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR
from pptx.enum.text import MSO_ANCHOR
from pptx.oxml.ns import qn
HERE = pathlib.Path(__file__).parent; DECK = HERE.parent / 'presentation_eda_20261019'
F, OUT = DECK / sys.argv[1], DECK / sys.argv[2]
SUB, LINE, FAINT, INK = (RGBColor.from_string(c) for c in ['4B5563', 'E5E7EB', '9CA3AF', '111827'])
FONT = 'Noto Sans KR'
T = pd.read_csv(HERE / 'out' / 'compare_4.csv', index_col=0)
COLS = ['전부 755점', '대각선 제외 630점', '통과 점만', 'ECC']
HEAD = ['모서리 전부\n755점', '대각선 모서리 제외\n630점', '통과한 점만\n약 570점', 'ECC\nQR 3개 단위']

def cell(row, col):
    v = str(T.loc[row, col])
    if v == 'nan': return ['-']
    v = v.replace('T-', 'Θ -').replace('T+', 'Θ +').replace('Y-', 'Y -').replace('Y+', 'Y +').replace('Y.05-', 'Y 0.05 홀수 -')
    v = re.sub(r'(\d\.\d+)~\1(?!\d)', r'\1', v)   # 0.018~0.018 -> 0.018
    if row == '±3σ 밖':
        n, rest = v.split(': '); items = [x.strip() for x in rest.split(',')]
        common = ['Θ -0.34→-0.35', 'Y -1.85→-1.95']
        if col == 'ECC': return [f'{n} (왼쪽 2칸 +)', ', '.join(x for x in items if x not in common)]
        return [n, ', '.join(items)]
    if row == '사진 공통 가로':
        a, b, c = [x.strip() for x in v.split(',')]; return [f'{a},  {c}', b]
    if row in ('점 고유 가로', '점 고유 세로') and '(QR 단위)' in v: return [v.replace(' (QR 단위)', ''), '(QR 하나 단위)']
    if row == '한 칸 점들끼리 σ(중앙)': return [v + '  (Y / Θ)']
    return [v]

ROWS = [('Y 한 칸 (0.1mm)', 'Y 한 칸'), ('Θ 한 칸 (0.01)', 'Θ 한 칸'), ('평균 ±3σ 밖 칸', '±3σ 밖'), ('ECC와 칸마다 차이', 'ECC와 칸마다 차이'),
        ('점 고유 흔들림 가로', '점 고유 가로'), ('점 고유 흔들림 세로', '점 고유 세로'), ('사진 공통 흔들림 가로', '사진 공통 가로'),
        ('한 칸 안 점들끼리 차이', '한 칸 점들끼리 σ(중앙)')]

prs = Presentation(F)
base = next(sl for sl in prs.slides if any(sh.has_text_frame and '이동 관계' in sh.text_frame.text for sh in sl.shapes))
s = prs.slides.add_slide(base.slide_layout)
for sh in list(s.placeholders): sh._element.getparent().remove(sh._element)
for sh in base.shapes:
    if sh.has_text_frame and (sh.text_frame.text.startswith('1-1') or '이동 관계' in sh.text_frame.text) or (sh.is_placeholder and 'Slide Number' in sh.name):
        s.shapes._spTree.append(copy.deepcopy(sh._element))
t = next(sh for sh in s.shapes if sh.has_text_frame and '이동 관계' in sh.text_frame.text)
rr = t.text_frame.paragraphs[0].runs; rr[0].text = '측정 방법 4가지 비교 (단위 px)'
for r in rr[1:]: r.text = ''

def text(x, y, w, h, lines, size, color=SUB, bold=False, mid=True):
    tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h)); tf = tb.text_frame; tf.word_wrap = True
    if mid: tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    for m in ('margin_left', 'margin_right', 'margin_top', 'margin_bottom'): setattr(tf, m, 0)
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        r = p.add_run(); r.text = line; r.font.size = Pt(size); r.font.color.rgb = color; r.font.bold = bold; r.font.name = FONT
        rpr = r._r.get_or_add_rPr(); ea = rpr.find(qn('a:ea'))
        if ea is None: ea = rpr.makeelement(qn('a:ea'), {}); rpr.append(ea)
        ea.set('typeface', FONT)

def hline(y, color=LINE, w=0.75):
    ln = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(X0), Inches(y), Inches(X0 + WTOT), Inches(y))
    ln.line.color.rgb = color; ln.line.width = Pt(w)

X0, KW, CW, GAP = 0.8, 2.05, 2.37, 0.05
WTOT = KW + 4 * CW
y = 1.85; HH = 0.5
for j, h in enumerate(HEAD):
    text(X0 + KW + j * CW, y, CW - GAP, HH, h.split('\n'), 11, INK, True)
y += HH; hline(y, INK, 1)
LH = 0.18; PAD = 0.09
for lab, key in ROWS:
    cells = [cell(key, c) for c in COLS]; n = max(len(c) for c in cells); h = n * LH + 2 * PAD
    text(X0, y, KW - 0.1, h, [lab], 11)
    for j, c in enumerate(cells): text(X0 + KW + j * CW, y, CW - GAP, h, c, 10)
    y += h; hline(y)
text(X0, y + 0.12, WTOT, 0.45, ['한 칸 이동량은 네 방법이 같음. 대각선 모서리를 넣으면 점 하나의 흔들림과 점들끼리 차이만 커지고, 평균하면 사라짐.',
                                'ECC의 점 고유 흔들림은 QR 하나(픽셀 수백 개) 평균이라 모서리 한 점과 바로 비교할 수 없음.'], 9.5, FAINT, mid=False)
ns = s.notes_slide
if ns.notes_text_frame is None:
    body = next(sh for sh in base.notes_slide.shapes if sh.is_placeholder and sh.placeholder_format.type == 2)
    ns.shapes._spTree.append(copy.deepcopy(body._element))
ns.notes_text_frame.text = ('같은 280장을 네 가지 방법으로 쟀습니다. 모서리를 전부 쓴 경우, 대각선 모서리만 뺀 경우, 진단을 통과한 점만 쓴 경우, 그리고 기존 ECC입니다. '
                            '한 칸 이동량은 네 방법 모두 Y 1.436픽셀, Θ 0.804픽셀로 같습니다. 대각선 모서리를 넣으면 점 하나의 흔들림이 커지지만 평균하면 사라집니다. '
                            '사진 전체가 같이 흔들리는 양은 방법과 관계없이 0.015에서 0.035픽셀이라, 측정 방법보다 촬영 자체의 흔들림이 한계입니다.')
prs.save(OUT)
print('saved', OUT.name, len(prs.slides), 'slides')
