# 덱 맨 뒤에 '측정 방법별 Y/Θ 판정 정확도' 1장 추가 (4열 비교표와 같은 열: 모서리 755 / 대각선 제외 630 / 통과 점만 / ECC)
#   표는 글상자와 가는 선으로 그림. 숫자는 summary.csv (run_compare.py 결과)
#   사용: python add_judge_slide.py 입력.pptx 출력.pptx   (presentation_eda_20261019 폴더 기준, 입력 파일은 읽기만 함)
import copy, sys, pathlib, pandas as pd
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
S = pd.read_csv(HERE / 'summary.csv').set_index('method')
COLS = [('corner_all', '모서리 전부\n755점'), ('corner_nondiag', '대각선 모서리 제외\n630점'), ('corner_valid', '통과한 점만\n(점 그대로: 281장 모두 통과 258점)'), ('ecc', 'ECC\nQR 3개 아핀 18값')]
pc = lambda v: f'{v * 100:.1f}%'

def key(col, kind):
    if col == 'ecc': return 'ecc_affine18' if kind == 'P12' else None
    return f'{col}_{kind}'

def cell(col, kind, what):
    k = key(col, kind)
    if k is None: return ['-']
    r = S.loc[k]
    if what == 'resub': return [f'Y {pc(r.resub_Y)},  Θ {pc(r.resub_T)}'] + (['(학습 사진에 과적합)'] if kind == 'direct' else [])
    if what == 'evenodd': return [f'Y {pc(r.evenodd_Y)},  Θ {pc(r.evenodd_T)}']
    if what == 'rep': return [f'Y {pc(r.rep_Y)},  Θ {pc(r.rep_T)}', f'회당 {r.rep_wrong_per_round:.1f}장 틀림']
    if what == 'max': return [f'Y {r.rep_Ymax:.2f},  Θ {r.rep_Tmax:.2f}']

ROWS = [('요약 12계수 (P12)', None, None),
        ('재대입', 'P12', 'resub'), ('짝홀 교차검증', 'P12', 'evenodd'), ('반복 5분할 20회', 'P12', 'rep'), ('최대오차 (반복, 오분류 포함)', 'P12', 'max'),
        ('점 이동 그대로', None, None),
        ('재대입', 'direct', 'resub'), ('짝홀 교차검증', 'direct', 'evenodd'), ('반복 5분할 20회', 'direct', 'rep'), ('최대오차 (반복, 오분류 포함)', 'direct', 'max')]

prs = Presentation(F)
base = next(sl for sl in prs.slides if any(sh.has_text_frame and '이동 관계' in sh.text_frame.text for sh in sl.shapes))
s = prs.slides.add_slide(base.slide_layout)
for sh in list(s.placeholders): sh._element.getparent().remove(sh._element)
for sh in base.shapes:
    if sh.has_text_frame and (sh.text_frame.text.startswith('1-1') or '이동 관계' in sh.text_frame.text) or (sh.is_placeholder and 'Slide Number' in sh.name):
        s.shapes._spTree.append(copy.deepcopy(sh._element))
t = next(sh for sh in s.shapes if sh.has_text_frame and '이동 관계' in sh.text_frame.text)
rr = t.text_frame.paragraphs[0].runs; rr[0].text = '측정 방법별 Y/Θ 판정 정확도 (같은 판정 방법)'
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

X0, KW, CW, GAP = 0.8, 2.25, 2.37, 0.05; WTOT = KW + 4 * CW
def hline(y, color=LINE, w=0.75):
    ln = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(X0), Inches(y), Inches(X0 + WTOT), Inches(y)); ln.line.color.rgb = color; ln.line.width = Pt(w)

y = 1.8; HH = 0.5
for j, (_, h) in enumerate(COLS): text(X0 + KW + j * CW, y, CW - GAP, HH, h.split('\n'), 10.5 if j != 2 else 10.5, INK, True)
y += HH; hline(y, INK, 1)
LH, PAD = 0.175, 0.07
for lab, kind, what in ROWS:
    if kind is None:
        text(X0, y + 0.04, WTOT, 0.26, [lab], 10.5, INK, True); y += 0.3; continue
    cells = [cell(c, kind, what) for c, _ in COLS]; n = max(len(c) for c in cells); h = n * LH + 2 * PAD
    text(X0 + 0.15, y, KW - 0.2, h, [lab], 10.5)
    for j, c in enumerate(cells): text(X0 + KW + j * CW, y, CW - GAP, h, c, 10)
    y += h; hline(y)
r = S.loc['ref205_P12']
text(X0, y + 0.1, WTOT, 0.75, [
    f'기준: 기존 205점 + 12계수 = 반복 5분할 Y {pc(r.rep_Y)}, Θ {pc(r.rep_T)}, 회당 {r.rep_wrong_per_round:.1f}장 틀림.  어느 방법도 20회 중 한 번도 280장을 모두 맞히지 못함.',
    '점 그대로 쓰면 특징 수(516~1510)가 학습 사진 수보다 많아 재대입은 100%지만 교차검증은 다른 방법과 비슷함. 점끼리 상관을 버린 대각 공분산은 Θ 68%로 나빠짐 (참고).',
    '판정: center 고정, 원점 통과 Y/T 2차 곡선, 학습 잔차 공분산으로 Y만/Θ만 비교 (center_fixed_20261015 와 같은 코드, 같은 저장 분할).'], 9, FAINT, mid=False)
ns = s.notes_slide
if ns.notes_text_frame is None:
    body = next(sh for sh in base.notes_slide.shapes if sh.is_placeholder and sh.placeholder_format.type == 2)
    ns.shapes._spTree.append(copy.deepcopy(body._element))
b = S.loc['corner_nondiag_P12']; e = S.loc['ecc_affine18']
ns.notes_text_frame.text = (f'같은 판정 방법에 측정값만 바꿔 넣었습니다. 교차검증으로 보면 대각선 모서리를 뺀 630점을 12계수로 요약한 것이 회당 {b.rep_wrong_per_round:.1f}장 틀려 가장 적고, '
                            f'ECC가 {e.rep_wrong_per_round:.1f}장, 기존 205점이 {r.rep_wrong_per_round:.1f}장으로 차이는 크지 않습니다. 모서리를 전부 넣고 12계수로 요약하면 대각선 모서리의 잡음 때문에 오히려 나빠집니다. '
                            '점을 그대로 쓰면 학습 사진에는 완벽히 맞지만 새 사진에서는 비슷한 수준이라, 측정 방법보다 촬영 자체의 흔들림이 한계입니다.')
prs.save(OUT)
print('saved', OUT.name, len(prs.slides), 'slides')
