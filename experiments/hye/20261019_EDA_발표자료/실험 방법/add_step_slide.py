# '② 이동 관계' 다음에 '③ 한 칸 이동량' 슬라이드를 추가 (같은 스타일: 머리말, 제목, 왼쪽 그림, 오른쪽 항목/값/가는 선)
#   사용: python add_step_slide.py 입력.pptx 출력.pptx     숫자는 step_analysis.py 결과(img/step_px.csv)에서 계산
import copy, sys, pathlib, pandas as pd
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR
from pptx.oxml.ns import qn
HERE = pathlib.Path(__file__).parent
F, OUT = HERE / sys.argv[1], HERE / sys.argv[2]
SUB, LINE, FAINT = RGBColor.from_string('4B5563'), RGBColor.from_string('E5E7EB'), RGBColor.from_string('9CA3AF')
FONT, E = 'Noto Sans KR', 914400
KX, KW, VX, VW, LX, LW = 7.21, 1.3, 8.51, 4.03, 7.21, 5.33
TOP, BOTTOM, LH = 2.03, 6.43, 0.2

S = pd.read_csv(HERE / 'img' / 'step_px.csv'); S['axis'] = S.trial.str[0]
st = S[S.kind == 'step']; g = st.groupby('axis').px
mY, sY, nY = g.mean()['Y'], g.std()['Y'], g.size()['Y']; mT, sT, nT = g.mean()['Θ'], g.std()['Θ'], g.size()['Θ']
tr = st.groupby('trial').px.mean()
nm = lambda f: ('Θ ' if f[0] == 'T' else 'Y ') + f[1:]
out = st[((st.axis == 'Y') & ((st.px - mY).abs() >= 3 * sY)) | ((st.axis == 'Θ') & ((st.px - mT).abs() >= 3 * sT))]
first = S[S.kind == 'first'].set_index('step_to').px
ROWS = [
    ('Y 한 칸', [f'평균 {mY:.3f}px  (σ {sY:.3f}, {nY}칸)', f'0.1mm당, 1mm = {mY * 10:.2f}px']),
    ('Θ 한 칸', [f'평균 {mT:.3f}px  (σ {sT:.3f}, {nT}칸)', f'0.01당, Θ 1 = {mT * 100:.1f}px']),
    ('trial별 평균', [f'Y {tr.filter(like="Y").min():.3f} ~ {tr.filter(like="Y").max():.3f}px', f'Θ {tr.filter(like="Θ").min():.3f} ~ {tr.filter(like="Θ").max():.3f}px  (trial끼리 차이 거의 없음)']),
    ('평균 ±3σ 밖', [f'{nm(r.step_from)} → {r.step_to[1:]}:  {r.px:.2f}px' for _, r in out.sort_values(['axis', 'step_to'], ascending=[False, True]).iterrows()]),
    ('첫 칸', [f'Y +0.10 {first["Y+0.10"]:.2f}, Y -0.10 {first["Y-0.10"]:.2f}px',
                         f'Y +0.05 {first["Y+0.05"]:.2f}, Y -0.05 {first["Y-0.05"]:.2f}px',
                         f'Θ +0.01 {first["T+0.01"]:.2f}, Θ -0.01 {first["T-0.01"]:.2f}px',
                         'center → 첫 사진. 0점 어긋남으로 크게 다름']),
]
NOTES = (f'명령값이 한 칸 바뀔 때 QR이 몇 픽셀 움직이는지 같은 trial 안의 이웃 사진끼리 쟀습니다. Y 한 칸(0.1mm)은 평균 {mY:.3f}픽셀, '
         f'Θ 한 칸(0.01)은 평균 {mT:.3f}픽셀이고, trial끼리 차이는 거의 없습니다. 평균에서 3시그마 넘게 벗어난 칸은 {len(out)}개입니다. '
         'Y -1.85에서 -1.95 칸은 그 trial의 첫 촬영(-1.95부터 0 쪽으로 찍음)이라 방향이 바뀐 직후입니다. '
         'center에서 첫 사진으로 가는 칸은 0점 어긋남 때문에 평균과 크게 다릅니다.')

prs = Presentation(F)
slides = list(prs.slides)
base = next(sl for sl in slides if any(sh.has_text_frame and '이동 관계' in sh.text_frame.text for sh in sl.shapes))
base_idx = slides.index(base)
s = prs.slides.add_slide(base.slide_layout)
for sh in list(s.placeholders): sh._element.getparent().remove(sh._element)
for sh in base.shapes:   # 머리말, 제목, 쪽 번호 복사
    if sh.has_text_frame and (sh.text_frame.text.startswith('1-1') or '이동 관계' in sh.text_frame.text) or (sh.is_placeholder and 'Slide Number' in sh.name):
        s.shapes._spTree.append(copy.deepcopy(sh._element))
t = next(sh for sh in s.shapes if sh.has_text_frame and '이동 관계' in sh.text_frame.text)
r0 = t.text_frame.paragraphs[0].runs
r0[0].text = '분석에 필요한 정보: ③ 한 칸 이동량'
for r in r0[1:]: r.text = ''

def text(x, y, w, h, lines, size, color=SUB):
    tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h)); tf = tb.text_frame; tf.word_wrap = True
    for m in ('margin_left', 'margin_right', 'margin_top', 'margin_bottom'): setattr(tf, m, 0)
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        r = p.add_run(); r.text = line; r.font.size = Pt(size); r.font.color.rgb = color; r.font.name = FONT
        rpr = r._r.get_or_add_rPr(); ea = rpr.find(qn('a:ea'))
        if ea is None: ea = rpr.makeelement(qn('a:ea'), {}); rpr.append(ea)
        ea.set('typeface', FONT)

s.shapes.add_picture(str(HERE / 'img' / 'step_px.png'), Inches(0.8), Inches(TOP), Inches(5.78), Inches(5.78 * 4.4 / 5.8))
text(0.8, TOP + 5.78 * 4.4 / 5.8 + 0.05, 5.78, 0.25, ['같은 trial 안 이웃 사진끼리 QR 3개 가로 이동 평균의 차이 (ECC로 잰 값)'], 9, FAINT)
pad = (BOTTOM - TOP - sum(len(v) for _, v in ROWS) * LH) / len(ROWS)
y = TOP
for k, vals in ROWS:
    text(KX, y + pad / 2 - 0.02, KW, 0.3, [k], 12)
    text(VX, y + pad / 2, VW, len(vals) * LH, vals, 11)
    y += len(vals) * LH + pad
    ln = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(LX), Inches(y), Inches(LX + LW), Inches(y))
    ln.line.color.rgb = LINE; ln.line.width = Pt(0.75)
ns = s.notes_slide
if ns.notes_text_frame is None:
    body = next(sh for sh in base.notes_slide.shapes if sh.is_placeholder and sh.placeholder_format.type == 2)
    ns.shapes._spTree.append(copy.deepcopy(body._element))
ns.notes_text_frame.text = NOTES
lst = prs.slides._sldIdLst; el = list(lst)[-1]; lst.remove(el); lst.insert(base_idx + 1, el)
prs.save(OUT)
print('saved', OUT.name, len(prs.slides), 'slides')
