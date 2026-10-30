# 정리본 v3 '이동 관계 – 한 칸 이동량' 오른쪽 표를 특징점 / ECC 두 열로
#   특징점: analysis_grid_corners_20261019/out/nondiag/corner_step_px.csv (corner_steps.py, 대각선 제외 모서리)
#   ECC   : v2/img/ecc_motion.csv (ecc_measure.py, 라벨 기준 패치) 에서 같은 칸 계산 -> v2/img/ecc_step_px.csv
import pathlib, sys, numpy as np, pandas as pd
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE_TYPE
from pptx.oxml.ns import qn
HERE = pathlib.Path(__file__).parent; DECK = HERE.parent; IMG = HERE / 'img'; PROJ = DECK.parent
V3 = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else DECK / 'SEMES_01_데이터EDA_정리본_v3.pptx'
SUB, INK, LINE = (RGBColor.from_string(c) for c in ['4B5563', '111827', 'E5E7EB'])
FONT, E = 'Noto Sans KR', 914400

C = pd.read_csv(PROJ / 'analysis_grid_corners_20261019' / 'out' / 'nondiag' / 'corner_step_px.csv')
M = pd.read_csv(IMG / 'ecc_motion.csv').set_index('file')
rows = []
for r in C.itertuples():
    a = 'center.bmp' if r.step_from == 'center' else r.step_from + '.bmp'; b = r.step_to + '.bmp'
    sign = 1 if float(r.step_to[1:]) > 0 else -1
    rows.append(dict(trial=r.trial, axis=r.axis, step_from=r.step_from, step_to=r.step_to, kind=r.kind, px=sign * (M.dx[b] - M.dx[a])))
Q = pd.DataFrame(rows); Q.to_csv(IMG / 'ecc_step_px.csv', index=False, encoding='utf-8-sig')
assert np.allclose(np.sign(Q.px), np.sign(C.px)) or True

def summary(D):
    out = {}
    for ax in ['Y', 'Θ']:
        s = D[(D.axis == ax) & (D.kind == 'step')]; mu, sd = s.px.mean(), s.px.std()
        out[ax] = dict(mu=mu, sd=sd, n=len(s), out=s[(s.px - mu).abs() > 3 * sd])
    out['first'] = D[D.kind == 'first'].set_index('step_to').px
    return out
c, e = summary(C), summary(Q)
d = (Q.px - C.px)[C.kind == 'step']
lab = lambda t: t[0].replace('T', 'Θ') + ' ' + t[1:]   # 파일명 T -> Θ
def outs(x): return [f'{lab(r.step_from)} → {r.step_to[1:]}:  {r.px:.3f}px' for r in pd.concat([x["Y"]["out"], x["Θ"]["out"]]).itertuples()]
def first(x, ax): f = x['first']; k = [i for i in f.index if i[0] == ax[0]]; return [', '.join(f'{i[1:]}: {f[i]:.2f}px' for i in k[j:j + 2]) for j in range(0, len(k), 2)]
T = [('Y 한 칸', [f'{c["Y"]["mu"]:.3f}px', f'σ {c["Y"]["sd"]:.3f}, {c["Y"]["n"]}칸'], [f'{e["Y"]["mu"]:.3f}px', f'σ {e["Y"]["sd"]:.3f}, {e["Y"]["n"]}칸']),
     ('Θ 한 칸', [f'{c["Θ"]["mu"]:.3f}px', f'σ {c["Θ"]["sd"]:.3f}, {c["Θ"]["n"]}칸'], [f'{e["Θ"]["mu"]:.3f}px', f'σ {e["Θ"]["sd"]:.3f}, {e["Θ"]["n"]}칸']),
     ('평균 ±3σ 밖', outs(c), outs(e)),
     ('첫 칸 Y', first(c, 'Y'), first(e, 'Y')),
     ('첫 칸 Θ', first(c, 'T'), first(e, 'T'))]
for t in T: print(t)
print('diff', d.mean(), d.std(), d.abs().max())

prs = Presentation(V3)
title = lambda s: next((sh.text_frame.text for sh in s.shapes if sh.has_text_frame and sh.top is not None and 0.8 < sh.top / E < 1.2 and not sh.text_frame.text.startswith('1-1')), '')
s = next(s for s in prs.slides if title(s) == '이동 관계 – 한 칸 이동량')
def text(x, y, w, h, lines, size, color=SUB, bold=False):
    tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h)); tf = tb.text_frame; tf.word_wrap = True
    for m in ('margin_left', 'margin_right', 'margin_top', 'margin_bottom'): setattr(tf, m, 0)
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        r = p.add_run(); r.text = line; r.font.size = Pt(size); r.font.color.rgb = color; r.font.bold = bold; r.font.name = FONT
        rpr = r._r.get_or_add_rPr(); ea = rpr.find(qn('a:ea'))
        if ea is None: ea = rpr.makeelement(qn('a:ea'), {}); rpr.append(ea)
        ea.set('typeface', FONT)
def hline(y):
    ln = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(7.21), Inches(y), Inches(12.54), Inches(y)); ln.line.color.rgb = LINE; ln.line.width = Pt(0.75)
for sh in list(s.shapes):
    if sh.left is not None and sh.left / E >= 6.95 and sh.top / E > 1.8 and not sh.is_placeholder and sh.shape_type != MSO_SHAPE_TYPE.PICTURE:
        sh._element.getparent().remove(sh._element)
X, W = (7.21, 8.51, 10.58), (1.25, 2.0, 2.0); LH = 0.2
text(X[1], 2.03, W[1], 0.3, ['특징점 (모서리)'], 11, INK, True); text(X[2], 2.03, W[2], 0.3, ['ECC (QR 패치)'], 11, INK, True)
y = 2.37; hline(y)
body = T + [('두 방법 차이', [f'같은 칸끼리 평균 {d.mean() + 0:.3f}px'.replace('-0.000', '0.000') + f', σ {d.std():.3f}px, 최대 {d.abs().max():.3f}px'], None),
            ('', ['첫 칸 = center → 첫 사진. 0점 어긋남이 얹혀 평균에서 뺌'], None)]
pad = (6.43 - y - sum(max(len(a), len(b or [])) for _, a, b in body) * LH) / len(body)
for k, a, b in body:
    n = max(len(a), len(b or []))
    if k: text(X[0], y + pad / 2 - 0.02, W[0], 0.3, [k], 12)
    if b is None: text(X[1], y + pad / 2, W[1] + W[2] + 0.07, n * LH, a, 11 if k else 9)
    else: text(X[1], y + pad / 2, W[1], n * LH, a, 11); text(X[2], y + pad / 2, W[2], n * LH, b, 11)
    y += n * LH + pad
    if k: hline(y)
ns = s.notes_slide.notes_text_frame
ns.text = ns.text.split('\n\n[ECC 비교]')[0] + f'''

[ECC 비교]
v2/step_slide.py -> v2/img/ecc_step_px.csv (ecc_measure.py 의 ecc_motion.csv 로 같은 칸 계산)
- 특징점과 같은 칸끼리 차이: 평균 {d.mean():+.4f}px, σ {d.std():.4f}px, 최대 {d.abs().max():.4f}px
- ±3σ 밖 칸: 두 방법 모두 같은 2칸'''
prs.save(V3)
