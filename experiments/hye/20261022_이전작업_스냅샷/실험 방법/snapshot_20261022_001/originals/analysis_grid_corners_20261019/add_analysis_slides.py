# 덱 맨 뒤에 모서리 분석 3장 추가: (1) 한 칸 이동량 (모서리 기반, ECC와 비교) (2) trial별 점별 흔들림 (3) 흔들림 지도
#   사용: python add_analysis_slides.py 입력.pptx 출력.pptx   (presentation_eda_20261019 폴더 기준)
#   숫자는 corner_steps.py 결과(out/corner_step_px.csv, corner_point_sigma.csv, photo_common_sigma.csv)에서 계산
import copy, sys, pathlib, numpy as np, pandas as pd
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR
from pptx.oxml.ns import qn
import os
RULE = os.environ.get('RULE', 'valid')
USE = '통과한' if RULE == 'valid' else '대각선 모서리를 뺀'          # 사용 점 설명
TAG = '' if RULE == 'valid' else ' (대각선 모서리만 제외)'         # 제목 꼬리표
HERE = pathlib.Path(__file__).parent; OUT_DIR = HERE / 'out' / ('' if RULE == 'valid' else RULE); DECK = HERE.parent / 'presentation_eda_20261019'
F, OUT = DECK / sys.argv[1], DECK / sys.argv[2]
SUB, LINE, FAINT, INK = (RGBColor.from_string(c) for c in ['4B5563', 'E5E7EB', '9CA3AF', '111827'])
FONT = 'Noto Sans KR'
KX, KW, VX, VW, LX, LW = 7.21, 1.3, 8.51, 4.03, 7.21, 5.33   # 오른쪽 '항목 / 값 / 가는 선' 표 (기초 정보와 같은 위치)
TOP, BOTTOM, LH = 2.03, 6.43, 0.2

prs = Presentation(F)
base = next(sl for sl in prs.slides if any(sh.has_text_frame and '이동 관계' in sh.text_frame.text for sh in sl.shapes))

def new_slide(title):
    s = prs.slides.add_slide(base.slide_layout)
    for sh in list(s.placeholders): sh._element.getparent().remove(sh._element)
    for sh in base.shapes:
        if sh.has_text_frame and (sh.text_frame.text.startswith('1-1') or '이동 관계' in sh.text_frame.text) or (sh.is_placeholder and 'Slide Number' in sh.name):
            s.shapes._spTree.append(copy.deepcopy(sh._element))
    t = next(sh for sh in s.shapes if sh.has_text_frame and '이동 관계' in sh.text_frame.text)
    rr = t.text_frame.paragraphs[0].runs; rr[0].text = title
    for r in rr[1:]: r.text = ''
    return s

def text(s, x, y, w, h, lines, size, color=SUB):
    tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h)); tf = tb.text_frame; tf.word_wrap = True
    for m in ('margin_left', 'margin_right', 'margin_top', 'margin_bottom'): setattr(tf, m, 0)
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        r = p.add_run(); r.text = line; r.font.size = Pt(size); r.font.color.rgb = color; r.font.name = FONT
        rpr = r._r.get_or_add_rPr(); ea = rpr.find(qn('a:ea'))
        if ea is None: ea = rpr.makeelement(qn('a:ea'), {}); rpr.append(ea)
        ea.set('typeface', FONT)

def rows_table(s, rows, top=TOP, bottom=BOTTOM):
    pad = (bottom - top - sum(len(v) for _, v in rows) * LH) / len(rows); y = top
    for k, vals in rows:
        text(s, KX, y + pad / 2 - 0.02, KW, 0.3, [k], 12)
        text(s, VX, y + pad / 2, VW, len(vals) * LH, vals, 11)
        y += len(vals) * LH + pad
        ln = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(LX), Inches(y), Inches(LX + LW), Inches(y))
        ln.line.color.rgb = LINE; ln.line.width = Pt(0.75)

def notes(s, t):
    ns = s.notes_slide
    if ns.notes_text_frame is None:
        body = next(sh for sh in base.notes_slide.shapes if sh.is_placeholder and sh.placeholder_format.type == 2)
        ns.shapes._spTree.append(copy.deepcopy(body._element))
    ns.notes_text_frame.text = t

# ---- 숫자
S = pd.read_csv(OUT_DIR / 'corner_step_px.csv'); st = S[S.kind == 'step']
E = pd.read_csv(DECK / 'img' / 'step_px.csv'); Es = E[E.kind == 'step']; Es = Es.assign(axis=np.where(Es.trial.str.startswith('Θ'), 'Θ', 'Y'))
g, ge = st.groupby('axis').px, Es.groupby('axis').px
cmp = S.merge(E[['step_from', 'step_to', 'px']], on=['step_from', 'step_to'], suffixes=('', '_ecc')); dd = cmp.px - cmp.px_ecc
lim = st.groupby('axis').px.agg(['mean', 'std']); st = st.join(lim, on='axis')
out = st[(st.px - st['mean']).abs() >= 3 * st['std']]
El = Es.join(Es.groupby('axis').px.agg(['mean', 'std']), on='axis'); n_ecc = int(((El.px - El['mean']).abs() >= 3 * El['std']).sum())
first = S[S.kind == 'first'].set_index('step_to').px
nm = lambda f: ('Θ ' if f[0] == 'T' else 'Y ') + f[1:]
R = pd.read_csv(OUT_DIR / 'corner_point_sigma.csv'); PH = pd.read_csv(OUT_DIR / 'photo_common_sigma.csv')
med = R.groupby(['comp', 'trial']).sigma_point.median()
cx = PH[PH.comp == 'dx'].set_index('trial').sigma_common

# ---- (1) 한 칸 이동량 (모서리)
s = new_slide('모서리 기반 한 칸 이동량' + TAG)
pw = 5.78; ph = pw * 4.4 / 5.8
s.shapes.add_picture(str(OUT_DIR / 'corner_step_px.png'), Inches(0.8), Inches(TOP), Inches(pw), Inches(ph))
text(s, 0.8, TOP + ph + 0.05, pw, 0.25, [f'같은 trial 안 이웃 사진, 두 사진 모두에서 {USE} 모서리의 가로 이동 차이 평균'], 9, FAINT)
rows_table(s, [
    ('Y 한 칸', [f'평균 {g.mean()["Y"]:.3f}px  (σ {g.std()["Y"]:.3f}, {g.size()["Y"]}칸)', f'ECC: {ge.mean()["Y"]:.3f}px  (σ {ge.std()["Y"]:.3f})']),
    ('Θ 한 칸', [f'평균 {g.mean()["Θ"]:.3f}px  (σ {g.std()["Θ"]:.3f}, {g.size()["Θ"]}칸)', f'ECC: {ge.mean()["Θ"]:.3f}px  (σ {ge.std()["Θ"]:.3f})']),
    ('ECC와 차이', [f'칸마다 평균 {dd.mean():+.3f}px, σ {dd.std():.3f}px, 최대 {dd.abs().max():.3f}px', '두 방법이 같은 이동을 재고 있음']),
    ('평균 ±3σ 밖', [f'{nm(r.step_from)} → {r.step_to[1:]}:  {r.px:.2f}px' for _, r in out.iterrows()] + [f'(ECC {n_ecc}칸 → 모서리 {len(out)}칸)']),
    ('첫 칸', [f'Y +0.10 {first["Y+0.10"]:.2f}, Y -0.10 {first["Y-0.10"]:.2f}px', f'Y +0.05 {first["Y+0.05"]:.2f}, Y -0.05 {first["Y-0.05"]:.2f}px',
               f'Θ +0.01 {first["T+0.01"]:.2f}, Θ -0.01 {first["T-0.01"]:.2f}px', 'center → 첫 사진. ECC와 같이 0점 영향 큼']),
])
notes(s, f'QR 격자 모서리 중 두 사진 모두 통과한 점으로 한 칸 이동량을 다시 쟀습니다. Y 한 칸은 {g.mean()["Y"]:.3f}픽셀, Θ 한 칸은 {g.mean()["Θ"]:.3f}픽셀로 ECC 결과와 같고, '
         f'흔들림은 조금 작습니다. 3시그마 밖 칸은 ECC 때 4칸에서 2칸으로 줄었고, Y -1.95는 방향을 바꾼 직후 촬영이라 여전히 남습니다.')

# ---- (2) trial별 점별 흔들림
s = new_slide('trial별 점별 흔들림' + TAG)
pw = 6.1; ph = pw * 3.1 / 7.2
s.shapes.add_picture(str(OUT_DIR / 'noise_box.png'), Inches(0.75), Inches(2.3), Inches(pw), Inches(ph))
text(s, 0.8, 2.3 + ph + 0.08, pw, 0.45, ['상자 = 점들의 25~75%, 선 = 5~95%, 숫자 = 중앙값.', '마름모 = 사진 공통 흔들림 (그 사진에서 모든 점이 같이 흔들린 양)'], 9, FAINT)
yt = [t for t in med['dx'].index if t.startswith('Y')]; tt = [t for t in med['dx'].index if t.startswith('Θ')]
rows_table(s, [
    ('재는 방법', [f'trial의 모든 사진에서 {USE} 점마다 이동량을', '명령값에 2차 곡선으로 맞추고 남은 값의 σ',
                 '남은 값 = 사진 공통 + 점 고유로 나눔']),
    ('점 고유 가로', [f'Y {min(med["dx"][yt]):.3f} ~ {max(med["dx"][yt]):.3f}px,  Θ {min(med["dx"][tt]):.3f} ~ {max(med["dx"][tt]):.3f}px', '(점들의 중앙값, trial끼리 거의 같음)']),
    ('점 고유 세로', [f'Y {min(med["dy"][yt]):.3f} ~ {max(med["dy"][yt]):.3f}px,  Θ {min(med["dy"][tt]):.3f} ~ {max(med["dy"][tt]):.3f}px']),
    ('사진 공통 가로', [f'Y 0.1 간격, Y 0.05 홀수 +쪽: {cx[["Y 0.1 간격 +", "Y 0.1 간격 -", "Y 0.05 홀수 +"]].min():.3f} ~ {cx[["Y 0.1 간격 +", "Y 0.1 간격 -", "Y 0.05 홀수 +"]].max():.3f}px',
                    f'Y 0.05 홀수 -쪽: {cx["Y 0.05 홀수 -"]:.3f}px  (방향 바뀐 첫 촬영 포함)', f'Θ +쪽 {cx["Θ +"]:.3f}, -쪽 {cx["Θ -"]:.3f}px']),
    ('의미', ['점 하나는 약 0.05px 흔들리지만, 점 수백 개를 평균하면', '거의 사라짐. 남는 건 사진 공통 흔들림 (0.014~0.032px)']),
])
notes(s, '점마다 trial 안에서 이동량을 명령값 곡선에 맞추고 남은 흔들림을 쟀습니다. 이걸 사진 전체가 같이 흔들린 부분과 점 하나만 흔들린 부분으로 나눴습니다. '
         '점 고유 흔들림은 가로 약 0.046픽셀, 세로 약 0.033픽셀로 trial끼리 거의 같습니다. 점을 많이 평균하면 이 부분은 줄어들고, 사진 공통 흔들림이 남습니다. 사진 공통 흔들림은 Θ trial에서 Y보다 두 배 정도 큽니다.')

# ---- (3) 흔들림 지도
s = new_slide('점 고유 흔들림 지도' + (TAG or ' (가로)'))
mw = 5.7; mh = mw * 654 / 2151
for j, (img, lab) in enumerate([('noise_map_Y.png', 'Y trial 4개 평균'), ('noise_map_T.png', 'Θ trial 2개 평균')]):
    x = 0.8 + j * (mw + 0.33)
    text(s, x, 2.15, mw, 0.3, [lab], 12, INK)
    s.shapes.add_picture(str(OUT_DIR / img), Inches(x), Inches(2.5), Inches(mw), Inches(mh))
s.shapes.add_picture(str(OUT_DIR / 'noise_colorbar.png'), Inches(0.8), Inches(2.5 + mh + 0.25), Inches(3.2), Inches(0.42))
x = R[R.comp == 'dx']; D0 = pd.read_csv(HERE / 'out' / 'corners_280.csv'); C0 = D0[D0.file == 'center.bmp'].set_index('id')
gx, gy = x.id.map(C0.grid_x), x.id.map(C0.grid_y)
fin = ((gx <= 7) & (gy <= 7)) | ((gx >= 18) & (gy <= 7)) | ((gx <= 7) & (gy >= 18))
qm = x.groupby('qr').sigma_point.median()
text(s, 4.3, 2.5 + mh + 0.25, 8.2, 1.2, [
    '진할수록 그 점의 가로 흔들림이 큼. ' + ('주황(불확실) 점은 빠짐.' if RULE == 'valid' else '대각선 모서리 125점은 빠짐.'),
    f'QR별 중앙값: 왼쪽 {qm[1]:.3f}, 가운데 {qm[2]:.3f}, 오른쪽 {qm[3]:.3f}px',
    f'큰 위치 네모(finder) 모서리 {x[fin].sigma_point.median():.3f}px,  나머지 {x[~fin].sigma_point.median():.3f}px',
    '뚜렷하게 몰린 곳은 없고, 왼쪽 QR과 finder 모서리가 조금 큼'], 11)
notes(s, '점마다 가로 흔들림을 지도에 그렸습니다. 진할수록 흔들림이 큽니다. 뚜렷하게 몰린 영역은 없고, 왼쪽 QR과 큰 위치 네모 모서리가 조금 큰 편입니다.')

prs.save(OUT)
print('saved', OUT.name, len(prs.slides), 'slides')
