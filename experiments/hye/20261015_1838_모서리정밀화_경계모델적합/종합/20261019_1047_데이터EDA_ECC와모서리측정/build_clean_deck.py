# 정리본 덱: ① 기초 정보 → ② 분석에 필요한 정보 → ③ 노이즈 → ④ 정리와 다음 데이터 → 부록(측정 방법)
#   입력 (읽기만 함): SEMES_01_데이터EDA_판정비교.pptx (사용자가 고친 앞 5장 포함, 분석 장들)
#                    SEMES_01_데이터EDA.pptx (예전 EDA 장: 짝 찾기, 0점, 촬영 순서, 결과, 다음 데이터)
#   출력: SEMES_01_데이터EDA_정리본.pptx
#   새로 만드는 장: 노이즈 한눈에 보기, 부록 표지, 모서리 맞춤 전수 결과(6 trial 한 장), 기준점 오차 검증
import copy, io, pathlib, pandas as pd
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR
from pptx.enum.text import MSO_ANCHOR
from pptx.oxml.ns import qn
HERE = pathlib.Path(__file__).parent; CORN = HERE.parent / 'analysis_grid_corners_20261019' / 'out'
SRC, OLD, OUT = HERE / 'SEMES_01_데이터EDA_판정비교.pptx', HERE / 'SEMES_01_데이터EDA.pptx', HERE / 'SEMES_01_데이터EDA_정리본.pptx'
SUB, LINE, FAINT, INK = (RGBColor.from_string(c) for c in ['4B5563', 'E5E7EB', '9CA3AF', '111827'])
FONT, E = 'Noto Sans KR', 914400
SEC1, SEC2, SEC3, SEC4, APX = '① 기초 정보', '② 분석에 필요한 정보', '③ 노이즈', '④ 정리와 다음 데이터', '부록  측정 방법'

prs = Presentation(SRC); old = Presentation(OLD)
src = list(prs.slides); olds = list(old.slides)
layout = {l.name: l for l in prs.slide_layouts}

# ---------- 도우미
def import_slide(o):
    """예전 덱의 장을 복사 (그림은 이미지 파일까지 옮김)"""
    s = prs.slides.add_slide(layout[o.slide_layout.name])
    for sh in list(s.placeholders): sh._element.getparent().remove(sh._element)
    for sh in o.shapes:
        if sh.top is not None and sh.top / E < 0.55 and sh.height == 0: continue   # 예전 머리말 위 짧은 선 (사용자 수정본에는 없음)
        el = copy.deepcopy(sh._element)
        for blip in el.iter(qn('a:blip')):
            rid = blip.get(qn('r:embed')); blob = o.part.related_part(rid).blob
            _, nid = s.part.get_or_add_image_part(io.BytesIO(blob)); blip.set(qn('r:embed'), nid)
        s.shapes._spTree.append(el)
    if o.has_notes_slide and o.notes_slide.notes_text_frame is not None:
        notes(s, o.notes_slide.notes_text_frame.text)
    return s

TEMPLATE = src[3]   # '이동 관계' 장: 머리말, 제목, 쪽 번호를 복사해 새 장의 틀로 씀
def new_slide(title, gray=True):
    s = prs.slides.add_slide(TEMPLATE.slide_layout if gray else layout['CONTENT'])
    for sh in list(s.placeholders): sh._element.getparent().remove(sh._element)
    for sh in TEMPLATE.shapes:
        if sh.has_text_frame and (sh.text_frame.text.startswith('1-1') or '이동 관계' in sh.text_frame.text) or (sh.is_placeholder and 'Slide Number' in sh.name):
            s.shapes._spTree.append(copy.deepcopy(sh._element))
    set_title(s, title); return s

def title_shape(s):
    cands = [sh for sh in s.shapes if sh.has_text_frame and sh.text_frame.text.strip() and not sh.text_frame.text.startswith('1-1')
             and sh.top / E < 1.2 and sh.top / E > 0.8]
    return cands[0] if cands else None

def set_title(s, title):
    t = title_shape(s); p = t.text_frame.paragraphs[0]
    p.runs[0].text = title
    for r in p.runs[1:]: r.text = ''
    for extra in t.text_frame.paragraphs[1:]:
        for r in extra.runs: r.text = ''

def set_eyebrow(s, sec):
    for sh in s.shapes:
        if sh.has_text_frame and sh.text_frame.text.startswith('1-1'):
            rr = sh.text_frame.paragraphs[0].runs
            if len(rr) >= 2: rr[0].text = '1-1  '; rr[1].text = f'20261012  |  {sec}'; [setattr(r, 'text', '') for r in rr[2:]]
            else: rr[0].text = f'1-1  20261012  |  {sec}'
            for r in rr: r.font.bold = False   # 머리말 굵기를 사용자 수정본과 같게

def replace_text(s, old_t, new_t):
    n = 0
    for sh in s.shapes:
        if sh.has_text_frame:
            for p in sh.text_frame.paragraphs:
                for r in p.runs:
                    if old_t in r.text: r.text = r.text.replace(old_t, new_t); n += 1
    assert n, (old_t, 'not found')

def text(s, x, y, w, h, lines, size, color=SUB, bold=False, mid=False):
    tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h)); tf = tb.text_frame; tf.word_wrap = True
    if mid: tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    for m in ('margin_left', 'margin_right', 'margin_top', 'margin_bottom'): setattr(tf, m, 0)
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        r = p.add_run(); r.text = line; r.font.size = Pt(size); r.font.color.rgb = color; r.font.bold = bold; r.font.name = FONT
        rpr = r._r.get_or_add_rPr(); ea = rpr.find(qn('a:ea'))
        if ea is None: ea = rpr.makeelement(qn('a:ea'), {}); rpr.append(ea)
        ea.set('typeface', FONT)

def hline(s, x, y, w, color=LINE, width=0.75):
    ln = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x), Inches(y), Inches(x + w), Inches(y)); ln.line.color.rgb = color; ln.line.width = Pt(width)

def notes(s, t):
    ns = s.notes_slide
    if ns.notes_text_frame is None:
        body = next(sh for sh in TEMPLATE.notes_slide.shapes if sh.is_placeholder and sh.placeholder_format.type == 2)
        ns.shapes._spTree.append(copy.deepcopy(body._element))
    ns.notes_text_frame.text = t

def grid_table(s, head, rows, colw, x0=0.8, y0=1.9, size=10.5, lh=0.18, pad=0.09):
    W = sum(colw); x = x0
    for j, h in enumerate(head): text(s, x, y0, colw[j] - 0.08, 0.3, [h], 10.5, INK, True, mid=True); x += colw[j]
    y = y0 + 0.36; hline(s, x0, y, W, INK, 1)
    for row in rows:
        cells = [c if isinstance(c, list) else [c] for c in row]; h = max(len(c) for c in cells) * lh + 2 * pad; x = x0
        for j, c in enumerate(cells): text(s, x, y, colw[j] - 0.08, h, c, size if j else 11, SUB, False, mid=True); x += colw[j]
        y += h; hline(s, x0, y, W)
    return y

# ---------- 새 장 만들기 (나중에 순서를 맞춤)
# ③ 노이즈 한눈에 보기
s_noise = new_slide('노이즈 한눈에 보기')
grid_table(s_noise, ['항목', '크기', '영향', '자세히'], [
    ['0점 어긋남', 'trial끼리 최대 44µm (약 0.6px)', ['정답 자체에 오차. center → 첫 사진 한 칸이 크게 틀어짐', '(Θ +0.01: 0.16px, 평균 0.80px)'], '8쪽'],
    ['방향 바뀐 직후', 'Y -1.85 → -1.95 한 칸 1.31px', '거꾸로 찍은 trial의 첫 사진. 평균 1.44px보다 작음', '5쪽'],
    ['사진 공통 흔들림', '가로 0.014 ~ 0.032px', ['사진 전체가 같이 흔들림. 점을 많이 평균해도 안 줄어듦', '판정 한계의 주원인 (Θ trial에서 2배)'], '10쪽'],
    ['점 하나 흔들림', '가로 0.046px, 세로 0.033px', '점 수백 개를 평균하면 거의 사라짐', '10쪽'],
    ['대각선 모서리', '0.16px (다른 점의 약 3.5배)', '검정 칸 두 개가 꼭짓점만 맞닿은 125점. 측정에서 뺌', '부록 1'],
    ['밝기 변화', 'center만 +1.2%, trial 안에서 서서히 변함', '템플릿 맞춤에 영향 가능', '9쪽'],
    ['세로 밀림', 'Θ trial에서 최대 0.4px', '가로 이동과 함께 판정에 섞일 수 있음', '9쪽'],
    ['센서 잡음', '밝기 단계 0.73', '작음', '2쪽'],
], [2.0, 3.1, 5.6, 1.0])
notes(s_noise, '이번 데이터에서 확인한 노이즈를 한 장에 모았습니다. 가장 큰 것은 0점 어긋남과 사진 전체가 같이 흔들리는 양이고, 점 하나의 흔들림은 평균하면 거의 사라집니다.')

# 부록 표지
s_apx = prs.slides.add_slide(layout['DIVIDER'])
for sh in list(s_apx.placeholders): sh._element.getparent().remove(sh._element)
hline(s_apx, 0.8, 3.15, 2.6, INK, 0.75)
text(s_apx, 0.8, 3.3, 8, 0.9, ['부록'], 40, INK, True)
text(s_apx, 0.8, 4.25, 10, 0.5, ['측정 방법: QR 격자 모서리 맞춤, 측정 방법 4가지 비교, 판정 정확도'], 15, SUB)

# 부록 2: 모서리 맞춤 전수 결과 (6 trial 한 장)
s_tr = new_slide('모서리 맞춤 전수 결과 (280장)')
T = pd.read_csv(CORN / 'trial_summary.csv')
NAME = {'Y 0.1 간격 +': 'Y 0.1 간격 +쪽', 'Y 0.1 간격 -': 'Y 0.1 간격 -쪽', 'Y 0.05 홀수 +': 'Y 0.05 홀수 +쪽', 'Y 0.05 홀수 -': 'Y 0.05 홀수 -쪽', 'Θ +': 'Θ +쪽', 'Θ -': 'Θ -쪽'}
mw = 3.79; mh = mw * 654 / 2151   # 3열 x 2행
for i, r in enumerate(T.itertuples()):
    x = 0.8 + (i % 3) * (mw + 0.18); y = 2.0 + (i // 3) * (mh + 0.95)
    text(s_tr, x, y, mw, 0.25, [f'{NAME[r.trial]}  ({r.photos}장)'], 11, INK, True)
    text(s_tr, x, y + 0.28, mw, 0.22, [f'초록 {r.always_ok}점, 주황 {r.sometimes_bad + r.always_bad}점 (항상 {r.always_bad}, 가끔 {r.sometimes_bad}), 통과율 {r.valid_rate}%'], 8.5, SUB)
    s_tr.shapes.add_picture(str(CORN / f'map_{i + 1}.png'), Inches(x), Inches(y + 0.55), Inches(mw), Inches(mh))
text(s_tr, 0.8, 2.0 + 2 * (mh + 0.95) + 0.05, 11.7, 0.25, ['초록 = 그 trial 모든 사진에서 통과, 주황 = 한 장이라도 불확실. 사진마다 통과 점 수는 591 ~ 621점. 항상 불확실한 125점은 모두 대각선 모서리.'], 9, FAINT)
notes(s_tr, '6개 trial 280장 전부에 같은 모서리 755점을 맞춘 결과입니다. 항상 불확실한 125점은 모두 대각선 모서리입니다.')

# 부록 7: 기준점 오차 검증
s_b = new_slide('검증: center 기준점이 1px 틀려도 이동량은 그대로')
B = pd.read_csv(HERE / 'img' / 'ref_bias_check.csv')
nm = lambda f: ('Θ ' if f[0] == 'T' else 'Y ') + f[1:-4]
end = grid_table(s_b, ['사진', '기존 기준점으로 잰 이동', '새 기준점으로 잰 이동', '차이'],
                 [[nm(r.file), f'{r.dx_a:.4f}px', f'{r.dx_b:.4f}px', f'{r.diff_px:.4f}px'] for r in B.itertuples()], [2.4, 3.1, 3.1, 2.4])
text(s_b, 0.8, end + 0.15, 11.0, 0.7, ['기존 기준점(OpenCV QR 검출 꼭짓점)과 사진에서 다시 잰 QR 경계 중심은 가로 1.0px, 세로 0.75px 다름.',
                                       '이동량은 QR 패치 전체를 맞춘 변환(ECC)으로 재므로, 기준점 차이는 최대 0.0007px만 영향 (Y/Θ 구분 단서 0.05px의 1/70).'], 10, SUB)
notes(s_b, 'center QR 위치를 처음에 1픽셀 정도 틀리게 잡았지만, 이동량은 QR 무늬 전체를 맞춰서 재기 때문에 영향은 0.001픽셀 이하입니다.')

# ---------- 예전 덱에서 가져올 장
o_pairs, o_zero, o_order, o_result, o_need, o_plan = (import_slide(olds[i - 1]) for i in (11, 10, 8, 12, 13, 14))

# ---------- 기존 장 손보기
replace_text(src[3], 'QR 약 14.2px 이동', 'QR 약 14.35px 이동 (한 칸 평균)')
replace_text(src[3], 'QR 약 80px 이동', 'QR 약 80.4px 이동 (한 칸 평균)')
replace_text(src[3], '한 단계 이동', '한 칸 이동')
text(src[3], 7.21, 6.55, 5.33, 0.4, ['그래프의 직선 하나로 맞추면 Y 1mm = 14.2px. 0점 어긋남까지 같이 맞춰서 조금 낮게 나옴.'], 9, FAINT)
replace_text(o_zero, '그래서 명령값과 보정한 값, 두 기준으로 채점하기로 했습니다.', '그래서 center → 첫 사진 한 칸이 평균과 크게 다릅니다 (Θ +0.01: 0.16px, 평균 0.80px).')
replace_text(o_zero, '파일 이름의 정답에 최대 44µm 오차가 섞여 있습니다.', '파일 이름의 정답에 최대 44µm(약 0.6px) 오차가 섞여 있습니다.')

# ---------- 순서, 머리말, 제목
PLAN = [  # (장, 머리말, 제목) 제목 None = 그대로
    (src[0], None, None),
    (src[1], SEC1, '기초 정보'),
    (src[2], SEC2, '기준 위치'), (src[3], SEC2, '이동 관계: 명령값과 픽셀은 직선'), (src[15], SEC2, '한 칸 이동량'), (o_pairs, SEC2, 'Y/Θ 구분 단서: 평균 이동으로는 구분 불가'),
    (s_noise, SEC3, None), (o_zero, SEC3, '0점 어긋남: 같은 "0" 명령이어도 멈춘 자리가 다름'), (o_order, SEC3, '촬영 순서, 밝기, 드리프트'), (src[16], SEC3, '점 위치 흔들림: 점 하나 vs 사진 공통'),
    (o_result, SEC4, '이번 데이터에서 알게 된 것, 모르는 것'), (o_need, SEC4, '다음 데이터: 필요한 것'), (o_plan, SEC4, '다음 데이터: 분석할 것'),
    (s_apx, None, None),
    (src[5], APX, '부록 1. QR 격자 모서리 맞춤: 통과와 불확실 기준'), (s_tr, APX, '부록 2. 모서리 맞춤 전수 결과 (280장)'), (src[17], APX, '부록 3. 점 고유 흔들림 지도'),
    (src[18], APX, '부록 4. 측정 방법 4가지 비교 (단위 px)'), (src[20], APX, '부록 5. 측정 방법별 판정: 280장 중 틀린 장수'),
    (src[19], APX, '부록 6. 측정 방법별 판정 정확도 상세'), (s_b, APX, '부록 7. 검증: center 기준점이 1px 틀려도 이동량은 그대로'),
]
for s, sec, title in PLAN:
    if sec: set_eyebrow(s, sec)
    if title: set_title(s, title)
keep = {id(s) for s, _, _ in PLAN}
lst = prs.slides._sldIdLst; by_slide = dict(zip([id(s) for s in prs.slides], list(lst)))
for el in list(lst): lst.remove(el)
for s, _, _ in PLAN: lst.append(by_slide[id(s)])
# 빠진 장(중복, trial 6장, ECC 한 칸 장)은 관계도 끊어서 파일에서 지움
dropped = [s for s in src if id(s) not in keep]
for s in dropped:
    for rel in list(prs.part.rels.values()):
        if rel.reltype.endswith('/slide') and rel.target_part is s.part: prs.part.drop_rel(rel.rId)
prs.save(OUT)
print('saved', OUT.name, len(PLAN), 'slides, dropped', len(dropped))
