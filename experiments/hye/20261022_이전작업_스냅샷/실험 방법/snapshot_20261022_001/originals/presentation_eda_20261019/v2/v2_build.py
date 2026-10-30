# 정리본 v2: 사용자가 고친 SEMES_01_데이터EDA_정리본.pptx (읽기만) 에 코멘트를 반영해 SEMES_01_데이터EDA_정리본_v2.pptx 로 저장
#   2번 기초 정보   : 발표 메모에 그림, 오버레이 기준, 코드 경로, 표 정보 출처
#   3번 기준 위치    : 표 맨 위에 '사진 전체 크기' 줄 추가, 메모에 원본 사진과 코드 경로
#   4번 기준 위치(특징점): 모서리 755점 그림과 표로 교체 (630점 사용, 대각선 125점 제외), 메모
#   5번 0점 위치     : trial별 0점 그림과 표, 아래 색 박스에 보정 방법, 메모
#   6번 앞에 새 장   : 이동 경로 (630점 평균), 6번 한 칸 이동량 메모에 정확한 단계
#   노이즈 7~10번    : 4분할 2장으로 교체 (제목 없이 그림과 짧은 설명만)
import copy, io, pathlib, numpy as np, pandas as pd
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE, MSO_SHAPE_TYPE
from pptx.enum.text import MSO_ANCHOR
from pptx.oxml.ns import qn
HERE = pathlib.Path(__file__).parent; DECK = HERE.parent; IMG = HERE / 'img'
SRC, OUT = DECK / 'SEMES_01_데이터EDA_정리본.pptx', DECK / 'SEMES_01_데이터EDA_정리본_v2.pptx'
SUB, LINE, FAINT, INK, ACC, ACCS = (RGBColor.from_string(c) for c in ['4B5563', 'E5E7EB', '9CA3AF', '111827', '0071E3', 'EAF3FD'])
FONT, E = 'Noto Sans KR', 914400
ROOTS = r'C:\Users\SSAFY\Downloads\[기업연계_세메스] 교보재\[기업연계_세메스] 교보재\extracted\SSAFY Test Image Set'
KX, KW, VX, VW, LW = 7.21, 1.3, 8.51, 4.03, 5.33

prs = Presentation(SRC); S = list(prs.slides)
def title_of(s):
    return next((sh.text_frame.text for sh in s.shapes if sh.has_text_frame and sh.top is not None and 0.8 < sh.top / E < 1.2 and not sh.text_frame.text.startswith('1-1')), '')
find = lambda key: next(s for s in S if key in title_of(s))
s2, s3, s4, s5, s6 = find('기초 정보'), find('전체 QR 기준'), find('특징점'), find('0점 위치'), find('한 칸 이동량')
noise_old = [s for s in S if title_of(s).startswith(('노이즈', '0점 어긋남', '촬영 순서', '점 위치 흔들림'))]

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
    return tb

def notes(s, t):
    ns = s.notes_slide
    if ns.notes_text_frame is None:
        body = next(sh for sh in S[3].notes_slide.shapes if sh.is_placeholder and sh.placeholder_format.type == 2)
        ns.shapes._spTree.append(copy.deepcopy(body._element))
    ns.notes_text_frame.text = t

def clear_right_table(s):
    """오른쪽 '항목 / 값 / 선' 표 지우기 (x >= 7.0in, 머리말과 제목 아래)"""
    for sh in list(s.shapes):
        if sh.left is not None and sh.left / E >= 6.95 and sh.top / E > 1.8 and sh.shape_type != MSO_SHAPE_TYPE.PICTURE and not sh.is_placeholder:
            sh._element.getparent().remove(sh._element)

def rows_table(s, rows, top=2.03, bottom=6.43, lh=0.2):
    pad = (bottom - top - sum(len(v) for _, v in rows) * lh) / len(rows); y = top
    for k, vals in rows:
        text(s, KX, y + pad / 2 - 0.02, KW, 0.3, [k], 12)
        text(s, VX, y + pad / 2, VW, len(vals) * lh, vals, 11)
        y += len(vals) * lh + pad
        ln = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(KX), Inches(y), Inches(KX + LW), Inches(y)); ln.line.color.rgb = LINE; ln.line.width = Pt(0.75)

def replace_picture(s, path, keep_width=True):
    pic = next(sh for sh in s.shapes if sh.shape_type == MSO_SHAPE_TYPE.PICTURE)
    x, y, w, h = pic.left, pic.top, pic.width, pic.height; pic._element.getparent().remove(pic._element)
    from PIL import Image; iw, ih = Image.open(path).size
    nh = int(w * ih / iw); s.shapes.add_picture(str(path), x, y + (h - nh) // 2, w, nh)

def set_caption(s, startswith, new):
    for sh in s.shapes:
        if sh.has_text_frame and sh.text_frame.text.startswith(startswith):
            rr = sh.text_frame.paragraphs[0].runs; rr[0].text = new
            for r in rr[1:]: r.text = ''
            return
    raise KeyError(startswith)

# ---------------- 2번 기초 정보: 메모
notes(s2, f'''[그림] 가운데 QR 확대 13장
- 원본: {ROOTS}\\center.bmp, Y\\Y-0.10 Y-2.00 Y+0.10 Y+2.00 Y-0.05 Y-1.95 Y+0.05 Y+1.95 .bmp, Theta\\T-0.01 T-1.00 T+0.01 T+1.00 .bmp
- 자르기: 모든 사진을 같은 창 x 1720~2080, y 970~1170 (가운데 QR + 최대 이동 여유)으로 잘라 4배 확대 (presentation_eda_20261019/make_qr_crops.py)
- 열: trial마다 -쪽 최대, -쪽 최소(첫 이동), +쪽 최소, +쪽 최대. 행: Y 0.1 간격, Y 0.05 홀수, Θ
[오버레이 기준] presentation_eda_20261019/make_qr_boxes.py, QR 네모는 qr_bbox.py
- QR 네모 = 사진에서 직접 잰 QR 모듈 경계: 밝은 QR 라벨 안쪽의 어두운 모듈이 차지하는 범위 (Otsu 이진화)
- 13장 모두: 파란 점선 = center 사진의 QR 위치
- 최소 이동 6장: + 초록 실선 = 그 사진의 QR 위치
- 최대 이동 6장: + 초록 점선 = 같은 trial 최소 이동 사진의 QR 위치, 빨강 실선 = 그 사진의 QR 위치
- 아래 px = 가로 이동량 (14_데이터_QR마커위치.csv 의 dCx, ECC로 잰 값)
[표 정보 출처]
- 총 장수, 구성: 원본 폴더 파일 목록 (center 1, Y 폴더 80, Theta 폴더 200)
- trial 수: 파일 수정 시각으로 촬영 순서를 복원해 6구간으로 나눔
- ADDRESS: 사진 속 라벨을 눈으로 읽음 / QR: OpenCV QR 디코딩 (codex_experiments_20261014/decoded_grid_corners_20261015/build.py)
- 해상도, 형식: BMP 헤더, 세 채널 값이 모두 같은지 픽셀로 확인
- 라벨 범위, 간격, 구조: 파일 이름 (T-0.35.bmp = Θ -0.35)
- 단위: Θ 1 = 약 80px 이동, Y 1mm = 약 14.4px 이동으로 회전 반경을 계산해 도 단위로 추정 (SEMES 확인 필요)''')

# ---------------- 3번 기준 위치 (전체 QR): 표에 사진 크기 줄 추가, 메모
clear_right_table(s3)
rows_table(s3, [('사진 전체 크기', ['3840 × 2160px']), ('QR 중심 좌표', ['L (1644, 1068)', 'C (1902, 1069)', 'R (2160, 1070)']), ('QR 크기', ['약 164 × 174px (폭 × 높이)']),
                ('QR 간격', ['258px (L와 C), 258px (C와 R)']), ('QR 줄 기울기', ['R이 L보다 2px 낮음 (약 0.2°)']), ('사용 영역', ['QR 3개', '판 나사 2개는 비교 실험에서만']),
                ('판까지 거리', ['약 320mm (추정)', 'Θ가 도 단위라고 가정'])])
notes(s3, f'''[원본] {ROOTS}\\center.bmp
[그림 코드] presentation_eda_20261019/make_ref_figure.py (QR 경계는 qr_bbox.py)
- QR 네모 = 밝은 QR 라벨 안쪽의 어두운 모듈 범위, 점 = 네모 가운데, 폭/높이/간격 = 네모 크기와 가운데끼리 거리
- 줄 기울기 = L과 R 가운데의 높이 차 / 가로 거리
- 판까지 거리 = Θ 1에 80px 이동, Y 1mm에 14.4px 이동 -> 회전 반경 약 32cm (Θ가 도 단위일 때)
- 이 좌표들은 정수 픽셀 경계로 잰 값. 이동량은 따로 서브픽셀로 잼 (부록 7: 기준점이 1px 틀려도 이동량 영향 0.0007px 이하)''')

# ---------------- 4번 기준 위치 (특징점)
F = pd.read_csv(IMG / 'corner_ref.csv').set_index('qr')
cx = {q: (F.x_min[q] + F.x_max[q]) / 2 for q in 'LCR'}; cy = {q: (F.y_min[q] + F.y_max[q]) / 2 for q in 'LCR'}
replace_picture(s4, IMG / 'ref_corners.png')
set_caption(s4, 'center 사진의 QR 3개', 'center 사진의 QR 격자 모서리. 초록 = 사용 630점, 주황 = 대각선 모서리 125점 (제외), 파랑 = 모서리 범위로 잡은 QR 네모')
clear_right_table(s4)
rows_table(s4, [('사진 전체 크기', ['3840 × 2160px']),
                ('모서리 수', [f'755점 (L {F.points.L}, C {F.points.C}, R {F.points.R})', 'QR 디코딩 격자에서 검정과 흰 칸이 만나는 점']),
                ('사용 점', [f'630점 (L {F.used.L}, C {F.used.C}, R {F.used.R})', '대각선 모서리 125점은 뺌 (흔들림 3배)']),
                ('QR 중심 좌표', [f'L ({cx["L"]:.0f}, {cy["L"]:.0f})  C ({cx["C"]:.0f}, {cy["C"]:.0f})  R ({cx["R"]:.0f}, {cy["R"]:.0f})', '(사용 점 범위의 가운데, 서브픽셀)']),
                ('QR 크기', [f'약 {np.mean([F.x_max[q] - F.x_min[q] for q in "LCR"]):.0f} × {np.mean([F.y_max[q] - F.y_min[q] for q in "LCR"]):.0f}px (바깥 모서리 사이)']),
                ('QR 간격', [f'{cx["C"] - cx["L"]:.1f}px (L와 C), {cx["R"] - cx["C"]:.1f}px (C와 R)']),
                ('center 진단 통과', ['755점 중 597점', '(통과와 무관하게 대각선만 빼고 630점을 씀)'])])
notes(s4, f'''[원본] {ROOTS}\\center.bmp
[모서리 맞춤] analysis_grid_corners_20261019/track_corners.py -> out/corners_280.csv (281장 x 755점)
- QR을 디코딩해 얻은 이상적인 격자에서 검정과 흰 칸이 만나는 꼭짓점 755개에 ID를 붙임 (codex_experiments_20261014/decoded_grid_corners_20261015)
- 사진마다 같은 ID의 점 주변 밝기에 흐려진 2x2 검정/흰 경계 모델을 맞춰 서브픽셀 위치를 구함 (창을 맞춘 위치로 옮겨 다시 맞춤)
- 대각선 모서리(검정 칸 두 개가 꼭짓점만 맞닿은 점, 125개)는 인쇄가 번져 모델과 모양이 달라 위치가 0.16px로 흔들려서 뺌
[그림 코드] presentation_eda_20261019/v2/v2_figs.py (ref_corners.png), 숫자 v2/v2_data.py (corner_ref.csv)
- QR 중심 = 사용 점들의 x, y 범위 가운데. 3번(모듈 경계로 잰 값)과 1px 안에서 같음''')

# ---------------- 5번 0점 위치
Z = pd.read_csv(IMG / 'zero_offsets.csv').set_index('trial')
replace_picture(s5, IMG / 'zero.png') if any(sh.shape_type == MSO_SHAPE_TYPE.PICTURE for sh in s5.shapes) else s5.shapes.add_picture(str(IMG / 'zero.png'), Inches(0.8), Inches(2.1), Inches(5.9), Inches(2.95))
for sh in list(s5.shapes):   # 비어 있던 '-' 글상자 지우기
    if sh.has_text_frame and sh.text_frame.text.strip() == '-': sh._element.getparent().remove(sh._element)
clear_right_table(s5)
yz = Z[Z.axis == 'Y']; tz = Z[Z.axis == 'Θ']
f = lambda t: f'{Z.zero_px[t]:+.2f}px'
rows_table(s5, [('0점이란', ['명령값 0에서 그 trial의 평균 가로 이동', '(trial 사진들을 직선으로 맞춘 절편, center = 0)']),
                ('Y trial 0점', [f'0.1 간격 +쪽 {f("Y 0.1 간격 +")} ({Z.zero_um["Y 0.1 간격 +"]:+.0f}µm),  -쪽 {f("Y 0.1 간격 -")} ({Z.zero_um["Y 0.1 간격 -"]:+.0f}µm)',
                                 f'0.05 홀수 +쪽 {f("Y 0.05 홀수 +")} ({Z.zero_um["Y 0.05 홀수 +"]:+.0f}µm),  -쪽 {f("Y 0.05 홀수 -")} ({Z.zero_um["Y 0.05 홀수 -"]:+.0f}µm)']),
                ('Θ trial 0점', [f'+쪽 {f("Θ +")} (Θ {Z.zero_cmd["Θ +"]:+.4f}),  -쪽 {f("Θ -")} (Θ {Z.zero_cmd["Θ -"]:+.4f})']),
                ('차이 크기', [f'Y trial끼리 최대 {yz.zero_px.max() - yz.zero_px.min():.2f}px = {yz.zero_um.max() - yz.zero_um.min():.0f}µm (한 칸의 약 절반)',
                              f'Θ trial끼리 {tz.zero_px.max() - tz.zero_px.min():.2f}px = Θ {tz.zero_cmd.max() - tz.zero_cmd.min():.4f} (한 칸의 약 80%)']),
                ('영향', ['첫 칸이 크게 틀어짐 (Θ +0.01: 0.16px, 평균 0.80px)', '0 근처 사진일수록 Y/Θ 판정이 흔들림'])],
           top=2.03, bottom=5.7)
box = s5.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(5.95), Inches(11.73), Inches(0.85)); box.adjustments[0] = 0.15
box.fill.solid(); box.fill.fore_color.rgb = ACCS; box.line.fill.background(); box.shadow.inherit = False
text(s5, 1.05, 6.02, 11.3, 0.75, ['보정 방법 (간단히)',
     '① 항상 같은 방향에서 다가가게 해서 백래시를 한쪽으로 고정   ② center 복귀 반복 촬영(다음 데이터 B)으로 방향별 0점을 재서 저장   ③ 운영 모델은 지금처럼 center 하나로 고정하고, 남는 0점 차이는 허용 오차 예산에 넣음'], 10.5, INK)
s5.shapes[-1].text_frame.paragraphs[0].runs[0].font.bold = True; s5.shapes[-1].text_frame.paragraphs[0].runs[0].font.color.rgb = ACC
notes(s5, '''[숫자] presentation_eda_20261019/v2/v2_data.py -> img/zero_offsets.csv, [그림] v2_figs.py -> img/zero.png
[방법] 모서리 630점(대각선 제외)의 사진별 평균 가로 이동을 trial마다 명령값에 직선으로 맞추고, 명령값 0에서의 값(절편)을 그 trial의 0점으로 봄. center는 0.
- 그림 세로축 = 평균 가로 이동에서 축별 공통 기울기를 빼고 남은 양. 마름모 = 각 trial 직선이 0을 지나는 높이 (= 0점)
- Y는 1mm = 14.4px 로 µm 환산, Θ는 Θ 1 = 80.5px 로 명령 단위 환산
- 이동 방향에 따라 다른 것으로 보아 백래시로 추정 (예전 ECC 분석의 -37, +7, -31, -30µm와 같은 경향)''')

# ---------------- 6번 앞에 '이동 경로' 장 (6번을 복사해 틀로 씀)
idx6 = list(prs.slides).index(s6)
sp = prs.slides.add_slide(s6.slide_layout)
for sh in list(sp.placeholders): sh._element.getparent().remove(sh._element)
for sh in s6.shapes:
    if sh.has_text_frame and (sh.text_frame.text.startswith('1-1') or '한 칸 이동량' in sh.text_frame.text) or (sh.is_placeholder and 'Slide Number' in sh.name):
        sp.shapes._spTree.append(copy.deepcopy(sh._element))
t = next(sh for sh in sp.shapes if sh.has_text_frame and '한 칸 이동량' in sh.text_frame.text)
rr = t.text_frame.paragraphs[0].runs; rr[0].text = '이동 관계 – 이동 경로'
for r in rr[1:]: r.text = ''
sp.shapes.add_picture(str(IMG / 'path.png'), Inches(0.8), Inches(2.4), Inches(6.0), Inches(3.0))
text(sp, 0.8, 5.5, 6.0, 0.45, ['모서리 630점의 사진별 평균 이동. 선은 명령값 크기 순 연결이고 촬영 시간 순 궤적이 아님.'], 9, FAINT)
M = pd.read_csv(IMG / 'photo_motion.csv'); mm = M[M.trial != 'center']; kd = np.polyfit(mm.dx, mm.dy, 1)
rows_table(sp, [('Y 기울기', [f'trial별 {yz.slope_px.min():.2f} ~ {yz.slope_px.max():.2f}px / 1mm']),
                ('Θ 기울기', [f'+쪽 {Z.slope_px["Θ +"]:.2f}px, -쪽 {Z.slope_px["Θ -"]:.2f}px / Θ 1', f'(+쪽과 -쪽이 {100 * abs(Z.slope_px["Θ -"] / Z.slope_px["Θ +"] - 1):.1f}% 다름)']),
                ('직선 잔차', [f'Y {yz.resid_px.min():.3f} ~ {yz.resid_px.max():.3f}px,  Θ {tz.resid_px.min():.3f} ~ {tz.resid_px.max():.3f}px (σ)']),
                ('세로 이동', [f'가로 이동의 {100 * kd[0]:.2f}% (이동 방향이 화면에서 {np.degrees(np.arctan(kd[0])):.2f}° 기울어짐)', 'trial마다 0 ~ 0.08px 위아래 차이']),
                ('사용 점', ['대각선 모서리를 뺀 630점 평균', '(사진마다 같은 점, 통과 여부 무관)'])])
notes(sp, '''[숫자] presentation_eda_20261019/v2/v2_data.py -> img/photo_motion.csv, zero_offsets.csv, [그림] v2_figs.py -> img/path.png
[방법] ① 사진마다 모서리 630점의 (가로, 세로) 이동 = 그 사진 위치 - center 위치  ② 630점 평균  ③ trial마다 명령값 크기 순으로 이어 그림
- 기울기 = trial마다 평균 가로 이동을 명령값 직선으로 맞춘 기울기, 벗어난 양 = 그 직선에서 남은 값의 표준편차
- 세로 이동은 가로 이동에 비례하는 부분(이동 방향 기울기)과 trial마다 다른 위아래 차이로 나뉨''')
lst = prs.slides._sldIdLst; el = list(lst)[-1]; lst.remove(el); lst.insert(idx6, el)

notes(s6, '''[숫자] analysis_grid_corners_20261019/corner_steps.py (RULE=nondiag) -> out/nondiag/corner_step_px.csv, [그림] corner_step_figure.py
[정확한 단계]
① 사진마다 모서리 630점(대각선 제외)의 서브픽셀 위치를 구함 (track_corners.py)
② 이동량 = 그 사진 위치 - center 위치 (같은 ID 점끼리)
③ 같은 trial에서 |명령값| 순서로 이웃한 두 사진을 짝지음 (Y는 한 칸 0.1mm, Θ는 0.01)
④ 두 사진의 가로 이동 차이를 630점 평균 (바깥쪽 방향을 + 로)
⑤ 축마다 평균, 표준편차, 평균 ±3σ 밖 칸을 구함. 첫 칸(center -> 첫 사진)은 0점 영향이 섞여 따로 표시
⑥ ECC(QR 패치 전체 맞춤) 결과와 같은 칸끼리 비교''')

# ---------------- 노이즈: 7~10번을 4분할 2장으로 교체
tmpl = noise_old[0]
PANELS = [[('n1.png', '촬영 순서', '6 trial을 58분에 걸쳐 찍음. 33~48분은 15분 공백, 마지막 trial은 -1.95에서 0 쪽으로 거꾸로 찍음.'),
           ('n2.png', '0점 어긋남', 'trial마다 0점이 다름. Y는 최대 0.65px(46µm), Θ +쪽은 -0.84px. 5쪽 참고.'),
           ('n3.png', '방향 바뀐 직후', '거꾸로 찍은 trial의 첫 칸만 1.31px로 평균 1.44px보다 작음. 백래시로 추정.'),
           ('n4.png', '밝기 변화', 'center만 QR 판 기준 +1.9% 밝고, trial 안에서도 서서히 변함. 센서 잡음은 0.73으로 작음.')],
          [('n5.png', '세로 이동', '세로 이동은 대부분 가로 이동에 비례 (이동 방향이 0.14° 기울어짐). 남는 흔들림은 0.01px 정도.'),
           ('n6.png', '사진 공통 흔들림', '사진 전체가 같이 흔들린 양. Y 0.013~0.014px (거꾸로 찍은 trial 0.027), Θ 0.029~0.032px. 판정 한계의 주원인.'),
           ('n7.png', '점 하나 흔들림', '점 하나는 가로 0.047px, 세로 0.033px. 630점을 평균하면 거의 사라짐.'),
           ('n8.png', '대각선 모서리', '검정 칸 두 개가 꼭짓점만 맞닿은 125점은 0.16px로 3배 넘게 흔들려 측정에서 뺌.')]]
pos = list(prs.slides).index(tmpl); new_noise = []
for page in PANELS:
    ns_ = prs.slides.add_slide(tmpl.slide_layout)
    for sh in list(ns_.placeholders): sh._element.getparent().remove(sh._element)
    for sh in tmpl.shapes:
        if sh.has_text_frame and sh.text_frame.text.startswith('1-1') or (sh.is_placeholder and 'Slide Number' in sh.name):
            ns_.shapes._spTree.append(copy.deepcopy(sh._element))
    cw = 5.75; ih = cw * 2.3 / 5.6
    for k, (img, lab, desc) in enumerate(page):
        x = 0.8 + (k % 2) * (cw + 0.23); y = 1.05 + (k // 2) * (ih + 0.62)
        text(ns_, x, y, cw, 0.26, [lab], 12, INK, True)
        text(ns_, x, y + 0.27, cw, 0.3, [desc], 9, SUB)
        ns_.shapes.add_picture(str(IMG / img), Inches(x), Inches(y + 0.6), Inches(cw), Inches(ih))
    notes(ns_, '[그림] presentation_eda_20261019/v2/v2_figs.py (' + ', '.join(p[0] for p in page) + '), 숫자 v2_data.py. 모서리 기준은 대각선 제외 630점.\n' + '\n'.join(f'- {p[1]}: {p[2]}' for p in page))
    new_noise.append(ns_)
# ---------------- 순서: 옛 노이즈 장 자리에 새 노이즈 2장, 옛 노이즈 장은 파일에서 지움
by = {id(s_): e for s_, e in zip(list(prs.slides), list(lst))}
order = []
for s_ in list(prs.slides):
    if s_ in new_noise: continue
    if s_ is noise_old[0]: order += new_noise
    if s_ in noise_old: continue
    order.append(s_)
for e in list(lst): lst.remove(e)
for s_ in order: lst.append(by[id(s_)])
for s_ in noise_old:
    for rel in list(prs.part.rels.values()):
        if rel.reltype.endswith('/slide') and rel.target_part is s_.part: prs.part.drop_rel(rel.rId)
prs.save(OUT)
print('saved', OUT.name, len(order), 'slides')
