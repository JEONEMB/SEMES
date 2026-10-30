# 정리본 v3 5번 '이동 관계 – ECC 분석' 채우기 (4번 특징점 장과 같은 틀)
#   그림: img/ecc_ref.png  center 사진의 QR 3개 ECC 기준 패치 (180 x 192px)
#   숫자: v2/ecc_measure.py 로 원본 281장을 새로 잰 img/ecc_motion.csv
import pathlib, numpy as np, pandas as pd, cv2
from PIL import Image, ImageDraw
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE_TYPE
from pptx.oxml.ns import qn
HERE = pathlib.Path(__file__).parent; DECK = HERE.parent; IMG = HERE / 'img'; PROJ = DECK.parent
ROOT = pathlib.Path(r"C:\Users\SSAFY\Downloads\[기업연계_세메스] 교보재\[기업연계_세메스] 교보재\extracted\SSAFY Test Image Set")
import sys
V3 = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else DECK / 'SEMES_01_데이터EDA_정리본_v3.pptx'   # 인자로 다른 파일을 주면 그 파일을 고침
SUB, LINE, FAINT = (RGBColor.from_string(c) for c in ['4B5563', 'E5E7EB', '9CA3AF'])
FONT, E = 'Noto Sans KR', 914400
KX, KW, VX, VW, LW = 7.21, 1.3, 8.51, 4.03, 5.33
BX = pd.read_csv(IMG / 'ecc_boxes.csv').set_index('qr')     # ecc_measure.py 가 center 사진의 QR 라벨에서 잡은 패치
BOXES = [tuple(int(BX.loc[q, k]) for k in ('x0', 'x1', 'y0', 'y1')) for q in 'LCR']
MARGIN = int(BX.label_x0.L - BX.x0.L)

# ---- 그림: 패치 3개가 다 들어가게 자르고, 4번 ref_corners.png 와 같은 배율
X0, Y0, X1, Y1, K = min(b[0] for b in BOXES) - 4, min(b[2] for b in BOXES) - 4, max(b[1] for b in BOXES) + 4, max(b[3] for b in BOXES) + 4, 3
g = cv2.imdecode(np.fromfile(str(ROOT / 'center.bmp'), np.uint8), cv2.IMREAD_GRAYSCALE)[Y0:Y1, X0:X1]
im = Image.fromarray(cv2.resize(g, None, fx=K, fy=K, interpolation=cv2.INTER_CUBIC)).convert('RGB'); d = ImageDraw.Draw(im)
for x0, x1, y0, y1 in BOXES:
    d.rectangle(((x0 - X0) * K, (y0 - Y0) * K, (x1 - X0) * K - 1, (y1 - Y0) * K - 1), outline=(0, 113, 227), width=5)
    cx, cy = ((x0 + x1) / 2 - X0) * K, ((y0 + y1) / 2 - Y0) * K; d.ellipse((cx - 10, cy - 10, cx + 10, cy + 10), fill=(0, 113, 227))
im.save(IMG / 'ecc_ref.png')

# ---- 숫자
M = pd.read_csv(IMG / 'ecc_motion.csv'); M = M[M.file != 'center.bmp']
cc = np.concatenate([M[f'{q}_cc'].values for q in 'LCR'])
mid = [((x0 + x1) / 2, (y0 + y1) / 2) for x0, x1, y0, y1 in BOXES]
SZ = [f'{b[1] - b[0]}×{b[3] - b[2]}' for b in BOXES]
rows = [('사진 전체 크기', ['3840 × 2160px']),
        ('기준 패치', [f'QR 라벨 전체 + 바깥 {MARGIN}px (세 QR 같은 규칙)', 'L ' + SZ[0] + ', C ' + SZ[1] + ', R ' + SZ[2] + 'px']),
        ('사용 픽셀', ['패치 전체 (점을 고르지 않음)', '라벨 테두리와 QR 모듈을 모두 씀']),
        ('패치 중심 좌표', ['  '.join(f'{n} ({x:.0f}, {y:.0f})' for n, (x, y) in zip('LCR', mid)), '(center 사진의 라벨 가운데)']),
        ('패치 간격', [f'{mid[1][0] - mid[0][0]:.1f}px (L와 C), {mid[2][0] - mid[1][0]:.1f}px (C와 R)']),
        ('맞추는 방법', ['① 템플릿 매칭으로 대략 위치 (±130px 안)', '② ECC 아핀으로 서브픽셀 정렬, 가운데 점 이동 사용']),
        ('맞춤 품질', [f'ECC 상관계수 최저 {cc.min():.3f}, 중앙값 {np.median(cc):.3f}', f'({len(M)}장 × QR 3개, 1 = center 와 똑같음)'])]

# ---- 슬라이드
prs = Presentation(V3)
def title(s):
    return next((sh.text_frame.text for sh in s.shapes if sh.has_text_frame and sh.top is not None and 0.8 < sh.top / E < 1.2 and not sh.text_frame.text.startswith('1-1')), '')
s = next(s for s in prs.slides if title(s) == '이동 관계 – ECC 분석')

def text(x, y, w, h, lines, size, color=SUB):
    tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h)); tf = tb.text_frame; tf.word_wrap = True
    for m in ('margin_left', 'margin_right', 'margin_top', 'margin_bottom'): setattr(tf, m, 0)
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        r = p.add_run(); r.text = line; r.font.size = Pt(size); r.font.color.rgb = color; r.font.name = FONT
        rpr = r._r.get_or_add_rPr(); ea = rpr.find(qn('a:ea'))
        if ea is None: ea = rpr.makeelement(qn('a:ea'), {}); rpr.append(ea)
        ea.set('typeface', FONT)

# 그림 교체 (위치와 크기는 4번과 같게)
pic = next(sh for sh in s.shapes if sh.shape_type == MSO_SHAPE_TYPE.PICTURE)
x, y, w, h = pic.left, pic.top, pic.width, pic.height; pic._element.getparent().remove(pic._element)
iw, ih = Image.open(IMG / 'ecc_ref.png').size; nh = int(w * ih / iw)   # 가로는 4번과 같게, 세로는 그림 비율대로
s.shapes.add_picture(str(IMG / 'ecc_ref.png'), x, y + (h - nh) // 2, w, nh)
cap = next(sh for sh in s.shapes if sh.has_text_frame and sh.text_frame.text.startswith('center 사진의'))
rs = cap.text_frame.paragraphs[0].runs; rs[0].text = f'center 사진의 QR 3개. 파랑 네모 = ECC 기준 패치 (QR 라벨 + 바깥 {MARGIN}px), 점 = 패치 가운데'
for r in rs[1:]: r.text = ''

# 오른쪽 표 다시 그림
for sh in list(s.shapes):
    if sh.left is not None and sh.left / E >= 6.95 and sh.top / E > 1.8 and not sh.is_placeholder and sh.shape_type != MSO_SHAPE_TYPE.PICTURE:
        sh._element.getparent().remove(sh._element)
top, bottom, lh = 2.03, 6.43, 0.2
pad = (bottom - top - sum(len(v) for _, v in rows) * lh) / len(rows); yy = top
for k, vals in rows:
    text(KX, yy + pad / 2 - 0.02, KW, 0.3, [k], 12); text(VX, yy + pad / 2, VW, len(vals) * lh, vals, 11)
    yy += len(vals) * lh + pad
    ln = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(KX), Inches(yy), Inches(KX + LW), Inches(yy)); ln.line.color.rgb = LINE; ln.line.width = Pt(0.75)

s.notes_slide.notes_text_frame.text = f'''[원본]
{ROOT / 'center.bmp'}

[측정 코드]
presentation_eda_20261019/v2/ecc_measure.py -> v2/img/ecc_motion.csv (원본 281장 x QR 3개, 새로 측정)

[그림 코드]
presentation_eda_20261019/v2/ecc_slide.py -> v2/img/ecc_ref.png

[방법]
① center 사진에서 QR 라벨(밝은 사각형)을 찾아 라벨 전체 + 바깥 {MARGIN}px 를 기준(템플릿)으로 둠 (v2/img/ecc_boxes.csv)
② 다른 사진마다 ±130px 범위에서 템플릿 매칭으로 대략 위치를 찾음
③ 그 자리에서 ECC 아핀(이동, 회전, 크기, 기울임)으로 패치 전체를 center 패치에 맞춤
④ 패치 가운데 점이 변환으로 옮겨진 양 = 그 QR의 이동 (회전, 크기 성분이 섞이지 않게 가운데 기준)
- 특징점 분석은 모서리 630점을 하나씩 찾아 평균, ECC는 QR마다 변환 하나로 패치 전체 밝기를 맞춤
- 맞춤 품질: ECC 상관계수 최저 {cc.min():.4f}, 중앙값 {np.median(cc):.4f}'''
prs.save(V3)
for k, v in rows: print(k, '|', ' / '.join(v))
