# '기준 위치' 슬라이드의 그림과 표 숫자를 실제 QR 경계 측정값(qr_bbox)으로 바꾼다
#   사용: python fix_ref_slide.py 입력.pptx 출력.pptx
import sys, pathlib, math
from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE
HERE = pathlib.Path(__file__).parent
F, OUT = HERE / sys.argv[1], HERE / sys.argv[2]
prs = Presentation(F)
s = next(sl for sl in prs.slides if any(sh.has_text_frame and sh.text_frame.text.endswith('기준 위치') for sh in sl.shapes))
pic = next(sh for sh in s.shapes if sh.shape_type == MSO_SHAPE_TYPE.PICTURE)
_, rid = s.part.get_or_add_image_part(str(HERE / 'img' / 'ref_center.png'))
pic._element.blipFill.blip.rEmbed = rid
L, C, R = (1644.0, 1068.0), (1901.5, 1069.0), (2160.0, 1070.0)   # make_ref_figure.py 출력 (QR 경계 중심)
tilt = math.degrees(math.atan2(R[1] - L[1], R[0] - L[0]))
NEW = {'L (1642, 1068)': 'L (1644, 1068)', 'C (1900, 1068)': 'C (1902, 1069)', 'R (2159, 1071)': 'R (2160, 1070)',
       '약 160 × 172px (폭 × 높이)': '약 164 × 174px (폭 × 높이)',
       '258px (L와 C), 259px (C와 R)': '258px (L와 C), 258px (C와 R)',
       'R이 L보다 3px 낮음 (약 0.3°)': f'R이 L보다 2px 낮음 (약 {tilt:.1f}°)'}
n = 0
for sh in s.shapes:
    if sh.has_text_frame:
        for p in sh.text_frame.paragraphs:
            for r in p.runs:
                if r.text in NEW: r.text = NEW[r.text]; n += 1
        if sh.text_frame.text.startswith('center 사진의 QR 3개'):
            sh.text_frame.paragraphs[0].runs[0].text = 'center 사진의 QR 3개. 파란 네모 = 사진에서 잰 QR 모듈 경계, 점 = 중심 (좌표 단위 px)'
prs.save(OUT)
print('replaced', n, 'texts')
