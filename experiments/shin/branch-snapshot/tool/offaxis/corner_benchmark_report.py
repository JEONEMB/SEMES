"""Standalone Korean visual report and inspectable point overlay, offline HTML."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'.analysis_packages'))
import base64
import io
import json
import html
import numpy as np
import pandas as pd
from PIL import Image
from PIL import ImageDraw,ImageFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT=ROOT/'docs/offaxis-eda/corner-benchmark'
DATA=ROOT/'data/SSAFY Test Image Set'


def figure_pipeline():
    fig,ax=plt.subplots(figsize=(15,5),layout='constrained');ax.set_xlim(0,15);ax.set_ylim(0,5);ax.axis('off')
    boxes=[(.2,3.1,'Known plane geometry\nQR modules / material corners'),(4.2,3.1,'Known virtual camera\nK, R, t + print retreat'),
           (8.2,3.1,'Forward image formation\nPSF, sensor integration, texture'),(12.1,3.1,'Synthetic image\nKnown ground truth'),
           (.2,.7,'Original 341 photographs\nNo calibrated K / actual pose'),(4.2,.7,'All corner candidates\nX crossings + L corners'),
           (8.2,.7,'Three point refiners\nSubpix / edge lines / model'),(12.1,.7,'Held-point projection\nConsistency, not truth')]
    for i,(x,y,label) in enumerate(boxes):
        w=2.7 if x<12 else 2.6
        box=FancyBboxPatch((x,y),w,1.1,boxstyle='round,pad=0.12',fc='#e9f4ff' if i<4 else '#edf8ef',ec='#3f6c8a')
        ax.add_patch(box);ax.text(x+w/2,y+.55,label,ha='center',va='center',fontsize=10)
        if i%4!=3:ax.annotate('',(x+3.85,y+.55),(x+w+.15,y+.55),arrowprops={'arrowstyle':'->','color':'#555555'})
    ax.text(7.5,4.75,'Two evaluation tracks prevent circular ground truth',ha='center',fontsize=17,weight='bold')
    ax.text(7.5,2.5,'Synthetic: ||detected - pi(K[R|t]X)||  |  Real: leave spatial blocks out of homography fitting',ha='center',fontsize=12)
    fig.savefig(OUT/'00_evaluation_pipeline.png',dpi=160);plt.close(fig)


def viewer_data(real):
    material=pd.read_csv(OUT/'material_projection.csv.gz')
    real=real.merge(material[['path','qr','method','id','corrected_dx','corrected_dy','corrected_error_px']],
                    on=['path','qr','method','id'],how='left',validate='one_to_one')
    paths=list(real.path.drop_duplicates())
    selected=['center.bmp','기준 위치/center1.bmp']
    for y,t in [(.05,.01),(.05,.1),(.5,1.),(1.,2.)]:
        found=real[real.y_mm.eq(y)&real.theta_deg.eq(t)&real.path.str.contains(r'\(1\)')].path.unique()
        selected.extend(found[:1])
    for prefix in ['Y/Y+1.00','Theta/T+1.00']:
        selected.extend([p for p in paths if p.startswith(prefix)][:1])
    output=[]
    for path in dict.fromkeys(selected):
        g=real[real.path.eq(path)].copy()
        finite=g[np.isfinite(g.x)&np.isfinite(g.y)]
        x0=int(finite.x.quantile(.001))-30;y0=int(finite.y.quantile(.001))-30
        x1=int(finite.x.quantile(.999))+30;y1=int(finite.y.quantile(.999))+30
        im=Image.open(DATA/path).convert('L').crop((x0,y0,x1,y1))
        buffer=io.BytesIO();im.save(buffer,format='PNG')
        points=[]
        for r in g.itertuples():
            if not np.isfinite([r.x,r.y]).all():continue
            points.append(dict(method=r.method,kind=r.kind,qr=r.qr,id=int(r.id),
                x=round(r.x-x0,5),y=round(r.y-y0,5),valid=bool(r.valid),
                dx=round(r.cv_dx,5) if np.isfinite(r.cv_dx) else None,
                dy=round(r.cv_dy,5) if np.isfinite(r.cv_dy) else None,
                error=round(r.cv_error_px,5) if np.isfinite(r.cv_error_px) else None,
                mdx=round(r.corrected_dx,5) if np.isfinite(r.corrected_dx) else None,
                mdy=round(r.corrected_dy,5) if np.isfinite(r.corrected_dy) else None,
                merror=round(r.corrected_error_px,5) if np.isfinite(r.corrected_error_px) else None))
        output.append(dict(path=path,image='data:image/png;base64,'+base64.b64encode(buffer.getvalue()).decode(),width=im.width,height=im.height,points=points))
    return output


def main():
    figure_pipeline()
    ss=pd.read_csv(OUT/'synthetic_summary.csv');rs=pd.read_csv(OUT/'real_summary.csv')
    free=pd.read_csv(OUT/'free_detection.csv');repeat=pd.read_csv(OUT/'repeatability.csv')
    material=pd.read_csv(OUT/'material_projection_summary.csv')
    parameters=pd.read_csv(OUT/'material_projection_parameters.csv')
    old=pd.read_csv(ROOT/'docs/offaxis-eda/ppt-method-v2/points.csv.gz')
    diagnostic=old.groupby('diagonal',as_index=False).agg(points=('id','count'),
        original_acceptance=('accepted','mean'),convergence=('converged','mean'),median_pixel_rmse=('rmse','median'))
    diagnostic['kind']=np.where(diagnostic.diagonal,'X','L')
    diagnostic.to_csv(OUT/'original_photometric_diagnostics.csv',index=False)
    real=pd.read_csv(OUT/'real_points.csv.gz')
    view=viewer_data(real)
    (OUT/'viewer_data.json').write_text(json.dumps(view,ensure_ascii=False),encoding='utf-8')
    syn=ss[ss.kind.eq('L')][['regime','method','coverage','median_px','rms_px','p95_px']].copy()
    actual=rs[rs.scope.eq('common_L')][['method','scored','median_px','rms_px','p95_px']].copy()
    formatters={c:(lambda x:f'{x:.4f}') for c in ['coverage','median_px','rms_px','p95_px']}
    synhtml=syn.to_html(index=False,border=0,formatters=formatters)
    realhtml=actual.to_html(index=False,border=0,formatters=formatters)
    materialhtml=material.to_html(index=False,border=0,formatters=formatters)
    textured=ss[ss.kind.eq('L')&ss.regime.eq('gap_texture')].sort_values('rms_px')
    winner=textured.iloc[0]
    materialwinner=material.sort_values('median_px').iloc[0]
    q=parameters[parameters.method.eq('quadrant_model')]
    retreat=q[['retreat_x_module','retreat_y_module']].median().to_numpy()
    # Korean overview card for inline delivery.
    card=Image.new('RGB',(1500,1100),'#f4f7fa');draw=ImageDraw.Draw(card)
    regular=lambda size:ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',size)
    bold=lambda size:ImageFont.truetype('C:/Windows/Fonts/malgunbd.ttf',size)
    draw.text((55,35),'QR 특징점 추출 비교',font=bold(44),fill='#19384d')
    draw.text((55,105),'실제 341장 · 가상 카메라 24자세 × 3조건 · 원본 변경 없음',font=regular(25),fill='#536a7a')
    steps=[('1  각진 곳으로 후보 확대','01_real_candidates.png'),('2  알려진 투영 정답으로 평가','synthetic_gap_texture.png'),('3  실제 사진은 보류점으로 검증','08_material_projection.png')]
    for i,(label,path) in enumerate(steps):
        x=55+i*480
        draw.rounded_rectangle((x,175,x+450,500),radius=15,fill='white')
        draw.text((x+15,190),label,font=bold(23),fill='#254d66')
        im=Image.open(OUT/path).convert('RGB')
        if i==0:im=im.crop((0,0,im.width//3,im.height))
        if i==2:im=im.crop((0,0,im.width//2,im.height))
        im.thumbnail((425,235));card.paste(im,(x+(450-im.width)//2,250+(235-im.height)//2))
    draw.text((55,540),'L형 모서리에서의 비교',font=bold(29),fill='#19384d')
    draw.text((65,600),'추출 방법',font=bold(23),fill='#344f63')
    draw.text((650,600),'합성 정확도 RMS',font=bold(23),fill='#344f63')
    draw.text((1050,600),'실제 보류 잔차 중앙값',font=bold(23),fill='#344f63')
    labels={'quadrant_model':'밝기 형상 적합','subpix':'일반 부화소 보정','edge_lines':'두 경계선 교점'}
    for i,method in enumerate(['quadrant_model','subpix','edge_lines']):
        y=650+i*75
        draw.rounded_rectangle((55,y-7,1440,y+55),radius=8,fill='#e0f1e8' if i==0 else 'white')
        sr=textured[textured.method.eq(method)].iloc[0];mr=material[material.method.eq(method)].iloc[0]
        draw.text((70,y),labels[method],font=bold(27) if i==0 else regular(27),fill='#214b3a' if i==0 else '#344f63')
        draw.text((710,y),f'{sr.rms_px:.3f} px',font=bold(28),fill='#214b3a' if i==0 else '#344f63')
        draw.text((1150,y),f'{mr.median_px:.3f} px',font=bold(28),fill='#214b3a' if i==0 else '#344f63')
    draw.text((60,905),'합성: 경계 후퇴 + 재료에 고정된 질감 / 실제: 후퇴량을 학습점에서만 보정한 투영',font=regular(23),fill='#536a7a')
    draw.text((60,960),'실제 투영 잔차는 절대 정확도가 아닙니다.',font=bold(28),fill='#a15a13')
    draw.text((60,1010),'코팅 질감은 영향 시험용이며 테플론 물성을 실측해 재현한 것은 아닙니다.',font=regular(23),fill='#536a7a')
    card.save(OUT/'summary.png')
    findings=f'''<ul><li>질감이 있는 합성 L형에서는 <b>{html.escape(winner['method'])}</b>가 가장 낮은 RMS <b>{winner.rms_px:.3f}px</b>를 보였습니다. 이것은 알려진 가상 조건의 정밀화 결과입니다.</li>
    <li>실제 사진의 경계 후퇴 보정 투영에서는 <b>{html.escape(materialwinner['method'])}</b>의 보류 잔차 중앙값이 <b>{materialwinner.median_px:.3f}px</b>였습니다. 추출점의 절대 오차가 아닙니다.</li>
    <li>밝기 형상 적합의 유효 경계 후퇴 추정 중앙값은 가로 {retreat[0]:.3f}, 세로 {retreat[1]:.3f} module입니다. 인쇄/코팅과 추출기 편향이 섞인 값이므로 실제 두께로 해석하지 않습니다.</li>
    <li>X형은 가상 대칭 중심의 위치 잔차가 작았지만 기존 밝기 적합 진단에서는 모두 탈락했습니다. 작은 투영/반복 잔차만으로 X형의 정확도를 주장할 수 없습니다. 같은 편향이 반복돼도 이런 잔차는 작아질 수 있습니다.</li>
    <li>일반 모서리는 안정적인 물리 ID와 대응시켜 추가하는 것이 좋습니다. 표면 질감의 자유 검출점은 많이 검출돼도 설계 도형 위 위치를 알 수 없어 이번 정답 평가에 넣지 않았습니다.</li></ul>'''
    freecoverage=free.matched_L.sum()/free.candidate_L.sum()
    captions=[('00_evaluation_pipeline.png','평가를 두 갈래로 나눴습니다. 합성 영상은 정답을 알고, 실제 사진은 독립된 정답이 없어 보류점 일관성을 측정합니다.'),
        ('01_real_candidates.png','X형 교점뿐 아니라 L형 모서리 전체와 자유 검출 후보를 표시했습니다. 자유 검출에서 생긴 점 모두가 정확한 QR 모서리는 아닙니다.'),
        ('02_raw_corner_closeups.png','원본 픽셀 확대. X형의 틈과 L형 경계, 표면 질감이 함께 보입니다. 노란 표시는 초기 격자 예측으로 실제 정답이 아닙니다.'),
        ('03_edge_intersection.png','L형 모서리의 두 변에서 여러 밝기 프로파일을 읽고, 각각 직선을 적합해 교점을 구합니다.'),
        ('synthetic_clean.png','가상 카메라의 투영과 광학 블러·픽셀 면적 적분으로 만든 매끈한 표면입니다.'),
        ('synthetic_gap_texture.png','검정 영역의 경계 후퇴와 상관된 질감을 추가한 합성 영상입니다. 테플론 BRDF를 측정해 복제한 영상은 아닙니다.'),
        ('05_synthetic_accuracy.png','알고 있는 투영 정답과의 거리입니다. 유효한 추출만의 분포이므로 실패율/coverage도 함께 확인해야 합니다.'),
        ('06_real_consistency.png','실제 사진에서는 해당 점을 제외하고 투영을 맞춘 뒤 잔차를 계산했습니다. 이 값은 절대 위치 정확도가 아닙니다.'),
        ('07_spatial_quality.png','위치별 중앙 잔차 지도입니다. 프레임마다 좋은 점만 고른 결과가 아니라 전체 위치를 표시했습니다.')]
    gallery=''.join(f'<figure><img loading="lazy" src="{p}"><figcaption>{html.escape(c)}</figcaption></figure>' for p,c in captions)
    gallery+= '<figure><img loading="lazy" src="08_material_projection.png"><figcaption>L형 경계 후퇴는 학습점에서만 보정했습니다. 보류점 일관성 진단이며 독립적인 물리 정답은 아닙니다.</figcaption></figure>'
    # Also write a readable markdown summary with all raw-data links.
    md=f'''# 모서리 추출 방법 비교: 투영 정답과 실제 사진

2026-10-26. 실제 341장, 가상 카메라 24자세 × 3조건. 원본 이미지는 변경하지 않았다.

## 주요 결과

질감 조건의 합성 L형에서 {winner['method']} RMS={winner.rms_px:.6f}px였다. 경계 후퇴를 학습점에서만 보정한 실제 L형의 투영 잔차 중앙값은 {materialwinner['method']}에서 {materialwinner.median_px:.6f}px였다. 실제 잔차는 추출점의 절대 정확도가 아니다. X형의 투영/반복 잔차는 작았지만 기존 밝기 적합 진단은 전부 탈락했다. 따라서 작은 기하 잔차만으로 X형이 더 정확하다고 주장하지 않는다.

## 평가에서 구분한 것

사용자 제안대로 이상적인 평면 위에 가상 특징점을 두고 핀홀 카메라 K[R|t]로 투영했다. 이 투영 위치와 검출 위치의 유클리드 거리를 합성 영상의 정확도로 사용했다. 실제 사진은 K·왜곡·실제 자세·인쇄물 모서리의 실측 좌표가 없으므로 같은 수준의 정답을 만들 수 없다. 실제 사진은 공간 블록 5-fold 보류 투영 잔차와 반복 일관성으로 평가했다. 평가할 점은 그 점의 homography 학습에서 제외했다.

각 QR에 개별 homography를 사용했으며 세 QR 사이 실제 간격이나 단일 보드 물리 좌표를 새로 알아냈다는 뜻이 아니다. 왜곡·판의 비평면성·인쇄 편차·코팅 경계 편향도 실제 잔차에 포함된다. 이 자료만으로 카메라 보정을 새로 추정해 정답으로 쓰지 않았다.

## 후보와 추출

QR 모듈 배치를 복원해 고정 ID를 붙였다. L형 모서리 630개, X형 가상 교점 125개다. L형에는 볼록 모서리, 오목 모서리, finder 경계가 포함된다. 기존 offaxis 분석도 이미 이 L630을 썼으므로, 새로 630개를 처음 발견했다고 주장하지 않는다. 운영 브랜치의 교점 중심 방식과 기존 분석을 구분한다.

자유 Shi–Tomasi 검출도 ROI 전체에서 수행했다. 고정 모듈 후보와 3px 이내 일대일 대응된 L 후보 비율은 {freecoverage:.2%}였다. 대응이 안 된 후보에는 배경/코팅 질감/라벨 외곽의 유효한 모서리가 함께 있을 수 있어 모두 오검출이라고 부르지 않는다. 다만 알려진 물리 ID가 없으므로 위치 정답 평가에는 포함하지 않았다.

비교 방법은 3×3 반창 cornerSubPix, 두 경계 프로파일에서 구한 직선 교점, 기존 9×9 Gaussian 사분면 밝기 적합이다. 기존 모델의 실제 사진 추출값은 저장된 v2 값을 재사용했고 원본 해시를 341장 확인했다. 나머지 실제 추출은 새로 실행했다. L형은 동일 ID의 공통 유효점에서도 비교하며 X형은 위치 집합이 달라 직접적인 개선 배수로 해석하지 않는다. 전처리 평활화는 하지 않았다. 경계 알고리즘의 미분/프로파일 보간은 추정 과정이다.

## 합성 영상의 정답 정의

카메라 초점거리 450px, 주점 (128,128), QR 한 모듈=가상 길이 1, 거리 48~54, 회전벡터 각 성분의 난수 범위를 코드에 고정했다. 실제 장비 보정값이 아니다. 입력 도형은 모듈당 64샘플, 센서 픽셀은 6×6 면적 적분으로 생성했다. 비등방 Gaussian PSF, 1.5 gray-level 센서 잡음, 인쇄 경계 후퇴 2/64 module, 상관된 질감 조건을 포함한다. 추출기의 해석적 Gaussian 식 자체로 영상을 만들지 않았다.

인쇄 경계 후퇴가 있으면 L형의 실제 재료 모서리와 이상적인 격자점은 다른 위치다. `error_px`는 후퇴한 실제 재료 모서리 투영과의 거리, `ideal_grid_error_px`는 원래 격자 투영과의 거리로 둘 다 저장했다. X형은 검정이 분리되면 실제로 존재하지 않는 가상 격자 교점이므로 X 결과를 실제 재료 모서리의 오차와 동일한 의미로 읽지 않는다.

테플론 코팅은 사용자 제공 정보다. 합성 질감은 영향 확인용이며 표면 반사모델을 실측하거나 테플론 물성을 맞춘 것은 아니다. 24가상 자세는 3조건에서 재사용하고 초기값 난수도 같게 유지했다. 각 자세의 L48/X24 위치는 성능을 보기 전 격자 순서에서 균등 선택했다. 검출 위치의 초기 추측은 정답 격자에 ±0.45px를 더해 제공했으므로 합성 정확도는 정밀화 단계 평가다. 자유 검출 전체 성능과 혼동하지 않는다.

## 결과

상세 수치는 `synthetic_summary.csv`, `real_summary.csv`, `repeatability.csv` 및 [시각 보고서](report.html)에 있다. 점별 CSV에는 실패와 모든 보류 잔차를 남겼다. 실제 사진 수십만 점을 독립 실험 수로 보지 않도록 프레임/QR 집계도 제공한다. 합성 자세와 원본 프레임은 범위가 제한돼 있으므로 전체 장비 조건의 일반 성능으로 일반화하지 않는다.

**유효 판정 주의:** 본 위치 비교의 valid는 유한 좌표·초기값 대비 3.6px 범위 등의 기하 조건이다. 기존 광학 적합의 원래 accepted와 다르다. 기존 L형 accepted는 96.24%, X형은 0%이며 X형의 픽셀 밝기 RMSE는 약 25로 원래 기준 15를 모두 초과했다. 유효한 좌표와 정상적인 광학 형상 적합을 혼동하지 않는다. X형의 작은 기하 잔차는 같은 편향이 반복되는 상황에서도 나올 수 있다. 원래 진단은 original_photometric_diagnostics.csv에 보존했다.

### 인쇄 경계 후퇴를 고려한 보류 투영

각 QR/방법/프레임의 학습점만으로 H의 8변수와 가로/세로 유효 후퇴량 2변수를 robust 적합하고, 공간 블록을 보류해 예측했다. L의 볼록/오목 모양에 따라 후퇴 부호를 정했다. 원래 격자 투영 잔차와 별도로 material_projection.csv.gz에 남겼다. 이 모델도 실제 K·렌즈 왜곡·광학/코팅 물성을 알아낸 것이 아니며, 추출기의 모양별 편향을 흡수할 수 있는 기술적 모델이다. 실제 모서리 정답을 제공하지는 않는다.

밝기 형상 적합의 유효 후퇴량 중앙값은 ({retreat[0]:.6f}, {retreat[1]:.6f}) module이다. 무잡음 투영 복원과 해석적 Jacobian 수치 검산을 수행했다. 두 모델의 잔차가 달라진 것을 추출 정확도가 그만큼 향상됐다고 해석하지 않는다.

## 문헌과 적용 범위

- [AprilTag 2, Wang & Olson (2016)](https://april.eecs.umich.edu/media/pdfs/wang2016iros.pdf): 경계 법선 방향의 gradient 가중 위치를 구하고 직선 적합 후 교점으로 모서리를 정밀화하는 접근을 참고했다. 이번 짧은 QR 내부 L변 프로파일 알고리즘은 논문 전체 구현이나 AprilTag 검출기가 아니다.
- [Canny/Devernay subpixel edges, Grompone von Gioi & Randall (2017)](https://www.ipol.im/pub/art/2017/216/): 부화소 경계 위치 추정의 근거로 참고했다. 제공 소스의 복제나 그 정확한 알고리즘 재현은 아니다.
- [Zhang (2000), A Flexible New Technique for Camera Calibration](https://www.microsoft.com/en-us/research/publication/a-flexible-new-technique-for-camera-calibration/): 평면 패턴과 투영, 렌즈 왜곡/비선형 보정을 구분하는 근거다. 본 실험에서 실제 K, 왜곡을 보정했다는 뜻이 아니다.
- [OpenCV feature detection documentation](https://docs.opencv.org/4.13.0/dd/d1a/group__imgproc__feature.html): 자유 검출과 cornerSubPix 비교군의 공식 정의.

## 재현과 파일

numpy,pandas,scipy,opencv-python-headless,matplotlib,qrcode,Pillow 필요. `tool/offaxis/corner_benchmark.py --views 24`, 이어서 `tool/offaxis/corner_benchmark_report.py` 실행. 추출 원시값을 재사용할 때는 `--reuse-real --reuse-synthetic`. Python 경로와 라이브러리 버전/소스 해시는 manifest에 기록했다.

- report.html: 오프라인 시각 보고서, 원본 사진 위 점/보류 투영 위치 선택·확대·오차 확인.
- real_points.csv.gz: 실제 사진 모든 후보/방법의 추출값·실패·보류 투영 잔차.
- synthetic_points.csv: 합성 투영 정답·격자 정답·추출값·오차·실패.
- synthetic_cameras.json: 알려진 가상 K,R,t,H,PSF와 인쇄 후퇴량.
- free_detection.csv: 자유 모서리 검출과 고정 ID 대응 범위.
- real_frame_summary.csv, repeatability.csv: 프레임/QR 집계 및 동일 명령 좌표의 상대변형 반복 잔차. 반복은 정지한 장비의 센서 잡음만이 아니라 기구 재현오차도 포함한다.
- manifest.json: 입력/소스 해시, 가정, 수치 검산, 라이브러리 버전.
- material_projection_summary.csv, material_projection.csv.gz: 후퇴 보정 투영의 보류점 결과. corner_material_projection.py 실행 후 report 생성.

커밋·push·운영 알고리즘 교체는 하지 않았다.
'''
    (OUT/'README.md').write_text(md,encoding='utf-8')
    page='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>QR 모서리 추출 비교</title><style>
    :root{font-family:system-ui,"Malgun Gothic",sans-serif;color:#203344;background:#f3f6f8}body{max-width:1200px;margin:auto;padding:28px}h1{font-size:32px}h2{margin-top:42px}p,li{line-height:1.8}.lead{background:#e4f3eb;padding:20px;border-radius:12px}.note{background:#fff2d7;padding:15px;border-radius:8px}figure{margin:24px 0;background:white;padding:14px;border-radius:12px}figure img{max-width:100%;display:block;margin:auto}figcaption{padding-top:12px;color:#4a6173}table{border-collapse:collapse;background:white;width:100%;font-size:14px}td,th{padding:10px;border-bottom:1px solid #ddd;text-align:right}td:first-child,th:first-child{text-align:left}.scroll{overflow-x:auto}button,select,input{font:inherit}select{max-width:96%;padding:8px;margin:4px}.viewer{background:white;border-radius:12px;padding:16px}canvas{width:100%;max-height:620px;image-rendering:pixelated;border:1px solid #aaa;touch-action:none}#detail{padding:10px;background:#edf4f8;min-height:48px;font-family:monospace;white-space:pre-wrap}a{color:#12699b}.links{display:flex;gap:20px;flex-wrap:wrap}.small{font-size:14px;color:#556b7b}
    </style><body><h1>QR 모서리 추출: 교점을 넘어 실제 경계까지</h1>
    <p class="lead">원본 341장과 알려진 가상 카메라 24자세 × 3조건 영상으로, X형 교점과 L형 모서리의 추출을 비교했습니다. 테플론 코팅 정보에 따라 재료에 고정된 질감을 고려했습니다.</p>
    <p class="note">합성 영상의 거리 오차는 정답 대비 정확도입니다. 실제 사진의 보류 투영 잔차는 기하 일관성입니다. 실제 K·왜곡·판의 실측 좌표가 없어 두 값을 같은 정확도로 해석하면 안 됩니다.</p>
    <h2>원본 위에서 직접 확인</h2><p>사진·방법·모서리 유형을 선택하세요. 배율을 올리면 스크롤로 이동할 수 있고, 점 근처에 마우스를 올리면 고정 ID와 잔차를 볼 수 있습니다. <b>주황 원은 실제 추출 위치, 청록 십자는 그 점을 제외하고 맞춘 평면 투영 위치</b>입니다.</p>
    <div class="viewer"><select id="frame"></select><select id="method"><option value="quadrant_model">밝기 형상 적합</option><option value="subpix">cornerSubPix</option><option value="edge_lines">두 경계선 교점</option></select><select id="kind"><option value="L">L형 모서리</option><option value="X">X형 교점</option></select><label>배율 <input id="zoom" type="range" min="1" max="6" value="1" step=".25"></label><div style="overflow:auto;max-height:650px"><canvas id="canvas"></canvas></div><div id="detail">점을 선택해 오차를 확인하세요. 청록 십자는 실측 정답이 아닙니다.</div></div>
    <h2>합성 영상: 투영 정답과의 거리</h2><p>L형 재료 모서리 기준, px 단위. coverage는 시도한 점 중 유효 추출 비율입니다. 중앙값만 좋아도 실패가 많으면 유리한 방법으로 판단할 수 없습니다.</p><div class="scroll">SYN_TABLE</div>
    <h2>실제 사진: 같은 L형 위치에서의 비교</h2><p>세 방법 모두 기하 조건이 유효하고 보류 투영이 가능한 동일 ID만 비교했습니다. 이 표에는 인쇄 편차·코팅·렌즈·평면 모델 오차도 남습니다.</p><p class="note">위치 비교의 valid는 광학 적합 통과와 다릅니다. 기존 밝기 적합의 원래 통과율은 L형 96.24%, X형 0%입니다. X형은 밝기 잔차가 커서 모두 탈락했으므로, 낮은 기하 잔차만으로 정확하다고 판정할 수 없습니다.</p><div class="scroll">REAL_TABLE</div>
    <h2>전체 과정과 결과 그림</h2>GALLERY
    <h2>실제 L형: 인쇄 경계와 격자를 구분한 투영</h2><p>학습점에서만 평면 투영 8변수와 가로/세로 경계 후퇴 2변수를 추정했습니다. 평가할 점은 포함하지 않았습니다. 추출기 편향도 이 값에 섞일 수 있어, 이 결과를 실측 정답이라고 부르지 않습니다.</p><div class="scroll">MATERIAL_TABLE</div>
    <h2>해석 및 다음 선택</h2><div id="findings">FINDINGS</div>
    <p>이번 결과만으로 Y·θ 한 틱 정확도 또는 5~10회 원점 도달을 주장할 수 없습니다. 추출 방법의 위치 편향과 반복성이 달라지면 기존 축 모델도 입력에 맞춰 다시 검증해야 합니다.</p>
    <h2>자료와 참고 문헌</h2><div class="links"><a href="README.md">분석 설명</a><a href="synthetic_summary.csv">합성 수치</a><a href="real_summary.csv">실제 수치</a><a href="repeatability.csv">반복 일관성</a><a href="manifest.json">재현 정보</a></div>
    <ul><li><a href="https://april.eecs.umich.edu/media/pdfs/wang2016iros.pdf">Wang & Olson, AprilTag 2 (2016)</a> — 경계 샘플과 직선 교점 접근 참고</li><li><a href="https://www.ipol.im/pub/art/2017/216/">Grompone von Gioi & Randall, Canny/Devernay subpixel edges (2017)</a> — 부화소 경계 검출 참고</li><li><a href="https://www.microsoft.com/en-us/research/publication/a-flexible-new-technique-for-camera-calibration/">Zhang, Camera Calibration (2000)</a> — 평면 투영과 보정의 구분</li></ul>
    <script>const DATA=VIEW_DATA;const canvas=document.getElementById('canvas'),ctx=canvas.getContext('2d'),frame=document.getElementById('frame'),method=document.getElementById('method'),kind=document.getElementById('kind'),zoom=document.getElementById('zoom'),detail=document.getElementById('detail');let img=new Image(),current=null,points=[];let mode=document.createElement('select');mode.innerHTML='<option value="material">경계 후퇴 보정 투영</option><option value="grid">원래 격자 투영</option>';kind.after(mode);mode.onchange=redraw;
    DATA.forEach((x,i)=>{let o=document.createElement('option');o.value=i;o.textContent=x.path;frame.appendChild(o)});
    function redraw(){if(!current||!img.complete)return;const z=+zoom.value;canvas.width=current.width*z;canvas.height=current.height*z;canvas.style.width=canvas.width+'px';canvas.style.maxHeight='none';ctx.imageSmoothingEnabled=false;ctx.drawImage(img,0,0,canvas.width,canvas.height);points=current.points.filter(p=>p.method===method.value&&p.kind===kind.value).map(p=>({...p,dx:mode.value==='material'&&p.mdx!==null?p.mdx:p.dx,dy:mode.value==='material'&&p.mdy!==null?p.mdy:p.dy,error:mode.value==='material'&&p.merror!==null?p.merror:p.error}));for(const p of points){const x=p.x*z,y=p.y*z;ctx.strokeStyle=p.valid?'#ff7b15':'#ff3158';ctx.lineWidth=1;ctx.beginPath();ctx.arc(x,y,Math.max(2,1.5*z),0,7);ctx.stroke();if(p.dx!==null){let px=(p.x-p.dx)*z,py=(p.y-p.dy)*z;ctx.strokeStyle='#04cbe2';ctx.beginPath();ctx.moveTo(px-2,py);ctx.lineTo(px+2,py);ctx.moveTo(px,py-2);ctx.lineTo(px,py+2);ctx.moveTo(px,py);ctx.lineTo(x,y);ctx.stroke()}}}
    function select(){current=DATA[+frame.value];img.onload=redraw;img.src=current.image}frame.onchange=select;method.onchange=redraw;kind.onchange=redraw;zoom.oninput=redraw;
    canvas.onmousemove=e=>{let r=canvas.getBoundingClientRect(),x=(e.clientX-r.left)*canvas.width/r.width/(+zoom.value),y=(e.clientY-r.top)*canvas.height/r.height/(+zoom.value),best=null,d=Infinity;for(const p of points){let dd=Math.hypot(x-p.x,y-p.y);if(dd<d){d=dd;best=p}}if(best&&d<8){detail.textContent=`${best.qr}:${best.id} | ${best.kind} | ${best.method} | valid=${best.valid}\nheld-point residual: ${best.error===null?'not scored':best.error.toFixed(4)+' px'} | dx=${best.dx} dy=${best.dy} (not absolute truth)`}};select();</script></body></html>'''
    # Figures and viewer are embedded so the HTML can be opened on its own.
    import re
    def embed(match):
        path=OUT/match.group(1)
        return 'src="data:image/png;base64,'+base64.b64encode(path.read_bytes()).decode()+'"'
    gallery=re.sub(r'src="([^"\n]+\.png)"',embed,gallery)
    page=page.replace('SYN_TABLE',synhtml).replace('REAL_TABLE',realhtml).replace('MATERIAL_TABLE',materialhtml).replace('FINDINGS',findings).replace('GALLERY',gallery).replace('VIEW_DATA',json.dumps(view,ensure_ascii=False).replace('</','<\\/'))
    (OUT/'report.html').write_text(page,encoding='utf-8')
    print('Report:',OUT/'report.html');print('Free L coverage',freecoverage);print(repeat.to_string(index=False))


if __name__=='__main__':main()
