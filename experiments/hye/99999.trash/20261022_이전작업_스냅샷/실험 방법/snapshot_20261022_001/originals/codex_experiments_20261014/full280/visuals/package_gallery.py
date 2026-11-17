from pathlib import Path
import base64,csv,io,json,zipfile
from PIL import Image

BASE=Path(__file__).resolve().parent
DEST=Path('C:/Users/SSAFY/.codex/visualizations/2026/10/14/01a13940-1576-7592-a2e8-b575c93d7993')
source=json.loads((BASE/'compact.json').read_text(encoding='utf8'))
def uri(file):
    with Image.open(file) as image:
        image=image.convert('L')
        image.thumbnail((736,300))
        buffer=io.BytesIO();image.save(buffer,'JPEG',quality=52,optimize=True)
    return 'data:image/jpeg;base64,'+base64.b64encode(buffer.getvalue()).decode('ascii')

data=dict(points=[[round(x,3),round(y,3)] for x,y in source['points']],regions=source['regions'],crop=source['crop'],center=uri(source['center_jpeg_path']),samples=[])
for s in source['samples']:
    r={k:s[k] for k in ['file','axis','value','pred_axis','pred_value','error_Y','error_T','mean_dx','mean_dy','arrow_scale']}
    r['D']=[[round(x,6),round(y,6)] for x,y in s['D']]
    r['photo']=uri(s['current_jpeg_path']);data['samples'].append(r)
template=(DEST/'qr-motion-gallery-template.html').read_text(encoding='utf8')
output=template.replace('__GALLERY_DATA__',json.dumps(data,ensure_ascii=False,separators=(',',':')))
assert '__GALLERY_DATA__' not in output
fragment=DEST/'qr-motion-gallery.html';fragment.write_text(output,encoding='utf8')
assert fragment.stat().st_size<1_000_000,fragment.stat().st_size

fields=['index','group','file','axis','value','pred_axis','pred_value','error_Y','error_T','mean_dx','mean_dy','arrow_scale']
with (BASE/'selected_20.csv').open('w',encoding='utf-8-sig',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows({k:s[k] for k in fields} for s in source['samples'])
readme='''특징점 이동 시각화 20장

01–10: Y/T 각각 명령값 절댓값이 가장 작은 5장. 같은 크기이면 + 방향 우선.
11–20: 앞의 사진을 제외한 나머지에서 Y/T 각 5장 무작위 선택(seed 20261014).

윗부분: 같은 좌표 범위의 센터 사진과 현재 사진, QR 내부 특징점 205개.
아랫부분: 센터 점 위치에서 출발한 실측 변위 벡터. 화살표만 표시 배율을 곱함.
화살표 배율은 그림마다 다를 수 있음. 사진 사이의 길이 비교는 실제 dx/dy 수치로 할 것.
픽셀 좌표는 오른쪽 +x, 아래쪽 +y. Y/T 명령 부호와 픽셀 방향은 서로 다른 좌표계임.
표: QR별 평균 픽셀 변위. 상단: 명령 정답, 모델 예측, Y/T 절대오차.

20장 모두 원본 이미지에서 특징점을 다시 추출했고 기존 좌표 캐시와 최대 차이 0.0px.
고정된 최종 모델의 기존 보정 데이터 재실행 결과이며 새 독립 사진의 성적은 아님.
T는 제공된 파일명 단위, Y는 mm. 허용오차 Y/T 각각 0.09.
'''
(BASE/'읽는방법.txt').write_text(readme,encoding='utf-8-sig')
archive=BASE.parent/'QR_특징점이동_20장.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
    for s in source['samples']:
        p=Path(s['png_path']);z.write(p,p.name)
    for name in ['읽는방법.txt','selected_20.csv','summary.json','data.json']:
        z.write(BASE/name,name)
print(json.dumps({'fragment':str(fragment),'bytes':fragment.stat().st_size,'archive':str(archive),'png_count':20},ensure_ascii=False))
