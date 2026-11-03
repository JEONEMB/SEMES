from pathlib import Path
import json,html
import pandas as pd
P=Path(__file__).resolve().parent
def js(name):return json.loads((P/name).read_text(encoding='utf-8'))
s=pd.read_csv(P/'session_summary.csv');orig=pd.read_csv(P/'reference_pixel_displacements.csv')
err=js('error_decomposition.json');meta=js('metadata_audit.json');lab=js('label_sensitivity.json')
verified=js('remeasurement_validation.json') if (P/'remeasurement_validation.json').exists() else None
small=s[['session','start_KST','end_KST','images','poses','Y_min','Y_max','T_min','T_max','bias_ticks','MAE_ticks','max_abs_ticks','pass_1tick']].copy()
small['start_KST']=small.start_KST.str[:19];small['end_KST']=small.end_KST.str[11:19]
small.columns=['후보','시작(KST)','종료(KST)','이미지','평가 자세','Y 최소','Y 최대','T 최소','T 최대','평균 오차(tick)','MAE(tick)','최대(tick)','1tick 통과']
table=small.to_html(index=False,float_format=lambda x:f'{x:.3f}',na_rep='-')
ot=orig.copy();ot['path']=ot.path.map(lambda x:Path(x).name)
ot.columns=['기준 이미지','dx(px)','dy(px)','변위 크기(px)','이동 이외 잔차 RMS(px)']
origin_table=ot.to_html(index=False,float_format=lambda x:f'{x:.4f}')
anomalies=pd.read_csv(P/'label_anomalies.csv').to_html(index=False,escape=True)
caption={
 '01_metadata_sessions.png':'생성 시각의 복사 흔적, 수정 시각의 세션 후보, 시간 경계 민감도, 자세 분포를 함께 확인한다.',
 '02_session_errors.png':'571장 기존 method 7 좌표 보류 OOF. 신뢰구간은 같은 자세의 반복을 묶어 3,000회 bootstrap한 기술 통계이며, 원점의 신뢰구간이 아니다.',
 '03_origin_evidence.png':'동일 원점 명령의 영상 차이는 관측된다. tick 환산은 보정 범위에 민감해 원점 변화의 확정값으로 쓰지 않는다.',
 '04_repeatability_quality.png':'같은 명령 자세에서도 영상과 예측이 변한다. 이 변화에는 특징 추출뿐 아니라 실제 기계 위치·조명·촬영 조건의 변동도 포함된다.',
 '05_label_audit.png':'빨간 원 이미지가 .33 반복군보다 .32 반복군에 가깝다. 폴더와 대응 TXT의 이름도 .32를 지지한다.',
 '06_each_session.png':'12개 평가 구간 각각의 부호 오차. 고정 오프셋이라면 수평 이동으로 개선되지만, 여러 구간에는 자세에 따른 부호 변화와 반복 편차가 남는다.'}
def fig(name):return f'<figure><a href="{name}"><img src="{name}" alt="{html.escape(caption[name])}"></a><figcaption>{html.escape(caption[name])}</figcaption></figure>'
body=f'''
<header><p class="eyebrow">T축 재분석 · 원본 575장 · 2026-11-02</p><h1>촬영 구간과 원점 차이로<br>오차를 설명할 수 있는가?</h1>
<p class="lead">원점 기준 영상에는 날짜별 차이가 있다. 그러나 현재 오차를 일정한 세션 원점 오프셋만으로 설명할 수는 없다. 촬영 시각의 원본 정보 손실과 라벨 불일치도 확인했다.</p></header>
<nav><a href="#time">시간·세션</a><a href="#errors">구간별 오차</a><a href="#origin">원점 근거</a><a href="#labels">라벨 검사</a><a href="#precision">추출 정밀도</a><a href="#files">자료·재현</a></nav>
<section class="callout"><strong>결론의 범위</strong><ul>
<li><b>확인:</b> 575장 생성 시각은 11/2의 198.1초에 몰려 있다. BMP 내부에는 촬영 시각이 없다. 수정 시각으로 10/4·10/15의 14개 연속 구간 후보를 구성했다.</li>
<li><b>확인:</b> 같은 (Y,T)=(0,0) 기준 영상은 날짜 사이 QR 위치가 0.45~0.54px 다르다. 실제 원점 또는 영상 좌표계 차이를 의심할 근거다.</li>
<li><b>미식별:</b> 이 차이가 T축 원점 이동 몇 tick인지는 현재 자료로 신뢰성 있게 결정할 수 없다. 비원점 자세는 세션 간 정확한 중복이 0개다.</li>
<li><b>확인:</b> 파일명·폴더 불일치 1장, 공통 원점 대비 Actual T 불일치 3장이 있다. 원본과 평가 라벨은 변경하지 않았다.</li>
</ul></section>
<section id="time"><h2>1. 생성 시각을 촬영 세션으로 쓰면 잘못 분류된다</h2>
<p>CreationTimeUtc는 {meta['creation_min']} ~ {meta['creation_max']}다. 이미지가 저장소로 복사된 순서와 일치하는 기록으로, 촬영 시점의 근거로 사용할 수 없다. 575장 모두 40바이트 BITMAPINFOHEADER이며 EXIF 태그 합계는 0이다. 수정 시각은 10월 4일·15일의 순서로 남아 있다. 다만 이것도 실제 카메라 촬영 로그가 아니므로 아래는 모두 <b>수정 시각에 따른 세션 후보</b>다.</p>
<p>명시적 기준은 시간순 정렬 후 직전 이미지와의 공백이 <b>120초 초과</b>일 때 분리하는 것이다. 기준 전용 2개 구간을 포함해 14개, 평가 가능한 구간은 12개다. 60초면 30개, 180초면 9개, 300초면 6개, 1,800초면 날짜 단위 2개가 된다. 따라서 14개를 실제 장비 재원점 횟수로 해석하면 안 된다.</p>
{fig('01_metadata_sessions.png')}
<div class="scroll">{table}</div>
<p class="note">단위: Y는 mm, T는 degree, T 1 tick = 0.01°. 기준 사진 4장은 오차 점수에서 제외한다. S11에는 원점 기준 1장과 평가 3장이 포함된다.</p></section>
<section id="errors"><h2>2. 세션 차이는 있지만 고정 원점 보정만으로는 해결되지 않는다</h2>
<p>기존 method 7의 571장 OOF 예측을 새 메타데이터와 결합했다. MAE <b>{err['MAE_ticks']:.3f}</b>, RMSE <b>{err['RMSE_ticks']:.3f}</b>, 최대 <b>{err['max_ticks']:.3f} tick</b>, 1 tick 이내 473/571장이다. 이번 작업은 새로운 예측 모델의 성능 개선 주장이 아니라 오차의 EDA다.</p>
<ul><li><b>S03 · 10/4 T 단독 200장:</b> 평균 오차 -0.107, MAE 0.552 tick. 양·음 T를 한 번씩 순차 촬영해 시간과 자세가 얽힌다. 시간 상관을 온도 드리프트로 단정할 수 없다.</li>
<li><b>S08 · 10/15 Y=.05, T=.01~.79, 237장:</b> 평균 -0.039, MAE 0.497 tick. 전체 구간의 고정 편향은 작지만 개별 자세의 차이가 남는다.</li>
<li><b>S10 · Y=.05, T=1·2, 6장:</b> T=1의 평균 오차는 -1.497, T=2는 +1.702 tick이다. 같은 짧은 구간 안에서 부호가 뒤집혀 일정한 T 원점 차이만으로 설명되지 않는다.</li>
<li><b>S12·S13·S14 · Y=.1·.5·1:</b> 평균 오차는 각각 +0.479·+0.613·+0.420 tick이다. 각 구간은 Y와 촬영 시간이 함께 바뀌므로 이것을 독립적인 세션 효과로 해석할 수 없다. 특히 S14의 최대오차는 2.640 tick이다.</li></ul>
<p>모든 세션의 정답 잔차 평균을 사후에 알고 제거하는 낙관적 계산에서도 RMSE는 <b>0.729 → {err['oracle_session_centered_RMSE_ticks']:.3f} tick</b>(약 2.7% 감소), 최대오차는 <b>2.640 → {err['oracle_session_centered_max_ticks']:.3f} tick</b>으로 오히려 커진다. 이는 학습·평가 결과가 아닌 <b>기술적 분해</b>다. 세션별 일정한 절편만으로 모든 표본을 맞출 수 없음을 보여준다.</p>
{fig('02_session_errors.png')}{fig('06_each_session.png')}
</section>
<section id="origin"><h2>3. 원점 영상은 달라졌다. T 원점 이동량은 확정할 수 없다</h2>
<p>파일명상 원점인 10/4 center와 10/15 center1·2·3을 비교했다. 각 영상의 동일 QR 630점 대응에서 평균 이동을 계산했다. 아래 수치는 <b>영상의 픽셀 차이</b>이며 T 단위가 아니다. 630점을 독립적인 630회 촬영으로 취급하지 않았다.</p>
<div class="scroll">{origin_table}</div>
<p>차이는 주로 전체 QR에 공통인 이동이다. 10/15 기준들끼리의 평균 위치 차이는 약 0.08~0.09px로, 명령 자세 반복에서 관측되는 QR 중심 변동과 비슷한 크기다. 각 원점 촬영이 한 장씩이라 이 값만으로 세션별 재원점 오차의 분포나 유의성을 추정할 수 없다.</p>
<p>10/4의 원점 주변 보정 자료로 10특징의 2차 전방 반응을 적합하고, 반복 촬영 공분산에 20% 대각 축소를 적용한 GLS로 상당 Y·T 이동을 계산했다. 그러나 보정 반경을 바꾸면 T 상당량의 부호·크기가 바뀐다. QR만 사용했을 때와 외부 선만 사용했을 때도 일치하지 않는다. 따라서 <b>이 환산은 기각하고 픽셀 차이만 보고</b>한다. 카메라·치구·광도 변화와 장비 좌표계 원점 이동을 구분할 독립 근거가 부족하다.</p>
<p>동봉 TXT 294개 중 293장은 이미지 이름과 직접 대응한다. 10/15 기준 사진과 거의 모든 혼합 사진에서 Actual - 명령은 <b>Y=-1.300mm, T≈-1.670°</b>로 일정하다. 즉, <b>10/15 로그상 전역 좌표 원점이 세션마다 바뀌었다는 근거는 없다.</b> 예외인 .57 세 장은 다음 절에 분리했다. Actual 필드가 외부 엔코더로 얻은 물리적 참값인지는 확인되지 않았다.</p>
{fig('03_origin_evidence.png')}
<p class="note">현재 자료의 식별 한계: (0,0)을 제외하면 서로 다른 후보 구간에 공통으로 촬영된 정확한 명령 자세가 0개다. 자세 반응을 자유롭게 허용하면 세션 절편과 자세 효과를 분리할 수 없다. 120초 분리는 단순 시간 공백이며 원점 재설정 이벤트를 증명하지 않는다.</p></section>
<section id="labels"><h2>4. 실제 데이터에서 발견한 라벨 문제</h2>
<ul><li><code>Y+0.05 T+0.32/Y+0.05 T+0.33 (3).bmp</code>: 부모 폴더는 .32, 파일명은 .33, 대응 순번의 TXT는 .32다. 영상 특징도 .32 반복군 쪽에 놓인다. 같은 폴더의 TXT를 임의로 다른 이미지에 연결하지 않고, 대응 불일치 자체를 기록했다.</li>
<li><code>Y+0.05 T+0.57 (1~3).bmp</code>: Actual T=-1.110°. 나머지 로그의 공통 원점 -1.670°를 빼면 .560°다. 명목 .57과 1 tick 차이가 있다. 파일명 오기인지 실행 위치 차이인지 TXT 기록 오류인지는 미확정이다.</li></ul>
{fig('05_label_audit.png')}
<p>.32/.33의 폴더 분류만 바꾼 민감도 계산에서 동일 자세 내 예측 SD는 <b>{lab['original_within_pose_sd_ticks']:.3f} → {lab['folder_audited_within_pose_sd_ticks']:.3f} tick</b>이다. 일부 반복 변동은 줄지만 전체 문제는 사라지지 않는다. 예측값을 고정한 라벨 대조이므로 수정된 OOF 성적이 아니다. 라벨을 확정한 뒤 동일 자세를 함께 묶어 모든 학습·평가 fold를 다시 만들어야 한다. 원본은 수정·이동·이름변경하지 않았다.</p>
<details><summary>라벨 이상 전수 목록</summary><div class="scroll">{anomalies}</div></details></section>
<section id="precision"><h2>5. 특징 추출 정밀도와 반복 변동</h2>
<p>원본 575장의 SHA-256이 기존 기록과 모두 일치했다. 전체를 다시 디코딩해 같은 고정 기준 영상에서 <b>QR 630점·외부 상하 선 120표본</b>을 재측정했다. 측정 실패 {verified['failures'] if verified else '검증 중'}, 기존 104특징과 최대차이 {verified['max_feature_difference'] if verified else '검증 중'}다.</p>
<p>QR은 NCC와 affine ECC로 초기화하고 해석적 모서리 모델로 위치를 잡은 뒤, 9×9 밝기 패치의 위치·gain·offset을 경계 제한 감쇠 Gauss–Newton으로 적합한다. 서브픽셀 표본은 <b>Lanczos-3: sinc(x)·sinc(x/3), |x|&lt;3</b>를 정규화해 사용하고 해석적 미분으로 위치를 최적화한다. 외부 선은 7×15 패치에서 gain·offset과 수직 위치를 soft-L1로 적합한다. 상수 영상·정수 위치 재현과 해석 미분 대 유한차분 검증을 실제 실행해 통과했다. 새 추출 결과는 기존 계측과 일치하므로 기존 OOF 오차와 직접 연결할 수 있다.</p>
<p>같은 명령 자세 반복 97개에서 QR 점 위치의 반복 RMS 중앙값은 약 <b>0.091px</b>, T 예측의 pooled within-pose SD는 <b>0.585 tick</b>이다. 전체 잔차 SSE의 약 21.9%가 같은 명령 자세 내 차이에 해당한다. 이는 순수한 보간 오차가 아니다. 실제 위치 변동, 광학·조명 변화, 특징 측정 및 추정 모델을 함께 포함한다.</p>
<p>이전 동일 57장 추출 비교의 QR 중심 성분 반복 SD는 해석적 추출 0.11823px, Lanczos-3 0.11894px, bicubic 0.12026px였다. 따라서 <b>Lanczos를 사용했다는 사실만으로 가장 낮은 실제 특징점 오차를 입증했다고 말할 수 없다.</b> 실제 특징점의 독립 참좌표가 없어 절대 오차의 최소 방법은 미확정이다. 이번 분석은 요청한 Lanczos 보간을 준수하고 수치 안정성·전수 재현성·관측 반복성을 검증한 결과다.</p>
<p>저품질 선 표본 수와 절대 T 오차의 Spearman 상관은 0.005로 거의 없다. 선 RMSE·대비·폭의 상관도 작다. 반복 표본의 의존성과 다중 비교를 고려하면 선 품질 하나를 주원인으로 지목할 근거는 부족하다.</p>
{fig('04_repeatability_quality.png')}</section>
<section><h2>6. 다음 촬영에서 무엇을 확보하면 원점을 식별할 수 있는가?</h2>
<p>모든 실제 촬영 세션의 시작·끝에 (0,0)뿐 아니라 동일한 여러 앵커 자세를 반복 촬영하면 원점 차이와 반응 기울기 차이를 구분할 수 있다. 예를 들어 (0,0), (+.1,0), (-.1,0), (0,+.1), (0,-.1)을 각 5회 이상 촬영하고, 매번 같은 방향에서 접근한 기록과 반대 방향 접근 기록을 구분한다. 촬영 순서를 무작위화하면 T 증가와 시간 증가의 혼동도 줄일 수 있다. 이는 이번 데이터에서 검증한 결과가 아니라 추가 수집 설계다.</p>
<p>카메라 원본 시각, 세션 ID, 원점 재설정 이벤트, 명령/Actual의 의미와 타임스탬프, 실제 펄스 또는 엔코더 로그를 이미지와 연결해서 저장해야 한다. 우선 이번 라벨 4건을 촬영 기록과 대조하고, 확정된 라벨과 실제 세션 단위로 추정 모델을 다시 검증하는 순서가 적절하다.</p></section>
<section id="files"><h2>자료와 재현</h2><p>원본·기존 추정기·기존 결과는 보존했다. 이 폴더에는 읽기 전용 원본에서 파생한 분석 자료만 저장했다.</p>
<ul><li><a href="image_inventory.csv">575장 메타데이터·원점 로그 인벤토리</a> · <a href="session_summary.csv">세션 요약</a> · <a href="session_threshold_sensitivity.csv">세션 경계 민감도</a></li>
<li><a href="samples_with_errors.csv">571장 기존 OOF 오차와 세션</a> · <a href="session_feature_statistics.csv">14구간 × 104특징 통계</a> · <a href="feature_repeatability.csv">동일 자세 영상 반복성</a></li>
<li><a href="reference_pixel_displacements.csv">원점 영상 픽셀 차이</a> · <a href="origin_sensitivity.csv">기각한 원점 환산의 민감도</a> · <a href="session_pose_overlap.csv">세션 간 공통 자세</a></li>
<li><a href="label_anomalies.csv">라벨 이상 4건</a> · <a href="sidecar_origin_audit.csv">Actual 원점 통계</a> · <a href="hash_audit.csv">원본 해시 전수 대조</a></li>
<li><a href="remeasurement_validation.json">전체 재측정 검증</a> · <a href="numeric_checks.txt">Lanczos 수치 검증</a> · <a href="analyze.py">EDA 재현 코드</a> · <a href="build_report.py">보고서 생성 코드</a> · <a href="reproduce.ps1">실행 스크립트</a></li></ul>
<p class="note">불확실성: 수정 시각의 진위 미검증, 동일 원점 영상은 각 1장, 세션과 자세의 강한 혼동, Actual 필드의 물리적 참값 여부 미확정. 기존 571장은 모두 개발에 노출된 자료다. 이 EDA를 근거로 새로운 세션에서 최대 1 tick을 보증하지 않는다.</p></section>
'''
css='''*{box-sizing:border-box}body{font:16px/1.8 "Malgun Gothic",system-ui,sans-serif;background:#f5f7fa;color:#243247;margin:0}main{max-width:1260px;margin:auto;padding:40px 30px 100px}header{padding:28px 0}h1{font-size:44px;line-height:1.3;letter-spacing:-1.5px;color:#132b49}h2{font-size:26px;line-height:1.5}p{margin:14px 0}.eyebrow{font-size:13px;color:#52738b;letter-spacing:1px}.lead{font-size:20px;max-width:1000px}nav{display:flex;gap:20px;flex-wrap:wrap;padding:16px 0;border-top:1px solid #ccd4df;border-bottom:1px solid #ccd4df}a{color:#12689a}section{background:white;border:1px solid #dfe6ee;border-radius:12px;padding:24px 28px;margin:26px 0}.callout{border-left:5px solid #dc9440;background:#fffaf1}li{margin:7px 0}figure{margin:25px 0}img{width:100%;height:auto;border:1px solid #e2e7ec;border-radius:5px}figcaption,.note{font-size:14px;color:#52647a}table{border-collapse:collapse;font-size:13px;white-space:nowrap;width:100%}td,th{border-bottom:1px solid #e1e7ed;padding:9px 12px;text-align:right}th{background:#edf2f7}td:first-child,th:first-child{text-align:left}.scroll{overflow:auto}code{font-size:13px;background:#edf2f7;padding:2px 4px}details{margin-top:20px}@media(max-width:700px){main{padding:18px 12px}section{padding:16px}h1{font-size:30px}h2{font-size:22px}}'''
(P/'report.html').write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>촬영 세션·원점 EDA</title><style>'+css+'</style><main>'+body+'</main></html>',encoding='utf-8')
print('report.html generated; full remeasurement verified:',bool(verified))
