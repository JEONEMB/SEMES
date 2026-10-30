"""Colleague-facing explanation using completed, preserved experiment results."""
from pathlib import Path
import base64,csv,html,json,re,shutil

HERE=Path(__file__).resolve().parent
BASE=HERE.parent
PREV=BASE/'pixel_weighted_gn_20261015'
def esc(x):return html.escape(str(x))
def main():
 HERE.mkdir(exist_ok=True);assets=HERE/'assets';assets.mkdir(exist_ok=True)
 for source,name in [(BASE/'measurement/gn17_spread14_features.png','QR205_features.png'),
  (PREV/'figures/01_axis_and_sign_accuracy.png','accuracy.png'),(PREV/'figures/02_max_errors_all_predictions.png','max_error.png'),
  (PREV/'figures/03_repeated_cv_failures.png','repeated_failures.png')]:shutil.copy2(source,assets/name)
 rows=list(csv.DictReader((PREV/'comparison_summary.csv').open(encoding='utf-8-sig')))
 names={'full280':'280장 재대입','interleaved':'짝홀 교차검증','stratified5':'층화 5분할','blocked_within_session5':'시행 내 연속구간 5분할','repeated20':'반복 5분할 20회 합계'}
 methods={'baseline':'기존 GN','pixel_noise_gn':'픽셀 잡음 가중 GN'}
 table=[]
 for r in rows:
  cells=[names[r['protocol']],methods[r['method']],
   f"{100*float(r['Y_axis_sign_accuracy']):.4f}% ({r['Y_axis_sign_correct']}/{r['Y_n']})",
   f"{100*float(r['T_axis_sign_accuracy']):.4f}% ({r['T_axis_sign_correct']}/{r['T_n']})",
   f"{float(r['vector_Y_max']):.6f}",f"{float(r['vector_T_max']):.6f}",f"{r['failed']}/{r['n']}"]
  table.append('<tr>'+''.join('<td>'+esc(x)+'</td>' for x in cells)+'</tr>')
 document='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>QR 기반 Y/T 이동 추정 알고리즘 설명</title><style>
 body{margin:0;color:#17354c;background:#f6f8fa;font:16px/1.85 "Malgun Gothic",sans-serif}main{max-width:1120px;padding:32px 28px 70px;margin:auto}h1{font-size:32px;line-height:1.4}h2{font-size:24px;margin-top:44px}h3{font-size:19px;margin-top:28px}p{max-width:1050px}a{color:#087b90}blockquote{margin:24px 0;padding:18px 24px;border-left:4px solid #087f91;background:#eaf2f6;font-size:18px}figure{margin:26px 0}img{display:block;width:100%;height:auto;background:white}figcaption{font-size:14px;color:#4b6174}.formula{padding:16px;background:#e7eef4;font-family:Consolas,"Malgun Gothic",monospace;overflow-wrap:anywhere}.scroll{overflow-x:auto}table{border-collapse:collapse;background:white;width:100%;font-size:14px}th,td{padding:10px 8px;text-align:left;border-bottom:1px solid #d9e2e9;vertical-align:top}th{background:#1b3b50;color:white}tr:nth-child(even){background:#edf3f7}.results td,.results th{white-space:nowrap}.flow{list-style:none;padding:0;display:grid;grid-template-columns:repeat(4,1fr);gap:14px}.flow li{padding:14px 0;border-top:3px solid #087f91}.flow strong{display:block;font-size:18px}.flow small{display:block;font-size:14px;color:#4b6174}nav a{display:inline-block;margin-right:16px}code{font-family:Consolas,monospace}li{margin:8px 0}@media(max-width:700px){main{padding:18px 14px 40px}h1{font-size:25px}.flow{grid-template-columns:repeat(2,1fr)}blockquote{font-size:16px}}@media print{body{background:white}main{padding:0}h2{break-after:avoid}figure,table,.flow{break-inside:avoid}nav{display:none}a{color:inherit}img{max-height:24cm;object-fit:contain}}
 </style><main><h1>QR 세 개로 Y/T 축·방향·이동량을 추정하는 방법</h1><p>동료 설명용 정리 · 2026-10-15 · 직전 완료된 실험 결과 기준</p>
 <nav><a href="#goal">목표·데이터</a><a href="#flow">전체 과정</a><a href="#qr">QR 처리</a><a href="#model">모델과 예측</a><a href="#results">결과표</a><a href="#limits">결론과 범위</a></nav>
 <blockquote>센터 사진의 QR 세 개에서 고른 205개 점을 현재 사진에서 다시 추적합니다. 점들의 이동을 판 전체의 12개 변형 계수로 요약하고, 미리 보정한 Y/T 이동 곡선과 비교합니다. 관측 그룹마다 다른 기준점 후보 여섯 개에 대해 이동량을 풀고, 공분산을 반영한 오차가 가장 작은 후보의 축·부호·이동량을 출력합니다.</blockquote>
 <h2 id="goal">1. 목표와 알고 있는 데이터</h2><p><strong>실행 입력은 센터 사진과 현재 사진 두 장, 그리고 사전에 학습해 저장한 보정 모델입니다.</strong> 현재 구현은 Y 또는 T 중 한 축만 움직였다는 조건을 사용합니다.</p>
 <table><tr><th>항목</th><th>내용</th></tr><tr><td>출력</td><td>Y/T 축, 명령 부호 +/−, 연속 이동량, (Y,T) 출력 벡터</td></tr><tr><td>합격 기준</td><td>Y와 T 각각 축·부호 100%, 두 축 각각 최대 절대오차 0.09 이내</td></tr><tr><td>최소 명령 간격</td><td>Y 0.05 mm / T 0.01. 허용오차 0.09와는 다른 값</td></tr><tr><td>데이터</td><td>센터 1장 + 이동 280장(Y 80장, T 200장), 영상 3840×2160</td></tr><tr><td>명령 범위</td><td>Y −2.00~+2.00 mm, T −1.00~+1.00, 두 폴더의 0 사진은 없음</td></tr><tr><td>정답</td><td>파일명 명령값. 실제 물리 위치를 별도 장비로 측정한 정답은 제공되지 않음</td></tr><tr><td>단위</td><td>Y는 mm. T는 물리 단위 미확인으로 원래 명령 단위를 유지</td></tr><tr><td>기준점</td><td>실제 센터 사진은 1장. 여섯 관측 그룹의 나머지 기준점은 회귀로 추정한 절편</td></tr></table>
 <h2 id="flow">2. 전체 데이터 처리 과정</h2><ol class="flow">
 <li><strong>① 센터의 QR 3개</strong><small>기준 영역과 특징점 ID 고정</small></li><li><strong>② 특징점 205개</strong><small>QR 내부에 넓게 분포한 모서리</small></li><li><strong>③ 같은 점 추적</strong><small>NCC 대략 탐색 · LK15 · GN17</small></li><li><strong>④ 이동 좌표 410개</strong><small>205점 × (Δx, Δy)</small></li><li><strong>⑤ 공간 변형 12개</strong><small>위치에 따른 2차 이동장 P12</small></li><li><strong>⑥ 후보 6개 계산</strong><small>기준점 + 축별 곡선에 이동량 적합</small></li><li><strong>⑦ 최소 비용 선택</strong><small>공분산을 반영해 축·부호 결정</small></li><li><strong>⑧ 결과 출력</strong><small>Y/T 값과 추적 품질 반환</small></li></ol>
 <p>학습 때에는 알려진 명령값으로 영상 변형과 이동량의 관계를 만듭니다. 실행 때에는 현재 사진의 정답 축·명령값·시행 ID를 입력하지 않습니다.</p>
 <h2 id="qr">3. QR을 어떻게 처리하고, 왜 그렇게 하는가</h2>
 <figure><img src="assets/QR205_features.png" alt="센터 사진의 QR 세 개 내부에 고정한 205개 특징점"><figcaption>실제 센터 영상에서 고정한 205개 점. QR별 69·66·70개이며, 최종 추적은 QR별 영상 범위 안에서 수행합니다.</figcaption></figure>
 <table><tr><th>처리</th><th>방법</th><th>이유</th></tr><tr><td>특징점 선택</td><td>센터 사진에서 모서리를 고르고 205개 ID를 고정</td><td>사진마다 다시 검출하면 서로 다른 모서리를 같은 점으로 대응시킬 수 있음</td></tr><tr><td>공간 분포</td><td>왼쪽·가운데·오른쪽 QR에 넓게 배치</td><td>평균 이동뿐 아니라 위치에 따른 작은 변형 차이를 측정</td></tr><tr><td>대략 이동</td><td>QR 전체 밝기 무늬의 정규화 상관(NCC), 가로 ±110/세로 ±22px 탐색</td><td>반복 QR 무늬 속에서 LK가 올바른 위치 근처에서 시작하도록 초기화</td></tr><tr><td>국소 추적</td><td>LK15와 앞뒤 추적 검사</td><td>같은 점의 연속 위치를 찾고 대응 실패를 검사</td></tr><tr><td>정밀 추적</td><td>17×17 창, 흐림 σ0.8, cubic 보간, GN 반복</td><td>정수 픽셀보다 작은 위치 차이 계산. 밝기 배율과 오프셋도 함께 적합</td></tr><tr><td>QR 범위 제한</td><td>QR별로 따로 잘라 흐림·보간·정밀 추적</td><td>QR만으로 평가한다는 조건을 지키고 나사·판 외부 정보 혼입 방지</td></tr><tr><td>유효점 검사</td><td>LK 성공, 앞뒤 일치, GN 보정량·수렴 검사. 전체 95% 이상, QR별 20점 이상 요구</td><td>추적 실패를 조용히 이동값으로 출력하지 않도록 검사. 유효 판정 자체가 실제 위치 정답을 보장하지는 않음</td></tr></table>
 <p><strong>QR을 해독해 문자나 ID로 이동량을 계산하는 방법은 아닙니다.</strong> QR의 모서리와 밝기 무늬를 위치 측정용으로 사용합니다. 현재 보정 모델은 같은 기준 영상과 QR 배치를 전제로 합니다.</p>
 <h2 id="model">4. 점 이동을 모델에 넣어 예측값을 구하는 방법</h2>
 <h3>4-1. 205개 점을 12개 공간 변형 계수로 요약</h3><p>점 i의 센터 위치가 (xᵢ,yᵢ), 현재 위치가 (x′ᵢ,y′ᵢ)이면 이동은 (Δxᵢ,Δyᵢ)=(x′ᵢ−xᵢ,y′ᵢ−yᵢ)입니다. 두 좌표 각각에 아래 공간식을 최소제곱으로 맞춥니다.</p>
 <div class="formula">Δxᵢ ≈ a₀ + a₁xᵢ + a₂yᵢ + a₃xᵢ² + a₄xᵢyᵢ + a₅yᵢ²<br>Δyᵢ ≈ b₀ + b₁xᵢ + b₂yᵢ + b₃xᵢ² + b₄xᵢyᵢ + b₅yᵢ²<br>특징 벡터 f = 두 좌표의 계수 12개</div>
 <p>실제로는 좌표를 정규화해서 계산합니다. 여러 점에 공통된 변형을 요약하고, 점별 흔들림이 그대로 분류기에 들어가는 것을 줄입니다. Y와 T 모두 큰 가로 이동으로 보일 수 있으므로, 평균 Δx 하나보다 위치에 따른 미세 변형이 중요합니다.</p>
 <h3>4-2. 학습: 명령 이동량과 영상 변형의 관계 보정</h3><div class="formula">예상 특징 f̂ₛ(v) = bₛ + u축·v + q축·v²</div>
 <p>v는 찾으려는 이동량, u와 q는 축별 선형·2차 응답, bₛ는 관측 그룹별 기준점입니다. 후보는 Y1+, Y1−, Y2+, Y2−, T+, T−의 여섯 개입니다. 이는 관측 자료에서 구성한 보정 그룹이며, 여섯 개의 독립적인 실제 0점 사진을 찍었다는 뜻은 아닙니다.</p>
 <p>0점 차이와 시간·접근 이력이 섞일 수 있어 현재 자료만으로 백래시나 온도를 원인으로 확정할 수 없습니다. 기준점 후보를 사용하는 성적은 이 보정 분포 안에서 해석해야 합니다.</p>
 <h3>4-3. 실행: 후보마다 이동량과 비용 계산</h3><div class="formula">잔차 rₛ(v) = 측정 특징 f − 예상 특징 f̂ₛ(v)<br>후보 비용 Cₛ = minᵥ rₛ(v)ᵀ Σ⁻¹ rₛ(v)<br>최종 후보 = 허용된 부호의 후보 중 비용이 가장 작은 것</div>
 <p>각 후보에서 이동량 v를 GN으로 8회 반복 계산합니다. Σ는 학습 잔차의 공분산입니다. 불안정한 변형 성분과 함께 흔들리는 관계를 반영해 후보 거리를 계산합니다. 이때 밝기 무늬에서 점 좌표를 찾는 앞단 GN과, 운동 곡선에서 이동량을 찾는 뒷단 GN은 서로 다른 계산입니다.</p>
 <p>선택된 후보가 Y이면 (Y=v,T=0), T이면 (Y=0,T=v)를 출력합니다. 두 축 동시 이동은 현재 구조가 표현하지 못합니다. 비용 차이는 후보 간 우열 지표이며 정답 확률이 아닙니다. 정답을 모르는 실행 사진의 실제 오차도 직접 출력할 수 없습니다.</p>
 <h3>4-4. 픽셀 잡음 가중 GN에서 바꾼 부분</h3><p>기존 GN은 창 안 픽셀의 밝기 잔차를 같은 무게로 맞춥니다. 새 후보는 밝기별 잡음의 역분산으로 픽셀 가중치를 주어 점 좌표를 측정합니다. 밝기 배율·오프셋도 같은 가중치로 적합합니다.</p>
 <div class="formula">밝기 잔차 rₚ = 현재 영상(W(p)) − a·센터 영상(p) − b<br>픽셀 가중치 wₚ ≈ 1 / (현재 영상 분산 + a²·센터 영상 분산)</div>
 <p>각 학습 분할의 Y 사진 8쌍에서 잡음 모델을 추정했습니다. 양 영상의 분산과 흐림을 고려한 대각 근사이며, 정지 반복 촬영으로 분리한 센서 잡음 모델은 아닙니다. 점 좌표를 얻은 뒤의 12계수 공간식과 여섯 후보 판정 구조는 유지했습니다.</p>
 <h3>4-5. 실제 저장 모델의 실행 예시</h3><p>익명 파일명으로 복사한 원본 영상을 실행한 결과입니다. 사진의 정답은 추론에 사용하지 않았습니다. 아래는 전체 280장으로 보정한 모델의 재대입 예시입니다.</p>
 <table><tr><th>사진의 명령 정답</th><th>가중 GN 모델 출력</th><th>명령값 대비 오차</th></tr><tr><td>Y +0.05</td><td>Y+, Y=+0.050717658, T=0</td><td>Y 0.000717658 mm</td></tr><tr><td>T −0.01</td><td>T−, Y=0, T=−0.010144621</td><td>T 0.000144621</td></tr></table>
 <h2 id="results">5. 완료된 비교 결과</h2><p><strong>전체 280장 재대입은 두 방법 모두 축·부호 280/280 정답입니다.</strong> 재대입에서 Y 최대오차는 감소했지만, 교차검증에서는 같은 크기의 개선이 이어지지 않았습니다.</p>
 <div class="scroll"><table class="results"><thead><tr><th>평가</th><th>방법</th><th>Y 축·부호</th><th>T 축·부호</th><th>Y 최대오차 mm</th><th>T 최대오차</th><th>전체 실패</th></tr></thead><tbody>'''+''.join(table)+'''</tbody></table></div>
 <p>위 평가에서는 축 정확도와 축·부호 정확도가 같았습니다. 최대오차는 오분류도 포함한 전체 (Y,T) 벡터 기준입니다. T 사진을 Y로 잘못 판정한 값도 Y 오차에 포함합니다. 실패는 축·부호 오류 또는 두 축 중 오차 0.09 초과 또는 거절입니다.</p>
 <figure><img src="assets/accuracy.png" alt="축과 부호 정확도 비교"><figcaption>두 방법을 같은 사진·같은 분할로 비교했습니다. 반복 평가의 분모는 같은 사진의 반복 예측 수입니다.</figcaption></figure>
 <figure><img src="assets/max_error.png" alt="전체 예측의 최대오차 비교"><figcaption>재대입에서 감소한 Y 오차가 교차검증에서는 같은 수준으로 감소하지 않았습니다.</figcaption></figure>
 <figure><img src="assets/repeated_failures.png" alt="20개 동일 분할의 실패 횟수 비교"><figcaption>각 라운드는 280장 전부를 한 번씩 학습에서 제외해 평가합니다. 오류 0 라운드는 기존 9/20회, 가중 GN 8/20회입니다.</figcaption></figure>
 <h2 id="limits">6. 동료에게 함께 전달할 결론</h2><ul>
 <li><strong>확보:</strong> QR 세 개만 사용하는 측정·모델·두 영상 추론 경로를 구현했고, 전체 280장 재대입에서는 축·부호와 허용오차를 모두 만족했습니다.</li>
 <li><strong>비교:</strong> 같은 분할에서 픽셀 가중 GN의 반복 실패는 13→12회입니다. Y 오류 1회는 사라졌지만 T 정확도는 99.7%로 같고, T−0.05 사진의 반복 오분류가 남습니다.</li>
 <li><strong>범위:</strong> 기존 사진으로 설정을 개발하고 평가했습니다. 반복 5,600건은 독립 사진 5,600장이 아니며, 새 촬영 시행의 100%를 검증한 성적이 아닙니다.</li>
 <li><strong>다음 검증:</strong> T−0.05의 특징점 이동과 후보 비용을 분석하고, v7b 2단계 판정을 같은 측정·정답·분할에서 비교하는 계산을 진행 중입니다. 이 문서의 표에는 아직 그 결과를 섞지 않았습니다.</li>
 </ul><p>원본·기존 모델·기존 결과는 보존했습니다. 자세한 실행 기록과 CSV는 <a href="../pixel_weighted_gn_20261015/픽셀잡음가중_GN_비교결과.html">픽셀 잡음 가중 GN 비교 보고서</a>에서 확인할 수 있습니다.</p>
 <p>설명 근거: <a href="../measurement/spread.py">QR별 추적</a> · <a href="../measurement/refine.py">GN 정밀화</a> · <a href="../zero_experiment.py">공간 모델·보정·후보 추론</a> · <a href="../infer.py">기존 실행 함수</a> · <a href="../pixel_weighted_gn_20261015/experiment.py">가중 GN 실험</a> · <a href="../pixel_weighted_gn_20261015/audit_checks.json">재현·보존 검사</a></p>
 </main></html>'''
 document=document.replace('직전 완료된 실험 결과 기준','기준 알고리즘과 픽셀 가중 GN의 완료 결과 기준')
 document=document.replace('T−0.05의 특징점 이동과 후보 비용을 분석하고, v7b 2단계 판정을 같은 측정·정답·분할에서 비교하는 계산을 진행 중입니다. 이 문서의 표에는 아직 그 결과를 섞지 않았습니다.',
  'T−0.05의 특징점·후보 비용 분석과 v7b 2단계 비교를 완료했습니다. 반복 실패는 기존 6후보 13/12회, v7b 2단계 154/138회로 이번 공통 조건에서는 v7b가 더 나빴습니다. 자세한 결과는 별도 분석 보고서에 있습니다.')
 document=document.replace('</main></html>','<p><a href="../t005_v7b_20261015/T005_원인분석_v7b_비교.html">추가 완료: T−0.05 원인 분석과 v7b 비교</a></p></main></html>')
 out=HERE/'동료설명_QR_YT_알고리즘.html';out.write_text(document,encoding='utf8')
 portable=document
 for p in assets.glob('*.png'):
  portable=portable.replace('src="assets/'+p.name+'"','src="data:image/png;base64,'+base64.b64encode(p.read_bytes()).decode()+'"')
 portable=re.sub(r'<a href="\.\./[^"]+">([^<]+)</a>',r'\1',portable)
 (HERE/'동료공유용_단일파일.html').write_text(portable,encoding='utf8')
 print(out)
if __name__=='__main__':main()
