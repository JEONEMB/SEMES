from pathlib import Path
from collections import Counter
import json,hashlib,re,csv,html
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
HERE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(name,obj):(HERE/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
S=json.loads((HERE/'summary.json').read_text(encoding='utf-8'));D={r['id']:r for r in S}
Q={r['id']:r for r in map(json.loads,(HERE/'requests.jsonl').read_text(encoding='utf-8').splitlines())}
P=json.loads((HERE/'data_provenance.json').read_text(encoding='utf-8'))
M=json.loads((HERE/'source_manifest.json').read_text(encoding='utf-8'))
commit=M['adahuber_commit']
source_url=f'https://github.com/XiaoouPan/adaHuber/blob/{commit}/src/adaHuber.cpp'
paper_url='https://www3.stat.sinica.edu.tw/statistica/oldpdf/A31n422.pdf'
real=[r for r in S if r['category']=='real']
for r in real:
 q=Q[r['id']];x=np.array(q['x'])[:,0];y=np.array(q['y']);xx=x-x.mean()
 m=float(xx@(y-y.mean())/(xx@xx));b=float(y.mean()-m*x.mean())
 r['ols_coef']=[b,m]; r['max_line_difference_from_ols_px']=float(np.max(np.abs((r['coef'][0]-b)+(r['coef'][1]-m)*x)))
 r['regression_converged_at_original_tolerance']=bool(r['regression_stage']['gradient_inf']<=1e-4)
 r['intercept_converged_at_original_tolerance']=bool(abs(r['mean_stage']['gradient'])<=1e-4)
save('real_data_diagnostics.json',real)

# Plot every input point. No subsampling, no fitted-point removal.
fig,axs=plt.subplots(4,2,figsize=(15,12),sharex=True)
for i,ref in enumerate(['R01','R02','R03','R04']):
 for j,t in enumerate(['upper_structure','lower_structure']):
  r=D[ref+'_'+t];q=Q[r['id']];x=np.array(q['x'])[:,0];y=np.array(q['y']);a=axs[i,j]
  a.scatter(x,y,s=3,alpha=.65,c='#6379a5',label='All Sobel maxima (integer px)')
  a.plot([x.min(),x.max()],r['coef'][0]+r['coef'][1]*np.array([x.min(),x.max()]),color='#c23532',lw=1.5,label='Original adaHuber adaptive')
  a.set_title(f'{ref} / {t.split("_")[0]} | angle={r["angle_deg"]:.6f} deg | n={len(x)}',fontsize=10)
  a.invert_yaxis();a.grid(alpha=.2);a.set_ylabel('Image y (px)')
  if i==3:a.set_xlabel('Image x (px)')
axs[0,0].legend(loc='upper left',fontsize=8)
fig.suptitle('Original code on every saved boundary point (no Gaussian / no rejection)',fontsize=13)
fig.tight_layout();fig.savefig(HERE/'all_points_and_lines.png',dpi=160);plt.close(fig)

# Original image bands + every measured point and its final line; preserve pixels in plot.
fig,axs=plt.subplots(4,2,figsize=(16,8))
targets=json.loads((HERE.parent/'reference_line_methods_108_20261030/configs/targets.json').read_text(encoding='utf-8'))['targets']
T={t['id']:t for t in targets}
for i,im in enumerate(P['images']):
 img=cv2.imdecode(np.frombuffer(Path(im['path']).read_bytes(),np.uint8),cv2.IMREAD_GRAYSCALE)
 for j,t in enumerate(['upper_structure','lower_structure']):
  r=D[im['id']+'_'+t];q=Q[r['id']];x=np.array(q['x'])[:,0];y=np.array(q['y']);a=axs[i,j];lo,hi=T[t]['band_y']
  a.imshow(img[lo-10:hi+11,:],cmap='gray',extent=(-.5,img.shape[1]-.5,hi+10.5,lo-10.5),aspect='auto',vmin=0,vmax=255)
  a.scatter(x,y,s=1,c='#21c9ed',alpha=.75)
  a.plot([x.min(),x.max()],r['coef'][0]+r['coef'][1]*np.array([x.min(),x.max()]),c='#ff6347',lw=1)
  a.set_title(im['id']+' '+t,fontsize=10);a.set_xlabel('x (px)');a.set_ylabel('y (px)')
fig.suptitle('Image-band inspection: cyan = every selected point, red = fitted line\nVertical direction enlarged for inspection; use numeric angle, not displayed tilt',fontsize=12)
fig.tight_layout();fig.savefig(HERE/'image_bands.png',dpi=160);plt.close(fig)

paper_alignment=[
 ('손실·거리','세로 잔차의 Huber 손실 / 식 (1.1), (3.2)','updateHuber: 잔차를 ±τ로 제한한 미분','일치'),
 ('τ 방정식','식 (3.6)의 검열된 잔차제곱 방정식','rootf1 + adaHuberReg의 rhs=(p+log(np))/n','p=1인 우리 입력에서 z=log(n)으로 일치'),
 ('시작값','one-step: OLS와 OLS 잔차로 시작 / p.2163','응답 MAD로 τ 초기화, 음의 미분값으로 계수 시작','불일치'),
 ('반복 계산','τ 방정식과 계수 방정식을 번갈아 풂','매번 경사하강 1회 후 τ 갱신','같은 식을 지향하나 계산 절차는 다름'),
 ('최종 추정량','one-step은 식 (3.6)의 계수·τ 쌍을 반환','adaptive에서도 마지막에 huberMean으로 절편 교체','one-step 그대로가 아님'),
 ('two-step과 비교','표준 Huber(잔차 MAD·1.345) 기울기 + adaptive 절편 / pp.2164–2165','adaptive는 adaptive 기울기 + adaptive 절편','논문 두 방식의 결합; standard 경로가 two-step에 더 가까움'),
 ('수치 설정','본문에 구현의 모든 종료값·상한이 명시되지는 않음','1e-4, 500, root 폭 0.001, 보폭 상한 100','코드 출처는 확인; 논문에서 이 수치의 최적성까지 검증한 것은 아님'),
 ('별도 점 제거','손실에 전체 관측 사용','선택 경로에 거리 기반 삭제·재피팅 없음','일치'),
 ('이론 적용 범위','독립 오차 등 정리의 가정 존재 / Condition 3.1','그 가정을 검사하지 않고 계산 수행','우리 영상에 확률적 보증을 적용한 검증은 아님')
]
save('paper_alignment.json',[dict(item=a,paper=b,code=c,verdict=d) for a,b,c,d in paper_alignment])

issues=[
 {'id':'A01','scope':'선택 경로','severity':'추정량 불일치','title':'adaptive가 논문의 one-step 또는 two-step을 그대로 반환하지 않음',
  'evidence':'식 (3.6) 기반 기울기 뒤에 식 (3.7)의 절편 재추정이 붙는다. 오염된 정규표본 시험에서 마지막 교체 전 미분 최대값은 7.94e-5, 반환 후 동일 τ로 재평가하면 0.0159864였다. 반환 τ는 절편 단계 τ=5.26128이 아니라 앞 단계 τ=4.46261이다.',
  'impact':'숫자 오류와 별도로, 이 결과를 논문의 one-step 추정량이라고 부를 수 없다. 현재 실제 8개에서는 OLS와 같아 이 차이의 영향이 작다.','lines':'226–268','test':'regression_normal_contaminated'},
 {'id':'A02','scope':'선택 경로의 절편 함수','severity':'수식 불일치·영향 재현','title':'스칼라 보폭 식의 괄호 누락으로 보이는 부분',
  'evidence':'huberMean과 huberMeanList의 a1=cross/derDiff*derDiff는 연산 순서상 cross와 같다. 같은 파일의 벡터식 a1=cross/(gradDiffᵀgradDiff)를 1차원으로 줄인 식과 다르다. 괄호만 고친 진단 복사본에서 lognormal 20점 시험의 반환 counter가 13→5, μ가 2.510059594→2.509990608로 변했다.',
  'impact':'저자의 의도는 확인하지 않았다. 두 결과 모두 각자의 종료 조건을 만족했으므로 이것만으로 원본의 모든 출력이 틀렸다고 판정하지 않는다. 실제 8개는 이 while에 진입하지 않아 원본/진단 결과가 같았다.','lines':'83–86, 120–123','test':'mean_lognormal_20'},
 {'id':'A03','scope':'선택 경로의 수치 계산','severity':'수치적 한계 재현','title':'절대 종료값 때문에 단위 축소 시 방정식이 충분히 풀리지 않음',
  'evidence':'[0,0,0.001] 평균 시험에서 τ² 구간 폭이 이미 0.001보다 작아 이분법을 한 번도 수행하지 않았다. τ 방정식 잔차는 0.157605였지만 반환됐다. noisy_line의 세로 좌표를 0.001배 한 뒤 기울기를 원단위로 복원하면 0.492515094, 원본은 0.498115381이다.',
  'impact':'현재 px 단위 8개 입력의 정상 종료와는 별개인 스케일 의존성이다. 자동으로 수치 설정을 교체하지 않았다.','lines':'18–29, 226–268','test':'mean_small_range / noisy_line_scale_0.001'},
 {'id':'A04','scope':'선택 경로의 입력 처리','severity':'예외 처리 부족 재현','title':'상수 가로 좌표의 0 표준편차를 검사하지 않음',
  'evidence':'X가 모두 같은 값이면 표준화에서 0으로 나눠 계수·τ가 NaN이 되지만 결과 구조와 iteration=1을 반환한다. R 진입 함수도 행·열 개수만 검사하므로 상수 X는 차단하지 않는다.',
  'impact':'실제 8개에는 해당하지 않는다. n=1 회귀도 C++ 직접 호출에서는 실패하지만 R 함수는 n≤p를 차단하므로 이를 R 인터페이스 결함으로 세지 않았다.','lines':'231–235; R adaHuber.reg','test':'constant_x'},
 {'id':'A05','scope':'선택 경로의 반환 정보','severity':'상태 정보 부족','title':'반복 제한 종료와 수렴, 절편 단계 상태가 반환값에서 구분되지 않음',
  'evidence':'계수·τ·counter만 반환한다. cap=0 시험은 미분값 0.00648978(1e-4보다 큼)인 채 결과를 반환한다. 절편 함수의 반복 횟수와 종료 사유도 최종 회귀 결과에 전달하지 않는다.',
  'impact':'값을 반환했다는 사실만으로 성공 처리하면 안 된다. 이번 보고서는 삽입 관측 로그로 각 단계의 실제 미분값을 검사했다.','lines':'247–268','test':'iteration_cap_0'},
 {'id':'B01','scope':'선택하지 않은 Lasso 경로','severity':'오류 재현','title':'잔차를 계산한 뒤 τ에는 원 응답의 제곱을 넣음',
  'evidence':'adaHuberLasso, adaHuberLassoList, cvAdaHuberLasso에서 res=Y-Zβ 직후 resSq=square(Y)를 사용한다. 진단 시험에서 해당 식만 residual로 바꾸면 τ가 15.254213→0.377832로 바뀐다.',
  'impact':'이번 저차원 직선 결과에는 영향 없음. 패키지 전체를 무오류라고 할 수 없는 근거이다.','lines':'457–461, 490–494, 544–548','test':'lasso_residual_probe'},
 {'id':'B02','scope':'선택하지 않은 공분산 경로','severity':'정답이 있는 반례 재현','title':'상수열 공분산 대각값을 0으로 만들지 못함',
  'evidence':'두 열이 각각 모두 5와 7이면 분산은 0이다. 원본 huberCov는 대각값 [25,49]를 반환했다. theta>평균²일 때만 평균²를 빼는 분기 때문이며 같을 때는 원래 제곱평균을 그대로 남긴다.',
  'impact':'이번 직선에는 영향 없음. 공분산 기능은 별도 오류가 있다.','lines':'177–182','test':'cov_constant'}
]
save('issues.json',issues)

# Function inventory makes the boundary of this audit explicit.
src=(HERE/'sources/adaHuber/src/adaHuber.cpp').read_text(encoding='utf-8')
active=re.sub(r'/\*.*?\*/','',src,flags=re.S)
functions=re.findall(r'^(?:[\w:<>]+\s+)+([A-Za-z_]\w*)\(',active,re.M)
onpath={'sgn','f1','rootf1','huberDer','huberMean','mad','standardize','updateHuber','adaHuberReg'}
other_executed={'huberMeanList','hMeanCov','huberCov','huberReg','softThresh','cmptLambdaLasso','updateHuberHd','lamm','adaHuberLassoList'}
inventory=[{'function':f,'review':'source read','execution':'selected adaptive path' if f in onpath else 'diagnostic execution' if f in other_executed else 'static review only'} for f in functions]
save('function_inventory.json',inventory)
filechecks=[]
for r in M['sources']:
 if 'file' in r:
  filechecks.append({'file':r['file'],'sha256':sha(HERE/r['file']),'matches_download':sha(HERE/r['file'])==r['sha256']})
# Record final publisher copies; bad URL previously yielded an HTML landing page.
bad=HERE/'sources/wang2021_published.pdf'
if bad.exists() and not bad.read_bytes().startswith(b'%PDF'):bad.rename(HERE/'sources/failed_published_url.html')
for name,url in [('wang2021_final.pdf',paper_url),('wang2021_accepted.pdf','https://www3.stat.sinica.edu.tw/ss_newpaper/SS-2019-0045_na.pdf'),('wang2021_supplement.pdf','https://www3.stat.sinica.edu.tw/preprint/supp/2019-0045_supp.pdf')]:
 p=HERE/'sources'/name
 if p.exists():filechecks.append({'file':str(p.relative_to(HERE)),'url':url,'sha256':sha(p),'pdf_header_valid':p.read_bytes().startswith(b'%PDF')})
verification={'job_count':len(S),'category_counts':dict(Counter(r['category'] for r in S)),
 'all_observation_results_equal_original':all(r.get('observed_matches_original',False) for r in S if 'exception' not in r),
 'real_fits_finite':all(np.isfinite(r['coef']).all() for r in real),
 'real_regression_stages_converged':sum(r['regression_converged_at_original_tolerance'] for r in real),
 'real_intercept_stages_converged':sum(r['intercept_converged_at_original_tolerance'] for r in real),
 'real_points_total':sum(r['n'] for r in real),'real_downweighted_points':sum(r['downweighted_points'] for r in real),
 'max_real_difference_from_ols_px':max(r['max_line_difference_from_ols_px'] for r in real),
 'file_checks':filechecks,
 'limitations':['C++ core compiled natively; R runtime/CRAN binary was not executed.','Independent Armadillo header-only backend; not a bitwise CRAN binary reproduction.','42 finite test inputs are not a proof of absence of all bugs.','No new theta/Y correction, hypothesis test, confidence interval, boundary-ground-truth accuracy claim, or production change.']}
save('verification.json',verification)

def esc(x):return html.escape(str(x))
def table(head,rows):
 return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+esc(x)+'</th>' for x in head)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+str(x)+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table></div>'
results_rows=[]
for i in range(1,5):
 u=D[f'R0{i}_upper_structure'];l=D[f'R0{i}_lower_structure']
 results_rows.append([f'R0{i}',*[f'{r[k]:.6f}' for r in [u,l] for k in ['angle_deg','rmse_px']]])
issue_html=''.join(f'<article><h3>{x["id"]} · {x["title"]}</h3><p class="meta">{x["scope"]} · {x["severity"]} · 원본 {x["lines"]}</p><p>{x["evidence"]}</p><p class="impact">{x["impact"]}</p><p>재현 입력: <code>{x["test"]}</code></p></article>' for x in issues)
audit_table=table(['비교 항목','논문','현재 공개 코드','판정'],[[esc(v) for v in row] for row in paper_alignment])
test_rows=[]
for r in S:
 if r['category']=='real':continue
 result=r.get('result',{})
 test_rows.append([esc(r['id']),esc(r['category']),esc(json.dumps(result)),esc(r.get('mean_stage',{}).get('gradient','—'))])
test_table=table(['시험 이름','분류','원본 반환값','절편/평균 미분값'],test_rows)
inventory_table=table(['함수','확인 범위'],[[esc(r['function']),esc(r['execution'])] for r in inventory])
body=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><title>adaHuber 논문·코드·실행 검증</title>
<style>body{{margin:0;background:#eef2f6;color:#1c2d40;font:16px/1.7 "Malgun Gothic",sans-serif}}main{{max-width:1180px;margin:auto;padding:42px 32px}}h1{{font-size:32px;line-height:1.35}}h2{{font-size:24px;margin-top:48px;border-bottom:2px solid #d9e1ec;padding-bottom:9px}}h3{{font-size:19px;margin-bottom:5px}}.lead{{background:#fff2d8;padding:22px;border-left:6px solid #d59a27}}.cards{{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin:24px 0}}.card,article{{background:white;padding:20px;border:1px solid #dce3ee;border-radius:8px}}.card strong{{font-size:23px;display:block}}article{{margin:16px 0}}.meta,small{{color:#53677f}}.impact{{border-left:3px solid #bacadb;padding-left:12px}}table{{width:100%;border-collapse:collapse;background:white;font-size:14px}}th,td{{text-align:left;border:1px solid #dce3ee;padding:11px;vertical-align:top}}th{{background:#e1e9f2}}td{{overflow-wrap:anywhere}}.scroll{{overflow-x:auto}}img{{width:100%;background:white;margin:12px 0}}a{{color:#1159a3}}code{{font-family:Consolas,monospace;background:#e6edf5;padding:2px 4px}}.note{{padding:16px;background:#ddeaf4}}details{{margin:15px 0}}footer{{font-size:13px;margin-top:36px}}@media(max-width:750px){{.cards{{grid-template-columns:1fr}}main{{padding:22px 12px}}}}</style>
<main><p class="meta">2026-11-01 · 기존 코드 채택 전 검증 · 결과 변경/선별 없음</p><h1>공개 adaHuber 구현을<br>우리 경계점에 그대로 써도 되는가?</h1>
<div class="lead"><b>현재 4개 기준 영상의 8개 직선은 정상 계산됐다. 그러나 공개 코드 전체가 논문과 완전히 일치하거나 오류가 없는 것은 아니다.</b><br>이번 입력에서는 최종 가중치가 전부 1이라 일반 최소제곱과 사실상 같았다. “튀는 경계점의 영향을 줄이는 효과”가 이 입력에서 발생했다고 평가할 수 없다.</div>
<div class="cards"><div class="card"><strong>8 / 8</strong>실제 입력: 유한 계수 + 두 단계 종료 조건 충족</div><div class="card"><strong>0 / 23,056</strong>최종 가중치가 감소한 실제 경계점</div><div class="card"><strong>42개 입력</strong>실제 8개 + 수학·예외·분리 경로 시험 34개</div></div>
<h2>1. 무엇을 실행했나</h2><p>1차 <b>center.bmp</b>, 2차 <b>center1.bmp·center2.bmp·center3.bmp</b>에서 저장한 상단·하단의 정수 Sobel 최대점 (x, y)을 그대로 사용했다. 상단은 영상당 2,712개, 하단은 3,052개이다. 원본 영상 SHA-256을 확인하고 Sobel을 다시 계산했으며, 저장된 모든 점의 x·y·응답값과 완전히 같았다.</p>
<p>중앙 가림 구간을 제외한 좌우 가시 영역을 <b>합쳐서 상단 1개·하단 1개</b>의 직선을 계산했다. Gaussian, 2px 제외, 응답값 8 제외, Huber 뒤 재피팅은 넣지 않았다. 이번 검증은 기존 점의 생성 규칙을 바꾸는 작업이 아니다.</p>
<p>원본 <code>adaHuberReg()</code> C++ 함수와 호출되는 계산 함수를 수정 없이 컴파일했다. R이 설치되지 않은 환경이므로 <b>R List 입출력만 JSON 연결 코드로 대체</b>하고 실제 Armadillo 행렬 계산을 사용했다. R 패키지 전체를 실행한 것과는 구분한다. 원본과 진단 로그 삽입본의 반환값은 모든 시험에서 동일했다.</p>
<p class="note">판정 기준은 코드가 정한 미분값 종료 조건(1e-4), 유한값 여부, 논문의 계산식 잔차, 알려진 수학적 성질이다. 별도의 px 합격선이나 성공 비율을 만들지 않았다. 좌표가 참 경계에 맞는지에 대한 절대 정확도는 정답 좌표가 없어 판정하지 않았다.</p>
<h2>2. 실제 기준 데이터 결과</h2>{table(['영상','상단 영상 각도 °','상단 RMS px','하단 영상 각도 °','하단 RMS px'],results_rows)}
<p>각도는 <code>atan(m)</code>이며 영상의 +x 방향과 이루는 선 각도이다. <b>기계 θ 보정량이 아니다.</b> RMS는 입력 점과 추정 직선의 세로 거리이다. 상단 4개 각도의 최대–최소는 0.002599°, 하단은 0.006812°이며, 여기서 동등성·허용오차 통과 판정을 내리지 않았다.</p>
<p>상단 τ는 6.7425–7.1502px, 하단 τ는 46.6407–49.7899px로 각 입력의 모든 잔차보다 컸다. 따라서 최종 가중치는 모두 1이었다. 독립적으로 계산한 OLS 직선과의 위치 차이는 모든 관측 x에서 최대 <b>{verification['max_real_difference_from_ols_px']:.3g}px</b>였다. 이것은 실제 경계 정답과의 오차가 아니라 두 계산 결과의 차이다.</p>
<img src="all_points_and_lines.png" alt="8개 입력의 모든 경계점과 원본 코드가 추정한 직선"><details><summary>원본 영상 위에서 선택점과 직선 보기</summary><p>세로 방향을 확대했다. 그림상 보이는 기울기를 실제 각도로 읽으면 안 된다. 청록색: 입력 점 전체, 주황색: 최종 직선.</p><img src="image_bands.png" alt="원본 영상 탐색 구간에 선택점 및 직선 표시"></details>
<h2>3. 논문과 코드 대조</h2><p><a href="{paper_url}">Wang et al. (2021) 최종 출판본</a>의 식 (2.11), (3.2), (3.6), (3.7), 3.2절과 보충자료를 확인했다. 2020년 Adaptive Huber 논문도 확인했지만 현재 공개 코드의 데이터 기반 τ 갱신 근거는 2021년 논문이다. 2021년 출판본 p.2175에서도 이 저장소를 직접 안내한다.</p>{audit_table}
<p>따라서 “논문을 그대로 구현했다”는 설명은 부정확하다. 핵심 손실과 τ 방정식은 대응하지만 초기화·계산 방법·최종 절편 처리·수치 종료값에는 차이가 있다. 특히 <code>method="adaptive"</code>는 논문의 두 방식을 결합한 결과여서 one-step의 이론을 그대로 붙일 수 없다.</p>
<h2>4. 코드 점검에서 나온 사항</h2>{issue_html}
<h2>5. 재현 방법과 검증 범위</h2><p>공개 저장소 고정 커밋: <code>{commit}</code>. 원본 파일의 다운로드 직후 해시와 최종 해시가 같다. 기존 프로젝트 코드·설정·이미지는 수정하지 않았다.</p>
<ol><li><code>fetch_sources.py</code>: 고정 커밋의 코드와 빌드용 헤더를 확보한다. 출판사 PDF는 Windows curl로 별도 확보했다.</li><li><code>prepare_build.py</code>: 원본과 관측용 복사본, 원인 확인용 복사본을 구분한다. <code>generated/*.diff</code>에서 모든 차이를 확인할 수 있다.</li><li><code>CMakeLists.txt</code>: C++17/MSVC 19.51, Armadillo 헤더 계산으로 실행 파일을 만든다. R·패키지를 시스템에 설치하지 않았다.</li><li><code>run_validation.py</code>: 원본 영상·입력 확인 → 42개 고정 시험 → 원본/관측본 일치 검사 → 단계별 수치 기록.</li><li><code>make_report.py</code>: 모든 입력 점 그래프, OLS 대조, 보고서를 만든다.</li></ol>
<p>진단 복사본은 원인을 확인하기 위한 비교 대상이며 현재 데이터 결과 표에는 원본 결과만 썼다. 괄호 보완을 기존 알고리즘에 적용하거나, 설정을 최적화하거나, 방법을 자동 교체하지 않았다.</p>
<details><summary>함수 전체 점검 목록과 실제 실행 범위</summary><p>원본 계산 소스 559줄과 R 진입 코드·생성된 호출 코드를 읽었다. 각 함수의 실행 여부는 아래와 같다. 모든 가능한 입력·실행 경로를 다 테스트했다는 의미가 아니다. 공분산·고차원 회귀의 모든 이론/예측 성능을 검증한 것도 아니다.</p>{inventory_table}</details>
<details><summary>실제 영상 외 34개 시험의 원본 반환값</summary><p>수학적 성질 및 예외 재현용 합성 입력이다. 학습용 증강 데이터가 아니다. JSON의 null은 NaN/비유한값을 표현한다. n=1 회귀는 R 진입부가 차단하지만, 직접 C++ 핵심 함수를 호출한 시험에는 포함했다.</p>{test_table}</details>
<h2>6. 이번에 내릴 수 있는 판단</h2><p><b>이 8개 입력에 대한 수치 계산은 정상 종료했다.</b> 실제 영상 경계의 정답 정확도나 θ 보정 성공은 이번 검증의 결론이 아니다.</p><p><b>공개 구현 전체를 검증된 무오류 기준으로 바로 채택하는 것은 권하지 않는다.</b> 논문과 다른 최종 추정량, 절편 계산식 불일치, 입력·종료 처리 한계를 먼저 구분해야 한다. 또한 이 설정에서는 우리 입력의 어느 점도 최종적으로 약하게 반영되지 않았으므로, 원하는 이상점 완화 효과가 얻어졌다고 볼 수 없다.</p>
<h2>근거·산출물</h2><ul><li><a href="{source_url}">고정 커밋 원본 C++ 코드</a> · <a href="sources/adaHuber/src/adaHuber.cpp">보존한 로컬 원본</a></li><li><a href="{paper_url}">2021년 최종 출판 논문</a> · <a href="sources/wang2021_final.pdf">로컬 PDF</a> · <a href="sources/wang2021_supplement.pdf">보충자료</a></li><li><a href="https://arxiv.org/pdf/1706.06991">2020년 Adaptive Huber 논문</a></li><li><a href="summary.json">전체 42개 결과</a> · <a href="real_data_results.csv">실제 8개 결과 CSV</a> · <a href="real_data_diagnostics.json">OLS·종료조건 대조</a></li><li><a href="issues.json">문제별 근거</a> · <a href="paper_alignment.json">논문 대조표</a> · <a href="verification.json">검증 및 해시 기록</a></li><li><a href="data_provenance.json">원본 영상 경로·입력 검증</a> · <a href="requests.jsonl">모든 실행 입력</a> · <a href="native_results.jsonl">원본 출력·단계 관측 로그</a></li></ul><footer>검토 결과는 명시한 커밋·데이터·실행 환경에 한정한다. 통계적 가설검정과 Y/θ 보정 알고리즘 변경은 수행하지 않았다.</footer></main></html>'''
(HERE/'report.html').write_text(body,encoding='utf-8')
text='adaHuber 검증 결과\n\n'+\
 '원본 commit: '+commit+'\n'+\
 '실제 4장, 상하단 8개 입력, 총 23,056점. Gaussian 및 점 제거 없음.\n'+\
 '8/8 유한 계수 및 원본 종료 조건 충족. 최종 가중치 감소: 0점. OLS 최대 위치 차이 '+str(verification['max_real_difference_from_ols_px'])+' px.\n\n'+\
 '\n\n'.join(x['id']+' '+x['title']+'\n'+x['evidence']+'\n'+x['impact'] for x in issues)+\
 '\n\n상세 내용과 그래프: report.html\n'+\
 '실행은 R 패키지 전체가 아니라, 원본 C++ 핵심 함수 + Armadillo + JSON 입출력 연결 코드.\n'
(HERE/'report.txt').write_text(text,encoding='utf-8')
save('output_hashes.json',{p.name:sha(p) for p in HERE.iterdir() if p.is_file() and p.name!='output_hashes.json'})
print('Report created. Original/observed comparisons:',verification['all_observation_results_equal_original'])
print('Function inventory:',dict(Counter(r['execution'] for r in inventory)))
