from pathlib import Path
import base64,hashlib,html,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import FuncFormatter
HERE=Path(__file__).resolve().parent
S=json.loads((HERE/'summary.json').read_text(encoding='utf-8'));M=json.loads((HERE/'metadata.json').read_text())
N=np.array(M['sample_counts']);STARTS=M['initializations'];CASES=[s['id'] for s in M['stopping']]
plt.rcParams.update({'font.family':font_manager.FontProperties(fname='C:/Windows/Fonts/malgun.ttf').get_name(),'axes.unicode_minus':False,'font.size':10})
names={'upper_structure':'상단','lower_structure':'하단','fixed_5':'초기 σ=5','fixed_11':'초기 σ=11','fixed_21':'초기 σ=21','paper_N':'논문 초기화 σ=N'}
def fmt(v):
 if v is None:return '계산 불가'
 if isinstance(v,int):return f'{v:,}'
 return f'{v:.7g}'
def table(headers,rows):
 return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+html.escape(str(v))+'</th>' for v in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table></div>'
def embed(name):return '<img src="data:image/png;base64,'+base64.b64encode((HERE/name).read_bytes()).decode()+'">'
def style_axis(ax):
 ax.grid(alpha=.2);ax.set_xlabel('표본 수 N');ax.yaxis.set_major_formatter(FuncFormatter(lambda v,p:f'{v:.0e}'));ax.legend(fontsize=8)

fig,axs=plt.subplots(3,2,figsize=(13,11),layout='constrained')
for ti,target in enumerate(['upper_structure','lower_structure']):
 rows=S['pooled'][target]['by_n']
 ax=axs[0,ti]
 for key,label in [('cap_800_to_1600','800→1600'),('cap_1600_to_3200','1600→3200'),('cap_3200_to_6400','3200→6400')]:
  ax.plot(N,[r['numerical_comparisons']['paper_N'][key]['mu']['max'] for r in rows],label=label,marker='.',lw=1)
 ax.set_yscale('symlog',linthresh=1e-6);ax.set_title(names[target]+' · 4번: 반복 한도별 μ 차이 최대');ax.set_ylabel('μ 차이 [px] · σ=N, 오차 점수 종료');style_axis(ax)
 ax=axs[1,ti]
 for key,label in [('median','중앙값'),('max','최댓값')]:
  ax.plot(N,[r['initialization_spread']['dual_1e-10']['mu'][key] for r in rows],label=label,marker='.',lw=1)
 ax.set_yscale('log');ax.set_title(names[target]+' · 3번: 시작 폭 5·11·21 사이 μ 범위');ax.set_ylabel('최대 μ - 최소 μ [px]');style_axis(ax)
 ax=axs[2,ti]
 for start in STARTS[:3]:
  ax.plot(N[1:],[r['window_adjacent'][start]['dual_1e-10']['max'] for r in rows[1:]],label=names[start],marker='.',lw=1)
 ax.set_yscale('log');ax.set_title(names[target]+' · 2번: N-2→N에 따른 μ 차이 최대');ax.set_ylabel('μ 차이 [px] · 시작 설정 고정');style_axis(ax)
fig.suptitle('4·3·2번 공동 비교 — 모든 위치·미수렴 반환값 포함',fontsize=15)
fig.savefig(HERE/'comparison.png',dpi=150);plt.close(fig)

# Show two actual input profiles and their own fitted curves. No original photo editing.
profiles={(p['ref'],p['target'],p['x']):p for p in map(json.loads,(HERE.parent/'gaussian_window_no_rejection_20261101/profiles.jsonl').read_text().splitlines())}
witnesses=[w for w in S['witnesses'] if (w['target'],w['n']) in [('upper_structure',21),('lower_structure',59)]]
fig,axs=plt.subplots(1,2,figsize=(13,4.5),layout='constrained')
for ax,w in zip(axs,witnesses):
 p=profiles[(w['ref'],w['target'],w['x'])];h=(w['n']-1)//2
 yy=np.arange(p['coarse_y']-h,p['coarse_y']+h+1)
 values=np.array(p['sobel_samples'])[30-h:31+h]
 ax.scatter(yy-p['coarse_y'],values,s=18,color='black',label='입력 Sobel 표본',zorder=3)
 xx=np.linspace(yy[0],yy[-1],1200)
 for row in w['rows']:
  if row['stop']!='dual_1e-10' or row['initial']=='paper_N':continue
  pred=row['A']*np.exp(-(xx-row['mu'])**2/(2*(row['sigma_abs']+1e-12)**2))
  ax.plot(xx-p['coarse_y'],pred,label=names[row['initial']],lw=1.4)
 ax.set_title(f"{names[w['target']]} {w['ref']} x={w['x']}, N={w['n']}");ax.set_xlabel('최초 경계에서의 세로 거리 [px]');ax.set_ylabel('Sobel 응답');ax.grid(alpha=.2);ax.legend(fontsize=8)
fig.suptitle('같은 입력에서 시작 폭만 바꾼 실제 계산 사례',fontsize=14)
fig.savefig(HERE/'profile_examples.png',dpi=160);plt.close(fig)

parts=['''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Gaussian 4·3·2번 공동 비교</title><style>
body{font-family:'Malgun Gothic',sans-serif;background:#f2f5f9;color:#26364c;margin:0;line-height:1.7}main{max-width:1200px;margin:auto;padding:26px 24px 70px}h1{font-size:29px}h2{font-size:21px;margin-top:35px}.card{background:#fff;border:1px solid #d7dfeb;border-radius:8px;padding:17px 22px;margin:16px 0}.warn{border-left:5px solid #ba643b}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:13px;background:white}th,td{padding:9px 12px;white-space:nowrap;text-align:right;border-bottom:1px solid #dce3ec}th{background:#e7edf6}th:first-child,td:first-child{text-align:left}img{width:100%;margin:14px 0}a{color:#1856a4}select{font:inherit;padding:6px;margin:4px}code{background:#e5ebf3;padding:2px 5px}details{margin:12px 0}summary{cursor:pointer;font-weight:bold}small{color:#52657d}</style><main>
<p>2026-11-01 · 기준 이미지 4장 · 상·하단 · Gaussian 단계만</p><h1>4번 계산 종료 · 3번 시작값 · 2번 표본 수 공동 비교</h1>
<div class="card warn"><b>비교를 완료했다. 세 설정을 최종 확정하거나 기존 알고리즘에 적용하지는 않았다.</b>
<p>반복 한도를 늘려 해결되는 미수렴과, 시작값에 따라 다른 μ가 나오는 현상이 함께 있었다. 상단 21개는 시작 폭을 바꾸면 최대 약 1.967px 차이가 남았다. 하단 59개는 일부 계산에서 이미지 밖의 매우 큰 μ를 반환했다. 따라서 이전 표본 수 후보를 수치 검증 완료로 볼 수 없다.</p></div>
<h2>무엇을 비교했나</h2><div class="card"><p><b>총 23,056곳 × 30개 표본 수 × 4개 초기화 방식 × 7개 종료 조건 = 19,367,040개 반환값.</b> 실제 최적화 경로는 2,766,720개이며, 각 경로에서 종료 조건별 결과를 따로 저장했다.</p>
<p>상단 10,848곳, 하단 12,208곳. 가로 1px의 모든 가시 열을 유지했다. 약한 응답·탐색 띠 끝점·미수렴·음수 진폭·2px 초과 이동을 이유로 버리지 않았다. 새로운 위치 제한, 이상치 제거, 합격 px 기준도 넣지 않았다.</p>
<p><b>기존 σ=N 결과의 재현:</b> 이전 800회 조건의 691,680개 결과가 A·μ·σ·제곱오차 합·반복 수·종료 상태까지 모두 정확히 일치했다.</p></div>''']
parts.append(table(['항목','실험 조건','수치의 출처와 의미'],[
 ['2번 표본 수','N=3,5,…,61','3~21은 논문 비교 범위. 23~61은 기존에 논의한 확장 범위. 최적 범위를 뜻하지 않음.'],
 ['3번 시작 폭','σ=5·11·21','이전에 비교한 폭을 재사용한 탐색값. 논문 지정값이나 모든 시작값의 대표성 보증이 아님.'],
 ['3번 고정한 값','A=최초 Sobel 응답, μ=최초 y, simplex 간격=(0.2A,1,5.5)','같은 위치에서 N·초기 폭 이외의 시작 조건을 고정. 기존 11표본의 탐색 간격을 사용.'],
 ['별도 재현 항목','σ=N, A=해당 표본 최대값, simplex 폭 간격=0.5N','이전 결과 재현용. 고정 시작값 비교와 합치지 않음.'],
 ['4번 반복 한도','800·1600·3200·6400','기존 한도를 두 배씩 늘린 비교. 6400은 이번 계산 예산이며 충분성 보증이 아님.'],
 ['4번 기존 종료','오차 점수 상대 차이 ≤ 1e-10','기존 구현. 파라미터 변화 조건 없음.'],
 ['4번 추가 비교','오차 점수와 파라미터 상대 차이 모두 ≤ 1e-6 / 1e-8 / 1e-10','공식 구현의 두 조건 원리를 참고한 실험. 수치와 정규화는 이번 비교 설계이며 논문 처방이 아님.']]))
parts.append('''<details><summary>종료 조건과 실험 한계의 정확한 정의</summary><div class="card"><p>기존 종료: <code>max(f)-min(f) ≤ tol × (|best f|+1e-30)</code>. f는 같은 입력 표본에 대한 제곱오차 합이다.</p>
<p>두 조건 종료는 여기에 <code>max |q_vertex−q_best|/(1+|q_best−origin|) ≤ tol</code>을 함께 요구한다. q는 A·μ·σ, origin은 (0, 최초 경계 y, 0)이다. 영상 좌표의 원점이 μ의 상대 크기를 좌우하지 않도록 최초 위치를 기준으로 사용했다. <b>MATLAB 코드를 그대로 복제한 것은 아니며, 이 정규화 역시 이번 실험 선택이다.</b></p>
<p>이 수치들은 점을 제거하는 조건이나 μ의 정확도 허용값이 아니다. μ가 이미지 밖이어도 그대로 기록한다. 두 조건이 모두 같은 6400회 한도에 도달했다면 반환값 차이가 0이어도 충분한 계산이 확인된 것은 아니다.</p>
<p>시작값 비교는 σ에 대한 제한된 비교다. A·μ의 다른 시작값이나 다른 simplex 크기까지 전수 검증하지 않았다. Gaussian 모델의 표본 적합성, Huber, 최종 직선, θ 보정 정확도는 이번 실행 범위가 아니다.</p></div></details>
<h2>4번 — 반복 횟수만 늘리면 해결되나</h2><div class="card"><p>상단 N=21·기존 σ=N·오차 점수만 종료하는 조건에서 미수렴은 <b>800회 52곳 → 1600회 9곳 → 3200회 0곳 → 6400회 0곳</b>이었다. 3200→6400의 μ 차이는 전부 0이었다.</p><p>그러나 파라미터 조건도 함께 요구하면 같은 조건에서 <b>6400회 미수렴 52곳</b>이 남았다. 오차 점수가 더 이상 변하지 않는 것과 A·μ·σ가 모두 안정되는 것은 같지 않다. 따라서 3200회를 전체 데이터의 확정값으로 선택하지 않았다.</p></div>
<h2>3번 — 같은 표본에서도 시작값에 따라 결과가 다른가</h2><div class="card"><p>상단 N=21, 두 조건 1e-10·최대 6400회에서 시작 σ=5·11·21에 따른 μ 범위의 최댓값은 <b>1.966760692px</b>였다. 모든 10,848곳을 포함한 수치다.</p>
<p>아래 상단 사례에서 σ=5는 수렴으로 종료했지만 SSE가 더 컸다. σ=11·21은 더 작은 SSE에 도달했으나 파라미터 수렴 조건을 충족하지 못했다. <b>‘수렴’ 표시만으로 좋은 경계 위치를 고를 수 없었다.</b></p></div>''')
parts.append(embed('profile_examples.png'))
for w in witnesses:
 rows=[]
 for r in w['rows']:
  if r['stop']=='dual_1e-10':rows.append([names[r['initial']],fmt(r['mu']),fmt(r['sigma_abs']),fmt(r['sse']),int(r['iterations']),'조건 충족' if r['converged'] else '한도 도달 · 유지'])
 parts.append(f"<p><b>{names[w['target']]} {w['ref']}, x={w['x']}px, 최초 y={w['coarse_y']}px, N={w['n']}</b></p>"+table(['시작 방식','μ [px]','최종 σ [px]','SSE','반복 수','종료 상태'],rows))
parts.append('''<p>큰 좌표는 실제 구조물이 이동한 거리가 아니라 제한 없는 계산이 반환한 값이다. 하단의 매우 넓은 Gaussian은 관측 표본 구간에서 거의 수평선처럼 보일 수 있다. 그림은 실제 입력 구간만 그렸으며, μ 값 자체는 표에 숨기지 않고 표시했다.</p>
<h2>2번 — 상단 21개·하단 59개를 확정할 수 있나</h2><div class="card"><p>같은 초기화와 같은 종료 조건 안에서 N만 바꾼 결과를 비교했다. 시작값별로 표본 수의 영향이 달랐으므로 <b>이전 후보를 최종 확정하지 않았다.</b></p><p>아래 표는 상단 N=13~21, 하단 N=59~61의 μ 최댓값−최솟값이다. 모두 두 조건 1e-10·최대 6400회의 반환값이며 미수렴도 포함한다.</p></div>''')
rows=[]
for target,lo,hi in [('upper_structure',13,21),('lower_structure',59,61)]:
 for r in S['pooled'][target]['window_ranges']:
  if r['lo']==lo and r['hi']==hi and r['stop']=='dual_1e-10':
   rows.append([names[target],f'{lo}~{hi}',names[r['initial']],fmt(r['mu_range']['median']),fmt(r['mu_range']['max'])])
parts.append(table(['영역','N 범위','시작 방식','μ 범위 중앙값 [px]','μ 범위 최댓값 [px]'],rows))
parts.append(embed('comparison.png'))
parts.append('''<small>모든 위치를 포함한 집계다. 위쪽 반복 한도 그래프는 0을 표시하기 위한 symlog 축(표시 전환점 1e-6)을 사용하며, 중간·아래 그래프는 로그축이다. 축 설정은 판정이나 데이터 제거에 사용하지 않았다.</small>
<h2>모든 N의 비교값 조회</h2><div class="card"><label>영역 <select id="target"><option value="upper_structure">상단</option><option value="lower_structure">하단</option></select></label><label>초기화 <select id="start"><option value="fixed_5">σ=5</option><option value="fixed_11">σ=11</option><option value="fixed_21">σ=21</option><option value="paper_N">논문 초기화 σ=N</option></select></label>
<p>선택 초기화의 계산 조건 변화, 같은 N의 시작 폭 3종 차이, 선택 초기화에서 N을 2 늘렸을 때의 차이를 나란히 표시한다. 모두 위치별 절대차의 최댓값이다.</p><div id="overview"></div>
<p>미수렴 숫자는 왼쪽부터 기존 800·1600·3200·6400 / 두 조건 1e-6·1e-8·1e-10 순서다. 미수렴 포함 여부에 따라 표본을 바꾸지 않았다.</p></div>
<h2>원시 결과·코드·근거</h2><div class="card"><p><a href="numerical_comparison.csv">전체 N별 수치 비교 CSV</a> · <a href="summary.json">모든 요약·극값 사례</a> · <a href="metadata.json">원시 파일 형식·x 좌표</a> · <a href="provenance.json">사전 기록한 실험 설계·출처·해시</a> · <a href="verification.json">검증 기록</a> · <a href="experiment.cpp">계산 코드</a> · <a href="read_profile.py">개별 위치 결과 읽기</a></p>
<p>원시 반환값은 raw/의 8개 binary64 파일에 모두 저장했다. 다음 명령은 새 계산 없이 상단 예시의 28개 결과를 읽는다.</p><pre>python read_profile.py R01 upper_structure 95 21</pre>
<p>실험용 종료 조건만 새 폴더에서 비교했으며 기존 알고리즘 코드와 설정은 바꾸지 않았다. 원본 이미지·캐시·기존 코드의 실행 전후 해시를 확인했다.</p>
<p><a href="https://www.researchgate.net/publication/283556095_Subpixel_edge_detection_in_blurry_and_noisy_images">Fabijańska(2015), §3.2·§4.1</a>: 표본 수를 바꾼 μ 비교. 자동 최적 N 선택 규칙이나 본 실험의 종료 수치는 제공하지 않는다.</p>
<p><a href="https://www.annals-csis.org/Volume_2/pliks/136.pdf">Fabijańska(2014), §III.B</a>: Gaussian·Nelder–Mead·기본 초기값. 본 실험의 비교용 고정 σ와 simplex 간격을 보증하는 자료는 아니다.</p>
<p><a href="https://www.mathworks.com/help/matlab/ref/fminsearch.html">MathWorks fminsearch</a>: 함수값·파라미터 양쪽 종료 조건의 근거. 이 실험은 동일 구현이나 기본값의 복제가 아니다.</p>
<p><a href="https://www.itl.nist.gov/div898/handbook/pmd/section1/pmd142.htm">NIST 비선형 최소제곱</a>: 시작값에 따라 수렴 및 도달하는 국소해가 달라질 수 있다는 근거. 본 데이터의 합격 기준은 아니다.</p></div>''')
payload={t:v['by_n'] for t,v in S['pooled'].items()}
parts.append('<script>const DATA='+json.dumps(payload,separators=(',',':'),allow_nan=False)+';')
parts.append('''const starts=['fixed_5','fixed_11','fixed_21','paper_N'];const el=id=>document.getElementById(id);const f=v=>v===null?'계산 불가':Number(v).toPrecision(7);function draw(){const target=el('target').value,initial=el('start').value,si=starts.indexOf(initial);let h='<div class="scroll"><table><thead><tr><th>N</th><th>800→1600</th><th>1600→3200</th><th>3200→6400</th><th>점수만→두 조건</th><th>두 조건 1e-8→1e-10</th><th>시작 폭 3종 차이</th><th>N-2→N 차이</th><th>미수렴 수</th></tr></thead><tbody>';for(const r of DATA[target]){const c=r.numerical_comparisons[initial];h+='<tr><td>'+r.n+'</td>'+['cap_800_to_1600','cap_1600_to_3200','cap_3200_to_6400','function_only_to_dual_1e-10','dual_1e-8_to_1e-10'].map(k=>'<td>'+f(c[k].mu.max)+'</td>').join('')+'<td>'+f(r.initialization_spread['dual_1e-10'].mu.max)+'</td><td>'+(r.window_adjacent[initial]['dual_1e-10']?f(r.window_adjacent[initial]['dual_1e-10'].max):'—')+'</td><td>'+r.nonconverged[si].join(' / ')+'</td></tr>'}el('overview').innerHTML=h+'</tbody></table></div>'}el('target').onchange=draw;el('start').onchange=draw;draw();</script></main></html>''')
(HERE/'report.html').write_text(''.join(parts),encoding='utf-8')
digest='''4·3·2번 공동 비교 완료. 설정값은 확정·적용하지 않음.
4번: 상단 N=21, 기존 sigma=N, 오차 점수만 종료할 때 미수렴 800회52곳 / 1600회9곳 / 3200회0곳 / 6400회0곳. 파라미터 조건도 요구하면 6400회 미수렴52곳.
3번: 상단 N=21, 동일 데이터·동일 시작 높이/중심/탐색 간격, 초기 sigma 5/11/21에서 최종 mu 범위 최대 1.966760692px. 두 조건1e-10·6400회 기준, 모든 반환값 포함.
2번: 초기값별로 표본 수 효과가 달라 상단21·하단59 확정 보류. 하단 일부 계산은 수렴 상태에서도 이미지 밖의 매우 큰 좌표를 반환.
모든23056곳, N3..61홀수, 초기화4종, 종료조건7종. 19367040개 반환값 유지. 기존691680개 baseline 완전일치.
초기값5/11/21, 고정 simplex 간격, 반복 한도, 파라미터 정규화·종료 기준은 공개한 실험 조건이며 논문이 정한 최적값·정확도 기준이 아니다.
Gaussian 모델 적합성, Huber, 최종 직선·theta 보정 정확도는 이번 범위에 포함하지 않음.
'''
(HERE/'report.txt').write_text(digest,encoding='utf-8')
files=['experiment.cpp','CMakeLists.txt','prepare.py','original_nelder_mead.hpp','summarize.py','make_report.py','read_profile.py','provenance.json','metadata.json','summary.json','verification.json','numerical_comparison.csv','comparison.png','profile_examples.png','report.html','report.txt']
files += [g['file'] for g in M['groups']]
(HERE/'output_hashes.json').write_text(json.dumps({p:hashlib.sha256((HERE/p).read_bytes()).hexdigest() for p in files},indent=2),encoding='utf-8')
print('Report generated; all return files hashed.')
