from pathlib import Path
import base64,hashlib,html,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import FuncFormatter
HERE=Path(__file__).resolve().parent
FONT=font_manager.FontProperties(fname='C:/Windows/Fonts/malgun.ttf').get_name()
plt.rcParams.update({'font.family':FONT,'axes.unicode_minus':False,'font.size':10})
S=json.loads((HERE/'summary.json').read_text(encoding='utf-8'))
Z=np.load(HERE/'arrays.npz');NS=Z['ns']
def f(x):return '—' if x is None else format(x,'.7g')
def table(headers,rows):
 return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+html.escape(str(v))+'</th>' for v in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table></div>'
fig,axs=plt.subplots(2,2,figsize=(13,8),layout='constrained')
for ti,target in enumerate(['upper_structure','lower_structure']):
 for j,field in enumerate(['median','max']):
  ax=axs[ti,j]
  for r,color in zip(['R01','R02','R03','R04'],['#2257a5','#e08028','#228b6b','#9a4bba']):
   row=next(x for x in S['groups'] if x['ref']==r and x['target']==target)
   vals=[x['adjacent_abs_delta'][field] for x in row['by_n'][1:]]
   ax.plot(NS[1:],vals,label=r,color=color,marker='.',markersize=3,linewidth=1.2)
  ax.set_yscale('log');ax.set_xlabel('표본 수 N (직전 N-2와 비교)');ax.set_ylabel('|μ(N) - μ(N-2)| [px], 로그축')
  ax.yaxis.set_major_formatter(FuncFormatter(lambda value, position: f'{value:.0e}'))
  ax.set_title(('상단' if ti==0 else '하단')+' · '+('전체 위치의 중앙값' if j==0 else '전체 위치의 최댓값'))
  ax.grid(alpha=.2);ax.legend(ncol=4,fontsize=8)
fig.suptitle('가로 1px · 모든 위치 유지 · Gaussian 반환 위치의 표본 수 민감도',fontsize=15)
fig.savefig(HERE/'window_changes.png',dpi=160);plt.close(fig)
payload={}
for g in S['groups']:
 key=g['ref']+'_'+g['target'];d=Z[key+'_data']
 payload[key]={'x':d[:,0,0].astype(int).tolist(),'coarse':d[:,0,1].astype(int).tolist(),'mu':d[:,:,8].tolist(),'converged':d[:,:,12].astype(int).tolist(),'iterations':d[:,:,11].astype(int).tolist()}
(HERE/'explorer_data.js').write_text('window.DATA='+json.dumps({'ns':NS.tolist(),'groups':payload},separators=(',',':'),allow_nan=False)+';',encoding='utf-8')
img=base64.b64encode((HERE/'window_changes.png').read_bytes()).decode()
parts=['''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Gaussian 표본 수 비교 — 제외 조건 없음</title><style>
body{font-family:'Malgun Gothic',sans-serif;margin:0;background:#f3f5f8;color:#24334a;line-height:1.65}main{max-width:1180px;padding:28px 24px 70px;margin:auto}h1{font-size:28px}h2{font-size:21px;margin-top:32px}.card{background:white;border:1px solid #dbe2ec;border-radius:8px;padding:18px 22px;margin:16px 0}.warn{border-left:5px solid #bb6538}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;background:white;font-size:13px}th,td{padding:9px 12px;border-bottom:1px solid #dbe2ec;white-space:nowrap;text-align:right}th:first-child,td:first-child{text-align:left}th{background:#e8edf5}img{width:100%}a{color:#1752a5}select,input,button{font:inherit;padding:4px;margin:4px}button{cursor:pointer}canvas{width:100%;height:320px;background:#fff}small{color:#53647a}code{background:#edf1f7;padding:2px 5px}.bad{color:#a64c20}details{margin:14px 0}summary{cursor:pointer;font-weight:bold}.controls{display:flex;gap:14px;align-items:center;flex-wrap:wrap}</style><main>
<p>2026-11-01 · 기준 이미지 4장 · 상·하단 외부 구조물</p><h1>Gaussian 주변 표본 수 비교</h1>
<div class="card warn"><b>상단: 21개를 작업용 후보로 제시. 하단: 표본 수 미결정.</b><p>상단은 N=13~21에서 전체 10,848곳의 μ 변화 폭 최댓값이 0.00075345px였다. 다만 N=21의 52곳은 기존 계산 종료 조건을 충족하지 못했으므로 수치 검증 완료로 표현하지 않는다. 하단은 이전 59개 선택을 지지하는 전체 위치의 안정 구간을 확인하지 못했다. 큰 반환값도 제외하지 않았다.</p></div>
<h2>이번에 수행한 범위</h2><div class="card">
<p>Fabijańska(2015) 그림 3·표 1처럼 같은 위치에서 N=3,5,…,21을 바꾸어 μ를 비교했다. 이전에 요청한 확장 범위 N=23,25,…,61도 따로 포함했다. <b>61은 이번에 계산한 범위의 끝이며, 안정 판정 기준이 아니다.</b></p>
<p>R01=center.bmp, R02=center1.bmp, R03=center2.bmp, R04=center3.bmp. 상단은 사진마다 2,712개 열, 하단은 3,052개 열이다. 총 23,056곳 × 30개 표본 수 = <b>691,680회</b>를 계산했다. 원본 영상·기존 설정은 수정하지 않았다.</p>
<p>기존 구조물 탐색 영역과 중앙 가림 구간은 유지했다. 가로 1px마다 세로 Sobel 응답의 최댓값 위치를 고정한 뒤, 그 위치 주변을 위아래 대칭으로 1px씩 읽었다. 표본은 탐색 띠 밖까지 실제 이미지에서 읽으며, 띠 경계에서 잘라내지 않는다.</p>
<p><b>제거한 제외 조건:</b> 응답 8 미만, 탐색 띠 끝점, 계산 미수렴, 음수 진폭, 초기 위치 대비 2px 초과. 어느 조건도 행 삭제나 표본 수 선정용 표본 필터로 사용하지 않았다. 미수렴 여부는 기록만 했다. 새로운 잔차·봉우리·이동량 제한도 넣지 않았다.</p>
<p><b>초기값:</b> A=사용 표본의 최대 응답, μ=최초 위치, σ=N. 이 초기화는 2014 선행 논문의 III.B를 따른다. 2015 논문은 구체적인 계산 시작값을 이 정도로 명시하지 않는다. 이전 실험의 σ=11 고정과 다르다.</p>
<p><b>그대로 둔 수치 구현:</b> 기존 Nelder–Mead, 최대 800회, 상대 목적함수 벌어짐 10⁻¹⁰, 시작 simplex 간격 (0.2A, 1, 0.5N), 폭 계산 |σ|+10⁻¹². 이 값들은 논문이 정한 값이 아니라 기존 구현 설정이다. 따라서 이 보고서는 논문 전체 구현의 동일 재현이 아니라 <b>표본 수 비교 절차의 적용</b>이다.</p>
<p>Huber 피팅, 최종 직선·기울기 계산, θ 보정, 모델 적합성 통과 판정은 수행하지 않았다.</p></div>
<h2>상단 후보와 하단 미결정의 근거</h2>''']
rows=[]
for g in S['groups']:
 if g['target']=='upper_structure':
  p=g['ranges']['13-21']; n=next(x for x in g['by_n'] if x['n']==21)
  rows.append([g['ref'],'상단',g['columns'],'13~21',f(p['median']),f(p['max']),n['nonconverged']])
 else:
  p=g['ranges']['59-61'];n=next(x for x in g['by_n'] if x['n']==61)
  rows.append([g['ref'],'하단',g['columns'],'59~61',f(p['median']),f(p['max']),n['nonconverged']])
parts.append(table(['사진','영역','전체 위치 수','비교 표본 수','위치 변화 폭 중앙값 [px]','위치 변화 폭 최대 [px]','마지막 N의 미수렴 수'],rows))
parts.append('''<div class="card"><p>위치 변화 폭은 <b>동일한 열에서 표본 수를 바꿨을 때 나온 μ의 최댓값−최솟값</b>이다. 열 사이의 구조물 높이 차이가 아니다. 표에서 중앙값은 단순 분포 요약이며, 이를 기준으로 나머지 위치를 버리거나 통과 판정을 하지 않았다.</p><p>상단 21개는 논문 후속 실험에서도 사용한 수이며, 이번 전체 위치의 13~21개 비교에서도 변화가 작아 작업용 후보로 선택했다. 논문에 자동 최소 표본 선택 규칙이 있는 것은 아니다. 미수렴 52곳도 포함했으므로 확정된 수치 정확도는 아니다.</p><p>하단 R04의 매우 큰 수치는 물체가 그렇게 이동했다는 뜻이 아니라, 제한 없는 Gaussian 계산이 반환한 μ이다. 이것을 잘라내거나 없애면 이전과 같은 여과 문제가 되므로 그대로 기록했다. 수렴으로 표시된 반환값에서도 발생했다.</p></div>''')
parts.append('<img src="data:image/png;base64,'+img+'" alt="모든 위치의 표본 수 변화에 따른 중앙값과 최댓값">')
parts.append('''<h2>모든 위치의 μ 직접 확인</h2><div class="card"><p>논문 표 1처럼 한 위치의 N별 μ를 볼 수 있다. 모든 23,056곳을 포함하며 결과가 좋은 위치만 골라 놓지 않았다. 기본 위치는 가시 범위의 첫 열이다.</p><div class="controls"><label>사진 <select id="ref"><option>R01</option><option>R02</option><option>R03</option><option>R04</option></select></label><label>영역 <select id="target"><option value="upper_structure">상단</option><option value="lower_structure">하단</option></select></label><label>x <input id="xvalue" type="number" value="24" style="width:80px"></label><button id="go">위치 보기</button><label>그래프 표시 <select id="endn"><option value="21">3~21개</option><option value="61">3~61개</option></select></label></div><input id="slider" type="range" min="0" max="2711" value="0" style="width:98%"><p id="caption"></p><canvas id="chart"></canvas><div id="pointtable"></div></div>
<h2>전체 위치를 포함한 수치 요약</h2><p>각 사진·영역의 N별 전체 위치를 그대로 비교했다. 큰 값과 미수렴을 제거한 별도 표는 만들지 않았다.</p>''')
for g in S['groups']:
 rows=[]
 for row in g['by_n']:
  delta=row['adjacent_abs_delta']
  rows.append([row['n'],g['columns'],row['nonconverged'],row['nonfinite'],f(delta['min']) if delta else '—',f(delta['median']) if delta else '—',f(delta['max']) if delta else '—'])
 parts.append('<details><summary>'+g['ref']+' · '+('상단' if g['target']=='upper_structure' else '하단')+'</summary>'+table(['N','전체 위치','미수렴','비유한값','직전 N 대비 변화 최소 [px]','중앙값 [px]','최대 [px]'],rows)+'</details>')
parts.append('''<h2>계산·출처 기록</h2><div class="card"><p><a href="fits.csv">모든 691,680회 반환값 CSV</a> · <a href="window_comparison.csv">N별 요약 CSV</a> · <a href="summary.json">요약 JSON</a> · <a href="profiles.jsonl">모든 최초 위치와 Sobel 표본</a> · <a href="provenance.json">설정·원본 경로·해시</a> · <a href="verification.json">재현 확인</a> · <a href="experiment.cpp">실험 코드</a></p><p>기존 가로 2px 위치 11,528곳과의 Sobel 표본·최초 위치는 정확히 일치했다. 원본 4장과 기존 코드·설정 파일의 SHA-256은 실행 전후 동일하다. 1px 간격의 모든 가시 열 × 모든 N의 행 수를 확인했다.</p><p><a href="https://www.researchgate.net/publication/283556095_Subpixel_edge_detection_in_blurry_and_noisy_images">Fabijańska (2015), §3.2 그림 3·표 1, §4.1</a>: N=3~21 비교 및 후속 21개 사용. 안정 임계값·최소 안정 N 자동 선택 규칙은 명시되지 않았다.</p><p><a href="https://www.annals-csis.org/Volume_2/pliks/136.pdf">Fabijańska (2014), §III.B</a>: Gaussian 모델·Nelder–Mead·초기값 설정. 최대 800회·10⁻¹⁰·현재 simplex 간격은 논문이 지정한 값이 아니다.</p></div>
<script src="explorer_data.js"></script><script>
const el=id=>document.getElementById(id);let index=0;
function group(){return DATA.groups[el('ref').value+'_'+el('target').value]}
function fmt(v){return Number.isFinite(v)?(Math.abs(v)>1e6?v.toExponential(9):v.toFixed(9)):'계산 불가'}
function draw(){const g=group(),ns=DATA.ns,mu=g.mu[index],cy=g.coarse[index];el('xvalue').value=g.x[index];el('slider').value=index;el('caption').textContent=`x=${g.x[index]}px, 최초 경계 y=${cy}px · 모든 N의 반환값 유지`;
const canvas=el('chart'),ctx=canvas.getContext('2d'),W=canvas.clientWidth,H=320;canvas.width=W*devicePixelRatio;canvas.height=H*devicePixelRatio;ctx.scale(devicePixelRatio,devicePixelRatio);ctx.clearRect(0,0,W,H);const end=+el('endn').value,values=mu.filter((v,i)=>ns[i]<=end).map(v=>v-cy),finite=values.filter(Number.isFinite);let lo=Math.min(...finite),hi=Math.max(...finite);if(lo===hi){lo-=1;hi+=1}const span=hi-lo;lo-=span*.08;hi+=span*.08;const left=105,right=W-25,top=25,bottom=270;const X=n=>left+(n-3)/(end-3)*(right-left),Y=v=>bottom-(v-lo)/(hi-lo)*(bottom-top);ctx.font='12px Malgun Gothic';ctx.fillStyle='#26364c';ctx.fillText('μ - 최초 y [px]',left,15);ctx.strokeStyle='#dbe3ef';for(let k=0;k<=4;k++){let v=lo+(hi-lo)*k/4;ctx.beginPath();ctx.moveTo(left,Y(v));ctx.lineTo(right,Y(v));ctx.stroke();ctx.fillText(v.toExponential(3),4,Y(v)+4)}ctx.strokeStyle='#1b5cba';ctx.lineWidth=2;ctx.beginPath();let started=false;values.forEach((v,i)=>{if(!Number.isFinite(v))return;if(!started){ctx.moveTo(X(ns[i]),Y(v));started=true}else ctx.lineTo(X(ns[i]),Y(v))});ctx.stroke();values.forEach((v,i)=>{if(!Number.isFinite(v))return;ctx.fillStyle=g.converged[index][i]?'#1b5cba':'#c45b28';ctx.beginPath();ctx.arc(X(ns[i]),Y(v),3,0,7);ctx.fill()});ctx.fillStyle='#26364c';for(let n=3;n<=end;n+=end===21?2:10)ctx.fillText(n,X(n)-5,bottom+20);ctx.fillText('표본 수 N · 주황색은 미수렴 반환값',left,bottom+43);
let h='<div class="scroll"><table><thead><tr><th>N</th><th>μ [px] — 논문처럼 4자리 표시</th><th>μ [px] — 9자리 표시</th><th>직전 N과 차이 [px]</th><th>반복 수</th><th>계산 종료 상태</th></tr></thead><tbody>';ns.forEach((n,i)=>h+=`<tr><td>${n}</td><td>${mu[i].toFixed(4)}</td><td>${fmt(mu[i])}</td><td>${i?fmt(mu[i]-mu[i-1]):'—'}</td><td>${g.iterations[index][i]}</td><td class="${g.converged[index][i]?'':'bad'}">${g.converged[index][i]?'종료 조건 충족':'미수렴 · 유지'}</td></tr>`);el('pointtable').innerHTML=h+'</tbody></table></div>'}
function reset(){index=0;el('slider').max=group().x.length-1;draw()}
el('ref').onchange=reset;el('target').onchange=reset;el('endn').onchange=draw;el('slider').oninput=e=>{index=+e.target.value;draw()};el('go').onclick=()=>{let i=group().x.indexOf(+el('xvalue').value);if(i<0){el('caption').textContent='해당 x는 기존 가시 범위에 포함되지 않습니다.';return}index=i;draw()};window.addEventListener('resize',draw);draw();
</script></main></html>''')
(HERE/'report.html').write_text(''.join(parts),encoding='utf-8')
digest='''상단: 21개 작업용 후보. N=13~21에서 전체 10,848곳의 변화 폭 최대 0.0007534483px. N=21 미수렴 52곳은 삭제하지 않았다.
하단: 미결정. 59→61개 변경 시 사진별 최대 차이: R01 6.579080px, R02 0.962935px, R03 3.173187px, R04 3397671.177925px(이미지 밖의 수치 반환).
기준 4장, 가로1px, N=3..61 홀수, 691680회. 2px 이동 제한·최소 응답 제한·미수렴 제외·공통 채택 표본 필터 없음.
초기 sigma=N. 최대800회·tol1e-10 및 simplex 시작 간격은 기존 수치 구현 설정으로 남아 있다.
전체 설정과 각 위치의 모든 반환값은 report.html, fits.csv, profiles.jsonl, provenance.json에 기록했다.
'''
(HERE/'report.txt').write_text(digest,encoding='utf-8')
files=['experiment.cpp','original_nelder_mead.hpp','CMakeLists.txt','run_config.json','prepare.py','summarize.py','make_report.py','fits.csv','profiles.jsonl','runtime.json','summary.json','arrays.npz','window_comparison.csv','verification.json','provenance.json','window_changes.png','explorer_data.js','report.html','report.txt']
(HERE/'output_hashes.json').write_text(json.dumps({p:hashlib.sha256((HERE/p).read_bytes()).hexdigest() for p in files},indent=2),encoding='utf-8')
print('Report and all-position explorer written. No source production settings changed.')
