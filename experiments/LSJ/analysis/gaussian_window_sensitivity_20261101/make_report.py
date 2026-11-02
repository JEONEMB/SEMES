"""Render the completed Gaussian-only sensitivity comparison with explicit denominators."""
import base64
import hashlib
import html
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import FuncFormatter, NullFormatter

HERE=Path(__file__).resolve().parent
BASE=HERE.parent/'reference_line_methods_108_20261030'
font=Path('C:/Windows/Fonts/malgun.ttf')
if font.exists():
    font_manager.fontManager.addfont(str(font))
    plt.rcParams['font.family']=['Malgun Gothic','DejaVu Sans']
plt.rcParams.update({'axes.unicode_minus':False,'figure.dpi':130,'savefig.dpi':150,
                     'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                     'mathtext.fontset':'dejavusans'})
N=np.arange(3,22,2)
COLORS=['#2458bd','#d26c16','#168573','#9a45a2']
CN={'upper_structure':'상단 구조물','lower_structure':'하단 구조물'}

def fmt(v):
    if v is None: return '—'
    if v==0: return '0'
    return f'{v:.6g}'

def img(name,caption):
    b=base64.b64encode((HERE/name).read_bytes()).decode()
    return f'<figure><img src="data:image/png;base64,{b}" alt="{html.escape(caption)}"><figcaption>{caption}</figcaption></figure>'

def table(headers,rows):
    return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+str(v)+'</th>' for v in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+str(v)+'</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table></div>'

def main():
    s=json.loads((HERE/'summary.json').read_text(encoding='utf-8'))
    z=np.load(HERE/'paired_arrays.npz')
    profiles=json.loads((HERE/'profiles.json').read_text())
    pmap={(p['ref'],p['target'],p['x']):p for p in profiles}
    provenance=json.loads((HERE/'provenance.json').read_text(encoding='utf-8'))
    groups=s['groups']
    # Plot 1: window sensitivity uses the same accepted columns across all main windows.
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for ri,target in enumerate(CN):
        for ci,ref in enumerate(['R01','R02','R03','R04']):
            g=next(g for g in groups if g['ref']==ref and g['target']==target)
            ws=g['windows'][1:]
            axes[ri,0].plot(N[2:],[w['abs_adjacent_delta_common_px']['p95'] for w in ws[1:]],'-o',color=COLORS[ci],label=ref)
            axes[ri,1].plot(N[1:],[w['selected_percent'] for w in ws],'-o',color=COLORS[ci],label=ref)
        axes[ri,0].set_title(CN[target]+' · 표본을 2개 늘렸을 때 위치 차이의 P95')
        axes[ri,0].set_ylabel('|μ(N) − μ(N-2)| [px] · symlog 축')
        axes[ri,0].set_yscale('symlog',linthresh=1e-5)
        axes[ri,0].yaxis.set_major_formatter(FuncFormatter(lambda value,pos:f'{value:g}'))
        axes[ri,0].yaxis.set_minor_formatter(NullFormatter())
        if target=='lower_structure':
            axes[ri,0].set_yscale('linear')
            axes[ri,0].set_ylim(bottom=0)
            axes[ri,0].set_ylabel('|μ(N) − μ(N-2)| [px]')
        axes[ri,0].axvline(11,color='#777',ls=':',lw=1)
        axes[ri,1].set_title(CN[target]+' · 기존 조건으로 채택되는 점 비율')
        axes[ri,1].set_ylabel('채택 비율 [%]'); axes[ri,1].set_ylim(0,103)
        for ax in axes[ri]:
            ax.set_xticks(N[1:]); ax.set_xlabel('경계 한 점에 사용하는 표본 개수')
            ax.grid(alpha=.2); ax.legend(ncol=2)
    fig.suptitle('같은 초기 경계에서 표본 수만 변경 · 주 비교: 초기 σ=11px 고정',fontsize=15)
    fig.savefig(HERE/'window_comparison.png'); plt.close(fig)

    # Plot 2: individual-column spans; occlusion gaps are left blank.
    fig,axes=plt.subplots(2,1,figsize=(12,7),layout='constrained')
    for ax,target in zip(axes,CN):
        for ci,ref in enumerate(['R01','R02','R03','R04']):
            key=ref+'_'+target; common=z[key+'_common']; mu=z[key+'_mu'][:,:,0]
            span=np.ptp(mu[common,4:],axis=1)
            ax.scatter(z[key+'_xs'][common],span,s=4,alpha=.5,label=ref,color=COLORS[ci])
        ax.set_title(CN[target]+' · 각 위치에서 11~21개 결과의 최대−최소')
        ax.set_yscale('log'); ax.set_ylabel('위치 변화 폭 [px] · log 축')
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value,pos:f'{value:g}'))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel('이미지 가로 좌표 x [px]'); ax.grid(alpha=.2); ax.legend(ncol=4)
    fig.savefig(HERE/'spatial_sensitivity.png'); plt.close(fig)

    # Plot 3: 3 fixed starts are not pooled or optimized; each is its own control series.
    fig,axes=plt.subplots(1,2,figsize=(12,4.4),layout='constrained')
    for ax,target in zip(axes,CN):
        for ci,ref in enumerate(['R01','R02','R03','R04']):
            g=next(g for g in groups if g['ref']==ref and g['target']==target)
            ax.plot(N[1:],[r['three_start_mu_span_selected_px']['max'] for r in g['initialization_audit'][1:]],'-o',color=COLORS[ci],label=ref)
        ax.set_title(CN[target]+' · 초기값 3종 사이 최대 위치 차이')
        ax.set_xticks(N[1:]); ax.set_xlabel('표본 개수'); ax.set_ylabel('μ 최대−최소 [px] · log 축')
        ax.set_yscale('log'); ax.grid(alpha=.2); ax.legend(ncol=2)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value,pos:f'{value:g}'))
        ax.yaxis.set_minor_formatter(NullFormatter())
    fig.savefig(HERE/'initialization_check.png'); plt.close(fig)

    # Plot 4: deterministic representative examples, not best-looking hand-selected columns.
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    examples=[]
    for ri,target in enumerate(CN):
        key='R01_'+target; xs=z[key+'_xs']; common=z[key+'_common']; mu=z[key+'_mu'][:,:,0]
        positions=np.flatnonzero(common)
        spans=np.ptp(mu[common,1:],axis=1)
        ordered=positions[np.argsort(spans,kind='stable')]
        ix=int(ordered[len(ordered)//2])
        x=int(xs[ix]); p=pmap[('R01',target,x)]; cy=p['coarse_y']
        offsets=np.arange(-10,11); values=np.array(p['sobel_samples_m10_p10'])
        axes[ri,0].plot(offsets,values,'ko',ms=4,label='원본 Sobel 표본')
        for n,color in [(5,'#d26c16'),(11,'#2458bd'),(21,'#168573')]:
            ni=int((n-3)//2); hw=(n-1)/2; u=np.linspace(-hw,hw,301)
            A=z[key+'_A'][ix,ni,0]; m=mu[ix,ni]-cy; sig=z[key+'_sigma'][ix,ni,0]
            axes[ri,0].plot(u,A*np.exp(-((u-m)**2)/(2*(sig+1e-12)**2)),color=color,label=f'{n}개 · 중심 {m:+.4f}px')
            axes[ri,0].axvline(m,color=color,ls=':',alpha=.7)
        axes[ri,0].set_title(f'R01 {CN[target]} x={x}px · 한 점의 피팅 예시')
        axes[ri,0].set_xlabel('초기 경계 y에서의 상대 위치 [px]'); axes[ri,0].set_ylabel('Sobel 세로 밝기 변화량')
        axes[ri,0].legend(fontsize=9)
        axes[ri,1].plot(N[1:],mu[ix,1:]-cy,'-o',color='#2458bd')
        axes[ri,1].set_title('표본 수에 따라 이 점의 위치가 어떻게 변하는가')
        axes[ri,1].set_xticks(N[1:]); axes[ri,1].set_xlabel('표본 개수'); axes[ri,1].set_ylabel('추정 μ − 초기 경계 y [px]')
        for ax in axes[ri]: ax.grid(alpha=.2)
        examples.append({'ref':'R01','target':target,'x':x,'coarse_y':cy,
            'selection':'All-window accepted R01 columns ranked by 5..21 mu span; middle rank chosen.',
            'mu_by_n':{str(n):float(mu[ix,ni]) for ni,n in enumerate(N) if n>=5}})
    fig.savefig(HERE/'profile_examples.png'); plt.close(fig)
    (HERE/'example_provenance.json').write_text(json.dumps(examples,ensure_ascii=False,indent=2),encoding='utf-8')

    range_rows=[]
    for g in groups:
        span=g['span_11_to_21_same_common_px']
        range_rows.append([g['ref'],CN[g['target']],f"{g['common_selected_5_to_21']:,} / {g['eligible_columns']:,}",
                           fmt(span['median']),fmt(span['p95']),fmt(span['max'])])
    window_sections=[]
    for g in groups:
        rows=[]
        for w in g['windows']:
            st=w['statuses']; delta=w['abs_delta_from_11_common_px']
            rows.append([w['n'],f"±{w['half_width_px']}px",w['selected_count'],
                         st.get('not_converged',0),st.get('shift_over_2px',0),
                         *([fmt(delta['median']),fmt(delta['p95']),fmt(delta['max'])] if w['n']>=5 else ['—','—','—'])])
        window_sections.append(f"<details><summary>{g['ref']} · {CN[g['target']]} — 표본별 수치</summary>"+
            table(['표본 수','범위','채택','수렴 실패','±2px 초과','11개 대비 |차이| 중앙값','P95','최대'],rows)+
            '<p>위치 차이는 5~21개에서 모두 채택된 동일 위치를 기준으로 계산했습니다. 3개는 참고용으로 채택·제외 개수만 표시했습니다.</p></details>')
    # Write a concise plain text record, plus a self-contained HTML report.
    text=[
        'Gaussian 표본 수 민감도 검증 · 2026-11-01',
        '범위: 1·2차 기준 4장 × 상·하단. QR·Huber·theta 보정 성능은 평가하지 않음.',
        '현재 E01의 초기 경계를 고정. 새 경계 선택 3방식은 미구현/미적용.',
        '주 비교: 초기 sigma=11, simplex sigma step=5.5 고정; N=5,7,...,21. N=3은 참고.',
        '초기값 점검: sigma=5와21의 별도 고정 계열. 결과를 합치거나 최선 결과를 선택하지 않음.',
        '원본 4장 SHA256 확인. OpenCV 3.4.15와 기존 Nelder-Mead 함수 사용.',
        '120개 저장 프로파일의 Sobel 표본 및 11개 Gaussian mu 재현 차이=0. 8그룹 채택 마스크 동일.',
        '상단: 초기값11 고정 시 표본11~21의 위치 변화 폭 P95는 이미지별 약0.000006~0.000008px. 최대는 R04에서0.0157953px.',
        '하단: 같은 조건의 변화 폭 중앙값0.5245~0.5922px, P95 1.3661~1.4123px, 최대2.5678px.',
        '공통 비교 위치: 상단5397/5424; 하단2176/6104. 제외 결과를 숨기지 않도록 표본별 실패 수를 별도 기록.',
        '상단11개는 유지 검토 후보. 상단17~21에서는 다른 초기값에 의한 일부 차이도 있으므로 무조건 확대할 근거 없음.',
        '하단은 현재 선택/모델/솔버/채택조건 아래에서 표본 수만 정해 안정성을 확보했다고 말할 수 없음.',
        '통과용 px 임계값은 임의로 만들지 않음. P95는 점 위치 차이의95백분위이며 신뢰구간 아님.',
        '물리적 경계 정답 없음. 이 수치는 실제 위치 오차나 theta 각도 오차가 아니며 로봇±0.005도 판정에 사용하지 않음.',
        '방향 고정의 적합성, Gaussian 채택기준/수렴조건의 타당성은 이번 표본 수 점검으로 해결되지 않음.',
        '최종 설정은 변경하지 않음. 원본 코드·설정·기존 결과 해시 불변 확인.',
    ]
    (HERE/'report.txt').write_text('\n'.join(text)+'\n',encoding='utf-8')
    page='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Gaussian 표본 수 검증 · 2026-11-01</title><style>
body{font-family:'Malgun Gothic',sans-serif;color:#203043;background:#f3f5f8;margin:0;line-height:1.7}main{max-width:1120px;margin:auto;padding:36px 22px 80px}h1{font-size:30px;line-height:1.3}h2{font-size:23px;margin-top:42px}p{max-width:1000px}.lead,.note{padding:20px 24px;border-radius:10px;background:white;border-left:5px solid #2458bd}.note{border-color:#d26c16}.cards{display:grid;grid-template-columns:1fr 1fr;gap:16px}.card{padding:20px;background:white;border-radius:10px}.card strong{display:block;font-size:20px}figure{margin:22px 0;background:white;border-radius:10px;overflow:hidden}img{display:block;width:100%;height:auto}figcaption{padding:10px 18px;font-size:14px;color:#46566b}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;background:white;font-size:14px}th,td{padding:10px 12px;border-bottom:1px solid #dbe2eb;text-align:right;white-space:nowrap}th{background:#e8eef7}th:first-child,td:first-child{text-align:left}details{margin:14px 0;background:white;padding:14px;border-radius:8px}summary{cursor:pointer;font-weight:bold}code{background:#e8eef7;padding:2px 5px}a{color:#2458bd}.small{font-size:14px;color:#536174}li{margin:8px 0}@media(max-width:650px){.cards{grid-template-columns:1fr}h1{font-size:25px}}
</style><main><p class="small">E01 · 외부 구조물 · 기준 이미지 4장 · 2026-11-01</p>
<h1>주변 픽셀을 몇 개 읽어야<br>경계 위치가 안정적으로 나오는가?</h1>
<div class="lead"><b>상단은 11개를 유지할 근거가 생겼지만, 하단은 표본 수만 정해서 해결됐다고 볼 수 없습니다.</b><br>
같은 초기 경계에서 표본 수를 바꾼 민감도 결과입니다. 실제 정답과의 오차나 θ 보정 성공 판정은 아닙니다.</div>
<h2>무엇을 바꿨나</h2><p>경계 한 점을 구할 때 읽는 표본을 <b>5·7·9·11·13·15·17·19·21개</b>로 변경했습니다. 표본 간격은1px이므로 범위는 ±2~±10px입니다. 3개는 참고용으로만 계산했습니다.</p>
<p>초기 경계는 기존 E01의 열별 양의 Sobel 최대점으로 한 번 정하고 고정했습니다. 새로 논의한 세 가지 선택 방식은 적용하지 않았습니다. 상단과 하단을 따로 비교했고 Huber 직선 피팅은 실행하지 않았습니다.</p>
<p>가로 측정 위치는 장당 상단1,356개·하단1,526개, 총11,528개입니다. 이것이 독립된 촬영 표본11,528개라는 뜻은 아닙니다. 촬영 이미지는4장입니다.</p>
<h2>결과를 읽는 방법</h2><p><b>한 위치의 변화 폭 = 표본 수별 Gaussian 중심 μ 중 최댓값 − 최솟값</b>입니다. 예를 들어5개·11개·21개로 구한 μ가 서로 다르면, 실제 물체가 움직인 것이 아니라 같은 사진의 계산 결과가 설정에 따라 달라진 것입니다.</p>
<p>P95는 비교한 점 중95%의 변화 폭이 그 값 이하라는 뜻입니다. 95% 신뢰구간이나 합격선이 아닙니다. 아래 주 비교는 초기 Gaussian 폭을11px로 고정했고, 공통으로 채택된 동일 위치끼리 비교했습니다.</p>
<div class="cards"><div class="card"><strong>상단: 11개 유지 검토</strong>11~21개 비교에서 이미지별 P95는 약0.000006~0.000008px입니다. 일부 위치에서는 최대0.0158px의 변화가 있습니다. 11개가 유일한 최적값이라는 결론은 아닙니다.</div>
<div class="card"><strong>하단: 표본 수 확정 보류</strong>같은 비교에서 중앙값 약0.52~0.59px, P95 약1.37~1.41px, 최대2.57px로 달라집니다. 21개까지 늘려도 동일 위치가 유지되는 구간을 확보했다고 보기 어렵습니다.</div></div>
'''
    page+=img('window_comparison.png','왼쪽: 표본을 2개 늘렸을 때의 위치 차이. 가로축 7은 5→7개, 11은 9→11개 비교를 뜻합니다. 오른쪽: 수렴하고 기존 ±2px 조건을 만족한 비율. 차이가 작아 보여도 채택된 점이 크게 줄었는지 함께 확인해야 합니다.')
    page+='<h2>11~21개에서 한 점의 위치가 변한 폭</h2><p>5~21개에서 모두 기존 조건을 만족한 위치만 사용했습니다. 상단은총5,397/5,424개, 하단은2,176/6,104개입니다. 특히 하단의 이 표는 상대적으로 상태가 좋은 일부 점에 대한 결과입니다.</p>'
    page+=table(['이미지','대상','공통 비교점 / 전체','변화 폭 중앙값 [px]','P95 [px]','최대 [px]'],range_rows)
    page+=img('spatial_sensitivity.png','로그 축으로 위치별 변화를 표시했습니다. 중앙의 빈 영역은 장착물로 가려져 원래부터 제외한 구간입니다. 한 점씩의 차이이며 최종 직선 각도 차이가 아닙니다.')
    page+='<h2>실제 한 점에서는 어떻게 달라졌나</h2><p>R01에서5~21개 변화 폭을 정렬한 뒤 가운데 순위의 점을 상단·하단 각각 하나씩 표시했습니다. 잘 나온 점만 골라 제시한 예시가 아니며, 전체 분포는 위 표와 그래프에 있습니다.</p>'
    page+=img('profile_examples.png','검정 점은 원본 Sobel 표본, 색 선은 해당 범위에 맞춘 Gaussian입니다. 세로 점선은 각 곡선의 중심 μ입니다. 오른쪽은 같은 점의 표본 수별 위치입니다.')
    page+='<h2>초기값 때문에 생긴 차이도 확인</h2><p>원래 코드는 표본 수를 바꾸면 초기σ와 초기 탐색 간격도 함께 바뀝니다. 이를 분리하기 위해 주 비교는 초기σ=11px, 폭 탐색 간격=5.5px로 고정했습니다. 추가로 초기σ=5와21을 각각 고정한 계열을 계산했습니다. 초기값별 결과를 합치거나 가장 유리한 결과를 선택하지 않았습니다.</p>'
    page+=img('initialization_check.png','같은 표본 수에서도 초기값5·11·21에 따라 중심이 달라지는지 확인했습니다. 상단17~21개에서도 일부 위치가 초기값에 민감하므로, 표본은 클수록 좋다고 판단할 수 없습니다.')
    page+='<div class="note"><b>단순히 수렴했다는 것만으로 경계 위치가 유효하지는 않았습니다.</b> 하단에서는 제약 없는 Gaussian 피팅이 매우 먼 중심을 반환한 사례도 있습니다. 기존±2px 조건에서 제외되며, 원시 결과는 fits.csv에 모두 남겼습니다. 이 숫자를 물리적 경계 오차로 해석하지 않습니다. 기존2px 조건 자체를 정당화한 실험도 아닙니다.</div>'
    page+='<h2>표본 수별 상세 수치</h2>'+''.join(window_sections)
    page+='<h2>이번 결과로 무엇을 결정할 수 있나</h2><ol><li><b>상단:</b> 11개를 다음 단계의 후보로 유지할 수 있습니다. 11개가 최적이거나 실제 경계 위치가 정확하다는 보증은 아닙니다.</li><li><b>하단:</b> 5~21개 중 하나를 현재 결과만 보고 최종 채택하지 않습니다. 선택률만 높다는 이유로5개를 선택하면 표본 수 의존성을 놓치게 됩니다.</li><li><b>미해결 범위:</b> 세로 방향 고정의 적합성, Gaussian 모델이 해당 경계에 맞는지, 초기값·수렴·2px 채택기준은 별도 항목입니다. 이번에는 수정하지 않았습니다.</li></ol>'
    page+='<h2>재현 확인과 파일</h2><ul><li>원본4장의 SHA-256이 기존 목록과 일치했습니다.</li><li>기존과 같은 OpenCV3.4.15, 기존 Nelder–Mead 함수를 그대로 추출해 사용했습니다.</li><li>기존에 저장된120개 프로파일의 Sobel 표본과11개 피팅 중심μ 차이는0이었습니다. 8개그룹 전체의 초기 좌표와 채택 마스크도 일치했습니다.</li><li>수렴 조건800회·10⁻¹⁰과 기존2px 조건은 변경하지 않았으며, 추가적인 통과 기준도 만들지 않았습니다.</li><li>원본 코드·설정·기존 결과 파일의 전후 해시가 동일합니다. 새 결과는 이 폴더에만 저장했습니다.</li></ul>'
    page+='<p><a href="summary.json">요약 수치JSON</a> · <a href="fits.csv">모든 피팅 결과CSV</a> · <a href="profiles.json">원본21개 Sobel 표본</a> · <a href="provenance.json">조건·입력·원본 해시</a> · <a href="report.txt">텍스트 요약</a></p>'
    page+='<details><summary>재실행 순서</summary><pre>python prepare.py\ncmake -S . -B build -G "Visual Studio 18 2026" -A x64 -T v141 -DOpenCV_DIR=C:/Users/SSAFY/opencv/build\ncmake --build build --config Release\nbuild/Release/gaussian_window_sensitivity.exe\npython summarize.py\npython make_report.py</pre><p>이 폴더를 작업 디렉터리로 실행합니다. 같은 이름의 이번 실험 출력만 다시 작성됩니다. 기존 측정 파이프라인은 실행하지 않습니다.</p></details>'
    page+='<h2>근거 자료</h2><p><a href="https://www.annals-csis.org/Volume_2/pliks/136.pdf">Fabijańska(2014), Gaussian-Based Approach to Subpixel Detection of Blurred and Unsharp Edges</a>: Gaussian 중심을 경계 위치로 사용하는 기반 방법. 현재 코드의 11개 고정값 자체를 정당화하는 근거는 아닙니다.</p><p><a href="https://www.researchgate.net/publication/283556095_Subpixel_edge_detection_in_blurry_and_noisy_images">Fabijańska(2015), Subpixel Edge Detection in Blurry and Noisy Images, 그림 3·표 1</a>: 표본 범위 3~21개를 변경하며 중심 위치를 비교한 선행 예. 이번 후보 범위의 출발점이며 보편적인 최적 범위라는 뜻은 아닙니다.</p>'
    page+='</main></html>'
    (HERE/'report.html').write_text(page,encoding='utf-8')
    outputs=['experiment.cpp','prepare.py','summarize.py','make_report.py','original_nelder_mead.hpp',
             'fits.csv','profiles.json','summary.json','paired_arrays.npz','runtime.json',
             'example_provenance.json','report.html','report.txt',
             'window_comparison.png','spatial_sensitivity.png','initialization_check.png','profile_examples.png']
    hashes={name:hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in outputs}
    (HERE/'output_hashes.json').write_text(json.dumps(hashes,indent=2),encoding='utf-8')
    print('Wrote report.html, report.txt and 4 figures. Outputs hashed.')

if __name__=='__main__': main()
