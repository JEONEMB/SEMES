from pathlib import Path
import json,hashlib,html
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, NullFormatter
H=Path(__file__).resolve().parent
NAMES={'qr':'QR만','upper':'QR + 위쪽 선','lower':'QR + 아래쪽 선','both':'QR + 위·아래 선'}
ORDER=['qr','upper','lower','both']

def table(df):return '<div class="scroll">'+df.to_html(index=False,border=0,float_format=lambda v:f'{v:.4f}')+'</div>'
def link(p,t):return f'<a href="{html.escape(p)}">{html.escape(t)}</a>'

def main():
    s=pd.read_csv(H/'static_summary.csv');p=pd.read_csv(H/'static_predictions.csv');st=pd.read_csv(H/'stencil_bounded_summary.csv');sim=pd.read_csv(H/'control/simulation_summary.csv');extra=pd.read_csv(H/'control/exploratory/simulation_summary.csv');cases=pd.read_csv(H/'control/simulation_cases.csv');m=pd.read_csv(H/'manifest.csv')
    sources=json.loads((H/'research/sources.json').read_text(encoding='utf-8'))['sources'];rows=[]
    for cond in ORDER:
        g=s[(s.condition_name==cond)&(s.protocol=='development_oof')];r=g[g.dataset.astype(str)=='all'].iloc[0];new=g[g.dataset.astype(str)=='20261020'].iloc[0]
        rows.append({'관측':NAMES[cond],'전체 Y 평균':r.Y_MAE,'전체 Y 최대':r.Y_max,'전체 T 평균':r.T_MAE,'전체 T 최대':r.T_max,'전체 동시 통과':f'{int(r.passed)}/341','20261020 통과':f'{int(new.passed)}/60'})
    comp=pd.DataFrame(rows);comp.to_csv(H/'comparison.csv',index=False,encoding='utf-8-sig')
    newtable=[]
    for cond in ORDER:
        r=s[(s.condition_name==cond)&(s.protocol=='development_oof')&(s.dataset.astype(str)=='20261020')].iloc[0]
        newtable.append({'관측':NAMES[cond],'Y 평균':r.Y_MAE,'Y 최대':r.Y_max,'T 평균':r.T_MAE,'T 최대':r.T_max,'동시 통과':f'{int(r.passed)}/60'})
    strows=[]
    for cond in ORDER:
        for frames in [1,3,4]:
            r=st[(st.condition==cond)&(st.frames==frames)].iloc[0];strows.append({'관측':NAMES[cond],'사진 수':frames,'알려진 이동 수':frames-1,'Y 평균':r.Y_MAE,'Y 최대':r.Y_max,'T 평균':r.T_MAE,'T 최대':r.T_max,'조합 통과율':f'{100*r.pass_fraction:.1f}%','조합 수':int(r.n)})
    selected_scenario='offset_gain_crossaxis_noise';simrows=[]
    for cond in ORDER:
        for source,policy,label in [(sim,'fixed_jacobian','사전 교정 반응 유지'),(sim,'probe2_broyden','작은 탐색2 + 갱신'),(sim,'probe3_broyden','작은 탐색3 + 갱신'),(extra,'rls_probe2_large','큰 탐색2 + 잡음 제한 RLS')]:
            z=source[(source.condition==cond)&(source.scenario==selected_scenario)&(source.policy==policy)&(source.dataset.astype(str)=='all')]
            if len(z)!=1:raise ValueError((cond,policy,z.shape))
            r=z.iloc[0];simrows.append({'관측':NAMES[cond],'제어 방법':label,'종료 Y 평균':r.Y_MAE,'종료 Y 최대':r.Y_max,'종료 T 평균':r.T_MAE,'종료 T 최대':r.T_max,'종료 오차 통과':f'{int(r.passed)}/341','평균 사용 턴':r.mean_turns,'영상 정지':int(r.visual_stops),'잘못된 영상 정지':int(r.false_visual_stops)})
    simcompare=pd.DataFrame(simrows);simcompare.to_csv(H/'simulation_comparison.csv',index=False,encoding='utf-8-sig')
    failures=p[(p.condition_name=='both')&(p.protocol=='development_oof')&(~p.passed)].copy()
    failtab=failures[['file','Y','T','predY','predT','eY','eT']].rename(columns={'file':'사진','predY':'Y 예측','predT':'T 예측','eY':'Y 오차','eT':'T 오차'});failtab.to_csv(H/'both_failure_cases.csv',index=False,encoding='utf-8-sig')
    plt.rcParams['font.family']='Malgun Gothic';plt.rcParams['axes.unicode_minus']=False
    fig,axs=plt.subplots(1,2,figsize=(12,4.7),layout='constrained')
    x=np.arange(4);colors=['#7c8fa3','#df9f34','#4488b8','#168c80']
    axs[0].bar(x,[int(r['전체 동시 통과'].split('/')[0]) for r in rows],color=colors);axs[0].set_xticks(x,[NAMES[c] for c in ORDER]);axs[0].set_ylim(0,360);axs[0].set_title('한 장 · 전체341 개발 OOF');axs[0].set_ylabel('두 축 허용오차 동시 통과 수')
    for i,c in enumerate(ORDER):
        z=st[st.condition==c].sort_values('frames');axs[1].plot(z.frames,z.pass_fraction*100,'o-',label=NAMES[c],color=colors[i])
    axs[1].set_xticks([1,3,4],['사진1장\n이동0','사진3장\n이동2','사진4장\n이동3']);axs[1].set_ylim(0,100);axs[1].set_title('기존48장 조합 · 실제 제어 궤적 아님');axs[1].set_ylabel('조합의 두 축 허용오차 통과율 (%)');axs[1].legend();axs[1].grid(alpha=.2)
    fig.savefig(H/'comparison.png',dpi=150);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(12,4.3),layout='constrained')
    for j,(source,policy,label) in enumerate([(sim,'fixed_jacobian','사전 교정'),(sim,'probe2_broyden','작은 탐색2'),(sim,'probe3_broyden','작은 탐색3'),(extra,'rls_probe2_large','큰 탐색2+RLS')]):
        z=source[(source.scenario==selected_scenario)&(source.policy==policy)&(source.dataset.astype(str)=='all')].set_index('condition').loc[ORDER]
        axs[0].bar(x+(j-1.5)*.2,z.passed/341*100,width=.2,label=label)
        axs[1].plot(x,z.Y_max,'o-',label=label)
    axs[0].set_ylim(0,100);axs[0].set_ylabel('20턴 종료 시 가정 정답 통과율 (%)');axs[1].axhline(.09,c='#d34f39',ls='--');axs[1].set_ylabel('종료 Y 최대오차');axs[1].set_yscale('log')
    axs[1].yaxis.set_major_formatter(FuncFormatter(lambda value,pos:f'{value:g}'))
    axs[1].yaxis.set_minor_formatter(NullFormatter())
    for ax in axs:ax.set_xticks(x,[NAMES[c] for c in ORDER]);ax.grid(axis='y',alpha=.2);ax.legend(fontsize=9)
    fig.suptitle('가정 시뮬레이션만: 원점 차이 + 축 이득/간섭 + 반복 잡음');fig.savefig(H/'simulation.png',dpi=150);plt.close(fig)
    before=json.loads((H/'preservation_before.json').read_text());changes=[]
    for rel,digest in before['files'].items():
        f=H.parent/rel
        if not f.exists() or hashlib.sha256(f.read_bytes()).hexdigest()!=digest:changes.append(rel)
    assert not changes,changes
    (H/'preservation_after.json').write_text(json.dumps(dict(checked=len(before['files']),changed=changes),indent=2))
    initial=int(np.all(abs(m[['Y','T']].to_numpy())<=.09+1e-12,axis=1).sum())
    sourcehtml=''.join(f'<li><a href="{html.escape(a["url"])}">{html.escape(a["title"])}</a><p>{html.escape(a["use_here"])}<br><small>{html.escape(a["limit"])}</small></p></li>' for a in sources)
    page=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>QR·가로선 4조건과 최대20턴 비전 피드백</title><style>body{{margin:0;background:#f3f6f8;color:#193241;font:16px/1.75 'Malgun Gothic',sans-serif}}main{{max-width:1160px;margin:auto;padding:38px 26px}}h1{{font-size:36px;line-height:1.35}}h2{{margin-top:48px;border-top:1px solid #cbd6df;padding-top:25px}}.tag{{font-weight:700;color:#087b70}}.lead{{font-size:20px}}.box{{background:#e1f1ed;border-left:5px solid #168c80;padding:20px;margin:22px 0}}.warn{{background:#fff0dc;border-color:#be7530}}.cards{{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}}.card{{padding:20px;background:white;border-radius:10px}}.big{{font-size:29px;font-weight:800}}small,.muted{{color:#546c7c}}a{{color:#086f92}}table{{border-collapse:collapse;width:100%;font-size:14px;background:white}}th,td{{border-bottom:1px solid #dfe7ec;padding:10px;text-align:right}}th:first-child,td:first-child{{text-align:left}}th{{background:#e7edf1}}.scroll{{overflow:auto}}img{{max-width:100%;height:auto}}details{{padding:17px;background:white;border-radius:8px;margin:18px 0}}summary{{cursor:pointer;font-weight:bold}}.flow{{background:#1b3444;color:#edf8fb;padding:24px;border-radius:8px;font-size:18px}}code{{background:#dfe8ed;padding:2px 5px}}footer{{margin-top:40px;color:#536e7c}}li{{margin:12px 0}}li p{{margin:0;font-size:14px}}@media(max-width:700px){{.cards{{grid-template-columns:1fr}}main{{padding:20px 12px}}h1{{font-size:28px}}}}</style><main><div class="tag">2026.10.23 · 사용자 허용 범위 반영 · 볼트·너트 제외</div><h1>위쪽·아래쪽 가로선을 따로 검증하고,<br>2~3회 탐색과 최대20턴 피드백을 비교했습니다.</h1>
<p class="lead"><b>가로선은 Y/T 구분에 도움이 됐습니다.</b> 한 장의 전체 평가에서는 위·아래를 함께 사용했을 때 가장 많이 통과했습니다. 여러 관측을 쓰는 방식도 일부 조건에서 개선됐지만, 작은 탐색 이동은 잡음 때문에 오히려 나빠질 수 있었습니다.</p>
<div class="cards"><div class="card">관측 비교<div class="big">4조건</div><small>QR만 / 위 / 아래 / 양쪽</small></div><div class="card">한 장 · QR+양쪽<div class="big">337 / 341</div><small>개발 OOF 두 축≤0.09, 전수 달성 아님</small></div><div class="card">실제 로봇20턴 시험<div class="big">미실행</div><small>정적 사진 비교와 가정 시뮬레이션만</small></div></div>
<div class="box warn"><b>세 종류의 결과를 구분해야 합니다.</b><br>① 실제341장으로 한 장 추정 평가 ② 그중48장을 묶어 상대 명령 차이를 알려준 여러 사진 추정 ③ 별도 수식 환경에서 제어한 시뮬레이션. ②는 실제 연속 촬영이 아니고, ③은 첫 관측부터 수식이 생성한 특징입니다. 새 촬영·실제 센터 복귀 성공률로 표현하지 않습니다.</div>
<h2>1. 허용된 선만 읽었습니다</h2><p>기존 외부5영역 모델은 사용하지 않았습니다. 위·아래 가로선 각각 왼쪽12개+오른쪽12개의 좁은 구간을 고정했습니다. 각64px 폭에서 수직 밝기 프로파일을 평균하고, 기준 선에 대한 수직 이동만 연속값으로 측정했습니다. 볼트·너트·중앙 장치·손잡이는 선 측정 마스크 밖입니다. 일반 배경 텍스처로 가로 이동을 추적하지 않습니다.</p>
<img src="lines/center_mask_overview.png" alt="기준사진 위아래 가로선 측정 마스크"><p class="muted">전체3840×2160 기준사진의 축소 개요. {link('lines/center_mask_strips_1x.png','실제1배 선 구간 그림')} · {link('lines/read_mask.png','실제 선 읽기 마스크')} · {link('lines/protocol.json','고정 좌표·실패 기준')}</p><p>두 선 모두341/341장 측정에 성공했습니다. 마스크 밖을0 또는 무작위 픽셀로 바꿔도 선 측정값·품질 지표가 완전히 동일함을 확인했습니다. 이것은 <b>사용 영역과 측정 재현성</b> 검사이며 자세 정확도 검사는 아닙니다. 긴 가로선은 수평 방향으로 미끄러지는 움직임 자체를 구분하기 어렵고, 주로 높이·기울기 변화가 정보를 줍니다.</p>
<h2>2. 한 장 추정 — 같은 파이프라인의4조건</h2>{table(comp)}<p>고정4fold와 내부 선택을 사용했고, 혼합 명령의 반복3장을 같은 시험fold로 묶었습니다. 모든 관측 조건에 같은 QR 직접 정합·같은 후보6개를 적용했습니다. 실패는 분모에 포함하며 이번4조건은 측정 실패0입니다. Y/T 모두0.09 이하여야 통과합니다.</p>
<p class="muted">여기서 QR만은 새 비교용54좌표 표현·공통 교정 절차입니다. 이전 QR630 개선 후보는 전체244/341,20261020은53/60이었습니다. 새 QR 단독188/341을 기존 최고 성적으로 대체하지 않습니다. 이번 선 추가 효과는 공통 파이프라인 안의 비교이며, 과거 QR 후보와도 별도로 대조해야 합니다.</p>
<details><summary>20261020의60장만 보기</summary>{table(pd.DataFrame(newtable))}<p>양쪽 선을 사용해도4장이 Y 허용오차를 넘습니다. 재대입60/60은 이 OOF 성적을 대신하지 않습니다.</p></details>
<details><summary>QR+위·아래에서 남은4장</summary>{table(failtab)}</details>
<h2>3. 왜 선이 구분을 돕나</h2><p>각 학습fold에서 관측의 Y/T 반응을 비교했습니다. 단위 길이로 정규화한 반응 방향의 조건수는 QR만 약831, 위쪽98, 아래쪽50, 양쪽44였습니다. Y=0.09 변화를 T로 최대한 상쇄한 뒤 남는 픽셀 차이 RMS는 QR만0.00221px, 위쪽0.01554px, 아래쪽0.03072px, 양쪽0.03005px입니다.</p><p>가로선 좌우 높이 차이는 QR의 공통 가로 이동과 다른 반응을 보여 구분 단서를 추가했습니다. 실제 깊이·로봇 회전축을 복원한 결과는 아닙니다. 반복 잡음까지 고려한 단순 신호 대 잡음 진단은 약1.2~1.4수준이라, 기하 조건이 좋아졌다는 사실만으로0.09를 보장할 수 없습니다. {link('diagnostics/separation_summary.csv','학습fold 진단')} · {link('diagnostics/observed_line_shifts.png','실제 선 높이 반응')}</p>
<h2>4. 사진을3~4장 보면 달라지나</h2><div class="flow">초기 관측 I₀ → 알려진 Y 이동 → I₁ → 알려진 T 이동 → I₂<br>Jᵧ ≈ (s₁−s₀)/ΔY,　Jₜ ≈ (s₂−s₁)/ΔT<br>필요하면3번째 탐색 → 영상 오차를 줄이는 작은 보정 → 재촬영·반응 갱신</div><p>이 방법은 두 축의 국소 영상 반응을 알아내는 데 도움이 됩니다. 다만 선형 모델 s=Aq에서는 이동에 따른 차이가 Δs=A·u여서 초기 q와 무관합니다. A의 두 열이 같은 방향이면 탐색만으로 모호성이 사라지지 않습니다. 관측의 차이와 충분한 탐색 신호가 모두 필요합니다. {link('research/sources.json','관련 원문7개와 적용 범위')}</p>
<p>실제 혼합 사진의4×4명령 격자에서 인접 사각형9개, 시작점4개씩 총36개 경로 설정을 만들었습니다. 해당 사각형의4조건·반복12장을 전부 학습에서 제외한 뒤, 같은 교정 모델로1/3/4장을 비교했습니다. 3장은 이동2회,4장은 이동3회에 해당하는 <b>기존 사진 조합</b>입니다. 모든 반복 조합을 사용했으며 총48장으로부터 만든 서로 의존적인 조합입니다.</p>
<img src="comparison.png" alt="한장4조건과여러정적사진조합비교"><details><summary>사진 조합의 평균·최대오차와 통과율</summary>{table(pd.DataFrame(strows))}</details>
<p>위쪽 선은 조합 통과율68.5%→80.9%→82.0%, 양쪽 선은68.5%→80.0%→80.9%로 개선됐습니다. QR만은21.3%→14.1%→11.1%로 나빠졌습니다. 따라서 <b>관측을 늘리는 것만으로 해결되지는 않습니다.</b> 이 비교는 일부 학습 조건을 더 엄격히 제거했고 고정한2차 교정 모델을 쓰므로 앞의341장 성적과 직접 같은 분모로 비교할 수 없습니다.</p>
<p class="muted">정답은 평가기에서 기존 사진과 상대 명령 차이를 구성하는 데만 사용했습니다. 추론에는 영상 특징과 누적 상대 이동만 전달했습니다. 저장 격자의 이동 간격은 Y0.05~0.5,T0.09~1로, 현장에 권장하는 작은 탐색량이 아닙니다. 해당 사진을 얻을 행동을 실제 미지의 자세에서 선택한 실험도 아닙니다. 프레임수에 따른 solver 차이를 감사에서 수정했고, 모두 같은 solver를 사용했습니다. 무제한 역산의 큰 외삽 오류를 확인한 뒤 학습 범위 기반 제한을 추가한 보조 결과이며, {link('stencil_summary.csv','무제한 결과')}도 보존했습니다. {link('stencil_bounded_protocol.json','상세 조건')}</p>
<h2>5. 최대20턴 피드백 — 가정 시뮬레이션</h2><p>제어기는 학습fold의 영상 반응·잡음만 사용하고, 환경은 별도의 시험fold로 맞춘2차 순방향 수식입니다. 환경이 실제 로봇을 정확히 모사한다는 검증은 없습니다. 초기 사진부터 이 수식이 만든 특징을 사용했습니다. 무잡음·원점 일치, 반복 잡음, 원점 차이+축 이득/간섭, 추가 드리프트+백래시의4가정을 비교했습니다.</p><p>첫 관측은0턴, 탐색·보정·확인 재촬영을 모두 합쳐 최대20턴입니다. 제어기는 실제 정답을 보지 않고 영상 오차로 정지하며, 종료 후 평가기만 가정 정답 Y/T를 검사합니다.341개 시작 상태 중 {initial}개는 시작부터 두 축0.09 안에 있습니다. 이들을 움직여 밖으로 보내는 실패도 최종 평가에 포함합니다.</p>
<div class="box warn"><b>작은 탐색2~3회를 고정적으로 추가한 방식은 잡음에서 악화했습니다.</b><br>사전 작은 탐색량 Y0.05,T0.01에서 두 축을 구분하는 변화/차분 잡음 비가1보다 낮았습니다. 잡음으로 추정한 J를 그대로 쓰면 잘못된 방향으로 보정할 수 있습니다. 다음 숫자는 실제 복귀 성적이 아닙니다.</div>
<p>아래는 원점 차이+축 이득/간섭+반복 잡음 가정입니다. 처음3정책의 결과를 본 후 탐색량Y0.20,T0.04와 잡음·여기량 검사를 넣은 RLS3후보를 보조 실험으로 추가했습니다. 이 숫자는 시뮬레이션용 설정이며 실제 장치의 허용 이동량이나 T 물리 각도 단위가 아닙니다.</p>
<img src="simulation.png" alt="시뮬레이션성공수와최대Y오차"><details open><summary>20턴 종료 결과 — 실제 로봇 시험 아님</summary>{table(simcompare)}</details>
<p><b>양쪽 선+보완 탐색2회는254→262/341로 통과 수가 늘었지만, Y 최대오차는0.1651→0.2496으로 악화했습니다.</b> 세 번째 탐색이 항상 더 좋지도 않았습니다. 충분히 큰 변화만 갱신에 쓰는 보완안은 폭주를 줄였으나 전수0.09 목표는 달성하지 못했습니다.</p>
<p>양쪽 선의 위 조건에서는 평균20턴을 모두 사용하고 영상 기준 정지0건이었습니다. 종료 시 가정 정답이0.09 안에 있는 것과, 제어기가 센터라고 확인하고 일찍 멈추는 것은 다릅니다. 무잡음·원점 일치 가정의 사전 교정 방식은341/341,평균약5.6턴이지만 이는 이상적인 수식 환경 결과입니다. 실제 생산 시간을 초로 환산하지 않았습니다.</p>
<details><summary>전체 시뮬레이션·재현 파일</summary><p>{link('control/protocol.json','처음 고정한정책·4시나리오')} · {link('control/simulation_summary.csv','처음3정책 전체성적')} · {link('control/simulation_cases.csv','개별초기상태·최초진입턴·정지판정')} · {link('control/simulation_audit.json','환경적합오차·학습분할')} · {link('control/exploratory/protocol.json','보완정책설정')} · {link('control/exploratory/simulation_summary.csv','보완3정책전체성적')}</p><p>환경의2차 근사 잔차가 반복 잡음보다 큰 구간도 있습니다. 이상 환경의341/341이나 가정된 백래시 결과를 실물 로봇 보장으로 사용하지 않습니다.</p></details>
<h2>6. 현재 제공한 실행물과 다음 실측의 기준</h2><p>{link('predict.py','한 장 추론 CLI')}는 허용된4조건만 선택할 수 있습니다. {link('feedback_session.py','피드백 세션 CLI')}는 현재 영상과 실제로 실행한 상대 명령을 받아 다음 보정 제안을 반환하고, 하드웨어 명령을 전송하지 않습니다. 중심 영상·관측 추출·교정·20턴 카운터를 연결했고 기존 이미지의 API 재현을 검사했습니다. 실제 장치에 연결하기 전 이동 한계·방향·정착 조건은 확인이 필요합니다.</p>
<p>실측에서는 같은 초기 자세와 반복 수로4관측×정책을 비교하고, 초기 관측부터 모든 영상·실행 명령·촬영/계산/이동/정착 시간을 저장해야 합니다. 독립 계측의 종료 잔여 Y/T,두 축≤0.09 통과율,최대오차,종료/실패 턴 분포,총 소요 시간,잘못된 정지를 평가합니다. 이번에 추가 촬영이나 실제 로봇 구동은 하지 않았습니다.</p>
<p>{link('static_predictions.csv','사진별한장예측')} · {link('static_selection.json','내부후보선택')} · {link('image_api_verification.json','이미지API재현')} · {link('lines/verification.json','금지영역배제검사')} · {link('research/review.json','정적·다중사진코드감사')} · {link('preservation_after.json','기존257파일보존검사')} · {link('../../ALGORITHM_RULES.md','최신공통규칙')}</p>
<h2>7. 근거 자료</h2><details><summary>원문·저자 자료7개와 적용 범위</summary><ol>{sourcehtml}</ol></details><footer>현재341장은 모두 개발에서 이미 본 자료입니다. 정답은 파일명 명령값이며 실제 자세의 독립 계측값이 아닙니다. 외부 볼트·너트·브래킷·손잡이는 사용하지 않았습니다. 가로선 범위 허용은 이전 금지된 외부5영역 실험을 재승인한 것이 아닙니다. 기존 결과는 보존했습니다.</footer></main></html>'''
    page=page.replace('<h2>7. 근거 자료</h2>', '<p>'+link('control/results_verification.json','32,736건 제어 결과 전수 산술 검증')+' · '+link('control/independent_review.json','제어 분할·온라인 입력·궤적 재현 독립 코드 감사')+'</p><h2>7. 근거 자료</h2>')
    (H/'report.html').write_text(page,encoding='utf-8');print('report built; prior files preserved',len(before['files']))

if __name__=='__main__':main()
