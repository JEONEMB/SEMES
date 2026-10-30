"""Human-readable leakage audit, all-ten failure analysis and complete comparisons."""
from pathlib import Path
import json,hashlib,html,re
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
H=Path(__file__).resolve().parent;OLD=H.parent/'joint_yt_disentangle_20261022'
plt.rcParams.update({'font.family':'Malgun Gothic','axes.unicode_minus':False,'font.size':10})
E=html.escape

def table(frame):return frame.to_html(index=False,escape=False,border=0,classes='data',float_format=lambda v:f'{v:.4f}')
def href(path,label):return f'<a href="{E(str(path))}">{E(label)}</a>'
def bullets(items):return '<ul>'+''.join('<li>'+E(x)+'</li>' for x in items)+'</ul>'

def main():
    previous=pd.read_csv(OLD/'summary.csv');current=pd.read_csv(H/'summary.csv');robust=pd.read_csv(H/'robust/exploratory_nested_summary.csv');phase=pd.read_csv(H/'phase_summary.csv')
    compare=pd.read_csv(H/'robust/exploratory_nested_comparison.csv').set_index('index');d=pd.read_csv(H/'diagnostics/all60_diagnostics.csv').set_index('index')
    cases=json.loads((H/'diagnostics/per_image_analysis.json').read_text(encoding='utf-8'));inventory=json.loads((H/'audit/source_inventory.json').read_text(encoding='utf-8'));diagnostic=json.loads((H/'diagnostics/audit.json').read_text(encoding='utf-8'))
    before=json.loads((H/'preservation_before.json').read_text())['files'];changed=[]
    for name,sha in before.items():
        p=H.parent/name
        if not p.exists() or hashlib.sha256(p.read_bytes()).hexdigest()!=sha:changed.append(name)
    preservation=dict(files_checked=len(before),changed=changed,all_preserved=not changed)
    (H/'preservation_after.json').write_text(json.dumps(preservation,indent=2))
    assert not changed,changed

    def select(frame,ds,protocol,method=None):
        mask=(frame.dataset.astype(str)==str(ds))&(frame.protocol==protocol)
        if method is not None:mask&=frame.method==method
        return frame[mask].iloc[0]
    definitions=[('직전 선택 절차',previous,'nested_group_oof','pass_first'),('사전 후보 4개 추가: 주 비교',current,'nested_development_oof','extended'),('사전 후보 4개만 선택',current,'nested_development_oof','robust_only'),('추가 탐색 16개: 개선 후보',robust,'exploratory_nested16_oof',None),('주기성 보정 가설 비교',phase,'nested_development_oof',None)]
    metricrows=[]
    for name,frame,protocol,method in definitions:
        r=select(frame,20261020,protocol,method);old=select(frame,20261014,protocol,method)
        metricrows.append({'개발 평가':name,'Y 평균오차':r.Y_MAE,'Y 최대오차':r.Y_max,'T 평균오차':r.T_MAE,'T 최대오차':r.T_max,'두 축 동시 통과':f'{int(r.passed)}/60','기존 281장 통과':f'{int(old.passed)}/281'})
    metricframe=pd.DataFrame(metricrows);metricframe.to_csv(H/'comparison_summary.csv',index=False,encoding='utf-8-sig')
    n=compare[compare.dataset==20261020];failing=n[~n.passed_candidate];failing.to_csv(H/'remaining_failures.csv',encoding='utf-8-sig')
    original_failures=n[~n.passed_prior].copy();original_failures.to_csv(H/'ten_before_after.csv',encoding='utf-8-sig')
    n.to_csv(H/'all60_before_after.csv',encoding='utf-8-sig')

    fig,axes=plt.subplots(1,2,figsize=(14,6),gridspec_kw={'width_ratios':[1,1.4]})
    colors=np.where(n.transition=='recovered','#15966d',np.where(n.transition=='new_failure','#d04c42','#426b9e'))
    axes[0].scatter(n.eY_prior,n.eY_candidate,c=colors,s=43,alpha=.85)
    axes[0].plot([0,.28],[0,.28],color='#a5afba',lw=1);axes[0].axhline(.09,color='#d04c42',ls='--');axes[0].axvline(.09,color='#d04c42',ls='--')
    axes[0].set(xlabel='직전 Y 절대오차',ylabel='개선 후보 Y 절대오차',title='전체 60장 개발 비교 · 아래쪽일수록 개선',xlim=(0,.28),ylim=(0,.28));axes[0].grid(alpha=.15)
    ids=[int(c['index']) for c in cases]+[335];rr=n.loc[ids];yy=np.arange(len(rr))
    for j,(_,r) in enumerate(rr.iterrows()):axes[1].plot([r.eY_prior,r.eY_candidate],[j,j],color='#b8c3ce',lw=2)
    axes[1].scatter(rr.eY_prior,yy,c='#8a97a6',label='직전',s=38,zorder=3);axes[1].scatter(rr.eY_candidate,yy,c=np.where(rr.passed_candidate,'#15966d','#d04c42'),label='개선 후보: 초록 통과 / 빨강 실패',s=48,zorder=4)
    axes[1].axvline(.09,color='#d04c42',ls='--');axes[1].set_yticks(yy,[f'Y{r.Y:g} T{r["T"]:g} · 반복{int(d.loc[i,"repeat"])}'+(' (새 실패)' if i==335 else '') for i,r in rr.iterrows()]);axes[1].invert_yaxis();axes[1].set(xlabel='Y 절대오차',title='기존 실패 10장 + 새로 실패한 1장');axes[1].legend(loc='lower right',fontsize=8);axes[1].grid(axis='x',alpha=.15)
    fig.suptitle('독립 테스트 아님 · 같은 341장을 반복 사용한 개발 평가',fontsize=14,color='#9d3c2b');fig.tight_layout();fig.savefig(H/'error_comparison.png',dpi=150);plt.close(fig)

    tenrows=[];cards=[]
    for c in cases:
        i=c['index'];r=compare.loc[i];e=d.loc[i];diag=c['diagnosis'];status='회복' if r.passed_candidate else '실패 유지'
        tenrows.append({'사진':href('#case'+str(i),f'Y{r.Y:g} / T{r["T"]:g} / 반복{int(e["repeat"])}'),'직전 Y 오차':r.eY_prior,'개선 Y 오차':r.eY_candidate,'개선 T 오차':r.eT_candidate,'변화':status,'확인된 오류 양상':E(diag['mechanism'])})
        peers=pd.DataFrame(c['other_repeats'])[['file','predY','predT','eY','eT','passed']];peers.columns=['같은 명령의 다른 사진','직전 Y 추정','직전 T 추정','Y 절대오차','T 절대오차','통과']
        source=Path(e.source_path).as_uri()
        cards.append(f'''<article class="case" id="case{i}"><h3>{E(c['file'])} <span class="pill {'ok' if r.passed_candidate else 'bad'}">{status}</span></h3>
<p class="lead">{E(diag['mechanism'])}</p><p>직전 Y 추정 <b>{r.predY_prior:.4f}</b> → 개선 후보 <b>{r.predY_candidate:.4f}</b> / 명령 Y={r.Y:g}, T={r['T']:g}</p>
<p>직전 Y 부호오차 <b>{e.signed_error_Y:+.4f}</b> = 같은 명령 3장의 평균 편향 <b>{e.condition_bias_Y:+.4f}</b> + 이 사진의 평균 대비 편차 <b>{e.individual_pred_deviation_Y:+.4f}</b></p>
<h4>관측한 근거</h4>{bullets(diag['observations'])}<h4>가능한 원인과 한계</h4>{bullets(diag['hypotheses'])}
<div class="scroll">{table(peers)}</div><p class="small">같은 명령의 평균은 사후 진단용입니다. 시험 예측 보정에 사용하지 않았습니다. {E(c['limit'])}</p><a href="{E(source)}">원본 사진 열기</a></article>''')
    train=select(robust,20261020,'resub341');oldrob=select(robust,20261014,'exploratory_nested16_oof');newrob=select(robust,20261020,'exploratory_nested16_oof')
    result=dict(independent_test_performed=False,reason='User confirmed no additional data; all341 images already exposed during development',direct_fold_label_leak_found=False,adaptive_evaluation_contamination=True,primary_prior_plus_preregistered4_passed=50,exploratory_nested16_passed=int(newrob.passed),exploratory_nested16_Y_MAE=float(newrob.Y_MAE),exploratory_nested16_Y_max=float(newrob.Y_max),exploratory_nested16_T_MAE=float(newrob.T_MAE),exploratory_nested16_T_max=float(newrob.T_max),recovered_old_failures=4,remaining_old_failures=6,new_failures=1,old281_passed=int(oldrob.passed),development_goal_met=False,independent_generalization_goal_status='NOT TESTED',production_algorithm_replaced=False,preserved_prior_files=len(before),candidate_model='robust/exploratory_nested_model.joblib',candidate_model_sha256=hashlib.sha256((H/'robust/exploratory_nested_model.joblib').read_bytes()).hexdigest())
    (H/'result.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    doc=f'''<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Y/T 실패 10장 분석 · 평가 누출 감사</title>
<style>*{{box-sizing:border-box}}body{{margin:0;background:#f3f6fa;color:#1d2d3d;font:16px/1.65 "Malgun Gothic",sans-serif}}main{{max-width:1260px;margin:auto;padding:40px 32px 70px}}h1{{font-size:36px;line-height:1.3;margin:12px 0}}h2{{font-size:25px;margin:0 0 15px}}h3{{font-size:20px}}h4{{margin:15px 0 4px}}p{{margin:10px 0}}a{{color:#1b64ad}}section,.case{{background:white;border:1px solid #d9e2eb;border-radius:13px;padding:25px;margin:22px 0;box-shadow:0 3px 12px #18345006}}.eyebrow{{color:#5d7185;font-size:13px;letter-spacing:.08em}}.notice{{border-left:5px solid #cf6439;background:#fff3e9;padding:18px 22px;margin:24px 0}}.cards{{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}}.stat{{border:1px solid #d5dfe9;background:white;border-radius:12px;padding:20px}}.stat b{{display:block;font-size:30px;color:#183b62}}.stat span,.small{{font-size:13px;color:#617285}}.scroll{{overflow:auto}}table{{border-collapse:collapse;width:100%;font-size:13px}}th{{background:#eaf0f7;text-align:left;white-space:nowrap}}td,th{{border-bottom:1px solid #e0e7ee;padding:10px;vertical-align:top}}td{{min-width:65px}}.dataframe td:nth-last-child(1){{min-width:95px}}img{{width:100%;height:auto;border:1px solid #e1e7ed;border-radius:8px}}.pill{{font-size:12px;border-radius:30px;padding:4px 10px;white-space:nowrap}}.ok{{color:#086744;background:#dcf4e8}}.bad{{color:#a4322b;background:#ffe8e2}}.lead{{font-weight:600}}nav{{display:flex;flex-wrap:wrap;gap:16px;margin:22px 0}}li{{margin:7px 0}}.case{{border-left:4px solid #7a9abf}}code{{font-family:Consolas,monospace;background:#edf2f7;padding:2px 5px}}@media(max-width:750px){{main{{padding:20px 12px}}h1{{font-size:28px}}.cards{{grid-template-columns:1fr}}section,.case{{padding:17px}}}}</style></head>
<body><main><div class="eyebrow">2026-10-22 · 기존 알고리즘 보존 · 새 실험 폴더</div><h1>틀린 10장을 분석하고,<br>성능 평가의 범위를 바로잡았습니다.</h1>
<p>목표: 사진 한 장과 고정 센터 사진으로 Y/T를 추정하고, 두 축 오차를 모두 0.09 이하로 줄이기.</p>
<div class="notice"><b>현재 341장은 모두 개발 자료입니다.</b> 오답과 교차검증 점수를 반복 관찰해 방법을 바꿨으므로, 아래 점수를 독립적인 새 사진 성능으로 해석할 수 없습니다. 사용자가 추가 데이터가 없다고 확인했습니다. 이번 작업에서 새 촬영 성능은 <b>미평가</b>입니다.</div>
<div class="cards"><div class="stat"><span>개발 탐색 비교 · 두 축 동시 통과</span><b>50 → 53 / 60</b><span>기존 실패 4장 회복 · 새 실패 1장</span></div><div class="stat"><span>개선 후보 · 최대 Y 오차</span><b>{newrob.Y_max:.4f}</b><span>목표 0.09 초과 · 목표 미달</span></div><div class="stat"><span>결과를 보기 전 고정한 주 비교</span><b>50 → 50 / 60</b><span>후보 추가만으로 개선되지 않음</span></div></div>
<nav><a href="#leak">평가 누출</a><a href="#metrics">전체 성능 비교</a><a href="#ten">10장 개별 분석</a><a href="#change">추가한 방법</a><a href="#future">독립 평가 준비</a></nav>
<section id="leak"><h2>데이터 누출 지적에 대한 결론</h2>
<p><b>평가 오염이 있었습니다.</b> 확인한 코드에서는 외부 시험 사진의 명령을 해당 모델의 학습·정규화·공분산 계산에 직접 넣거나, 파일명을 추론에 제공한 문제는 발견하지 못했습니다. 그러나 같은 외부 시험 성적과 실패 사진을 보고 모델 계열·특징을 반복 변경했습니다. 이 정보가 개발에 들어갔으므로, 교차검증이라는 이름만으로 독립 평가가 되지 않습니다.</p>
<p>테스트 결과로 모델을 선택하면 성능 추정이 낙관적으로 치우칠 수 있습니다. <a href="https://scikit-learn.org/stable/common_pitfalls.html#data-leakage">scikit-learn 공식 문서</a>와 <a href="https://jmlr.org/papers/v11/cawley10a.html">Cawley·Talbot의 모델 선택 편향 연구</a>에 근거해 기존 점수를 개발 평가로 한정했습니다.</p>
<ul><li>같은 Y/T 명령의 새 사진 3장은 같은 외부 fold에 두었습니다. 새 센터 3장도 하나의 fold로 묶어 학습에서 제외했습니다.</li><li>각 fold의 내부 선택·공분산·사진 쌍·반복 잡음은 그 fold의 학습 부분에서만 계산했습니다.</li><li>분할을 새로 만들거나 파일명을 바꿔도 이미 본 사진이 독립 테스트로 바뀌지는 않습니다.</li><li>학습한 60장을 다시 넣은 60/60은 재대입 성적입니다. 새 사진 성능의 근거로 사용하지 않았습니다.</li></ul>
<p class="small">근거 파일: <a href="audit/evaluation_audit.json">코드·분할 감사</a> · <a href="audit/source_inventory.json">원시 사진 중복 조사</a></p></section>
<section id="metrics"><h2>방법을 골랐던 과정까지 포함한 개발 성능</h2><p>20261020: 혼합 57장 + 센터 3장. 한 장에서 |Y 오차| ≤ 0.09 그리고 |T 오차| ≤ 0.09일 때만 통과입니다. 표의 오차 정답은 파일명 명령값이며 실제 로봇 자세 계측값은 없습니다.</p>
<div class="scroll">{table(metricframe)}</div><p><b>주 비교는 개선 없음(50/60).</b> 사전 지정 4개 강건화 후보를 기존 15개 후보와 같은 내부 분할에서 비교했으나, 외부 fold의 선택 모델은 기존과 같았습니다. 그 뒤 16개 후보를 내부에서 선택하는 추가 탐색을 했고 53/60이 나왔습니다. 탐색 시점과 후보를 모두 기록했습니다.</p>
<p>개선 후보는 기존 실패 10장 중 <b>4장을 회복</b>했고, 나머지 6장은 계속 실패했습니다. 기존 통과 50장 중 <b>Y=1,T=2 반복1이 새로 실패</b>했습니다(Y 오차 0.0653 → 0.1048). 기존 20261014의 281장은 192 → 191장으로 소폭 나빠졌습니다. 전체 341장 통과는 242 → 244장입니다.</p>
<img src="error_comparison.png" alt="전체60장과 기존실패10장 및 새실패1장의 오차 비교"><p class="small">그래프는 개발 OOF 비교입니다. 조건별 테스트 사진은 그 fold 모델의 학습에서 제외했지만, 같은 데이터에 대한 반복적인 개발 노출은 이미 있었습니다.</p>
<details><summary>학습 재대입 성적 — 새 사진 평가와 구분</summary><p>새 최종 후보로 학습했던 60장을 다시 추정하면 {int(train.passed)}/60, Y 평균 {train.Y_MAE:.4f} / 최대 {train.Y_max:.4f}, T 평균 {train.T_MAE:.4f} / 최대 {train.T_max:.4f}입니다. 이 수치로 새 사진의 0.09 이내를 주장하지 않습니다.</p></details></section>
<section id="ten"><h2>실패한 10장을 하나씩 비교했습니다</h2><p>이 10장은 모두 Y만 기준을 넘었고, T는 기준 이내였습니다. QR 측정 실패로 처리된 사진은 없습니다. 아래의 원인은 저장된 추정기의 오차 양상에 대한 진단입니다. 실제 기계 반복 오차와 영상 측정 오차를 확정적으로 구분할 별도 자세 계측은 없습니다.</p>
<div class="scroll">{table(pd.DataFrame(tenrows))}</div><p class="small">사진 이름을 누르면 같은 명령의 다른 반복 사진, 오차 분해와 구체적인 근거로 이동합니다.</p></section>
<section><h2>10장에서 공통으로 확인한 분리 문제</h2><p>학습 자료에서 추정한 Y/T의 원좌표 이동 방향 cosine은 약 <b>0.999993</b>으로 매우 비슷합니다. 전체 60장의 Y/T 부호오차 상관은 <b>−0.9572</b>였습니다. Y를 높게 추정할 때 T를 약간 낮게 추정하는 식으로 축 사이의 오차가 함께 움직였습니다.</p>
<p>최악 사진 <b>Y=1,T=2 반복2</b>의 Y 오차는 −0.2574입니다. 같은 명령 3장의 평균 편향 −0.0751과 해당 사진의 평균 대비 편차 −0.1823으로 나뉩니다. 해당 편차의 일차 근사에서는 전체 공통 이동 +0.0061, QR 사이 상대 이동 −0.0422, QR 내부 형상 변화 −0.1462가 기여했습니다. 세 QR 모두 같은 방향으로 기여해 특정 QR 하나의 실패로 설명되지 않습니다.</p>
<p>이 사진의 경계 측정 RMSE p95는 전체 60장 중 낮은 쪽 8.3백분위입니다. RMSE·실제 픽셀 선명도만으로 10장 실패를 구분할 근거는 없었습니다. 경계 맞춤 sigma 중앙값은 전 사진에서 하한 0.35에 거의 고정되어, 흐림 비교 지표로 해석하지 않았습니다.</p>
<img src="diagnostics/ten_error_decomposition.png" alt="실패10장 조건평균편향과 개별반복편차"><p class="small">조건평균 분해는 사후 설명용입니다. 추론에서 시험 사진의 명령이나 다른 반복 사진을 제공하지 않았습니다. 이동 방향이 비슷하다는 것은 이 모델의 민감도 근거이며, 단일 사진으로 분리가 원천적으로 불가능하다는 증명은 아닙니다.</p></section>
{''.join(cards)}
<section id="change"><h2>이번에 추가한 개선과 채택 여부</h2><ol><li><b>조건별 균형 가중:</b> 사진 수가 많은 기존 단일 축 자료가 잡음 공분산을 지배하지 않도록, 날짜별·명령 조건별 가중치를 먼저 맞췄습니다.</li><li><b>큰 잔차의 영향 완화:</b> 학습 자료 안에서 Huber 반복 가중을 적용했습니다. 학습 사진과 630점을 제거하지 않고 양수 가중치를 유지했습니다.</li><li><b>조건 단위 가중 앙상블:</b> 학습 조건마다 양수 랜덤 가중치를 주어 12개 모델을 학습하고 평균했습니다. 특정 조건 하나에 민감한지를 줄이려는 후보입니다.</li><li><b>주기적인 서브픽셀 오차 가설:</b> 학습 조건 내부 반복만으로 픽셀 위치의 주기적 형상 성분을 추정했습니다. 원본 측정 좌표는 보존하고 모델 특징에서만 보정했습니다. <a href="https://opg.optica.org/oe/abstract.cfm?uri=oe-23-15-19242">서브픽셀 주기 편향 연구</a>를 참고한 가설 실험이며, 현재 영상의 원인이 pixel-locking이라는 증거는 확보하지 못했습니다.</li></ol>
<p>추가 탐색의 각 외부 fold는 학습 자료의 내부 평가로 후보를 골랐습니다. 전체 학습 자료로 고른 최종 저장 후보는 <b>조건별 균형 가중 + shrink=0.01</b>이며 Huber·앙상블은 꺼져 있습니다. 따라서 모든 추가 요소가 항상 개선했다고 설명하지 않습니다. 주기성 보정도 전체 학습의 최종 선택에서는 꺼졌습니다.</p>
<p>실험용 새 추론 코드 <a href="predict.py">predict.py</a>는 이미지 바이트와 고정 센터로 630점을 재측정해 Y/T 및 반대 보정값을 출력합니다. 실제 두 이미지의 재측정 결과가 저장 특징 추론과 일치함을 확인했습니다. 로봇 동작이나 기존 웹 알고리즘 변경은 하지 않았습니다.</p>
<p class="small">재현 파일: <a href="protocol.json">주 비교 프로토콜</a> · <a href="robust/exploratory_nested_protocol.json">추가 탐색 프로토콜</a> · <a href="robust/exploratory_nested_verification.json">저장 예측 검증</a> · <a href="robust/phase_review.json">주기성 가설 독립 코드 검토</a> · <a href="smoke_test.json">이미지 실행 일관성 검사</a></p></section>
<section id="future"><h2>독립 평가를 위한 준비와 현재 한계</h2><p>프로젝트의 원시·변환 사진 711개를 파일 SHA와 디코딩 픽셀 SHA로 조사했습니다. 기존 개발 사진 341개, 복사·변환 368개, 합성 테스트용 2개이며 아직 쓰지 않은 실제 촬영 자료는 확인되지 않았습니다. 사용자도 추가 데이터가 없다고 확인했습니다.</p>
<p>새 사진이 생기면 모델·측정 코드·환경·평가 기준을 고정하고, 정답을 읽기 전에 익명 사진 ID로 예측을 저장한 뒤 별도 정답 파일로 한 번 채점하도록 <a href="audit/future_validator.py">평가 실행기</a>를 만들었습니다. 기존 파일·동일 픽셀 복사·배치 내 중복을 거부하고, 추적 실패와 보류도 전체 분모에 남깁니다. 임의의 새 촬영 성적을 만들어 넣지 않았습니다.</p>
<div class="notice"><b>결론: 새 사진에서도 두 축 모두 0.09 이내라는 목표는 아직 검증·달성하지 못했습니다.</b> 현재 개발 평가에서도 7장이 실패합니다. 실패한 사진을 제외하거나 정답으로 출력을 보정하면 목표를 충족한 것으로 계산하지 않습니다.</div>
<p>기존 실험의 코드·모델·예측·보고서 <b>{len(before)}개 파일 SHA가 모두 보존</b>됐습니다. 규칙은 <a href="../../ALGORITHM_RULES.md">ALGORITHM_RULES.md</a> 한 곳에서 관리합니다.</p>
<p class="small">다운로드: <a href="all60_before_after.csv">전체60장 전후 비교</a> · <a href="ten_before_after.csv">원래 실패10장 전후</a> · <a href="remaining_failures.csv">개선 후 실패7장</a> · <a href="diagnostics/per_image_analysis.json">10장 상세 근거</a> · <a href="result.json">결과 요약</a> · <a href="preservation_after.json">기존 결과 보존 확인</a></p></section>
</main></body></html>'''
    (H/'report.html').write_text(doc,encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
