from pathlib import Path
import html,json,hashlib,ast
import pandas as pd,numpy as np
H=Path(__file__).resolve().parent;OLD=H.parent/'joint_yt_visual_feedback_20261023'
N={'qr':'QR만','upper':'QR + 위쪽 선','lower':'QR + 아래쪽 선','both':'QR + 위·아래 선'}

def table(df):return '<div class="scroll">'+df.to_html(index=False,border=0,float_format=lambda v:f'{v:.4f}')+'</div>'
def link(path,label):return f'<a href="{html.escape(path)}">{html.escape(label)}</a>'

def main():
    g=pd.read_csv(H/'guarded_summary.csv');p=pd.read_csv(H/'simulation_summary.csv');old=pd.read_csv(OLD/'control/simulation_summary.csv');r=pd.read_csv(H/'recorded_summary.csv');rg=pd.read_csv(H/'recorded_guarded_summary.csv');m=pd.read_csv(OLD/'manifest.csv')
    scenario='offset_gain_crossaxis_noise';rows=[];counts=[]
    for c,label in N.items():
        v=g[(g.condition==c)&(g.scenario==scenario)&(g.policy=='guarded_adaptive2')&(g.dataset=='all')].iloc[0]
        mixed=g[(g.condition==c)&(g.scenario==scenario)&(g.policy=='guarded_adaptive2')&(g.dataset=='20261020')].iloc[0]
        rows.append({'観測':label,'Y平均':v.Y_MAE,'Y最大':v.Y_max,'T平均':v.T_MAE,'T最大':v.T_max,'終了時通過':f'{int(v.passed)}/341','平均ターン':v.mean_turns,'画像停止':int(v.visual_stops),'誤停止':int(v.false_visual_stops),'20261020通過':f'{int(mixed.passed)}/60'})
        def pick(df,policy):return int(df[(df.condition==c)&(df.scenario==scenario)&(df.policy==policy)&(df.dataset=='all')].iloc[0].passed)
        counts.append({'관측':label,'기존 사전 반응 고정':pick(old,'fixed_jacobian'),'이동만으로 새로 식별·2회':pick(p,'motion_probe2'),'사전 반응+작은 탐색3':pick(g,'guarded_small3'),'사전 반응+탐색량 조절2':pick(g,'guarded_adaptive2'),'사전 반응+탐색량 조절3':pick(g,'guarded_adaptive3')})
    # Use Korean presentation labels; CSV source fields remain explicit.
    cols={'観測':'관측','Y平均':'Y 평균','Y最大':'Y 최대','T平均':'T 평균','T最大':'T 최대','終了時通過':'종료 시 통과','平均ターン':'평균 턴','画像停止':'영상 정지','誤停止':'잘못된 정지','20261020通過':'20261020 통과'}
    result=pd.DataFrame(rows).rename(columns=cols);result.to_csv(H/'comparison.csv',index=False,encoding='utf-8-sig')
    both=g[(g.condition=='both')&(g.scenario==scenario)&(g.policy=='guarded_adaptive2')&(g.dataset=='all')].iloc[0]
    base=old[(old.condition=='both')&(old.scenario==scenario)&(old.policy=='fixed_jacobian')&(old.dataset=='all')].iloc[0]
    real=[]
    for c,label in N.items():
        for frames in [3,4]:
            a=r[(r.condition==c)&(r.frames==frames)].iloc[0];b=rg[(rg.condition==c)&(rg.frames==frames)].iloc[0]
            real.append({'관측':label,'사진 수':frames,'이동만 식별 통과':f'{int(a.passed)}/{int(a.n)}','이동만 식별 보류':int(a.n-a.estimated),'사전 반응+이동 통과':f'{int(b.passed)}/{int(b.n)}','사전+이동 Y 평균':b.Y_MAE,'사전+이동 Y 최대':b.Y_max,'사전+이동 T 평균':b.T_MAE,'사전+이동 T 최대':b.T_max})
    inside=np.all(abs(m[['Y','T']].to_numpy())<=.09+1e-12,axis=1)
    cases=pd.read_csv(H/'guarded_cases.csv');z=cases[(cases.condition=='both')&(cases.scenario==scenario)&(cases.policy=='guarded_adaptive2')];outside=z[~inside[z['index'].to_numpy()]]
    endpoint=dict(initial_already_within=int(inside.sum()),initial_outside=len(outside),outside_to_final_within=int(outside.passed.sum()))
    (H/'endpoint_context.json').write_text(json.dumps(endpoint,indent=2))
    scene=[]
    labels={'ideal_zero_offset':'원점 일치·무잡음','repeat_noise_zero_offset':'원점 일치·반복 잡음','offset_gain_crossaxis_noise':'원점 차이·축 이득/간섭·잡음','offset_gain_crossaxis_drift_backlash_noise':'추가 드리프트·백래시'}
    for name,label in labels.items():
        v=g[(g.condition=='both')&(g.scenario==name)&(g.policy=='guarded_adaptive2')&(g.dataset=='all')].iloc[0]
        scene.append({'환경 가정':label,'Y 평균':v.Y_MAE,'Y 최대':v.Y_max,'T 평균':v.T_MAE,'T 최대':v.T_max,'종료 시 통과':f'{int(v.passed)}/341','평균 턴':v.mean_turns,'영상 정지':int(v.visual_stops),'잘못된 정지':int(v.false_visual_stops)})
    page=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>이동 전후 분석 전용 로직 · 2026.10.26</title>
<style>body{{font:16px/1.75 'Malgun Gothic',sans-serif;color:#193544;background:#f2f6f7;margin:0}}main{{max-width:1150px;margin:auto;padding:35px 24px}}h1{{font-size:34px;line-height:1.4}}h2{{margin-top:40px}}.box{{padding:20px;background:#e1f0eb;border-left:5px solid #118676;margin:22px 0}}.warn{{background:#fff0dd;border-color:#bf7023}}table{{border-collapse:collapse;background:white;width:100%;font-size:14px}}th,td{{padding:9px;text-align:right;border-bottom:1px solid #dce5e9}}th:first-child,td:first-child{{text-align:left}}.scroll{{overflow:auto}}.flow,pre{{background:#173847;color:white;padding:22px;border-radius:7px;overflow:auto}}a{{color:#06718a}}small{{color:#526b78}}details{{background:white;margin:18px 0;padding:16px}}code{{background:#e0e9ed;padding:2px 4px}}</style><main>
<small>2026.10.26 · QR / 위쪽 / 아래쪽 / 양쪽 · 기존 결과 보존</small><h1>실행한 이동과 전후 영상으로 반응을 구하는<br>별도 피드백 로직을 구현했습니다.</h1>
<div class="box">한 장 추정 모델을 반복 호출하는 대신, 실행한 상대 Y·T 명령과 영상 변화 이력을 저장합니다. 탐색 → 두 축 반응 식별 → 보정 제안 → 실행 결과 재관측 → 반응 갱신을 최대20턴으로 연결했습니다.</div>
<div class="box warn"><b>코드 구현·오프라인 비교까지 완료했습니다. 실물 로봇은 구동하지 않았습니다.</b> 현재 자료에는 실제 연속 이동 촬영이 없습니다. 아래는 기존 정지 사진 조합과 가정 시뮬레이션이며, 새 사진·실제20턴 복귀 성능으로 해석할 수 없습니다. 전수 두 축0.09 목표도 달성하지 못했습니다.</div>
<h2>1. 이동 전용 처리 흐름</h2><div class="flow">기준 센터 특징 s*와 첫 관측 s₀<br>→ Y 탐색 제안 → 실행한 상대 명령 u₁ + 새 영상 s₁<br>→ T 탐색 제안 → 실행한 상대 명령 u₂ + 새 영상 s₂<br>→ 선택한3회 탐색: 반대 이동으로 재관측<br>→ 이력에서 Jᵧ·Jₜ 추정 → 센터 영상 오차를 줄일 ΔY·ΔT 계산<br>→ 이동량 제한·제안 → 실행값과 새 영상 → 반응 갱신·재계산</div>
<p>행동을 제안한 뒤 확인된 실행 명령을 받기 전에는 다음 턴을 진행하지 않습니다. 제안과 다르게 일부만 실행했더라도 <b>확인된 실행값</b>을 사용합니다. 이는 명령값이며 실제 기계 변위를 별도 측정했다는 뜻은 아닙니다. 첫 관측0턴, 탐색·보정·확인 촬영은 모두20턴 안에 포함합니다.</p>
<p>현재 자세 정답·파일명·시행 번호는 제어기에 전달하지 않습니다. 중심 기준은 원래 고정 센터입니다. 이력 회귀의 절편은 영상 변화의 기울기를 구하기 위한 값이며 센터 기준을 현재 사진으로 바꾸지 않습니다.</p>
<h2>2. 두 모드를 구분했습니다</h2><p><b>이동만으로 식별:</b> 학습된 J 없이 시작합니다. 누적 실행 명령과 영상 특징에 절편을 포함한 최소제곱을 적용해 두 반응을 구합니다. 입력이 두 축을 충분히 움직이지 않았거나 영상의 구분 신호가 약하면 <code>insufficient_separation</code>으로 보정 제안을 보류합니다.</p>
<p><b>사전 반응+이동 갱신:</b> 학습fold에서만 구한 J를 초기값으로 사용합니다. 영상 변화가 충분하고 사전 반응과의 차이가 과도하지 않을 때만 규제한 추정으로 갱신합니다. 갱신이 거부되면 기존 반응을 유지합니다. 이 모드를 세션 영상만으로 식별한 결과라고 부르지 않습니다. 두 모드 모두 특징 투영과 잡음 크기는 기존 학습 자료에서 구합니다.</p>
<p>보완안은 학습 자료의 약한 축 반응과 잡음으로 탐색량을 계산하고 지정 상한 안에서만 늘립니다. 실험의 최소값Y0.05/T0.01과 상한각0.25는 <b>시뮬레이션 설정</b>입니다. 실제 허용 이동량·T 각도 단위는 확인되지 않았으며 이미지 CLI에서는 탐색량과 상한을 직접 입력해야 합니다. 잡음 판정 지표는 보조 진단이며 자세 오차의 통계적 보증이 아닙니다.</p>
<h2>3. 기존 사진으로 이동 전후 반응을 확인한 결과</h2>{table(pd.DataFrame(real))}
<p>혼합48장의36개 경로 설정에서 반복 사진 조합을 모두 비교했습니다. 각 경로의4조건·반복12장을 학습에서 제외했습니다.3장은972조합,4장은2916조합이며 서로 의존적입니다. 이는 실제 연속 촬영이나 제어기가 스스로 선택한 탐색 경로가 아닙니다. 세 번째 기록 이동은 Y만 반대로 움직이는 격자 경로이고, 제어기의 명령합0 탐색과도 구분합니다.</p>
<p>표는 관측된 마지막 사진의 현재 자세를 ‘제안한 전체 보정량의 반대값’으로 추정해 평가했습니다. 보정 후 실제 도착한 사진은 없습니다. 보류도 전체 통과율의 분모에 포함하며, 이동만 식별 모드의 오차는 추정이 나온 조합에서만 계산되므로 {link('recorded_summary.csv','추정 수·조건부 평균/최대오차')}와 함께 봐야 합니다. 정지 사진의 여러 장 역산 성적을 이동 제어 성공률로 다시 사용하지 않았습니다.</p>
<h2>4. 최대20턴 제어: 같은 가정 환경에서 비교</h2><p>아래는 원점 차이·축 이득/간섭·반복 잡음을 가정한 결과입니다. 이전 제어기와 같은16개 분할 교정/환경 수식,341개 시작 상태,잡음 난수를 사용했습니다. 환경의 정답은 평가기에만 있습니다. 초기 관측부터 수식이 만든 특징이므로 실제 사진 재생이 아닙니다.</p>{table(result)}
<p><b>양쪽 선의 보완 탐색2회는 기존 사전 반응 고정 {int(base.passed)}/341 → {int(both.passed)}/341로 늘었지만, Y 최대오차는 {base.Y_max:.4f} → {both.Y_max:.4f}로 커졌습니다.</b> 통과 수 개선만으로 채택하거나0.09를 보장할 수 없습니다. 평균20턴·영상 정지0건이라 ‘센터임을 확인하고 빨리 정지했다’는 결과도 아닙니다.</p>
<details><summary>정책별 종료 시 통과 수 비교 · 각341개</summary>{table(pd.DataFrame(counts))}<p>탐색량과 추정 규제가 함께 바뀐 보완안입니다. 갱신 알고리즘 하나만의 효과로 해석하지 않습니다. 초기 이동만 식별 결과를 본 뒤 보완안을 추가했으며 독립 시험으로 부르지 않습니다.</p></details>
<p>341개 중 처음부터 두 축0.09 안에 있던 상태는 {endpoint['initial_already_within']}개입니다. 보완 탐색2회·양쪽 선에서 초기 바깥 {endpoint['initial_outside']}개 중 종료 시 안으로 들어온 것은 {endpoint['outside_to_final_within']}개입니다. 시작부터 안에 있는 경우도 최종 위치로 다시 평가했습니다.</p>
<details><summary>양쪽 선·보완 탐색2회의 환경 가정별 결과</summary>{table(pd.DataFrame(scene))}</details>
<h2>5. 무엇이 아직 어려운가</h2><p>작은 이동만으로 약한 축 반응을 새로 추정하면 잡음이 반응 방향처럼 보일 수 있습니다. 따라서 영상 차이가 생겼다는 사실만으로 Y/T를 구분했다고 판정하지 않습니다. 긴 가로선의 높이·기울기 정보도 사용하되 볼트·너트·손잡이·일반 배경은 제외했습니다.</p>
<p>또한 국소 반응 J는 센터까지의 전체 변화가 선형이라는 보장이 아닙니다. 원점 차이를 b라 하면 s=b+Jq에서 탐색 변화는 Δs=Ju입니다. 움직임만으로 J를 알아도 b와 현재 자세 q를 따로 확정할 수는 없습니다. 실제 고정 센터와 물리 원점의 관계, 국소 모델의 유효 범위, 반복 잡음을 검증해야 합니다.</p>
<h2>6. 실행 인터페이스와 기록</h2><p>{link('session.py','이미지 피드백 CLI')}의 <code>start</code>에 첫 사진·관측 조건·탐색량·이동 상한을 넣습니다. 이후 <code>observe</code>에 새 사진과 실제 실행한 <code>--dy / --dt</code>를 전달하면 다음 이동 제안을 반환합니다. <code>inspect</code>에서 턴별 제안·실행·관측 변화·예측 변화·반응 갱신/거부 이유를 확인할 수 있습니다. 하드웨어 명령을 전송하는 경로는 없습니다.</p>
<pre>python session.py start STATE.joblib FIRST.bmp --condition both \\\n+  --probe-y Y_PROBE --probe-t T_PROBE \\\n+  --max-step-y Y_CAP --max-step-t T_CAP \\\n+  --mode prior-guarded --adaptive-probes --probes 2

python session.py observe STATE.joblib NEXT.bmp --dy EXECUTED_Y --dt EXECUTED_T
python session.py inspect STATE.joblib</pre>
<p>QR만은<code>qr</code>,위쪽은<code>upper</code>,아래쪽은<code>lower</code>,양쪽은<code>both</code>입니다. 학습된 반응 없이 시작하려면<code>--mode online-only</code>를 사용하고<code>--adaptive-probes</code>는 빼야 합니다. 숫자 자리는 장치에서 확인한 명령 단위와 한계로 채웁니다. 유효 영상 측정이 실패하면 실행 턴을 기록하고 종료합니다.</p>
<p>{link('motion.py','이동 분석·제어 로직')} · {link('software_checks.json','알려진 선형 응답·거부·20턴 소프트웨어 검사')} · {link('cli_verification.json','실제 이미지 입력 CLI 검사')} · {link('verification.json','전수 결과·이력 재계산 및 재현')} · {link('protocol.json','처음 이동만 식별 실험 조건')} · {link('guarded_protocol.json','사후 보완 실험 조건')} · {link('simulation_summary.csv','이동만 식별 시뮬레이션 전체')} · {link('guarded_summary.csv','보완 시뮬레이션 전체')} · {link('../joint_yt_visual_feedback_20261023/report.html','이전4관측 비교')} · {link('../joint_yt_visual_feedback_20261023/research/sources.json','기존 조사한 원문7개와 적용 범위')} · {link('../../ALGORITHM_RULES.md','공통 규칙')}</p>
<p><small>모든341장은 이미 사용한 개발 자료입니다. 기존 수치 결과·모델·코드는 보존했습니다. 과거 동결 목록과 다른 감사 문서1개는10월23일 수정된 문구이며 이번 실행 전부터 있던 차이임을 검증 파일에 공개했습니다. 현재 실험의 소프트웨어 검사는 실제0.09 복귀 성공의 증거가 아닙니다.</small></p></main></html>'''
    (H/'report.html').write_text(page,encoding='utf-8')
    print(result.to_string(index=False))
if __name__=='__main__':main()
