from pathlib import Path
import sys,json,html,hashlib
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import cv2
H=Path(__file__).resolve().parent;ROOT=H.parents[1]
sys.path.insert(0,str(H.parent/'joint_yt_generalization_20261022'))
from data import load
from metrics import metric

def table(df):return df.to_html(index=False,border=0,classes='data',escape=True,float_format=lambda x:f'{x:.4f}')
def a(path,text):return f'<a href="{html.escape(path)}">{html.escape(text)}</a>'

def main():
    m,_=load();research=json.loads((H/'research/research.json').read_text(encoding='utf-8'));par=pd.read_csv(H/'parallax_predictions.csv');oof=par[(par.method=='qr_external')&(par.protocol=='development_nested_oof')].sort_values('index');summary=[]
    def add(label,df,method=None,protocol=None):
        if method is not None:df=df[df.method==method]
        if protocol is not None:df=df[df.protocol==protocol]
        n=df[df.dataset.astype(str)=='20261020'].iloc[0];old=df[df.dataset.astype(str)=='20261014'].iloc[0]
        summary.append({'방법':label,'Y 평균':n.Y_MAE,'Y 최대':n.Y_max,'T 평균':n.T_MAE,'T 최대':n.T_max,'20261020 동시 통과':f'{int(n.passed)}/60','20261014 동시 통과':f'{int(old.passed)}/281'})
    add('직전 QR 630점 개선 후보',pd.read_csv(H.parent/'joint_yt_erroraudit_20261022/robust/exploratory_nested_summary.csv'),protocol='exploratory_nested16_oof')
    add('새 QR별 직접 정합',pd.read_csv(H/'direct_summary.csv'),protocol='development_nested_oof')
    add('새 QR 공동 정합',pd.read_csv(H/'shared_summary.csv'),protocol='development_nested_oof')
    gen=pd.read_csv(H/'generative/summary_all.csv');add('기하 생성모델 — 사후 최선 보조 후보',gen[(gen.experiment=='ecc5_new')&(gen.candidate==3)])
    pixels=pd.read_csv(H/'pixel_summary.csv');add('원픽셀 PLS',pixels,'pixel_pls','development_group_oof');add('원픽셀 CNN',pixels,'pixel_cnn','development_group_oof')
    ps=pd.read_csv(H/'parallax_summary.csv');add('외부 볼트·브래킷 5곳만',ps,'external','development_nested_oof');add('QR + 외부 볼트·브래킷 5곳',ps,'qr_external','development_nested_oof')
    comp=pd.DataFrame(summary);comp.to_csv(H/'comparison.csv',index=False,encoding='utf-8-sig')
    date=pd.read_csv(H/'date_stress_summary.csv');date=date[date.method=='qr_external'].rename(columns={'train_date':'학습 날짜','test_date':'평가 날짜','Y_MAE':'Y 평균','Y_max':'Y 최대','T_MAE':'T 평균','T_max':'T 최대','passed':'통과','n':'전체'})
    by=[]
    for label,sub in [('전체',oof),('20261014: 단일축280+center1',oof[oof.dataset==20261014]),('20261020: 혼합57+center3',oof[oof.dataset==20261020])]:
        r=metric(sub[['Y','T']],sub[['predY','predT']]);by.append({'범위':label,'Y 평균':r['Y_MAE'],'Y 최대':r['Y_max'],'T 평균':r['T_MAE'],'T 최대':r['T_max'],'동시 통과':f"{r['passed']}/{r['n']}",'측정 실패':r['failures']})
    oldfail=json.loads((H.parent/'joint_yt_erroraudit_20261022/diagnostics/per_image_analysis.json').read_text(encoding='utf-8'));fail=[]
    for r in oldfail:
        z=oof[oof['index']==r['index']].iloc[0];fail.append({'사진':z.file,'과거 Y 오차':r['evidence']['eY'],'새 Y 오차':z.eY,'새 T 오차':z.eT,'새 Y 예측':z.predY,'새 T 예측':z.predT})
    fail=pd.DataFrame(fail);fail.to_csv(H/'prior10_comparison.csv',index=False,encoding='utf-8-sig')
    # Scientific comparison figures: errors, not classification accuracy.
    plt.rcParams['font.family']='Malgun Gothic';plt.rcParams['axes.unicode_minus']=False
    fig,ax=plt.subplots(1,2,figsize=(13,4.7),layout='constrained')
    short=['직전 QR','QR 직접','QR 공동','기하 생성','픽셀 PLS','픽셀 CNN','외부만','QR+외부']
    colors=['#9ba6b2']*6+['#17a9a5','#056b69']
    for j,(key,title) in enumerate([('Y 최대','20261020 · Y 최대 절대오차'),('T 최대','20261020 · T 최대 절대오차')]):
        ax[j].barh(short,comp[key],color=colors);ax[j].axvline(.09,color='#e05539',ls='--',label='허용오차 0.09');ax[j].invert_yaxis();ax[j].set_title(title);ax[j].grid(axis='x',alpha=.15);ax[j].legend();ax[j].set_xlabel('명령값 단위 (T 물리 단위 미확인)')
    fig.savefig(H/'comparison.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for j,axis in enumerate(['Y','T']):
        for ds,c in [(20261014,'#6696bd'),(20261020,'#d87727')]:
            z=oof[oof.dataset==ds];ax[j].scatter(z[axis],z['pred'+axis],s=15,c=c,label=str(ds),alpha=.75)
        lo=min(oof[axis].min(),oof['pred'+axis].min());hi=max(oof[axis].max(),oof['pred'+axis].max());ax[j].plot([lo,hi],[lo,hi],c='#303e47');ax[j].set_xlabel(f'{axis} 명령 정답');ax[j].set_ylabel(f'{axis} OOF 예측');ax[j].set_title('QR+외부 / 전체341 개발 평가');ax[j].legend()
    fig.savefig(H/'predictions.png',dpi=150);plt.close(fig)
    raw=(ROOT/'hw-analysis/data/hw/bmp/20261014/center.bmp').read_bytes();im=cv2.imdecode(np.frombuffer(raw,np.uint8),0);pr=json.loads((H/'parallax/protocol.json').read_text(encoding='utf-8'))
    fig,ax=plt.subplots(figsize=(12.8,7.2),dpi=100);ax.imshow(im,cmap='gray',vmin=0,vmax=255)
    for k,r in enumerate(pr['rois']):
        x0,y0,x1,y1=r['box'];ax.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,fill=False,edgecolor='#00ddbe',lw=2));ax.text(x0,y0-20,f'E{k+1}',color='#00ddbe',weight='bold',fontsize=13)
    for k,(x0,y0,x1,y1) in enumerate([(1552,969,1736,1167),(1809,971,1993,1169),(2067,973,2252,1171)]):
        ax.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,fill=False,edgecolor='#ffd35b',lw=2));ax.text(x0,y0-20,f'QR{k+1}',color='#ffd35b',fontsize=11)
    ax.axis('off');fig.subplots_adjust(left=0,right=1,bottom=0,top=1);fig.savefig(H/'regions_overview.png',dpi=100);plt.close(fig)
    sources=''.join(f'<li><a href="{html.escape(s["url"])}">{html.escape(s["title"])}</a><p>{html.escape(s["supports"])}</p></li>' for s in research['sources'])
    before=json.loads((H/'preservation_before.json').read_text());changed=[];missing=[]
    for rel,sha in before['files'].items():
        p=H.parent/rel
        if not p.exists():missing.append(rel)
        elif hashlib.sha256(p.read_bytes()).hexdigest()!=sha:changed.append(rel)
    preserve=dict(checked=len(before['files']),changed=changed,missing=missing,ALGORITHM_RULES='new experiment permission recorded intentionally; not part of preservation snapshot');(H/'preservation_after.json').write_text(json.dumps(preserve,indent=2))
    assert not changed and not missing,preserve
    text=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Y/T 원점 재설계 — 관측 영역을 바꾼 결과</title>
<style>body{{margin:0;background:#f3f6f8;color:#192c3b;font:16px/1.7 'Malgun Gothic',sans-serif}}main{{max-width:1160px;margin:auto;padding:44px 28px}}h1{{font-size:38px;line-height:1.3;letter-spacing:-1px}}h2{{margin-top:54px;border-top:1px solid #cad7df;padding-top:24px}}h3{{margin-top:28px}}a{{color:#076d8e}}.tag{{color:#087e79;font-weight:700}}.lead{{font-size:20px;max-width:1000px}}.box{{padding:20px 24px;margin:22px 0;border-radius:10px;background:#e1f3ef;border-left:5px solid #137d70}}.warn{{background:#fff0de;border-color:#cd7735}}.cards{{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}}.card{{padding:22px;background:white;border-radius:12px;box-shadow:0 3px 14px #263d4b0a}}.big{{font-size:34px;font-weight:800}}small,.muted{{color:#557080}}table{{border-collapse:collapse;width:100%;font-size:14px;background:white}}th,td{{padding:11px 12px;border-bottom:1px solid #e0e8ed;text-align:right}}th:first-child,td:first-child{{text-align:left}}th{{background:#e6eef2}}.scroll{{overflow:auto}}img{{max-width:100%;height:auto;background:white}}.pipeline{{padding:24px;background:#192f3c;color:#e8f7f7;border-radius:12px;font-size:19px;line-height:2}}details{{background:white;padding:16px;margin:16px 0;border-radius:10px}}summary{{cursor:pointer;font-weight:700}}li{{margin:12px 0}}li p{{margin:0;font-size:14px;color:#526775}}code{{background:#e4edf1;padding:2px 5px}}footer{{margin-top:50px;color:#58707f}}@media(max-width:700px){{.cards{{grid-template-columns:1fr}}main{{padding:22px 14px}}h1{{font-size:29px}}}}</style>
<main><div class="tag">2026.10.22 · 원점 재설계 · 별도 실험</div><h1>관측 영역을 바꿔<br>Y와 T의 구분 정보를 늘렸습니다.</h1>
<p class="lead">고정 center와 현재 사진에서 <b>QR 3개 + 외부 볼트·브래킷 5곳</b>의 움직임을 직접 측정했습니다. 기존 630점 알고리즘과 결과를 보존하고, 측정부터 새로 만든 모델을 비교했습니다.</p>
<div class="cards"><div class="card">전체 개발 OOF<div class="big">341 / 341</div><small>Y와 T 모두 절대오차≤0.09</small></div><div class="card">전체 Y 최대오차<div class="big">0.03307</div><small>평균 0.00883</small></div><div class="card">전체 T 최대오차<div class="big">0.00688</div><small>평균 0.00151 · T 명령값 단위</small></div></div>
<div class="box warn"><b>현재 자료의 개발 목표는 달성. 새 촬영 성능 보장은 아직 아닙니다.</b><br>341장 모두 이전 개발에서 이미 본 자료입니다. 사진별 예측을 만들 때 해당 시험 fold를 학습에서 제외했지만, 방법 설계까지 독립이었던 새 시험은 아닙니다. 추가 실제 촬영 자료는 없으며 로봇 복귀도 실행하지 않았습니다.</div>
<h2>1. 무엇을 바꿨나</h2><p>QR 내부에서는 Y와 T가 거의 같은 방향으로 크게 움직입니다. 두 축을 나누려면 아주 작은 형태 차이에 의존해야 했습니다. 이번에는 영상에서 서로 떨어진 구조물의 움직임 차이를 함께 사용했습니다. 깊이가 달라질 때 평행이동과 회전의 영상 반응이 달라지는 점은 <a href="https://web.mit.edu/amcp/OldFiles/drg/Chaumette_Part_I.pdf">Chaumette·Hutchinson의 영상 제어식</a>을 참고했습니다. 실제 각 부품의 깊이와 로봇 T의 회전축을 복원한 것은 아닙니다.</p>
<div class="pipeline">고정 center + 현재 이미지 바이트<br>↓ QR 3개와 외부 고정 영역 5곳의 원픽셀 정합<br>↓ 각 영역 affine / projective 변환 → 고정 격자 변위<br>↓ 학습 자료에서 교정한 두 축 영상 반응과 잡음 공분산으로 역산<br>↓ 연속 Y·T 출력</div>
<p>최종 실험에서는 외부 affine 정합 + 선형 순방향 교정/역산이 모든 외부 fold와 전체 학습의 내부 검증에서 선택됐습니다. 교정 특징에는 변위 자체와 변위×평균 가로이동 항을 사용하며, 그 특징 공간에서 명령에 대한 선형 순방향 모델을 적합합니다. 현재 파일명·정답·촬영 날짜·반복 번호는 추론 입력이 아닙니다. 출력값 반올림, 정답 목록 조회, 한 축 강제 0 처리를 하지 않습니다.</p>
<img src="regions_overview.png" alt="원본 center의 QR와 외부 관측 영역"><p class="muted">원본3840×2160을 1280×720으로 축소한 개요(1/3배; 화면 크기에 따라 추가 축소). 노랑: QR, 초록: E1·E2 벽 볼트, E3·E4 위쪽 브래킷 볼트, E5 아래 손잡이 고정부. {a('parallax/selected_center_regions_1x.png','외부 영역의 1배 crop')} · {a('parallax/protocol.json','고정 영역·실패 기준')}</p>
<p>외부 5곳은 center의 텍스처만 보고 먼저 고정했습니다. 제안된 빈 배경 ROI 하나는 실제 손잡이 고정부로 옮긴 뒤 프로토콜을 저장했고, 341장 성적을 보고 영역을 바꾸지 않았습니다. 각 영역의 9개 격자는 영상 변환을 읽는 가상 투영점이며, 물리 코너 9개를 독립 검출했다는 뜻이 아닙니다. 외부 부품이 따로 움직이거나 가려지는 환경은 별도 검증이 필요합니다.</p>
<h2>2. 새 측정·추론 계통을 실제로 비교했습니다</h2><p>18개 논문·공식 자료에서 7개 접근을 검토했습니다. 원영상 정합, 기하 생성, 원픽셀 학습, 외부 구조 관측을 구현했고 카메라 자기보정도 수치적으로 시도했습니다. 아래는 <b>20261020의 60장</b>에 대한 개발 OOF 오차입니다. 평균·최대오차를 먼저, 두 축 허용오차 동시 통과 수를 뒤에 표시했습니다.</p><div class="scroll">{table(comp)}</div>
<p class="muted">기하 생성 행은 보조 조합 20개 중 결과를 본 뒤 고른 최선으로, nested 선택 성적이 아닙니다. 나머지 선택 방법은 학습 내부 검증을 사용했습니다(CNN은 고정350epoch·seed·구조). QR 계열과 PLS의 내부 선택은 새 날짜 실패 수를 우선했고, 외부 관측 실험은 두 날짜의 실패율에 동일 비중을 두었습니다. 외부 fold와 사진은 같습니다. 모든 성적은 이미 노출된 개발 자료의 비교입니다.</p>
<img src="comparison.png" alt="각 방법의 Y T 최대오차와 0.09 기준">
<details><summary>각 접근에서 추가한 것과 결과 해석</summary><ul><li><b>직접 영상 정합:</b> 630개 경계 좌표의 재사용을 벗어나 QR 내부 픽셀에서 affine/H를 새로 측정. QR별4개 관측×3개 교정기, 공동4개×3개 교정기를 비교했습니다. 공동 제약은 형태 잡음을 줄였지만 Y/T 성적은 더 나빴습니다.</li><li><b>기하 생성:</b> Y,T에서 예상 homography를 만들고 가장 맞는 두 값을 역산. rational, Y×T, exp(YA)exp(TB), 역순 합성의 4구조를 시험했습니다. optimizer 실패0이어도 명령 오차는 컸습니다. {a('generative/summary_all.csv','전체20조합 성적')}</li><li><b>원픽셀 학습:</b> PLS4/12/32차원과 공유 CNN42,860파라미터를 구현했습니다. 큰 이동은 별도 좌표 입력으로 남겼습니다. PLS 전수 재대입은341/341이지만 OOF 혼합 날짜는36/60이므로 암기 성적을 일반화로 쓰지 않았습니다.</li><li><b>카메라 보정·PnP:</b> 실제 QR 격자와 학습 영상으로 Zhang 선형 보정을 시도했으나, QR3개 공통 해의 행렬이4개 학습fold 모두 양정치가 아니었습니다. 개별 QR의 보정값도 불일치했습니다. 임의 K·회전단위를 넣은 PnP를 성공 모델로 발표하지 않았습니다. {a('research/calibration_diagnostic.json','수치 진단')}</li><li><b>외부 구조 관측:</b> 외부만/QR+외부 각2개 정합×3개 교정기. 관측 자체에 새 공간 반응이 추가되었습니다. 외부만으로도341/341, QR를 더하면 최대오차가 소폭 줄었습니다.</li><li><b>능동 관측:</b> 작은 실제 구동 뒤 재촬영하여 모호성을 줄이는 접근은 조사만 했습니다. 새 사진·실제 구동을 실행하지 않았습니다.</li><li><b>카메라·표적 재설계:</b> 깊이·시야 배치와 마커 설계를 조사했습니다. 새 장비나 마커를 설치한 성적은 없습니다.</li></ul></details>
<h2>3. 왜 외부 영역이 효과가 있었나</h2><p>각 외부 학습fold의 혼합 조건 평균으로 선형 반응을 적합했습니다. QR 직접 관측의 Y/T 방향 cosine은 약0.999997, 열을 정규화한 조건수는 약831이었습니다. QR+외부에서는 각각 약0.997819, 30으로 개선됐습니다. Y를0.09 바꾸고 T로 큰 공통 이동을 상쇄한 뒤 남는 좌표 차이의 RMS는 <b>0.00221px → 0.05707px</b>로 약26배 커졌습니다.</p><p>즉, 이 학습 모델에서는 두 축을 구분하는 영상 단서가 더 커졌습니다. 이것은 국소 경험 모델의 진단이며 실제 깊이 복원, 모든 자세의 식별 가능성, 새 설치의 일반화 보장은 아닙니다. {a('spatial_response_diagnostic.csv','fold별 진단 수치')}</p>
<h2>4. 전체 성적과 평가 경계</h2><div class="scroll">{table(pd.DataFrame(by))}</div><p>20261014는 단일축280장+center1, 20261020은 실제 혼합57장+center3입니다. Y/T 명령이 같은 혼합 반복3장은 항상 같은 외부 시험fold에 묶었습니다. 기존4fold와 내부 분할을 그대로 사용했고, 정규화·교정·공분산·후보 선택은 각 학습 분할 안에서 계산했습니다. 외부측정은341장×5영역 모두 성공했고 실패가 있었다면 전체 분모에 포함하는 코드입니다.</p>
<img src="predictions.png" alt="전체 OOF 명령값 대 예측값">
<div class="box warn"><b>날짜를 통째로 빼면 같은 결과가 나오지 않습니다.</b><br>20261014만 학습해20261020을 예측하는 추가 스트레스 시험에서는36/60만 통과했습니다. 날짜 변화와 처음 보는 혼합 조건이 동시에 바뀌므로 둘 중 하나만의 영향이라고 단정할 수 없습니다. 현재 최종 후보는 두 날짜의 교정 자료를 함께 사용하는 모델입니다.</div><div class="scroll">{table(date[['학습 날짜','평가 날짜','전체','Y 평균','Y 최대','T 평균','T 최대','통과']])}</div>
<p class="muted">날짜 시험도 이미 본 날짜의 사후 개발 진단입니다. 역방향은 원래20261014 center를 알려진 기준 영상으로 계속 쓰므로 완전히 보지 못한 기준/설치 이전 시험이 아닙니다. {a('date_stress_protocol.json','날짜 시험 조건')}</p>
<details><summary>과거 실패10장도 모두 개선됐는지</summary><p>과거10장뿐 아니라 전체341장을 함께 평가했습니다. 이10장은 모두 새 OOF에서 통과했습니다.</p><div class="scroll">{table(fail)}</div>{a('prior10_comparison.csv','사진별 수치 CSV')}</details>
<h2>5. 실행·검증·보존</h2><p>{preserve['checked']}개 기존 결과 파일의 SHA-256을 시작 전과 비교해 변경·누락0개를 확인했습니다. {a('../../ALGORITHM_RULES.md','알고리즘 공통 규칙')}에는 이번 별도 재설계 허용만 기록했습니다. 기존 운영 알고리즘은 자동 교체하지 않았습니다.</p><p>독립 코드 검토에서 실패 검증행의 분모 처리와 전체 실패 시 지표 계산의 잠재 결함을 고쳤습니다. 현재 모든 측정은 성공이어서 저장된 성능 수치에는 영향이 없었습니다. 실패 주입 검사와 모델·원영상 재현 기록을 함께 보존합니다.</p>
<p>현재 이미지로 별도 후보를 실행할 수 있습니다: <code>python predict.py &lt;현재사진경로&gt; --method qr_external</code>. 입력 경로는 바이트 읽기에만 사용됩니다. 모델은 Y/T를 출력하며 로봇 명령을 보내지 않습니다. 출력의 반대 부호를 실제 구동해 센터에 복귀하는지는 별도 폐루프 측정이 필요합니다.</p>
<p>{a('predict.py','이미지 추론 코드')} · {a('qr_external_model.joblib','전체 교정 모델')} · {a('parallax_predictions.csv','전수 OOF·재대입 예측')} · {a('parallax_selection.json','내부 선택·분할')} · {a('parallax_fixed_predictions.csv','모든 고정 후보 OOF')} · {a('preservation_after.json','기존 결과 보존 검사')} · {a('review.json','코드·분할 감사')} · {a('generative/raw_parallax_audit/audit.json','341장 원본 재추출 감사')} · {a('parallax/qa/api_verification.json','최악 사진 정합·추론 QA')} · {a('inference_verification.json','이미지 추론 재현')} · {a('frozen_manifest.json','후보 고정 해시')}</p>
<h2>6. 문헌과 적용 범위</h2><p>원문·저자기관·공식 구현 자료를 기준으로 조사했습니다. 논문의 전체 시스템을 그대로 재현했다고 주장하지 않으며, 각 아이디어의 적용과 제약은 {a('research/research.json','연구 기록')}에 구분했습니다.</p><details><summary>18개 출처 펼치기</summary><ol>{sources}</ol></details>
<footer>정답은 파일명에 기록된 명령값입니다. 별도 계측한 실제 자세가 아닙니다. Y는 프로젝트의 명령 단위, T의 물리 각도 단위는 미확인입니다. 현재 QR 내용·구조물·카메라 환경에서 개발했으며 QR 변경·조명·가림·다른 설치·실제 복귀 오차는 독립 검증하지 않았습니다.</footer></main></html>'''
    (H/'report.html').write_text(text,encoding='utf-8')
    print('report ready; preserved',preserve['checked'])

if __name__=='__main__':main()
