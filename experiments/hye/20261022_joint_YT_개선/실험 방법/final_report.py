"""Error-first Korean report; all exploration and limitations remain visible."""
import json,html
import numpy as np,pandas as pd
from engine import H,B
from experiment import metrics

def read(name):return json.loads((H/name).read_text())
def table(df):return df.to_html(index=False,escape=False,float_format=lambda v:f'{v:.5f}')

def main():
    summary=pd.read_csv(H/'summary.csv');summary=summary[summary.subset=='all'].set_index('method')
    fixed=read('fixedfeature_summary.json');coord=read('coordinate_summary.json');full=read('fullfield_summary.json');linear=read('linear_audit.json')
    entries=[('기존 Y-or-T (참고)',summary.loc['existing_Y_or_T'].to_dict()),('첫 동시 추정',summary.loc['joint_nested_selected'].to_dict()),('4단계 순차 탐색',summary.loc['improved_nested'].to_dict()),('원래 특징+비선형 회귀·증강',fixed['real_oof']),('630점 선형+품질 통계',full['real_oof']),('630점 좌표 전용 선형',coord['real_oof'])]
    rows=[]
    for label,s in entries:rows.append({'모델':label,'Y 평균 절대오차':s['Y_MAE'],'Y 최대 절대오차':s['Y_max'],'T 평균 절대오차':s['T_MAE'],'T 최대 절대오차':s['T_max'],'두 축 허용오차 동시 통과':f"{int(s['passed'])}/280 ({s['pass_rate']*100:.1f}%)"})
    main_table=pd.DataFrame(rows);main_table.to_csv(H/'comparison_errors_first.csv',index=False,encoding='utf-8-sig')
    stages=[]
    for label,key in [('0 공통 원점 학습','stage0_common_origin'),('A 관측 가중 집계 추가','stageA_selected'),('B 공간 특징 변경 추가','stageB_selected'),('C 회귀 변경 추가','stageC_selected'),('D 안정화 추가','improved_nested')]:
        s=summary.loc[key];stages.append({'순차 단계':label,'Y 평균':s.Y_MAE,'Y 최대':s.Y_max,'T 평균':s.T_MAE,'T 최대':s.T_max,'보조 통과 수':int(s.passed)})
    methods=[{'단계':'A 관측 집계','실제 추가 방법 수':6,'적용한 방법':'밝기 RMSE 가중, 창 안정성 가중, 대비 종합 가중, Huber, soft-L1, Cauchy','추가 위치':'같은 630점 좌표 → 공간 회귀. 모든 점 가중치≥0.001. 픽셀 경계 측정기는 유지.'}, {'단계':'B 특징 표현','실제 추가 방법 수':6,'적용한 방법':'전체 affine, 전체 cubic, QR별 affine, QR별 quadratic, QR별 4분면, homography','추가 위치':'공간 회귀 → 회귀 입력 특징. 원래 global quadratic(P12) 포함 총7개.'},{'단계':'C 회귀·역추정','실제 추가 방법 수':8,'적용한 방법':'Ridge, BayesianRidge, ARD, PCR, PLS, RBF KernelRidge, epsilon-SVR, ExtraTrees','추가 위치':'특징 → Y/T 두 연속 출력. 추가로 GLS 공분산5종과 이차·강건 손실 비교, 총28개 설정.'},{'단계':'D 추론 안정화','실제 추가 방법 수':7,'적용한 방법':'bootstrap 평균, bootstrap 중앙값, subagging, 특징 잡음 증강, convex mixup, 가산 Y+T 증강, 축 균형 재표본화','추가 위치':'선택한 C모델의 학습/앙상블 단계. 원래 none 포함 총8개.'}]
    diags=[read(f'diagnostic_fold{f}.json') for f in range(4)]
    fixedp=pd.read_csv(H/'fixedfeature_oof.csv');coordp=pd.read_csv(H/'coordinate_oof.csv')
    oldp=pd.read_csv(B/'joint_yt_20261022/predictions.csv');oldp=oldp[(oldp.method=='joint_nested_selected')&(oldp.protocol=='oof4')]
    corr=float(np.corrcoef((oldp.pred_Y-oldp.true_Y),(oldp.pred_T-oldp.true_T))[0,1])
    detail=[]
    for label,pp in [('비선형+증강',fixedp),('좌표 전용 선형',coordp)]:
        for a in ['Y','T']:
            g=pp[pp.axis==a];s=metrics(g[['pred_Y','pred_T']].to_numpy(),g[['true_Y','true_T']].to_numpy());detail.append({'모델':label,'사진':a,'Y 평균':s['Y_MAE'],'Y 최대':s['Y_max'],'T 평균':s['T_MAE'],'T 최대':s['T_max'],'비활성 축 평균 절대출력':s['inactive_MAE']})
    worst=fixedp.copy();worst['Y오차']=abs(worst.pred_Y-worst.true_Y);worst['T오차']=abs(worst.pred_T-worst.true_T);worst=worst.sort_values('Y오차',ascending=False).head(6)
    mixture=[]
    for label,d in [('4단계 비선형',read('synthetic_mixed_summary.json')),('원래 특징+비선형·증강',fixed['synthetic_mixed']),('요약 특징 선형',linear['synthetic_mixed']),('630점 좌표 전용 선형',coord['synthetic_mixed'])]:mixture.append({'모델':label,'Y 평균':d['Y_MAE'],'Y 최대':d['Y_max'],'T 평균':d['T_MAE'],'T 최대':d['T_max'],'보조 통과율(합성만)':f"{d['pass_rate']*100:.2f}%"})
    selections=[]
    for item in fixed['selection']:
        if item['stage']=='fixed_D':selections.append({'외부 fold / 최종':item['fold'],'측정':item['config']['measurement'],'특징':item['config']['feature'],'회귀':item['config']['model']['family'],'안정화':item['config']['policy']})
    research=read('research.json');refs=''.join('<li><a href="'+html.escape(r['url'])+'">'+html.escape(r['title'])+'</a><br><b>실제 적용:</b> '+html.escape(r['applied'])+'<br><small>'+html.escape(r['scope'])+'</small></li>' for r in research)
    audit=read('audit.json');verify=read('verification.json');selected=read('selected_verification.json') if (H/'selected_verification.json').exists() else {'status':'running'}
    files=[('사진별 전체 후보 예측','predictions.csv'),('순차 탐색 내부 선택','selection.json'),('원래 특징 개선 예측','fixedfeature_oof.csv'),('원래 특징 개선 내부 선택','fixedfeature_summary.json'),('좌표 전용 선형 예측','coordinate_oof.csv'),('좌표 전용 선형 설정','coordinate_summary.json'),('방법 목록','registered_methods.json'),('논문·자료 적용 목록','research.json'),('모든 모델 통제 실험','controlled_model_summary.csv'),('추론·보존 검증','selected_verification.json')]
    links=' · '.join('<a href="'+v+'">'+k+'</a>' for k,v in files)
    text='''<!doctype html><html lang="ko"><meta charset="utf-8"><title>Y·T 동시 추정 오차 분석·개선</title><style>
    body{font:16px system-ui,"Malgun Gothic",sans-serif;max-width:1280px;margin:36px auto;padding:0 24px;line-height:1.7;color:#183247;background:#fafcfe}h1,h2{color:#164e77}h2{margin-top:36px}a{color:#146f9c}table{border-collapse:collapse;width:100%;font-size:13px;background:white;margin:18px 0}th,td{padding:10px;border:1px solid #d4e0ea;text-align:right}td:first-child,th:first-child{text-align:left}th{background:#e9f1f8}.box{padding:18px 22px;background:#e9f2f8;border-left:5px solid #24729d;border-radius:4px}.limit{background:#fff1d9;border-color:#cc932f}img{width:100%;max-width:1200px}code{background:#e8eef4;padding:2px 4px}li{margin-bottom:12px}small{color:#536c80}.scroll{overflow-x:auto}summary{cursor:pointer;color:#166b94}</style>
    <h1>Y·T 동시 추정: 오차가 커지는 이유와 개선 결과</h1>
    <p><a href="../../ALGORITHM_RULES.md">알고리즘 공통 규칙</a> · 2026-10-22 · 기존 알고리즘·웹·첫 실험은 보존</p>
    <div class="box"><b>동시 추정은 회귀다.</b> Y/T 분류 정확도를 주지표로 쓰지 않는다. 아래 표는 Y·T 평균·최대 절대오차를 먼저 표시한다. '두 축 허용오차 동시 통과율'은 |Y오차|≤0.09 및 |T오차|≤0.09를 모두 만족한 사진의 비율이며 분류 정확도가 아니다. T 물리 단위는 미확인이다.</div>
    <h2>현재 데이터의 오차 비교</h2>'''+table(main_table)+'''<p>비선형+증강 모델은 첫 동시 모델 대비 Y 평균오차를 약71%, T 평균오차를 약57% 줄였다. 가산 구조를 유지하는 좌표 전용 선형 모델은 Y 평균오차를 약23%, T 평균오차를 약22% 줄였다. 모두 같은280장 외부시험 예측이며 재대입 성적이 아니다.</p>
    <div class="box limit"><b>아직 동시 추정 문제를 해결했다고 볼 수 없다.</b> 비선형 모델의 높은 단일 축 통과율은 실제 혼합 자세 성능을 보장하지 않는다. 합성 혼합 진단에서도 약했다. 좌표 선형 모델은 개선됐지만 Y 최대오차0.3504로 허용0.09를 넘는다. 현재 운영의 기존 Y-or-T 모델은 교체하지 않았다.</div>
    <h2>왜 첫 동시 모델이 흔들렸나</h2>
    <ol><li><b>Y와 T 이동 모양이 거의 나란하다.</b> 각 학습 fold에서 잔차로 백색화한 두 방향의 유사도는 '''+f"{min(d['cosine'] for d in diags):.4f}~{max(d['cosine'] for d in diags):.4f}"+'''. 구분 신호가 약해 작은 잔차를 두 축으로 분해할 때 Y를 늘리고 T를 줄이는 상쇄 오차가 생긴다. 첫 모델의 Y/T 오차 상관은 '''+f'{corr:.4f}'+'''.</li>
    <li><b>원래 알고리즘에는 '한 축만 움직였다'는 강한 제한이 있었다.</b> 동시 모델은 그 제한을 풀어 두 값을 자유롭게 맞춘다. 더 일반적인 문제를 풀면서 측정 불확실성이 드러난다. 특징점 수630개가 곧630개의 독립 구분 단서를 뜻하지 않는다.</li>
    <li><b>공간 요약과 모델 선택이 상호작용한다.</b> GLS 기준으로 고른 A/B 표현이 트리 회귀에도 유리한 것은 아니었다. 원래 P12를 보존한 통제 실험이 순차 변경보다 좋았다. 추가한 모든 방법이 개선된 것처럼 보고하지 않는다.</li>
    <li><b>현재 학습·시험 사진은 좌표축 위에만 있다.</b> Y 사진의 T명령은0이고 T 사진의 Y명령은0이다. Y×T 상호작용 항은 항상0이므로 이 자료로 식별할 수 없다. 비선형 모델이 이 분포를 학습하면 단일 축 점수는 좋아져도 혼합 자세에서는 무너질 수 있다.</li></ol>
    <h2>각 단계에 실제 추가한 방법</h2>'''+table(pd.DataFrame(methods))+'''<p>A/B/C/D는 기준 포함7·7·28·8개 설정을 비교했다. 추가로 630점 전체 정보와2×2/4×4/8×8 공간집계에 선형 Ridge/PCR/PLS/GLS를 적용한95개 설정을 비교했다. 매개변수만 바꾼 여러 설정을 서로 다른 논문 방법으로 세지 않았다.</p>
    <h2>단계를 순서대로 추가했을 때</h2>'''+table(pd.DataFrame(stages))+'''<p>A의 가중 집계가 OOF를 악화시키기도 했고, B의 표현 변경도 회귀모델과 함께 봐야 했다. 최종 별도 개선 경로는 <b>원래630점 경계 측정 → 원래 비가중 P12 → ExtraTrees 두 값 회귀 → 학습 잔차 기반 잡음 증강</b>이다. 다른 축을0으로 만드는 후처리·축 분류 게이트는 넣지 않았다. 다만 회귀 자체가 단일 축 학습 분포를 반영할 수 있다.</p>'''+table(pd.DataFrame(selections))+'''<p>좌표 전용 선형 경로는 <b>630점 전체 변위 → 학습 잔차 공분산 축소 GLS</b>를 기본으로 하며, 선택된 표현·공분산은 coordinate_summary.json에 fold별로 공개했다. 공통 절편은 학습 자료로만 추정한다.</p>
    <h2>오차와 단일 축 성능 그림</h2><img src="final_analysis.png" alt="모델별 Y T 평균 최대 절대오차 및 합성 혼합 진단">
    <h2>Y 사진과 T 사진을 따로 본 오차</h2>'''+table(pd.DataFrame(detail))+'''<p>Y 사진 정답은(명령Y,0), T 사진은(0,명령T)이다. 비활성 축 오차도 전체 오차에서 빼지 않았다.</p>
    <h2>개선 비선형 모델의 큰 오차 사진</h2>'''+table(worst[['file','true_Y','true_T','pred_Y','pred_T','Y오차','T오차']])+'''
    <h2>합성 혼합 진단 — 실제 혼합 사진 성적 아님</h2><p>각 외부시험 fold의 Y특징+T특징−학습 공통 원점으로4000개 특징쌍을 만들었다. 특징을 더할 수 있다는 가정 자체가 검증되지 않았다. 특히 품질 통계가 포함된 표현에는 이 가정이 더 약하다. 아래는 모델의 단일 축 분포 의존성을 드러내는 보조 진단이다.</p>'''+table(pd.DataFrame(mixture))+'''
    <h2>평가와 검증</h2><p>저장된 고정4-fold: 학습210/시험70씩 총280장. 모든 정규화·절편·공분산·차원축소·증강·방법 선택은 외부 학습 분할 안에서만 추정했다. 내부3-fold 선택 점수는 최초 합의대로 두 축 허용오차 실패 수 → 최대오차 → 평균오차 순이다. 이번 설명에서 오차를 먼저 보여주도록 바꿨지만, 수행한 선택 기준을 사후 변경하지 않았다.</p>
    <p>기존 fold를 여러 개발 실험에 반복 사용했다. 내부 선택을 지켰어도 독립적인 새 촬영 검증을 대신하지 않는다. 파일명 명령값은 평가 정답으로만 사용하며 실제 계측 자세라고 주장하지 않는다. 픽셀 잡음·밝기·흐림·QR내용 변경과 실제 혼합 촬영의 일반화는 별도로 남아 있다.</p>
    <p>보존 해시 확인 '''+str(audit['preserved_files'])+'''개 파일. 기준 P12 특징 재계산 일치(최대 차이 '''+f"{verify['original_feature_max_delta']:.2e}"+'''), GLS 백색화의 수치 등가성·Woodbury 공분산 계산·280장 분할·오차 재계산·내부 선택 점수 일치를 확인했다. 이미지 바이트 입력2장의 예측도 캐시 예측과 비교했다. 검증 기록은 아래 파일에서 확인할 수 있다.</p>
    <h2>찾아보고 적용한 논문·공식 자료13개</h2><ol>'''+refs+'''</ol>
    <h2>코드·모델·결과</h2><p>'''+links+'''</p><p>실험용 모델: <a href="fixedfeature_model.joblib">현재 사진 오차를 줄인 비선형 모델</a> · <a href="coordinate_model.joblib">좌표 전용 선형 동시 모델</a>. 웹 운영 모델은 그대로다.</p><p>새 사진 실험: <code>python predict.py --center [고정center 경로] --current [현재 사진 경로] --model coordinate_model.joblib</code>. 재현: experiment.py → linear_audit.py → fullfield.py → coordinate_only.py → refine.py → verify.py --images → verify_selected.py → final_report.py.</p></html>'''
    (H/'report.html').write_text(text,encoding='utf-8');make_plot(entries,fixedp,coordp,oldp,fixed,coord)
    (H/'final_results.json').write_text(json.dumps(dict(comparison=rows,original_regression_error_correlation=corr,verifications=selected),indent=2,ensure_ascii=False))

def make_plot(entries,fixedp,coordp,oldp,fixed,coord):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(2,2,figsize=(13,9));short=['Existing OR','First joint','Staged search','P12 + trees/noise','Full + quality','Full coordinates'];xx=np.arange(len(entries))
    for ax,mean,maxi,title in [(axs[0,0],'Y_MAE','Y_max','Y absolute error'),(axs[0,1],'T_MAE','T_max','T absolute error (unit unconfirmed)')]:
        ax.bar(xx-.18,[s[mean] for _,s in entries],.36,label='Mean');ax.bar(xx+.18,[s[maxi] for _,s in entries],.36,label='Maximum');ax.axhline(.09,color='#b34f36',ls='--',label='Tolerance0.09');ax.set_xticks(xx,short,rotation=25,ha='right');ax.set_title(title);ax.legend(fontsize=8)
    for label,p,color in [('First joint',oldp,'#c66848'),('P12 + trees/noise',fixedp,'#1b6b97'),('Full coordinates',coordp,'#418357')]:axs[1,0].scatter(p.pred_Y-p.true_Y,p.pred_T-p.true_T,s=10,alpha=.45,label=label,color=color)
    axs[1,0].set_xlabel('Y error');axs[1,0].set_ylabel('T error');axs[1,0].legend(fontsize=8);axs[1,0].set_title('Error cancellation between axes')
    axs[1,1].bar(['P12 + trees/noise','Full coordinates'],[fixed['synthetic_mixed']['Y_MAE'],coord['synthetic_mixed']['Y_MAE']],label='Y MAE');axs[1,1].set_title('Synthetic mixed-feature diagnostic ONLY');axs[1,1].set_ylabel('Y absolute error (mean)');axs[1,1].axhline(.09,color='#b34f36',ls='--')
    for ax in axs.ravel():ax.grid(axis='y',alpha=.2)
    fig.suptitle('Regression errors first | historical4-fold OOF | real mixed photographs unavailable');fig.tight_layout();fig.savefig(H/'final_analysis.png',dpi=160);plt.close(fig)

if __name__=='__main__':main()
