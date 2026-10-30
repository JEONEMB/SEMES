from pathlib import Path
import html, json, sys, hashlib
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
BASE=HERE.parent
LABELS={
 'baseline':'기존: 공간 OLS + 기존 공분산',
 'ols_euclidean':'공간 OLS + 동일 가중',
 'ols_wls':'공간 OLS + 대각 WLS',
 'ols_gls_lw':'공간 OLS + 공분산 LW',
 'spatial_wls':'공간 WLS + 기존 공분산',
 'spatial_gls_lw':'공간 GLS(LW) + 기존 공분산',
 'spatial_gls_lw_metric_lw':'공간 GLS(LW) + 공분산 LW',
 'direct410_identity':'410변위 직접 + 동일 가중',
 'direct410_diagonal':'410변위 직접 + 대각 WLS',
 'direct410_ledoit_wolf':'410변위 직접 + GLS(LW)',
}

def main():
    metrics=pd.read_csv(HERE/'metrics.csv')
    raw=pd.read_csv(HERE/'direct410/metrics.csv')
    raw=raw[raw.method!='baseline12_fixed001'].copy()
    raw['protocol']=raw.apply(lambda r: 'full280' if r.protocol=='resubstitution' else
        f'repeated5_seed{int(r.seed)-7000}' if r.protocol=='repeated_stratified5' else
        'session_out_missing_origin',axis=1)
    raw['correct']=raw.axis_sign_correct
    raw['failed']=raw.n-raw.strict_pass09_count
    raw['Y_correct']=np.rint(raw.Y_axis_sign_accuracy*raw.Y_n).astype(int)
    raw['T_correct']=np.rint(raw.T_axis_sign_accuracy*raw.T_n).astype(int)
    metrics=pd.concat([metrics,raw],ignore_index=True)
    metrics.to_csv(HERE/'combined_metrics.csv',index=False,encoding='utf-8-sig')
    pred=pd.read_csv(HERE/'predictions.csv')
    direct=pd.read_csv(HERE/'direct410/all_predictions.csv')
    direct=direct[direct.method!='baseline12_fixed001'].copy()
    direct['protocol']=direct.apply(lambda r: 'full280' if r.protocol=='resubstitution' else
        f'repeated5_seed{int(r.seed)-7000}' if r.protocol=='repeated_stratified5' else
        'session_out_missing_origin',axis=1)
    direct=direct.rename(columns={'joint_correct':'correct','strict_pass09':'pass09',
                                'error_Y_abs':'error_Y','error_T_abs':'error_T'})
    pred=pd.concat([pred,direct],ignore_index=True)
    summaries=[];paired=[]
    repeat=pred[pred.protocol.str.startswith('repeated5_seed')]
    baseline=repeat[repeat.method=='baseline'][['protocol','file','pass09']].rename(columns={'pass09':'baseline_pass'})
    for method,label in LABELS.items():
        f=metrics[(metrics.method==method)&(metrics.protocol=='full280')].iloc[0]
        cv=metrics[(metrics.method==method)&metrics.protocol.str.startswith('repeated5_seed')]
        assert len(cv)==20 and (cv.n==280).all()
        p=repeat[repeat.method==method]
        p=p.merge(baseline,on=['protocol','file'],validate='one_to_one')
        fixed=int((~p.baseline_pass&p.pass09).sum());new=int((p.baseline_pass&~p.pass09).sum())
        assert len(p)==5600
        row=dict(method=method,label=label,full_correct=int(f.correct),full_failed=int(f.failed),
            full_Y_max=float(f.vector_Y_max),full_T_max=float(f.vector_T_max),
            cv_predictions=5600,cv_failed=int(cv.failed.sum()),cv_perfect_rounds=int((cv.failed==0).sum()),
            cv_failed_per_round_min=int(cv.failed.min()),cv_failed_per_round_max=int(cv.failed.max()),
            cv_Y_correct=int(cv.Y_correct.sum()),cv_Y_total=1600,cv_T_correct=int(cv.T_correct.sum()),cv_T_total=4000,
            cv_Y_accuracy=float(cv.Y_correct.sum()/1600),cv_T_accuracy=float(cv.T_correct.sum()/4000),
            cv_Y_max=float(cv.vector_Y_max.max()),cv_T_max=float(cv.vector_T_max.max()),
            fixed_vs_baseline=fixed,new_failures_vs_baseline=new,
            unique_failed_images=int(p.loc[~p.pass09,'file'].nunique()))
        summaries.append(row)
        for protocol,pp in p.groupby('protocol'):
            paired.append(dict(method=method,protocol=protocol,baseline_failed=int((~pp.baseline_pass).sum()),
                method_failed=int((~pp.pass09).sum()),fixed=int((~pp.baseline_pass&pp.pass09).sum()),
                newly_failed=int((pp.baseline_pass&~pp.pass09).sum())))
    summary=pd.DataFrame(summaries)
    summary.to_csv(HERE/'comparison_summary.csv',index=False,encoding='utf-8-sig')
    (HERE/'comparison_summary.json').write_text(json.dumps(summaries,ensure_ascii=False,indent=2),encoding='utf8')
    pd.DataFrame(paired).to_csv(HERE/'paired_comparison.csv',index=False,encoding='utf-8-sig')
    failures=repeat[~repeat.pass09].groupby(['method','file','axis','value']).size().reset_index(name='failures_out_of_20')
    failures.to_csv(HERE/'failed_image_frequency.csv',index=False,encoding='utf-8-sig')
    z=np.load(BASE/'measurement/gn17_spread14.npz')
    np.save(HERE/'models/center_displacement.npy',z['D'][z['names']=='center.bmp'][0].astype(float))
    for path in (HERE/'models').glob('*.json'):
        obj=json.loads(path.read_text(encoding='utf8'))
        obj['point_feature_path']='../../measurement/gn17_spread14_features.npz'
        path.write_text(json.dumps(obj,indent=2),encoding='utf8')

    def table(rows, columns):
        out='<div class="scroll"><table><thead><tr>'+''.join('<th>'+html.escape(label)+'</th>' for key,label in columns)+'</tr></thead><tbody>'
        for row in rows:
            out+='<tr>'+''.join('<td>'+html.escape(str(row[key]))+'</td>' for key,label in columns)+'</tr>'
        return out+'</tbody></table></div>'
    full=[];cv=[]
    for r in summaries:
        full.append(dict(label=r['label'],correct=f"{r['full_correct']}/280",Y=f"{r['full_Y_max']:.6f}",T=f"{r['full_T_max']:.6f}"))
        cv.append(dict(label=r['label'],Y=f"{r['cv_Y_correct']}/1600 ({r['cv_Y_accuracy']*100:.4f}%)",
            T=f"{r['cv_T_correct']}/4000 ({r['cv_T_accuracy']*100:.4f}%)",fail=r['cv_failed'],
            perfect=f"{r['cv_perfect_rounds']}/20",maxY=f"{r['cv_Y_max']:.6f}",maxT=f"{r['cv_T_max']:.6f}"))
    failure_rows=failures.copy();failure_rows['method']=failure_rows.method.map(LABELS)
    content='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>GLS 적용 비교 — 기존 모델 보존</title><style>
body{font-family:'Malgun Gothic',sans-serif;color:#17324a;background:#f4f7fa;margin:0;line-height:1.7}main{max-width:1250px;margin:auto;padding:36px 26px 64px}h1{font-size:32px;line-height:1.3}h2{font-size:24px;margin-top:42px}p,li{font-size:16px}table{border-collapse:collapse;width:100%;background:white;font-size:14px}th,td{padding:12px 10px;border-bottom:1px solid #d7e1e9;text-align:left}th{background:#17324a;color:white;white-space:nowrap}tr:nth-child(even){background:#edf3f7}.scroll{overflow:auto}img{width:100%;height:auto;background:white}a{color:#006f87}code{background:#e5edf2;padding:2px 5px}pre{padding:18px;background:#e5edf2;overflow:auto}.note{border-left:4px solid #cc6b21;padding-left:18px}.lead{font-size:20px}small{color:#486174}
</style><main><h1>GLS 적용 전후 비교</h1><p>2026-10-14 · QR 3개 / 고정 특징점 205개 / Y 80장·T 200장</p>
<p class="lead"><strong>기존 모델은 그대로 보존했다. GLS로 학습 자료의 이동량 오차는 줄었지만, 학습에서 제외한 사진의 축·부호 판정은 개선되지 않았다.</strong></p>
<p class="note">이 자료는 기존 280장으로 수행한 개발 실험이다. 전체 재대입과 학습 제외 평가를 구분한다. 반복 검증의 5,600건은 새로운 사진 5,600장이 아니라 같은 280장을 20가지 분할로 평가한 횟수다.</p>
<h2>1. 무엇을 기록하고 고정했나</h2><p>기존 모델, 추론·학습·추적 코드, 205점 원시 측정 캐시, 280장 결과, 기존 교차검증 결과를 <a href="baseline_snapshot/hashes.json">baseline_snapshot</a>에 복사하고 SHA-256을 기록했다. 실험 후 원본과 사본의 해시 일치를 확인했다. 기존 배포 모델과 이전 PPT는 변경하지 않았다.</p>
<p>같은 특징점, 같은 명령 정답, 같은 6개 0점 후보, 축별 공유 2차 곡선, 같은 부호 제한·최적화와 같은 분할을 사용했다. 7개 주 비교는 특징점 요약·후단 가중 방법을 바꾸고, 추가 3개는 12차원 요약을 생략해 410개 좌표를 직접 사용했다. 추가 3개는 표현력도 달라져 순수한 가중치 비교와 구분해야 한다.</p>
<h2>2. GLS를 어디에 적용했나</h2><p>기존 방식은 205점×2방향의 변위를 공간 OLS로 12계수로 요약한 뒤, 학습 잔차의 공분산을 반영해 Y/T 후보를 선택한다. 이미 후단에는 일반화 최소제곱 형태의 비용이 있었다. 이번 실험의 변화는 다음과 같다.</p>
<ul><li><strong>동일 가중:</strong> 상관관계·분산 차이를 모두 무시하는 대조군.</li><li><strong>WLS:</strong> 좌표별 오차 분산은 반영하되 좌표 사이 상관관계는 무시.</li><li><strong>공분산 LW:</strong> 후단 12계수의 공분산을 Ledoit–Wolf 방식으로 축소 추정.</li><li><strong>공간 GLS(LW):</strong> 410개 원시 변위의 오차 공분산을 추정해 12계수를 추출하는 단계부터 가중.</li><li><strong>410변위 직접:</strong> 요약 없이 원시 변위마다 곡선을 학습하고 동일 가중/WLS/GLS 비용을 비교.</li></ul>
<pre>기존 공간 OLS:  f = (HᵀH)⁻¹Hᵀd
공간 GLS:      f = (HᵀΣ⁻¹H)⁻¹HᵀΣ⁻¹d
이동 후보 비용: (f − b − s·v − c·v²)ᵀΩ⁻¹(f − b − s·v − c·v²)</pre>
<p>d는 410개 픽셀 변위, H는 고정 공간 설계행렬이다. Σ는 각 학습 분할에서 원시 변위의 이동 곡선을 적합한 잔차로 추정했다. 평가 사진의 정답·시행 ID로 Σ나 변환을 맞추지 않았다. 후단 Ω도 학습 잔차로만 추정했다. Σ는 실측 센서 잡음만의 공분산이 아니라 모델 미설명 오차도 포함하므로 정확히는 feasible GLS(FGLS)다.</p>
<p>모든 출력에 동일한 설명변수를 사용하는 다변량 곡선 회귀에서 출력 공분산만 바꾸면 회귀계수 자체는 OLS와 같다. 이번 변화의 효과는 공간 요약과 후보 비용에서 발생한다. 계산은 NumPy로 구현했고, LW 공분산은 scikit-learn을 사용했다. 원리 참고: <a href="https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.GLS.html">GLS 공식 문서</a>, <a href="https://scikit-learn.org/stable/modules/generated/sklearn.covariance.LedoitWolf.html">Ledoit–Wolf 공식 문서</a>.</p>
<h2>3. 학습에 사용한 280장 전체에 다시 적용</h2><p>정답 수는 축+부호를 함께 맞힌 수다. 최대오차는 오분류도 포함한 Y/T 출력 벡터의 절대오차다. Y는 mm, T는 원자료의 명령 단위다.</p>'''
    content+=table(full,[('label','방법'),('correct','축+부호 정답'),('Y','Y 최대오차'),('T','T 최대오차')])
    content+='''<p>공간 GLS(LW)의 재대입 최대오차는 Y 0.036241→0.002881, T 0.002864→0.001258로 감소했다. 이 숫자만으로 채택하면 안 된다. 아래 학습 제외 결과에서는 오류가 늘었다.</p>
<h2>4. 같은 20가지 분할로 학습 제외 평가</h2><p>각 회차는 5분할이다. 280장을 56장씩 나누고, 나머지 224장으로 모든 계수와 공분산을 다시 학습해 제외한 56장을 예측했다. 다섯 묶음을 합쳐 매 회차 280장 전부를 한 번씩 평가했다. 난수 시드 7000~7019를 모든 방법에 동일하게 적용했다. 분할·학습 범위는 <a href="splits.json">splits.json</a>에 기록했다.</p>'''
    content+=table(cv,[('label','방법'),('Y','Y 축+부호'),('T','T 축+부호'),('fail','누적 실패/5600'),('perfect','280장 모두 통과한 회차'),('maxY','Y 최대오차'),('maxT','T 최대오차')])
    content+='''<p>실패는 축·부호 오류 또는 두 축 중 하나라도 오차 0.09 초과다. 기존 방식은 누적 13회, 공간 GLS(LW)는 66회, 직접 410 GLS는 108회 실패했다. 이번에 정한 GLS 설정에서는 기존 방식보다 강건한 대안을 찾지 못했다.</p>
<img src="figures/01_repeated_failures_and_perfect_rounds.png" alt="7가지 가중 방법의 반복 검증 실패와 전부 합격 회차 비교">
<img src="figures/02_max_vector_errors_including_misclassification.png" alt="오분류를 포함한 Y와 T 최대 절대오차 비교">
<h2>5. 왜 재대입만 좋아졌을까</h2><p>GLS는 모든 점을 독립 정보처럼 세지 않고 흔들림과 상관관계를 반영한다. 하지만 여기서는 공분산 자체도 제한된 학습 자료에서 추정해야 한다. 원시 410차원은 한 분할의 학습 사진 224장보다 많아 공분산 축소가 필요하다. 기존 자료에 잘 맞는 가중치가 학습 제외 자료에서도 잘 맞는지는 별개다. 관측된 결과는 이러한 일반화 문제와 일치하지만, 단일 원인을 확정한 것은 아니다.</p>
<p>또한 여섯 0점 후보를 사용하는 구조는 그대로다. GLS는 새로운 시행의 알 수 없는 0점을 자동으로 해결하지 않는다. 전체 시행 제외 진단도 <a href="combined_metrics.csv">전체 지표</a>에 보존했다. 그 진단에서는 T+ 또는 T−를 통째로 빼면 해당 부호 분기 자체가 학습에 없어 구조적으로 실패하므로, 주 비교와 섞지 않았다.</p>
<h2>6. 검증 및 남긴 파일</h2><ul><li>원본 및 기존 모델 해시 유지, 23개 기존 평가 결과 재현, 기존 280개 예측과 최대 차이 2.91×10⁻⁹ 미만.</li><li>각 방법·평가 회차마다 280개 고유 사진이 모두 포함되는지 확인.</li><li>학습 제외 사진의 특징·정답·그룹을 크게 바꿔도 학습된 GLS 변환·모델·공분산이 변하지 않음을 확인.</li><li>동일 가중의 공간 GLS가 공간 OLS와 일치하는지 확인.</li><li>직접 410 방식은 별도 구현에서도 기존 기준 성적을 재현.</li></ul>
<p><a href="comparison_summary.csv">10가지 방법 요약 CSV</a> · <a href="paired_comparison.csv">같은 분할의 개선·악화 비교</a> · <a href="failed_image_frequency.csv">실패 사진과 빈도</a> · <a href="checks.json">검증 결과</a> · <a href="compare.py">주 실험 코드</a> · <a href="direct410/run_direct410.py">직접 410 실험 코드</a> · <a href="predict.py">실험 모델 두 이미지 추론 코드</a></p>
<p>후보 7개의 전체 학습 모델은 models 폴더에 별도로 보존했다. 기존 infer.py의 기본 모델은 교체하지 않았다. 통계 설정은 이번 비교의 개발 선택이며, 최종 독립 성적을 얻으려면 새 촬영 시행을 고정 모델로 평가해야 한다.</p>
</main></html>'''
    (HERE/'GLS_비교결과.html').write_text(content,encoding='utf8')
    print(summary.to_string(index=False))

if __name__=='__main__':main()
