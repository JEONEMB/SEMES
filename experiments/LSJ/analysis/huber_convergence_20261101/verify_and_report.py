"""Read saved runs; verify comparisons and generate a standalone report/plots."""
from pathlib import Path
import hashlib
import html
import json
import math

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
load = lambda name: json.loads((HERE/name).read_text(encoding='utf-8'))
results, history, summary = load('results.json'), load('histories.json'), load('summary.json')
provenance, settings = load('provenance.json'), load('settings.json')
data = np.load(HERE/'inputs.npz')
by = {(r['input'],r['config']):r for r in results}
hist = {(r['input'],r['config']):r for r in history}
keys = sorted(data.files)
caps = {c['id']:c for c in settings['comparisons']}
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()

checks = []
for item in provenance['source_snapshots']:
    ok = sha(item['installed_file']) == item['sha256'] == sha(HERE/item['snapshot'])
    assert ok
    checks.append({'source':item['module'],'unchanged':ok})
assert sha(provenance['profile_file']) == provenance['profile_sha256']
assert len(results)==72 and len(keys)==8

baseline_differences=[]
late_history=[]
for key in keys:
    x,y = data[key].T
    baseline=by[key,'A']
    a=np.array([baseline['intercept_px'],baseline['slope']])
    for label in caps:
        r=by[key,label]; h=hist[key,label]
        assert len(h['params_b_m'])==len(h['scale_px'])==len(h['deviance'])==r['iteration']
        assert abs(h['deviance'][-1]-h['deviance'][-2])==r['deviance_delta']
        assert r['tol_met'] == (r['finite'] and r['deviance_delta']<=r['tol'])
        assert np.array_equal(np.array(h['params_b_m'][-1]),
                              np.array([r['intercept_px'],r['slope']]))
        # Avoid subtracting two nearly equal full image ordinates: use coefficient differences.
        b=np.array([r['intercept_px'],r['slope']])
        db,dm=b-a
        difference=np.max(abs(db+dm*x))
        baseline_differences.append({'input':key,'config':label,
                                    'max_line_difference_px':float(difference),
                                    'angle_difference_deg':float(np.degrees(np.arctan(b[1])-np.arctan(a[1])))})
        if label in ['B','C']:
            assert np.array_equal(a,b)
    # Observe all iterates of the tightest run after the default stopping iteration.
    hh=np.array(hist[key,'I']['params_b_m'])
    late=hh[baseline['iteration']-1:]-a
    # Difference of two straight lines attains its absolute max at an x-range endpoint.
    err=np.maximum(abs(late[:,0]+late[:,1]*x.min()),abs(late[:,0]+late[:,1]*x.max()))
    late_history.append({'input':key,'start_iteration':baseline['iteration'],
                         'end_iteration':len(hh),'max_departure_from_default_px':float(err.max()),
                         'last_deviance_change':by[key,'I']['deviance_delta'],
                         'last_stop_reason':by[key,'I']['stop_reason']})
worst=max(baseline_differences,key=lambda z:z['max_line_difference_px'])
worst_angle=max(abs(r['angle_difference_deg']) for r in baseline_differences)
validation={'source_checks':checks,'profile_hash_unchanged':True,
            'runs':len(results),'history_lengths_and_final_coefficients_match':True,
            'termination_classified_by_actual_deviance_delta':True,
            'cap_50_100_200_at_1e8_bitwise_identical':True,
            'max_default_to_other_setting_difference':worst,
            'max_default_to_other_angle_difference_deg':worst_angle,
            'baseline_downweighted_points':sum(by[k,'A']['used_downweighted_points'] for k in keys),
            'baseline_warnings':[w for k in keys for w in by[k,'A']['warnings']],
            'late_history':late_history,
            'conclusion_scope':'Numerical convergence on these 8 point sets only; no physical-accuracy or theta success claim.',
            'acceptance':'No newly invented px/degree pass threshold.'}
(HERE/'verification.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2),encoding='utf-8')

# Figure 1: every source point, no selection after fitting.
fig, axes=plt.subplots(4,2,figsize=(14,15), constrained_layout=True)
for i,ref in enumerate(['R01','R02','R03','R04']):
    for j,target in enumerate(['upper_structure','lower_structure']):
        key=ref+'_'+target; r=by[key,'A']; x,y=data[key].T; ax=axes[i,j]
        resid=y-(r['intercept_px']+r['slope']*x)
        w=np.minimum(1,r['tau_px']/np.maximum(abs(resid),np.finfo(float).tiny))
        # Colors represent observational final-residual weights; no points are removed.
        ax.scatter(x,y,c=w,s=4,cmap='viridis',vmin=0,vmax=1,rasterized=True)
        xx=np.array([x.min(),x.max()])
        ax.plot(xx,r['intercept_px']+r['slope']*xx,color='#e3503a',lw=1.5)
        ax.invert_yaxis(); ax.grid(alpha=.2)
        ax.set_title(f"{ref} {'upper' if j==0 else 'lower'} | {r['image_angle_deg']:.9f} deg | {r['iteration']} iterations")
        ax.set_xlabel('Image x (px)'); ax.set_ylabel('Image y (px)')
fig.suptitle('Ordinary Huber, default settings: all input points + fitted line\nAxes are independently scaled; color is final-residual weight (0 to 1).',fontsize=15)
fig.savefig(HERE/'points_and_lines.png',dpi=140); plt.close(fig)

# Figure 2: tightening the stop rule versus stability of the actual line.
fig,axes=plt.subplots(4,2,figsize=(14,14),constrained_layout=True)
for row,key in enumerate(keys):
    ax=axes.flat[row]; x=data[key][:,0]; a=by[key,'A']
    hh=hist[key,'I']; params=np.array(hh['params_b_m'])
    dif=params-np.array([a['intercept_px'],a['slope']])
    err=np.maximum(abs(dif[:,0]+dif[:,1]*x.min()),abs(dif[:,0]+dif[:,1]*x.max()))
    iterations=np.arange(1,len(params)+1)
    ax.plot(iterations,err,color='#1560a7',lw=1)
    ax.set_yscale('symlog',linthresh=1e-13)
    ax.axvline(a['iteration'],color='#c94d33',ls='--',label='Default stop')
    ax.set_title(key+' | strict stop: '+by[key,'I']['stop_reason'])
    ax.set_xlabel('Iteration');ax.set_ylabel('Max line difference from default (px)')
    ax.grid(alpha=.2); ax.legend(fontsize=8)
fig.suptitle('Continued iteration at tolerance 1e-12 (cap 800)\nZero is shown; the plot scale is not an acceptance threshold.',fontsize=15)
fig.savefig(HERE/'convergence_history.png',dpi=140);plt.close(fig)

fmt=lambda x:f'{x:.6g}'
friendly=lambda k:k.replace('_upper_structure',' 상단').replace('_lower_structure',' 하단')
config_rows=''.join(f"<tr><td>{c}</td><td>{r['maxiter']}</td><td>{r['tol']:.0e}</td><td>{r['tol_met_count']}/8</td><td>{r['cap_only_count']}/8</td><td>{r['iteration_min']}–{r['iteration_max']}</td></tr>" for c,r in summary['per_config'].items())
base_rows=''.join(f"<tr><td>{friendly(k)}</td><td>{by[k,'A']['n']:,}</td><td>{by[k,'A']['image_angle_deg']:.9f}</td><td>{by[k,'A']['iteration']}</td><td>{by[k,'A']['deviance_delta']:.3e}</td><td>{by[k,'A']['scale_px']:.6f}</td><td>{by[k,'A']['tau_px']:.6f}</td><td>{by[k,'A']['used_downweighted_points']}</td></tr>" for k in keys)
comparison_rows=''.join(f"<tr><td>{a.replace('_',' → ')}</td><td>{r['max_line_difference_px']:.3e}</td><td>{r['max_abs_angle_difference_deg']:.3e}</td><td>{friendly(r['worst_input'])}</td></tr>" for a,r in summary['max_comparison_differences'].items())
late_rows=''.join(f"<tr><td>{friendly(r['input'])}</td><td>{r['start_iteration']} → {r['end_iteration']}</td><td>{r['max_departure_from_default_px']:.3e}</td><td>{r['last_deviance_change']:.3e}</td><td>{r['last_stop_reason']}</td></tr>" for r in late_history)
all_rows=''.join(f"<tr><td>{friendly(r['input'])}</td><td>{r['config']}</td><td>{r['iteration']}</td><td>{r['stop_reason']}</td><td>{r['deviance_delta']:.3e}</td><td>{r['image_angle_deg']:.12f}</td><td>{r['runtime_ms']:.2f}</td></tr>" for r in results)
source_rows=''.join(f"<li><a href='{s['snapshot']}'>{s['module']}</a> <small>SHA-256: {s['sha256']}</small></li>" for s in provenance['source_snapshots'])
image_rows=''.join(f"<li><b>{r['id']}</b>: <code>{html.escape(r['path'])}</code></li>" for r in provenance['image_checks'])

report=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>일반 Huber: 반복 상한·종료 조건 검증</title>
<style>body{{background:#eef2f6;color:#203247;font:16px/1.75 'Malgun Gothic',sans-serif;margin:0}}main{{max-width:1200px;margin:auto;padding:36px 28px}}h1{{font-size:30px;line-height:1.4}}h2{{font-size:23px;margin-top:42px;border-bottom:2px solid #cbd6e3;padding-bottom:8px}}.lead{{background:#e4f3eb;padding:22px;border-left:6px solid #24734f}}.note{{background:#fff2d7;padding:18px;margin:18px 0}}table{{width:100%;border-collapse:collapse;background:white;font-size:14px}}th,td{{border:1px solid #d6dfe9;padding:10px;text-align:left}}th{{background:#dfe8f2}}.scroll{{overflow:auto}}img{{width:100%;background:white}}code{{overflow-wrap:anywhere}}a{{color:#115ca2}}small{{display:block;overflow-wrap:anywhere;color:#56687a}}details{{margin:18px 0}}.meta{{color:#586b7c}}@media(max-width:700px){{main{{padding:20px 10px}}}}</style>
<main><p class="meta">2026-11-01 · 일반 Huber · 3번(반복 상한), 4번(종료 조건) 수렴성 시험</p>
<h1>기본 50회 · 10⁻⁸ 설정으로 계산이 충분했는가?</h1>
<div class="lead"><b>현재 8개 입력에서는 기본 설정을 유지할 근거가 확인됐다.</b><br>
8개 모두 9–17회에 기본 종료 조건을 충족했다. 상한을 100·200회로 늘려도 계수는 완전히 같았다.
더 엄격한 설정 및 최대 800회 실행과 비교한 최종 직선 위치 차이는 최대 <b>{worst['max_line_difference_px']:.3e}px</b>였다.</div>
<div class="note"><b>엄격한 설정이 모두 수렴했다는 뜻은 아니다.</b> 10⁻¹⁰에서는 2개, 10⁻¹²에서는 7개가 800회에서도 종료 조건을 충족하지 못했다. 이 실행은 반복 제한 종료로 기록했다. 엄격 실행의 마지막 값을 정답으로 간주하지 않았다. 기록된 직선 변화량은 계산 설정 사이의 차이이며 실제 경계 정확도나 θ 보정 오차가 아니다.</div>

<h2>1. 입력·방법을 어떻게 고정했나</h2>
<p>1차 center.bmp와 2차 center1·center2·center3.bmp의 상·하단 경계점 8세트, 총 23,056개를 사용했다. 상단은 영상당 2,712개, 하단은 3,052개이다. 이전 검증에서 원본 Sobel 계산과 대조한 캐시의 SHA-256과 원본 영상 해시가 그대로임을 확인했다.</p>
<p>가림 구간을 제외한 기존 좌우 가시 영역의 점을 합쳐 상단 하나, 하단 하나의 직선을 구했다. Gaussian, 응답값 임계에 의한 삭제, 거리 기준 삭제·재피팅을 추가하지 않았다. 원래 이미지의 x와 절편을 사용하며 임의 좌표 정규화도 추가하지 않았다.</p>
<p>공식 PyPI의 <b>statsmodels 0.15.0</b>을 이 폴더 내부 전용 가상환경에 설치해 그대로 실행했다. 라이브러리 소스를 수정하거나 C++로 재구현하지 않았다. 매 실행에 새 RLM 객체를 만들었다.</p>
<ul><li>직선: y = mx + b, 세로 잔차 r = y − (mx+b)</li><li>HuberT: 기준 상수 1.345</li><li>s = median(|r|) / 0.6744897501960817, τ = 1.345s. <code>center=0</code>인 실제 RLM 코드와 동일.</li><li>초기값: 기본 최소제곱(OLS). s는 매 반복 갱신.</li><li>종료 평가값: 원본 코드의 <code>deviance</code>. 반복 상한과 tol 이외 설정은 동일.</li></ul>
<p>deviance는 <code>rho(resid / tmp_results.scale).sum()</code>으로 계산된다. 가중치 계산에 쓰는 MAD 척도 <code>self.scale</code>와 코드상 별개 변수이다. 비교에서는 이 원본 계산식을 변경하지 않았다. 10⁻⁸은 px 또는 각도 허용오차가 아니다.</p>

<h2>2. 비교 조건과 실제 종료</h2>
<p>A는 공식 기본 설정이다. B–E는 사전에 설명한 민감도 시험 조건이다. D·E가 200회 제한에 걸린 뒤 F–I를 추가해 상한을 400·800회로 늘렸다. 이 확대는 횟수 부족 여부를 확인하기 위한 시험이며 논문의 최적값이나 새 운영 설정이 아니다.</p>
<div class="scroll"><table><tr><th>설정</th><th>최대 반복</th><th>tol</th><th>종료 조건 충족</th><th>반복 제한 종료</th><th>실제 반복 범위</th></tr>{config_rows}</table></div>
<p>반복 수는 라이브러리 <code>fit_history.iteration</code> 그대로이며 초기 OLS를 1로 센다. '종료 조건 충족'은 실제 마지막 두 deviance의 유한한 차이가 tol 이하인지 직접 확인한 결과이다. 총 72개 조건 실행에서 유한한 직선을 얻었다. 조건 충족 실패를 숫자가 반환됐다는 이유로 성공 처리하지 않았다.</p>

<h2>3. 기본 설정의 결과</h2>
<div class="scroll"><table><tr><th>입력</th><th>점 수</th><th>영상 선 각도 °</th><th>반복 수</th><th>마지막 deviance 변화</th><th>s px</th><th>τ px</th><th>가중치 감소 점 수</th></tr>{base_rows}</table></div>
<p>각도는 atan(m)이며 기계 θ가 아니다. 가중치 감소 점 수는 라이브러리가 마지막 가중 최소제곱에 실제 사용한 값 기준이다. 총 {validation['baseline_downweighted_points']:,}개 점의 가중치가 1보다 작았으며 점을 삭제한 것은 아니다. 이는 보조 관찰값으로, 가중치가 줄어야 수렴 검증을 통과한다는 조건을 만들지 않았다.</p>
<img src="points_and_lines.png" alt="기준 4장 상하단 입력점 전체와 일반 Huber 직선">
<p class="meta">점은 전체 표시했다. 색은 최종 반환 계수·s에서 다시 계산한 가중치이다. 마지막 실제 사용 가중치와의 미세한 차이도 results.json에 기록했다. 그래프의 x·y 축 배율은 다르므로 그림의 기울기를 실제 각도로 읽으면 안 된다.</p>

<h2>4. 설정 변경으로 직선이 얼마나 달라졌나</h2>
<p>같은 입력의 두 직선에 대해 <b>max |(b₂−b₁)+(m₂−m₁)x|</b>를 측정 영역에서 계산했다. 매우 가까운 두 영상 좌표를 직접 빼는 대신 계수 차이로 계산했다.</p>
<div class="scroll"><table><tr><th>비교</th><th>최대 위치 차이 px</th><th>최대 각도 차이 °</th><th>위치 차이 최대 입력</th></tr>{comparison_rows}</table></div>
<p>기본 A와 나머지 모든 설정의 최종 결과 차이는 최대 {worst['max_line_difference_px']:.3e}px, 각도는 최대 {worst_angle:.3e}°였다. 이 수치를 실제 영상 경계의 정확도로 해석하지 않는다. 임의의 px·각도 합격선을 새로 설정하지 않았다.</p>

<h2>5. 엄격한 설정은 왜 반복 제한에 걸렸나</h2>
<p>반복 제한에 걸린 실행에서는 최종 직선이 기본 결과 부근에서 극히 작게 변하지만 deviance의 작은 변화가 지정 tol 아래로 내려가지 않았다. 원인은 수치 계산의 정밀도와 평가값 민감도에 관련됐을 가능성이 있으나, 이번 비교만으로 하나의 원인으로 확정하지 않는다.</p>
<p>아래 표는 가장 엄격한 I 실행에서, 기본 A가 멈춘 반복 이후 <b>모든 중간 직선</b>이 A로부터 얼마나 벗어났는지도 확인한 것이다. 마지막 직선만 우연히 비슷했는지 구분한다.</p>
<div class="scroll"><table><tr><th>입력</th><th>관찰 반복 구간</th><th>A로부터 최대 위치 차이 px</th><th>마지막 deviance 변화</th><th>I 종료 사유</th></tr>{late_rows}</table></div>
<img src="convergence_history.png" alt="엄격한 계산의 매 반복 직선과 기본 결과 사이 최대 위치 차이">
<p>선의 변화는 반복 내내 기록했다. 800회 제한에 걸린 실행의 수학적 수렴을 증명했다고 주장하지 않는다. 다만 이들 설정으로 계산을 연장해도 기본 직선이 의미 있게 이동하는 현상은 관찰되지 않았다.</p>

<h2>6. 이번 결과로 정하는 범위</h2>
<p><b>3번: 현재 8개 입력에서 최대 50회는 충분했다.</b> 실제로 모두 17회 이내에 기본 종료 조건을 충족했고 상한 증가가 결과에 영향을 주지 않았다.</p>
<p><b>4번: 이번 입력의 수치 계산에는 기본 10⁻⁸을 유지하는 것을 권한다.</b> 더 엄격하게 계산해도 직선 변화는 위에 기록한 규모였으며, 엄격한 tol은 오히려 반복 제한에 도달했다. 이는 현재 입력에 대한 수치적 근거이며 다른 촬영 조건 전체의 보증이나 θ 보정 성능 검증이 아니다.</p>
<p>더 작은 tol이 자동으로 더 정확한 경계나 더 좋은 θ 보정을 뜻하지 않는다. 이번 작업은 τ 기준값 탐색, 점 선택 변경, 새로운 통계 가설검정, Y/θ 알고리즘 변경을 포함하지 않는다.</p>

<details><summary>72개 실행 전체 기록</summary><div class="scroll"><table><tr><th>입력</th><th>설정</th><th>반복 수</th><th>종료 사유</th><th>deviance 변화</th><th>영상 각도 °</th><th>단일 실행 ms</th></tr>{all_rows}</table></div><p>실행 시간은 반복 측정한 성능 벤치마크가 아닌 해당 실행의 참고 기록이다.</p></details>
<details><summary>원본 이미지 경로</summary><ul>{image_rows}</ul></details>
<h2>근거와 재현 파일</h2>
<ul><li><a href="https://www.statsmodels.org/stable/generated/statsmodels.robust.robust_linear_model.RLM.fit.html">공식 RLM.fit 설정 문서</a> · <a href="https://www.statsmodels.org/stable/_modules/statsmodels/robust/robust_linear_model.html">공식 소스</a></li>
<li><a href="https://www.statsmodels.org/stable/generated/statsmodels.robust.norms.HuberT.html">HuberT 기준값</a> · <a href="https://www.mathworks.com/help/stats/robustfit.html">1.345의 정규오차 효율 근거</a> (MATLAB 전처리는 사용하지 않음)</li>
<li><a href="settings.json">고정 설정 및 비교 조건</a> · <a href="summary.json">결과 요약</a> · <a href="verification.json">교차 확인 기록</a></li>
<li><a href="results.csv">72개 실행 CSV</a> · <a href="comparisons.csv">직선 차이 CSV</a> · <a href="histories.json">매 반복 계수·척도·deviance</a></li>
<li><a href="provenance.json">입력 해시·환경·버전</a> · <a href="install_report.json">공식 wheel 설치 출처·해시</a> · <a href="run_experiment.py">실행 코드</a> · <a href="verify_and_report.py">검산·보고서 코드</a></li>{source_rows}</ul>
<p class="meta">실행: <code>.venv/Scripts/python.exe run_experiment.py</code>. 보고서 생성은 기존 Python의 matplotlib을 사용했다. 원본 이미지·기존 알고리즘 코드는 수정하지 않았다.</p></main></html>'''
(HERE/'report.html').write_text(report,encoding='utf-8')
(HERE/'report.txt').write_text(
    '일반 Huber 수렴성 검증\n'
    '기준 4장 상하단 8입력, 23056점, 9설정 72회.\n'
    '기본 50회/1e-8: 8개 모두 9~17회에 종료 조건 충족.\n'
    '상한 100/200, tol=1e-8: 기본값과 계수 완전 동일.\n'
    f'다른 설정과 기본 결과의 최종 위치 차이 최대 {worst["max_line_difference_px"]:.12g}px.\n'
    'tol=1e-10: 800회에서도 2개 반복 제한, tol=1e-12: 800회에서도 7개 반복 제한.\n'
    '권고: 현재 입력에서는 기본 50회/1e-8 유지. 물리적 정확도·theta 보정 성공 판정 아님.\n',encoding='utf-8')
outputs={str(p.relative_to(HERE)):sha(p) for p in HERE.iterdir()
         if p.is_file() and p.name not in ['output_hashes.json','install.log','execution.log']}
(HERE/'output_hashes.json').write_text(json.dumps(outputs,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(validation,ensure_ascii=False,indent=2))
