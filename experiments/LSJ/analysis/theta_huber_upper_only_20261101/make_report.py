from pathlib import Path
import json
import html
import hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]


def read(p):return json.loads(p.read_text(encoding='utf-8'))
def f(v,n=6):return '—' if v is None else f'{v:.{n}f}'
def esc(v):return html.escape(str(v))


def main():
    s=read(HERE/'summary.json'); rows=read(HERE/'results.json')
    oldrows=read(ROOT/'analysis/theta_huber_holdout_20261101/results.json')
    old={r['id']:r for r in oldrows}
    cfg=read(HERE/'frozen_calibration.json')['model']
    plt.rcParams.update({'font.family':'Malgun Gothic','axes.unicode_minus':False,'font.size':10})
    fig,axes=plt.subplots(1,2,figsize=(14,5.8),layout='constrained')
    for ax,d in zip(axes,(2,3)):
        rr=[r for r in rows if r['dataset']==d and r['execution_success']]
        ax.scatter([r['theta_deg'] for r in rr],[old[r['id']]['theta_error_deg'] for r in rr],
                   s=28,c='#9da6b3',alpha=.65,label='기존 상·하단 결합')
        ax.scatter([r['theta_deg'] for r in rr],[r['theta_error_deg'] for r in rr],
                   s=28,c='#1264b5',alpha=.85,label='상단 전용')
        ax.axhspan(-.005,.005,color='#2a9458',alpha=.13,label='허용 범위 ±0.005°')
        ax.axhline(0,color='#555',lw=.8)
        for y in (-.005,.005):ax.axhline(y,color='#268254',ls='--',lw=1)
        ax.set(title=f'{d}차 동일 사진: 추정 θ - 라벨 θ',xlabel='라벨 θ (°)',ylabel='추정 오차 (°)',ylim=(-.04,.04))
        ax.grid(alpha=.17)
        ax.legend(loc='upper center',bbox_to_anchor=(.5,-.16),ncol=2,fontsize=9)
    fig.savefig(HERE/'comparison.png',dpi=160)
    fig.savefig(HERE/'comparison.svg')
    plt.close(fig)

    table=[]
    for label,b,n in [('2차 20장',s['baseline_by_dataset']['2'],s['by_dataset']['2']),
                       ('3차 234장',s['baseline_by_dataset']['3'],s['by_dataset']['3']),
                       ('전체 254장',s['baseline_overall'],s['overall'])]:
        for method,r in [('기존 상·하단',b),('상단 전용',n)]:
            table.append(f'<tr><th>{label}</th><td>{method}</td><td>{r["within_tolerance"]}/{r["images"]} ({r["pass_rate_percent_all_images"]:.1f}%)</td>'
                f'<td>{f(r.get("mae_deg"))}</td><td>{f(r.get("rmse_deg"))}</td><td>{f(r.get("max_absolute_error_deg"))}</td><td>{r["execution_failures"]}</td></tr>')
    detail=[]
    for r in rows:
        b=old[r['id']]
        state='failure' if not r['execution_success'] else 'pass' if r['within_tolerance'] else 'outside'
        change='실행 실패' if not r['execution_success'] else '감소' if r['absolute_theta_error_deg']<b['absolute_theta_error_deg'] else '증가' if r['absolute_theta_error_deg']>b['absolute_theta_error_deg'] else '동일'
        name={'failure':'실행 실패','pass':'통과','outside':'범위 초과'}[state]
        detail.append(f'<tr data-d="{r["dataset"]}" data-s="{state}"><td>{r["id"]}</td><td>{r["dataset"]}차</td><td>{r["y_mm"]:g}</td><td>{r["theta_deg"]:g}</td>'
            f'<td>{f(b.get("theta_error_deg"))}</td><td>{f(r.get("theta_error_deg"))}</td><td>{change}</td><td class="{state}">{name}</td>'
            f'<td class="file" title="{esc(r["path"])}">{esc(r["file"])}</td></tr>')
    groups=[]
    for r in s['by_dataset_y']:
        groups.append(f'<tr><td>{r["dataset"]}차</td><td>{r["y_mm"]:g}</td><td>{r["within_tolerance"]}/{r["images"]}</td>'
                      f'<td>{f(r.get("mean_error_deg"))}</td><td>{f(r.get("max_absolute_error_deg"))}</td></tr>')
    n=s['overall'];b=s['baseline_overall'];p=s['paired']
    text=f'''<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>상단 전용 θ 보정 · 기존 결합 모델 비교</title><style>
body{{margin:0;background:#f3f6fa;color:#14233b;font:15px/1.7 Malgun Gothic,Arial,sans-serif}}main{{max-width:1280px;margin:auto;padding:28px}}
h1{{font-size:27px}}h2{{font-size:21px;margin-top:32px}}.box{{background:#fff;border:1px solid #d8e0ea;border-radius:10px;padding:22px;margin:18px 0}}
.outcome{{border-left:6px solid #2c70ad}}.note{{color:#53637a;font-size:14px}}.big{{font-size:26px;font-weight:bold}}.scroll{{overflow:auto}}
table{{border-collapse:collapse;background:white;width:100%;font-size:14px}}th,td{{padding:10px;border-bottom:1px solid #dce3ec;text-align:right;white-space:nowrap}}
th:first-child,td:first-child{{text-align:left}}thead{{background:#eaf0f8}}img{{width:100%;height:auto}}a{{color:#175ba6}}code{{font-family:Consolas,monospace}}
.pass{{color:#167d42}}.outside,.failure{{color:#b6472c}}.file{{max-width:330px;white-space:normal;word-break:break-word;text-align:left;font-size:12px}}select{{padding:8px;margin-right:14px}}
</style></head><body><main><h1>하단을 제외한 상단 전용 θ 보정 비교</h1><p>2026-11-01 · 동일 학습 40장 · 동일 테스트 254장 · 성공 기준 ±0.005°</p>
<div class="box outcome"><strong>상단 전용에서 평균·최대 오차는 줄었지만, 통과율 개선은 작았고 목표 정확도에는 미달했다.</strong>
<p class="big">{b['within_tolerance']}/254 ({b['pass_rate_percent_all_images']:.1f}%) → {n['within_tolerance']}/254 ({n['pass_rate_percent_all_images']:.1f}%)</p>
<p>평균 절대오차 {b['mae_deg']:.6f}° → {n['mae_deg']:.6f}°<br>최대 절대오차 {b['max_absolute_error_deg']:.6f}° → {n['max_absolute_error_deg']:.6f}°</p>
<p>상단 전용도 실행 실패는 {n['execution_failures']}장이다. 하단 제외만으로 ±0.005° 문제가 해결되지는 않았다.</p></div>
<h2>데이터별 비교</h2><div class="scroll"><table><thead><tr><th>데이터</th><th>방식</th><th>±0.005° 이내</th><th>평균 절대오차 (°)</th><th>RMSE (°)</th><th>최대 절대오차 (°)</th><th>실행 실패</th></tr></thead><tbody>{''.join(table)}</tbody></table></div>
<h2>같은 사진에서 무엇이 달라졌는가</h2><div class="box"><ul>
<li>절대오차 감소: {p['absolute_error_decreased']}장. 증가: {p['absolute_error_increased']}장.</li>
<li>기존 실패 → 상단 전용 통과: {p['old_fail_new_pass']}장.</li>
<li>기존 통과 → 상단 전용 실패: {p['old_pass_new_fail']}장.</li>
<li>두 방식 모두 통과: {p['both_pass']}장. 두 방식 모두 실패: {p['both_fail']}장.</li>
</ul><p>통과 이미지 수가 순증한 것은 {n['within_tolerance']-b['within_tolerance']}장이다. 기존 통과 사진이 모두 유지되면서 4장만 추가된 결과는 아니다.</p></div>
<div class="box"><img src="comparison.png" alt="기존 결합 방식과 상단 전용 방식의 동일 사진별 theta 오차 비교"><p class="note">회색: 기존 결합. 파란색: 상단 전용. 녹색 띠: ±0.005°. 두 패널은 같은 세로축 범위를 사용한다.</p></div>
<h2>상단 전용의 Y 조건별 결과</h2><table><thead><tr><th>데이터</th><th>Y 라벨 (mm)</th><th>허용 범위 이내</th><th>평균 부호 오차 (°)</th><th>최대 절대오차 (°)</th></tr></thead><tbody>{''.join(groups)}</tbody></table>
<p class="note">구간마다 θ 구성이 달라 Y만의 영향으로 해석하지 않는다. 상단 전용 전체의 평균 부호 오차는 {n['mean_error_deg']:.6f}°다.</p>
<h2>정확히 무엇을 바꿨는가</h2><div class="box"><p>상단 경계점 추출과 Huber 직선 피팅은 기존 소스를 그대로 재사용했다. <strong>254장 모두에서 상단 측정 출력이 기존과 정확히 같았다.</strong></p>
<table><thead><tr><th>항목</th><th>기존</th><th>상단 전용</th></tr></thead><tbody>
<tr><td>측정 특징</td><td>상단·하단 영상 각도</td><td>상단 영상 각도만</td></tr>
<tr><td>환산식</td><td>상·하단 φ = b + aY×Y + aθ×θ</td><td>상단 φ = b + a×θ</td></tr>
<tr><td>추정</td><td>두 식을 풀어 Y·θ 추정</td><td>θ = (φ − b) / a</td></tr>
<tr><td>Y 영향 처리</td><td>내부 Y 추정으로 분리 시도</td><td>별도 보정 없음, Y 라벨 미사용</td></tr>
</tbody></table><p>상단 각도 하나로 미지수 Y와 θ를 동시에 풀 수 없으므로 환산식을 함께 변경했다. 기존과 동일한 학습용 40장으로 상단 전용 계수만 다시 계산했다.</p>
<p><code>b = {cfg['intercept_deg']:.17g}</code><br><code>a = {cfg['per_theta_deg']:.17g}</code></p>
<p>상단 ROI, 가림 범위, Sobel, 일반 Huber, 학습·테스트 이미지 구분, θ 목표 0°, ±0.005° 기준은 유지했다. 테스트 후 계수 수정은 하지 않았다.</p></div>
<h2>하단이 원인이라고 볼 수 있는가</h2><div class="box"><p><strong>상단 전용 모델이 기존 결합 모델보다 평균·최대 오차가 작았다는 것은 확인했다.</strong> 그러나 통과율은 32.3%에서 33.9%로 조금 올랐고, 112장은 절대오차가 더 커졌다.</p>
<p>하단 특징 제거와 함께 Y·θ 환산 구조도 바뀌었으므로, 개선분 전체를 하단 경계점 추출 오류 때문이라고 단정할 수 없다. 이 비교는 현재 두 모델의 성능 차이를 보여준다.</p>
<p>하단을 전혀 사용하지 않는지는 별도로 확인했다. 학습 이미지 복사본에서 상단 영역 아래를 검정/흰색으로 바꾸어도 상단 기울기와 보정치가 동일했고, 새 실행 파일은 하단 추정 코드를 링크하지 않는다.</p></div>
<h2>사진별 결과</h2><p>데이터 <select id="d"><option value="all">전체</option><option value="2">2차</option><option value="3">3차</option></select>
상단 전용 결과 <select id="s"><option value="all">전체</option><option value="pass">통과</option><option value="outside">범위 초과</option><option value="failure">실행 실패</option></select><span id="count"></span></p>
<div class="scroll"><table id="details"><thead><tr><th>ID</th><th>데이터</th><th>Y (mm)</th><th>θ 라벨 (°)</th><th>기존 오차 (°)</th><th>상단 전용 오차 (°)</th><th>절대오차</th><th>상단 전용 결과</th><th>파일</th></tr></thead><tbody>{''.join(detail)}</tbody></table></div>
<h2>평가 범위와 기록</h2><p>오차는 추정 θ − θ 라벨이다. 실제 기계 이동 후 측정이 아니다. 기존 실패를 본 뒤 같은 테스트 254장을 재사용한 비교이므로 독립 최종 검증으로 해석하지 않는다. 원본 이미지·TXT·기존 모델은 수정하지 않았다.</p>
<p><a href="summary.json">요약 수치</a> · <a href="results.json">전체 실행 출력</a> · <a href="protocol.json">고정 조건·해시</a> · <a href="frozen_calibration.json">상단 전용 계수</a> · <a href="../../algorithm/theta_huber_upper_only_20261101/README.md">수정 코드·실행 방법</a></p>
</main><script>function filter(){{let count=0;const d=document.getElementById('d').value,s=document.getElementById('s').value;document.querySelectorAll('#details tbody tr').forEach(r=>{{const show=(d==='all'||r.dataset.d===d)&&(s==='all'||r.dataset.s===s);r.hidden=!show;if(show)count++;}});document.getElementById('count').textContent=count+'장 표시';}}document.getElementById('d').onchange=filter;document.getElementById('s').onchange=filter;filter();</script></body></html>'''
    (HERE/'report.html').write_text(text,encoding='utf-8')
    names=['summary.json','results.json','protocol.json','holdout_manifest.json','frozen_calibration.json','comparison.png','comparison.svg','report.html','evaluate.py','make_report.py']
    (HERE/'output_hashes.json').write_text(json.dumps({n:hashlib.sha256((HERE/n).read_bytes()).hexdigest() for n in names},indent=2),encoding='utf-8')
    print('report.html and comparison.png written')


if __name__=='__main__':main()
