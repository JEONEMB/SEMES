"""Render the completed frozen-model evaluation; no model fitting or new inference."""
from pathlib import Path
import hashlib
import html
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE=Path(__file__).resolve().parent


def read(name):
    return json.loads((HERE/name).read_text(encoding='utf-8'))


def esc(value):
    return html.escape(str(value))


def num(value,n=6):
    return '—' if value is None else f'{value:.{n}f}'


def main():
    summary=read('summary.json'); rows=read('results.json'); protocol=read('protocol.json')
    tol=summary['tolerance_deg']
    plt.rcParams.update({'font.family':'Malgun Gothic','axes.unicode_minus':False,'font.size':10})
    fig,axes=plt.subplots(1,2,figsize=(14,5),layout='constrained')
    for ax,d in zip(axes,(2,3)):
        selected=[r for r in rows if r['dataset']==d and r['execution_success']]
        groupkey='y_mm' if d==2 else 'repeat'
        groups=sorted({r[groupkey] for r in selected})
        for g in groups:
            part=[r for r in selected if r[groupkey]==g]
            label=f'Y={g:g} mm' if d==2 else f'({g})번'
            ax.scatter([r['theta_deg'] for r in part],[r['theta_error_deg'] for r in part],s=29,label=label,alpha=.85)
        ax.axhspan(-tol,tol,color='#1d8e58',alpha=.12,label='허용 범위 ±0.005°')
        ax.axhline(0,color='#555',lw=.8)
        ax.axhline(tol,color='#1d8e58',ls='--',lw=1)
        ax.axhline(-tol,color='#1d8e58',ls='--',lw=1)
        ax.set(title=f'{d}차 테스트: 추정 θ - 라벨 θ',xlabel='라벨 θ (°)',ylabel='추정 오차 (°)')
        ax.grid(alpha=.18); ax.legend(fontsize=8,loc='best')
    fig.savefig(HERE/'theta_error.png',dpi=170)
    fig.savefig(HERE/'theta_error.svg')
    plt.close(fig)

    statrows=[]
    for label,s in [('2차 (3)번',summary['by_dataset']['2']),('3차 전체',summary['by_dataset']['3']),('전체',summary['overall'])]:
        statrows.append(f'<tr><th>{label}</th><td>{s["images"]}</td><td>{s["execution_failures"]}</td>'
            f'<td>{s["within_tolerance"]} / {s["images"]}</td><td>{num(s["pass_rate_percent_all_images"],1)}%</td>'
            f'<td>{num(s.get("mae_deg"))}</td><td>{num(s.get("rmse_deg"))}</td><td>{num(s.get("max_absolute_error_deg"))}</td></tr>')
    groups=[]
    for r in summary['by_dataset_y']:
        groups.append(f'<tr><td>{r["dataset"]}차</td><td>{r["y_mm"]:g}</td><td>{r["within_tolerance"]} / {r["images"]}</td>'
            f'<td>{num(r.get("mean_error_deg"))}</td><td>{num(r.get("max_absolute_error_deg"))}</td></tr>')
    details=[]
    for r in rows:
        status='failure' if not r['execution_success'] else 'pass' if r['within_tolerance'] else 'outside'
        status_label={'failure':'실행 실패','pass':'통과','outside':'범위 초과'}[status]
        out=r.get('output',{})
        details.append(f'<tr data-dataset="{r["dataset"]}" data-status="{status}"><td>{r["id"]}</td><td>{r["dataset"]}차</td>'
            f'<td>{r["y_mm"]:g}</td><td>{r["theta_deg"]:g}</td><td>{num(out.get("theta_estimate_deg"))}</td>'
            f'<td>{num(out.get("theta_correction_deg"))}</td><td>{num(r.get("theta_error_deg"))}</td>'
            f'<td class="{status}">{status_label}</td><td class="filename" title="{esc(r["path"])}">{esc(r["file"])}</td></tr>')
    overall=summary['overall']
    headline='이번 테스트 전부 허용 범위 이내' if overall['within_tolerance']==overall['images'] else '현재 모델은 모든 테스트 이미지에서 목표 정확도를 충족하지 못함'
    worst=summary['worst_images'][0] if summary['worst_images'] else None
    worsttext=(f'{worst["dataset"]}차 {esc(worst["id"])}: Y={worst["y_mm"]:g}mm, θ 라벨={worst["theta_deg"]:g}° → '
               f'θ 추정={worst["estimate_deg"]:.6f}°, 오차={worst["error_deg"]:+.6f}°.' if worst else '실행 가능한 이미지 없음.')
    text=f'''<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>θ 보정 고정 모델 테스트 · 2026-11-01</title><style>
body{{font-family:Malgun Gothic,Arial,sans-serif;background:#f4f6f9;color:#162234;margin:0;line-height:1.65}}main{{max-width:1250px;margin:auto;padding:30px}}
h1{{font-size:27px}}h2{{margin-top:32px;font-size:21px}}.box{{background:white;border:1px solid #dce2e9;border-radius:10px;padding:20px;margin:18px 0}}
.outcome{{border-left:6px solid #c24c32}}.note{{color:#536174;font-size:14px}}table{{border-collapse:collapse;width:100%;background:white;font-size:14px}}
th,td{{border-bottom:1px solid #e0e5ec;padding:10px;text-align:right;white-space:nowrap}}th:first-child,td:first-child{{text-align:left}}thead{{background:#eaf0f7}}
img{{width:100%;height:auto}}.scroll{{overflow:auto}}.outside,.failure{{color:#ba3d27;font-weight:bold}}.pass{{color:#16804b}}code{{font-family:Consolas,monospace}}
.filename{{text-align:left;max-width:370px;white-space:normal;word-break:break-word;font-size:12px}}select{{padding:8px;margin-right:15px}}a{{color:#1756a2}}.metric{{font-size:24px;font-weight:bold}}
</style></head><body><main><h1>θ 보정 고정 모델 테스트</h1><p>2026-11-01 · 2차 (3)번 20장 + 3차 전체 234장 · 학습 계수 및 C++ 실행 파일 고정</p>
<div class="box outcome"><strong>{headline}</strong><p class="metric">허용 범위 이내 {overall['within_tolerance']} / {overall['images']}장 ({overall['pass_rate_percent_all_images']:.1f}%)</p>
<p>기준: |추정 θ − 라벨 θ| ≤ 0.005°. 전체 분모에 실행 실패도 포함한다.</p><p>실행 실패 {overall['execution_failures']}장. {worsttext}</p></div>
<h2>데이터별 결과</h2><div class="scroll"><table><thead><tr><th>데이터</th><th>장수</th><th>실행 실패</th><th>허용 범위 이내</th><th>비율</th><th>평균 절대오차 (°)</th><th>RMSE (°)</th><th>최대 절대오차 (°)</th></tr></thead><tbody>{''.join(statrows)}</tbody></table></div>
<p class="note">3차가 전체의 대부분이므로 데이터별 결과도 따로 표시했다. 평균·RMSE·최대 오차는 실행 성공 이미지 기준이다.</p>
<h2>오차 분포</h2><div class="box"><img src="theta_error.png" alt="2차와 3차 테스트의 theta 라벨 대비 추정 오차 산점도"><p class="note">녹색 띠가 ±0.005°다. 점 하나가 이미지 한 장이다. 두 그래프의 세로축 눈금은 각각 표시된 값을 따른다.</p></div>
<h2>Y 조건별 결과</h2><table><thead><tr><th>데이터</th><th>라벨 Y (mm)</th><th>허용 범위 이내</th><th>평균 부호 오차 (°)</th><th>최대 절대오차 (°)</th></tr></thead><tbody>{''.join(groups)}</tbody></table>
<p class="note">촬영 조건별 결과 집계다. 각 Y 그룹의 θ 구성이 다르므로 이 표만으로 오차 원인을 Y 영향으로 확정하지 않는다.</p>
<h2>어떻게 테스트했는가</h2><div class="box"><ol>
<li>2차 (1)·(2)번 40장으로 이미 계산한 계수와 C++ 실행 파일을 고정했다.</li>
<li>테스트 254장의 원본 이미지 해시를 확인한 뒤 각 이미지를 C++ 실행 파일에 한 장씩 입력했다. 실행 파일에는 라벨을 전달하지 않았다.</li>
<li>상·하단 세로 Sobel → 일반 Huber → 두 각도를 이용한 θ 계산을 끝까지 실행했다.</li>
<li>추정 θ와 사용자 확인 정정이 반영된 θ 라벨을 비교했다. 반올림 전 값으로 ±0.005° 이내 여부를 판정했다.</li>
<li>코드·계수·데이터 분리 파일이 실행 전후 동일한지 해시로 다시 확인했다.</li></ol>
<p><strong>장비 이동 후의 실제 잔류 오차를 측정한 시험은 아니다.</strong> θ 라벨을 기준으로 이미지 한 장에서 반환한 θ 보정치의 정확도를 평가했다. 명령대로 정확히 이동한다고 가정하면 예상 잔류 오차는 <code>라벨 θ + 반환 보정치 − 목표 θ</code>이며 추정 오차와 부호만 반대다.</p>
<p>학습에 쓰지 않은 고정 내부 테스트다. 이 자료는 이전 분석에서 이미 관찰한 자료이므로 완전히 새로운 독립 데이터 검증으로 표현하지 않는다.</p>
<p>3차 0.32° 파일명 오타 1건과 0.57°의 TXT 값 정정 3건은 기존 사용자 확인 내역을 그대로 적용했다. 원본 파일은 수정하지 않았다.</p>
<p>처리시간 중앙값: {overall.get('cli_elapsed_ms_median',0):.1f}ms/장, 95백분위: {overall.get('cli_elapsed_ms_p95',0):.1f}ms/장. 프로세스 실행·파일 읽기를 포함한 CLI 시간이며, 순수 알고리즘 연산시간과는 다르다.</p></div>
<h2>이미지별 전체 결과</h2><p><label>데이터 <select id="dataset"><option value="all">전체</option><option value="2">2차</option><option value="3">3차</option></select></label>
<label>결과 <select id="status"><option value="all">전체</option><option value="outside">범위 초과</option><option value="pass">통과</option><option value="failure">실행 실패</option></select></label><span id="count"></span></p>
<div class="scroll"><table id="details"><thead><tr><th>ID</th><th>데이터</th><th>Y (mm)</th><th>θ 라벨 (°)</th><th>θ 추정 (°)</th><th>θ 보정치 (°)</th><th>추정−라벨 (°)</th><th>결과</th><th>파일</th></tr></thead><tbody>{''.join(details)}</tbody></table></div>
<h2>재현 자료</h2><p><a href="summary.json">요약 수치</a> · <a href="results.json">254장 원시 출력·오차</a> · <a href="protocol.json">테스트 기준·해시</a> · <a href="frozen_calibration.json">고정 계수</a> · <a href="holdout_manifest.json">테스트 이미지 목록</a> · <a href="evaluate.py">실행 스크립트</a></p>
<p class="note">이 보고서는 고정 모델의 평가 결과다. 결과를 보고 계수·추출 조건을 변경하거나 다시 학습하지 않았다.</p>
</main><script>function filter(){{let n=0;const d=document.getElementById('dataset').value,s=document.getElementById('status').value;document.querySelectorAll('#details tbody tr').forEach(r=>{{const show=(d==='all'||r.dataset.dataset===d)&&(s==='all'||r.dataset.status===s);r.hidden=!show;if(show)n++;}});document.getElementById('count').textContent=n+'장 표시';}}document.getElementById('dataset').onchange=filter;document.getElementById('status').onchange=filter;filter();</script></body></html>'''
    (HERE/'report.html').write_text(text,encoding='utf-8')
    outputs=['summary.json','results.json','protocol.json','holdout_manifest.json','frozen_calibration.json','condition_summary.json','report.html','theta_error.png','theta_error.svg','evaluate.py','make_report.py']
    (HERE/'output_hashes.json').write_text(json.dumps({p:hashlib.sha256((HERE/p).read_bytes()).hexdigest() for p in outputs},indent=2),encoding='utf-8')
    print('report.html and theta_error.png written')


if __name__=='__main__':
    main()
