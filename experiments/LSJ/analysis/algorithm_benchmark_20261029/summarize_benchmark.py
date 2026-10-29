"""Summarize the frozen-model offline test. No refitting, rebasing, or pass threshold selection."""
import collections
import html
import json
import math
import statistics
from pathlib import Path

HERE=Path(__file__).resolve().parent
read=lambda name:json.loads((HERE/name).read_text(encoding='utf-8-sig'))
manifest=read('test_manifest.json'); native=read('native_results.json')
inputs={c['id']:c for c in manifest['cases']}
rows=[dict(inputs[r['id']],**r) for r in native['results']]
assert len(rows)==len(inputs)==341
def state(r): return r.get('decision',{}).get('status',r.get('error_code','unknown'))
def valid(r): return bool(r['measurement_valid'])
def decision(r): return r.get('decision',{})
def theta(r): return decision(r)['theta']['estimated_theta_deg'] if valid(r) else None
def ycmd(r):
    d=decision(r)
    return d['command_y_mm'] if d.get('y_evaluated') and d.get('command_available') else None
def percentile(values,p):
    x=sorted(values)
    if not x: return None
    k=(len(x)-1)*p; lo=int(k); hi=min(lo+1,len(x)-1)
    return x[lo]+(x[hi]-x[lo])*(k-lo)
def errors(values):
    if not values: return {'n':0,'mae':None,'rmse':None,'max_abs':None,'bias':None}
    return {'n':len(values),'mae':statistics.mean(map(abs,values)),
        'rmse':math.sqrt(statistics.mean(x*x for x in values)),
        'max_abs':max(map(abs,values)),'bias':statistics.mean(values)}
def timing(rs,key):
    x=[r[key] for r in rs]
    return {'n':len(x),'median':percentile(x,.5),'p95':percentile(x,.95),'max':max(x) if x else None}
def overview(name,rs):
    return {'name':name,'count':len(rs),'measurement_valid':sum(valid(r) for r in rs),
        'nonzero_command':sum(bool(decision(r).get('command_available')) and
            (decision(r).get('command_y_mm',0)!=0 or decision(r).get('command_theta_deg',0)!=0) for r in rs),
        'already_in_image_target':sum(bool(decision(r).get('aligned')) for r in rs),
        'command_unavailable':sum(valid(r) and not decision(r).get('command_available') for r in rs),
        'measurement_failed':sum(not valid(r) for r in rs),
        'y_evaluated':sum(bool(decision(r).get('y_evaluated')) for r in rs),
        'statuses':dict(collections.Counter(state(r) for r in rs)),
        'estimate_ms':timing(rs,'estimate_ms'),'load_and_decode_ms':timing(rs,'load_ms'),
        'total_ms':timing(rs,'total_ms')}

partitions=[('1차 기준 사진',[r for r in rows if r['dataset']=='first' and r['is_reference']]),
    ('1차 Y 단독',[r for r in rows if r['dataset']=='first' and r['axis']=='y']),
    ('1차 θ 단독',[r for r in rows if r['dataset']=='first' and r['axis']=='theta']),
    ('2차 기준 사진',[r for r in rows if r['dataset']=='second' and r['is_reference']]),
    ('2차 Y·θ 혼합',[r for r in rows if r['dataset']=='second' and not r['is_reference']])]
overview_rows=[overview(name,rs) for name,rs in partitions]
groups=collections.defaultdict(list)
for r in rows: groups[(r['dataset'],r['group'])].append(r)
delta_checks=[]
for (dataset,group),rs in groups.items():
    if dataset!='first' or rs[0]['is_reference']: continue
    rs.sort(key=lambda r:r['time_metadata'])
    axis=rs[0]['axis']; key='label_theta_deg' if axis=='theta' else 'label_y_mm'
    get=theta if axis=='theta' else ycmd
    pairs=[]
    for a,b in zip(rs,rs[1:]):
        pa,pb=get(a),get(b)
        if pa is None or pb is None: continue
        label_delta=b[key]-a[key]
        predicted_delta=pb-pa
        # Correction has opposite sign to pose. Y is a tick command to a RANGE, not a point estimate.
        error=predicted_delta-label_delta if axis=='theta' else predicted_delta+label_delta
        pairs.append({'from':a['id'],'to':b['id'],'label_delta':label_delta,
            'predicted_delta':predicted_delta,'error':error})
    delta_checks.append({'dataset':dataset,'group':group,'axis':axis,
        'metric':'adjacent theta estimate difference minus nominal label difference' if axis=='theta' else
                 'adjacent issued Y correction difference plus nominal label difference (conditional, quantized, target range)',
        'unit':'degree' if axis=='theta' else 'mm','eligible_pairs':len(rs)-1,
        'stats':errors([p['error'] for p in pairs]),'pairs':pairs})

# Second data: fixed Y, adjacent theta conditions. Require all 3 estimates per condition.
second_cond=[]
repeats=[]
for (dataset,group),rs in groups.items():
    if dataset!='second': continue
    estimates=[theta(r) for r in rs]; ys=[ycmd(r) for r in rs]
    complete=all(v is not None for v in estimates) and len(estimates)==3
    complete_y=all(v is not None for v in ys) and len(ys)==3
    repeats.append({'condition':group,'ids':[r['id'] for r in rs],'count':len(rs),
        'valid_theta_count':sum(v is not None for v in estimates),
        'theta_estimates_deg':estimates,'theta_range_deg':max(estimates)-min(estimates) if complete else None,
        'y_command_range_mm':max(ys)-min(ys) if complete_y else None,
        'statuses':dict(collections.Counter(state(r) for r in rs))})
    if not rs[0]['is_reference']:
        second_cond.append({'condition':group,'y':rs[0]['label_y_mm'],'t':rs[0]['label_theta_deg'],
            'mean_estimated_theta':statistics.mean(estimates) if complete else None,
            'mean_command_y':statistics.mean(ys) if complete_y else None})
for axis in ['theta','y']:
    fixed_key='y' if axis=='theta' else 't'; vary='t' if axis=='theta' else 'y'
    predicted_key='mean_estimated_theta' if axis=='theta' else 'mean_command_y'
    for fixed in sorted({c[fixed_key] for c in second_cond}):
        cs=sorted([c for c in second_cond if c[fixed_key]==fixed],key=lambda c:c[vary])
        pairs=[]
        for a,b in zip(cs,cs[1:]):
            if a[predicted_key] is None or b[predicted_key] is None: continue
            ld=b[vary]-a[vary]; pd=b[predicted_key]-a[predicted_key]
            pairs.append({'from':a['condition'],'to':b['condition'],'label_delta':ld,
                'from_label':a[vary],'to_label':b[vary],
                'within_nominal_theta_1_degree':a['t']<=1 and b['t']<=1,
                'predicted_delta':pd,'error':pd-ld if axis=='theta' else pd+ld})
        delta_checks.append({'dataset':'second','group':f'{fixed_key}={fixed}','axis':axis,
            'metric':'condition means; all three outputs required; adjacent nominal change comparison',
            'unit':'degree' if axis=='theta' else 'mm','eligible_pairs':max(0,len(cs)-1),
            'stats':errors([p['error'] for p in pairs]),
            'nominal_theta_up_to_1_degree_stats':errors([p['error'] for p in pairs if p['within_nominal_theta_1_degree']]),
            'pairs':pairs})

summary={'scope':'341 real images, frozen model, one independent inference each; not closed-loop robot accuracy',
    'overview':overview_rows,'all_cases':overview('All',rows),
    'non_reference_cases':overview('Reference photos excluded',[r for r in rows if not r['is_reference']]),
    'per_capture_or_condition':[overview(f'{d}:{g}',rs) for (d,g),rs in groups.items()],
    'second_large_theta':overview('2차 nominal theta=2 degrees',[r for r in rows if r['dataset']=='second' and r['label_theta_deg']==2]),
    'delta_reproduction':delta_checks,'second_repeatability':repeats,
    'timing':{'initialization_ms':native['initialization_ms'],'warmup_images':1,'repeats_per_image':1,
        'opencv_version':native['opencv_version'],'threads':native['opencv_threads'],
        'processor':manifest['processor'],'note':'Local warm-run latency; not a guaranteed production latency or hardware control-cycle time'},
    'limits':['First data already used for calibration; replay is not independent test accuracy',
        'Second data is a separate acquisition but already explored; not a blind holdout',
        'Nominal labels have a different reference from the current target, so absolute correction error is not scored',
        'Adjacent pairs share images and are not independent trials; descriptive stats only',
        'Y error metrics are conditional on passing theta and obtaining a command; report coverage',
        'No movement-followed-by-image data; no empirical convergence or 20-move success rate',
        'Missing or failed predictions remain in denominators; no pass threshold chosen after observing results']}
(HERE/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')

def fmt(v,n=4): return '—' if v is None else f'{v:.{n}f}'
lines=['# 기존 촬영본 알고리즘 성능 점검 — 2026-10-29','',
    '현재 수정본과 설정을 고정하고 1차 281장·2차 60장 전체를 한 번씩 입력했습니다. 이미지 목록은 `test_manifest.json`, 원시 출력은 `native_results.json`, 상세 집계는 `summary.json`입니다. 원본은 복사·수정하지 않았습니다.','',
    '**현재 순차 알고리즘은 θ 단계에서 대부분 정지했습니다.** 341장 중 비영점 명령은 29장, θ 명령 불가는 311장, 기준 사진 자기 비교의 목표 내 판정은 1장입니다. 기준 사진 외에는 Y 단계에 도달한 사진이 없어 Y 보정 정확도는 이번 실행으로 평가하지 못했습니다. 기존 코드 회귀 시험 통과와 실제 촬영본에서의 보정 가능성은 다른 결과입니다.','',
    '**명령 산출은 실제 정렬 성공이 아닙니다.** 정렬 완료 열은 이동 전 현재 이미지가 코드의 목표 조건에 들어왔다는 뜻입니다. 도달 불가도 실패 건수에 포함합니다.','',
    '| 자료 | 장수 | 비영점 명령 산출 | 현재 목표 내 | 측정 후 명령 불가 | 측정 실패 | Y 평가 도달 |',
    '|---|---:|---:|---:|---:|---:|---:|']
for r in overview_rows:
    lines.append(f"| {r['name']} | {r['count']} | {r['nonzero_command']} | {r['already_in_image_target']} | {r['command_unavailable']} | {r['measurement_failed']} | {r['y_evaluated']} |")
lines+=['','## 반환 상태','']
for r in overview_rows: lines.append(f"- {r['name']}: "+', '.join(f'{k}={v}' for k,v in r['statuses'].items()))
example=next(r for r in rows if r['dataset']=='first' and r['file'].replace('\\','/')=='Theta/T+0.10.bmp')
angles=decision(example)['theta']['lines']
lines+=['',f"예: `Theta/T+0.10.bmp`의 두 선을 각각 로봇 θ로 환산하면 {angles[0]['estimated_theta_deg']:.6f}°와 {angles[1]['estimated_theta_deg']:.6f}°입니다. 현재 모델은 이 두 선을 함께 ±0.005° 조건으로 보정할 수 없다고 판정했습니다. 이는 현재 특징·환산·판정 조합의 결과이며 실제 기계 오차의 원인을 확정한 것이 아닙니다.",
    '', '전체 311건 중 `theta_target_conflict`는 273건, `theta_no_reachable_tick`은 38건입니다. 실행 예외는 없었지만, 예외가 없다는 사실이 영상 특징의 정확한 검출을 보증하지는 않습니다.']
lines+=['','## 처리시간','',
    '기준 이미지 초기화와 1회 예열을 제외한 이미지별 1회 실행입니다. 모델 추정 시간은 이미지 읽기·디코딩을 제외합니다. 실패한 호출 시간도 전체 집계에 포함합니다.','',
    '| 자료 | 추정 중앙값(ms) | 추정 P95(ms) | 읽기 포함 중앙값(ms) | 읽기 포함 P95(ms) |','|---|---:|---:|---:|---:|']
for r in overview_rows:
    lines.append(f"| {r['name']} | {fmt(r['estimate_ms']['median'],2)} | {fmt(r['estimate_ms']['p95'],2)} | {fmt(r['total_ms']['median'],2)} | {fmt(r['total_ms']['p95'],2)} |")
lines+=['','## 명목 변화량 재현','',
    'θ: 같은 촬영 구간 또는 같은 Y 조건에서 `추정 θ 차이 − 라벨 θ 차이`를 계산합니다. 2차는 각 조건의 3장 모두 추정되었을 때만 평균 간 차이를 계산합니다. 기준점 오프셋을 새로 맞추거나 회귀를 다시 하지 않았습니다.','',
    'Y: 실제 순차 알고리즘이 Y까지 통과해 명령을 낸 경우에만 `Y 명령 차이 + Y 라벨 차이`를 계산합니다. 범위 목표와 0.05mm 양자화를 포함한 **명령 변화량**의 비교입니다. Y 연속 위치 추정 정확도와 다릅니다. 계산 불가인 경우 0점 처리하지 않습니다.','',
    '| 자료/구간 | 축 | 사용/가능 쌍 | MAE | RMS | 최대 절댓값 | 단위 |','|---|---|---:|---:|---:|---:|---|']
for r in delta_checks:
    s=r['stats']; lines.append(f"| {r['dataset']} / {r['group']} | {r['axis']} | {s['n']}/{r['eligible_pairs']} | {fmt(s['mae'],6)} | {fmt(s['rmse'],6)} | {fmt(s['max_abs'],6)} | {r['unit']} |")
lines+=['','위 2차 θ 표는 1→2° 구간도 포함합니다. 기존 기울기 자료의 범위와 분리해서 볼 수 있도록 **두 조건의 θ 라벨이 모두 1° 이하인 쌍**을 별도로 집계합니다.','',
    '| 2차 고정 Y 조건 | 쌍 수 | 변화량 MAE(°) | RMS(°) | 최대 절댓값(°) |','|---|---:|---:|---:|---:|']
for r in delta_checks:
    if r['dataset']=='second' and r['axis']=='theta':
        s=r['nominal_theta_up_to_1_degree_stats']
        lines.append(f"| {r['group']} | {s['n']} | {fmt(s['mae'],6)} | {fmt(s['rmse'],6)} | {fmt(s['max_abs'],6)} |")
lines+=['','인접 쌍은 이미지를 공유하므로 독립 반복 실험이 아닙니다. 위 숫자는 기술 통계이며, 전체 이미지의 보정 정확도나 허용오차 합격률로 해석하지 않습니다.','',
    '## 동일 조건 3장 반복 출력','',
    '| 2차 조건 | θ 추정 가능 | θ 추정 최댓값−최솟값(°) | Y 명령 최댓값−최솟값(mm) |','|---|---:|---:|---:|']
for r in repeats: lines.append(f"| {r['condition']} | {r['valid_theta_count']}/3 | {fmt(r['theta_range_deg'],6)} | {fmt(r['y_command_range_mm'],4)} |")
lines+=['','## 평가 경계','',
    '- 1차는 기울기·목표 계산에 이미 사용한 자료이므로 재현 검사입니다. 2차도 이전 탐색에 사용한 적이 있어 완전한 블라인드 테스트는 아닙니다.',
    '- 2차 기준은 현재 알고리즘의 1차 기준과 다릅니다. 라벨의 음수를 절대 정답 보정량으로 정하지 않았습니다.',
    '- θ=2°인 2차 사진 12장은 기존 기울기 자료의 ±1° 범위 밖 조건으로 별도 집계했습니다. 전체 실패율에서 제외하지 않았습니다.',
    '- θ 단계에서 정지하면 Y를 강제로 실행하지 않았습니다. 따라서 Y 지표의 표본 수와 단계 도달률을 함께 봐야 합니다.',
    '- 저장된 사진을 제어 명령에 따라 이동한 뒤의 사진으로 대체하거나, 모델 예측으로 실제 성공을 판정하지 않았습니다. 실장비 정확도·20회 내 수렴률은 이 시험으로 측정하지 못합니다.',
    '- 상세 실행 환경·소스 해시·입력 해시는 manifest와 integrity 파일에 있습니다. 알고리즘·기준 범위·허용오차는 변경하지 않았습니다.','',
    '## 재현','',
    '`python run_benchmark.py` 후 `python summarize_benchmark.py`를 실행합니다. MSVC v141와 OpenCV 3.4.15가 필요하며 로컬 경로는 실행 스크립트에 있습니다.']
(HERE/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

# Standalone per-image inspection. No external assets or services.
esc=lambda x:html.escape(str(x))
body=[]
for r in rows:
    d=decision(r); t=d.get('theta',{}); lc=t.get('lines',[])
    body.append('<tr>'+''.join('<td>'+esc(x)+'</td>' for x in [r['id'],r['file'],r['label_y_mm'],r['label_theta_deg'],state(r),
        fmt(theta(r),6),fmt(d.get('command_theta_deg'),4),fmt(d.get('command_y_mm'),4),
        d.get('command_available','—'),d.get('y_evaluated','—'),
        fmt(lc[0]['estimated_theta_deg'],6) if lc else '—',fmt(lc[1]['estimated_theta_deg'],6) if lc else '—',
        fmt(r['estimate_ms'],2),r.get('error','')])+'</tr>')
headers=['ID','파일','Y 라벨(mm)','θ 라벨(°)','상태','추정 θ(°)','θ 명령(°)','Y 명령(mm)','명령 가능','Y 평가','왼쪽 θ(°)','오른쪽 θ(°)','추정 ms','측정 오류']
document='''<!doctype html><html lang="ko"><meta charset="utf-8"><title>촬영본 알고리즘 평가</title>
<style>body{font-family:system-ui,sans-serif;margin:24px;color:#17202a}table{border-collapse:collapse;font-size:13px;width:100%}th,td{padding:8px;border:1px solid #ddd;text-align:left}th{background:#eef2f7;position:sticky;top:0}tr:nth-child(even){background:#f8fafc}input{padding:10px;width:420px;max-width:90%}pre{white-space:pre-wrap;background:#f5f7fa;padding:16px}</style>
<h1>1·2차 촬영본 341장 — 고정 알고리즘 실행 결과</h1>
<p>명령 가능은 실제 정렬 성공이 아닙니다. 정지·실패 행의 명령 0은 실행 지시가 아닙니다. Y가 평가되지 않은 행도 모두 표시합니다.</p>
<p><a href="report.md">해석 및 수치 보고서</a> · <a href="summary.json">상세 집계</a> · <a href="test_manifest.json">테스트 목록</a></p>
<details><summary>집계표 보기</summary><pre>'''+esc('\n'.join(lines[:25]))+'''</pre></details>
<p><input id="q" placeholder="파일명·상태·ID 검색" aria-label="결과 검색"> <span id="n">341건</span></p><table><thead><tr>'''+''.join('<th>'+esc(h)+'</th>' for h in headers)+'''</tr></thead><tbody>'''+''.join(body)+'''</tbody></table>
<script>const rows=[...document.querySelectorAll('tbody tr')];document.getElementById('q').addEventListener('input',e=>{let n=0;const q=e.target.value.toLowerCase();for(const r of rows){r.hidden=!r.textContent.toLowerCase().includes(q);if(!r.hidden)n++;}document.getElementById('n').textContent=n+'건';});</script></html>'''
(HERE/'results.html').write_text(document,encoding='utf-8')
print(json.dumps({'overview':overview_rows,'all':summary['all_cases'],'delta_metrics':[{k:v for k,v in r.items() if k!='pairs'} for r in delta_checks]},ensure_ascii=False,indent=2))
