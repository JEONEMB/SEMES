"""Compile sourced research and proposals. Does not run or modify pose models."""
from pathlib import Path
import csv, hashlib, html, json, zipfile

HERE=Path(__file__).resolve().parent
BASE=HERE.parent
FILES={'audit':'audit_research.json','geometry':'geometry_research.json',
       'photo':'photometric_research.json','root':'root_research.json'}
ORDER=[('audit','A1'),('audit','A2'),('audit','A3')]+[('geometry',f'G{i:02}') for i in range(1,8)]+[('photo',f'P{i:02}') for i in range(1,8)]+[('audit','A4'),('root','covariance_shift'),('audit','A5')]
SHORT=[
 ('새 시행의 0점','새 원점이 이동량과 섞임','알려진 물리 기준의 새 앵커로 절편 보정 후 양축 기울기 확인','추가 기준 측정','먼저'),
 ('시간·온도 드리프트','시간에 따라 편향·배율이 달라짐','예열 안정성 확인, 기준 앵커 보간, 온도·시간 보정','기준 반복 또는 상태 기록','먼저'),
 ('백래시·접근 이력','같은 명령도 실제 위치가 다름','양방향 접근 시험, 접근 방향 통일, 이력 모델·실제 위치 보정','접근 이력 또는 실제 위치','먼저'),
 ('카메라 이동·재장착','카메라와 판의 운동이 섞임','세계 고정 기준으로 카메라 운동 분리 또는 재장착 후 재교정','판 밖 고정 기준 또는 고정 조건','먼저'),
 ('해상도·crop·줌·왜곡','같은 이동의 픽셀 좌표·배율이 바뀜','알려진 resize/crop 좌표 복원, 광학 변경은 내부 파라미터 재교정','사전 카메라·좌표 보정','조건별'),
 ('높이 Z·평면 밖 기울기','확대·원근 변화가 Y/T처럼 보임','알려진 판 기하와 카메라로 평면 자세 추정, Z·tilt를 함께 고려','판 치수·카메라·기구 보정','먼저'),
 ('Y와 T 동시 이동','현재 단일축 모델이 다른 축을 0 처리','Y/T 공동 추정과 혼합항 검증, 관측 가능성 검사','혼합 이동 보정 자료','먼저'),
 ('기존 범위 밖 큰 이동','검색 실패·2차 곡선 외삽','전역 QR 재검출과 다중 스케일 초기화, 범위별 보정·범위 밖 판정','확장 범위 보정 자료','조건별'),
 ('진짜 정지·0 근처','새 정지 사진을 Y/T로 강제 분류','정지 반복 분포로 CENTER/보류 구간 보정, 최소 이동과 분리 평가','독립 정지·최소 이동 반복','먼저'),
 ('판 휨·QR 부착 위치 변화','고정 강체·점 배치 가정이 깨짐','QR별 변형 진단, 기하 재측량, 강체 부분 선택 또는 변형 모델','형상·부착 상태 보정','조건별'),
 ('밝기·노출·감마 변화','패치 밝기 차이가 추적 편향으로 연결','기존 gain/offset 유지, 비포화 강건 광도 적합, 필요시 응답 선형화','선형 변화는 두 영상; 비선형은 교정','먼저'),
 ('국소 그림자·반사·포화','일부 무늬가 바뀌거나 사라짐','포화·반사 마스크, 강건 손실, 확산 조명·편광·노출 조절','복구 불가능 영역은 재촬영','조건별'),
 ('초점 이탈·모션 블러','모서리 위치가 불안정해짐','초점·노출·정착 조건 개선, 흐림 측정과 선택적 복원 비교','심한 흐림은 재촬영','조건별'),
 ('잡음·압축·센서 패턴','모서리와 위치의 작은 변화가 왜곡됨','무손실 저장, dark/flat-field, 약한 잡음 제거와 위치 편향 검증','센서 교정·조건별 반복','조건별'),
 ('가림·오염·QR 누락','유효점과 공간 범위가 부족해짐','유효점 기반 적합·강건 추정, 남은 점의 분포와 조건수 검사','관측 부족 시 재촬영','먼저'),
 ('반복 무늬·표식 교체','다른 모서리를 같은 점으로 대응','QR ID·기하·양방향 검사, RANSAC, 교체 표식의 새 템플릿','새 표식의 기준 대응','먼저'),
 ('롤링셔터·촬영 중 운동','행마다 다른 시각의 자세가 섞임','정착 후 촬영, 동기 트리거·스트로브, 필요시 행별 시간 모델','노출·행 시간·운동 정보','조건별'),
 ('명령 정답·노출 시각 불일치','영상이 맞아도 잘못된 값으로 채점','명령과 실제 위치 분리, 노출 시각에 센서 정답 연결','독립 위치·동기화 정보','먼저'),
 ('잡음 공분산 변화','학습된 가중치가 새 오차에 부적합','QR 블록·저차원 공분산과 강건 손실을 분리 검증','새 조건 반복 교정','조건별'),
 ('새 분포·잘못된 확신','작은 비용을 높은 정확도로 오인','시행·일자 분리 평가, OOD 진단, 별도 신뢰도 보정·고정 시험','독립 검증과 정답 피드백','먼저'),
]

KEYS={
 'audit':dict(change='situation',effect='image_and_model_effect',detect='detection_metrics',
              correction='concrete_correction_proposal',extra='additional_data',limit='two_images_only_limit',
              current='current_model_connection',validation='validation_experiment',evidence='evidence_classification'),
 'geometry':dict(change='situation',effect='effect',detect='detection',correction='correction',
                 extra='additional_calibration_data',limit='two_images_limit',current='current_code_connection',
                 validation='validation_plan',evidence='evidence_status'),
 'photo':dict(change='subcases',effect='impact',detect='detection',correction='correction_steps',
              extra='extra_information',limit='two_image_limit',current='current_connection',
              validation='validation_design',evidence='decision'),
 'root':dict(change='change',effect='failure',detect='detect',correction='correction',extra='extra_data',
             limit='two_image_limit',current='current_status',validation='validation',evidence='evidence_level'),
}
INNER={'status':'実施状態'.replace('実施状態','실행 상태'),'design':'설계','split':'분리','report':'보고 지표',
       'method':'방법','formulas':'식과 계산','cautions':'적용 한계','synthetic':'합성 진단안','real':'실제 촬영안',
       'compare':'비교 후보','metrics':'지표'}
def esc(s):return html.escape(str(s))
def render(value):
    if isinstance(value,list):return '<ul>'+''.join('<li>'+render(x)+'</li>' for x in value)+'</ul>'
    if isinstance(value,dict):return '<dl>'+''.join('<dt>'+esc(INNER.get(k,k))+'</dt><dd>'+render(v)+'</dd>' for k,v in value.items())+'</dl>'
    return esc(value)
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    docs={k:json.loads((HERE/v).read_text(encoding='utf8')) for k,v in FILES.items()}
    sources={f'{k}:{s["id"]}':s for k,d in docs.items() for s in d['sources']}
    normalized=[]
    for number,((group,cid),short) in enumerate(zip(ORDER,SHORT),1):
        case=next(c for c in docs[group]['cases'] if c['id']==cid)
        row=dict(number=number,id=f'{group}:{cid}',title=short[0],problem=short[1],short_correction=short[2],
                 data_requirement=short[3],priority=short[4],source_file=FILES[group],original_title=case['title'])
        for key,field in KEYS[group].items():row[key]=case.get(field,'')
        row['sources']=[sources[f'{group}:{sid}'] for sid in case['source_ids']]
        assert row['sources'] and all(row[k] for k in ['effect','detect','correction','extra','limit','current','validation'])
        normalized.append(row)
    assert len(normalized)==20
    (HERE/'cases_20.json').write_text(json.dumps(normalized,ensure_ascii=False,indent=2),encoding='utf8')
    with (HERE/'cases_20.csv').open('w',encoding='utf-8-sig',newline='') as f:
        fields=['number','title','problem','short_correction','data_requirement','priority']
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        w.writerows({k:r[k] for k in fields} for r in normalized)
    root=docs['root']
    unique_sources={s['url']:s for s in sources.values()}
    (HERE/'sources.json').write_text(json.dumps(list(unique_sources.values()),ensure_ascii=False,indent=2),encoding='utf8')
    pieces=['''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>새 데이터 20가지 문제 상황과 보정 방법 조사</title>
<style>body{margin:0;font-family:'Malgun Gothic',sans-serif;color:#18344a;background:#f5f7f9;line-height:1.75}main{max-width:1160px;margin:auto;padding:38px 28px 70px}h1{font-size:32px;line-height:1.35}h2{font-size:24px;margin-top:42px}h3{font-size:20px;margin-top:32px}p,li,dd{font-size:16px}a{color:#006d87}dt{font-weight:bold;margin-top:14px}dd{margin-left:0}ul{padding-left:24px;margin:7px 0 14px}li ul{margin-top:5px}table{width:100%;border-collapse:collapse;background:white;font-size:14px}th,td{padding:12px 10px;border-bottom:1px solid #d5dfe7;text-align:left;vertical-align:top}th{background:#18344a;color:white}tbody tr:nth-child(even){background:#edf3f6}.scroll{overflow:auto}.meta{color:#506778}.case{border-top:1px solid #c9d7e1;margin-top:34px;padding-top:8px}.refs{font-size:14px}.refs li{font-size:14px}.lead{font-size:20px}.formula{padding:16px;background:#e6eef3;overflow:auto;font-family:Consolas,'Malgun Gothic',monospace}nav{margin:20px 0}nav a{display:inline-block;margin:3px 16px 3px 0}small{color:#506778}code{font-family:Consolas,'Malgun Gothic',monospace}@media(max-width:600px){main{padding:24px 16px}h1{font-size:27px}th,td{padding:9px 8px}}@media print{body{background:white}main{padding:0;max-width:none}a{color:inherit}.scroll{overflow:visible}.case{break-before:auto}h2,h3,dt{break-after:avoid}}
</style><main><h1>새 데이터의 20가지 문제 상황과 보정 방법</h1>
<p class="meta">2026-10-14 · 센터·현재 사진 두 장으로 Y/T 추정 · 문헌 조사와 현재 구현 분석</p>
<p class="lead"><strong>보정 방법은 변화 원인에 따라 달라진다. 새 0점·카메라 이동·구동 이력처럼 두 사진만으로 원인을 분리할 수 없는 경우에는, 더 복잡한 회귀모델보다 기준 측정과 관측 조건을 먼저 갖춰야 한다.</strong></p>
<p>20개 상황마다 어떤 데이터가 달라지는지, 오류가 생기는 이유, 감지 지표, 구체적인 보정 후보, 필요한 추가 데이터, 두 사진의 한계와 검증 실험을 정리했다. 공식 문서와 원논문은 방법의 근거다. 아래 보정안의 SEMES 데이터 성능을 새로 측정한 것은 아니다. 이미 수행한 GLS 실험 결과와 새 제안을 명확히 구분한다.</p>
<nav><a href="#summary">20가지 요약</a><a href="#constraints">현재 구현의 한계</a><a href="#modeling">모델링 방향</a><a href="#details">사례별 상세</a><a href="#validation">수집 및 검증 설계</a><a href="#files">원문과 기록</a></nav>
<h2 id="summary">20가지 상황과 우선 보정 후보</h2>
<p>‘추가 데이터’는 런타임 입력과 사전 보정을 구분해야 한다. 사전에 카메라·기준점을 교정해 저장하면 런타임 두 장을 유지할 수 있는 경우가 있다. 반면 임의의 카메라 이동·미지 원점·이력은 해당 정보나 제약이 없으면 두 장만으로 보장할 수 없다. ‘먼저’는 현재 목표와 코드에 근거한 검증 우선순위이며 발생 확률을 측정한 순위가 아니다.</p>
<div class="scroll"><table><thead><tr><th>번호·상황</th><th>문제</th><th>보정 후보</th><th>추가 기준 또는 조건</th></tr></thead><tbody>''']
    for r in normalized:
        pieces.append(f'<tr><td><a href="#case-{r["number"]}">{r["number"]}. {esc(r["title"])}</a></td><td>{esc(r["problem"])}</td><td>{esc(r["short_correction"])}</td><td>{esc(r["data_requirement"])}</td></tr>')
    pieces.append('''</tbody></table></div><h2 id="constraints">현재 코드에서 확인한 범위</h2><p>현재 6개 절편은 관측 그룹의 패턴이다. 별도로 계측한 6개 실제 0점 또는 백래시 원인이라고 단정하지 않는다. ± 명령값의 부호와 마지막 접근 방향도 다른 변수다.</p><ul>''')
    for fact in root['current_implementation_facts']:
        pieces.append(f'<li>{esc(fact["fact"])} <a href="{esc(fact["file"])}">코드 {esc(fact["file"].split("/")[-1])} {fact["line"]}행</a></li>')
    pieces.append('''</ul><p>원래 목표는 축·부호 100%와 Y/T 각각의 최대 절대오차 0.09 이내다. Y 0.05/T 0.01은 최소 명령 간격이다. T의 물리 단위는 아직 확인되지 않았다. 입력을 거절하는 품질 관리가 필요하더라도, 거절한 사진을 제외하고 전체 100% 달성이라고 보고하면 안 된다.</p>
<h2 id="modeling">다음 모델링 비교의 순서</h2><p><strong>첫 단계는 정답과 기준 좌표의 정의다.</strong> ‘센터 영상에 대한 실제 위치’, ‘각 시행의 명령 이동량’, ‘센터로 되돌릴 구동 명령’은 서로 다른 목표다. 원점·백래시가 있으면 같은 이미지로 이 세 값을 동시에 보장할 수 없다. 영상 측정과 운동 관계, 기준점·촬영 조건을 분리해 모델을 비교한다.</p>
<div class="formula">영상 특징 f ≈ 기준점 b + 운동 응답 F(Y,T) + 관측 가능한 환경 효과 + 잔차</div>
<p>새 기준점 b를 아무 제약 없이 매 사진 자유롭게 추정하면 실제 Y/T를 흡수할 수 있다. 따라서 알려진 기준 위치나 사전에 확인한 제약이 필요하다. 환경 효과를 추가할 때도 Y/T의 영상 변화와 구분 가능한지 확인해야 한다.</p>
<div class="scroll"><table><thead><tr><th>후보</th><th>역할과 비교 방법</th><th>추가 데이터</th><th>현재 상태</th></tr></thead><tbody>''')
    for model in root['recommended_models']:
        pieces.append('<tr>'+''.join('<td>'+esc(v)+'</td>' for v in [model['id']+' '+model['name'],model['purpose'],model['additional_data'],model['status']])+'</tr>')
    pieces.append('''</tbody></table></div><p>GLS는 위 구조에서 오차의 가중 방법이다. 미지의 원점이나 물체·카메라 운동의 식별 문제를 해결하는 물리 정보가 새로 생기는 것은 아니다. 실제 이전 비교에서 공간 GLS의 학습 재대입 오차는 줄었지만, 같은 280장의 20회 학습 제외 평가 실패는 기존 13회에서 66회로 늘었다. <a href="../gls_comparison_20261014/GLS_비교결과.html">기존 GLS 비교 결과</a>.</p>
<h2 id="details">사례별 상세 조사</h2><p>각 사례의 실험은 검증 설계 제안이다. 예시 심각도·촬영 수·임계값 후보를 실제 장비의 검증 완료 사양으로 해석하지 않는다.</p>''')
    headings=[('change','어떤 데이터가 달라지는가'),('effect','어떤 오류가 생기는가'),('current','현재 구현과의 연결'),
              ('detect','어떻게 감지할 것인가'),('correction','어떤 보정을 비교할 것인가'),('extra','추가로 필요한 데이터'),
              ('limit','두 사진만으로 가능한 범위와 한계'),('validation','검증 실험 설계'),('evidence','해석과 적용 조건')]
    for r in normalized:
        pieces.append(f'<section class="case" id="case-{r["number"]}"><h3>{r["number"]}. {esc(r["title"])}</h3><dl>')
        for key,label in headings:
            if r[key]:pieces.append(f'<dt>{label}</dt><dd>{render(r[key])}</dd>')
        pieces.append('</dl><p><strong>근거 자료</strong></p><ul class="refs">')
        for source in r['sources']:
            pieces.append('<li><a href="'+esc(source['url'])+'">'+esc(source['title'])+'</a> — '+esc(source.get('supported_claim',''))+'</li>')
        pieces.append('</ul></section>')
    pieces.append('<h2 id="validation">새 데이터 수집과 검증 설계</h2><p>'+esc(root['validation_plan']['scope'])+'</p><dl>')
    plan_labels={'first_design':'시행 구성','pilot_counts':'탐색용 촬영 수 예시','capture_protocol':'촬영 순서와 기록',
                 'separation':'학습과 시험의 분리','reference_policy':'센터와 보정 자료 허용 범위','synthetic_policy':'합성 자료의 역할',
                 'stress_policy':'요인별 시험','metrics':'필수 지표','acceptance':'원래 목표의 합격 판정',
                 'zero_policy':'정지와 최소 이동','unit_policy':'단위와 정답'}
    for key,label in plan_labels.items():pieces.append('<dt>'+label+'</dt><dd>'+render(root['validation_plan'][key])+'</dd>')
    pieces.append('''</dl><p>우선 M0 기준과 M1 추적 품질 개선을 같은 조건으로 비교하고, 실제 기준 위치 자료를 확보한 조건에서는 M2를 비교한다. 혼합 이동과 기하 변화까지 운용 범위를 넓힐 경우 M3/M4를 추가한다. 한 번에 모든 보정을 켜서 원인을 알 수 없게 만들지 않는다.</p>
<h2 id="files">기록과 출처</h2><p><a href="cases_20.csv">20개 상황 요약 CSV</a> · <a href="cases_20.json">상세 통합 JSON</a> · <a href="sources.json">출처와 지원 주장</a> · <a href="validation_plan.json">검증 계획 JSON</a> · <a href="research_checks.json">완전성·기존 파일 보존 검사</a></p>
<p>보고서 생성 코드와 분야별 원자료를 같은 폴더에 저장했다. 현 장비가 인용한 제조사의 제품이라는 뜻은 아니며, 해당 기능을 사용할 때에는 실제 장비의 지원 여부가 필요하다. 조사 과정에서 원본 이미지·기존 모델·기존 성능표를 수정하지 않았다.</p></main></html>''')
    output=HERE/'새데이터_20가지_보정방법_조사.html'
    output.write_text(''.join(pieces),encoding='utf8')
    (HERE/'validation_plan.json').write_text(json.dumps(root['validation_plan'],ensure_ascii=False,indent=2),encoding='utf8')
    ledger=json.loads((BASE/'gls_comparison_20261014/baseline_snapshot/hashes.json').read_text(encoding='utf8'))
    unchanged=all(sha(BASE/name)==digest for name,digest in ledger.items())
    assert unchanged
    checks=dict(cases=20,all_cases_have_sources=True,all_cases_have_detection_correction_limits_and_validation=True,
                primary_source_records=len(sources),unique_source_urls=len(unique_sources),
                existing_16_baseline_files_unchanged=unchanged,new_experiments_run=False,
                proposed_validation_only=True,research_files={name:sha(HERE/name) for name in FILES.values()})
    (HERE/'research_checks.json').write_text(json.dumps(checks,indent=2),encoding='utf8')
    print(json.dumps(checks,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
