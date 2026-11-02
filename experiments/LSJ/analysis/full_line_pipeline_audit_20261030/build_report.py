"""Render an audit of existing artifacts; no detector, fit or statistical test runs."""
from pathlib import Path
from collections import Counter
from html import escape
import json
import os
import re
from urllib.parse import quote

from audit_content import ALL, FITTED, SEG, FINDINGS, METHODS, SOURCES

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
SRC = ROOT / 'analysis/reference_line_methods_108_20261030'
SEL = ROOT / 'analysis/reference_line_methods_selection_20261030'
H = escape

def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))

def write(name, value):
    (HERE/name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')

def rel(p):
    return quote(os.path.relpath(p, HERE).replace('\\', '/'), safe='/#:')

def link(href, label):
    return f'<a href="{H(href, quote=True)}">{H(str(label))}</a>'

def table(headers, rows):
    return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+H(x)+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+str(x)+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table></div>'

def refs(ids):
    return ' · '.join(link(SOURCES[s][1], s+' '+SOURCES[s][0]) for s in ids)

def f_links(ids):
    return ' '.join(link('#'+s,s) for s in ids)

def evidence_location(item, views):
    name, needle = item
    p = (SEL/name.split('/',1)[1]) if name.startswith('selection/') else SRC/name
    lines = p.read_text(encoding='utf-8-sig').splitlines()
    hits = [i for i,s in enumerate(lines,1) if needle in s]
    if not hits:
        raise ValueError(f'Missing evidence: {name}: {needle}')
    key = p.relative_to(ROOT).as_posix()
    return dict(path=key, line=hits[0], other_matches=hits[1:], needle=needle,
                view=quote(views[key]['view'],safe='/')+'#L'+str(hits[0]), sha256=views[key]['sha256'])

def parameter_status(mid, key):
    n=mid[1:]; ext=mid.startswith('E')
    if 'examples_every' in key or key=='max_candidates_saved':
        return '저장용', '예시·후보 보관량이며 정확도의 합격 기준이 아님.'
    if n=='01':
        if key=='gradient': return '설명값', '현재 Sobel-y 구현에 고정. 문자열 변경으로 계산이 바뀌지 않음.'
        if key=='init_sigma': return '원문 기반·설명값', '초기 σ=이웃 크기는 원문 근거. 현재 코드에 고정된 초기화이며 설정 문자열로 분기하지 않음.'
        return '자체 설정·미검증', 'Gaussian 원리와 별개로 창·임계·종료값은 현재 영상에서 적정성이 검증되지 않음.'
    if n=='02': return '자체 적용값·미검증', '저자 검출기는 그대로이나 평활 폭과 두 임계의 현재 값은 별도 선택임.'
    if n=='03': return '자체 적용값·미검증', '표준 API의 파라미터. 검색 격자·득표·거리 등의 선택이 정확도를 입증하지 않음.'
    if n=='04':
        if key=='scale': return '요구사항에 따른 변경', '0.8→1.0. 원본 해상도 요구를 따른 선택이며 기본 Gaussian subsampling은 생략됨.'
        if key=='sigma_scale': return '현재 비활성', 'scale=1인 현재 분기에서는 적용되지 않음.'
        return '저자 기본값', '출처는 확인됨. 이 데이터에서의 적정성까지 검증된 것은 아님.'
    if n=='05':
        if key=='k_edge': return '원문 근사 상수', 'Gaussian 흐림과 ramp 폭의 근사 대응 1.8. 원문 조건에 의존함.'
        if key=='blur_sigma_px': return '물리량 가정·미검증', '광학 흐림 측정값이 아님. coarse_blur_sigma와 다른 양임.'
        if key=='Np': return '원문 허용 형태·선택값', '7×7 적분 마스크는 원문과 양립. 현재 영상의 최적 창이라는 검증은 없음.'
        return '자체 설정·미검증', '수치적분·거친 검출·채택 조건의 적정성 비교가 없음.'
    if n=='06': return '변형 설정·미검증', '3×9 기본형과 경사·밝기차·중앙 이동 제한. 기본 공식의 근거와 창/임계의 검증은 구분.'
    if n=='07':
        if ext and key in ('ed_grad_thresh','ed_anchor_thresh'):
            return '저자 기본값 변경', '약한 구조물 경계 때문에 36/8에서 12/2로 낮춤. 이유는 있으나 비교 검증 없음.'
        return '저자 기본값 대응', '고정 소스의 기본 동작에 대응. 이 영상에서 최선이라는 근거는 아님.'
    if n=='08':
        if ext and key in ('canny_th1','canny_th2'):
            return '저자 기본값 변경', '50/50에서 10/20으로 낮춤. 약한 경계 대응 의도이며 적정값은 미검증.'
        return '공식 기본값 대응', 'do_merge=false에서만 저장 지지점과 선분의 순서 대응을 지원. 거리값은 기본 √2의 소수 근사.'
    if n=='09':
        if ext and key in ('gradientThreshold','anchorThreshold'):
            return '저자 기본값 변경', '30/8에서 10/4로 변경. 약한 경계 대응 의도이며 적정값은 미검증.'
        return '저자 기본값 대응', '유효 설정은 소스 기본값까지 포함해야 함. listJunctionSizes는 별도 기록 필요.'
    if n=='10':
        if key in ('threshold','contour_mode'): return '설명값', '현재 Otsu/RETR_EXTERNAL 호출에 고정. 문자열 변경이 실행에 반영되지 않음.'
        return '자체 설정·미검증', '2% 다각형 근사와 양끝 8% 제외는 프로젝트 선택.'
    if n=='11': return '자체 적용값·미검증', 'Shi–Tomasi/cornerSubPix의 존재는 현재 코너 선택과 수렴값의 검증이 아님.'
    return '기존 구현 이식·미검증', '이전 Python에서 가져온 선택값. 이력은 있지만 정확도 검증을 대신하지 못함.'

def main():
    views={x['source']:x for x in read(HERE/'source_views.json')}
    cfg=read(SRC/'configs/methods.json')
    targets=read(SRC/'configs/targets.json')
    manifest=read(SRC/'configs/inputs_manifest.json')
    checks=read(HERE/'artifact_checks.json')
    vendor=read(HERE/'third_party_comparison.json')
    records=read(SRC/'results/line_measurements.json')['records']
    enriched=[]
    for f in FINDINGS:
        f=dict(f)
        f['code_evidence']=[evidence_location(e,views) for e in f['evidence']]
        enriched.append(f)
    write('findings.json',enriched)

    params=[]
    for mid in ALL:
        n=int(mid[1:]); group=f'E{n:02}/Q{n:02}' if n<=9 else mid
        kind='external' if mid.startswith('E') else 'qr'
        for key,value in cfg['methods'][group][kind].items():
            status,note=parameter_status(mid,key)
            params.append(dict(method=mid,group=group,kind=kind,key=key,value=value,status=status,note=note))
    write('method_parameter_ledger.json',params)

    categories=Counter(f['category'] for f in enriched)
    bymethod={m:[r for r in records if r['method_id']==m] for m in ALL}
    texts=[]; body=[]
    def add(title,content,plain='',id=None):
        body.append(f'<section id="{H(id or title)}"><h2>{H(title)}</h2>{content}</section>')
        texts.extend(['',title,'='*len(title),plain])

    lead='핵심 문제는 논문에 근거한 검출기 뒤에 붙인 선택·제외·재피팅 규칙이 충분히 검증되지 않았는데, 최종 수치를 방법 자체의 성능처럼 비교하기 쉬운 구조라는 점입니다.'
    intro='<p class="lead">'+H(lead)+'</p><p>21개 방법과 공통 처리에서 <b>32개 검토 항목</b>을 정리했습니다. 32개 모두 계산 오류라는 뜻은 아닙니다. 아래에서 <b>확인된 논리 문제·설명 문제</b>, <b>특정 입력에서 발생할 위험</b>, <b>추가 근거가 필요한 설정·변형</b>을 구분합니다.</p>'
    intro+=table(['분류','항목 수','의미'],[[H(k),v,H({'코드 논리 문제':'구현의 비교 조건이 목적과 맞지 않음.','기록·설명 문제':'실제 계산과 설명 또는 실행 기록이 맞지 않음.','조건부 코드 위험':'현재 결과의 오류가 관측된 것과 달리, 방어 조건이 부족함.','레퍼런스 대비 변형':'변형이 존재함. 그 자체가 오류는 아니며 영향 검증이 필요함.','설정 근거·검증 부족':'출처 또는 선택 의도와 실제 적정성 검증을 구분해야 함.'}[k])] for k,v in categories.items()])
    intro+='<div class="notice"><b>이번에 한 일:</b> 기존 코드·설정·저장 결과·일차 자료를 읽고 대조했습니다.<br><b>하지 않은 일:</b> 원본 수정, 검출/피팅 재실행, 임계값 변경, 통계 검정, 후보 재선정, Git/Jira 작업.<br>사진에서 실제 경계의 정답을 새로 측정하거나 θ 보정 정확도를 판정한 보고서가 아닙니다.</div>'
    add('1. 점검 결론',intro,lead+'\n'+json.dumps(categories,ensure_ascii=False)+'\n원본 수정·재측정·통계 검정·후보 변경 없음.','summary')

    quick=[('F01·F02·F03','공통 Huber/제외/수렴값','1.345라는 숫자의 출처는 있으나 1.345px, 2/1px 제외와 현재 종료값의 적정성은 검증되지 않았음.','점 추출부터 나온 방법 차이와 공통 후처리 효과를 구분해야 함.'),
           ('F28','누락 기록도 동일하다고 판단할 수 있음','only_in_a/b가 있어도 동일 여부는 공통 ID 값만으로 판단함.','비교 프로그램의 조건을 고쳐야 함. 기존 216개 전체 반복 기록은 ID도 같아 그 결론은 유지.'),
           ('F24·F27','설명·설정의 적용 범위 불일치','Q10의 일정한 0.5px 오프셋 설명은 근거 부족. Q10~Q12는 공통 band 제한을 사용하지 않음.','설명 수정 및 실제 유효 설정 분리 필요.'),
           ('F04·F11','피팅 사용점의 의미','최종 마스크가 마지막 실제 피팅 입력과 다를 수 있음. 선분법의 최종 점은 검출 화소가 아닌 인공 표본.','저장·그림의 점과 잔차가 무엇을 뜻하는지 바로잡아야 함.'),
           ('F12·F16·F17·F19','방법별 변형·가정','Gaussian 부분 참조, LSD scale=1, Zernike 흐림 폭 가정, PAE 3×9 변형이 있음.','원문 또는 저자 기본값과 같은 성능이라고 볼 수 없음. 개별 비교가 필요.'),
           ('F31','선별 결과의 해석','기준 4장의 최대−최소 각도 폭으로 선별했으나 정답 경계 각도는 없음.','현재 후보는 임시 선택 기록이며 정확도 합격 목록은 아님.')]
    add('2. 먼저 볼 항목',table(['항목','내용','확인 결과','의미'],[[f_links(x.split('·')),H(t),H(d),H(e)] for x,t,d,e in quick]),'\n'.join(' | '.join(x) for x in quick),'first')

    scope='<p>외부 <b>E01~E09</b>는 상·하단 각각, QR <b>Q01~Q12</b>는 좌·중·우 각각을 점검했습니다. 4장 × (9×2 + 12×3) = <b>216개 저장 측정 기록</b>입니다. 108개 조합은 이미 계산된 선을 묶은 경우의 수로, 서로 독립된 108개 검출기가 아닙니다.</p>'
    scope+=table(['점검 대상','확인 범위','한계'],[
        ['자체 C++/설정/실행기','점 추출 → 후보 선택 → 직선 → 결과 저장 → 선별 로직','새 실행으로 수치적 정확성을 증명하지 않음.'],
        ['저자 코드','보관 공식 원본과 실제 소스의 차이 및 주요 호출·변경 경로','수천 줄의 모든 저자 코드에 대한 독립 재증명은 아님.'],
        ['원자료','4장 해시 일치, 크기·좌표·대상 정의, 기존 확대 그림','216개 모든 경계를 사람이 새로 정답 라벨링하지 않음.'],
        ['레퍼런스','원논문/공식 문서/저자 소스 중 실제 접근한 자료','PAE 원논문 전문, Devernay 논문 전문은 미확인. 출처별 제한은 부록에 명시.'],
        ['산출물','216개 기록, 432개 조합 기록, 선별 48개 복사 여부','기록의 일관성과 측정 정확도는 다름.']])
    scope+=table(['기준','원본 경로','해시 확인'],[[H(r['reference_id']),'<code>'+H(r['path'])+'</code>','manifest 일치'] for r in manifest['references']])
    add('3. 범위와 확인한 자료',scope,'외부 9개 + QR 12개. 기준 4장, 저장 측정 216개, 조합 432개.\n'+'\n'.join(r['reference_id']+' '+r['path'] for r in manifest['references']),'scope')

    retained=[
        'LSD와 Devernay의 실제 c/h는 보관된 공식 ZIP 원본과 같습니다. 모든 검출기 코어를 임의로 새로 만든 것은 아닙니다.',
        'EDLines·ELSED·FLD는 보관 원본과의 실제 diff에서 접근자/지지점 출력·이식 변경을 구분했습니다. 주 검출 계산식을 바꾼 정황은 해당 diff에서 발견하지 못했습니다.',
        '외부 구조물은 좌우 가시 영역의 점을 모아 상단 한 직선·하단 한 직선을 만듭니다. 좌우의 두 기울기를 평균하는 구현이 아닙니다.',
        'QR은 세 영역 모두 처리합니다. Q01~Q11은 밝은 바깥 테두리 상단을 목표로 하고, Q12는 파인더 중심 연결선으로 별도 분류합니다.',
        'Huber는 E01~E09 및 Q01~Q10에 적용됩니다. Q11·Q12는 두 점 연결이며 Huber를 적용하지 않습니다.',
        '원본 픽셀 좌표 복원·법선각/기울기 변환의 검토 경로에서 명백한 대수 오류는 찾지 못했습니다. 이는 영상 각도 φ와 로봇 θ가 같다는 뜻이 아닙니다.',
        '좌표는 x 오른쪽, y 아래, m=dy/dx, φ=atan2(dy,dx)입니다. 직선의 기울기 m은 px/px이며 φ는 영상의 각도입니다.'
    ]
    good='<ul>'+''.join('<li>'+H(s)+'</li>' for s in retained)+'</ul><p>'+refs(['S02','S04','S07','S08','S09'])+'</p>'
    add('4. 유지할 수 있는 부분',good,'\n'.join(retained),'retained')

    common_rows=[
        ('대상 ROI·가림','targets.json 전체값','R01·사용자 대상 지정·이전 정의','선정 이유는 있음. 같은 물리 경계인지·위치 변경 민감도는 미검증.','F07·F08·F27'),
        ('Huber C','1.345px','OpenCV 기본 상수와 같은 숫자','픽셀 잔차의 scale 근거 없음. 95% 효율을 정확도 95%로 해석 금지.','F01·F06'),
        ('IRLS 종료','최대 100회; 각도 <1e-13rad; 중심 x/y <1e-9px','자체 구현','수렴/한도 종료 미구분. 횟수·최종 변화량도 미저장.','F03'),
        ('hard trimming','외부 2px / QR 1px, 재피팅 2회','경계 혼동 제거용 초기값','점 제거의 타당성과 결과 영향 미검증.','F02·F04'),
        ('최소 점 수','외부 50 / QR 20','자체 초기값','인공 표본 수도 포함. 마지막 마스크 분류 후 재검사 없음.','F02·F04·F11'),
        ('검출 여백','외부 16px / QR 8px','주변 문맥 확보','모든 방법이 동일 적용하지 않음. 점 방식 일부는 별도 여백; Q10~Q12는 ROI 직접 사용.','F07·F27'),
        ('열 간격','외부 2px / QR 1px','이전 측정 간격·QR 짧은 길이','선택 의도는 있으나 간격 민감도 미검증. 모든 방법에 적용되는 공통 규칙은 아님.','F07·F27'),
        ('선분 후보','각도 3°/5°; 길이 20/10px; 양쪽 offset 3/2px; 밝기차 3/20','자체 consensus 전처리','저자 검출기와 별도 선택 모델. midpoint만 band 검사.','F08·F09'),
        ('선분 consensus','양 끝 거리 2/1px; 긴 선분 최대 300개에서 가설 생성','자체 설계','길이 합·동점·중복 지지 영향 미검증.','F09'),
        ('선분 최종 점','x 방향 1px 간격','선분을 가중 통합하려는 선택','호 길이 가중과 정확히 같지 않음. 원 검출 화소와 다른 점.','F10·F11'),
        ('최종 마스크','마지막 선에 대해 원 점 전체 재분류','진단·저장용','실제 마지막 fit 입력 마스크와 구분 필요.','F04'),
        ('계산값·표시값','fit 원정밀 / 중간 좌표 4~5자리 / 화면 점 2자리','저장 용량·표시','표시 좌표로 원 결과의 정확한 재피팅은 불가.','F30')]
    common_html=table(['단계','현재 값','출처/의도','판정','근거 항목'],[[H(a),H(b),H(c),H(d),f_links(e.split('·'))] for a,b,c,d,e in common_rows])
    common_html+='<div class="formula">점의 직교잔차 r → w = 1 (|r| ≤ 1.345px), 그 밖은 w = 1.345px / |r| → 가중 직교 직선 → 2px/1px 밖의 점 제거 → 재피팅</div><p>좌표에서 중심을 빼는 계산은 수치 안정화를 위한 이동입니다. 잔차를 표준편차 등의 scale로 나눈 <b>통계적 표준화</b>가 아닙니다. 다만 고정 px Huber 자체가 수학적으로 잘못된 것은 아닙니다. 적정 scale을 정하지 않은 상태에서 다른 입력 유형 전체에 같은 값을 적용한 타당성이 부족합니다. '+refs(['S10','S11'])+'</p>'
    common_html+='<details><summary>현재 대상 좌표 전체 보기</summary><pre>'+H(json.dumps(targets,ensure_ascii=False,indent=2))+'</pre></details>'
    add('5. 공통 처리와 고정값',common_html,'\n'.join(' | '.join(r) for r in common_rows),'common')

    method_html='<p>각 방법의 <b>검출 코어</b>와 <b>자체 선택·후처리</b>를 분리해 읽어야 합니다. 아래 성공 수는 기존 프로그램이 반환한 status이며 정확도 합격 수가 아닙니다.</p>'
    method_html+=table(['방법','방식','저장 성공/전체','공통 Huber','현재 후보'],[[link('#'+m,m),H(METHODS[m[1:]]['name']),f"{sum(r['status']=='ok' for r in bymethod[m])}/{len(bymethod[m])}",'적용' if m in FITTED else '미적용','활성' if m in ['E01','E02','E06','Q02','Q04'] else '비활성'] for m in ALL])
    method_text=[]
    for m in ALL:
        d=METHODS[m[1:]]; plist=[p for p in params if p['method']==m]
        ids=[f['id'] for f in enriched if m in f['methods']]
        method_html+=f'<article id="{m}" class="method"><h3>{m} · {H(d["name"])}</h3><p class="pipeline">{H(d["pipeline"])}</p><p><b>근거가 확인된 부분:</b> {H(d["kept"])}</p><p><b>판단:</b> {H(d["decision"])}</p><p><b>방법별 쟁점:</b> {f_links(d["ids"])}</p><p><b>공통 항목 포함:</b> {f_links(ids)}</p><p>{refs(d["src"])}</p>'
        method_html+='<details><summary>이 방법의 설정값과 근거 구분 — '+str(len(plist))+'개</summary>'+table(['설정','값','근거 분류','의미'],[[H(p['key']),'<code>'+H(json.dumps(p['value'],ensure_ascii=False))+'</code>',H(p['status']),H(p['note'])] for p in plist])+'</details>'
        gallery=[(SRC/f'figures/methods/{m}_R{i:02}.png',f'R{i:02} 기존 그림') for i in range(1,5)]
        method_html+='<p class="small">기존 결과를 새로 계산하지 않고 연결: '+' · '.join(link(rel(p),s) for p,s in gallery if p.exists())+'</p></article>'
        method_text.extend(['',m+' '+d['name'],d['pipeline'],'근거: '+d['kept'],'판단: '+d['decision'],'관련: '+', '.join(ids)])
        for p in plist:
            method_text.append(f"  {p['key']}={json.dumps(p['value'],ensure_ascii=False)} | {p['status']} | {p['note']}")
    add('6. 방법별 전수 점검 — 21개',method_html,'\n'.join(method_text),'methods')

    details='<p>‘제안’은 후속 수정 방향입니다. 이번 작업에서 적용하지 않았습니다. ‘관측/범위’가 정적 위험과 실제 발견을 구분합니다.</p>'
    finding_text=[]
    for f in enriched:
        details+=f'<article class="finding" id="{f["id"]}"><div class="tags">{H(f["category"])} · {H(f["priority"])} 검토</div><h3>{f["id"]} · {H(f["title"])}</h3><p class="small">대상: {H(", ".join(f["methods"]))}</p><dl>'
        finding_text.extend(['',f['id']+' '+f['title'],f['category']+' / '+f['priority'], '대상: '+', '.join(f['methods'])])
        for key,label in [('current','현재 코드'),('judgement','판정'),('effect','영향'),('observed','관측/범위'),('proposal','수정·검증 제안')]:
            details+='<dt>'+label+'</dt><dd>'+H(f[key])+'</dd>'
            finding_text.append(label+': '+f[key])
        details+='</dl><details><summary>코드 근거 '+str(len(f['code_evidence']))+'곳</summary><ul>'
        for e in f['code_evidence']:
            details+='<li>'+link(e['view'],e['path']+':'+str(e['line']))+'<br><code>'+H(e['needle'])+'</code></li>'
            finding_text.append(f"코드: {e['path']}:{e['line']} | {e['needle']}")
        details+='</ul></details>'
        if f['sources']:
            details+='<p class="small">출처: '+refs(f['sources'])+'</p>'
            finding_text.extend('출처: '+SOURCES[s][1] for s in f['sources'])
        details+='</article>'
    add('7. 발견 사항 — 32개',details,'\n'.join(finding_text),'findings')

    badhash=[r['path'] for r in checks['runtime_source_hashes'] if not r['matches_execution_log']]
    failed=[r for r in records if r['status']!='ok']
    obs_rows=[['입력 이미지','4장 모두 manifest 해시 일치'],['저장 측정','216개 / 성공 status 209개 / 실패 status 7개'],['기록 구조·유한값 등','점검한 필드의 불일치 0건'],['선별본','48개 모두 원본 기록과 동일'],['조합 참조','432개 참조·status 계산의 불일치 0건'],['외부 좌우 가시 영역','성공 저장값에서 한쪽 지지만 남은 사례 0건'],['최종 점의 band·최소 개수','점검한 성공 저장값에서 이탈·미달 사례 0건'],['기존 실행 해시','tools/make_figures.py만 현재 파일과 불일치. 측정 소스/설정은 일치'],['수렴 여부','Huber 종료 사유가 없어 과거 정상 수렴 여부 확인 불가']]
    datahtml=table(['확인 항목','결과'],[[H(a),H(b)] for a,b in obs_rows])
    datahtml+='<p><b>기록이 맞게 저장됐다는 것과, 직선이 실제 경계에 정확하다는 것은 별개입니다.</b> 현재 값에 드러나지 않은 방어 누락은 F04·F05·F08·F18에 따로 표시했습니다.</p>'
    datahtml+='<details><summary>기존 실패 기록 7개</summary><pre>'+H(json.dumps([{'measurement_id':r['measurement_id'],'reason':r.get('failure_reason',r.get('reason',r.get('status')))} for r in failed],ensure_ascii=False,indent=2))+'</pre></details>'
    datahtml+='<p>'+link('artifact_checks.json','기록 점검 원자료')+' · '+link('stored_observations.json','216개 기록별 확인 내용')+'</p>'
    add('8. 저장 결과에서 실제로 확인한 것',datahtml,'\n'.join(' | '.join(r) for r in obs_rows),'artifacts')

    vendorrows=[]
    for v in vendor:
        archive=v.get('pristine_equals_cached_official_archive',v.get('pristine_equals_cached_official_download'))
        vendorrows.append([H(v['group']+'/'+v['file']),'일치' if archive else '불일치','원본과 동일' if v['unchanged_from_pristine'] else link(v['actual_diff'],'변경 diff 보기')])
    add('9. 저자 소스의 변경 대조',table(['실제 사용 파일','보관 공식 원본 확인','실제 사용본 비교'],vendorrows)+'<p>'+link('third_party_comparison.json','소스 해시·비교 원자료')+'</p><p>‘원본과 동일’은 해당 파일의 구현 이력을 확인한 결과입니다. 뒤에 붙인 선택기와 재피팅, 설정값까지 원 저자 검증을 받았다는 의미는 아닙니다.</p>',json.dumps(vendor,ensure_ascii=False,indent=2),'vendor')

    nextsteps=[
        ('1','코드·설명 불일치부터 정정할 수정안 확정','누락 ID 비교(F28), 사용점/마스크 구분(F04), 0.5px 설명(F24), 유효 설정(F27), 해시 이력(F29).','검출 방법을 바꾸기 전에 해결 가능한 항목. 이번에는 미수정.'),
        ('2','무엇을 비교하는지 고정','원 검출 선분을 통합할지 실제 경계점을 재피팅할지, 입력별 직선 추정 가정을 명시(F06·F11).','모든 방법에 한 피팅을 덮어씌우기 전에 결정.'),
        ('3','근거 없는 값의 검증 계획 수립','Huber scale·trim·수렴값과 각 방법의 창/임계/모델 가정을 분리(F01~F03, F12~F26).','원문 기본값도 이 데이터의 최적값은 아님. 기본값 재현과 변경 효과를 구분.'),
        ('4','정한 범위만 다시 측정·평가','출력 정의를 고정한 후 후보 비교와 θ 반응 검증의 순서를 결정.','현재 4장 편차만으로 정확도 합격이나 기울기-θ 관계를 확정하지 않음.')]
    add('10. 다음 수정 순서 제안',table(['순서','작업','구체적인 내용','경계'],[[H(x) for x in row] for row in nextsteps])+'<p>현재 활성 후보 <b>E01·E06·E02 / Q02·Q04</b>는 그대로 보존했습니다. E05를 다시 넣거나 탈락 방법을 확정하지 않았습니다. <b>“전부 표준화해서 재실행”을 바로 하는 것은 적절하지 않습니다.</b> 입력 점의 종류부터 다른 만큼, 표준화 대상과 측정 정의를 먼저 정해야 합니다.</p>','\n'.join(' | '.join(x) for x in nextsteps),'next')

    sourcehtml=''; source_text=[]
    for s,(title,url,scope) in SOURCES.items():
        sourcehtml+=f'<article id="{s}"><h3>{H(s)} · {link(url,title)}</h3><p>{H(scope)}</p></article>'
        source_text.extend(['',s+' '+title,url,scope])
    sourcehtml+='<p>온라인 문서로 옮겨졌거나 원문 전문을 읽지 못한 항목은 위에 명시했습니다. 보관 공식 다운로드와 실제 사용 소스는 별도로 해시·diff를 확인했습니다. 레퍼런스의 존재만으로 프로젝트의 고정값을 정당화하지 않았습니다.</p>'
    add('11. 근거 자료와 열람 범위',sourcehtml,'\n'.join(source_text),'sources')

    auditfiles=[('findings.json','32개 항목과 코드 위치'),('method_parameter_ledger.json','21개 방법 설정값 분류'),('artifact_checks.json','기록·입력 해시 점검'),('third_party_comparison.json','외부 소스 비교'),('source_views.json','코드 스냅샷 목록·해시'),('baseline_manifest.json','점검 전 원본 파일 해시'),('preservation_check.json','점검 후 원본 보존 확인')]
    add('12. 재현 자료와 원본 보존','<p>점검 스크립트는 읽기·해시·자료 대조·보고서 생성만 수행합니다. 기존 검출기·회귀·통계 검정 코드를 호출하지 않습니다. 보존 확인 범위는 기존 3개 분석 폴더의 파일이며 build, __pycache__, .git, qa 폴더는 제외합니다.</p><ul>'+''.join('<li>'+link(p,l)+'</li>' for p,l in auditfiles)+'</ul><p>점검용 새 산출물: <code>'+H(str(HERE))+'</code></p>','원본 3개 분석 폴더는 baseline_manifest.json 대비 검증. preservation_check.json 참조.','preservation')

    nav=' · '.join(link('#'+s,t) for s,t in [('summary','결론'),('first','우선 항목'),('common','공통 처리'),('methods','21개 방법'),('findings','32개 상세'),('artifacts','저장 결과'),('next','다음 순서'),('sources','출처')])
    css='''*{box-sizing:border-box}html{scroll-behavior:smooth;scroll-padding-top:75px}body{margin:0;background:#f3f5f8;color:#202b37;font:16px/1.7 "Malgun Gothic","Segoe UI",sans-serif}header,main{max-width:1220px;margin:auto;padding:30px}header{padding-bottom:15px}h1{font-size:30px;line-height:1.4;margin:0 0 10px}h2{font-size:23px;margin:0 0 16px}h3{font-size:19px;margin:10px 0}nav{position:sticky;top:0;z-index:5;background:#102d44;padding:12px 20px;text-align:center;color:white}nav a{color:#d7efff}a{color:#075b9c;text-underline-offset:3px}section{background:#fff;border:1px solid #dce3eb;border-radius:10px;padding:28px;margin-bottom:24px}p{margin:12px 0}.lead{font-size:19px;font-weight:600}.notice{background:#fff5df;border-left:5px solid #d29617;padding:18px;margin-top:20px}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:14px;margin:15px 0}th,td{border:1px solid #dbe2ea;padding:10px 12px;text-align:left;vertical-align:top}th{background:#edf2f7;white-space:nowrap}td{min-width:90px;overflow-wrap:anywhere}td:first-child{min-width:100px}article.finding,article.method{border:1px solid #dce3eb;border-radius:8px;padding:20px;margin:22px 0}article:target{outline:3px solid #e2ae2a}.tags{font-size:13px;color:#7b4a00}.small{font-size:13px;color:#536373}.pipeline,.formula{background:#ecf5fc;padding:14px;border-radius:5px}.formula{font-family:Consolas,"Malgun Gothic",monospace}dt{font-weight:700;margin-top:10px}dd{margin:3px 0 12px;padding-left:14px;border-left:3px solid #dce7f2}details{border:1px solid #dce3eb;border-radius:6px;padding:10px 14px;margin:12px 0}summary{cursor:pointer;font-weight:600}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}code{font-size:.9em;overflow-wrap:anywhere}li{margin:8px 0}footer{max-width:1220px;margin:auto;padding:20px 30px;color:#627184}@media(max-width:700px){main,header{padding:15px}section{padding:17px}h1{font-size:24px}nav{position:static}}@media print{nav{display:none}body{background:white;font-size:11px}main{padding:0}section{break-inside:auto;border:none}article{break-inside:avoid}details>*{display:block}h1{font-size:22px}}'''
    doc='<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>직선 추출 파이프라인 전수 재점검</title><style>'+css+'</style></head><body><header><div class="small">2026-10-30 · 읽기 전용 코드/근거 감사</div><h1>직선 추출 파이프라인 전수 재점검</h1><p>외부 구조물 9개 · QR 12개 · 공통 피팅·선별·기록</p><p>'+link('report.txt','텍스트 전달본')+' · '+link('findings.json','항목별 JSON')+'</p></header><nav>'+nav+'</nav><main>'+''.join(body)+'</main><footer>새 측정 또는 통계 검증 없이, 기존 구현의 근거와 한계를 점검한 보고서입니다.</footer></body></html>'
    (HERE/'report.html').write_text(doc,encoding='utf-8')
    (HERE/'report.txt').write_text('직선 추출 파이프라인 전수 재점검\n2026-10-30\n'+lead+'\n\n'+ '\n'.join(texts),encoding='utf-8')
    write('report_summary.json',dict(findings=len(enriched),categories=dict(categories),methods=len(ALL),method_parameter_entries=len(params),stored_rows=len(records),active_methods=['E01','E06','E02','Q02','Q04'],scope='Read-only audit. No image measurement, fit, statistical test or original modification.'))
    print(json.dumps(read(HERE/'report_summary.json'),ensure_ascii=False))

if __name__=='__main__':
    main()
