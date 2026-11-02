"""Record only the requested five fixes and preserve the historical provenance gap honestly."""
import difflib,hashlib,json,re,shutil,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent.parent
BASE=ROOT/'analysis/reference_line_methods_108_20261030'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
subprocess.run([sys.executable,str(HERE/'safeguard.py'),'--verify'],check=True,stdout=subprocess.PIPE)
proof=json.loads((HERE/'preservation_check.json').read_text(encoding='utf-8'))
validation=json.loads((HERE/'validation.json').read_text(encoding='utf-8'))
assert len(validation['steps'])==8 and all(s['exit_code']==0 for s in validation['steps'])
counts={}
for name in ('03_fit_regression','04_pipeline_regression','05_numeric_records'):
    log=(HERE/'logs'/f'{name}.stdout.txt').read_text(encoding='utf-8-sig')
    assert '0 failure(s)' in log
    counts[name]=sum(line.startswith('PASS ') for line in log.splitlines())
for name in ('07_python_checks','08_display_regression'):
    log=(HERE/'logs'/f'{name}.stderr.txt').read_text(encoding='utf-8-sig')
    assert '\nOK\n' in log
    counts[name]=int(re.search(r'Ran (\d+) tests',log).group(1))

patches=HERE/'patches';patches.mkdir(exist_ok=True)
changes=[]
for path in proof['changed_files']:
    current=ROOT/path;relative=current.relative_to(BASE);before=HERE/'before'/relative
    original=before.read_text(encoding='utf-8-sig') if before.exists() else ''
    updated=current.read_text(encoding='utf-8-sig')
    dest=patches/(relative.as_posix().replace('/','__')+'.diff')
    dest.write_text(''.join(difflib.unified_diff(original.splitlines(True),updated.splitlines(True),
        fromfile='before/'+relative.as_posix(),tofile='after/'+relative.as_posix())),encoding='utf-8')
    changes.append({'path':path,'before_sha256':sha(before) if before.exists() else None,
                    'after_sha256':sha(current),'diff':dest.relative_to(HERE).as_posix()})

# A snapshot is not evidence of which historical code generated an existing figure.
baseline=json.loads((HERE/'before_manifest.json').read_text(encoding='utf-8'))
prefix=BASE.relative_to(ROOT).as_posix()+'/'
artifacts={path:value for path,value in baseline.items() if path.startswith(prefix) and
           (path[len(prefix):].startswith(('results/','figures/','logs/')) or path[len(prefix):] in ('report.html','report.md'))}
legacy={'kind':'preserved_existing_artifact_inventory_not_a_generation_record',
        'producer_version':'unknown where no historical generation record exists',
        'note':'Hashes describe unchanged historical files. Current generator source is not attributed to these artifacts. Future runs use separate append-only stage records.',
        'files_sha256':artifacts}
(HERE/'legacy_artifacts_inventory.json').write_text(json.dumps(legacy,ensure_ascii=False,indent=2),encoding='utf-8')
shutil.copyfile(BASE/'logs/hashes.json',HERE/'legacy_existing_hashes.json')
exe=HERE/'build/Release/line_methods.exe'
manifest={'scope':['F18 arithmetic guards only','F28 full/subset comparison','F24 contour explanation',
                   'F29 future stage provenance and preserved legacy gap','F30 calculation coordinate precision'],
          'changed_files':changes,'passed_checks':counts,'configuration_slots_checked':54,
          'all_configs_byte_identical':proof['all_config_files_byte_identical'],
          'historical_artifacts_preserved':proof['historical_artifacts_preserved'],
          'executable':str(exe),'executable_sha256':sha(exe),
          'captured_image_analysis_run':False,'statistical_validation_run':False,'historical_reports_regenerated':False,
          'legacy_generation_version_reconstructed':False}
(HERE/'change_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
cpp=sum(counts[k] for k in ('03_fit_regression','04_pipeline_regression','05_numeric_records'))
py=counts['07_python_checks']+counts['08_display_regression']
report=f'''코드·기록·설명 보완 5건 — 2026-10-31

적용 소스: {BASE}
새 실행 파일: {exe}
기존 build, 측정값, 그림, 보고서, 선별 결과는 그대로 보존함.
공통 1~6번 수정 위에 아래 5건만 보완함.

1. F18 — Zernike 계산 유효성 검사
- 1차 모멘트가 0인 경우, 분모가 0인 경우를 나눗셈 전에 거부함.
- 모멘트·분모·비율·판별식·위치 계산의 NaN/Inf/넘침을 검사함.
- 음수 판별식은 제곱근 계산 전에 거부함.
- 실패 후보는 선택점에 넣지 않고 원인을 저장함.
- 아직 계산되지 않은 진단값은 0을 꾸며 넣지 않고 JSON null로 기록함.
- 기존 정상 입력의 수식, 방향 조건, 이동 제한, 작은 폭의 기존 분기는 유지함.
- '얼마보다 작은 분모를 버린다'는 새 경험적 임계값은 추가하지 않음.
- 원문의 적용 조건, 흐림 폭, 64×64 수치적분 충분성 및 평균 제거 영향은 이번 검사로 검증한 것이 아님.

2. F28 — 전체 비교와 지정 부분 비교 구분
- 전체 비교: 계획된 measurement_id 집합과 개수가 양쪽 모두 정확히 맞아야 통과함.
- 지정 부분 비교: 별도로 선언한 ID가 기준 결과에 모두 있고, 부분 결과가 그 집합과 정확히 일치해야 통과함.
- 기대 ID는 출력 파일의 교집합이 아닌 입력 manifest·방법 목록·대상 목록에서 생성함.
- 누락·추가·중복 ID, 빠진 필드, 비정상 숫자, 빈 비교를 정상으로 처리하지 않음.
- 비교 불일치 시 파이프라인은 실패로 종료함.
- passed는 해당 비교 모드의 통과 여부, identical은 전체 ID 집합까지 같은 경우에만 true임.
- 비교 대상 필드도 결과에 명시함. 파일 전체의 바이트 일치라는 뜻은 아님.
- 예전 비교 JSON에는 필수 ID·개수 검사가 없으므로 보고서에서 '전체 일치 보증 불가'로 구분함.

3. F24 — Q10 윤곽점 설명 정정
이전: 실제 경계보다 항상 약 0.5px 위이며 일정한 차이라 기울기에 영향이 없다고 설명함.
정정: 가장 바깥 밝은 화소의 중심을 사용하며, 실제 경계와의 차이와 기울기 영향은 별도 검증 대상임.
코드 진단 문구와 보고서 생성 설명에 반영함. 좌표를 0.5px 이동시키거나 계산식을 바꾸지 않음.
과거 JSON/보고서는 원 기록으로 보존하므로 정정 내용은 이 문서와 새 생성 코드에서 확인함.

4. F29 — 측정·그림·보고서 생성 기록 분리
- 각 단계에서 logs/provenance/ 아래 고유 실행 ID의 시작/종료 JSON을 새로 생성함.
- 시작 기록에 코드·입력 해시, 실행 조건, UTC 및 한국 시각을 저장함.
- 종료 기록에 산출물 해시와 실행 중 코드·입력 변경 여부를 저장함.
- 코드/입력이 중간에 바뀌면 유효한 완료로 기록하지 않음.
- 기존 생성 기록의 산출물과 새 입력의 경로·해시가 일치할 때만 이전 실행과 연결함.
- 생성 기록 파일은 독점 생성하며 이전 실행 기록을 덮어쓰지 않음.
- 보고서 작성이 중간에 중단되어 종료 기록이 없으면 '시작 후 미완료' 기록이 남음.
- 이력 기록은 run_pipeline.py, make_figures.py, make_report.py 실행 경로에 연결함.
- 실행 파일을 직접 호출해 만든 과거/별도 산출물에는 해당 단계 기록이 없을 수 있으며, 그 경우 생성 코드 버전을 추정하지 않음.

과거 그림의 정확한 생성 코드 버전은 기존 자료만으로 복원하지 못함.
legacy_artifacts_inventory.json에 기존 산출물의 해시를 보존하고,
legacy_existing_hashes.json에 과거 해시 로그를 그대로 복사함.
이 파일들은 과거 생성 버전을 입증하는 기록이 아니라 기존 파일 보존 확인 자료임.
이번 작업에서는 실제 촬영본으로 그림·보고서를 재생성하지 않았음.

5. F30 — 계산용 좌표와 표시용 좌표 구분
- 공통 피팅에 넘기는 전체 입력 좌표를 finish_with_points에서 중앙 저장함.
- fit_points는 소수점 자릿수를 잘라 저장하지 않고 double JSON 왕복 정밀도를 유지함.
- fit_points_precision에 저장 정밀도와 점 집합의 의미를 기록함.
- Q11/Q12의 두 점 연결 입력도 line_input_points에 원래 계산값 그대로 저장함.
- 원 검출기가 float로 계산한 값을 double로 저장했다고 새 정밀도가 생긴다고 주장하지 않음.
- HTML 뷰어의 소수점 2자리 좌표는 표시용으로 유지하며, 재계산에는 원본 중간 JSON을 사용하도록 명시함.
- 과거 반올림 좌표에는 원정밀 저장을 소급 주장하지 않음.
- 실패 입력의 NaN/Inf는 JSON null로 표현되며 성공 피팅 좌표로 취급하지 않음.

검증 결과
- OpenCV 3.4.15 / MSVC v141 빌드 성공.
- C++ {cpp}개 검사 통과: 기존 피팅 {counts['03_fit_regression']}, 기존 선분·설정 {counts['04_pipeline_regression']}, Zernike 계산·좌표 저장 {counts['05_numeric_records']}.
- Python {py}개 시험 통과: 비교·생성 기록·좌표 표시·기존 기록 해석 및 표시 회귀 시험.
- 실제 실행 파일 --check-config로 설정 54개 대상 조합 확인.
- 모든 설정 파일 바이트 동일, 기존 결과·그림·보고서·선별본·검출기 원본 소스 보존 확인.
- 첫 Python 시험은 Windows TemporaryDirectory의 소유자 전용 권한 때문에 fixture에 접근하지 못함.
  작업 폴더의 상속 권한을 쓰는 임시 fixture로 변경한 뒤 통과함. 초기 로그는 logs/first_attempt/에 남김.
- 기존 외부 라이브러리 컴파일 경고는 로그에 보존했으며 해당 라이브러리를 수정하지 않음.

범위
실제 촬영 이미지 분석, 통계 검정, 임계값 조정, 후보 변경, Git/Jira 작업은 하지 않음.
이번 시험은 코드 동작 검증이며 실제 경계 정확도나 알고리즘 성능 검증이 아님.
Huber 상수·점 제외 거리·수렴 기준 등의 타당성 검토는 여전히 별도임.

근거
- validation.json, logs/: 실행 결과.
- preservation_check.json: 기존 파일 보존 확인.
- change_manifest.json: 변경 소스·실행 파일 해시.
- before/, patches/: 수정 전 소스 및 파일별 diff.

변경 파일
'''+ '\n'.join('- '+x['path'] for x in changes)+'\n'
(HERE/'changes.txt').write_text(report,encoding='utf-8')
print(json.dumps({'changed_files':len(changes),'cpp_checks':cpp,'python_tests':py,
                  'configs_unchanged':True,'historical_artifacts_preserved':True}))
