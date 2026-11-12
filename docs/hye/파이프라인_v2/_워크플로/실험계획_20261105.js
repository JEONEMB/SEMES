export const meta = {
  name: 'semes-pipeline-v2-plans',
  description: 'SEMES 파이프라인 v2: 19개 단위의 단계별 실험 계획 01_실험계획.md 작성. 실험 실행과 v3 는 하지 않는다.',
  phases: [
    { title: 'Plan', detail: '단계별 실험 계획 01_실험계획.md' },
  ],
}

const REPO = args.repo
const BUILD = args.build || 'cmake -S 실험 -B <임시 빌드 폴더> -DCMAKE_BUILD_TYPE=Release (이 컴퓨터에 설치된 OpenCV를 find_package로 찾는다)'
const V2 = REPO + '/docs/hye/파이프라인_v2'
const CANVAS = REPO + '/docs/hye/문헌연구/00005_파이프라인/00_파이프라인_검증현황.canvas'
const COMMON = `
공통 맥락 (반드시 지킬 것)
- 저장소: ${REPO} (현재 브랜치 그대로 사용, 브랜치 전환 금지). 계획: ${V2}/plan.md. 문헌 지침: ${REPO}/docs/hye/문헌연구/AGENTS.md. 알고리즘 규칙: ${REPO}/ALGORITHM_RULES.md. 먼저 이 세 파일을 읽는다.
- 과제: 로봇이 작업하다 웨이퍼 투입 전에 센터 자리로 오면, 카메라 사진(3840x2160 8bit)의 QR 3개와 뒷벽 위·아래 가로선만으로 센터 대비 Y(mm)·T(회전 명령 단위) 편차를 추정하고, 이동·재촬영을 최대 20턴 반복해 |eY|,|eT| ≤ 0.045 안에 정지시킨다. 1차는 joint()로 T만, 2차는 control()로 Y. 나사·볼트·일반 배경은 관측에 쓰지 않는다.
- 현재 모듈: ${REPO}/experiments/hye/20261028_1258_final_Y_T_joint (NCC→ECC 호모그래피 QR54 + 가로선 48블록 세로 프로파일 → 204차원 GLS), ${REPO}/experiments/hye/20261028_1327_final_Y_T control (NCC→LK15→17x17 스플라인 GN, 고정 536점, 센터 차감, P12, 6절편 후보 곡선 역산, Y-or-T).
- 단계 캔버스(목적·해결할 질문·잠정 후보·사이 보정·평가·현재 상태): ${CANVAS} (JSON. 노드 id로 찾는다).
- 핵심 수치: Y 1mm≈14.36px, T 1단위≈80.3px, 허용 ±0.045 = Y 0.65px. Y·T 영상 반응 cos 0.99999. 점 측정 잡음 0.009~0.015px, 536점의 유효 독립 정보 ≈19. 영점: 하루 안 가로 ≤0.84px(백래시 β −0.34px, 드리프트), 날짜 간 세로 +0.44px(카메라 이동, QR−가로선으로 상쇄, A15). 모든 자료는 명령한 자리에 멈춰 찍은 정지 사진이며 실제 작업 후 도착 사진은 없다. 341/575장은 개발에 노출된 자료다.
- 금지: 원본·기존 결과·기존 final 폴더 수정 금지. data 커밋 금지. git 커밋/푸시 금지(메인이 한다). 봇 차단 우회 금지. 확인하지 못한 서지·수치를 지어내지 않는다(실제로 OpenAlex/arXiv/Google Patents 등에서 확인한 것만 등록). 시험 정답을 학습에 쓰지 않는다.
- 커밋 메시지와 문서에 사람 이름이나 '누가 무엇을 참고했다'는 표기를 넣지 않는다(문헌 서지의 저자명은 예외).
- 이어하기 실행이다. 만들 파일이 이미 있고 요구한 구성을 다 갖췄으면 다시 쓰지 말고 확인만 한 뒤 그 내용으로 결과를 돌려준다. 일부만 있으면 빠진 부분만 채운다.
- 문체: 한국어, 쉬운 말, 짧은 문장. 문서에 '—'(em dash)와 '·'(가운뎃점) 대신 쉼표와 줄바꿈 목록을 쓴다.
`

const UNITS = [
  { code: 'P', dir: 'P_문제정의', node: 'sP', title: '[P] 문제 정의', focus: '왜 로봇 명령만으로 센터에 못 가는가, 비전 정렬 시스템 요구사항·환경·평가 기준(웨이퍼 이송 로봇 티칭, 비전 얼라인먼트 사례)' },
  { code: '0', dir: '0_기준만들기', node: 's0', title: '[0] 기준 만들기', focus: '센터(기준 자세) 정의: 기준 영상 1장 vs 다중 평균 vs 좌표 정의, 절대 위치 부호화 마커, 재설치·개체 간 전이, 기준 보정' },
  { code: '1', dir: '1_촬영', node: 's1', title: '[1] 촬영', focus: '작업 중 도착 후 촬영 조건: 카메라 예열 열 드리프트, 정착 시간·잔진동, 조명 플리커·노출, 다중 촬영 평균, 사이클 시간' },
  { code: '2-1', dir: '2_측정/2-1_대략위치', node: 's2_1', title: '[2-1] 대략 위치 찾기', focus: '정밀 정합 전 초기화: 템플릿 NCC, 피라미드, 위상상관, 코드 디코딩 기반 대응, 큰 이동·탐색 범위' },
  { code: '2-2a', dir: '2_측정/2-2_정밀측정/a_특징점', node: 's2_2', title: '[2-2a] 정밀 측정: 특징점 기반', focus: '특징점 계열만: 코너/안장점 검출(Harris, Shi-Tomasi, Förstner, ChESS, 안장점), 서술자(SIFT, ORB, AKAZE), 부화소 코너 정밀화, QR/체커보드 격자점 정밀도' },
  { code: '2-2b', dir: '2_측정/2-2_정밀측정/b_영역', node: 's2_2', title: '[2-2b] 정밀 측정: 영역 기반', focus: '영역 계열만: NCC/ZNCC 부화소, 위상상관, LK, IC-GN/DIC, ECC, 직접 호모그래피 정합, 밝기 변화 강건 정합' },
  { code: '2-2c', dir: '2_측정/2-2_정밀측정/c_선에지', node: 's2_2', title: '[2-2c] 정밀 측정: 선·에지 기반', focus: '선·에지 계열만: 에지 부화소 위치 추정(모멘트, 가우스 적합, ESF), 직선 검출(LSD, Hough) 정밀도, 프로파일 정합, 선 기반 자세 추정' },
  { code: '2-3', dir: '2_측정/2-3_사이보정', node: 's2_3', title: '[2-3] 사이 보정', focus: '측정 전후 보정: 부화소 위상 편향(peak locking) 보정, 보간 편향, 렌즈 왜곡 보정, 밝기 정규화, 정·역 일관성, 이상치 제거, 점별 신뢰 가중' },
  { code: '2-4', dir: '2_측정/2-4_품질판정', node: 's2_4', title: '[2-4] 측정 품질 판정', focus: '측정 실패·신뢰도 판정 지표: 정합 점수, 잔차, 불확도 정량화, 측정 유효성 판정 특허' },
  { code: '3', dir: '3_기준과비교', node: 's3', title: '[3] 기준과 비교', focus: '기준 대비 차분, 고정 기준점으로 카메라/장비 드리프트 상쇄, 기준 불확도 전파' },
  { code: '4', dir: '4_특징만들기', node: 's4', title: '[4] 특징 만들기', focus: '변위장 요약 표현(다항식, 아핀/호모그래피 파라미터, PCA/PLS), 상관 잡음 하 가중(GLS), 정보량(Fisher, CRLB), 점 방식 vs 영역 방식 정보량' },
  { code: '5a', dir: '5_YT계산/a_회귀ML', node: 's5', title: '[5a] Y/T 계산: 회귀·ML', focus: '공선성 회귀(Ridge, PLS, 축소 공분산 GLS), errors-in-variables(명령값 라벨 오차), 혼합/잠재 절편 회귀, 트리·부스팅·신경망 정확도와 시간, 외삽' },
  { code: '5b', dir: '5_YT계산/b_물리식', node: 's5', title: '[5b] Y/T 계산: 물리식', focus: '물리 모델 추정: 호모그래피 분해, 평면 자세(IPPE), 회전축·중심 교정과 식별성(소각), 자기 보정, 물리식+잔차 학습 하이브리드' },
  { code: '6', dir: '6_판정', node: 's6', title: '[6] 판정', focus: '추정 신뢰 판정: 예측 구간, conformal prediction, 분포 밖(OOD) 탐지, 재촬영·보류 정책, 다중 측정 평균' },
  { code: '7', dir: '7_명령만들기', node: 's7', title: '[7] 명령 만들기', focus: 'look-and-move 비전 보정 이득, 양자화 구동기, 과보정·데드밴드, 최대 이동 제한' },
  { code: '8', dir: '8_이동', node: 's8', title: '[8] 이동', focus: '스테이지·로봇 백래시, 로스트 모션, 단방향 접근, 반복정밀도 측정(ISO 9283, ISO 230-2)' },
  { code: '9', dir: '9_정지', node: 's9', title: '[9] 정지', focus: '정착 시간, 잔진동, 정지 후 드리프트, 촬영 타이밍' },
  { code: '10', dir: '10_단계전환', node: 's10', title: '[10] 단계 전환', focus: '결합된 축의 순차/동시 정렬, 2단계 정렬 전환 조건, 수렴 보장, 재이탈 감지·복귀, XYθ 얼라인먼트 특허' },
  { code: '11', dir: '11_검증', node: 's11', title: '[11] 검증', focus: '독립 기준 측정, 반복성·재현성(Gage R&R), 측정 불확도(GUM), 개체 간 전이 검증, 선택 편향 없는 평가' },
]

const SEL = { type: 'object', properties: {
  docs: { type: 'array', items: { type: 'object', properties: {
    kind: { type: 'string', enum: ['논문', '특허'] },
    folder: { type: 'string', description: 'absolute path of the per-document folder to create' },
    title: { type: 'string' }, citation: { type: 'string' }, url: { type: 'string' }, pdf_url: { type: 'string' },
    questions: { type: 'array', items: { type: 'integer' } }, why: { type: 'string' } },
    required: ['kind', 'folder', 'title', 'citation', 'url', 'why'] } },
  existing: { type: 'array', items: { type: 'object', properties: { path: { type: 'string' }, why: { type: 'string' } }, required: ['path', 'why'] } },
}, required: ['docs', 'existing'] }

const DOCRES = { type: 'object', properties: {
  folder: { type: 'string' }, pdf: { type: 'boolean' }, translation: { type: 'string', enum: ['full', 'abstract', 'none'] }, note: { type: 'string' } },
  required: ['folder', 'pdf', 'translation', 'note'] }

const PLANRES = { type: 'object', properties: {
  offline: { type: 'array', items: { type: 'object', properties: { id: { type: 'string' }, title: { type: 'string' }, question: { type: 'string' }, candidates: { type: 'array', items: { type: 'string' } } }, required: ['id', 'title', 'question'] } },
  robot_needed: { type: 'array', items: { type: 'string' } } }, required: ['offline', 'robot_needed'] }

const EXPRES = { type: 'object', properties: {
  folders: { type: 'array', items: { type: 'string' } },
  findings: { type: 'array', items: { type: 'string' } },
  recommendation_control: { type: 'string' }, recommendation_joint: { type: 'string' }, failed: { type: 'array', items: { type: 'string' } } },
  required: ['folders', 'findings', 'recommendation_control', 'recommendation_joint', 'failed'] }

const SELDIR = REPO + '/docs/hye/파이프라인_v2/_워크플로/선정'
function unitDir(u) { return `${V2}/단계/${u.dir}` }



const BUDGET = `
토큰 규칙 (어제 이 작업이 사용 한도로 죽었다. 반드시 지킨다)
- 긴 글은 Write 로 파일에 바로 쓰고, 쓴 파일을 다시 읽지 않는다. 응답에 본문을 되붙이지 않는다.
- 번역은 추출한 본문을 조각으로 나눠 조각마다 파일에 append 한다. 앞서 쓴 번역을 다시 context 에 올리지 않는다.
- pdftotext 결과는 파일로 저장하고 필요한 구간만 sed 로 읽는다. 전체를 한 번에 받지 않는다.
- 큰 도구 출력(파일 목록, 긴 로그)을 그대로 받지 않는다. wc, head, 집계값만 본다.
- 마지막 응답은 20줄 이내. 경로, 상태, 실패 사유만.
- 사용 한도 오류가 나면 재시도하지 말고 그 시점 상태를 남기고 끝낸다.
`


const results = await parallel(UNITS.map(u => () =>
  // D. experiment plan
  agent(`${COMMON}${BUDGET}
단계 "${u.title}"의 실험 계획을 쓴다: ${unitDir(u)}/01_실험계획.md
근거: ${unitDir(u)}/00_단계.md (후보 방법 표, 질문별 답), ALGORITHM_RULES.md, 기존 실험 ${REPO}/experiments/hye (특히 20261029_* 측정 비교, 20261102_* A0~A16·B1~B6 계획서와 보고서, 20261102_142904_결론_센터복귀_반복정밀도.md).
원칙: 후보는 문헌 기반 표에서 고른다. 같은 자료, 같은 분할(기존 A0 계층 분할과 캐시 ${REPO}/experiments/hye/20261102_124309_A0_분할측정고정_계층분할과점측정캐시 를 우선 재사용, control 비교는 validation/splits.json 고정 4-fold, joint 비교는 341장 외부 4-fold), 정확도(축별 평균·최대오차, |e|≤0.045 통과율)와 처리 시간(ms)을 함께. 날짜를 나눈 평가를 반드시 포함. 기존 final 모듈을 대조군으로 둔다. 시험 정답 사용 금지.
오프라인(data/clean 정지 사진)으로 할 수 있는 실험과, 실기 촬영이 필요한 실험(촬영 요청서 형태: 몇 자세, 몇 회, 어떤 조건)을 나눈다. 오프라인 실험은 이 단계 질문에 직접 답하는 것 1~3개로 하되 각각 실험 ID(V2-${u.code}-E1 형식), 질문, 후보, 고정 조건, 지표, 판정 기준(실험 전에 고정), 예상 소요를 적는다. 이미 같은 조건으로 끝난 실험이면 재사용하고 새로 돌리지 않는다고 적는다.
문제 정의·명령·이동·정지처럼 오프라인으로 답할 수 없는 단계는 오프라인 실험을 0개로 두어도 된다(대신 체크리스트나 촬영 요청서를 쓴다).`,
    { label: `plan:${u.code}`, phase: 'Plan', schema: PLANRES }).then(p => ({ code: u.code, plan: p }))
))

const done = results.filter(r => r && r.plan)
log(`실험 계획 ${done.length}/${UNITS.length}. 실패: ${UNITS.filter(u => !done.find(d => d.code === u.code)).map(u => u.code).join(', ') || '없음'}`)
return { 계획완료: done.length, 대상: UNITS.length, 오프라인실험: done.reduce((a, d) => a + (d.plan?.offline?.length || 0), 0) }
