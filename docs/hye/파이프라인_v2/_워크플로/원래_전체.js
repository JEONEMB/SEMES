export const meta = {
  name: 'semes-pipeline-v2',
  description: 'SEMES Y/T 파이프라인 v2: 단계별 논문·특허 조사 → 리뷰 → 실험계획 → 실험 → control/joint final_v2 재설계·검증',
  phases: [
    { title: 'Research', detail: '단계별 질문·기존 문헌 매핑·새 논문/특허 검색·선정' },
    { title: 'Docs', detail: '문헌별 폴더: 원문·번역·요약' },
    { title: 'Review', detail: '단계 페이지 00_단계.md: 리뷰 표·요약 블록·후보 재작성' },
    { title: 'Plan', detail: '단계별 실험 계획' },
    { title: 'Experiment', detail: 'C++ 실험 구현·실행·결과' },
    { title: 'Design', detail: 'control_v2 / joint_v2 설계' },
    { title: 'Build', detail: 'final_v2 구현·검증' },
    { title: 'Verify', detail: '적대적 검증·수정' },
    { title: 'Report', detail: '결과 보고' },
  ],
}

const REPO = '/Users/ihrong/hwlee/projects/2026-10_SEMES/S15P31A507'
const V2 = REPO + '/docs/hye/파이프라인_v2'
const CANVAS = REPO + '/docs/hye/문헌연구/00005_파이프라인/00_파이프라인_검증현황.canvas'
const COMMON = `
공통 맥락 (반드시 지킬 것)
- 저장소: ${REPO} (브랜치 feature/hye/pipeline-v2). 계획: ${V2}/plan.md. 문헌 지침: ${REPO}/docs/hye/문헌연구/AGENTS.md. 알고리즘 규칙: ${REPO}/ALGORITHM_RULES.md. 먼저 이 세 파일을 읽는다.
- 과제: 로봇이 작업하다 웨이퍼 투입 전에 센터 자리로 오면, 카메라 사진(3840x2160 8bit)의 QR 3개와 뒷벽 위·아래 가로선만으로 센터 대비 Y(mm)·T(회전 명령 단위) 편차를 추정하고, 이동·재촬영을 최대 20턴 반복해 |eY|,|eT| ≤ 0.045 안에 정지시킨다. 1차는 joint()로 T만, 2차는 control()로 Y. 나사·볼트·일반 배경은 관측에 쓰지 않는다.
- 현재 모듈: ${REPO}/experiments/hye/20261028_1258_final_Y_T_joint (NCC→ECC 호모그래피 QR54 + 가로선 48블록 세로 프로파일 → 204차원 GLS), ${REPO}/experiments/hye/20261028_1327_final_Y_T control (NCC→LK15→17x17 스플라인 GN, 고정 536점, 센터 차감, P12, 6절편 후보 곡선 역산, Y-or-T).
- 단계 캔버스(목적·해결할 질문·잠정 후보·사이 보정·평가·현재 상태): ${CANVAS} (JSON. 노드 id로 찾는다).
- 핵심 수치: Y 1mm≈14.36px, T 1단위≈80.3px, 허용 ±0.045 = Y 0.65px. Y·T 영상 반응 cos 0.99999. 점 측정 잡음 0.009~0.015px, 536점의 유효 독립 정보 ≈19. 영점: 하루 안 가로 ≤0.84px(백래시 β −0.34px, 드리프트), 날짜 간 세로 +0.44px(카메라 이동, QR−가로선으로 상쇄, A15). 모든 자료는 명령한 자리에 멈춰 찍은 정지 사진이며 실제 작업 후 도착 사진은 없다. 341/575장은 개발에 노출된 자료다.
- 금지: 원본·기존 결과·기존 final 폴더 수정 금지. data 커밋 금지. git 커밋/푸시 금지(메인이 한다). 봇 차단 우회 금지. 확인하지 못한 서지·수치를 지어내지 않는다(실제로 OpenAlex/arXiv/Google Patents 등에서 확인한 것만 등록). 시험 정답을 학습에 쓰지 않는다.
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

function unitDir(u) { return `${V2}/단계/${u.dir}` }

const results = await pipeline(UNITS,
  // A. research & select
  (prev, u) => agent(`${COMMON}
너는 단계 "${u.title}"의 문헌 조사 담당이다. 단계 폴더: ${unitDir(u)} (없으면 만든다. 하위에 논문/, 특허/).
조사 초점: ${u.focus}
1) 캔버스에서 노드 id "${u.node}"의 텍스트를 읽어 이 단계의 목적과 해결할 질문을 가져온다${u.code.startsWith('2-2') || u.code.startsWith('5') ? ' (이 노드는 여러 하위 단위가 공유하므로 위 조사 초점에 해당하는 부분만 다룬다)' : ''}. 질문이 부족하면 보강해 번호를 붙인다.
2) 기존 문헌(${REPO}/docs/hye/문헌연구/01_주제_관련, 02_QR_분석, 03_Y_T_분리, 04_Y_T_오차_보정 의 요약.md, 그리고 00002_문헌대장_v2.md)에서 이 단계 질문에 답하는 것을 찾아 existing으로 돌려준다(경로와 이유).
3) 새 논문·특허를 검색한다. 논문은 OpenAlex API(curl "https://api.openalex.org/works?search=...")와 arXiv API, 특허는 Google Patents(https://patents.google.com/?q=... 결과 페이지나 patent 페이지)로 확인한다. 기존 문헌과 중복되지 않는 것을 고른다. 새 문헌은 질문을 덮을 만큼 고르되 논문 3~5편, 특허 2~4건을 목표로 한다. 원문 PDF를 무료로 받을 수 있는 것을 우선한다(arXiv, 저자 공개본, 오픈액세스, 특허 공보 PDF). 실제로 존재를 확인한 것만 등록한다.
4) ${unitDir(u)}/조사방법론.md를 쓴다: 목적, 해결할 질문(번호), 검색어(블록 OR/AND), 검색 기록 표(날짜 2026-11-04, 경로, 검색어, 걸린 건수, 후보, 선정), 선정 이유.
5) 선정 문헌마다 폴더 경로를 정해 돌려준다: 논문은 ${unitDir(u)}/논문/NN_저자연도_출처_핵심어, 특허는 ${unitDir(u)}/특허/NN_공보번호_출원인_핵심어 (NN=01부터). 폴더는 만들기만 하고 내용은 다음 단계가 채운다.`,
    { label: `research:${u.code}`, phase: 'Research', schema: SEL }),
  // B. per-doc fetch, translate, summarize
  (sel, u) => parallel((sel?.docs || []).map(d => () => agent(`${COMMON}
문헌 하나를 정리한다. 단계 "${u.title}". 폴더: ${d.folder} (없으면 만든다)
문헌: [${d.kind}] ${d.title} / ${d.citation} / ${d.url} / PDF 후보: ${d.pdf_url || '없음'}
이 단계에서 답하려는 질문 번호: ${(d.questions || []).join(', ')} (질문 원문은 ${unitDir(u)}/조사방법론.md)
1) 원문 PDF를 curl로 받아 폴더에 저장한다(파일명 자유, 확장자 .pdf). 특허는 Google Patents 페이지의 PDF 링크(patentimages.storage.googleapis.com)를 쓴다. 받지 못하면 이유를 기록하고 ${V2}/원문_받을_목록.md 에 "| 단계 | 폴더 | 링크 | 이유 |" 형식 한 줄을 추가한다(파일이 없으면 표 머리와 함께 만든다). 봇 차단을 우회하지 않는다.
2) pdftotext -layout 으로 본문을 뽑아 번역.md를 쓴다. PDF가 있으면 전문 번역(초록, 본문, 그림·표 캡션, 부록. 참고문헌은 원문 그대로). 핵심 그림 1~3개는 pdftoppm -r 110 -png 로 해당 쪽을 이미지화해 그림/ (특허는 도면/)에 넣고 처음 언급되는 곳 근처에 ![](그림/파일.png)로 넣는다. PDF가 없으면 초록이나 공개된 청구항·요약만 번역하고 맨 위에 "번역 범위: 초록만"을 적는다. 특허는 Google Patents 본문 텍스트를 써도 된다.
3) 요약.md를 기존 형식(예: ${REPO}/docs/hye/문헌연구/03_Y_T_분리/논문/12_Hoerl1970_Technometrics_능형회귀/요약.md 를 열어 본다)으로 쓴다: 제목, 원제, 원문 파일, 저자/출원인, 서지, 핵심(근거 상태 표시: 원문 확인 / 초록만*), 정량 수치, 우리와의 관계(관련도 ◎○△, 답하는 질문 번호), 링크. 그리고 "## 리뷰 카드" 절에 다음 네 줄을 쓴다: 어떻게 해결했나 / 같은 점 / 다른 점 / 가져올 해결책 (우리 장비의 구체 조건과 수치에 맞춰). 마지막에 "## 이 단계 후보 방법으로서" 절: 방법 이름, 전제 조건, 장점, 단점, 우리 적용 조건.`,
    { label: `doc:${u.code}:${(d.folder || '').split('/').pop()}`, phase: 'Docs', schema: DOCRES }))).then(r => ({ sel, docs: r.filter(Boolean) })),
  // C. stage review page
  (st, u) => agent(`${COMMON}
단계 "${u.title}"의 단계 페이지를 쓴다: ${unitDir(u)}/00_단계.md
자료: ${unitDir(u)}/조사방법론.md, ${unitDir(u)}/논문/*/요약.md, ${unitDir(u)}/특허/*/요약.md, 그리고 기존 문헌 ${JSON.stringify((st?.sel?.existing || []).map(e => e.path))}. 문헌 처리 결과: ${JSON.stringify(st?.docs || [])}
구성 (이 순서, Obsidian 마크다운):
1. "# ${u.title}" 그리고 캔버스 노드 "${u.node}" 블록 내용(목적, 해결할 질문, 사이 보정, 평가, 현재 상태)을 이 단위 초점에 맞게 다듬어 그대로 옮긴다.
2. "## 문헌 리뷰 표" 논문 표, 특허 표를 따로. 열: 번호 | 문헌(링크는 요약.md 상대경로) | 답하는 질문 | 방법 | 핵심 수치 | 우리 적용 | 근거 상태(원문/초록만/기존). 기존 문헌은 "기존" 표시와 상대 링크.
3. "## 문헌별 요약" 문헌마다 블록 하나: "### 번호 저자 연도 (관련도)" 아래 어떻게 해결했나 / 같은 점 / 다른 점 / 가져올 해결책.
4. "## 후보 방법 (문헌 기반 재작성)" AGENTS.md 1절 규칙대로 잠정 후보를 문헌 근거로 다시 쓴다. 표: 방법 | 출처(문헌 번호) | 전제 조건 | 장점 | 단점 | 우리 적용 조건 | 이미 한 실험. 근거 없는 잠정 후보는 "근거 없음"으로 남긴다. 이미 한 실험은 ${REPO}/experiments/hye 의 폴더 이름과 보고서를 찾아 연결한다.
5. "## 질문별 답 (현재까지)" 질문 번호마다 문헌과 기존 실험으로 지금 답할 수 있는 것, 아직 모르는 것.
마지막 응답으로 이 단계의 핵심을 5줄 이내로 돌려준다.`,
    { label: `review:${u.code}`, phase: 'Review' }).then(s => ({ ...st, review: s })),
  // D. experiment plan
  (st, u) => agent(`${COMMON}
단계 "${u.title}"의 실험 계획을 쓴다: ${unitDir(u)}/01_실험계획.md
근거: ${unitDir(u)}/00_단계.md (후보 방법 표, 질문별 답), ALGORITHM_RULES.md, 기존 실험 ${REPO}/experiments/hye (특히 20261029_* 측정 비교, 20261102_* A0~A16·B1~B6 계획서와 보고서, 20261102_142904_결론_센터복귀_반복정밀도.md).
원칙: 후보는 문헌 기반 표에서 고른다. 같은 자료, 같은 분할(기존 A0 계층 분할과 캐시 ${REPO}/experiments/hye/20261102_124309_A0_분할측정고정_계층분할과점측정캐시 를 우선 재사용, control 비교는 validation/splits.json 고정 4-fold, joint 비교는 341장 외부 4-fold), 정확도(축별 평균·최대오차, |e|≤0.045 통과율)와 처리 시간(ms)을 함께. 날짜를 나눈 평가를 반드시 포함. 기존 final 모듈을 대조군으로 둔다. 시험 정답 사용 금지.
오프라인(data/clean 정지 사진)으로 할 수 있는 실험과, 실기 촬영이 필요한 실험(촬영 요청서 형태: 몇 자세, 몇 회, 어떤 조건)을 나눈다. 오프라인 실험은 이 단계 질문에 직접 답하는 것 1~3개로 하되 각각 실험 ID(V2-${u.code}-E1 형식), 질문, 후보, 고정 조건, 지표, 판정 기준(실험 전에 고정), 예상 소요를 적는다. 이미 같은 조건으로 끝난 실험이면 재사용하고 새로 돌리지 않는다고 적는다.
문제 정의·명령·이동·정지처럼 오프라인으로 답할 수 없는 단계는 오프라인 실험을 0개로 두어도 된다(대신 체크리스트나 촬영 요청서를 쓴다).`,
    { label: `plan:${u.code}`, phase: 'Plan', schema: PLANRES }).then(p => ({ ...st, plan: p })),
  // E. run experiments
  (st, u) => (st?.plan?.offline?.length ? agent(`${COMMON}
단계 "${u.title}"의 오프라인 실험을 구현하고 실행한다. 계획: ${unitDir(u)}/01_실험계획.md (실험 ${JSON.stringify(st.plan.offline.map(e => e.id))}).
- 실험마다 ${REPO}/experiments/hye/20261104_HHMMSS_V2_${u.code}_<주제>/ 를 만든다(HHMMSS는 date +%H%M%S). 구조는 기존 실험과 같게: 계획서.md, 실험/(CMakeLists.txt, src/), 결과/(보고서.md, results/*.csv, 이미지/).
- C++17로 작성한다. 이 맥에서는 Homebrew OpenCV를 쓴다: cmake -S 실험 -B /tmp/hyebuild/<id> -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/opt/homebrew/opt/opencv (find_package(OpenCV REQUIRED), 버전 고정 금지, OpenMP 없이). OpenCV가 아직 설치 중이면 /opt/homebrew/opt/opencv/lib/cmake 가 생길 때까지 1분 간격으로 최대 40분 기다린다. 코드는 OpenCV 3.4 API 범위로 쓴다. 공통 코드 ${REPO}/experiments/hye/00001_common 과 A0 캐시를 재사용할 수 있다. 기존 폴더는 수정하지 않는다(필요하면 복사).
- 계산이 큰 경우 사진 수를 줄이지 말고 캐시를 활용한다. 한 실험 실행이 40분을 넘을 것 같으면 같은 분할의 대표 부분집합으로 먼저 돌리고 그 사실을 보고서에 적는다.
- 보고서.md: 문제, 해결 방법, 결과 표(대조군 포함, 정확도+시간), 판정(계획서의 판정 기준 그대로), 튀는 값, 한계, 빌드 환경(macOS, Homebrew OpenCV 버전). 결과를 본 뒤 기준을 바꿨으면 그 사실을 적는다.
- 마지막으로 ${unitDir(u)}/02_실험결과.md 를 쓴다: 실험별 한 줄 답, 결과 표, 질문별 답 갱신, control_v2와 joint_v2에 넣을지 권고(근거 수치).
실패한 실험도 실패 이유와 함께 보고한다.`,
    { label: `exp:${u.code}`, phase: 'Experiment', schema: EXPRES }) : agent(`${COMMON}
단계 "${u.title}"는 오프라인 실험이 없다. ${unitDir(u)}/02_실험결과.md 를 쓴다: 오프라인으로 답할 수 없는 이유, 실기 촬영 요청 ${JSON.stringify(st?.plan?.robot_needed || [])} 을 정리한 요청서, 문헌과 기존 실험으로 정한 control_v2/joint_v2 설계 권고.`,
    { label: `exp:${u.code}`, phase: 'Experiment', schema: EXPRES })).then(x => ({ code: u.code, dir: unitDir(u), review: st?.review, plan: st?.plan, exp: x, ndocs: (st?.docs || []).length }))
)

const done = results.filter(Boolean)
log(`단계 ${done.length}/${UNITS.length} 완료. 실패 단계: ${UNITS.filter(u => !done.find(d => d.code === u.code)).map(u => u.code).join(', ') || '없음'}`)

const SUMMARY = JSON.stringify(done.map(d => ({ code: d.code, dir: d.dir, findings: d.exp?.findings, rec_control: d.exp?.recommendation_control, rec_joint: d.exp?.recommendation_joint, failed: d.exp?.failed })))

const MODS = [
  { key: 'control', old: `${REPO}/experiments/hye/20261028_1327_final_Y_T control`, neo: `${REPO}/experiments/hye/00000_final_v3_Y_T control`, v2: `${REPO}/experiments/hye/00000_final_20261104_1435_v2_control_강등`, role: '2차 Y 정렬(1차 후 T≈0). 원래 Y-or-T 출력. v3는 같은 공개 API(initialize_control, control, alignment_diagnostics)와 CMake 대상 semes_control, assets/ 구조를 유지한다.' },
  { key: 'joint', old: `${REPO}/experiments/hye/20261028_1258_final_Y_T_joint`, neo: `${REPO}/experiments/hye/00000_final_v3_Y_T_joint`, v2: `${REPO}/experiments/hye/00000_final_20261104_1435_v2_joint_강등`, role: '1차 T 정렬(Y도 함께 추정하지만 운용은 T만 사용). v3는 같은 공개 API(initialize_joint, joint, alignment_diagnostics)와 CMake 대상 semes_joint, assets/ 구조를 유지한다.' },
]

phase('Design')
const designs = await parallel(MODS.map(m => () => agent(`${COMMON}
너는 ${m.key}_v3 설계 담당이다. 역할: ${m.role}
출발점: 메인이 먼저 만든 ${m.v2} (README와 근거 실험 A17~A21 보고서)를 읽고 그보다 나아지는 것을 목표로 한다.
입력: 단계별 실험 결과 요약 ${SUMMARY}
그리고 각 단계 폴더의 00_단계.md, 02_실험결과.md (${V2}/단계/ 아래), 기존 모듈 ${m.old} 의 README·소스·validation, ALGORITHM_RULES.md.
${V2}/설계_${m.key}_v3.md 를 쓴다:
1. 목표와 기존 대비 개선하려는 약점(수치 근거)
2. 단계별 선택표: 단계 | 기존 | v2 선택 | 근거(문헌 번호, 실험 ID와 수치) | 바꾸지 않은 이유(유지 시)
3. 처리 흐름(함수 단위), 진단 출력(측정 품질, 신뢰 판정, 범위 밖), 실패 처리
4. 검증 계획: 기존 final과 같은 사진·같은 분할(control: validation/splits.json 4-fold 280장 + 날짜 교차; joint: 341장 외부 4-fold + 날짜 교차 + 1029 자료), 지표(|e|≤0.045 통과율, 축별 평균·최대, ms), 회귀(나빠짐) 금지 기준
5. 위험과 대안
설명 가능성을 우선한다: 모든 선택에 "왜"를 한 문장으로 적는다. 근거가 약한 변경은 넣지 않는다.`,
  { label: `design:${m.key}`, phase: 'Design' })))

const critiques = await parallel(MODS.map(m => () => agent(`${COMMON}
${V2}/설계_${m.key}_v3.md 를 비판적으로 검토한다. 각 선택이 문헌·실험 수치로 실제 뒷받침되는지 원문 파일(실험 보고서, results csv)을 열어 확인하고, 데이터 누출, 과적합, 기존보다 나빠질 위험, 설명 불가능한 선택을 찾는다. 문제마다 수정안을 제시한다. 그리고 설계 문서를 직접 고쳐 "## 검토 반영" 절에 무엇을 바꿨는지 적는다.`,
  { label: `critique:${m.key}`, phase: 'Design' })))

phase('Build')
const BUILDRES = { type: 'object', properties: {
  folder: { type: 'string' }, built: { type: 'boolean' },
  table: { type: 'string', description: 'markdown table v1 vs v2 on same images/splits' },
  regressions: { type: 'array', items: { type: 'string' } }, notes: { type: 'string' } },
  required: ['folder', 'built', 'table', 'regressions', 'notes'] }
const builds = await parallel(MODS.map(m => () => agent(`${COMMON}
${m.key}_v3 를 구현하고 검증한다. 설계: ${V2}/설계_${m.key}_v3.md (검토 반영 포함). 기존: ${m.old} (수정 금지, 복사해서 시작해도 된다). 새 폴더: ${m.neo}
- ${m.role}
- C++17, OpenCV 3.4 API 범위(팀 기준 MSVC v141 + OpenCV 3.4.15 호환). 헤더 선언/구현 분리, .cpp include 금지. 학습이 필요하면 tool/ 아래 별도 학습 도구로 만들고 학습 결과를 assets/에 저장, 출처·해시를 assets/provenance.json에.
- 이 맥에서 Homebrew OpenCV로 빌드·검증한다(cmake -DCMAKE_PREFIX_PATH=/opt/homebrew/opt/opencv). 저장소 루트 CMake와 experiments/selected_modules.cmake 는 바꾸지 않는다. 모듈 폴더만 단독 빌드할 수 있게 하거나 별도 검증용 CMake를 둔다.
- 검증: 설계 4절의 계획대로 v1(기존), v2(메인이 먼저 만든 ${m.v2})와 v3를 같은 사진·같은 분할에서 돌려 results/v3_validation_20261104/ 에 summary.csv, predictions.csv, 시간 측정을 저장한다. v1 수치도 이 맥에서 다시 돌려 같은 환경 비교를 한다(불가하면 기존 기록값과 비교하고 이유를 적는다). 날짜 교차 평가를 포함한다. 시험 정답·파일명은 추론에 쓰지 않는다.
- README.md: 함수 계약, 처리 흐름, 각 선택의 근거(설계 문서 링크), 검증 결과의 의미와 한계(개발 노출 자료, 실기 미검증).
- 나빠진 지표가 있으면 숨기지 말고 regressions에 적는다.`,
  { label: `build:${m.key}`, phase: 'Build', schema: BUILDRES })))

phase('Verify')
const VER = { type: 'object', properties: { issues: { type: 'array', items: { type: 'object', properties: { severity: { type: 'string', enum: ['blocker', 'major', 'minor'] }, what: { type: 'string' }, evidence: { type: 'string' }, fix: { type: 'string' } }, required: ['severity', 'what', 'evidence', 'fix'] } } }, required: ['issues'] }
const lenses = [
  { k: 'leak', p: '데이터 누출과 평가 공정성: 시험 정답·파일명·시험 fold 자료가 학습·선택·정규화·후보 선택에 들어갔는지, 분할이 v1과 정말 같은지, 반복 사진이 같은 fold에 묶였는지' },
  { k: 'repro', p: '재현과 수치 확인: 직접 빌드하고 검증 도구를 다시 돌려 보고된 표의 수치가 재현되는지, 시간 측정이 공정한지' },
  { k: 'design', p: '설계와 코드의 일치, 강건성: README·설계 문서의 주장과 실제 코드가 같은지, 실패 처리(예외, (0,0) 숨김 금지), 범위 밖 입력, OpenCV 3.4 API 호환, 헤더/구현 분리' },
]
const verified = await parallel(MODS.map((m, i) => () => parallel(lenses.map(l => () => agent(`${COMMON}
${m.neo} (${m.key}_v3)를 검증한다. 관점: ${l.p}. 구현 보고: ${JSON.stringify(builds[i] || {})}
문제를 찾지 못하면 빈 목록을 돌려준다. 추측이 아니라 파일·실행 결과를 증거로 적는다.`,
  { label: `verify:${m.key}:${l.k}`, phase: 'Verify', schema: VER }))).then(vs => ({ m, issues: vs.filter(Boolean).flatMap(v => v.issues) }))))

const fixes = await parallel(verified.filter(Boolean).map(v => () => {
  const serious = v.issues.filter(x => x.severity !== 'minor')
  if (!serious.length && !v.issues.length) return Promise.resolve({ key: v.m.key, fixed: 'no issues' })
  return agent(`${COMMON}
${v.m.neo} 의 검증 지적을 처리한다: ${JSON.stringify(v.issues)}
blocker와 major는 반드시 고치고 검증을 다시 돌려 results/v3_validation_20261104/ 를 갱신한다. minor는 고치거나 README 한계 절에 적는다. 무엇을 고쳤는지와 최종 v1, v2, v3 표를 돌려준다.`,
  { label: `fix:${v.m.key}`, phase: 'Verify' }).then(t => ({ key: v.m.key, fixed: t }))
}))

phase('Report')
const report = await agent(`${COMMON}
최종 결과 보고서 ${V2}/결과보고.md 를 쓴다. 독자: 팀원과 사용자(쉬운 말).
구성: 1) 한 문단 결론(v3가 v1·v2보다 무엇이 나아졌고 무엇은 아직인지) 2) 단계별 한 줄 요약 표(단계 | 해결할 질문 | 문헌 수 | 실험 | 답 | v2 반영) 3) control_v3, joint_v3 각각: 무엇을 바꿨고 왜, v1, v2, v3 표(같은 사진·분할, 정확도·시간), 나빠진 것 4) 아직 모르는 것과 실기 촬영 요청(B1·B2·B5·B6, 도착 조건 사진) 5) 원문을 받지 못한 문헌 목록 링크(${V2}/원문_받을_목록.md).
입력: 단계 요약 ${SUMMARY}, 구현 ${JSON.stringify(builds)}, 수정 ${JSON.stringify(fixes)}, 설계 문서 ${V2}/설계_control_v3.md, ${V2}/설계_joint_v3.md, 각 모듈 README와 results.
마지막 응답으로 결론 요약 10줄을 돌려준다.`, { label: 'report', phase: 'Report' })

return { units: done.map(d => ({ code: d.code, ndocs: d.ndocs, plan: d.plan, exp: d.exp })), builds, fixes, report }
