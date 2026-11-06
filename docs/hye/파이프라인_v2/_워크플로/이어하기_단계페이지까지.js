export const meta = {
  name: 'semes-pipeline-v2-stage-pages',
  description: 'SEMES 파이프라인 v2 이어하기(단계 페이지까지만): 남은 9개 단위의 문헌 정리와 00_단계.md 작성. 실험계획·실험·v3는 돌리지 않는다.',
  phases: [
    { title: 'Research', detail: '이전 run의 선정 결과 불러오기' },
    { title: 'Docs', detail: '문헌별 폴더: 원문, 번역, 요약' },
    { title: 'Review', detail: '단계 페이지 00_단계.md' },
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
- 현재 모듈: ${REPO}/experiments/hye/00000_final_Y_T_joint (NCC→ECC 호모그래피 QR54 + 가로선 48블록 세로 프로파일 → 204차원 GLS), ${REPO}/experiments/hye/00000_final_Y_T control (NCC→LK15→17x17 스플라인 GN, 고정 536점, 센터 차감, P12, 6절편 후보 곡선 역산, Y-or-T).
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


const REMAIN = ['3', '6', '7', '8', '9', '10', '11', '2-4', '5b']
const TARGET = UNITS.filter(u => REMAIN.includes(u.code))

const results = await pipeline(TARGET,
  // A. load selection saved from previous run (research already done)
  (prev, u) => agent(`파일 ${SELDIR}/${u.code}.json 을 읽는다. folder와 existing.path는 저장소 기준 상대 경로이니 앞에 ${REPO}/ 를 붙여 절대 경로로 바꾼다. docs 중 폴더에 번역.md와 요약.md가 둘 다 이미 있는 항목은 빼고 나머지를 돌려준다(kind, folder, title, citation, url, pdf_url, questions, why). existing은 그대로 돌려준다. 다른 작업은 하지 않는다.`,
    { label: `load:${u.code}`, phase: 'Research', schema: SEL, effort: 'low' }),
  // B. per-doc fetch, translate, summarize
  (sel, u) => parallel((sel?.docs || []).map(d => () => agent(`${COMMON}
문헌 하나를 정리한다(이전 실행이 사용 한도로 중단됨. 폴더에 이미 PDF, 번역.md, 요약.md 일부가 있으면 확인해 재사용하고 빠진 것만 채운다). 단계 "${u.title}". 폴더: ${d.folder} (없으면 만든다)
문헌: [${d.kind}] ${d.title} / ${d.citation} / ${d.url} / PDF 후보: ${d.pdf_url || '없음'}
이 단계에서 답하려는 질문 번호: ${(d.questions || []).join(', ')} (질문 원문은 ${unitDir(u)}/조사방법론.md)
1) 원문 PDF를 curl로 받아 폴더에 저장한다(파일명 자유, 확장자 .pdf). 특허는 Google Patents 페이지의 PDF 링크(patentimages.storage.googleapis.com)를 쓴다. 받지 못하면 이유를 기록하고 ${V2}/원문_받을_목록.md 에 "| 단계 | 폴더 | 링크 | 이유 |" 형식 한 줄을 추가한다(파일이 없으면 표 머리와 함께 만든다). 봇 차단을 우회하지 않는다.
2) pdftotext -layout 으로 본문을 뽑아 번역.md를 쓴다. PDF가 있으면 전문 번역(초록, 본문, 그림·표 캡션, 부록. 참고문헌은 원문 그대로). 핵심 그림 1~3개는 pdftoppm -r 110 -png 로 해당 쪽을 이미지화해 그림/ (특허는 도면/)에 넣고 처음 언급되는 곳 근처에 ![](그림/파일.png)로 넣는다. PDF가 없으면 초록이나 공개된 청구항·요약만 번역하고 맨 위에 "번역 범위: 초록만"을 적는다. 특허는 Google Patents 본문 텍스트를 써도 된다.
3) 요약.md를 기존 형식(예: ${REPO}/docs/hye/문헌연구/03_Y_T_분리/논문/12_Hoerl1970_Technometrics_능형회귀/요약.md 를 열어 본다)으로 쓴다: 제목, 원제, 원문 파일, 저자/출원인, 서지, 핵심(근거 상태 표시: 원문 확인 / 초록만*), 정량 수치, 우리와의 관계(관련도 ◎○△, 답하는 질문 번호), 링크. 그리고 "## 리뷰 카드" 절에 다음 네 줄을 쓴다: 어떻게 해결했나 / 같은 점 / 다른 점 / 가져올 해결책 (우리 장비의 구체 조건과 수치에 맞춰). 마지막에 "## 이 단계 후보 방법으로서" 절: 방법 이름, 전제 조건, 장점, 단점, 우리 적용 조건.`,
    { label: `doc:${u.code}:${(d.folder || '').split('/').pop()}`, phase: 'Docs', schema: DOCRES }))).then(r => ({ sel, docs: r.filter(Boolean) })),
  // C. stage review page
  (st, u) => agent(`${COMMON}
단계 "${u.title}"의 단계 페이지를 쓴다: ${unitDir(u)}/00_단계.md
자료: ${unitDir(u)}/조사방법론.md, ${unitDir(u)} 아래 모든 논문/*/요약.md, 특허/*/요약.md (하위 폴더 포함, 이전 실행에서 이미 정리된 문헌도 전부 포함), 그리고 기존 문헌 ${JSON.stringify((st?.sel?.existing || []).map(e => e.path))}. 이번 실행에서 새로 처리한 문헌 결과: ${JSON.stringify(st?.docs || [])}. 이미 00_단계.md가 있으면 확인해 위 구성에 맞게 완성한다.
구성 (이 순서, Obsidian 마크다운):
1. "# ${u.title}" 그리고 캔버스 노드 "${u.node}" 블록 내용(목적, 해결할 질문, 사이 보정, 평가, 현재 상태)을 이 단위 초점에 맞게 다듬어 그대로 옮긴다.
2. "## 문헌 리뷰 표" 논문 표, 특허 표를 따로. 열: 번호 | 문헌(링크는 요약.md 상대경로) | 답하는 질문 | 방법 | 핵심 수치 | 우리 적용 | 근거 상태(원문/초록만/기존). 기존 문헌은 "기존" 표시와 상대 링크.
3. "## 문헌별 요약" 문헌마다 블록 하나: "### 번호 저자 연도 (관련도)" 아래 어떻게 해결했나 / 같은 점 / 다른 점 / 가져올 해결책.
4. "## 후보 방법 (문헌 기반 재작성)" AGENTS.md 1절 규칙대로 잠정 후보를 문헌 근거로 다시 쓴다. 표: 방법 | 출처(문헌 번호) | 전제 조건 | 장점 | 단점 | 우리 적용 조건 | 이미 한 실험. 근거 없는 잠정 후보는 "근거 없음"으로 남긴다. 이미 한 실험은 ${REPO}/experiments/hye 의 폴더 이름과 보고서를 찾아 연결한다.
5. "## 질문별 답 (현재까지)" 질문 번호마다 문헌과 기존 실험으로 지금 답할 수 있는 것, 아직 모르는 것.
마지막 응답으로 이 단계의 핵심을 5줄 이내로 돌려준다.`,
    { label: `review:${u.code}`, phase: 'Review' }).then(s => ({ ...st, review: s }))
)

// A pipeline result object can exist even when the review agent failed.
// This is an execution-status check; verify actual artifacts with
// tool/audit_pipeline_v2.ps1 before reporting document completion.
const failedResponse = /hit your .*limit|usage limit|rate limit|사용 한도|You've hit/i
const completed = TARGET.filter((u, i) => {
  const review = results[i]?.review
  return typeof review === 'string' && review.trim().length > 0 && !failedResponse.test(review)
})
const failed = TARGET.filter(u => !completed.some(c => c.code === u.code)).map(u => u.code)
log(`단계 응답 ${completed.length}/${TARGET.length}. 실패: ${failed.join(', ') || '없음'}. 실제 파일은 tool/audit_pipeline_v2.ps1로 별도 검증한다.`)
return { 응답완료: completed.length, 대상: TARGET.length, 실패: failed, 파일검증: '미실행' }
