# 정리본 v4: SEMES_01_데이터EDA_정리본_v3.pptx (읽기만) -> SEMES_01_데이터EDA_정리본_v4.pptx
#   13~15번 (④ 정리와 다음 데이터) 를 지금까지 분석 결과에 맞게 다시 검토해서 고침, 16번 부록 표지 부제 고침
#   모든 발표 메모: [제목] 묶음 사이 빈 줄 (v3_build.py 와 같은 규칙)
#   숫자 출처: v2/img/zero_offsets.csv, photo_motion.csv, common_resid.csv, brightness.csv,
#             analysis_grid_corners_20261019/out/nondiag/corner_step_px.csv, analysis_judge_compare_20261019/summary.csv
import re, pathlib, numpy as np, pandas as pd
from pptx import Presentation
HERE = pathlib.Path(__file__).parent; DECK = HERE.parent; IMG = HERE / 'img'; PROJ = DECK.parent
SRC, OUT = DECK / 'SEMES_01_데이터EDA_정리본_v3.pptx', DECK / 'SEMES_01_데이터EDA_정리본_v4.pptx'
E = 914400
prs = Presentation(SRC); S = list(prs.slides)
title_of = lambda s: next((sh.text_frame.text for sh in s.shapes if sh.has_text_frame and sh.top is not None and 0.8 < sh.top / E < 1.2 and not sh.text_frame.text.startswith('1-1')), '')
s13 = next(s for s in S if title_of(s).startswith('이번 데이터에서'))
s14 = next(s for s in S if title_of(s).startswith('다음 데이터: 필요한'))
s15 = next(s for s in S if title_of(s).startswith('다음 데이터: 분석할'))
s16 = next(s for s in S if any(sh.has_text_frame and sh.text_frame.text.startswith('측정 방법: QR 격자') for sh in s.shapes))

# ---- 숫자
Z = pd.read_csv(IMG / 'zero_offsets.csv').set_index('trial'); yz, tz = Z[Z.axis == 'Y'], Z[Z.axis == 'Θ']
St = pd.read_csv(PROJ / 'analysis_grid_corners_20261019' / 'out' / 'nondiag' / 'corner_step_px.csv'); st = St[St.kind == 'step'].groupby('axis').px.mean()
Rc = pd.read_csv(IMG / 'common_resid.csv').groupby('trial').common_px.std()
B = pd.read_csv(IMG / 'brightness.csv'); bc = B[B.file == 'center.bmp'].plate_mean.iloc[0]; bo = B[B.file != 'center.bmp'].plate_mean.mean()
J = pd.read_csv(PROJ / 'analysis_judge_compare_20261019' / 'summary.csv').set_index('method')
M = pd.read_csv(IMG / 'photo_motion.csv'); mm = M[M.trial != 'center']; kd = np.polyfit(mm.dx, mm.dy, 1)[0]
N = dict(
    y_mm=st['Y'] * 10, t_one=st['Θ'] * 100, resid=max(Z.resid_px.max(), 0), y_zero_px=yz.zero_px.max() - yz.zero_px.min(), y_zero_um=yz.zero_um.max() - yz.zero_um.min(),
    t_zero_px=tz.zero_px.max() - tz.zero_px.min(), t_zero=tz.zero_cmd.max() - tz.zero_cmd.min(), c_lo=Rc.min(), c_hi=Rc.max(),
    tilt=np.degrees(np.arctan(kd)), bright=100 * (bc / bo - 1), wrong=J.loc['corner_nondiag_P12', 'rep_wrong_per_round'],
    wrong_lo=J[J.index.str.endswith(('P12', 'affine18'))].rep_wrong_per_round.drop('corner_all_P12').min(),
    wrong_hi=J[J.index.str.endswith(('P12', 'affine18'))].rep_wrong_per_round.drop('corner_all_P12').max(),
    gain_gap=100 * abs(Z.slope_px['Θ -'] / Z.slope_px['Θ +'] - 1))
print({k: round(v, 4) for k, v in N.items()})

def replace_runs(s, mapping):
    hit = set()
    for sh in s.shapes:
        if not sh.has_text_frame: continue
        t = sh.text_frame.text
        if t in mapping:
            p = sh.text_frame.paragraphs[0]; p.runs[0].text = mapping[t]
            for r in p.runs[1:]: r.text = ''
            for extra in sh.text_frame.paragraphs[1:]:
                for r in extra.runs: r.text = ''
            hit.add(t)
    miss = set(mapping) - hit; assert not miss, miss

def set_notes(s, t): s.notes_slide.notes_text_frame.text = t

# ---- 13번: 알게 된 것 / 확인할 수 없는 것
replace_runs(s13, {
    '깨끗한 사진': '깨끗한 사진', '흑백 4K, 노이즈 0.73': '흑백 4K 무압축, 노이즈 0.73',
    '거의 직선': '거의 직선', 'Y 1mm 14px, Θ 1 80px': f'Y 1mm {N["y_mm"]:.2f}px, Θ 1 {N["t_one"]:.1f}px',
    '정답에 오차': '0점 어긋남', '0점이 최대 44µm 다름': f'trial끼리 Y {N["y_zero_um"]:.0f}µm, Θ {N["t_zero"]:.3f}',
    '0 근처가 어려움': '사진 공통 흔들림', '|Y| 0.5mm 이하 구분 안 됨': f'{N["c_lo"]:.3f}~{N["c_hi"]:.3f}px, 판정 한계',
    '판 위 단서가 작음': '0 근처 판정', '판 위 점끼리 비율이 같음': f'280장 중 약 {N["wrong"]:.0f}장 틀림',
    '다른 현장': '다른 현장', '다른 QR, 다른 장비': '다른 QR 내용, 다른 장비',
})
set_notes(s13, f'''[알게 된 것]
- 깨끗한 사진: 3840x2160 무압축 BMP, R=G=B 흑백, 센서 잡음 0.73 (밝기 단계)
- 거의 직선: 모서리 630점 기준 한 칸 평균에서 Y 1mm = {N["y_mm"]:.2f}px, Θ 1 = {N["t_one"]:.1f}px. trial마다 직선으로 맞추고 남은 값 σ {N["resid"]:.3f}px 이하
- 0점 어긋남: trial마다 명령값 0의 위치가 다름. Y trial끼리 {N["y_zero_px"]:.2f}px = {N["y_zero_um"]:.0f}µm, Θ trial끼리 {N["t_zero_px"]:.2f}px = Θ {N["t_zero"]:.4f} (6쪽)
- 사진 공통 흔들림: 사진 전체가 같이 흔들린 양 σ {N["c_lo"]:.3f}~{N["c_hi"]:.3f}px. 점을 많이 평균해도 안 줄어들어 판정 한계의 주원인 (12쪽)
- 0 근처 판정: 측정 방법을 바꿔도 반복 5분할에서 회당 {N["wrong_lo"]:.1f}~{N["wrong_hi"]:.1f}장 틀림 (부록 5). 틀리는 사진은 0 근처 작은 이동
- 그 밖에: 이동 방향이 화면에서 {N["tilt"]:.2f}° 기울어짐, Θ +쪽과 -쪽 기울기 {N["gain_gap"]:.1f}% 차이, center만 밝기 {N["bright"]:+.1f}%

[확인할 수 없는 것]
- 반복성: 같은 자세를 다시 찍었을 때의 흔들림 (사진 공통 흔들림이 카메라 탓인지 로봇 탓인지)
- 두 축 동시 이동: Y와 Θ가 같이 틀어진 사진이 없음 (십자 구조)
- 환경 변화: 조명, 온도, 진동, 시간
- 다른 현장: QR 내용이 다른 위치, 다른 장비
- Θ 단위와 합격선: 도(°) 추정, 허용 오차 0.09의 해석 (SEMES 확인 필요)

[숫자 출처]
presentation_eda_20261019/v2/v4_build.py 가 아래 파일에서 계산
v2/img/zero_offsets.csv, common_resid.csv, brightness.csv, photo_motion.csv
analysis_grid_corners_20261019/out/nondiag/corner_step_px.csv
analysis_judge_compare_20261019/summary.csv''')

# ---- 14번: 필요한 것 (카드 내용은 유지, F 설명만 QR 내용 강조), 메모에 요청 이유
replace_runs(s14, {'다른 ADDRESS의 QR에서 9자세': '다른 ADDRESS(다른 QR 내용)에서 9자세'})
set_notes(s14, f'''[요청 이유 (이번 분석과 연결)]
- A 정지 반복 촬영: 사진 공통 흔들림 {N["c_lo"]:.3f}~{N["c_hi"]:.3f}px가 카메라 잡음인지 로봇 흔들림인지 가르기
- B 원점 복귀 반복: 0점 어긋남(Y {N["y_zero_um"]:.0f}µm, Θ {N["t_zero"]:.4f})이 복귀 방향마다 재현되는지, 백래시 크기
- C 혼합 격자: 지금은 십자 구조라 Y와 Θ가 같이 틀어진 경우를 시험할 수 없음
- D 왕복 스윕: Θ +쪽과 -쪽 기울기 {N["gain_gap"]:.1f}% 차이, 방향 바뀐 직후 한 칸이 작아지는 현상(Y -1.85→-1.95)이 반복되는지
- E 환경 변화: center만 밝기 {N["bright"]:+.1f}%, trial 안에서도 밝기가 변함. 조명, 예열, 진동, 초점 영향
- F 다른 위치, 다른 장비: QR 내용이 바뀌어도 모서리 630점 방식이 그대로 통하는지

[요청서 원본]
00_데이터요청서.md''')

# ---- 15번: 분석할 것 표의 '알고 싶은 것' 열을 이번 결과와 연결
tbl = next(sh for sh in s15.shapes if sh.has_table).table
want = [f'사진 공통 흔들림 {N["c_lo"]:.3f}~{N["c_hi"]:.3f}px의 원인', f'0점 어긋남({N["y_zero_um"]:.0f}µm)이 재현되나', '두 축이 섞여도 판정이 맞나',
        f'+쪽 -쪽 기울기 차({N["gain_gap"]:.1f}%), 이력', f'밝기({N["bright"]:+.1f}%), 온도 영향', '다른 QR에서도 630점 방식이 되나']
for r, w in zip(list(tbl.rows)[1:], want):
    p = r.cells[2].text_frame.paragraphs[0]; p.runs[0].text = w
    for x in p.runs[1:]: x.text = ''
for sh in s15.shapes:   # 강조 문장: 앞 '4번이 가장 중요합니다.' 는 파랑 굵게, 뒤는 원래 회색 그대로 (두 run 유지)
    if sh.has_text_frame and sh.text_frame.text.startswith('4번이 가장 중요합니다.'):
        rr = sh.text_frame.paragraphs[0].runs; assert len(rr) >= 2
        rr[1].text = f'지금 성적(회당 약 {N["wrong"]:.0f}장 틀림)은 같은 280장 안에서 나눠 본 결과라서, 처음 보는 데이터에서 다시 재야 진짜 성능을 알 수 있습니다.'
set_notes(s15, f'''[공통으로 먼저]
1. 형식, 노이즈가 이번과 같은지 (해상도, 무압축, 센서 잡음 0.73)
2. 촬영 순서와 시각 (드리프트, 공백, 거꾸로 찍은 구간)
3. 라벨 범위, 간격, Θ 단위
4. 지금 모델을 그대로 넣은 정확도. 지금 성적은 같은 280장 안의 교차검증이라 새 데이터가 진짜 시험

[데이터별로 알고 싶은 것]
- A: 사진 공통 흔들림 {N["c_lo"]:.3f}~{N["c_hi"]:.3f}px의 원인 (카메라 vs 로봇)
- B: 0점 어긋남 {N["y_zero_um"]:.0f}µm가 복귀 방향마다 재현되는지
- C: 두 축이 섞여도 판정이 맞는지
- D: +쪽 -쪽 기울기 차 {N["gain_gap"]:.1f}%, 이력
- E: 밝기 변화 {N["bright"]:+.1f}%, 온도
- F: 다른 QR 내용에서도 모서리 630점 방식이 되는지''')

# ---- 16번 부록 표지
replace_runs(s16, {'측정 방법: QR 격자 모서리 맞춤, 측정 방법 4가지 비교, 판정 정확도': '측정 방법: QR 격자 모서리 맞춤, 측정 방법 4가지 비교, 판정 정확도, 검증 (기준점, ECC 패치)'})

# ---- 모든 메모 형식
for s in prs.slides:
    if not s.has_notes_slide or s.notes_slide.notes_text_frame is None: continue
    t0 = s.notes_slide.notes_text_frame.text
    t1 = re.sub(r',\s*(\[[가-힣 ,()]+\])', r'\n\1', t0)
    t1 = re.sub(r'^(\[[가-힣 ,()]+\])[ \t]+', r'\1\n', t1, flags=re.M)
    out = []
    for l in [l.rstrip() for l in t1.split('\n')]:
        if re.fullmatch(r'\[[가-힣 ,()]+\]', l) and out and out[-1] != '': out.append('')
        if l == '' and (not out or out[-1] == ''): continue
        out.append(l)
    t1 = '\n'.join(out).strip()
    if t1 != t0: s.notes_slide.notes_text_frame.text = t1
prs.save(OUT); print('saved', OUT.name, len(prs.slides))
