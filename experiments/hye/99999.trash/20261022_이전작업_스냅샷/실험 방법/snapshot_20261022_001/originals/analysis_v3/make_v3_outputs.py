import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from pathlib import Path
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.workbook.properties import CalcProperties
HERE = Path(__file__).parent; PROJ = HERE.parent
V = pd.read_pickle(HERE/'v3.pkl'); R, per, names, X = V['R'], V['per'], V['names'], V['X']
twinT, isY, inrange = V['twinT'], V['isY'], V['inrange']
_g = R.특징점.str.startswith('G1'); _o = R.영역 != 'QR만'
R = pd.concat([R[~_g & ~_o], R[_g], R[_o]])                          # G1(QR만)을 QR만 구역 끝으로
best = R.loc[R.groupby('특징점', sort=False).Y_RMS.idxmin()]
qbest = best[best.영역 == 'QR만'].sort_values('Y_RMS').iloc[0]; obest = best[best.영역 != 'QR만'].sort_values('Y_RMS').iloc[0]

# ---------- 그림: 조합 x 알고리즘 Y RMS ----------
plt.rcParams['font.family'] = 'Malgun Gothic'; plt.rcParams['axes.unicode_minus'] = False
combos = list(dict.fromkeys(R.특징점)); algs = list(dict.fromkeys(R.알고리즘))
M = np.full((len(combos), len(algs)), np.nan); Nb = np.full_like(M, np.nan)
for _, r in R.iterrows():
    i, j = combos.index(r.특징점), algs.index(r.알고리즘); M[i, j] = r.Y_RMS * 1000; Nb[i, j] = r.Y불가
fig, a = plt.subplots(figsize=(15, 8.6))
Mc = np.log10(np.clip(M, 10, 1000))
a.imshow(Mc, cmap='Blues_r', vmin=1, vmax=3, aspect='auto')
for i in range(len(combos)):
    for j in range(len(algs)):
        if np.isnan(M[i, j]): a.text(j, i, '-', ha='center', va='center', color='#9a998f'); continue
        dark = Mc[i, j] < 1.9
        a.text(j, i, f'{M[i,j]:.0f}µm\n불가 {int(Nb[i,j])}', ha='center', va='center', fontsize=9, color='white' if dark else '#1f1f1e')
a.set_xticks(range(len(algs))); a.set_xticklabels([s.replace(' (', '\n(') for s in algs], fontsize=9)
a.set_yticks(range(len(combos))); a.set_yticklabels(combos, fontsize=9.5)
o1 = combos.index([c for c in combos if c.startswith('O1')][0])
a.axhline(o1 - 0.5, color='#eb6834', lw=2)
a.text(-0.45, o1 - 0.42, 'QR만 (위)  /  QR 밖 포함 (아래)', ha='left', va='bottom', color='#eb6834', fontsize=10)
a.set_title('특징점 조합 x 알고리즘: Y 오차 RMS (짙을수록 좋음) / Y 사진 80장 중 구분 불가 수', loc='left', fontsize=13, pad=10)
a.set_xticks(np.arange(-.5, len(algs)), minor=True); a.set_yticks(np.arange(-.5, len(combos)), minor=True)
a.grid(which='minor', color='white', lw=2); a.tick_params(which='minor', length=0)
for s in a.spines.values(): s.set_visible(False)
fig.text(0.01, 0.01, '짝/홀 2겹 교차검증 (어제 17번 결과표와 같은 분할). 구분 불가 = 추정 Θ가 짝의 Θ에서 3σ 안. 허용 오차 90µm. 색은 로그 눈금.', fontsize=9, color='#5f5e58')
plt.tight_layout(rect=(0, 0.03, 1, 1)); plt.savefig(PROJ/'26_v3_조합x알고리즘_히트맵.png', dpi=130, facecolor='white')

# ---------- 엑셀 ----------
F = 'Arial'; base = Font(name=F, size=10); bold = Font(name=F, size=10, bold=True)
hdr = Font(name=F, size=10, bold=True, color='FFFFFF'); hfill = PatternFill('solid', fgColor='1F4E78')
bad = PatternFill('solid', fgColor='FCE4D6'); good = PatternFill('solid', fgColor='E2EFDA'); thin = Side(style='thin', color='D9D9D9')

def header(ws, r, heads, widths):
    for j, h in enumerate(heads, 1):
        c = ws.cell(r, j, h); c.font = hdr; c.fill = hfill
        c.alignment = Alignment(horizontal='center', vertical='center', wrap_text=True)
    for j, w in enumerate(widths, 1): ws.column_dimensions[ws.cell(r, j).column_letter].width = w
    ws.row_dimensions[r].height = 42

def cv(v): return int(v) if isinstance(v, (np.integer,)) else (float(v) if isinstance(v, np.floating) else v)

cols = [('영역', '영역', 10), ('특징점', '특징점 조합', 40), ('알고리즘', '알고리즘', 34), ('T_RMS', 'Θ 오차 RMS', 10), ('T_최대', 'Θ 오차 최대', 10),
        ('Y_평균', 'Y 오차 평균 (mm)', 10), ('Y_RMS', 'Y 오차 RMS (mm)', 10), ('Y_최대', 'Y 오차 최대 (mm)', 10), ('Y_009안', 'Y 0.09mm 안 (%)', 10),
        ('Y_005안', 'Y 0.05mm 안 (%)', 10), ('Y불가', 'Y 사진 구분 불가 (/80)', 11), ('Θ불가', 'Θ 사진 구분 불가 (/71)', 11),
        ('오판율', '같은 축 오판율', 10), ('σΘ', 'σ (Θ, 판정 기준)', 10)]
fmt = {'T_RMS': '0.0000', 'T_최대': '0.0000', 'Y_평균': '0.000', 'Y_RMS': '0.000', 'Y_최대': '0.000', 'Y_009안': '0.0', 'Y_005안': '0.0', '오판율': '0.0%', 'σΘ': '0.0000'}

def table(ws, r0, df, bold_idx):
    header(ws, r0, [c[1] for c in cols], [c[2] for c in cols])
    for i, (ix, r) in enumerate(df.iterrows()):
        for j, (k, _, _) in enumerate(cols, 1):
            c = ws.cell(r0 + 1 + i, j, cv(r[k])); c.font = bold if ix in bold_idx else base; c.border = Border(bottom=thin)
            if k in fmt: c.number_format = fmt[k]
            if k == 'Y_RMS' and r[k] <= 0.09: c.fill = good

wb = Workbook(); S = wb.active; S.title = '요약'
S['A1'] = 'Y, Θ 구분표 v3: 어제 특징점 조합 13개 + 판 원근 변환(Q5b) x 알고리즘 C1~C6, 픽셀 직접 가우스-뉴턴(G1)'; S['A1'].font = Font(name=F, size=13, bold=True)
S['A2'] = (f'QR만 최고: {qbest.특징점} + {qbest.알고리즘} (Y RMS {qbest.Y_RMS:.3f}mm, 구분 불가 Y {qbest.Y불가}/80).   '
           f'QR 밖 포함 최고: {obest.특징점} + {obest.알고리즘} (Y RMS {obest.Y_RMS:.3f}mm, 구분 불가 Y {obest.Y불가}/80).'); S['A2'].font = base
S['A3'] = '굵은 행 = 조합마다 Y 오차가 가장 작은 알고리즘. 초록 = Y 오차 RMS가 허용 오차 0.09mm 이하. 순서는 어제 결과표와 같음. 판정 규칙과 한계는 설명 시트.'; S['A3'].font = base
table(S, 5, R, set(best.index)); S.freeze_panes = 'D6'
B2 = wb.create_sheet('조합별 최고'); table(B2, 1, best, set())

KEY = [('Q1 QR 3개 평균 이동량 (1개)', 'C1 이동량÷계수 (Y=0 가정)', 'Q1+C1 (기준선)'),
       ('Q7 QR 내부 특징점 (263개)', 'C5 야코비안 가중(GLS)', 'Q7+C5 (어제 QR만 최고)'),
       (qbest.특징점, qbest.알고리즘, f'{qbest.특징점.split(" ")[0]}+{qbest.알고리즘.split(" ")[0]} (오늘 QR만 최고)'),
       ('G1 QR 판 픽셀 전체 (특징점 없음)', 'G1 픽셀 직접 가우스-뉴턴 (Θ, Y + 방해 3)', 'G1 (픽셀 직접 가우스-뉴턴)'),
       ('O3 조합B2 브래킷 윗판+아래 (2개)', 'C5 야코비안 가중(GLS)', 'O3+C5 (어제 추천, QR 밖)'),
       (obest.특징점, obest.알고리즘, f'{obest.특징점.split(" ")[0]}+{obest.알고리즘.split(" ")[0]} (QR 밖 최고)')]
seen, K2 = set(), []
for k in KEY:
    if (k[0], k[1]) not in seen: seen.add((k[0], k[1])); K2.append(k)
KEY = K2
qb = per[(qbest.특징점, qbest.알고리즘)][1]
for sh, mask in [('Y', isY), ('Θ', (X[:, 1] == 0) & (X[:, 0] != 0))]:
    ws = wb.create_sheet(sh); idx = np.where(mask)[0]; idx = idx[np.argsort(X[idx, 1 if sh == 'Y' else 0])]
    heads = ['인덱스', '파일명', f'값 ({"mm" if sh == "Y" else "Θ"})', '짝의 Θ (보간)']
    for _, _, lab in KEY: heads += [f'{lab}\n추정 Θ', '추정 Y (mm)', '판정']
    header(ws, 1, heads, [7, 13, 9, 10] + [12, 10, 9] * len(KEY)); ws.freeze_panes = 'E2'
    for rr, i in enumerate(idx):
        R_ = rr + 2; still = (not np.isnan(twinT[i])) and (not qb[i])
        vals = [rr + 1, names[i], float(X[i, 1] if sh == 'Y' else X[i, 0]), None if np.isnan(twinT[i]) else float(twinT[i])]
        fm = [None, None, '0.00', '0.0000']
        for fn, an, _ in KEY:
            est, can = per[(fn, an)]
            j = '범위 밖' if np.isnan(twinT[i]) else ('가능' if can[i] else '불가')
            vals += [float(est[i, 0]), float(est[i, 1]), j]; fm += ['0.0000', '0.000', None]
        for j, (v, f_) in enumerate(zip(vals, fm), 1):
            c = ws.cell(R_, j, v); c.font = bold if still else base; c.border = Border(bottom=thin)
            if v == '불가': c.fill = bad
            if f_: c.number_format = f_
    ws.cell(len(idx) + 3, 1, f'굵은 행 = 오늘 QR만 최고 방법({KEY[2][2]})으로도 구분 불가. 주황 칸 = 그 방법에서 구분 불가.').font = base

E = wb.create_sheet('설명')
for k, t in enumerate([
    '데이터와 분할: 어제 17번 결과표와 완전히 같은 사진별 측정값(analysis_v3/v1_data)과 같은 짝/홀 2겹 교차검증. 어제 표 49행을 소수점까지 그대로 재현함을 확인함.',
    'C1~C5: 어제 알고리즘 그대로. C4, C5는 직선 모델을 한 번에 푸는 것으로, 가우스-뉴턴 1회 반복과 같음.',
    'C6 (새로 추가): 특징점 조합용 가우스-뉴턴. 곡선 성분(Θ², ΘY, Y²)까지 넣은 모델을 학습 사진으로 맞추고, C5 해에서 출발해 GLS 가중 가우스-뉴턴으로 반복. 오차가 줄 때만 이동(감쇠)해서 발산을 막음.',
    'Q5b (새로 추가): QR 3개 판 전체를 원근 변환으로 맞춘 뒤 판 귀퉁이 4개와 QR 중심 3개의 이동(14개). 어제 Q5와 같은 변환이지만 값의 표현을 바꿔 계산이 안정적.',
    'G1 (새로 추가): 픽셀 직접 가우스-뉴턴. 특징점 없이 QR 판 영역 픽셀 전부로 Θ, Y를 직접 푼다. 학습 사진 평균을 기준 영상으로 쓰고, Θ, Y와 섞이지 않는 방해 요소 3개(세로 이동, 화면 내 회전, 크기)를 함께 푼다. 방해 요소 없이 Θ, Y만 풀면 크게 틀림(예: Y+0.10을 Y 0.83으로).',
    '구분 판정(규칙3): Y 사진의 짝은 QR 가로 평균 위치가 같은 Θ 지점(보간), Θ 사진의 짝은 Θ=0인 Y 지점. 추정 Θ가 짝의 Θ에서 3σ 넘게 떨어지면 구분 가능. σ = 그 방법의 Θ 추정 오차 표준편차(정답값 오차 포함이라 보수적).',
    '주의: 한 가지 답만 내는 방법(예: Q1+C1은 항상 "Θ"라고 답함)은 Θ 사진 쪽 판정을 쉽게 통과함. 그래서 Y 사진 쪽 구분 불가 수가 더 의미 있는 지표.',
    '오판율: 같은 축의 정답 Θ에서 3σ 넘게 벗어난 비율. 낮을수록 판정 기준을 믿을 수 있음.',
    '한계: 짝/홀 분할은 같은 촬영 구간의 이웃 사진이 학습과 채점에 섞여 실제보다 좋게 나올 수 있음. 정답값 자체에 약 40µm 오차.',
    '코드: analysis_v3/common.py(어제 코드 재사용), gn_features.py(C6), gn_pixel.py와 run_g1.py(G1), run_v3.py(평가), make_v3_outputs.py(출력)']):
    E.cell(k + 1, 1, t).font = base
E.column_dimensions['A'].width = 180
wb.calculation = CalcProperties(fullCalcOnLoad=True)
wb.save(PROJ/'25_YΘ구분표_v3.xlsx')
print('ok |', qbest.특징점, '+', qbest.알고리즘, '|', obest.특징점, '+', obest.알고리즘)
