"""PPT 슬라이드용 데이터: 짝 기준 7가지(좌, 중, 우, 좌중, 좌우, 중우, 좌중우)의 v1 방식 판정과
다른 처리 방법들(v2, v3)의 판정을 같은 형식으로 모은다.

형식: dict(name, basis, values, rule, flagY{파일: 불가/가능}, flagT{파일: 불가/가능/범위 밖}, twinY{Y파일: 짝 Θ}, twinT{Θ파일: 짝 Y})
"""
import numpy as np, pandas as pd
from pathlib import Path
HERE = Path(__file__).parent; PROJ = HERE.parent
SCR = Path('C:/Users/SSAFY/AppData/Local/Temp/claude/C--Users-SSAFY-Desktop-hwlee-projects-2026-10-SEMES/0478c349-0ea5-4fa5-b878-4da27eb190c2/scratchpad')
q = pd.read_csv(PROJ/'14_데이터_QR마커위치.csv'); q['ax'] = q.file.str[0]
c = q[q.ax == 'c']
Ys = pd.concat([c, q[q.ax == 'Y']]).sort_values('value').reset_index(drop=True)
Ts = pd.concat([c, q[q.ax == 'T']]).sort_values('value').reset_index(drop=True)
Q = ['dLx', 'dCx', 'dRx']
TH = np.array([0.0449, 0.0274, 0.0478])          # v1 임계값 (같은 축 보간 잔차 3σ, 두 시트 중 큰 값) - 22번 엑셀 설명 시트와 같음
BASES = {'좌': ['dLx'], '중': ['dCx'], '우': ['dRx'], '좌중': ['dLx', 'dCx'], '좌우': ['dLx', 'dRx'], '중우': ['dCx', 'dRx'], '좌중우': ['dLx', 'dCx', 'dRx']}

def key(s, cols): return s[cols].mean(1).values

def pairing(basis):
    """짝 기준 basis로 v1 판정. 상대 시트 곡선에서 basis 값이 같은 지점을 보간하고, 좌, 중, 우 가로 px 차이로 판정."""
    cols = BASES[basis]; out = dict(flagY={}, flagT={}, twinY={}, twinT={})
    for s, ref, fl, tw in [(Ys, Ts, 'flagY', 'twinY'), (Ts, Ys, 'flagT', 'twinT')]:
        m = key(ref, cols); ms = key(s, cols); o = np.argsort(m); m = m[o]; R = ref.iloc[o]
        for i, r in s.iterrows():
            if r.file == 'center.bmp': continue
            if ms[i] < m.min() or ms[i] > m.max(): out[fl][r.file] = '범위 밖'; continue
            d = np.array([r[k] - np.interp(ms[i], m, R[k].values) for k in Q])
            out[fl][r.file] = '불가' if (np.abs(d) <= TH).all() else '가능'
            out[tw][r.file] = float(np.interp(ms[i], m, R.value.values))
    return out

def pairing_best():
    """최적 짝: 상대 축 곡선 위에서 좌, 중, 우 차이가 임계값 대비 가장 작아지는 지점(가장 닮은 지점)을 짝으로 삼는다.
    가장 불리한(헷갈리기 쉬운) 짝이라 '구분 불가'를 가장 엄격하게 센다."""
    out = dict(flagY={}, flagT={}, twinY={}, twinT={})
    for s, ref, fl, tw in [(Ys, Ts, 'flagY', 'twinY'), (Ts, Ys, 'flagT', 'twinT')]:
        o = np.argsort(ref.value.values); R = ref.iloc[o]
        g = np.linspace(R.value.min(), R.value.max(), 20001)                 # 상대 축 값을 촘촘히
        curves = np.stack([np.interp(g, R.value.values, R[k].values) for k in Q], 1)
        lo, hi = curves.mean(1).min(), curves.mean(1).max()
        for i, r in s.iterrows():
            if r.file == 'center.bmp': continue
            if r[Q].mean() < lo or r[Q].mean() > hi: out[fl][r.file] = '범위 밖'; continue
            sc = (np.abs(r[Q].values.astype(float) - curves) / TH).max(1); j = sc.argmin()
            out[fl][r.file] = '불가' if sc[j] <= 1 else '가능'; out[tw][r.file] = float(g[j])
    return out

def twins_lcr():
    p = pairing('좌중우'); return p['twinY'], p['twinT']

def from_flags(fy, ft):
    ty, tt = twins_lcr(); return dict(flagY=fy, flagT=ft, twinY=ty, twinT=tt)

def methods():
    """다른 처리 방법들. 짝 기준은 모두 좌중우 평균으로 고정."""
    M = {}
    J = pd.read_pickle(PROJ/'analysis_v2/judge.pkl'); res = J['res']
    for key_, lab in [('v2-b (QR 네모 가로, 세로, 폭, 높이 12개)', 'v2-b'), ('v2-a1 (QR별 내부 픽셀 ECC affine 18개)', 'v2-a1'),
                      ('v2-a2 (판 전체 원근 변환 14개)', 'v2-a2')]:
        r = res[key_]
        fy = {f: v for f, v in zip(J['Ys'].file, r['out']['Y'].규칙2) if f != 'center.bmp'}
        ft = {f: ('범위 밖' if v == '범위 밖' else v) for f, v in zip(J['Ts'].file, r['out']['Θ'].규칙2) if f != 'center.bmp'}
        M[lab] = from_flags(fy, ft)
    V = pd.read_pickle(HERE/'v3.pkl'); names = V['names']; X = V['X']; per = V['per']
    for (fn, an), lab in [(('Q7 QR 내부 특징점 (263개)', 'C5 야코비안 가중(GLS)'), 'Q7+C5'),
                          (('G1 QR 판 픽셀 전체 (특징점 없음)', 'G1 픽셀 직접 가우스-뉴턴 (Θ, Y + 방해 3)'), 'G1'),
                          (('O3 조합B2 브래킷 윗판+아래 (2개)', 'C5 야코비안 가중(GLS)'), 'O3+C5')]:
        est, can = per[(fn, an)]; fy, ft = {}, {}
        for i, n in enumerate(names):
            if n == 'center.bmp': continue
            if X[i, 0] == 0: fy[n] = '가능' if can[i] else '불가'
            else: ft[n] = ('가능' if can[i] else '불가') if V['inrange'][i] else '범위 밖'
        M[lab] = from_flags(fy, ft)
        M[lab]['est'] = {n: est[i] for i, n in enumerate(names)}
    return M

def val(f): return 0.0 if f == 'center.bmp' else float(f[1:-4])

def summary(d):
    """세트 수와 경계. 가운데부터 연속으로 전부 구분 불가인 범위, 그 바깥으로 전부 구분 가능해지는 시작점."""
    out = {}
    for ax, fl in [('Y', d['flagY']), ('Θ', d['flagT'])]:
        items = sorted(((abs(val(f)), v) for f, v in fl.items() if v != '범위 밖'))
        n_bad = sum(v == '불가' for _, v in items)
        mags = sorted(set(a for a, _ in items))
        allbad = 0.0
        for a in mags:
            if all(v == '불가' for b, v in items if b == a): allbad = a
            else: break
        allgood = None
        for a in reversed(mags):
            if all(v == '가능' for b, v in items if b == a): allgood = a
            else: break
        out[ax] = dict(n_bad=n_bad, n=len(items), allbad=allbad, allgood=allgood,
                       maxbad=max([a for a, v in items if v == '불가'], default=0.0))
    return out

if __name__ == '__main__':
    for b in list(BASES) + ['최적']:
        s = summary(pairing(b) if b != '최적' else pairing_best())
        print(f'{b:4s} Y 불가 {s["Y"]["n_bad"]:2d}/{s["Y"]["n"]} (|Y|≤{s["Y"]["allbad"]:.2f} 전부 불가, ≥{s["Y"]["allgood"]:.2f} 전부 가능, 불가 최대 {s["Y"]["maxbad"]:.2f}) | '
              f'Θ 불가 {s["Θ"]["n_bad"]:2d}/{s["Θ"]["n"]} (|Θ|≤{s["Θ"]["allbad"]:.2f} 전부 불가, ≥{s["Θ"]["allgood"]:.2f} 전부 가능)')
    for k, d in methods().items():
        s = summary(d)
        print(f'{k:6s} Y 불가 {s["Y"]["n_bad"]:2d}/{s["Y"]["n"]} (|Y|≤{s["Y"]["allbad"]:.2f} 전부 불가, ≥{s["Y"]["allgood"]:.2f} 전부 가능, 불가 최대 {s["Y"]["maxbad"]:.2f}) | '
              f'Θ 불가 {s["Θ"]["n_bad"]:2d}/{s["Θ"]["n"]} (|Θ|≤{s["Θ"]["allbad"]:.2f} 전부 불가, ≥{s["Θ"]["allgood"]:.2f} 전부 가능)')
