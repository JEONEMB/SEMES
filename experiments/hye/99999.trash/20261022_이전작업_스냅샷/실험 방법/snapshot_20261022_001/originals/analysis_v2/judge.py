"""버전별 Y/Θ 구분 판정. 짝 기준(같은 QR 위치)은 v1과 같이 QR 3개 가로 평균 px.
규칙1: 값마다 3σ 임계값 (v1 방식)  규칙2: 마할라노비스 거리 (v2-c, 잡음 공분산 반영, 99.73%)"""
import numpy as np, pandas as pd
from scipy.stats import chi2
from pathlib import Path
B = Path(__file__).parent.parent
q = pd.read_csv(B/'14_데이터_QR마커위치.csv')
f15 = pd.read_csv(B/'15_데이터_파인더_볼트이동량.csv')[['file','finder_dx','finder_dy','bolt_dx']]
ecc = pd.read_csv(B/'analysis_agent/qr_measurements.csv'); ecc = ecc[[c for c in ecc.columns if c=='file' or (c[:2] in ('q0','q1','q2') and not c.endswith(('cc','match')))]]
hom = pd.read_csv(Path(__file__).parent/'homography_features.csv').drop(columns=['cc','h20','h21'])
D = q.merge(f15,on='file').merge(ecc,on='file').merge(hom,on='file')
D['ax'] = D.file.str[0]; D['m'] = D[['dLx','dCx','dRx']].mean(1)
X3 = ['dLx','dCx','dRx']
V = {
 'v1 (QR 네모 가로 3개)': X3,
 'v2-b (QR 네모 가로, 세로, 폭, 높이 12개)': X3+['dLy','dCy','dRy','Lw','Cw','Rw','Lh','Ch','Rh'],
 'v2-a1 (QR별 내부 픽셀 ECC affine 18개)': list(ecc.columns[1:]),
 'v2-a2 (판 전체 원근 변환 14개)': list(hom.columns[1:]),
 'v2-a3 (QR만 전부 합침)': X3+['dLy','dCy','dRy','Lw','Cw','Rw','Lh','Ch','Rh']+list(ecc.columns[1:])+list(hom.columns[1:]),
 'v2-d (참고: QR + 뒤쪽 볼트)': X3+['bolt_dx'],
}
def sheet(a):
    c = D[D.ax=='c']; return pd.concat([c, D[D.ax==a]]).sort_values('m').reset_index(drop=True)
Ys, Ts = sheet('Y'), sheet('T')
def resid(row, ref, F):
    return np.array([row[f]-np.interp(row.m, ref.m.values, ref[f].values) for f in F])
def loo(s, F):
    return np.array([resid(s.iloc[i], s.drop(s.index[i]), F) for i in range(1,len(s)-1)])
res = {}
for name, F in V.items():
    L = np.vstack([loo(Ys,F), loo(Ts,F)])
    nY = len(Ys)-2; sd = np.maximum(L[:nY].std(0), L[nY:].std(0)); th = 3*sd  # v1과 같은 기준: 두 시트 중 큰 잡음
    S = np.cov(L.T) if len(F)>1 else np.array([[L.var()]])
    S = S + np.eye(len(F))*1e-9*np.trace(S)/len(F)
    Si = np.linalg.inv(S); dthr = chi2.ppf(0.9973, len(F))
    md = lambda r: float(r@Si@r)
    # 같은 축 오판율 (LOO 잔차가 '구분 가능'으로 나오는 비율)
    fp1 = np.mean([(np.abs(r)>th).any() for r in L]); fp2 = np.mean([md(r)>dthr for r in L])
    out = {}
    for sh, s, ref in [('Y',Ys,Ts),('Θ',Ts,Ys)]:
        rows=[]
        for _,r in s.iterrows():
            if r.file=='center.bmp': rows.append(('기준','기준',np.nan)); continue
            if r.m<ref.m.min() or r.m>ref.m.max(): rows.append(('범위 밖','범위 밖',np.nan)); continue
            rr = resid(r, ref, F); d2 = md(rr)
            rows.append(('불가' if (np.abs(rr)<=th).all() else '가능', '불가' if d2<=dthr else '가능', d2))
        out[sh] = pd.DataFrame(rows, columns=['규칙1','규칙2','거리'])
    res[name] = dict(F=F, fp1=fp1, fp2=fp2, out=out, dthr=dthr)
    c = lambda sh,k: int((out[sh][k]=='불가').sum())
    print(f'{name}\n   규칙1 불가 Y {c("Y","규칙1")}/80 Θ {c("Θ","규칙1")}/71  오판율 {fp1:.1%} | 규칙2 불가 Y {c("Y","규칙2")}/80 Θ {c("Θ","규칙2")}/71  오판율 {fp2:.1%}')
pd.to_pickle(dict(res=res, Ys=Ys, Ts=Ts), Path(__file__).parent/'judge.pkl')
