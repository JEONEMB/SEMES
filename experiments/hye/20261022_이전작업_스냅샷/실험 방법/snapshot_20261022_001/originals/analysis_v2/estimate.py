"""버전별 특징으로 Θ, Y를 직접 추정 (선형 회귀, 짝/홀 2겹 교차검증). 오차를 허용 오차 0.09mm와 비교."""
import numpy as np, pandas as pd
from pathlib import Path
J = pd.read_pickle(Path(__file__).parent/'judge.pkl'); res = J['res']
B = Path(__file__).parent.parent
import importlib.util, sys
spec = importlib.util.spec_from_file_location('j', Path(__file__).parent/'judge.py')
# judge.py의 D를 다시 만들지 않고 피클의 Ys, Ts에서 합침
D = pd.concat([J['Ys'], J['Ts'][J['Ts'].file!='center.bmp']]).reset_index(drop=True)
T = np.where(D.ax=='T', D.value, 0.); Y = np.where(D.ax=='Y', D.value, 0.)
import sys
LAM = float(sys.argv[1]) if len(sys.argv)>1 else 1e-9
rows = []
for name, r in res.items():
    X = D[r['F']].values; X = (X-X.mean(0))/X.std(0); A = np.c_[X, np.ones(len(D))]
    lam = LAM*len(D); P = np.eye(A.shape[1]); P[-1,-1] = 0
    pT = np.zeros(len(D)); pY = np.zeros(len(D))
    for k in range(2):
        tr = (np.arange(len(D))%2)!=k; te = ~tr
        for tgt, out in [(T,pT),(Y,pY)]:
            w = np.linalg.solve(A[tr].T@A[tr]+lam*P, A[tr].T@tgt[tr]); out[te] = A[te]@w
    eY = np.abs(pY-Y)*1000; eT = np.abs(pT-T)
    small = (np.abs(Y)<=0.5)&(np.abs(T)<=0.09)
    rows.append(dict(버전=name, Y_RMS_um=np.sqrt((eY**2).mean()), Y_최대_um=eY.max(), Y_90um초과=int((eY>90).sum()),
                     Θ_RMS=np.sqrt((eT**2).mean()), 작은틀어짐_Y_RMS_um=np.sqrt((eY[small]**2).mean())))
o = pd.DataFrame(rows); o.to_csv(Path(__file__).parent/'estimate_result.csv', index=False, encoding='utf-8-sig')
print(o.round(4).to_string(index=False))
