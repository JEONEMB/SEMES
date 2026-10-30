"""ts33 보충: 반복40에서 '피해 있는' 틀림 (축 틀리고 보정 후 0.09 밖) + 1단계 학습 범위(LOC) 민감도. 결과: out/72_2단계_LOC민감도.csv"""
import ts33
from ts33 import *
def harm(rp):
    w = rp[rp.axis != rp.pred_axis]
    t = np.abs(w.cmd.values); p = np.abs(w.pred.values)
    return ((t > 0.09) | (p > 0.09)).astype(int), w
MEAS = {'17px': load(8), '33px': load33()}; rows = []
for mn, (F, D) in MEAS.items():
    Fe = feats(F, D, 'P12')
    for loc in [(0.3, 0.06), (0.45, 0.09), (0.6, 0.12), (1.0, 0.2), (2.0, 1.0)]:
        ts33.LOC = {'Y': loc[0], 'Θ': loc[1]}
        rp = repeat40(Fe, lambda F=F: TwoStage(F, 'P12')); h, w = harm(rp)
        per = (rp.axis != rp.pred_axis).groupby(rp.rep).sum(); ph = pd.Series(h, index=w.rep.values).groupby(level=0).sum().reindex(range(40), fill_value=0)
        r = dict(측정=mn, LOC_Y=loc[0], LOC_Θ=loc[1], 축정확도=(rp.axis == rp.pred_axis).mean() * 100, 반복당틀림=per.mean(), 반복당_피해틀림=ph.mean(), 전부맞은반복=int((per == 0).sum()), 피해없는반복=int((ph == 0).sum()))
        rows.append(r); print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
pd.DataFrame(rows).round(3).to_csv(OUT/'72_2단계_LOC민감도.csv', index=False, encoding='utf-8-sig')
