"""G1 전체 실행: 어제와 같은 짝/홀 2겹 교차검증.
학습 사진으로 변형 모델(B)과 평균 기준 영상을 만들고, 채점 사진을 5자유도(Θ, Y + 방해 요소 3개) 가우스-뉴턴으로 푼다.
출발점 두 곳 중 비용이 낮은 답을 쓴다."""
import numpy as np, time
from gn_pixel import *
est = np.full(X.shape, np.nan); nui = np.full((len(X), 3), np.nan); cost = np.full(len(X), np.nan); t0 = time.time()
for f, (tr, te) in enumerate(folds):
    B = np.linalg.lstsq(np.c_[X[tr], np.ones(tr.sum())], CD[tr], rcond=None)[0]
    Tm = mean_template([n for n, t in zip(names, tr) if t])
    print(f'fold {f}: 기준 영상 완료 ({time.time()-t0:.0f}초)', flush=True)
    for k, i in enumerate(np.where(te)[0]):
        n = names[i]; m = q.loc[n, ['dLx', 'dCx', 'dRx']].mean()
        p, c, it = solve_multi(read(n), B, m, Tm); est[i] = p[:2]; nui[i] = p[2:]; cost[i] = c
        if k % 20 == 0: print(f'  {k}/{te.sum()} {n} ({time.time()-t0:.0f}초)', flush=True)
np.save(HERE/'g1_est.npy', est); np.save(HERE/'g1_nuisance.npy', nui); np.save(HERE/'g1_cost.npy', cost)
e = est - X; print('완료. Θ RMS %.4f  Y RMS %.4f mm  Y 최대 %.3f mm' % (np.sqrt((e[:,0]**2).mean()), np.sqrt((e[:,1]**2).mean()), np.abs(e[:,1]).max()))
