# 폐루프(측정 → 반대로 이동 → 재측정) 몬테카를로 : 4차에서 잰 오차 성분이 '멈춤 판정'과 최종 위치에 주는 영향
import numpy as np, json
rng=np.random.default_rng(7)
STOP_Y,STOP_T=0.045,0.005; SPEC_Y,SPEC_T=0.090,0.010
def run(n=20000, noise=(0.0034,0.0022), gain=(0,0), cross=0.0, bias=(0,0), back=(0,0), rep=(0.0075,0.002), scale=(0,0), maxit=20):
    ok=goal=0; its=[]; hit=0; fy=[]; ft=[]; fa=0
    for _ in range(n):
        y=rng.uniform(-1,1); t=rng.uniform(-0.5,0.5); ly=rng.choice([-1.0,1.0]); lt=rng.choice([-1.0,1.0])  # 직전 이동 방향 (이력 모름 → 무작위)
        for it in range(maxit+1):
            yh=y*(1+gain[0])+cross*t+bias[0]+rng.normal(0,noise[0]); th=t*(1+gain[1])+bias[1]+rng.normal(0,noise[1])
            if abs(yh)<=STOP_Y and abs(th)<=STOP_T: break
            if it==maxit: hit+=1; break
            my,mt=-yh,-th
            def act(m,last,b,sc,r):
                a=m*(1+sc)
                if last!=0 and np.sign(m)!=np.sign(last): a=np.sign(m)*max(abs(a)-b,0.0)
                return a+rng.normal(0,r)
            y+=act(my,ly,back[0],scale[0],rep[0]); t+=act(mt,lt,back[1],scale[1],rep[1]); ly,lt=my,mt
        its.append(it); fy.append(y); ft.append(t)
        inspec=abs(y)<=SPEC_Y and abs(t)<=SPEC_T; ok+=inspec; goal+=abs(y)<=0.045 and abs(t)<=SPEC_T
    fy=np.array(fy)*1000; ft=np.array(ft)
    return dict(spec=100*ok/n, goal=100*goal/n, moves=float(np.mean(its)), p95=float(np.percentile(its,95)), limit=100*hit/n, ym=float(fy.mean()), ysd=float(fy.std()), tm=float(ft.mean()), tsd=float(ft.std()), ymax=float(np.percentile(np.abs(fy),99.5)), tmax=float(np.percentile(np.abs(ft),99.5)))
G=(0.0136,0.0078); X=-0.028   # 4차에서 잰 배율·교차
S={
 '① 측정 잡음만 (Y 3.4 µm, Θ 0.0022°)':dict(),
 '② + 수식 배율·교차 오차 (Y +1.4%, Θ +0.78%, Θ→Y −28 µm/°)':dict(gain=G,cross=X),
 '③ + 로봇 백래시 Y 40 µm · Θ 0.006°':dict(gain=G,cross=X,back=(0.040,0.006)),
 '④ + 로봇 백래시 Y 80 µm · Θ 0.006° (4차 최악 점프 수준)':dict(gain=G,cross=X,back=(0.080,0.006)),
 '⑤ ③ + 기준 사진 2시간 경과 (Y −15 µm, Θ +0.0022°)':dict(gain=G,cross=X,back=(0.040,0.006),bias=(-0.015,0.0022)),
 '⑥ ③ + 다른 날 기준 사진 (4차 수준 Y −31.5 µm, Θ +0.0051°)':dict(gain=G,cross=X,back=(0.040,0.006),bias=(-0.0315,0.0051)),
 '⑦ ③ + 다른 날 기준 사진 (10/04↔10/15 수준 Y ±96 µm, Θ −0.018°)':dict(gain=G,cross=X,back=(0.040,0.006),bias=(-0.096,-0.018)),
}
R={k:run(**v) for k,v in S.items()}
for k,v in R.items(): print(k[:40].ljust(40),'spec %.1f goal %.1f moves %.2f p95 %d limit %.2f | Y %+.1f±%.1f (99.5%% %.0f) T %+.4f±%.4f (99.5%% %.4f)'%(v['spec'],v['goal'],v['moves'],v['p95'],v['limit'],v['ym'],v['ysd'],v['ymax'],v['tm'],v['tsd'],v['tmax']))
json.dump(R,open('loop.json','w'),indent=1,ensure_ascii=False)
