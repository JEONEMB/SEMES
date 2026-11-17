from lib import *
import numpy as np, pickle
M0=load_yml('/mnt/user-data/uploads/PJT/config/model_s1d1.yml')
C2=pickle.load(open('cands2.pkl','rb'))
MN=C2['E 기울기=1+4차(세트별상수·균형) / 상수=1차-같은날센터 s1d2']
rows=[l.split() for l in open('/home/claude/align/a.py').read().split('rows="""')[1].split('"""')[0].splitlines()]
d=np.array([[float(x) for x in r] for r in rows])
n,yt,tt,ym,tm=d.T
def fwd(M,Y,T): return ev(M['s'],Y,T)[0], ev(M['d'],Y,T)[0]
out=[]
for i in range(len(n)):
    s,dd=fwd(M0,ym[i],tm[i])
    y1,t1=invert(MN,s,dd)                       # 같은 기준 사진(장비 center) 그대로
    y2,t2=invert(MN,s+0.21048,dd+0.279621)      # 장비가 center.bmp(1차) 기준이었다면 → center1 기준으로 환산
    out.append((n[i],yt[i],ym[i]-yt[i],y1-yt[i],y2-yt[i],tt[i],tm[i]-tt[i],t1-tt[i],t2-tt[i]))
o=np.array(out); m=o[:,0]!=19
for i,lab in [(2,'기존'),(3,'새수식(장비 기준사진 그대로)'),(4,'새수식(center.bmp→center1 환산)')]:
    e=o[:,i]; print(f"Y {lab:28s} mean={e.mean():+.4f} sd={e.std(ddof=1):.4f} rms={np.sqrt((e**2).mean()):.4f} max={abs(e).max():.3f} pos_mean={e[o[:,1]>0].mean():+.4f} neg_mean={e[o[:,1]<0].mean():+.4f}")
for i,lab in [(6,'기존'),(7,'새(그대로)'),(8,'새(환산)')]:
    e=o[m,i]; print(f"T {lab:10s} mean={e.mean():+.4f} sd={e.std(ddof=1):.4f} max={abs(e).max():.3f}")
print('rows 33-35:',np.round(o[32:,[3,4]],3))
p=np.polyfit(o[:,1],o[:,3],1); print('new-model Y err vs Y fit (slope,int):',p)
np.save('semes.npy',o)
