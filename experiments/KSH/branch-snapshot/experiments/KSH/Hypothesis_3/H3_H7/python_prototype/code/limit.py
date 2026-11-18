import numpy as np, json, re
exec(open('field.py').read().split("if __name__")[0])
bx,by=['1','u','u2','v','uv'],['1','u','v','uv']
K,M,C,E,E2=run(bx,by,True)
# 배경 시차 정답 (파이썬 시제품 측정값) : (Y,T,shot) 로 이름 맞춤
SP='/tmp/claude-0/-home-claude/0114ab36-a469-5e68-a006-eed126d9175b/scratchpad'
import sys; sys.argv=['x']; cwd=__import__('os').getcwd(); __import__('os').chdir(SP+'/center')
exec(open('an.py').read().split("for setname,cols")[0], globals())
cols=[0,1,2,3,8,9,10,11,12,13]; Pp,Cp,_=fit(train,cols)
parT={}
c1p=[i for i in range(len(names)) if names[i]=='center1.bmp'][0]
for i in range(len(names)):
    s,y,t,sh=info[i] if False else (S[i],Y[i],T[i],SH[i])
    ref=DX[[i]][:,cols] - (DX[c1p][cols]-Pp[3] if s==2 and not names[i].startswith('center') else 0)
    e=est(Pp,Cp,ref)[0]; parT[(int(s),round(float(y),3),round(float(t),3),int(sh),names[i].startswith('center'))]=e
__import__('os').chdir(cwd)
D=np.load('corr.npz',allow_pickle=True); nm=list(D['names']); inf=D['info']
key=lambda i:(int(inf[i,0]),round(float(inf[i,1]),3),round(float(inf[i,2]),3),int(inf[i,3]),nm[i].startswith('center'))
def cmp(E,idx,label):
    pt=np.array([parT[key(i)] for i in idx]); e=E[idx]
    dT=e[:,1]-pt[:,1]; dY=(e[:,0]-pt[:,0])*1000; dL=pt[:,1]-inf[idx,2].astype(float)
    print(f'{label:20s} vs parallax truth: Θ RMS {np.sqrt((dT**2).mean()):.4f} (bias {dT.mean():+.4f}) ±0.01 {np.sum(np.abs(dT)<=0.01)}/{len(idx)} | parallax vs label Θ RMS {np.sqrt((dL**2).mean()):.4f}')
    return dT
T1=test1t; Y1=test1y
d1=cmp(E,T1,'set1 test Θ sweep'); cmp(E,Y1,'set1 test Y sweep'); cmp(E2,s2,'set2')
# 점 잡음만으로 생기는 한계 : 영상 안 기저 적합 잔차로 계수 공분산 → Θ σ
sig=[]
for i in T1:
    m=OK[i]; A=np.stack([BASIS[b] for b in bx],1)[m]; d=(Q[i,m,0]-P0[m,0]); c,*_=np.linalg.lstsq(A,d,rcond=None); r=d-A@c
    Cc=np.linalg.inv(A.T@A)*r.var(); sig.append(np.sqrt(Cc[2,2]))
print('point-noise-only σ of u2 coeff (px per (1000px)^2):',np.mean(sig).round(4),'→ Θ σ ≈',(np.mean(sig)/abs(M[1][2])).round(4),'deg  (u2 sensitivity',M[1][2].round(3),')')
