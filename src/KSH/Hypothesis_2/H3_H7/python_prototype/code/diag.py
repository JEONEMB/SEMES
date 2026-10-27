import numpy as np,json,sys
from KSH.Hypothesis_2.H3_H7.python_prototype.code.phys import *
D=np.load('corr.npz',allow_pickle=True); P0,Q,OK,names,info=D['P0'],D['Q'],D['OK'],list(D['names']),D['info']
S,Yl,Tl=info[:,0].astype(int),info[:,1],info[:,2]
C=json.load(open(sys.argv[1])); g=C['g']; X=plate_points(P0,g)
rows=[]
for i in range(len(names)):
    if S[i]!=1: continue
    pr=project(X,Yl[i]+g.get("Y0",0)+g.get("bY",0)*np.sign(Yl[i]),Tl[i]+g.get("T0",0)+g.get("bT",0)*np.sign(Tl[i]),0,g); r=pr-Q[i]; r=r[OK[i]]; xs=P0[OK[i],0]
    A=np.c_[np.ones_like(xs),(xs-1900)/400,((xs-1900)/400)**2]; c,*_=np.linalg.lstsq(A,r[:,0],rcond=None)
    rows.append((Yl[i],Tl[i],*c, np.std(r[:,0]-A@c)))
R=np.array(rows)
for lab,m in (('Y sweep',R[:,1]==0),('T sweep',R[:,0]==0)):
    v=R[m]; key=v[:,0] if lab[0]=='Y' else v[:,1]; o=np.argsort(key)
    print(lab,'cols: val, mean_x, slope, curv, within-sd')
    for j in o[::8]: print('   ',np.round(v[j,[0 if lab[0]=='Y' else 1,2,3,4,5]],4))
