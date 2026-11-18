import numpy as np, pandas as pd, re
D=pd.read_csv('data/features_719x4ref.csv')
D['file']=D['file'].astype(str)
BAD={('S4','Y-0.8 T-0.96')}
D=D[~D.apply(lambda r:(r.set,r.file) in BAD,axis=1)].reset_index(drop=True)
def basis(Y,T,o): return np.column_stack([np.ones_like(Y),Y]+[T**k for k in range(1,o+1)])
def fit1(Y,T,v,o,w=None):
    A=basis(Y,T,o)
    if w is not None: sw=np.sqrt(w); x,*_=np.linalg.lstsq(A*sw[:,None],v*sw,rcond=None)
    else: x,*_=np.linalg.lstsq(A,v,rcond=None)
    r=A@x-v; return dict(c=x[0],a=x[1],b=list(x[2:]),sigma=max(1e-6,np.sqrt(np.mean(r**2))))
def fit(df,ref,os_=1,od=1,w=None):
    Y=df.Y.values;T=df['T'].values
    return dict(s=fit1(Y,T,df[f's{ref}'].values,os_,w),d=fit1(Y,T,df[f'd{ref}'].values,od,w))
def ev(m,Y,T):
    v=m['c']+m['a']*Y; dt=0; tp=1.0
    for k,b in enumerate(m['b']): dt+=(k+1)*b*tp; tp*=T; v+=b*tp
    return v,m['a'],dt
def invert(M,s,d):
    S,Dm=M['s'],M['d']; det=S['a']*Dm['b'][0]-Dm['a']*S['b'][0]
    rs,rd=s-S['c'],d-Dm['c']; y=(rs*Dm['b'][0]-rd*S['b'][0])/det; t=(S['a']*rd-Dm['a']*rs)/det
    ws,wd=1/S['sigma']**2,1/Dm['sigma']**2
    for _ in range(30):
        vs,sy,st=ev(S,y,t); vd,dy,dt=ev(Dm,y,t); es,ed=s-vs,d-vd
        a00=ws*sy*sy+wd*dy*dy;a01=ws*sy*st+wd*dy*dt;a11=ws*st*st+wd*dt*dt
        b0=ws*sy*es+wd*dy*ed;b1=ws*st*es+wd*dt*ed;dd=a00*a11-a01*a01
        y+=(a11*b0-a01*b1)/dd; t+=(a00*b1-a01*b0)/dd
    return y,t
def predict(M,df,ref):
    out=np.array([invert(M,s,d) for s,d in zip(df[f's{ref}'],df[f'd{ref}'])])
    df=df.copy(); df['ye']=out[:,0]; df['te']=out[:,1]; df['ey']=(df.ye-df.Y)*1000; df['et']=df.te-df['T']; return df
def met(df):
    ey,et=df.ey.values,df.et.values
    return dict(n=len(df),ym=ey.mean(),ysd=ey.std(),yrms=np.sqrt((ey**2).mean()),ymax=np.abs(ey).max(),
      tm=et.mean(),tsd=et.std(),trms=np.sqrt((et**2).mean()),tmax=np.abs(et).max(),
      spec=int(((abs(ey)<=90)&(abs(et)<=0.01)).sum()),half=int(((abs(ey)<=45)&(abs(et)<=0.005)).sum()))
def load_yml(p):
    t=open(p,encoding='utf-8-sig').read()
    def blk(n):
        b=t.split(n+':')[1]
        return dict(c=float(re.search(r'c_px:\s*([-\d.e+]+)',b)[1]),a=float(re.search(r'a_px_per_mm:\s*([-\d.e+]+)',b)[1]),
          b=[float(x) for x in re.search(r'theta_px_per_deg_k:\s*\[([^\]]*)\]',b)[1].split(',')],sigma=float(re.search(r'sigma_px:\s*([-\d.e+]+)',b)[1]))
    return dict(s=blk('\n   s'),d=blk('\n   d'))
