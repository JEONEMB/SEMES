"""v7 판정기: 사진 1장의 요약값 f (center 대비) → (축, 값).
가설 a ∈ {Y, Θ}마다 f = b_a + s_a v + c_a v² + 잡음(Σ). 비용 = 마할라노비스 잔차²를 v에 대해 최소화. 비용 작은 축 → 그 v.
0점 처리:
  'center' : b = 0 (center 사진을 그대로 0점으로 믿음)
  'learn'  : b를 학습 사진으로 맞춤 (center와 실제 0점의 평균 차이를 흡수, 배포 가능)
  'oracle' : 시험 사진이 속한 구간의 0점을 앎 (배포 불가, 0점이 완벽히 맞으면 어디까지 가나 = 상한)
라벨: 'raw' 명령값 / 'corr' 보정본 (v4 ①, 폴드마다 학습 사진으로만)"""
import io, contextlib, sys
from lib7 import *
sys.path.insert(0,'../analysis_v5')
with contextlib.redirect_stdout(io.StringIO()):
    import step4_8 as V5
FOLD={n:V5.FOLD[n] for n in V5.FOLD}
CORR={k:{n:V5.CORR[k][i] for i,n in enumerate(V5.NAMES)} for k in (0,1)}
AXA=np.array([AXIS[n] for n in NAMES]); CMD=np.array([VAL[n] for n in NAMES]); SESA=np.array([session_of(n) for n in NAMES])
FOLDA=np.array([FOLD.get(n,-1) for n in NAMES])
THETA_TO_Y=80.66/14.39
GRID={'Y':np.linspace(-2.6,2.6,5201),'Θ':np.linspace(-1.3,1.3,5201)}

def cov_est(R,kind):
    S=np.cov(R.T)
    if kind=='diag': return np.diag(np.diag(S))
    if kind=='shrink':
        from sklearn.covariance import LedoitWolf
        return LedoitWolf().fit(R).covariance_
    return S

def fit(X,lab,tr,quad,zero):
    """반환: 축마다 (b, s, c), 세션 절편(oracle용), Σ"""
    M={}; res=[]; sesb={}
    for a in ['Y','Θ']:
        m=tr&(AXA==a); v=lab[m]
        if zero=='oracle':
            ss=[s for s in SES if s[0]==a[0]]
            Ses=np.stack([(SESA[m]==s).astype(float) for s in ss],1)
            A=np.c_[v,v**2 if quad else np.zeros_like(v),Ses]
            co=np.linalg.lstsq(A,X[m],rcond=None)[0]; M[a]=(np.zeros(X.shape[1]),co[0],co[1])
            for j,s in enumerate(ss): sesb[s]=co[2+j]
        elif zero=='shared':
            mm=tr&(AXA!='C'); vv=lab[mm]; iy=(AXA[mm]=='Y').astype(float); it=1-iy
            A=np.c_[np.ones(mm.sum()),vv*iy,vv*it]+0
            if quad: A=np.c_[A,vv*vv*iy,vv*vv*it]
            co=np.linalg.lstsq(A,X[mm],rcond=None)[0]; j=1 if a=='Y' else 2
            M[a]=(co[0],co[j],co[j+2] if quad else np.zeros_like(co[0]))
        else:
            cols=[v]+([v**2] if quad else [])+([np.ones_like(v)] if zero=='learn' else [])
            A=np.stack(cols,1); co=np.linalg.lstsq(A,X[m],rcond=None)[0]
            s=co[0]; c=co[1] if quad else np.zeros_like(s); b=co[-1] if zero=='learn' else np.zeros_like(s); M[a]=(b,s,c)
        b,s,c=M[a]; r=X[m]-b-np.outer(v,s)-np.outer(v**2,c)
        if zero=='oracle': r=r-np.stack([sesb[x] for x in SESA[m]])
        res.append(r)
    return M,sesb,np.vstack(res)

def solve(Xt,M,W,sesb=None,ses=None):
    """Xt: 시험 사진들. 반환 축별 (값, 비용)"""
    out={}
    for a in ['Y','Θ']:
        b,s,c=M[a]; Z=Xt-b
        if sesb is not None: Z=Z-np.stack([sesb[x] for x in ses])
        g=GRID[a]; P=np.outer(g,s)+np.outer(g**2,c)          # grid x p
        # cost(v) = (z-P)ᵀW(z-P) = zWz -2 zWP + PWP
        PW=P@W; t1=np.einsum('ij,ij->i',PW,P); t2=Z@W@P.T     # (n, grid)
        cost=t1[None,:]-2*t2; k=cost.argmin(1); zz=np.einsum('ij,jk,ik->i',Z,W,Z)
        out[a]=(g[k],cost[np.arange(len(k)),k]+zz)
    return out

def run(X,quad=True,zero='learn',label='corr',cov='full',ret_cost=False):
    recs=[]
    for k in (0,1):
        te=FOLDA==k; tr=(FOLDA>=0)&~te
        lab=CMD if label=='raw' else np.array([CORR[k][n] for n in NAMES])
        act=np.array([CORR[k][n] for n in NAMES])
        M,sesb,R=fit(X,lab,tr,quad,zero); S=cov_est(R,cov); W=np.linalg.pinv(S)
        idx=np.where(te)[0]
        out=solve(X[idx],M,W,sesb if zero=='oracle' else None,SESA[idx])
        for j,i in enumerate(idx):
            pa='Y' if out['Y'][1][j]<=out['Θ'][1][j] else 'Θ'
            recs.append(dict(file=NAMES[i],axis=AXA[i],session=SESA[i],cmd=CMD[i],act=act[i],pred_axis=pa,pred=out[pa][0][j],
                             cost_Y=out['Y'][1][j],cost_T=out['Θ'][1][j],val_Y=out['Y'][0][j],val_T=out['Θ'][0][j]))
    return pd.DataFrame(recs)

def evaluate(df,basis):
    t=df[basis].values; p=df.pred.values; ok=(df.pred_axis==df.axis).values
    y,th=(df.axis=='Y').values,(df.axis=='Θ').values; e=np.abs(p-t)
    passed=np.where(ok,e<=0.09,(np.abs(t)<=0.09)&(np.abs(p)<=0.09))
    bad=~ok; yeq=np.abs(df.cmd.values)*np.where(y,1,THETA_TO_Y)
    r=dict(축정확도=ok.mean()*100,축틀림_Y=int((bad&y).sum()),축틀림_Θ=int((bad&th).sum()),
           틀림최대_Yeq=yeq[bad].max() if bad.any() else 0.0)
    for m,nm in [(y,'Y'),(th,'Θ')]:
        mm=m&ok; r[f'{nm}_평균오차']=e[mm].mean()*1000; r[f'{nm}_최대오차']=e[mm].max()*1000
    r['통과09']=passed.mean()*100; r['통과09_Y']=passed[y].mean()*100; r['통과09_Θ']=passed[th].mean()*100
    return r
def both(df):
    a=evaluate(df,'cmd'); b=evaluate(df,'act'); out={}
    for k,v in a.items(): out[f'명령값_{k}']=v
    for k,v in b.items(): out[f'실제_{k}']=v
    return out

# ---------- 추가 판정 방법 ----------
def run_zcov(X,quad=True,label='corr'):
    """0점 학습 + 구간 0점 흔들림을 잡음으로 (Σ = 구간안 Σw + 구간 0점 Σz). 배포 가능."""
    recs=[]
    for k in (0,1):
        te=FOLDA==k; tr=(FOLDA>=0)&~te
        lab=CMD if label=='raw' else np.array([CORR[k][n] for n in NAMES]); act=np.array([CORR[k][n] for n in NAMES])
        Mo,sesb,Ro=fit(X,lab,tr,quad,'oracle'); M,_,_=fit(X,lab,tr,quad,'learn')
        Sw=np.cov(Ro.T); B=np.stack([sesb[s] for s in SES]); Sz=np.cov(B.T); W=np.linalg.pinv(Sw+Sz)
        idx=np.where(te)[0]; out=solve(X[idx],M,W)
        for j,i in enumerate(idx):
            pa='Y' if out['Y'][1][j]<=out['Θ'][1][j] else 'Θ'
            recs.append(dict(file=NAMES[i],axis=AXA[i],session=SESA[i],cmd=CMD[i],act=act[i],pred_axis=pa,pred=out[pa][0][j],cost_Y=out['Y'][1][j],cost_T=out['Θ'][1][j]))
    return pd.DataFrame(recs)

def run_joint(X,label='corr'):
    """Y와 Θ를 동시에 추정: f = b + S_Y y + S_T t (+2차) → GLS. 축 = QR 평균 이동으로 환산한 크기가 큰 쪽. 값 = 그 축만의 가설로 다시 맞춘 값."""
    recs=[]
    for k in (0,1):
        te=FOLDA==k; tr=(FOLDA>=0)&~te
        lab=CMD if label=='raw' else np.array([CORR[k][n] for n in NAMES]); act=np.array([CORR[k][n] for n in NAMES])
        m=tr; y=np.where(AXA[m]=='Y',lab[m],0); t=np.where(AXA[m]=='Θ',lab[m],0)
        A=np.c_[np.ones(m.sum()),y,t,y*y,t*t]; co=np.linalg.lstsq(A,X[m],rcond=None)[0]; R=X[m]-A@co; W=np.linalg.pinv(np.cov(R.T))
        M,_,_=fit(X,lab,tr,True,'learn')
        idx=np.where(te)[0]; single=solve(X[idx],M,W)
        for j,i in enumerate(idx):
            yy,tt=0.,0.
            for _ in range(8):   # 가우스-뉴턴
                J=np.c_[co[1]+2*co[3]*yy,co[2]+2*co[4]*tt]; r=X[i]-(co[0]+co[1]*yy+co[2]*tt+co[3]*yy*yy+co[4]*tt*tt)
                d=np.linalg.solve(J.T@W@J,J.T@W@r); yy+=d[0]; tt+=d[1]
            pa='Y' if abs(yy)>=abs(tt)*THETA_TO_Y else 'Θ'
            recs.append(dict(file=NAMES[i],axis=AXA[i],session=SESA[i],cmd=CMD[i],act=act[i],pred_axis=pa,pred=single[pa][0][j],est_Y=yy,est_T=tt))
    return pd.DataFrame(recs)

def run_ml(X,model,label='corr'):
    """머신러닝: 축은 분류기, 값은 축별 GLS 가설로. 입력 = 요약값을 QR 평균 이동 크기로 나눈 '모양' + 크기."""
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.neighbors import KNeighborsClassifier
    mk={'로지스틱':lambda:LogisticRegression(C=10,max_iter=5000),'SVM(RBF)':lambda:SVC(C=10,gamma='scale'),
        'GBM':lambda:GradientBoostingClassifier(random_state=0),'kNN':lambda:KNeighborsClassifier(5)}[model]
    recs=[]
    for k in (0,1):
        te=FOLDA==k; tr=(FOLDA>=0)&~te
        lab=CMD if label=='raw' else np.array([CORR[k][n] for n in NAMES]); act=np.array([CORR[k][n] for n in NAMES])
        M,_,R=fit(X,lab,tr,True,'learn'); W=np.linalg.pinv(np.cov(R.T))
        # 모양 특징: 평균 이동 방향 성분을 뺀 나머지를 크기로 나눔
        mean_dir=(M['Y'][1]+M['Θ'][1]*1/THETA_TO_Y)/2; mean_dir/=np.linalg.norm(mean_dir)
        def feat(Xa):
            Z=Xa-(M['Y'][0]+M['Θ'][0])/2; mag=Z@mean_dir; sh=Z-np.outer(mag,mean_dir)
            return np.c_[sh/np.maximum(np.abs(mag),0.3)[:,None]*np.sign(mag)[:,None],np.abs(mag)]
        clf=make_pipeline(StandardScaler(),mk()); clf.fit(feat(X[tr]),AXA[tr])
        idx=np.where(te)[0]; pa_all=clf.predict(feat(X[idx])); single=solve(X[idx],M,W)
        for j,i in enumerate(idx):
            pa=pa_all[j]; recs.append(dict(file=NAMES[i],axis=AXA[i],session=SESA[i],cmd=CMD[i],act=act[i],pred_axis=pa,pred=single[pa][0][j]))
    return pd.DataFrame(recs)

def run_curv(X,cols,label='corr'):
    """가장 단순한 규칙: 곡률 κ = 가운데 QR dx − 양쪽 평균, 평균 이동 m. 축마다 κ = b + s m 직선 → 가까운 쪽."""
    i={c:j for j,c in enumerate(cols)}; L,Cc,Rr=X[:,i['L_dx_0']],X[:,i['C_dx_0']],X[:,i['R_dx_0']]
    m=(L+Cc+Rr)/3; kap=Cc-(L+Rr)/2; recs=[]
    for k in (0,1):
        te=FOLDA==k; tr=(FOLDA>=0)&~te
        lab=CMD if label=='raw' else np.array([CORR[k][n] for n in NAMES]); act=np.array([CORR[k][n] for n in NAMES])
        P={}
        for a in ['Y','Θ']:
            mm=tr&(AXA==a); P[a]=(np.polyfit(m[mm],kap[mm],1),np.polyfit(lab[mm],m[mm],1))
        for i_ in np.where(te)[0]:
            d={a:abs(kap[i_]-np.polyval(P[a][0],m[i_])) for a in P}; pa=min(d,key=d.get)
            s,b=P[pa][1]; recs.append(dict(file=NAMES[i_],axis=AXA[i_],session=SESA[i_],cmd=CMD[i_],act=act[i_],pred_axis=pa,pred=(m[i_]-b)/s))
    return pd.DataFrame(recs)
