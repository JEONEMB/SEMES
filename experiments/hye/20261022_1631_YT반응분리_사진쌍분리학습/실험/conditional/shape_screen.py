from screen import H,load,features,metric,np,pd,threadpool_limits
from conditional_model import ScalarGLS
from forward_model import ForwardGLS

def main():
    m,a=load();D=a['D'];y=m[['Y','T']].to_numpy();new=m.dataset==20261020
    global_shape=D-D.mean(1)[:,None,:];local_shape=D.copy()
    for qr in range(3):
        mask=a['regions']==qr;local_shape[:,mask]-=D[:,mask].mean(1)[:,None,:]
    bank=dict(global_shape=global_shape.reshape(len(D),-1),qr_shape=local_shape.reshape(len(D),-1))
    for k,x in list(bank.items()):bank[k+'_phase']=np.c_[x,x*(D[:,:,0].mean(1)/100)[:,None]]
    h=D.mean(1);records=[];preds=[];diagnostics=[]
    configs=[dict(feature=f,shrink=s,t_model=tm,h_degree=d) for f in bank for s in [.01,.1,.5] for tm in ['scalar','joint'] for d in [1,2]]
    for k,c in enumerate(configs):
        X=bank[c['feature']];p=np.full(y.shape,np.nan)
        for fold in range(4):
            tr=m.fold!=fold;te=m.fold==fold;yy,tt=y[tr].T
            if c['t_model']=='scalar':T=ScalarGLS(c['shrink']).fit(X[tr],tt).predict(X[te])
            else:T=ForwardGLS(1,c['shrink']).fit(X[tr],y[tr]).predict(X[te])[:,1]
            A=np.c_[np.ones(tr.sum()),yy,tt] if c['h_degree']==1 else np.c_[np.ones(tr.sum()),yy,tt,yy*yy,yy*tt,tt*tt]
            co=np.linalg.lstsq(A,h[tr],rcond=1e-10)[0]
            # Given predicted T, invert both average x/y responses for Y by Gauss-Newton.
            z=h[te]-co[0]-T[:,None]*co[2];Y=z@co[1]/(co[1]@co[1])
            if c['h_degree']==2:
                for _ in range(12):
                    r=h[te]-(co[0]+Y[:,None]*co[1]+T[:,None]*co[2]+Y[:,None]**2*co[3]+(Y*T)[:,None]*co[4]+T[:,None]**2*co[5])
                    j=co[1]+2*Y[:,None]*co[3]+T[:,None]*co[4]
                    Y+=np.clip(np.sum(r*j,axis=1)/np.maximum(np.sum(j*j,axis=1),1e-12),-.5,.5)
            p[te]=np.c_[Y,T]
            if k==0:diagnostics.append(dict(fold=fold,horizontal_Y_slope=float(co[1,0]),horizontal_T_slope=float(co[2,0]),T_error_amplification_into_Y=float(co[2,0]/co[1,0])))
        for scope,mask in [('new60',new),('old281',~new),('all341',np.ones(len(m),bool))]:records.append(dict(candidate=k,**c,scope=scope,**metric(y[mask],p[mask])))
        pp=m[['index','dataset','file','Y','T','scope','condition','fold']].copy();pp['candidate']=k;pp['predY']=p[:,0];pp['predT']=p[:,1];preds.append(pp)
        if k%12==0:print(k,c,metric(y[new],p[new]),flush=True)
    pd.DataFrame(records).to_csv(H/'shape_summary.csv',index=False);pd.concat(preds,ignore_index=True).to_csv(H/'shape_predictions.csv',index=False)
    pd.DataFrame(diagnostics).to_csv(H/'shape_sensitivity.csv',index=False)
    df=pd.DataFrame(records);print(df[df.scope=='new60'].sort_values(['passed','Y_max'],ascending=[False,True]).head(10).to_string(index=False),flush=True)

if __name__=='__main__':
    with threadpool_limits(1):main()
