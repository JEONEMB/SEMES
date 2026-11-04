import numpy as np, json
exec(open('limit.py').read().split("d1=cmp")[0])
def h7(Ee,idx,label):
    idx=np.array(idx); tT=Tl[idx]; tY=Yl[idx]; eT=Ee[idx,1]; eY=Ee[idx,0]
    cal=np.isin(np.round(tT,2),[0.01,1.0]); te=~cal
    g,o=np.linalg.lstsq(np.c_[tT[cal],np.ones(cal.sum())],eT[cal],rcond=None)[0]
    oy,ky=np.linalg.lstsq(np.c_[np.ones(cal.sum()),tT[cal]],eY[cal]-tY[cal],rcond=None)[0]
    cT=(eT-o)/g; cY=eY-oy-ky*cT
    out={}
    for nm,(a,b) in (('before',(eT,eY)),('after',(cT,cY))):
        dT=a[te]-tT[te]; dY=(b[te]-tY[te])*1000
        out[nm]=dict(rmsT=float(np.sqrt((dT**2).mean())),passT=int(np.sum(np.abs(dT)<=0.01)),rmsY=float(np.sqrt((dY**2).mean())),passY=int(np.sum(np.abs(dY)<=90)),n=int(te.sum()))
    print(f"{label}: H7 gain {g:.4f} off {o:+.4f}°  before Θ RMS {out['before']['rmsT']:.4f} pass {out['before']['passT']}/{out['before']['n']}  →  after Θ RMS {out['after']['rmsT']:.4f} pass {out['after']['passT']}/{out['after']['n']}, Y ±90 {out['after']['passY']}")
    return out
res={}
res['H3_field_H7']=h7(E2,s2,'H3 선형화 물리 모델')
json.dump(res,open('h7_field.json','w'),indent=1)
