import json, re, numpy as np, pandas as pd, os
R=[json.loads(l) for l in open('/mnt/user-data/uploads/QR_Image/_work_20261116_개선/text_roi/abs_features.jsonl',encoding='utf-8')]
rows=[]
for r in R:
    p=r['path']; parts=p.split('/'); stem=os.path.splitext(parts[-1])[0]
    st={'1st':'S1','2nd':'S2','3rd':'S3','4th':'S4'}[parts[0]]
    if stem.startswith('center'): Y=T=0.0; st='C0' if parts[0]=='1st' else 'C2'
    else:
        lab=parts[1] if parts[0] in('2nd','3rd') else stem
        m=re.match(r'(?:Y([+-][\d.]+))?\s*(?:T([+-][\d.]+))?',lab.replace('Theta','')); Y=float(m[1]) if m[1] else 0.0; T=float(m[2]) if m[2] else 0.0
    rows.append(dict(path=p,set=st,file=stem,Y=Y,T=T,s_abs=r['s_abs'],d_abs=r['d_abs'],qx=r['qx'],bx=r['bx']))
A=pd.DataFrame(rows)
A=A[~((A.set=='S4')&(A.file=='Y-0.8 T-0.96'))]
print(A.set.value_counts()); A.to_pickle('abs.pkl')
F=pd.read_csv('data/features_719x4ref.csv'); 
# sanity: label agreement counts per set
print(A.groupby('set')[['Y','T']].agg(['min','max']))
