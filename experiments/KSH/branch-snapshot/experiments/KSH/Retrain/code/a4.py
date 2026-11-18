import pandas as pd, numpy as np, re
def lab(f):
    st,n=f[:-4].split('__',1)
    m=re.match(r'^(?:Y([+-][\d.]+))?\s*(?:T([+-][\d.]+))?',n)
    return st,n,(float(m[1]) if m and m[1] else 0.0),(float(m[2]) if m and m[2] else 0.0)
def load(p):
    d=pd.read_csv(p); L=[lab(f) for f in d.file]; d['set']=[x[0] for x in L]; d['name']=[x[1] for x in L]; d['Y']=[x[2] for x in L]; d['T']=[x[3] for x in L]; return d
mt={p.split('|')[1][:-4]:int(p.split('|')[0]) for p in open('/home/claude/cmp/ksh/mt.txt').read().strip().split(';')}
