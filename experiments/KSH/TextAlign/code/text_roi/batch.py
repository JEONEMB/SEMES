import os, sys, json, glob
sys.path.insert(0, os.path.dirname(__file__)); from absmeas import measure
root=sys.argv[1]; out=sys.argv[2]
done=set()
if os.path.exists(out):
    for l in open(out,encoding='utf-8'): done.add(json.loads(l)['path'])
fs=sorted(glob.glob(root+'/**/*.bmp',recursive=True))
with open(out,'a',encoding='utf-8') as f:
    for p in fs:
        rp=os.path.relpath(p,root)
        if rp in done or 'ROI' in rp: continue
        try: r=measure(p); r['path']=rp; r['ok']=1
        except Exception as e: r=dict(path=rp,ok=0,err=str(e))
        f.write(json.dumps(r,ensure_ascii=False)+'\n'); f.flush()
print('done',len(fs))
