# 시제품 측정 : img 폴더의 모든 bmp 에 대해 랜드마크 이동(center.bmp 대비)을 측정해 json 으로 저장
# 사용 : python dev_run.py <img 폴더> <출력 json> [제한 시간 s]   (중간에 멈춰도 이어서 측정)
import cv2,numpy as np,json,sys,os,time
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from legacy.KSH.Hypothesis_2.Center_check.tools.reg import measure
IMG=sys.argv[1]; out=sys.argv[2]; tlim=float(sys.argv[3]) if len(sys.argv)>3 else 1e9
files=sorted(f for f in os.listdir(IMG) if f.lower().endswith('.bmp'))
ref=cv2.imdecode(np.fromfile(os.path.join(IMG,'center.bmp'),np.uint8),0)
res=json.load(open(out)) if os.path.exists(out) else {}
t0=time.time()
for f in files:
    if f in res: continue
    if time.time()-t0>tlim: break
    g=cv2.imdecode(np.fromfile(os.path.join(IMG,f),np.uint8),0)
    m=measure(ref,g); res[f]={k:[round(v[0],4),round(v[1],4),round(v[2],3)] for k,v in m.items()}
    json.dump(res,open(out,'w'))
print(len(res),'/',len(files))
