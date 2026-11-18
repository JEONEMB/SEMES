"""Image-only deployment engine, timing decomposition and lean result (no plots)."""
from pathlib import Path
import time,json,importlib
import cv2,numpy as np
HERE=Path(__file__).resolve().parent
ROOT=Path(r'C:\Users\SSAFY\Desktop\hwlee\projects\2026-10_SEMES\hw-analysis\data\hw\bmp\20261014')
class Engine:
    def __init__(self,variant='optimized'):
        t=time.perf_counter();self.variant=variant
        module=importlib.import_module(variant);self.center=cv2.imdecode(np.fromfile(ROOT/'center.bmp',np.uint8),0)
        self.extractor=module.StructuralExtractor(self.center);self.model=json.loads((HERE/'assets/model.json').read_text())
        p=self.extractor.points;u,v=(p[:,0]-1906)/486,(p[:,1]-1070)/122;self.B=np.c_[np.ones(len(p)),u,v,u*u,u*v,v*v]
        # Fixed geometry pseudoinverse gives the identical unweighted least-squares solution.
        self.pinv=np.linalg.pinv(self.B);self.W=np.array(self.model['W']);self.init_seconds=time.perf_counter()-t
        cv2.setNumThreads(1)
    def predict(self,raw,diagnostics=False):
        start=time.perf_counter();img=cv2.imdecode(np.frombuffer(raw,np.uint8),0);decoded=time.perf_counter()
        if img is None or img.shape!=self.center.shape:raise ValueError('Fixed-center original image dimensions required')
        r=self.extractor.extract(img);measured=time.perf_counter()
        if not np.isfinite(r['D']).all():raise ValueError('Nonfinite measurement')
        C=np.linalg.lstsq(self.B,r['D'],rcond=None)[0] if self.variant=='baseline' else self.pinv@r['D']
        X=C.ravel()-np.array(self.model['zero_feature']);regressed=time.perf_counter();choices=[]
        for key,(axis,b,s,c) in self.model['branches'].items():
            z=self.W@(X-np.array(b));s=self.W@np.array(s);c=self.W@np.array(c);value=float(z@s/(s@s))
            for _ in range(8):
                j=s+2*value*c;value+=float(j@(z-value*s-value*value*c)/max(j@j,1e-20))
            choices.append(dict(axis=axis,value=value,cost=float(np.sum((z-value*s-value*value*c)**2))))
        best=min(choices,key=lambda a:a['cost'])
        if np.array_equal(img,self.center):best=dict(axis='CENTER',value=0.,cost=0.)
        end=time.perf_counter();timings=dict(decode=decoded-start,**r['timings'],measurement_other=(measured-decoded)-sum(r['timings'].values()),regression=regressed-measured,model=end-regressed,total=end-start)
        out=dict(**best,Y=best['value'] if best['axis']=='Y' else 0.,T=best['value'] if best['axis']=='T' else 0.,points=630,valid=int(r['valid'].sum()),timings=timings,model='v1_edge630_boundary_fixed',variant=self.variant,cache_hit=False)
        if diagnostics:out['measurement']=r
        return out
