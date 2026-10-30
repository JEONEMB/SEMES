"""Rebuild the primary independent image measurements from the original BMPs."""
from pathlib import Path
import subprocess,sys,concurrent.futures,time,json
import numpy as np
import zero_experiment as Z

def main():
    for file,args in [('measurement/measure.py',[]),('measurement/refine.py',['--window','17']),('measurement/refine.py',['--window','31']),('measurement/spread.py',[])]:
        subprocess.run([sys.executable,str(Z.HERE/file),*args],check=True)
    sys.path.insert(0,str(Z.HERE/'direct_alignment'))
    from direct_pair import DirectPairExtractor
    ext=DirectPairExtractor(Z.ROOT/'center.bmp',inset=10)
    paths=[Z.ROOT/'center.bmp']+sorted((Z.ROOT/'Theta').glob('*.bmp'))+sorted((Z.ROOT/'Y').glob('*.bmp'))
    start=time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        rows=list(pool.map(ext.extract,paths))
    np.savez_compressed(Z.HERE/'direct_alignment/direct_p12_newinit.npz',names=np.array([p.name for p in paths]),**{k:np.array([r[k] for r in rows]) for k in rows[0]})
    print('Rebuilt dense alignment from raw image pairs in',time.time()-start,'seconds')

if __name__=='__main__':main()
