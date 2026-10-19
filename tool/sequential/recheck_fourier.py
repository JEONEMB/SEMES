"""Recollect corner/Fourier without changing the fixed sequential estimator or split."""
import argparse,concurrent.futures,json,subprocess,time,hashlib
from pathlib import Path
import pandas as pd
from evaluate import run

def main(repo,out):
    repo=repo.resolve();out=out.resolve();out.mkdir(parents=True,exist_ok=True)
    paths=[repo/'data/SSAFY Test Image Set/center.bmp']+sorted((repo/'data/SSAFY Test Image Set/Theta').glob('*.bmp'))+sorted((repo/'data/SSAFY Test Image Set/Y').glob('*.bmp'))
    jobs=[];started=time.perf_counter()
    for method in ['corner','fourier']:
        for k in range(4):
            d=out/method/f'worker_{k}';d.mkdir(parents=True,exist_ok=True);listing=d/'images.txt';listing.write_text('\n'.join(map(str,paths[k::4])),encoding='utf-8');jobs.append((method,k,d,listing))
    def work(job):
        method,k,d,listing=job
        with (d/'log.txt').open('w') as log:subprocess.run([str(repo/'tool/qr-subpixel/build/Release/validate.exe'),str(listing),str(d),method],stdout=log,stderr=log,check=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(work,jobs))
    all_summary={}
    for method in ['corner','fourier']:
        directory=out/method
        for file in ['points.csv','frames.csv','qr-status.csv']:
            tables=[]
            for k in range(4):
                t=pd.read_csv(directory/f'worker_{k}'/file);t['frame']=t.frame*4+k;tables.append(t)
            t=pd.concat(tables).sort_values('frame')
            if file=='points.csv':t=t.rename(columns={'valid':'accepted'})
            t.to_csv(directory/file,index=False)
        metadata={'workers':4,'images':len(paths),'method':method,'total_collection_seconds':time.perf_counter()-started}
        (directory/'parallel_run.json').write_text(json.dumps(metadata,indent=2))
        run(directory,directory/'evaluation',repo)
        summary=json.loads((directory/'evaluation/summary.json').read_text());f=pd.read_csv(directory/'frames.csv')
        all_summary[method]={'frame_count':len(f),'three_qr_valid':int((f.valid_features==3).sum()),'evaluation':summary}
    manifest={'start_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),'workers':4,'images':len(paths),'source_sha256':{}}
    for p in ['include/qr_refinement.hpp','include/pipeline_settings.hpp','src/qr_features.cpp','tool/sequential/sequential.py','tool/sequential/evaluate.py']:
        manifest['source_sha256'][p]=hashlib.sha256((repo/p).read_bytes()).hexdigest()
    (out/'recheck_summary.json').write_text(json.dumps(all_summary,indent=2));(out/'recheck_manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(all_summary,indent=2))


def common_control(out,repo):
    import shutil
    tables={m:pd.read_csv(out/m/'points.csv') for m in ['corner','fourier']}
    keys=['frame','payload','id']
    accepted=[pd.MultiIndex.from_frame(t[t.accepted==1][keys]) for t in tables.values()]
    common=accepted[0].intersection(accepted[1]);summaries={}
    for method,t in tables.items():
        directory=out/('common-'+method);directory.mkdir(exist_ok=True)
        t['accepted']=(t.accepted==1)&pd.MultiIndex.from_frame(t[keys]).isin(common)
        t['accepted']=t.accepted.astype(int);t.to_csv(directory/'points.csv',index=False)
        for name in ['frames.csv','parallel_run.json']:shutil.copyfile(out/method/name,directory/name)
        run(directory,directory/'evaluation',repo)
        summaries[method]=json.loads((directory/'evaluation/summary.json').read_text())
    (out/'common_summary.json').write_text(json.dumps(summaries,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,default=Path('.'));p.add_argument('--output',type=Path,required=True);a=p.parse_args();main(a.repo,a.output);common_control(a.output.resolve(),a.repo.resolve())
