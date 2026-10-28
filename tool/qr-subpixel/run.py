"""Collect all candidates with the unchanged pipeline; four single-thread workers."""
import argparse,csv,json,subprocess,time
from pathlib import Path

HERE=Path(__file__).resolve().parent
LOCAL_REPO=HERE.parents[1]
REPO=LOCAL_REPO if (LOCAL_REPO/'src/qr_features.cpp').exists() else Path(r'C:\Users\SSAFY\Documents\GitHub\S15P31A507')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--pipeline',choices=['fourier','model','corner']);parser.add_argument('--repo',type=Path,default=REPO);args=parser.parse_args()
    repo=args.repo
    exe=HERE/'build/Release/compare.exe'
    paths=[repo/'data/SSAFY Test Image Set/center.bmp']
    paths+=sorted((repo/'data/SSAFY Test Image Set/Theta').glob('*.bmp'))
    paths+=sorted((repo/'data/SSAFY Test Image Set/Y').glob('*.bmp'))
    out=HERE/('pipeline-'+args.pipeline if args.pipeline else 'results');out.mkdir(exist_ok=True)
    assignments=[list(enumerate(paths))[i::4] for i in range(4)]
    processes=[];start=time.time()
    for i,chunk in enumerate(assignments):
        worker=out/f'worker_{i}';worker.mkdir(exist_ok=True)
        images=worker/'images.txt';images.write_text('\n'.join(str(p) for _,p in chunk),encoding='utf-8')
        log=open(worker/'progress.log','w')
        command=([str(HERE/'build/Release/validate.exe'),str(images),str(worker),args.pipeline] if args.pipeline else [str(exe),'collect',str(images),str(worker)])
        proc=subprocess.Popen(command,stdout=log,stderr=log,creationflags=subprocess.CREATE_NO_WINDOW)
        processes.append((proc,log))
    for proc,log in processes:
        rc=proc.wait();log.close()
        if rc:raise RuntimeError(f'worker exited {rc}')
    for filename in ['points.csv','frames.csv',('qr-status.csv' if args.pipeline else 'candidates.csv')]:
        rows=[];fields=None
        for i,chunk in enumerate(assignments):
            with open(out/f'worker_{i}'/filename,newline='') as handle:
                reader=csv.DictReader(handle);fields=reader.fieldnames
                for row in reader:
                    row['frame']=str(chunk[int(row['frame'])][0]);rows.append(row)
        rows.sort(key=lambda r:int(r['frame']))
        with open(out/filename,'w',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    if not args.pipeline:
        subprocess.run([str(HERE/'build/Release/augment.exe'),str(out)],check=True,creationflags=subprocess.CREATE_NO_WINDOW)
        subprocess.run([str(exe),'synthetic',str(out)],check=True,creationflags=subprocess.CREATE_NO_WINDOW)
    (out/'collection.json').write_text(json.dumps({'frames':len(paths),'workers':4,'opencv_threads_per_worker':1,'seconds':time.time()-start,'base_commit':'28f6759','pipeline_mode':args.pipeline,'initial_values':'fresh unchanged candidate projection; old cornerSubPix explicitly selected for comparison'},indent=2))
    print('Collected',len(paths),'frames in',round(time.time()-start,1),'seconds',flush=True)

if __name__=='__main__':main()
