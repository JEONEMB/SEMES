"""Check integrated acceptance and downstream estimates without tuning."""
import json
from pathlib import Path
import numpy as np
from evaluate import HERE,read,split_frames,evaluate,write

def main():
    comparison=read(HERE/'results/points.csv');summaries={};predictions=[]
    selected={'fourier':'fourier_a24_centered','model':'model_r3'}
    for method,variant in selected.items():
        directory=HERE/('pipeline-'+method)
        if not (directory/'collection.json').exists():continue
        frames=read(directory/'frames.csv');pts=read(directory/'points.csv');qr=read(directory/'qr-status.csv')
        maps={}
        for p in pts:
            if p['valid']=='1':maps.setdefault(int(p['frame']),{})[p['payload']+':'+p['id']]=np.array([float(p['x']),float(p['y'])])
        train,val,test=split_frames(frames);summary,rows=evaluate(train+val,test,maps)
        ref={(p['frame'],p['payload'],p['id']):np.array([float(p['x']),float(p['y'])]) for p in comparison if p['method']==variant and p['valid']=='1'}
        differences=[]
        for p in pts:
            key=(p['frame'],p['payload'],p['id'])
            if key in ref and p['valid']=='1':differences.append(float(np.linalg.norm(np.array([float(p['x']),float(p['y'])])-ref[key])))
        summaries[method]={'frames':len(frames),'plate_success':sum(f['plate']=='1' for f in frames),'all_3_qr_decoded':sum(f['decoded']=='3' for f in frames),'all_3_qr_features_valid':sum(f['valid_features']=='3' for f in frames),'qr_feature_success':sum(q['status']=='ok' for q in qr),'total_qr':len(qr),'accepted_points':sum(p['valid']=='1' for p in pts),'coordinate_difference_vs_experiment_median_px':float(np.median(differences)),'coordinate_difference_vs_experiment_max_px':float(np.max(differences)),'holdout':summary,'collection':json.loads((directory/'collection.json').read_text())}
        predictions.extend([{'method':method,**r} for r in rows])
    (HERE/'results/pipeline-summary.json').write_text(json.dumps(summaries,indent=2))
    write(HERE/'results/pipeline-holdout.csv',predictions)
    print(json.dumps(summaries,indent=2))

if __name__=='__main__':main()
