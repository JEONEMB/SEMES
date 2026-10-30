"""Audit-only model, feature and path checks; requires the main recheck's full280 output."""
import json
from pathlib import Path
import subprocess
import time
from recheck_20261027 import ROOT, OUT, BUILD, DATA, exe

def main():
    runs=[]
    def run(name,args,cwd=ROOT):
        start=time.monotonic()
        with (OUT/(name+'.log')).open('w',encoding='utf-8') as f:
            p=subprocess.run([str(x) for x in args],cwd=cwd,stdout=f,stderr=subprocess.STDOUT)
        runs.append(dict(name=name,command=[str(x) for x in args],cwd=str(cwd),exit_code=p.returncode,seconds=time.monotonic()-start))
        (OUT/'supplement_commands.json').write_text(json.dumps(runs,ensure_ascii=False,indent=2),encoding='utf-8')
        print(name,p.returncode,flush=True)
    features=ROOT.parent/'codex_experiments_20261014/center_patch_v2_20261021'
    corners=ROOT.parent/'analysis_grid_corners_20261019/out/corners_280.csv'
    run('parity280',[exe('audit'),'models/edge630',DATA+'/center.bmp',corners,features/'v1_edge630_features.npy',features/'tracks.npz',ROOT.parent/'motion_web_20261022/verification_630/recomputed280.csv',ROOT.parent/'motion_speed_630_20261022/results/compiled280.csv',OUT/'full280',OUT/'parity280'])
    run('aggregation',[exe('aggregation_check'),corners,features/'tracks.npz',ROOT.parent/'codex_experiments_20261014/joint_yt_improved_20261022/features',OUT/'aggregation'])
    joint=ROOT.parent/'joint_center_630_20261022/results'
    run('joint_replay341',[exe('joint_check'),'models/edge630',DATA+'/center.bmp','models/joint630/model.npz',joint/'measurements.npz',joint/'manifest.csv',joint/'local_surfaces/predictions.csv',OUT/'joint_replay341'])
    run('joint_fresh341',[exe('corpus'),'models/edge630',DATA+'/center.bmp','models/joint630/model.npz',ROOT.parent/'hw-analysis/data/hw/bmp',OUT/'joint_fresh341'])
    # Explicit absolute paths must work outside the source working directory.
    run('predict_other_cwd',[exe('main'),'predict',ROOT/'models/edge630',(ROOT/DATA/'center.bmp').resolve(),(ROOT/DATA/'Y/Y+0.50.bmp').resolve(),OUT/'other_cwd'],cwd=ROOT.parent)
    run('preservation_after',[exe('verify_manifest'),'..','experiments/inventory_20261023_01/files.csv','config/port_progress.json',OUT/'preservation_after'])

if __name__=='__main__':main()
