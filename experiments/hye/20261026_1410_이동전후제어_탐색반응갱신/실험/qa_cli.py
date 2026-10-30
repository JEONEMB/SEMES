"""CLI replay of existing center bytes/failed decode; no hardware interaction."""
from pathlib import Path
import subprocess,sys,json,uuid
H=Path(__file__).resolve().parent

def main():
    root=H/'qa'/str(uuid.uuid4());root.mkdir(parents=True)
    center=H.parents[1]/'hw-analysis/data/hw/bmp/20261014/center.bmp';assert center.exists()
    outputs=[]
    def call(*args):
        r=subprocess.run([sys.executable,str(H/'session.py'),*map(str,args)],capture_output=True,text=True,check=True)
        assert 'NaN' not in r.stdout and 'Infinity' not in r.stdout
        v=json.loads(r.stdout);assert v['hardware_command_sent'] is False;outputs.append(v);return v
    for mode,expected in [('prior-guarded','visual_goal_confirmed'),('online-only','insufficient_separation')]:
        state=root/(mode+'.joblib')
        first=call('start',state,center,'--condition','both','--probe-y','.05','--probe-t','.01','--max-step-y','.25','--max-step-t','.25','--mode',mode)
        assert first['turns']==0 and first['suggested_relative_command']==[.05,0.]
        for turn in range(1,4):
            r=call('observe',state,center,'--dy','0','--dt','0');assert r['turns']==turn
        assert r['status']==expected and r['suggested_relative_command'] is None
        log=call('inspect',state);assert len(log['events'])==3
    bad=root/'invalid.bmp';bad.write_bytes(b'not an image')
    state=root/'failed-capture.joblib'
    call('start',state,center,'--condition','both','--probe-y','.05','--probe-t','.01','--max-step-y','.25','--max-step-t','.25')
    r=call('observe',state,bad,'--dy','.01','--dt','0')
    assert r['turns']==1 and r['status']=='measurement_failed' and r['suggested_relative_command'] is None
    assert call('inspect',state)['events'][0]['status']=='measurement_failed'
    (H/'cli_verification.json').write_text(json.dumps(dict(scope='Existing center replay and decode failure only; all acknowledged increments are synthetic test metadata, not robot movements.',calls=len(outputs),prior_mode_retains_prior_on_zero_execution=True,online_only_rejects_zero_execution=True,failed_capture_counts_acknowledged_move=True,hardware_commands_sent=0,session_directory=str(root)),indent=2))
    print('CLI checks passed',len(outputs),'calls; hardware commands0')
if __name__=='__main__':main()
