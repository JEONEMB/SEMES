"""Image-session oracle comparison; no Python runtime dependency."""
import sys,json,subprocess
from pathlib import Path
import numpy as np,joblib
sys.dont_write_bytecode=True
out=Path(__file__).resolve().parent;final=out.parents[1];root=final.parent
old=root/'codex_experiments_20261014/joint_yt_visual_feedback_20261023';sys.path.insert(0,str(old));sys.path.insert(0,str(old/'control'))
from image_observations import ObservationExtractor
from controller import POLICIES
extractor=ObservationExtractor();image=final/'data/raw/hw/bmp/20261014/center.bmp';maximum=0.;calls=0
for condition in ['qr','upper','lower','both']:
    cal=joblib.load(old/f'control/full_calibration_{condition}.joblib')['calibration'];observation=extractor.extract_bytes(image.read_bytes(),condition);s=cal.project(observation['X'])
    for policy in POLICIES:
        directory=out/(condition+'_'+policy);directory.mkdir(exist_ok=False);state=directory/'state.json';c=cal.controller(policy).start(s,np.zeros_like(s))
        for step in range(4):
            if step==0:arguments=['start',str(state),str(image),str(final/f'models/feedback_full341_frozen/{condition}.json'),policy]
            else:c.observe(s,np.zeros(2));arguments=['observe',str(state),str(image),'0','0']
            action=c.next_action();run=subprocess.run([str(final/'build/modern/bin/Release/feedback_image_session.exe'),*arguments],capture_output=True,text=True,check=True);actual=json.loads(run.stdout);(directory/f'observation_{step}.json').write_text(run.stdout);calls+=1
            assert actual['status']==c.status and actual['turns']==c.turn
            if action is None:assert actual['suggested_relative_command'] is None
            else:assert np.max(abs(np.array(actual['suggested_relative_command'])-action))<1e-6
            delta=abs(actual['feature_rms']-c.feature_rms);maximum=max(maximum,delta);assert delta<1e-6
            if action is None:break
        assert c.status=='visual_goal_confirmed'
summary=dict(passed=True,conditions=4,policies=3,calls=calls,feature_rms_tolerance=1e-6,max_feature_rms_difference=maximum,scope='Native image extraction, declared frozen training calibration and JSON state reload; no hardware action')
(out/'checks.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
