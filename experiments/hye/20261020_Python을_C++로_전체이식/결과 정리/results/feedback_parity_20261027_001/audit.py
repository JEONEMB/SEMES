"""Historical policy oracle; never used by the C++ runtime."""
import sys,json,subprocess,importlib.util
from pathlib import Path
import numpy as np
sys.dont_write_bytecode=True
out=Path(__file__).resolve().parent;final=out.parents[1];root=final.parent
spec=importlib.util.spec_from_file_location('old_feedback_oracle',root/'codex_experiments_20261014/joint_yt_visual_feedback_20261023/control/controller.py');mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
J=np.array([[120.,20.],[-80.,300.],[50.,-90.],[200.,150.]]);maximum=0.;calls=0;records=[]
def clean(x):
    if isinstance(x,list):return [clean(v) for v in x]
    if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
    if isinstance(x,float) and not np.isfinite(x):return None
    return x
for policy in mod.POLICIES:
    for scenario in ['linear','partial_ack','response_change','failed_capture','turn_limit','initial_goal']:
        directory=out/(policy+'_'+scenario);directory.mkdir(exist_ok=False);q=np.array([.65,-.4]) if scenario!='initial_goal' else np.zeros(2);model=J.copy();goal=np.array([7.,-2.,5.,1.]);c=mod.FeedbackController(J,policy).start(goal+model@q,goal);state=None
        for step in range(22):
            action=c.next_action()
            if step==0:request=dict(jacobian=J.tolist(),policy=policy,observation=c.s.tolist(),goal=goal.tolist())
            else:request=dict(state=state,observation=c.s.tolist(),executed=executed.tolist())
            path=directory/f'input_{step}.json';path.write_text(json.dumps(clean(request)));destination=directory/f'output_{step}';subprocess.run([str(final/'build/modern/bin/Release/feedback_step.exe'),str(path),str(destination)],check=True,capture_output=True,text=True);state=json.loads((destination/'state.json').read_text());calls+=1
            assert state['status']==c.status and state['turn']==c.turn and state['streak']==c.streak,(policy,scenario,step)
            fields={'J':c.J,'correction':c.correction,'feature_rms':c.feature_rms,'trust':c.trust,'action':action}
            for key,value in fields.items():
                if value is None:assert state[key] is None;continue
                expected=np.asarray(value,float);actual=np.asarray(state[key],float);assert np.array_equal(np.isfinite(actual),np.isfinite(expected)),(key,actual,expected)
                finite=np.isfinite(expected)
                if np.any(finite):
                    delta=float(np.max(abs(actual[finite]-expected[finite])));maximum=max(maximum,delta);assert delta<=1e-9,(policy,scenario,step,key,delta)
            if action is None:break
            executed=action.copy()
            if scenario=='partial_ack':executed*=np.array([.73,.81]);executed+=np.array([.001,-.0003])
            if scenario=='turn_limit':executed*=0
            q+=executed
            if scenario=='response_change' and step==4:model*=1.3
            observation=goal+model@q
            if scenario=='failed_capture' and step==0:observation[:]=np.nan
            c.observe(observation,executed)
        records.append(dict(policy=policy,scenario=scenario,status=c.status,turn=c.turn))
summary=dict(passed=True,scenarios=len(records),calls=calls,absolute_tolerance=1e-9,max_difference=maximum,results=records)
(out/'checks.json').write_text(json.dumps(summary,indent=2));print(json.dumps({k:v for k,v in summary.items() if k!='results'},indent=2))
