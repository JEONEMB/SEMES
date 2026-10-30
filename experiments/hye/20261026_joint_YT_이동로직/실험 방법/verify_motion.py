"""Independent known synthetic responses; software checks, not robot trials."""
from pathlib import Path
import json,inspect
import numpy as np
from motion import MotionController,Settings,choose_probe_amplitudes
H=Path(__file__).resolve().parent

def main():
    checks={}
    assert list(inspect.signature(MotionController.start).parameters)==['self','s','s0']
    assert list(inspect.signature(MotionController.observe).parameters)==['self','s','executed_u']
    checks['online_start_observe_signatures_have_no_absolute_pose']=True
    J=np.array([[120.,20.],[-80.,300.],[50.,-90.],[200.,150.]])
    cfg=Settings(probe_y=.2,probe_t=.04,probe_count=2)
    q=np.array([.65,-.4]);goal=np.array([7.,-2.,5.,1.])
    c=MotionController(cfg).start(goal+J@q,goal)
    assert c.J is None
    for _ in range(2):
        u=c.next_action();assert np.array_equal(u,c.next_action())
        q+=u;c.observe(goal+J@q,u)
    assert np.allclose(c.J,J,atol=1e-10)
    assert np.allclose(c.correction,-q,atol=1e-10)
    checks['response_identified_from_executed_moves']=True
    checks['correction_targets_current_pose_after_probes']=True
    for _ in range(20):
        u=c.next_action()
        if u is None:break
        q+=u;c.observe(goal+J@q,u)
    assert c.turn<=20 and c.status=='visual_goal_confirmed' and np.max(abs(q))<=.09
    checks['known_linear_fixture']=dict(turns=c.turn,terminal_q=q.tolist(),status=c.status)
    # Imperfect execution: use acknowledged inputs, not the suggested vectors.
    c=MotionController(cfg).start(J@np.array([.2,.3]),np.zeros(4));q=np.array([.2,.3])
    for ack in [np.array([.13,.003]),np.array([-.01,.027])]:
        c.next_action();q+=ack;c.observe(J@q,ack)
    assert np.allclose(c.J,J,atol=1e-9)
    checks['partial_and_cross_axis_acknowledgements_used']=True
    # Complete actuator refusal cannot be mistaken for two-axis excitation.
    c=MotionController(Settings()).start(np.ones(4)*10,np.zeros(4))
    for _ in range(3):c.next_action();c.observe(np.ones(4)*10,np.zeros(2))
    assert c.status=='insufficient_separation' and c.next_action() is None
    checks['unexecuted_probes_rejected']=True
    # Rank-one visual response is ambiguous even with two independent inputs.
    one=np.c_[J[:,0],2*J[:,0]];c=MotionController(cfg).start(one@np.array([.5,.3]),np.zeros(4));q=np.array([.5,.3])
    for _ in range(2):
        u=c.next_action();q+=u;c.observe(one@q,u)
    assert c.J is None and c.status=='insufficient_separation'
    checks['rank_one_visual_response_rejected']=True
    c=MotionController(cfg).start(np.ones(4),np.zeros(4));c.next_action();c.observe(np.full(4,np.nan),np.zeros(2))
    assert c.turn==1 and c.status=='measurement_failed' and c.next_action() is None
    assert c.snapshot()['feature_rms'] is None
    checks['failed_capture_counted_and_terminated']=True
    # Never stop from latent truth: this fixture is deliberately far away.
    c=MotionController(cfg).start(J@np.array([100.,100.]),np.zeros(4));q=np.array([100.,100.])
    for _ in range(20):
        u=c.next_action();assert u is not None
        assert np.all(abs(u)<=[cfg.max_step_y,cfg.max_step_t]);q+=u;c.observe(J@q,u)
    assert c.status=='turn_limit' and c.turn==20 and c.next_action() is None
    checks['all_actions_capped_and_twenty_turn_limit_enforced']=True
    # No persistent session can silently count the same observation twice.
    try:c.observe(J@q,np.zeros(2));raise AssertionError('unexpected')
    except ValueError:pass
    checks['state_machine_rejects_unrequested_observation']=True
    c=MotionController(cfg,J).start(J@np.array([.2,.3]),np.zeros(4))
    for _ in range(2):c.next_action();c.observe(c.s.copy(),np.zeros(2))
    assert c.status=='running' and np.array_equal(c.J,J) and c.identification_updates==0
    checks['guarded_mode_keeps_declared_prior_on_uninformative_moves']=True
    c=MotionController(cfg,J).start(J@np.array([.2,.3]),np.zeros(4));s=c.s.copy()
    for _ in range(2):
        u=c.next_action();s+=3*J@u;c.observe(s,u)
    assert np.array_equal(c.J,J) and c.diagnostics['reason']=='response_change_exceeds_guard'
    checks['large_unsupported_response_change_rejected']=True
    amp=choose_probe_amplitudes(J,[.05,.01],[.25,.25],1.)
    assert np.all(amp>=[.05,.01]) and np.all(amp<=[.25,.25])
    checks['adaptive_probe_respects_supplied_caps']=True
    (H/'software_checks.json').write_text(json.dumps(dict(scope='Synthetic API checks; no physical trial',checks=checks),indent=2),encoding='utf-8')
    print(json.dumps(checks,indent=2))
if __name__=='__main__':main()
