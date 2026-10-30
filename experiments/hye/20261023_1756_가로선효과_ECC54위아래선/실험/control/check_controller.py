"""Synthetic API/turn-accounting tests; these are not robot performance tests."""
from pathlib import Path
import inspect,json
import numpy as np
from controller import FeedbackController,PARAMETERS
from simulate import protocol

def main():
    J=np.array([[10.,3.],[2.,12.],[4.,-1.]])
    assert list(inspect.signature(FeedbackController.start).parameters)==['self','s','s0']
    assert list(inspect.signature(FeedbackController.observe).parameters)==['self','s','executed_u']
    assert list(inspect.signature(FeedbackController.next_action).parameters)==['self']
    probe_checks=[]
    for count in [2,3]:
        c=FeedbackController(J,policy=f'probe{count}_broyden').start(np.array([10.,5.,3.]),np.zeros(3))
        s=c.s.copy();seen=[]
        for _ in range(count):
            u=c.next_action();seen.append(u.tolist());s=s+J@u;c.observe(s,u)
        assert np.allclose(seen[:2],[[.05,0.],[0.,.01]])
        if count==3:assert np.allclose(seen[2],[-.05,-.01])
        assert c.turn==count and np.allclose(c.J,J)
        probe_checks.append(dict(policy=c.policy,turns=c.turn,probes=seen))
    # Known linear synthetic plant differs from controller's Jacobian.
    actual=np.array([[10.3,3.1],[1.8,11.7],[3.9,-.8]])
    q=np.array([.7,-.4]);c=FeedbackController(J,'probe2_broyden').start(actual@q,np.zeros(3))
    for _ in range(25):
        u=c.next_action()
        if u is None:break
        q=q+u;c.observe(actual@q,u)
    assert c.turn<=20 and np.max(abs(q))<.09
    failed=FeedbackController(J).start(np.full(3,np.nan),np.zeros(3))
    assert failed.status=='measurement_failed' and failed.next_action() is None
    limit=FeedbackController(J,'fixed_jacobian').start(np.ones(3)*100,np.zeros(3))
    for _ in range(20):
        u=limit.next_action();assert u is not None;limit.observe(np.ones(3)*100,u)
    assert limit.turn==20 and limit.status=='turn_limit' and limit.next_action() is None
    out=dict(status='passed',scope='Synthetic software tests, not measured return-to-center performance',online_API_has_no_absolute_pose_labels=True,probe_checks=probe_checks,independent_linear_fixture_terminal_q=q.tolist(),fixture_turns=c.turn,invalid_measurement_rejected=True,twenty_action_limit_enforced=True,zero_confirmation_counts_against_budget=True)
    here=Path(__file__).resolve().parent
    (here/'controller_checks.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    (here/'protocol.json').write_text(json.dumps(protocol(),indent=2),encoding='utf-8')
    print(json.dumps(out,indent=2))

if __name__=='__main__':main()
