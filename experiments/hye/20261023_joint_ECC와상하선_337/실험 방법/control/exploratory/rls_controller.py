"""Exploratory guarded secant regression, added AFTER primary results.

No absolute state enters this class. Only visual observations and acknowledged
relative commands are used online. Existing primary controller is unmodified.
"""
from pathlib import Path
import sys,numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from controller import FeedbackController

POLICIES=('rls_feedback_only','rls_probe2_large','rls_probe3_large')
SETTINGS=dict(probe=[.20,.04],history_pairs=12,forgetting=.95,prior_precision=2.,
              minimum_predicted_difference_noise_ratio=1.,minimum_two_axis_SNR=1.5,
              huber_innovation_RMS=2.5,minimum_noise_floor_improvement=.25,
              maximum_relative_response_change=.5,update_blend=.5)

class GuardedRLS(FeedbackController):
    def __init__(self,jacobian,policy):
        if policy not in POLICIES:raise ValueError(policy)
        super().__init__(jacobian,'fixed_jacobian');self.policy=policy
        self.probes=0 if policy=='rls_feedback_only' else (2 if policy=='rls_probe2_large' else 3)
        d=np.asarray(SETTINGS['probe']);self.probe_actions=[np.array([d[0],0.]),np.array([0.,d[1]]),-d]
        self.history=[];self.update_attempts=0;self.updates_accepted=0
        self.gate_counts=dict(small_signal=0,unexcited=0,noise_floor=0,response_change=0)
        self.last_two_axis_SNR=0.

    def _update(self,delta,u):
        K=len(delta);difference_noise=np.sqrt(2*K)
        if np.linalg.norm(self.prior@u)<SETTINGS['minimum_predicted_difference_noise_ratio']*difference_noise:
            self.gate_counts['small_signal']+=1;return
        innovation=delta-self.J@u
        rms=float(np.sqrt(np.mean(innovation**2)))/np.sqrt(2)
        weight=min(1.,SETTINGS['huber_innovation_RMS']/max(rms,1e-12))
        self.history.append((u.copy(),delta.copy(),weight));self.history=self.history[-SETTINGS['history_pairs']:]
        if len(self.history)<2:self.gate_counts['unexcited']+=1;return
        self.update_attempts+=1
        scale=np.asarray(SETTINGS['probe']);U=np.array([r[0]/scale for r in self.history]).T;S=np.array([r[1] for r in self.history]).T
        W=np.array([r[2] for r in self.history])*SETTINGS['forgetting']**np.arange(len(self.history)-1,-1,-1)
        G=(U*W)@U.T;eigen,V=np.linalg.eigh(G)
        if eigen.min()<1e-8:self.gate_counts['unexcited']+=1;return
        root=(V*np.sqrt(np.maximum(eigen,0)))@V.T
        prior_scaled=self.prior*scale[None]
        unregularized=((S*W)@U.T)@np.linalg.inv(G)
        snr=min(np.linalg.svd(prior_scaled@root,compute_uv=False)[-1],np.linalg.svd(unregularized@root,compute_uv=False)[-1])/difference_noise
        self.last_two_axis_SNR=float(snr)
        if snr<SETTINGS['minimum_two_axis_SNR']:self.gate_counts['unexcited']+=1;return
        ridge=SETTINGS['prior_precision']*np.eye(2)
        proposed_scaled=((S*W)@U.T+prior_scaled@ridge)@np.linalg.inv(G+ridge)
        proposed=proposed_scaled/scale[None]
        prior_error=np.sum((S-prior_scaled@U)**2,axis=0)
        new_error=np.sum((S-proposed_scaled@U)**2,axis=0)
        if float(np.average(prior_error-new_error,weights=W))<SETTINGS['minimum_noise_floor_improvement']*2*K:
            self.gate_counts['noise_floor']+=1;return
        # Reject a noise-induced weak-axis rotation/amplification, even when
        # overall feature-space norm is dominated by the strong direction.
        relative=np.linalg.lstsq(self.prior,proposed,rcond=None)[0]
        if np.linalg.norm(relative-np.eye(2),2)>SETTINGS['maximum_relative_response_change']:
            self.gate_counts['response_change']+=1;return
        self.J=(1-SETTINGS['update_blend'])*self.J+SETTINGS['update_blend']*proposed
        self.updates_accepted+=1

    def observe(self,s,executed_u):
        if self.status!='running':raise ValueError('Controller is not awaiting an observation')
        s=np.array(s,float);u=np.array(executed_u,float)
        if u.shape!=(2,) or s.shape!=self.s.shape:raise ValueError('Feedback input shape mismatch')
        old=self.s;delta=s-old;old_norm=float(np.linalg.norm(old-self.s0));self.turn+=1
        if np.isfinite(s).all() and np.linalg.norm(u)>1e-12:self._update(delta,u)
        if self.last_kind=='feedback' and np.isfinite(s).all():
            ratio=float(np.linalg.norm(s-self.s0))/max(old_norm,1e-12)
            if ratio>1.05:self.trust=max(.1,self.trust*.5)
            elif ratio<.8:self.trust=min(1.,self.trust*1.2)
        self.s=s;self._assess()
