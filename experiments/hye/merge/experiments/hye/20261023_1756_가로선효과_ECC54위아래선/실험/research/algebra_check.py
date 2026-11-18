"""Check the stated linear identities on artificial matrices, not robot data."""
from pathlib import Path
import json
import numpy as np

H = Path(__file__).resolve().parent
A = np.array([[1., 2.], [2., 4.], [-1., -2.]])
q = np.array([.3, -.2])
null = np.array([2., -1.])
b = np.array([.1, .2, .3])
inputs = [np.array([.04, 0.]), np.array([0., .07]), np.array([-.03, .02])]
states_a, states_b = [q.copy()], [q+null]
for u in inputs:
    states_a.append(states_a[-1]+u)
    states_b.append(states_b[-1]+u)
hist_a = np.array([A@x+b for x in states_a])
hist_b = np.array([A@x+b for x in states_b])
rank_deficient_history_difference = float(np.max(abs(hist_a-hist_b)))
probe_formula_error = float(max(np.max(abs(hist_a[k+1]-hist_a[k]-A@u)) for k,u in enumerate(inputs)))

B = np.array([[1., 1.], [0., .01], [.2, .3]])
q2 = np.array([.5, .8])
s0 = B@q2
s1 = B@(q2+inputs[0])
s2 = B@(q2+inputs[0]+inputs[1])
J = np.column_stack([(s1-s0)/inputs[0][0], (s2-s1)/inputs[1][1]])
q_after_probe = q2+inputs[0]+inputs[1]
u = -np.linalg.lstsq(J, s2, rcond=None)[0]
out = {
    'scope': 'Artificial algebra checks only; no image, trained robot model, real trajectory or physical performance claim.',
    'rank_deficient_A_rank': int(np.linalg.matrix_rank(A)),
    'different_initial_states_same_observation_history_max_difference': rank_deficient_history_difference,
    'delta_s_equals_A_u_max_difference': probe_formula_error,
    'full_rank_B_rank': int(np.linalg.matrix_rank(B)),
    'two_probe_J_reconstruction_max_difference': float(np.max(abs(J-B))),
    'ideal_linear_noise_free_feedback_residual_command': (q_after_probe+u).tolist(),
    'explanation': 'Independent input directions recover an unknown linear J. They do not remove a nullspace in the observation map. A full-rank, noise-free ideal map supports direct feature-error feedback; this is not a real robot guarantee.'
}
assert rank_deficient_history_difference < 1e-12
assert probe_formula_error < 1e-12
assert np.max(abs(J-B)) < 1e-12
assert np.max(abs(q_after_probe+u)) < 1e-12
(H/'algebra_check.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps(out))
