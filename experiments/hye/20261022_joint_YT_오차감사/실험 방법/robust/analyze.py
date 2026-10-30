"""Descriptive comparison after the fixed screen; no model changes or tuning."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

H = Path(__file__).resolve().parent
P = H.parents[1] / 'joint_yt_disentangle_20261022'
old = pd.read_csv(P / 'predictions.csv')
old = old[(old.method == 'pass_first') & (old.protocol == 'nested_group_oof')]
pred = pd.read_csv(H / 'predictions.csv')
summary = pd.read_csv(H / 'summary.csv')
ranked = summary[summary.dataset.astype(str) == '20261020'].sort_values(['passed', 'Y_max', 'Y_MAE'], ascending=[False, True, True])
best = int(ranked.iloc[0].candidate)
chosen = pred[pred.candidate == best]
comparison = old[['index', 'predY', 'predT', 'eY', 'eT', 'passed']].merge(chosen, on='index', suffixes=('_prior', '_candidate'))
comparison['transition'] = np.select([
    comparison.passed_prior & comparison.passed_candidate,
    ~comparison.passed_prior & comparison.passed_candidate,
    comparison.passed_prior & ~comparison.passed_candidate],
    ['kept_pass', 'recovered', 'new_failure'], default='still_failure')
comparison.to_csv(H / 'development_comparison_all341.csv', index=False, encoding='utf-8-sig')
new = comparison[comparison.dataset == 20261020]
new[~new.passed_prior].to_csv(H / 'prior10_failures_comparison.csv', index=False, encoding='utf-8-sig')
out = dict(
    screen_best_candidate=best,
    warning='Post-hoc descriptive development comparison: candidate selected using these OOF outcomes; not fair independent validation against prior nested selection.',
    new60_transition_counts=new.transition.value_counts().to_dict(),
    old281_transition_counts=comparison[comparison.dataset == 20261014].transition.value_counts().to_dict(),
    best_new60=ranked.iloc[0].to_dict(),
    original_representative_candidates=[4,5,6,7],
    original_representative_new60_pass_counts=[46,46,45,46],
    all_points_preserved=630,
    notes=['Candidate3 improves pass count but its worst Y error is slightly worse than the prior nested model.',
           'Candidate3 old281 pass count187 is worse than prior192; all341 pass count241 is worse than prior242.',
           'A priori four representatives did not improve the prior50/60. Their identities were not changed after viewing results.'])
(H / 'analysis.json').write_text(json.dumps(out, indent=2), encoding='utf-8')
print(json.dumps(out, indent=2))
