# Label-only evaluation (2026-11-04)

The five saved nominal_exclude8 fold models were fitted on original labels with all eight EDA flags excluded from training and selection. The relabel mode scores all 571 non-reference photos with these models. The filename typo uses the corrected pose's fold to keep sibling frames out of training. The other 563 predictions reproduce their prior raw predictions within 1e-12 degrees.

The evaluation manifest gives the folder value for the typo and the median of the three local sequence estimates for each of the three 0.57 photos. For the four repeat offsets it uses nominal T + common_dx / 78.19604712044145 (degrees). The denominator is the median of the three QR horizontal T derivatives in local_motion_jacobian.csv. This provisional conversion assumes fixed Y. The image motion alone does not establish the true T offset. See the nominal-label sensitivity row in summary.csv.

Run t_label_review with OLD_EXPERIMENT REVIEW_EXPERIMENT REVIEW_EXPERIMENT/evaluation_labels.csv relabel.
