# Forward GLS C++ port

Original: codex_experiments_20261014/joint_yt_generalization_20261022/forward_model.py. Implementation: include/forward_gls.hpp and tool/forward_gls.cpp, built in the modern preset. Native LAPACK comes from the copied, pinned OpenBLAS runtime; it does not import Python.

The port implements polynomial values and analytic derivatives, training command scales, weighted least squares with rcond=1e-10, per-feature residual scales, shrinkage covariance through the Woodbury identity, the symmetric projected Gram matrix, linear initialization and bounded polynomial inversion with the original 35-iteration update/acceptance sequence. It serializes the fitted matrices as JSON and can reload them for prediction. Online prediction accepts feature matrices only.

Validation uses the existing 341 development observations with fold0 excluded from fitting. Six configurations (degree1/2/3, shrink0.01/0.1) match original predictions within the declared 1e-6 command tolerance. Maximum observed difference is 7.76e-7; linear differences are below 7.69e-13. This is numerical parity on exposed data, not independent performance evidence. Results: results/forward_gls_parity_20261027_001/checks.json. The native implementation supports degree1..8; only degrees1..3 were compared here.

Input to `forward_gls training.json new-output` has X, commands, degree, shrink and optional weights/test_X. For prediction from a serialized model, supply model and X. Each output must use a new directory. The separate ObservationModel sequence optimizer has not been marked complete by this port.
