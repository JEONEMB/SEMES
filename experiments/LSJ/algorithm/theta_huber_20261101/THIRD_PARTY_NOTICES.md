# Third-party implementation reference

The IRLS/MAD/Huber/deviance algorithm in `src/huber_line.cpp` is a C++ adaptation of the line-estimation portions of statsmodels 0.15.0:

- `statsmodels.robust.robust_linear_model.RLM.fit`, `deviance`, `_estimate_scale`
- `statsmodels.robust.norms.HuberT`
- `statsmodels.robust.scale.mad`
- `statsmodels.regression._tools._MinimalWLS`

Statsmodels is distributed under the BSD 3-Clause license; see `licenses/statsmodels-LICENSE.txt` for the complete copyright/license notice, copied from the installed distribution. The C++ adapter uses OpenCV SVD rather than NumPy pseudoinverse and explicitly reports undefined scales/non-finite results rather than returning NaN. It does not implement statsmodels covariance estimation.

Source snapshots already verified in this workspace are under `analysis/huber_convergence_20261101/sources/`. Their hashes are recorded in `validation/source_provenance.json`.

OpenCV 3.4.15 is an existing external build dependency and is not vendored in this source folder.
