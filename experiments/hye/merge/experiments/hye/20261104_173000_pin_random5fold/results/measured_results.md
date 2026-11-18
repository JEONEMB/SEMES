# Measured random-image 5-fold development results

571 non-center images. Y tolerance 0.045, T tolerance 0.01. Failures remain in denominators.

| Model | Valid | Y pass | T pass | Both pass | MAE Y | MAE T | Max Y | Max T | Fold both min-max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| pin_only_linear | 571/571 | 284 | 397 | 281 (49.21191%) | 0.05258 | 0.00797 | 0.22219 | 0.03351 | 45.21739-53.50877% |
| pin_only_quadratic | 571/571 | 326 | 436 | 323 (56.56743%) | 0.04666 | 0.00687 | 0.21018 | 0.03103 | 53.91304-60.52632% |
| qr_pin_linear | 571/571 | 571 | 569 | 569 (99.64974%) | 0.00832 | 0.00210 | 0.04128 | 0.01069 | 99.12281-100.00000% |
| qr_pin_quadratic | 571/571 | 570 | 570 | 569 (99.64974%) | 0.00802 | 0.00168 | 0.05972 | 0.01144 | 99.12281-100.00000% |
| v1_joint_GLS | 571/571 | 549 | 556 | 548 (95.97198%) | 0.01630 | 0.00305 | 0.20570 | 0.03868 | 93.04348-99.12281% |
| v2_joint | 571/571 | 523 | 547 | 515 (90.19264%) | 0.02112 | 0.00397 | 0.19637 | 0.04234 | 87.71930-93.85965% |

## Surrogate simulation, not physical robot trials

Training-only forward response, fixed held-out residual, noiseless ideal motion, simultaneous correction.

| Model | World degree | True pass | False stop | Mean turns | Max turns |
|---|---:|---:|---:|---:|---:|
| pin_only_linear | 1 | 271/571 | 300 | 2.45359 | 9 |
| pin_only_linear | 2 | 292/571 | 279 | 2.45534 | 9 |
| pin_only_quadratic | 1 | 259/571 | 312 | 2.47461 | 10 |
| pin_only_quadratic | 2 | 278/571 | 293 | 2.47636 | 9 |
| qr_pin_linear | 1 | 568/571 | 3 | 2.44308 | 8 |
| qr_pin_linear | 2 | 568/571 | 3 | 2.44308 | 8 |
| qr_pin_quadratic | 1 | 552/571 | 19 | 2.44308 | 9 |
| qr_pin_quadratic | 2 | 566/571 | 5 | 2.44308 | 9 |
| v1_joint_GLS | 1 | 547/571 | 24 | 2.45884 | 9 |
| v1_joint_GLS | 2 | 544/571 | 27 | 2.45709 | 9 |
| v2_joint | 1 | 511/571 | 60 | 2.46060 | 9 |
| v2_joint | 2 | 525/571 | 46 | 2.45534 | 9 |
