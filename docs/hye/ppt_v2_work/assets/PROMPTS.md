# 생성 이미지 프롬프트 세트

방식: 내장 `image_gen.imagegen`, 방법마다 별도 호출. joint의 3×3 가상 위치는 원본 생성 뒤 편집 호출로 수정했다.

공통 지시: scientific educational concept diagram, white background, navy/electric blue/cyan palette, large legible technical visuals, sparse English labels, no performance numbers, no watermark. 실제 실험결과로 보이지 않도록 원리 개념도임을 슬라이드에 표시한다.

| 이미지 | 개별 생성 지시 |
|---|---|
| harris.png | L-shaped black-white corner, local window, perpendicular x/y displacement arrows, corner-response heatmap with peak |
| shitomasi.png | Flat area / edge / corner windows, gradient-eigenvalue ellipses, large minimum eigenvalue at corner |
| forstner.png | Corner gradient constraints, subpixel intersection and uncertainty ellipse |
| fast.png | 16 circular samples around center pixel, contiguous brighter arc, corner example |
| sift.png | Gaussian scale-space planes, cross-scale extremum and gradient orientation descriptor |
| surf.png | Hessian box filters and four-corner integral-image rectangle |
| orb.png | Rotated patches, orientation arrows, pixel-pair binary descriptors and matches |
| brisk.png | Concentric sampling rings, short/long sampling pairs and binary strip |
| akaze.png | Nonlinear scale space preserving edges, corresponding corner and binary descriptor |
| qrstructure.png | Schematic QR module grid, fixed structural point ID across images, excluded diagonal-only contact |
| ncc.png | Reference template, search window, integer correlation peak |
| phase.png | Translated patches, FFT and cross-power spectrum, inverse-transform shift peak |
| parabolic.png | Three sampled points around an inverted parabola, fractional vertex between integer samples |
| subpix.png | Coarse and refined corner, image gradient constraints and fractional pixel grid |
| lk.png | Reference/current local windows, translated brightness profiles, dx/dy |
| fb.png | Forward p-to-q and backward q-to-p-prime tracking, return error, incorrect-match example |
| gn.png | Iterative template contour alignment, smooth spline curve, gain/offset |
| ecc.png | Perspective-skewed patch, homography warp and increasing correlation |
| joint.png | Exactly three horizontal QR patches, upper/lower long lines at another depth, QR54 + line48, 102 observations, 204 features, GLS, (Y,T). Edit: exactly nine open virtual grid circles in each QR, no measured hardware points |
| control.png | Three QR patches, one reference and current image, persistent point ID, LK+GN630, fixed inner536, P12, two conceptual response curves, six learned feature intercepts, output Y-or-T |
| flow.png | Camera observation, joint T correction, recapture, control Y correction, recapture, both-axis check, loops and maximum20 moves |
| feature_overview.png | Two reference/current images with persistent physical corner IDs and displacement vectors |
| region_overview.png | Reference/current patches, warp grid and successive intensity residual reduction |

최종 발표자료: `../../2026-10-29_중간_시각화수정.pptx`
각 그림은 개념 설명용이며 QR 무늬, 점 배치, 장비 형상을 실측 결과로 취급하지 않는다. 수집현황 차트는 별도로 실제 manifest의 파일명 명령값으로 작성했다.

