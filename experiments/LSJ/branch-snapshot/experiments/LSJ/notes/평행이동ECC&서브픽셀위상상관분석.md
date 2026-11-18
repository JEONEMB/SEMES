영상 처리 및 컴퓨터 비전에서 평행이동 ECC(Enhanced Correlation Coefficient)와 서브픽셀 위상 상관 분석(Subpixel Phase Correlation)은 두 이미지 간의 미세한 위치 이동($\Delta x, \Delta y$)을 1픽셀 미만의 정밀도(Subpixel accuracy)로 찾아내는 대표적인 정합(Image Registration) 알고리즘입니다.

두 기법은 모두 서브픽셀 단위의 평행이동을 추정하지만, 문제를 푸는 수학적 공간(공간 도메인 vs 주파수 도메인)과 강인성(Robustness)에서 뚜렷한 차이를 보입니다.

---

## 1. 평행이동 ECC (Translation Enhanced Correlation Coefficient)

ECC는 Evangelidis와 Psarakis(2008)가 제안한 공간 도메인(Spatial Domain) 기반의 반복적 최적화 알고리즘입니다. 평행이동뿐만 아니라 회전, 어파인(Affine), 호모그래피(Homography) 변환까지 확장할 수 있으나, 평행이동 모드에서는 오직 $x, y$ 축의 이동 파라미터만 추정합니다.

* **핵심 원리 (제로 평균 정규화 상관계수 극대화):**
* 참조 이미지(Template) $I_r(\mathbf{x})$와 워핑된 입력 이미지 $I_w(\mathbf{x}; \mathbf{p})$의 픽셀 밝기 평균을 뺀 뒤, 두 벡터의 코사인 유사도(정규화된 내적)를 최대화하는 이동 벡터 $\mathbf{p} = [\Delta x, \Delta y]^T$를 찾습니다.
* 목적 함수는 다음과 같이 정의됩니다 ($\bar{I}_r, \bar{I}_w$는 각 영역의 평균이 제거된 제로 평균 밝기 벡터):

$$E(\mathbf{p}) = \left\Vert{} \frac{\bar{I}_r}{\Vert{}\bar{I}_r\Vert{}} - \frac{\bar{I}_w(\mathbf{p})}{\Vert{}\bar{I}_w(\mathbf{p})\Vert{}} \right\Vert{}^2$$




* **조명 및 대비 변화에 대한 불변성 (Photometric Invariance):**
* 픽셀값의 평균을 빼고 노름(Norm)으로 나누어 정규화하기 때문에, 두 이미지 사이에 **전체적인 밝기 오프셋(Offset)이나 대비(Contrast) 스케일 변화**가 발생해도 영향을 받지 않고 정합을 수행합니다.


* **서브픽셀 추정 방식 (반복적 기울기 기반 최적화):**
* 비선형 목적 함수를 테일러 급수로 선형화한 뒤, 이미지의 공간 기울기(Spatial Gradient, $\nabla I$)와 헤시안(Hessian) 행렬을 이용해 파라미터 증분 $\Delta \mathbf{p}$를 반복(Iterative) 계산합니다.
* 매 반복마다 **바이리니어(Bilinear) 또는 바이큐빅(Bicubic) 보간법**으로 이미지를 미세하게 이동시키며 수렴 기준(예: $\epsilon = 10^{-4}$)에 도달할 때까지 정밀도를 높입니다.



---

## 2. 서브픽셀 위상 상관 분석 (Subpixel Phase Correlation)

위상 상관 분석은 푸리에 변환(Fourier Transform)의 이동 정리(Shift Theorem)를 활용하여 주파수 도메인(Frequency Domain)에서 평행이동량을 한 번에 계산하는 기법입니다.

* **핵심 원리 (푸리에 이동 정리):**
* 공간 도메인에서의 평행이동 $(x - \Delta x, y - \Delta y)$는 주파수 도메인에서 진폭(Magnitude)의 변화 없이 선형적인 위상 변화(Phase Shift)로 변환됩니다.
* 두 이미지 $I_1, I_2$의 2차원 고속 푸리에 변환(2D FFT)을 각각 $F_1(u, v), F_2(u, v)$라 할 때, 정규화된 교차 전력 스펙트럼(Normalized Cross-Power Spectrum) $R(u, v)$는 다음과 같습니다 ($F_2^*$는 켤레복소수):
$$R(u, v) = \frac{F_1(u, v) F_2^*(u, v)}{\vert{}F_1(u, v) F_2^*(u, v)\vert{}} = e^{-i 2\pi \left(\frac{u \Delta x}{M} + \frac{v \Delta y}{N}\right)}$$


* **정수 픽셀 피크 검출:**
* $R(u, v)$에 역푸리에 변환(IFFT)을 취하면 평행이동 위치 $(\Delta x, \Delta y)$에서만 날카로운 델타 함수(Dirac delta impulse) 형태의 피크가 나타납니다.


* **서브픽셀 정밀도 달성 방식 (3가지 주요 기법):**
1. **주변 피크 곡면 피팅 / 무게중심법 (Centroid / Parabolic Fitting):** IFFT 결과의 최대 피크 픽셀과 주변 $3 \times 3$ 또는 $5 \times 5$ 이웃 값을 이용해 2차 곡면(포물선)이나 가우시안 피팅, 혹은 가중치 무게중심을 구해 소수점 위치를 추정합니다. 연산이 가장 빠릅니다(OpenCV의 `cv2.phaseCorrelate` 기본 방식).
2. **국소 DFT 업샘플링 (Guizar-Sicairos 기법):** 전체 이미지를 $k$배 제로패딩(Zero-padding)하여 거대한 IFFT를 수행하는 대신, 정수 피크 주변의 아주 좁은 영역($1.5 \times 1.5$ 픽셀 등)에 대해서만 행렬 곱 기반 이산 푸리에 변환(Matrix-multiply DFT)을 수행하여 적은 메모리로 $1/100$ 픽셀 이하의 초고정밀도를 달성합니다.
3. **위상 평면 직접 피팅 (SVD / 최소자승법):** IFFT를 거치지 않고 $R(u, v)$의 위상각(Phase Angle)이 이루는 2차원 경사 평면의 기울기를 특이값 분해(SVD)나 최소자승법으로 직접 추정합니다.



---

## 3. 평행이동 ECC vs 서브픽셀 위상 상관 비교

실제 프로젝트나 비전 파이프라인에 적용할 때 고려해야 할 두 기법의 핵심 차이점은 다음과 같습니다.

| 비교 항목 | 평행이동 ECC (`cv2.findTransformECC`) | 서브픽셀 위상 상관 (`cv2.phaseCorrelate`) |
| --- | --- | --- |
| **연산 도메인** | 공간 도메인 (픽셀 밝기 및 기울기 직접 활용) | 주파수 도메인 (2D FFT 및 교차 전력 스펙트럼 활용) |
| **연산 방식 및 속도** | **반복적 최적화 (Iterative)** — 상대적으로 느림 | **단일 패스 (Closed-form)** — 매우 빠르고 연산 시간이 일정함 |
| **수렴 범위 (Capture Range)** | **좁음** — 초기 위치가 실제 이동량과 멀면 지역 최적해(Local Minima)에 빠짐 (피라미드 기법 병행 필요) | **전역적 (Global)** — 이미지 크기의 절반까지 큰 이동량도 한 번에 정확히 검출 |
| **조명 변화 강인성** | **매우 우수** — 선형적인 대비 및 밝기 변화에 완벽히 불변 | **우수** — 위상만 정규화하므로 균일한 밝기 변화 및 주파수 노이즈에 강함 |
| **영역 마스킹 (ROI Mask)** | **지원함** — 특정 배경이나 가려짐(Occlusion) 영역을 제외한 정합 가능 | **불가** — 전체 이미지에 대해 FFT를 수행하므로 픽셀 단위 마스크 적용이 어려움 (해닝 윈도우 등만 적용) |
| **확장성** | 회전, 유클리드, 어파인, 호모그래피로 즉시 확장 가능 | 순수 평행이동만 가능 (회전/스케일 추정 시 로그-폴라 변환 추가 필요) |

---

## 4. 실무 적용 및 결합 팁

* **하이브리드 파이프라인 (Coarse-to-Fine 전략):**
* 두 이미지 간 이동량이 크고 동시에 극도의 정밀도나 ROI 마스킹이 필요한 경우, **1단계로 위상 상관 분석**을 수행해 전역적인 초기 이동량$(\Delta x_0, \Delta y_0)$을 빠르게 구하고, 이 값을 2단계 ECC 알고리즘의 초기값(`warp_matrix`)으로 입력하면 수렴 실패 없이 빠르고 정밀한 정합이 가능합니다.


* **경계 현상(Edge Effect) 주의:**
* 서브픽셀 위상 상관을 사용할 때는 푸리에 변환의 주기성 가정으로 인해 이미지 외곽 경계선이 강한 고주파 노이즈(Spectral Leakage)를 유발합니다. 반드시 입력 이미지에 **해닝 윈도우(Hanning Window)** 등을 씌워 외곽을 부드럽게 감쇠시킨 후 적용해야 정밀도가 유지됩니다.



아래의 인터랙티브 시각화 도구를 통해 두 알고리즘이 이미지 이동량과 조명 변화, 노이즈 환경에서 어떻게 다르게 작동하는지 단계별로 비교해 보세요.