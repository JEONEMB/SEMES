// =====================================================================================
// phase_align.hpp — 위상(Local Phase) 기반 Y·Θ 분리 알고리즘
//
// 근거 논문 : Yinan Miao, "Vision-based Displacement Measurements using Local Image Phase
//             for Structural Dynamics Applications", 전남대 박사학위논문, 2023
//             - 2장 : 복소 Gabor 필터 위상으로 서브픽셀 변위, 비선형 위상(위상 기울기 이상) 판별
//             - 4장 : 위상 기반 광류(POF) + 신뢰도 척도(필터 응답 크기, 비선형 위상)로 나쁜 화소 제외
//
// 처리 흐름 (영상 한 장)
//   1) 거친 단계 : QR 띠 전체를 위상 상관(phaseCorrelate)으로 맞춰 정수 이동(ix, iy)을 구함
//                  → 위상이 한 주기(λ) 안에서만 모호하지 않으므로, 큰 이동(Θ 2° ≈ 160 px)을 먼저 제거
//   2) 정밀 단계 : QR 3개(L·C·R) 영역마다 "가로 방향(α = 0°) 복소 Gabor 필터"의 화소별 위상 변화로
//                  가로 이동 dx(x, y)를 구함 :  dx = ix − Δφ / (∂φ/∂x)            (논문 Eq. 8, 35)
//   3) 신뢰도    : 응답 크기가 작거나(무늬 없음), 위상 기울기가 2π/λ 에서 벗어난(비선형 위상) 화소는 제외
//   4) QR 별 맞춤: dx(x) = t + s·(x − x_c)/100   →  t = QR 이동(px), s = 가로 배율 변화(px / 100 px)
//                  Θ(팬 회전)는 원근 때문에 왼쪽·오른쪽 QR 의 배율 s 를 반대로 바꿈 → Y 와 구분되는 단서
//   5) 추정      : 특징 6개 [tL, sL, tC, sC, tR, sR] = A·[Y, Θ] + c ,  잡음 공분산 Σ
//                  → 일반화 최소제곱(GLS)으로 (Y, Θ) 추정
//   5') 3변수    : 공통 배율 변화 m(ppm, 카메라–판 거리·초점 흔들림)을 방해 변수로 함께 추정
//                  m 이 특징에 주는 영향은 기하로 정함 : t_i += m·1e-6·(x_i − x0),  s_i += m·1e-4
// =====================================================================================
#ifndef __PHASE_ALIGN_HPP__
#define __PHASE_ALIGN_HPP__

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

namespace pa {

const int NUM_QR   = 3;      // L, C, R
const int NUM_FEAT = 6;      // tL, sL, tC, sC, tR, sR

// ---------------------------------------------------------------------------------
// 매개변수 (기본값은 교보재 Y·Θ 스윕에서 비교해 고른 값)
// ---------------------------------------------------------------------------------
struct PhaseParams {
    double lambda   = 12.0;  // Gabor 물결 주기 λ (px)
    double sigmaX   = 4.0;   // 가로(물결 방향) 가우시안 σ (px)
    double sigmaY   = 6.0;   // 세로 가우시안 σ (px) — 세로로 이어진 경계 정보를 모음
    int    roiPadX  = 6;     // QR 사각형 바깥으로 넓히는 가로 여유 (바깥 경계 포함)
    int    roiInsetY= 8;     // QR 사각형 위·아래에서 안쪽으로 줄이는 행 수
    double ampPercentile = 40.0;  // 응답 크기 하위 몇 %를 버릴지 (신뢰도 척도 ①)
    double maxGradErr    = 0.30;  // |측정 위상 기울기 − 2π/λ| / (2π/λ) 허용치 (신뢰도 척도 ②, 비선형 위상)
    double maxResidualFrac = 0.40;// |dx − ix| 가 이 비율·λ 보다 크면 위상 감김으로 보고 제외
    double minValidRatio = 0.15;  // QR 영역에서 유효 화소 비율 최소값
    double maxFitStd     = 0.60;  // QR 별 맞춤 잔차 표준편차(px) 최대값
    bool   useMagnification = false; // true : [Y, Θ, 공통 배율] 3변수,  false : [Y, Θ] 2변수 (기본)
                                     //   3변수는 효과 1% 이내로 확인됨 → 비교용으로만 남김 (results/1st_stage_3var)
    int    rowSplit = 1;         // QR 행을 위에서부터 몇 조각으로 나눌지 (1 = 나누지 않음, 2 = 위·아래, 3 = 상·중·하)
    int    rowPart  = 0;         // 사용할 조각 번호 0 … rowSplit−1 (위에서부터). 분할 시험용
                                 //   조각끼리 필터 범위가 겹치지 않도록 경계마다 ±halfY() 행은 양쪽 모두 제외

    int halfX() const { return (int)std::ceil(3.0 * sigmaX); }
    int halfY() const { return (int)std::ceil(3.0 * sigmaY); }
};

// 기준 영상에서 찾은 QR 밝은 사각형
struct QRBox {
    cv::Rect    rect;
    cv::Point2f centroid;
};

// QR 하나의 측정 결과
struct QRMeasure {
    double shift     = 0;   // t : QR 가로 이동 (px, 기준 대비, +는 오른쪽)
    double scale     = 0;   // s : 가로 배율 변화 (px / 100 px)
    double validRatio= 0;   // 유효 화소 비율
    double fitStd    = 0;   // 맞춤 잔차 표준편차 (px)
    double gradSpread= 0;   // 유효 화소 위상 기울기의 상대 흩어짐 (한 장 안에서 본 SDPGM 성격의 지표)
    int    nValid    = 0;
    bool   valid     = false;
};

// 영상 한 장의 측정 결과
struct Features {
    QRMeasure qr[NUM_QR];
    double coarseDx = 0, coarseDy = 0;   // 거친 단계 위상 상관 결과 (px)
    double f[NUM_FEAT] = {0, 0, 0, 0, 0, 0};
    bool   valid = false;                 // 특징을 모두 구했는지
    int    numValidQR = 0;
};

// 선형 모델 : f = A·[Y, Θ] + c ,  잡음 공분산 Σ (학습 잔차)
struct Calibration {
    double a[NUM_FEAT] = {0};    // ∂f/∂Y (px/mm 또는 (px/100px)/mm)
    double b[NUM_FEAT] = {0};    // ∂f/∂Θ (…/deg)
    double c[NUM_FEAT] = {0};    // 절편
    cv::Mat cov;                 // NUM_FEAT × NUM_FEAT, CV_64F (3변수일 때는 배율 성분을 뺀 잔차의 공분산)
    double n[NUM_FEAT] = {0};    // 공통 배율 1 ppm 이 각 특징에 주는 변화 (기하로 계산)
    bool   useMag = false;       // 3변수 모델로 학습했는지
    bool   ok = false;

    double sigma(int k) const { return cov.empty() ? 1.0 : std::sqrt(cov.at<double>(k, k)); }
    bool save(const std::string& path, const PhaseParams& p) const;
    bool load(const std::string& path, PhaseParams* p = nullptr);
};

// 추정 결과 (기준 대비 오차)
struct Estimate {
    double y = 0, t = 0;         // mm, deg
    double sdY = 0, sdT = 0;     // 모델 잡음에서 전파된 표준편차
    double mag = 0, sdMag = 0;   // 공통 배율 변화 (ppm, 3변수일 때만)
    bool   valid = false;
};

// 학습·평가 샘플
struct Sample {
    std::string path;
    char   sweep = 'C';          // 'C' center, 'Y' Y 스윕, 'T' Θ 스윕
    double trueY = 0, trueT = 0;
    Features feat;
    double estY = 0, estT = 0;
    bool   train = false;
};

// 밝은 사각형 후보 → 피치 규칙으로 L·C·R 선택 (기준 영상에만 사용)
bool findQRBoxes(const cv::Mat& gray, QRBox boxes[NUM_QR], double pitch = 257.0);

// ---------------------------------------------------------------------------------
// 위상 기반 정렬기
// ---------------------------------------------------------------------------------
class PhaseAligner {
public:
    explicit PhaseAligner(const PhaseParams& p = PhaseParams()) : params_(p) { buildKernel(); }

    bool setReference(const cv::Mat& refImage);       // 기준(보정 완료) 영상 등록
    bool hasReference() const { return hasRef_; }
    void setParams(const PhaseParams& p);              // 매개변수 변경 (기준 위상 다시 계산)
    const PhaseParams& params() const { return params_; }

    Features measure(const cv::Mat& image) const;      // 거친 단계 + 정밀 단계 + 특징
    void setCalibration(const Calibration& c) { calib_ = c; }
    const Calibration& calibration() const { return calib_; }
    Estimate estimate(const Features& f) const;        // GLS 로 (Y, Θ)

    // 시각화용
    const QRBox* referenceBoxes() const { return refBox_; }
    // 공통 배율 1 ppm 이 특징 6개에 주는 변화 (x0 = 영상 가로 중심을 광축으로 가정)
    void magnificationDirection(double n[NUM_FEAT]) const;
    cv::Rect roi(int q) const { return roi_[q]; }

private:
    void buildKernel();
    // 영역 r 의 복소 응답을 열 x−1 … x+w 범위로 계산 (위상 기울기용 양옆 1열 포함)
    void response(const cv::Mat& g32, cv::Rect r, cv::Mat& re, cv::Mat& im) const;
    QRMeasure measureQR(const cv::Mat& g32, int q, int ix, int iy) const;

    PhaseParams params_;
    Calibration calib_;
    bool    hasRef_ = false;
    cv::Mat ref32_;
    QRBox   refBox_[NUM_QR];
    cv::Rect roi_[NUM_QR];
    cv::Rect band_;                       // 거친 단계 위상 상관 영역
    cv::Mat refBand_, hann_;
    cv::Mat refRe_[NUM_QR], refIm_[NUM_QR];
    cv::Mat kx_cos_, kx_sin_, ky_;
};

// ---------------------------------------------------------------------------------
// 학습·평가 도우미
// ---------------------------------------------------------------------------------
// magDir == nullptr : 2변수,  magDir 지정 : 3변수 (공통 배율을 방해 변수로)
Calibration fitCalibration(const std::vector<Sample>& samples, const double* magDir = nullptr);
std::vector<Sample> buildSampleList(const std::string& imgDir,
                                    double maxY = 2.0, double stepY = 0.05,
                                    double maxT = 1.0, double stepT = 0.01);

} // namespace pa

#endif // __PHASE_ALIGN_HPP__
