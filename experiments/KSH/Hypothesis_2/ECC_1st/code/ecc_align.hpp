// =====================================================================================
// ecc_align.hpp — 가설 H2 : ECC 전역 원근 정합으로 Y·Θ 분리
//
//  H2 정의 : QR 3개를 모두 포함하는 띠 전체를 ECC(정규화 상관) 정합 한 번으로 맞추면,
//            원근(키스톤) 성분을 약 700 px 의 긴 기준선 위에서 직접 추정할 수 있어
//            Θ 측정 잡음이 위상(Gabor) 방식(QR 별 약 200 px 기준선)보다 작아지고 Y·Θ 를 더 정확히 구분한다.
//  검증 결과 요약은 results/h2_ecc/H2_결과.md
//
//  근거 논문
//    - Evangelidis & Psarakis (2008), ECC : 정규화 상관을 최대화하는 정합 (OpenCV findTransformECC)
//    - Pan, Li, Tong (2013), IC-GN + ZNSSD : 같은 계열, 밝기·대비 변화에 강하고 화소 전체를 한 번에 맞춤
//    - Sutton et al. (2008) : Θ(팬) 는 화면에서 이동 + 원근(위치에 따라 다른 배율)으로 나타남
//    - Grédiac et al. (2016) : 배율(변형률) 잡음 ∝ 1 / (기준선 길이)²  → 띠 전체를 한 번에 맞추는 근거
//
//  처리 흐름 (영상 한 장)
//    1) 거친 단계 : 기준 띠와 현재 띠의 위상 상관(phaseCorrelate) → 정수 수준 이동 (Θ 2° ≈ 160 px 도 처리)
//    2) 정밀 단계 (method 로 선택, QR 영역 마스크만 사용)
//       method 1 (기본, "심화 ECC") : ECC 와 같은 정규화 상관(ZNCC) 기준을 원근 다항 워프로 직접 최적화
//            u(x,y) = p0 + p1·x̂ + p2·x̂² + p3·ŷ + p4·x̂ŷ        (가로 이동 : 이동·배율·원근 곡률·기울기)
//            v(x,y) = p5 + p6·ŷ + p7·x̂ + p8·x̂ŷ               (세로 이동 : 키스톤 포함)
//            x̂, ŷ = 띠 중심 기준, 띠 반폭으로 나눈 좌표 → 매개변수 크기가 비슷해 수치적으로 안정
//            워프가 매개변수에 선형이라 double 정밀도 Gauss-Newton 이 수 회 만에 수렴 (Pan 2013 IC-GN 계열)
//       method 0 : OpenCV findTransformECC 호모그래피 (비교용. float 계산·수렴 판정 때문에 작은 이동에서
//                  배율 값이 반복 설정에 따라 0.005 ~ 0.01 px/100px 흔들림 → Θ 신호와 같은 크기)
//    3) 특징      : 호모그래피로 QR 3개 중심의 가로 이동 t 와 가로 배율 s 를 계산
//                   f = [tL, sL, tC, sC, tR, sR]   (t : px,  s : px / 100 px)
//    4) 추정      : f = a·Y + b·Θ + q·Θ² + c ,  잡음 공분산 Σ  → 일반화 최소제곱(Gauss-Newton)으로 (Y, Θ)
//
//  OpenCV 3.4.15 에 있는 함수만 사용 (findTransformECC, phaseCorrelate, FileStorage)
// =====================================================================================
#ifndef __ECC_ALIGN_HPP__
#define __ECC_ALIGN_HPP__

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

namespace ea {

static const int NUM_QR   = 3;
static const int NUM_FEAT = 6;   // [tL, sL, tC, sC, tR, sR]

// ---------------------------------------------------------------------------------
// 매개변수 (모델 파일에 함께 저장 → MFC 앱도 같은 값으로 측정)
// ---------------------------------------------------------------------------------
struct EccParams {
    int    method     = 1;      // 1 : ZNCC Gauss-Newton 다항 워프 (심화 ECC, 기본),  0 : OpenCV ECC 호모그래피
    int    bandPadX   = 150;    // 띠 : 바깥 QR 에서 가로로 넓히는 여유 (px)
    int    bandPadY   = 24;     // 띠 : QR 위·아래 여유 (px)
    int    searchX    = 240;    // 현재 영상에서 더 넓게 잘라 올 가로 여유 (Θ ±2° ≈ ±160 px 이동 대비)
    int    searchY    = 48;     // 세로 여유
    int    maskPad    = 10;     // ECC 마스크 : QR 사각형 바깥으로 넓히는 폭 (QR 테두리 경계 포함)
    double blurSigma  = 1.0;    // 정합 전 가우시안 σ (method 1 : 기울기 계산 안정화.  method 0 은 0 권장 — 내부 5×5 블러)
    int    maxIter    = 50;     // 최대 반복 (method 0 은 100 권장)
    double eps        = 1e-6;   // 수렴 기준 : method 1 = QR 중심 이동 변화(px),  method 0 = 상관 증가량 (1e-8 권장)
    double derivStep  = 50.0;   // 배율 s 계산 시 중심 좌우 거리 (px)
    double qrPitch    = 257.0;  // QR 중심 간격 (px) — 기준 영상에서 QR 찾기
};

// 기준 영상에서 찾은 QR 밝은 사각형 (영상 좌표)
struct QRBox {
    cv::Rect    rect;
    cv::Point2f centroid;
};

// 영상 한 장의 측정 결과
struct Measure {
    bool    ok = false;
    double  coarseDx = 0, coarseDy = 0;   // 거친 이동 (px)
    cv::Mat H;                            // 3×3 CV_64F : 기준 띠 좌표 → 현재 영상의 같은 띠 좌표
    double  cc = 0;                       // 상관 계수 (1 에 가까울수록 잘 맞음)
    int     iters = 0;                    // 반복 횟수 (method 1)
    double  p[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};   // 다항 워프 매개변수 (method 1)
    double  f[NUM_FEAT] = {0, 0, 0, 0, 0, 0};
    double  vC = 0;                       // 가운데 QR 세로 이동 (px, 참고용)
    double  ms = 0;                       // 측정 시간
};

// 보정 모델 : f = a·Y + b·Θ + q·Θ² + c,  Σ = 잔차 공분산
struct Model {
    EccParams p;
    double a[NUM_FEAT] = {0}, b[NUM_FEAT] = {0}, q[NUM_FEAT] = {0}, c[NUM_FEAT] = {0};
    cv::Mat cov;                 // NUM_FEAT × NUM_FEAT, CV_64F
    bool   quad = true;          // Θ² 항 사용 여부
    bool   ok = false;
    int    nTrain = 0;
    std::string note;            // 학습 데이터 설명

    double sigma(int k) const { return cov.empty() ? 1.0 : std::sqrt(cov.at<double>(k, k)); }
    bool save(const std::string& path) const;
    bool load(const std::string& path);
};

// 추정 결과 (기준 대비)
struct Estimate {
    double y = 0, t = 0;         // mm, deg
    double sdY = 0, sdT = 0;     // 모델 잡음에서 전파된 표준편차
    bool   valid = false;
};

// 학습용 샘플
struct Sample {
    std::vector<double> f;       // 특징 6개
    double y = 0, t = 0;         // 정답
};

bool findQRBoxes(const cv::Mat& gray8, QRBox boxes[NUM_QR], double pitch);

// ---------------------------------------------------------------------------------
// ECC 정렬기
// ---------------------------------------------------------------------------------
class EccAligner {
public:
    explicit EccAligner(const EccParams& p = EccParams()) : p_(p) {}

    void setParams(const EccParams& p) { p_ = p; }
    const EccParams& params() const { return p_; }

    bool setReference(const cv::Mat& image);           // 기준(Y = 0, Θ = 0) 영상 등록
    bool hasReference() const { return hasRef_; }
    Measure measure(const cv::Mat& image) const;        // 거친 단계 + ECC + 특징

    const QRBox* boxes() const { return box_; }
    cv::Rect band() const { return band_; }             // 띠 (영상 좌표)
    cv::Point2f qrCenter(int q) const;                   // QR 중심 (띠 좌표)

private:
    bool measureEcc(const cv::Mat& in32, double ox, double oy, Measure& m) const;   // method 0
    bool measureGN(const cv::Mat& in32, double ox, double oy, Measure& m) const;    // method 1

    EccParams p_;
    bool      hasRef_ = false;
    QRBox     box_[NUM_QR];
    cv::Rect  band_;
    cv::Mat   ref32_;            // 기준 띠 (CV_32F)
    cv::Mat   mask_;             // 기준 띠 마스크 (CV_8U)
    cv::Mat   hann_;
    // method 1 : 마스크 화소 목록 (띠 좌표), 정규화 좌표, 기준 밝기(평균 0, 노름 1)
    std::vector<cv::Point2f> px_;
    std::vector<double> xh_, yh_, tn_;
    double xc_ = 0, yc_ = 0, S_ = 1;     // 정규화 : x̂ = (x − xc) / S
};

// 학습 : 최소제곱으로 a, b, q, c 와 잔차 공분산
Model fitModel(const std::vector<Sample>& train, const EccParams& p, bool quad, const std::string& note);
// 추정 : GLS (Θ² 항이 있으면 Gauss-Newton 반복)
Estimate estimate(const Model& m, const double f[NUM_FEAT]);

} // namespace ea

#endif // __ECC_ALIGN_HPP__
