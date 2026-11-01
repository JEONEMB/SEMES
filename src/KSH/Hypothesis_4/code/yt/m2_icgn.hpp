// =====================================================================================
// m2_icgn.hpp — M2 : IC-GN 아핀 정합 (QR 별 이동 + 늘어남) 과 서브픽셀 치우침 줄이기
//
//  근거 : Pan, Li, Tong 2013 — IC-GN + ZNSSD (역합성 갱신이라 Hessian 을 한 번만 계산)
//         Liu et al. 2021 — DPG : 이웃 두 정수 중심에서 구한 서브픽셀 값을 결합해 보간 치우침 감소
//  모델 : 부분 영역 중심 기준 좌표 ξ 에서  x' = c + u + (1+ux)·ξx + uy·ξy ,  y' = c + v + vx·ξx + (1+vy)·ξy
//         ux = 가로 늘어남(변형률 ε_x) — 트랙 A 의 원근 단서
//  보간 : 직접 배정밀도로 계산 (cv::remap 은 1/32 px 고정소수점이라 쓰지 않음)
//         Linear = 선형, Cubic = Keys 3차 (a = −0.5), BSpline = 사전 필터를 거친 3차 B-스플라인 (Unser 1993 —
//         DIC 문헌에서 보간 치우침이 가장 작은 축에 속함), Dual = 선형·Keys 결과의 평균
//         (프로토타입에서는 OpenCV 3차(a = −0.75)와 선형의 치우침 부호가 반대였으나, Keys(a = −0.5)는 같은 부호 → 검증 V2-1 참고)
//  사전 필터 : Pan, Xie, Wang 2013 "Bias error reduction of digital image correlation using Gaussian pre-filtering"
//         — 고주파(나이퀴스트 근처) 성분이 보간 치우침의 원인이므로 σ ≈ 0.5~1 px 가우시안으로 두 영상을 같이 흐림
// =====================================================================================
#ifndef YT_M2_ICGN_HPP
#define YT_M2_ICGN_HPP

#include "yt_common.hpp"

namespace yt {
namespace m2 {

enum class Interp { Linear, Cubic, BSpline, Dual };
const char* interpName(Interp i);

// 채택한 설정 : B-스플라인 3차 + 가우시안 사전 필터 σ = 1.0 px (선택 근거 : 검증 V2-1, build/explore_sigma.txt)
constexpr Interp kInterp = Interp::BSpline;
constexpr double kSigma = 1.0;

struct Affine {
    double u = 0, ux = 0, uy = 0, v = 0, vx = 0, vy = 0;
};

struct Result {
    Affine p;
    double zncc = 0;   // 수렴 후 ZNCC (1 = 완전 일치)
    int iters = 0;
    bool ok = false;
};

// ref 의 (cx, cy) 중심 반폭 hw 부분 영역이 cur 에서 어떻게 옮겨지고 늘어났는지
//  sigma > 0 : 두 영상 모두 가우시안 사전 필터(Pan 2013, 보간 치우침 감소) 후 정합
Result icgn(const cv::Mat& ref, const cv::Mat& cur, int cx, int cy, int hw, const Affine& init, Interp interp,
            double sigma = 0.0, int maxIter = 60);

// DPG 시연 (1D, x 이동만) : 정수 이동 gi 를 중심으로 1단계 기울기법 / 두 정수 중심 결합
double gradientOneStep(const cv::Mat& ref, const cv::Mat& cur, int cx, int cy, int hw, int gi);
double dpg(const cv::Mat& ref, const cv::Mat& cur, int cx, int cy, int hw, int giFloor);

// 검증 : 합성(부분 화소 이동, 늘어남 주입, DPG) + 실제 사진 (m2_affine.csv 저장 — 트랙 A 입력)
bool verify(const std::vector<Sample>& ds, const std::string& outDir, const std::string& m1Csv);

} // namespace m2
} // namespace yt

#endif
