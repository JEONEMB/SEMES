// =====================================================================================
// ksh/fine_align.hpp — ② Fine Alignment (정밀 정렬) : 화소 이하 평행이동
//
//  icgn       : 역합성 가우스–뉴턴(IC-GN), ZNSSD 비교 (Pan, Li, Tong 2013) + 보간 커널 선택 — 채택
//  opencv_ecc : cv::findTransformECC (선형 보간 + 5×5 가우시안 고정) — 이전 방식, 비교용
//  출발점(init)이 정답 골짜기 안(±0.5 px 정도)에 있어야 수렴 → Coarse Alignment 결과를 init 으로 씀
// =====================================================================================
#ifndef KSH_FINE_ALIGN_HPP
#define KSH_FINE_ALIGN_HPP

#include "ksh/interp_kernel.hpp"

#include <opencv2/core.hpp>

namespace ksh {

enum class FineMethod { icgn, opencv_ecc };

struct FineParams {
    FineMethod method = FineMethod::icgn;
    Kernel kernel = Kernel::omoms3;
    double gauss_sigma_px = 0.0;   // 가우시안 사전 필터 σ (0 = 안 씀). 실험계획 2 에서 0 채택
    int max_iter = 60;
    double eps_px = 1e-5;          // 걸음 크기가 이보다 작으면 멈춤
    int search_margin_px = 28;     // 현재 영상에서 잘라 올 여유 (init 에서 이만큼까지만 움직일 수 있음)
    double min_zncc = 0.0;         // 정합 상관 하한 (0 = 검사 안 함). 낮으면 가림·반사 등으로 보고 실패
    int ecc_max_iter = 100;        // opencv_ecc 전용
    double ecc_eps = 1e-7;
};

struct FineResult {
    cv::Point2d shift_px;   // 기준 영역이 현재 영상에서 옮겨진 양 (현재 − 기준)
    double zncc = 0.0;      // 정규화 상관 (1 에 가까울수록 잘 맞음)
    bool converged = false; // 반복이 멈춤 조건(eps_px)을 만족해 끝났는지 (max_iter 를 다 쓰면 false)
    int iterations = 0;     // 반복 횟수 (IC-GN)
};

// 기준 영상의 영역 ref_rect_px 가 현재 영상에서 init_px 근처로 옮겨졌다고 보고 화소 이하로 다듬음
bool icgn_translation(const cv::Mat &ref32, const cv::Mat &cur32, const cv::Rect &ref_rect_px, const cv::Point2d &init_px,
                      const FineParams &params, FineResult &out);
bool ecc_translation(const cv::Mat &ref32, const cv::Mat &cur32, const cv::Rect &ref_rect_px, const cv::Point2d &init_px,
                     const FineParams &params, FineResult &out);

// params.method 에 따라 위 둘 중 하나 호출
//   계산이 끝났어도 미수렴(IC-GN) 이거나 zncc < params.min_zncc 이면 false (out 에 값은 채워 둠 → 실패 이유 확인용)
bool fine_alignment(const cv::Mat &ref32, const cv::Mat &cur32, const cv::Rect &ref_rect_px, const cv::Point2d &init_px,
                    const FineParams &params, FineResult &out);

} // namespace ksh

#endif
