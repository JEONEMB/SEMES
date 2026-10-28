// =====================================================================================
// m1_phase_ecc.hpp — M1 : 위상 상관 + ECC 로 영역 평행이동 측정
//
//  근거 : Dong et al. 2018 (위상 상관, 1/10~1/100 px), Evangelidis & Psarakis 2008 (ECC, ZNCC 최대화)
//  절차 : ① 플레이트 주변 900 × 400 창의 위상 상관 → 거친 이동 (모델·라벨 값 사용 안 함)
//         ② 영역마다 (벽은 2 배 창 위상 상관으로 한 번 더 보정) → ECC 평행이동으로 정밀화
//  주의 : cv::phaseCorrelate 에 창 함수를 넘기면 입력을 제자리에서 곱함 → 항상 복사본을 넘김
// =====================================================================================
#ifndef YT_M1_PHASE_ECC_HPP
#define YT_M1_PHASE_ECC_HPP

#include "yt_common.hpp"

namespace yt {
namespace m1 {

// b 가 a 대비 이동한 양 (px). 같은 크기 영상
cv::Point2d phaseShift(const cv::Mat& a, const cv::Mat& b);

struct Shift {
    cv::Point2d d;      // 기준 대비 이동 (px)
    double cc = 0.0;    // ECC 상관 (1 에 가까울수록 신뢰)
    bool ok = false;
};

// tmpl 이 img 안에서 init 만큼 이동해 있다고 보고 ECC 평행이동 정밀화 (같은 크기)
Shift eccTranslation(const cv::Mat& tmpl, const cv::Mat& img, const cv::Point2d& init);
// ECC 미수렴(예외) 누적 건수 — 측정 실패로 처리된 수 (콘솔 문구 대신 이 값으로 보고)
int eccFailCount();
void resetEccFailCount();

// 플레이트 주변 거친 이동. center = 기준 영상 좌표의 창 중심
cv::Point2d coarseShift(const cv::Mat& ref, const cv::Mat& cur, const cv::Point& center, const cv::Size& win);

// 영역 r (기준 좌표) 의 이동. wide = 2 배 창 위상 상관을 먼저 (벽), false = 바로 ECC (QR : 반복 무늬라 넓은 창 금지)
Shift measureRegion(const cv::Mat& ref, const cv::Mat& cur, const Region& r, const cv::Point2d& guess, bool wide);

struct Frame {
    cv::Point2d coarse;
    std::vector<Shift> r;   // regions() 순서
    double ms = 0;
};
Frame measureFrame(const cv::Mat& ref, const cv::Mat& cur);

// 검증 : 합성(정답 아는 이동) + 실제 사진. 결과 outDir/m1_*.csv, RESULT.md. 실제 측정은 m1_shifts.csv 로 저장 (트랙 B 입력)
bool verify(const std::vector<Sample>& ds, const std::string& outDir);

} // namespace m1
} // namespace yt

#endif
