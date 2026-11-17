// =====================================================================================
// ksh/hole_locate.hpp — ② 뒷벽 홀(깊이가 다른 배경) 중심 (기준 사진 없이, 설정 파일의 절대 좌표 박스로)
//
//  박스 = 설정 좌표 + ①에서 구한 QR 가로 이동 dx (홀은 플레이트와 함께 화면에서 움직이고, 차이는 시차 d 뿐)
//  박스 안 : 크게 흐린 영상(σ 6)의 최소점 = 홀 대략 위치
//          → 그 둘레 (2·half+1)² 창에서 방사 대칭 중심 (Parthasarathy 2012, Nature Methods 9:724) 3회 반복
//  방사 대칭 중심 : 모든 화소의 밝기 기울기 방향 직선이 가장 가깝게 모이는 점 (가중 최소제곱, 닫힌 해)
//   → 임계값·윤곽선 없이 서브픽셀. 홀 4곳(W1~W4, 좌우·상하 대칭)의 평균을 쓴다
// =====================================================================================
#ifndef KSH_HOLE_LOCATE_HPP
#define KSH_HOLE_LOCATE_HPP

#include "ksh/qr_locate.hpp"

#include <opencv2/core.hpp>

#include <string>
#include <vector>

namespace ksh {

struct HoleParams {
    std::vector<BoxSpec> boxes;      // W1 ~ W4
    int search_px = 40;              // 박스 바깥 여유
    int window_half_px = 34;         // 방사 대칭 창 반폭 (홀 지름 약 45 px 보다 크게)
    double blur_sigma_px = 1.0;      // 방사 대칭 전 가우시안
    double locate_sigma_px = 6.0;    // 대략 위치용 큰 흐림 (locate_radius_px = 0 일 때)
    int locate_radius_px = 0;        // 대략 위치 : 0 = σ locate_sigma_px 흐림의 최소점 (Python 검토판), > 0 = 이 반지름 원판 평균이 가장 어두운 곳
    double min_contrast = 10.0;      // 박스 테두리 밝기 − 홀 최소 밝기 하한 (가림·조명 이상 검출)
    int robust_iter = 0;             // 방사 대칭 강건 반복 횟수 (0 = 끔 = Python 검토판과 같음. 2 로 하면 먼지에 강하지만 흔들림 약 8 % 증가)
    double max_line_dist_px = 4.0;   // 강건 반복 때 중심에서 이보다 먼 기울기 직선 제외 (먼지·이물 영향 차단)
    double max_center_gap_px = 12.0; // 방사 대칭 중심과 어두운 부분 무게중심의 차이 상한 (실측 최대 7.9 px — W3 는 그림자로 원래 7.7 px)
    double max_spread_px = 5.0;      // 홀 4곳의 (측정 − 박스) 가로 차이 폭 상한 (실측 최대 3.4 px @ Θ 2°. 한 곳만 엉뚱하면 실패 — features 에서 검사)
};

struct HoleResult {
    std::string name;
    bool ok = false;
    std::string message;
    cv::Rect box_px;                 // 이번 사진에서 쓴 박스 (설정 + dx)
    cv::Point2d center_px;           // 홀 중심 (서브픽셀, 사진 좌표)
    double contrast = 0.0;
    double center_gap_px = 0.0;      // 방사 대칭 중심 − 어두운 부분 무게중심 거리
};

// 방사 대칭 중심 (창 좌표, 화소 중심 = 정수 + 0.5 기준이 아닌 배열 좌표). 실패하면 false
bool radial_symmetry_center(const cv::Mat &roi32, double &xc, double &yc, int robust_iter = 0, double max_line_dist_px = 4.0);

// 박스 하나 (dx = ①의 QR 가로 이동)
HoleResult locate_hole(const cv::Mat &gray8, const BoxSpec &box, const HoleParams &params, double dx_px);

} // namespace ksh

#endif
