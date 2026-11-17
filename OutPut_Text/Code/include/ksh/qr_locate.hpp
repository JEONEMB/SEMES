// =====================================================================================
// ksh/qr_locate.hpp — ① QR 3개 위치 (기준 사진 없이, 설정 파일의 절대 좌표 박스로)
//
//  ①-1 대략 위치 (coarse) : QR 띠의 가로 프로파일에서 QR 사이 어두운 틈 2곳을 찾아 설정 좌표 대비 가로 이동 dx
//        QR 안 검은 모듈은 닫힘 연산(21 px)으로 메워, 밝은 사각형 사이의 틈(약 37 px)만 어둡게 남긴다
//  ①-2 정밀 위치 : 박스(설정 좌표 + dx) 안에서 밝은 사각형 4변을 행·열마다 서브픽셀(포물선 보간)로 잡고
//        변마다 강건 직선 맞춤(3σ, 3회) → 두 중간선의 교점 = 사각형 중심
//  결과 center_px.x (3개 평균) 가 특징 s, dx 는 배경 홀 박스를 옮기는 데도 쓴다
//  (2026-11-16 Python 검토판 absmeas.py 와 같은 계산 — 교보재 사진 719장 실패 0장)
// =====================================================================================
#ifndef KSH_QR_LOCATE_HPP
#define KSH_QR_LOCATE_HPP

#include <opencv2/core.hpp>

#include <string>
#include <vector>

namespace ksh {

// 설정 파일의 박스 하나 (사진 절대 좌표, 정사각형 = 중심 ± hw)
struct BoxSpec {
    std::string name;
    int cx_px = 0, cy_px = 0, hw_px = 0;
};

struct QrParams {
    std::vector<BoxSpec> boxes;      // QR_L, QR_C, QR_R (가로 순서 무관 — 내부에서 정렬)
    int search_px = 40;              // 박스 바깥으로 더 보는 여유 (사각형 변이 박스 경계 근처여도 잡히게)
    int coarse_range_px = 350;       // ①-1 대략 위치에서 찾는 가로 이동 범위 (±)
    double coarse_max_spread_px = 15;// 찾은 틈들이 설정 간격과 이 이상 어긋나면 실패
    int close_kernel_px = 21;        // QR 모듈 메우는 닫힘 연산 크기 (틈 37 px 보다 작아야 함)
    double gap_dark_frac = 0.6;      // 틈 = 프로파일 중앙값 × 이 값보다 어두운 구간
    int gap_min_px = 12, gap_max_px = 80;
    int edge_inset_px = 8;           // 변을 잡을 때 모서리 쪽은 이만큼 빼고 씀
    int edge_half_window_px = 6;     // 대략 변 위치 ± 이 폭에서 서브픽셀 변 찾기
    double blur_sigma_px = 1.0;      // 변 찾기 전 가우시안
    bool edge_gauss_peak = true;     // 서브픽셀 보간 : true = 로그 포물선(가우시안), false = 포물선 (Python 검토판과 같음)
};

struct QrResult {
    bool ok = false;
    std::string message;
    double coarse_dx_px = 0.0;               // ①-1 설정 좌표 대비 가로 이동
    std::vector<cv::Rect> boxes_px;          // 이번 사진에서 쓴 박스 (설정 + dx)
    std::vector<cv::Point2d> centers_px;     // 사각형 중심 (설정의 boxes 순서)
    cv::Point2d center_px;                   // 3개 평균 (s = center_px.x)
};

// ①-1 만 (진단·초기 세팅용). 실패하면 false 와 이유
bool coarse_qr_shift(const cv::Mat &gray8, const QrParams &params, double &dx_px, std::string &message);

// ①-2 박스 하나의 사각형 중심 (box = 사진 좌표의 중심·반폭, margin = 여유). 실패하면 false
bool qr_square_center(const cv::Mat &gray8, const BoxSpec &box, const QrParams &params, cv::Point2d &center_px, std::string &message);

// ① 전체 : gray8 = CV_8UC1
QrResult locate_qr(const cv::Mat &gray8, const QrParams &params);

} // namespace ksh

#endif
