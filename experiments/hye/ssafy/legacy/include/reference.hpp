#pragma once
#include <opencv2/core.hpp>
#include <filesystem>
#include <string>

// 그날 센터 사진으로 측정 기준 자료를 다시 만든다(기준 다시 잡기).
// 카메라가 무거워 날마다 화면이 세로로 밀리면(2026-11-13: QR +14 px, 바 +8~14 px) 원래 기준 자료로는 바 측정(±11.5 px)이 실패한다.
// 방법은 experiments/hye/20261113_1345_…/실험/make_rev_assets.py 와 같다(그 실험에서 계수를 그대로 두고 정확도가 유지됨을 확인).
//   QR 상자   원래 센터의 QR 조각을 새 센터에서 NCC 로 찾은 정수 이동만큼 옮긴다(크기 그대로)
//   630점     QR 마다 원래 조각 → 새 조각 ECC homography 로 옮긴다
//   나사      원래 중심에서 출발해 새 센터의 기울기 크기 무게중심(3회)
//   바 48칸   가로 범위 그대로, 원래 기준 행 + (QR 세로 이동의 절반) 주변 ±20 행에서 같은 부호의 가장 뚜렷한 가장자리
namespace final00 {
struct ReferenceReport {
    cv::Point qr_shift[3];          // QR 상자의 정수 이동 (새 − 원래, px)
    double qr_ncc[3], qr_ecc[3];    // QR 찾기 NCC, 호모그래피 ECC 상관
    cv::Point screw_shift[4];       // 나사 중심 이동 (px)
    int bar_shift_min[2], bar_shift_max[2];   // 위, 아래 바 기준 행 이동의 범위 (px)
    std::string folder;             // 새 기준 자료 폴더
    bool absolute_kept = true;      // Estimator::set_reference: 절대 영점을 이었는가
    double bridge_y = 0, bridge_t = 0;   // Estimator::set_reference: 새 기준 출력에 더하는 값
};

// base: 원래 기준 자료 폴더(center.bmp, points.json, geometry.json, line_protocol.json, site.json).
// center: 새 센터 사진(3840×2160 8비트, 1, 3, 4채널). out: 새 기준 자료를 쓸 폴더(없으면 만든다).
// QR, 나사, 바를 찾지 못하면 예외를 던진다.
// relocate_bars=false: 바 칸을 다시 찾지 않고 원래 line_protocol.json 을 그대로 쓴다(바를 쓰지 않는 모델, 값 5개).
ReferenceReport make_reference(const std::filesystem::path& base, const cv::Mat& center, const std::filesystem::path& out, bool relocate_bars = true);
}
