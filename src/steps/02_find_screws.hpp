#pragma once
// 2단계: 나사 4개가 기준 사진에서 가로로 몇 px 움직였는지 잰다.
//
// 나사마다 세 번에 걸쳐 좁혀 간다
//   1. locate_screw          NCC 로 대략 어디 있는지 찾는다 (1 px 단위)
//   2. refine_by_brightness  밝기를 맞춰 가며 조금씩 밀어 위치를 다듬는다 (소수 px)
//   3. refine_by_edges       나사 테두리(밝기가 급히 바뀌는 곳)만 남긴 그림으로 정밀하게 맞춘다 (ECC)
// 마지막 3 의 결과를 쓴다. 3 은 시작 위치가 가까워야 제대로 맞기 때문에 1, 2 가 먼저 필요하다.
//
// 나사를 못 찾으면 그 나사만 found = false 로 두고 나머지는 계속 잰다.
#include "common/reference.hpp"
#include <array>
#include <string>

namespace alignment {

struct ScrewShift {
    bool found = false;
    double dx = 0;            // 가로 이동 (px, 오른쪽이 +)
    double dy = 0;            // 세로 이동 (px, 아래가 +). Y·T 계산에는 쓰지 않는다
    double similarity = 0;    // 1 에서 찾았을 때의 NCC (0~1)
    std::string problem;      // 못 찾았을 때 이유
};

class ScrewFinder {
public:
    explicit ScrewFinder(const Reference& reference);

    // qr_dx, qr_dy: 1단계에서 구한 QR 이동. 나사도 비슷하게 움직였을 것이라 그 주변부터 찾는다.
    // wanted[k] 가 false 인 나사는 찾지 않는다.
    std::array<ScrewShift, 4> find_screw_shifts(const cv::Mat& gray, double qr_dx, double qr_dy, const std::array<bool, 4>& wanted) const;

    // 아래 셋은 나사 하나를 재는 세 단계다. 단계별 값을 확인할 수 있게 따로 부를 수 있다.
    bool locate_screw(const cv::Mat& gray, int screw_index, cv::Point expected_shift, cv::Point& shift, double& similarity) const;
    bool refine_by_brightness(const cv::Mat& gray, int screw_index, cv::Point start, double& dx, double& dy) const;
    bool refine_by_edges(const cv::Mat& gray, int screw_index, cv::Point start, double& dx, double& dy) const;

private:
    ScrewShift find_one(const cv::Mat& gray, int screw_index, cv::Point expected_shift) const;

    cv::Point centers_[4];
    cv::Mat brightness_templates_[4];   // 기준 사진의 나사 조각 (91×91)
    cv::Mat edge_templates_[4];         // 기준 사진의 나사 테두리 그림 (141×141)
    cv::Mat edge_mask_;                 // 테두리 맞춤에 쓰는 원 (반지름 45 px)
};

}   // namespace alignment
