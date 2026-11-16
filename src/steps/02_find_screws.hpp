#pragma once
// 2단계: 나사별 가로 이동량(px)
// 처리: 닮은 위치 찾기 -> 밝기 차이 보정 -> 테두리 위치 맞추기 -> 둘레 영역 맞추기
// 중간 실패 시 해당 나사만 제외. 머리 일부가 가려지거나 오염된 것으로 보이면 일부 가림으로 표시 (측정 실패 처리)
#include "common/reference.hpp"
#include <array>
#include <string>

namespace alignment {

struct ScrewShift {
    bool found = false;
    double dx = 0;            // 가로 이동 (px, 오른쪽이 +). 머리 테두리 맞춤 결과
    double dy = 0;            // 세로 이동(px, 아래가 +). Y, T 계산 제외
    double area_dx = 0;       // 가로 이동 (px). 나사 둘레 121×121 영역 밝기 맞춤 결과. Y, T 계산에 dx 와 함께 쓴다
    double similarity = 0;    // 처음 찾은 위치의 닮은 정도 (0~1)
    double shape_similarity = 0;   // 최종 위치에서 흐린 기준 조각과의 닮은 정도 (0~1). 일부 가림 점검용
    bool partial = false;     // 머리 일부 가림 또는 오염 의심 (found = false)
    std::string problem;      // 검출 또는 위치 맞춤 실패 사유
    std::string trace;        // 실제 수행한 하위 단계의 중간값 (진단 기록)
};

class ScrewFinder {
public:
    // ScrewFinder: 나사 기준 영상 준비
    //   입력: reference (기준 사진과 나사 위치)
    //   출력: 나사 탐색기
    explicit ScrewFinder(const Reference& reference);

    // find_screw_shifts: 나사별 이동량 측정
    //   입력: gray (회색 사진), qr_dx와 qr_dy (QR 이동량 px), wanted (사용할 나사)
    //   출력: 나사 1~4의 이동량과 검출 결과
    std::array<ScrewShift, 4> find_screw_shifts(const cv::Mat& gray, double qr_dx, double qr_dy, const std::array<bool, 4>& wanted, bool record = false) const;

    // locate_screw: 기준 사진과 가장 닮은 위치 찾기
    //   입력: gray (회색 사진), screw_index (0~3), expected_shift (예상 이동량 px)
    //   출력: 성공 여부, shift (정수 px), similarity (닮은 정도)
    bool locate_screw(const cv::Mat& gray, int screw_index, cv::Point expected_shift, cv::Point& shift, double& similarity) const;

    // refine_by_brightness: 밝기 차이를 줄여 위치 맞추기
    //   입력: gray (회색 사진), screw_index (0~3), start (초기 이동량 px)
    //   출력: 성공 여부, dx와 dy (정밀 이동량 px)
    bool refine_by_brightness(const cv::Mat& gray, int screw_index, cv::Point start, double& dx, double& dy) const;

    // refine_by_edges: 테두리 위치 맞추기
    //   입력: gray (회색 사진), screw_index (0~3), start (초기 이동량 px)
    //   출력: 성공 여부, dx와 dy (정밀 이동량 px)
    bool refine_by_edges(const cv::Mat& gray, int screw_index, cv::Point start, double& dx, double& dy) const;

    // refine_by_area: 나사 둘레 영역(121×121) 밝기 맞추기
    //   입력: gray (회색 사진), screw_index (0~3), start (초기 이동량 px)
    //   출력: 성공 여부, dx와 dy (정밀 이동량 px)
    bool refine_by_area(const cv::Mat& gray, int screw_index, cv::Point start, double& dx, double& dy) const;

private:
    // find_one: 나사 하나의 위치를 4단계로 맞추기
    //   입력: gray (회색 사진), screw_index (0~3), expected_shift (예상 이동량 px)
    //   출력: ScrewShift (이동량, 검출 여부, 실패 사유)
    ScrewShift find_one(const cv::Mat& gray, int screw_index, cv::Point expected_shift, bool record) const;

    // shape_similarity_at: 최종 위치에서 흐린 기준 조각과의 닮은 정도
    //   입력: gray (회색 사진), screw_index (0~3), dx와 dy (최종 이동량 px)
    //   출력: 닮은 정도 (-1~1). 사진 밖이면 -1
    double shape_similarity_at(const cv::Mat& gray, int screw_index, double dx, double dy) const;

    cv::Point centers_[4];
    cv::Mat locate_templates_[4];       // 초기 탐색용 나사 영역(91×91, 8비트)
    cv::Mat brightness_templates_[4];   // 밝기 비교용 기준 사진. 소수 밝기값 사용
    cv::Mat edge_templates_[4];         // 기준 사진의 나사 윤곽선 영상 (141×141)
    cv::Mat edge_mask_;                 // 테두리 맞춤에 쓰는 원 (반지름 45 px)
    cv::Mat area_templates_[4];         // 둘레 영역 맞춤용 기준 사진 (121×121, 소수)
    cv::Mat shape_templates_[4];        // 일부 가림 점검용 흐린 기준 조각 (91×91, 소수)
};

}   // alignment
