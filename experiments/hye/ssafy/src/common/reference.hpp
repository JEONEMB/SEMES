#pragma once
// 기준 사진, QR 영역, 나사 중심 좌표
#include <opencv2/core.hpp>
#include <string>
#include <vector>

namespace alignment {

const int kPhotoWidth = 3840;
const int kPhotoHeight = 2160;
const cv::Rect kPhotoRect(0, 0, kPhotoWidth, kPhotoHeight);   // 사진 전체

// inside_photo: 사진 경계 확인
//   입력: area (확인할 사각 영역)
//   출력: 영역 전체가 사진 안이면 true
inline bool inside_photo(const cv::Rect& area) { return (area & kPhotoRect) == area; }

struct Reference {
    cv::Mat photo;        // 기준 사진 (회색, 3840×2160)
    cv::Rect qr[3];       // QR 사각 영역, 왼쪽부터
    cv::Point screw[4];   // 나사 중심. 1 왼쪽 위, 2 오른쪽 위, 3 왼쪽 아래, 4 오른쪽 아래
    double trained_y_mm[2] = {0, 0};    // 계산식을 만드는 데 사용한 Y 범위 (최소, 최대)
    double trained_t_deg[2] = {0, 0};   // 계산식을 만드는 데 사용한 T 범위
};

// load_reference: 기준 자료 읽기
//   입력: data_folder (center.bmp, reference.json)
//   출력: 성공 여부, reference (기준 사진과 위치), error (실패 사유)
bool load_reference(const std::string& data_folder, Reference& reference, std::string& error);

// to_gray: 회색 영상 변환
//   입력: photo (3840x2160, 밝기 0~255의 회색 또는 컬러 BGR, BGRA)
//   출력: 성공 여부, gray (회색 사진), error (실패 사유)
bool to_gray(const cv::Mat& photo, cv::Mat& gray, std::string& error);

// average_gray: 여러 사진의 같은 위치 밝기 평균
//   입력: photos (같은 크기의 입력 사진 목록)
//   출력: 성공 여부, gray (평균 회색 사진), error (실패 사유)
bool average_gray(const std::vector<cv::Mat>& photos, cv::Mat& gray, std::string& error);

}   // alignment
