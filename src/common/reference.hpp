#pragma once
// 기준: 기준 사진과 그 안의 QR·나사 위치. 모든 측정은 "기준에서 얼마나 움직였나" 를 잰다.
#include <opencv2/core.hpp>
#include <string>
#include <vector>

namespace alignment {

const int kPhotoWidth = 3840;
const int kPhotoHeight = 2160;

struct Reference {
    cv::Mat photo;        // 기준 사진 (회색, 3840×2160)
    cv::Rect qr[3];       // QR 상자, 왼쪽부터
    cv::Point screw[4];   // 나사 중심. 1 왼쪽 위, 2 오른쪽 위, 3 왼쪽 아래, 4 오른쪽 아래
    double trained_y_mm[2] = {0, 0};    // 계수를 학습한 Y 범위 (최소, 최대)
    double trained_t_deg[2] = {0, 0};   // 계수를 학습한 T 범위
};

// model 폴더의 center.bmp 와 reference.json 을 읽는다. 실패하면 false 와 이유.
bool load_reference(const std::string& data_folder, Reference& reference, std::string& error);

// 사진을 회색으로 바꾼다. 크기나 형식이 다르면 false 와 이유.
bool to_gray(const cv::Mat& photo, cv::Mat& gray, std::string& error);

// 여러 장의 화소 평균 (회색). 한 장이면 그대로.
bool average_gray(const std::vector<cv::Mat>& photos, cv::Mat& gray, std::string& error);

}   // namespace alignment
