// =====================================================================================
// measure.hpp  -  사진마다 QR 3개(L, C, R)가 center 사진 대비 얼마나 움직였는지 잰다
//
//   1. QR 라벨 찾기   center 사진에서 밝은 QR 라벨(흰 사각형) 3개를 찾는다
//   2. ECC 패치       라벨 + 바깥 8px 를 기준 패치로, 사진마다 템플릿 매칭 -> ECC 아핀 정렬
//                    이동 = 패치 가운데 점이 옮겨진 양
//   3. 모서리 특징점   라벨 안 QR 격자 모서리를 center 에서 찾고, 사진마다 ECC 로 예측한 자리에서
//                    "흐린 2x2 모듈" 밝기 모델을 맞춰 서브픽셀 위치를 구한다.
//                    이동 = 대각선 모서리를 뺀 점들의 평균
// =====================================================================================
#pragma once

#include <array>
#include <map>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

namespace hw {

constexpr int kQrCount = 3;
const char *const kQrNames[kQrCount] = {"L", "C", "R"};

// center 사진의 QR 하나
struct QrRef {
    cv::Rect label;          // 밝은 라벨 사각형
    cv::Rect patch;          // ECC 기준 패치 = 라벨 + 바깥 kPatchMargin
    cv::Point2d labelCenter; // 라벨 가운데
    double labelMean = 0;    // 라벨 안 평균 밝기
};
constexpr int kPatchMargin = 8;

// center 사진에서 QR 라벨 3개를 찾는다 (왼쪽부터 L, C, R). 못 찾으면 프로그램 종료.
std::array<QrRef, kQrCount> findQrLabels(const cv::Mat &centerGray);

// ECC 결과 (QR 하나)
struct EccMotion {
    double dx = 0, dy = 0;   // 패치 가운데 점의 이동 (px)
    double cc = 0;           // ECC 상관계수 (1 = center 와 똑같음)
    double rotDeg = 0;       // 아핀 변환의 회전 (도)
    double scale = 1;        // 아핀 변환의 크기 변화
    cv::Matx23d warp;        // 패치 좌표 -> 사진 좌표 (정수 오프셋 포함)
    bool ok = false;
};

class EccMeasurer {
public:
    EccMeasurer(const cv::Mat &centerGray, const std::array<QrRef, kQrCount> &qrs);
    EccMotion measure(const cv::Mat &gray, int q) const;
private:
    std::array<cv::Mat, kQrCount> templates_;   // center 패치 (float)
    std::array<QrRef, kQrCount> qrs_;
};

// QR 격자 모서리 하나
struct Corner {
    cv::Point2f p;           // center 사진에서의 위치 (서브픽셀)
    int qr = 0;              // 0=L, 1=C, 2=R
    bool diagonal = false;   // 검정 칸 두 개가 꼭짓점만 맞닿은 모서리 (흔들림이 커서 평균에서 뺌)
    std::array<int, 4> black{};   // 꼭짓점 주변 4칸이 검정인지: 왼위, 오른위, 왼아래, 오른아래
};

// 모서리 하나를 밝기 모델로 맞춘 결과
struct CornerFit {
    cv::Point2d p;           // 꼭짓점 위치
    double rmse = 0;         // 모델과 실제 밝기 차이
    double contrast = 0;     // 검정과 흰색 밝기 차
    double blur = 0;         // 경계 흐림 σ (px)
    bool ok = false;         // 통과 기준을 모두 만족
    std::string reason;      // 탈락했으면 첫 번째 이유
};
// start 근처에서 모서리를 맞춘다 (창 다시 맞추기 최대 3번)
CornerFit fitCorner(const cv::Mat &gray, const Corner &c, cv::Point2d start);

// center 사진에서 모서리를 찾는다
std::vector<Corner> findCorners(const cv::Mat &centerGray, const std::array<QrRef, kQrCount> &qrs);

// 사진 한 장에서 모서리 위치를 다시 찾는다. ECC 변환으로 자리를 예측하고 서브픽셀로 맞춘다.
// 반환: 모서리마다 위치 (못 찾으면 NaN)
std::vector<cv::Point2f> trackCorners(const cv::Mat &gray, const std::vector<Corner> &corners,
                                      const std::array<QrRef, kQrCount> &qrs,
                                      const std::array<EccMotion, kQrCount> &ecc);

}  // namespace hw
