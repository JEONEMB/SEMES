// =====================================================================================
// measurements.hpp  -  묶음의 모든 사진을 한 번 읽어서 분석에 필요한 값을 모아 둔다
//   (사진 읽기가 가장 오래 걸리므로 한 번만 읽고, 분석과 슬라이드는 이 값만 쓴다)
// =====================================================================================
#pragma once

#include "core.hpp"
#include "measure.hpp"

namespace hw {

struct ShotMeasure {
    std::array<EccMotion, kQrCount> ecc;   // QR 별 ECC
    double eccDx = kNaN, eccDy = kNaN;       // QR 3개 평균
    double cornerDx = kNaN, cornerDy = kNaN; // 대각선 아닌 모서리 평균 (모든 사진에서 추적된 점만)
    std::vector<cv::Point2f> pos;          // 모서리마다 위치 (NaN = 실패)
    double plateMean = kNaN;                // QR 라벨 3개 안 평균 밝기 (움직인 자리 기준)
    double wholeMean = kNaN;                // 사진 전체 평균 밝기
};

struct Measurements {
    std::array<QrRef, kQrCount> qrs;       // center(기준) 사진의 QR
    std::vector<Corner> corners;           // center 사진의 모서리
    std::vector<bool> cornerUsable;        // 모든 사진에서 추적에 성공한 모서리
    std::vector<ShotMeasure> shots;        // DataSet::shots 와 같은 순서
    cv::Size imageSize;
    int channels = 0;
    bool grayAsColor = false;              // 3채널 값이 모두 같은지 (기준 사진으로 판정)
};

// 묶음 전체 측정. 기준 사진은 ds.centers[0]
Measurements measureSet(const DataSet &ds);

// 측정값 CSV (사진별, 모서리별) 저장
void writeMeasureCsv(const DataSet &ds, const Measurements &m, const fs::path &dir);

}  // namespace hw
