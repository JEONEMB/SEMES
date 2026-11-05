// =====================================================================================
// real_compare.hpp — 운동 모델 예측을 실제 사진과 비교
//
//  영역(QR 3개 + 배경 벽)마다 기준 영상 대비 실제 이동량을 재고, 모델이 예측한 이동량과 비교한다.
//   측정 : 위상 상관(넓은 창, 거친 이동) → ECC 평행이동(정밀) — 모델 값은 쓰지 않음
//   기준 : 1차 세트 = center.bmp, 2차 세트 = center1.bmp (세트마다 그 세트의 기준으로 비교)
//  batch : 모든 사진의 영역별 (측정, 예측, 잔차) CSV + 1차 세트로 영역별 감도 회귀
//          → 감도 a_k (px/mm), b_k (px/°) 로 실효 초점거리·회전축 위치를 다시 맞춘 모델(fitted) 저장
// =====================================================================================
#ifndef MM_REAL_COMPARE_HPP
#define MM_REAL_COMPARE_HPP

#include "camera_model.hpp"
#include <opencv2/core.hpp>
#include <map>
#include <string>
#include <vector>

namespace mm {

struct ImageEntry {
    std::string path, stem;
    Pose pose;
    int set_id = 1;
    bool is_ref = false;   // center / centerN
};
std::vector<ImageEntry> scanImages(const std::string& dir);

// 비교 영역 (기준 영상 좌표, Center_check landmarks.yml 에서 가져옴)
struct Region {
    std::string name;
    int group;             // 0 = 플레이트(QR), 1 = 위쪽 벽, 2 = 아래쪽 벽 (door 쪽)
    cv::Point2d c;         // 중심 (px)
    cv::Size half;         // 반폭·반높이 (px)
};
const std::vector<Region>& defaultRegions();

struct RegionResult {
    cv::Point2d meas, pred;   // 이동량 (px)
    double ecc = 0;           // ECC 상관 (1 에 가까울수록 신뢰)
    bool ok = false;
};

struct CompareResult {
    ImageEntry e;
    std::string ref;
    cv::Point2d coarse;                 // 플레이트 주변 거친 이동 (px)
    std::vector<RegionResult> r;        // defaultRegions() 순서
};

class RealComparer {
public:
    RealComparer(const CameraModel& cam, std::string imgDir) : cam_(cam), dir_(std::move(imgDir)) {}
    // panel != nullptr 이면 비교 화면(1600 × 900)도 만든다
    bool compare(const ImageEntry& e, CompareResult& out, cv::Mat* panel = nullptr);
    void setModel(const CameraModel& cam) { cam_ = cam; }

private:
    const cv::Mat& reference(int set_id);
    double regionDepth(int group) const;
    CameraModel cam_;
    std::string dir_;
    std::map<int, cv::Mat> refs_;       // set_id → 기준 영상 (CV_32F 회색)
    std::map<int, std::string> refNames_;
};

// 전체 사진 일괄 비교 + 모델 재보정 결과 저장 (outDir 에 CSV·txt·yml)
void runBatch(const CameraModel& cam, const std::vector<ImageEntry>& entries, const std::string& imgDir,
              const std::string& outDir);

} // namespace mm

#endif
