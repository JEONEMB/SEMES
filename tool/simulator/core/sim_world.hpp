// =====================================================================================
// sim_world.hpp — 실제 사진으로 만든 '가상 장면' : 운동장(motion field) + 파노라마 + 사진 합성
//
//  운동장 : 기준 사진을 80 px 격자로 나누고, 격자마다 실제 사진(1차 첫 번째 측정 : Θ 스윕 + Y 스윕)에서
//          잰 이동을 자세의 식으로 맞춤
//              dx = c + aY·Y + bT·Θ + qT·Θ²     (dy 도 같은 꼴)
//          → 깊이(시차), 회전축 어긋남, 원근 휨, 렌즈 왜곡이 모두 '측정값' 으로 들어감 (물리 가정 없음)
//          → 직선 모서리(벽 구조 가로선 등)는 선에 수직인 성분만 1차원으로 재서 넣음 (조리개 문제)
//          → 질감이 없는 격자(민무늬 벽)는 이웃 값으로 채움. 질감이 없으면 이동이 틀려도 사진이 거의 같음
//  파노라마 : 기준 사진(center)을 가운데 두고, 바깥 테두리는 다른 자세의 사진을 운동장으로 되돌려 채운 넓은 사진
//  합성 : 자세 (Y, Θ) → 각 화소가 파노라마의 어디를 보는지 역사상 → cv::remap → 촬영 잡음 추가 → 8 bit
// =====================================================================================
#ifndef SIM_WORLD_HPP
#define SIM_WORLD_HPP

#include "sim_config.hpp"

#include <opencv2/core.hpp>
#include <functional>
#include <string>
#include <vector>

namespace sim {

struct Pose {
    double y_mm = 0.0;
    double t_deg = 0.0;
};

// 실제 사진 한 장 (파일 이름에서 읽은 라벨)
struct Photo {
    std::string stem, path;
    Pose pose;
    int set = 1;          // 1 = 1차 (10/04), 2 = 2차 (10/15), 3 = 3차 (폴더 3rd)
    bool is_ref = false;  // center, centerN
};
bool parseLabel(const std::string& stem, Photo& p);
std::vector<Photo> scanPhotos(const std::string& dir);            // 하위 폴더까지, .bmp 우선 (경로 = UTF-8)
const Photo* findPhoto(const std::vector<Photo>& v, const std::string& stem);
// 1차 첫 번째 측정 여부 (Y 0.05 mm 홀수 값만 약 48분 뒤 두 번째 측정 → 원점이 0.2 px 어긋나 제외)
bool firstSession(const Photo& p);

// 진행 알림 (0~1, 문구). 작업 스레드에서 불림
using Progress = std::function<void(double, const std::string&)>;

class World {
public:
    // 실제 사진으로 운동장·파노라마 만들기 (수 분). coarse_* : 거친 이동 창 (alignment.yml 과 같은 값)
    bool build(const SimConfig& cfg, const cv::Point& coarse_center_px, const cv::Size& coarse_window_px,
               const Progress& progress, std::string& message);
    bool save(const std::string& dir) const;   // panorama.png, panorama_covered.png, field.yml, field_report.md
    bool load(const std::string& dir, const SimConfig& cfg);
    bool ready() const { return !canvas_.empty() && !coef_[0].empty(); }

    // 자세 → 합성 사진 (8UC1, 기준 사진 크기). rng 가 nullptr 이면 잡음 없음
    cv::Mat render(const Pose& pose, cv::RNG* rng) const;
    // 기준 사진(center) 그대로 + 새 촬영 잡음 — 모델을 학습할 때와 같은 기준 사진
    cv::Mat renderReference(cv::RNG* rng) const;
    // 현재 사진의 네 모서리가 파노라마의 어디인지 (파노라마 좌표, 화면 표시용)
    std::vector<cv::Point2f> footprint(const Pose& pose) const;
    // 기준 사진 화소 p 의 이동 (현재 − 기준) — 상수항 포함 여부는 설정 값
    cv::Point2d displacement(const cv::Point2d& p, const Pose& pose) const;

    const cv::Mat& panorama() const { return canvas_; }           // 8UC1
    const cv::Mat& covered() const { return covered_; }           // 8UC1 : 255 = 실제 사진으로 채움, 0 = 가장자리 복제
    cv::Point offset() const { return cv::Point(mx_, my_); }      // 파노라마에서 기준 사진 왼쪽 위
    cv::Size frameSize() const { return frame_; }
    // 격자 계수 지도 (CV_32F, gy × gx) : k = 0 c, 1 aY, 2 bT, 3 qT2 / axis 0 = dx, 1 = dy
    const cv::Mat& coef(int axis, int k) const { return coef_[axis * 4 + k]; }
    const cv::Mat& valid() const { return valid_; }               // 8UC1 : 255 = 측정, 0 = 채움
    const cv::Mat& residual() const { return resid_; }            // CV_32F : 맞춤 잔차 RMS (px)
    const cv::Mat& edge1d() const { return edge_; }               // 8UC1 : 1 = dx 만 측정(세로 모서리), 2 = dy 만 측정(가로 모서리), 0 = 아님
    int step() const { return step_; }
    // 영점 사진 : 참 자세 (0, 0) 이 이 사진. refPose = 그 사진의 1차 지령 좌표 자세 (1차 center 면 0)
    const std::string& referencePhoto() const { return refName_; }
    Pose referencePose() const { return refPose_; }
    std::string report() const { return report_; }

private:
    cv::Mat canvas_, covered_;
    cv::Mat coef_[8];            // 기준 사진 격자 (gy × gx)
    cv::Mat coefCanvas_[8];      // 파노라마 격자 (여유 칸 포함, 가장자리 복제)
    cv::Mat valid_, resid_, edge_;
    std::string refName_ = "center";
    Pose refPose_;
    cv::Size frame_{3840, 2160};
    int step_ = 80, mx_ = 160, my_ = 80;
    bool useOffset_ = true;
    double noiseSigma_ = 0.71, gain_ = 1.0;
    std::string report_;

    void makeCanvasField();
    // 파노라마 격자 좌표에서 이동 (양선형 보간)
    cv::Point2d sampleCanvas(double ux_ref, double vy_ref, const Pose& pose, bool withOffset) const;
    // 현재 사진 격자마다 '기준 사진 좌표 − 현재 좌표' (역사상) — 저해상도
    void inverseLow(const Pose& pose, cv::Mat& ix, cv::Mat& iy) const;
};

} // namespace sim

#endif
