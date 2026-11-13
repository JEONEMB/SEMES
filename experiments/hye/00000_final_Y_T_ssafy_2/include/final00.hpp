#pragma once
#include "alignment.hpp"
#include "reference.hpp"
#include <vector>
#include <array>
#include <memory>
#include <string>

// 값 7개 구성의 1차(joint) 추정기.
//   값 7개 = QR 3개 NCC 이동(정수 + 포물선)의 가로 평균 1 + 위·아래 바 기울기 2 + 나사 4개의 가로 이동(기울기 영상 맞춤) 4
//   (2026-11-13 전에는 QR 값이 QR 536점 추적 이동의 가로 평균이었다. 바꾸면서 계수를 다시 학습했다)
// 출력 (y, t) 는 절대 영점(센터 사진과 같은 화면) 기준 추정값이다. 이동 명령이 아니다.
namespace final00 {
extern const char* const kValueNames[7];

struct Measurement {
    std::array<double, 7> values;   // kValueNames 순서, px (바 기울기는 px / 1000 px)
    int valid_points;               // 측정한 QR 수(3). 예전에는 고정 536점 중 유효한 점 수였다.
    double qr_correlation, line_correlation;
};

// 사진 한 장에서 값 7개를 잰다. 모델 없이 쓸 수 있다(학습 자료를 만들 때 쓴다).
class Measurer {
public:
    // assets: center.bmp, points.json, geometry.json, line_protocol.json 이 있는 폴더(UTF-8 경로)
    explicit Measurer(const std::string& assets);
    ~Measurer();
    // image: 3840×2160 8비트, 1·3·4채널. 입력은 바꾸지 않는다. 측정이 성립하지 않으면 예외를 던진다.
    Measurement measure(const cv::Mat& image) const;
    // 그날 센터 사진으로 측정 기준을 다시 만든다(reference.hpp). 원래 assets 는 바꾸지 않고 임시 폴더에 새 기준 자료를 쓴다.
    // 실패하면 예외를 던지고 이전 기준이 그대로 남는다.
    ReferenceReport set_reference(const cv::Mat& center);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    std::string base_, current_;   // 원래 assets, 지금 측정 기준 자료 폴더
};

struct Result {
    Alignment_t pose;
    Measurement measurement;
    bool outside_training_range;
};

class Estimator {
public:
    // assets: Measurer 의 자료 + model7.tsv + provenance.json. 자료 해시가 provenance 와 다르면 예외를 던진다.
    explicit Estimator(const std::string& assets);
    Result predict(const cv::Mat& image) const;
    // 기준 다시 잡기: centers 는 그날 센터 사진 1장 이상(여러 장이면 화소 평균을 쓴다).
    // 측정 기준을 그 사진으로 다시 만들고, 그 사진의 출력이 (0, 0) 이 되도록 이후 출력에서 그 값을 뺀다. 계수는 그대로다.
    // 부르지 않으면 assets 의 절대 영점 그대로 동작한다.
    ReferenceReport set_reference(const std::vector<cv::Mat>& centers);
    bool has_reference() const { return has_reference_; }
    // 새 기준에서 등록한 센터 사진을 계수로 계산한 값(값이 모두 0 이라 계수의 절편 몫). 이후 출력에서 뺀다.
    std::array<double, 2> reference_offset() const { return offset_; }
private:
    std::array<double, 2> raw(const Measurement& m) const;
    Measurer measurer_;
    std::array<double, 2> offset_{{0, 0}};
    bool has_reference_ = false;
    std::array<double, 7> mean_, scale_;
    std::array<std::array<double, 2>, 8> weights_;   // 첫 행은 절편. 열은 Y, T
    std::array<double, 2> minimum_, maximum_;
};
}
