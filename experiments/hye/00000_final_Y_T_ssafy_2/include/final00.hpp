#pragma once
#include "alignment.hpp"
#include <array>
#include <memory>
#include <string>

// 값 7개 구성의 1차(joint) 추정기.
//   값 7개 = QR 536점 이동의 가로 평균 1 + 위·아래 바 기울기 2 + 나사 4개의 가로 이동(기울기 영상 맞춤) 4
// 출력 (y, t) 는 절대 영점(센터 사진과 같은 화면) 기준 추정값이다. 이동 명령이 아니다.
namespace final00 {
extern const char* const kValueNames[7];

struct Measurement {
    std::array<double, 7> values;   // kValueNames 순서, px (바 기울기는 px / 1000 px)
    int valid_points;               // 고정 536점 중 유효한 점 수. 536 이어야 측정이 성립한다.
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
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
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
private:
    Measurer measurer_;
    std::array<double, 7> mean_, scale_;
    std::array<std::array<double, 2>, 8> weights_;   // 첫 행은 절편. 열은 Y, T
    std::array<double, 2> minimum_, maximum_;
};
}
