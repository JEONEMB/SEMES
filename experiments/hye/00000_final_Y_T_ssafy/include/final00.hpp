#pragma once
#include "alignment.hpp"
#include "reference.hpp"
#include <vector>
#include <array>
#include <limits>
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
    // qr_method: "" 이면 assets/site.json 의 qr_method(없으면 "track536"). "track536"(QR 536점 추적) 또는 "ncc"(QR 3개 NCC 포물선)로 고를 수 있다.
    explicit Measurer(const std::string& assets, const std::string& qr_method = "");
    const std::string& qr_method() const;   // 실제로 쓰는 QR 방식
    ~Measurer();
    // image: 3840×2160 8비트, 1·3·4채널. 입력은 바꾸지 않는다. 측정이 성립하지 않으면 예외를 던진다.
    Measurement measure(const cv::Mat& image) const;
    // 그날 센터 사진으로 측정 기준을 다시 만든다(reference.hpp). 원래 assets 는 바꾸지 않고 임시 폴더에 새 기준 자료를 쓴다.
    // 실패하면 예외를 던지고 이전 기준이 그대로 남는다.
    ReferenceReport set_reference(const cv::Mat& center);
    // false 면 바 기울기를 재지 않는다(값 2, 3 은 0, line_correlation 은 NaN). 바를 쓰지 않는 모델(값 5개)에서 Estimator 가 끈다.
    // 기준 다시 잡기에서도 바 칸을 찾지 않는다.
    void set_use_bars(bool use) { use_bars_ = use; }
    bool use_bars() const { return use_bars_; }
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    std::string base_, current_;   // 원래 assets, 지금 측정 기준 자료 폴더
    std::string method_;           // 요청한 QR 방식("" = site.json)
    bool use_bars_ = true;
};

struct Result {
    Alignment_t pose;
    Measurement measurement;
    bool outside_training_range;
    // 측정값 일치 점검(2026-11-13, 실험 20261113_2230). 자세는 Y, T 두 개인데 값은 5~7개라, 정상이면 값들이 한 자세로 함께 설명된다.
    // health = max_j |값_j − 2차식_j(출력 자세) − 기준 잔차_j| / 정상 흔들림_j. health_threshold 를 넘으면 값 하나가 다른 값과 맞지 않는다(나사 가림·반사 등).
    // assets 에 점검식이 없으면 둘 다 NaN 이고 점검하지 않는다. 화면 전체가 같이 바뀌는 변화(회전, 가로 밀림, 판–나사 밀림)는 잡지 못한다(영점 업데이트의 몫).
    double health = std::numeric_limits<double>::quiet_NaN();
    double health_threshold = std::numeric_limits<double>::quiet_NaN();
    int health_worst = -1;           // 가장 맞지 않는 값의 kValueNames 번호
    bool health_failed() const { return health > health_threshold; }   // NaN 이면 false
};

class Estimator {
public:
    // assets: Measurer 의 자료 + model7.tsv + provenance.json. 자료 해시가 provenance 와 다르면 예외를 던진다.
    // qr_method: "" 이면 site.json 의 기본 방식과 model7.tsv. 다른 방식을 고르면 provenance.json 의 alt_models.<방식> 계수를 쓴다(없으면 예외).
    explicit Estimator(const std::string& assets, const std::string& qr_method = "");
    const std::string& qr_method() const { return measurer_.qr_method(); }
    const std::string& model_id() const { return model_id_; }
    // 출력 = 계수 출력 + 측정 기준 보정(bridge) − 영점(zero). 기본(아무 등록 없음)은 절대 영점: assets 의 center.bmp 와 같은 화면이 (0, 0).
    Result predict(const cv::Mat& image) const;

    // [영점 업데이트] centers(1장 이상, 여러 장이면 화소 평균)의 지금 출력을 새 영점으로 기억하고 이후 출력에서 뺀다.
    // 돌려주는 값 = 그 사진의 절대 영점 기준 자세(그날 센터가 절대 영점에서 떨어진 양). 측정 기준은 바꾸지 않는다.
    std::array<double, 2> set_zero(const std::vector<cv::Mat>& centers);
    void set_zero_value(double y, double t) { zero_ = {y, t}; }   // 저장해 둔 영점(절대 영점 기준)을 다시 넣는다
    void clear_zero() { zero_ = {0, 0}; }                          // 절대 영점으로 되돌린다
    std::array<double, 2> zero() const { return zero_; }

    // [측정 기준 다시 잡기] 화면이 크게 밀려 측정이 실패할 때만 쓴다. 측정 기준(센터 사진, QR·나사·바 위치)을 centers 로 다시 만든다.
    // 계수와 영점은 그대로다. 절대 영점을 잇기 위해 같은 사진을 바꾸기 전 기준으로도 재서 그 차이(bridge)를 더한다.
    // 바꾸기 전 기준으로 잴 수 없으면 절대 영점을 잃는다: 그 사진이 (0, 0) 이 되고 영점은 지워진다(report.absolute_kept = false).
    ReferenceReport set_reference(const std::vector<cv::Mat>& centers);
    bool absolute_kept() const { return absolute_kept_; }
    // 측정값 일치 점검식이 있는지. set_zero 와 set_reference 는 그 센터 사진의 잔차를 기준 잔차로 삼는다(그날 화면 상태를 0 으로).
    bool has_health() const { return !health_.empty(); }
private:
    struct HealthRow { int value; std::array<double, 6> c; double scale; };   // 값 = c·(1, Y, T, Y², YT, T²)
    std::array<double, 2> raw(const Measurement& m) const;
    std::array<double, 7> residual(const Measurement& m) const;   // 점검에 쓰는 값만 채운다(나머지 0)
    std::vector<HealthRow> health_;
    double health_threshold_ = std::numeric_limits<double>::quiet_NaN();
    std::array<double, 7> health_offset_{};
    cv::Mat average(const std::vector<cv::Mat>& centers) const;
    Measurer measurer_;
    std::array<double, 2> bridge_{{0, 0}}, zero_{{0, 0}};
    bool absolute_kept_ = true;
    std::string model_id_;
    std::array<double, 7> mean_, scale_;
    std::array<std::array<double, 2>, 8> weights_;   // 첫 행은 절편. 열은 Y, T
    std::array<double, 2> minimum_, maximum_;
};
}
