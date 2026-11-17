#include "final00.hpp"
#include "control_internal.hpp"
#include "final04.hpp"
#include "screw_gradient.hpp"
#include "site.hpp"
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace final00 {
namespace fs = std::filesystem;
const char* const kValueNames[7] = {"qr_dx_mean", "bar_tilt_upper", "bar_tilt_lower", "screw1_x", "screw2_x", "screw3_x", "screw4_x"};

namespace {
std::vector<unsigned char> selected_points(const fs::path& assets) {
    cv::FileStorage file((assets / "points.json").u8string(), cv::FileStorage::READ);
    if (!file.isOpened()) throw std::runtime_error("points.json 이 없다");
    std::vector<unsigned char> selected;
    for (const auto& row : file["points"]) selected.push_back(static_cast<unsigned char>(int(row[3])));
    if (selected.size() != 630 || std::count(selected.begin(), selected.end(), 1) != 536)
        throw std::runtime_error("points.json: 630점 중 536점 선택이 아니다");
    return selected;
}
cv::Mat to_gray(const cv::Mat& input) {
    if (input.empty() || input.depth() != CV_8U || input.size() != cv::Size(3840, 2160))
        throw std::invalid_argument("입력은 3840x2160 8비트 영상이어야 한다");
    cv::Mat gray;
    if (input.channels() == 1) gray = input;
    else if (input.channels() == 3) cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);
    else if (input.channels() == 4) cv::cvtColor(input, gray, cv::COLOR_BGRA2GRAY);
    else throw std::invalid_argument("채널 수는 1, 3, 4 중 하나여야 한다");
    return gray;
}
}

struct Measurer::Impl {
    bool site_loaded;   // 다른 멤버보다 먼저 site.json 을 읽는다
    legacy_control::Tracker tracker;
    std::vector<unsigned char> selected;
    alignment_detail::Observer observer;
    cv::Mat center;
    final04_pin::Observer pins;
    ScrewGradient screws;
    std::vector<double> block_x;
    explicit Impl(const fs::path& assets)
        : site_loaded((site::load(assets), true)), tracker(assets), selected(selected_points(assets)), observer(assets),
          center(alignment_detail::read_gray(assets / "center.bmp")), pins(center), screws(center), block_x(observer.block_centers()) {}
};

Measurer::Measurer(const std::string& assets) : impl_(std::make_unique<Impl>(fs::u8path(assets))) {}
Measurer::~Measurer() = default;

Measurement Measurer::measure(const cv::Mat& image) const {
    const cv::Mat gray = to_gray(image);
    const auto tracks = impl_->tracker.measure(gray);
    Measurement m{};
    m.valid_points = impl_->tracker.valid_selected(tracks);
    if (m.valid_points != 536) throw std::runtime_error("QR 고정 536점 중 유효하지 않은 점이 있다 (유효 " + std::to_string(m.valid_points) + ")");
    double dx = 0;
    for (int p = 0; p < 630; ++p) if (impl_->selected[p]) dx += tracks.displacement.at<float>(p, 0);
    const auto observation = impl_->observer.measure(gray);
    const auto pin = impl_->pins.measure(gray);
    const cv::Mat f66 = final04::features(observation.values, pin.displacement, impl_->block_x);
    const cv::Mat screw = impl_->screws.measure(gray, pin.displacement);
    if (!cv::checkRange(screw)) throw std::runtime_error("나사 기울기 영상 맞춤에 실패했다");
    m.values = {dx / 536.0, f66.at<double>(55), f66.at<double>(57), screw.at<double>(0), screw.at<double>(2), screw.at<double>(4), screw.at<double>(6)};
    m.qr_correlation = observation.qr_correlation; m.line_correlation = observation.line_correlation;
    return m;
}

Estimator::Estimator(const std::string& assets) : measurer_(assets) {
    const fs::path root = fs::u8path(assets);
    cv::FileStorage f((root / "provenance.json").u8string(), cv::FileStorage::READ);
    if (!f.isOpened()) throw std::runtime_error("provenance.json 이 없다");
    std::string id; f["model_id"] >> id;
    if (id.rfind("final00_huber7_", 0) != 0 || int(f["features"]) != 7) throw std::runtime_error("모델 식별자가 다르다");   // 장비별 모델 이름을 받는다(원본은 SEMES 702장 모델 하나만 받는다)
    for (const std::string name : {"center.bmp", "points.json", "geometry.json", "line_protocol.json", "model7.tsv"}) {
        std::string expected; f[name + "_sha256"] >> expected;
        if (expected.size() != 64 || alignment_detail::sha256(root / name) != expected) throw std::runtime_error("자료 해시가 다르다: " + name);
    }
    for (const auto* key : {"training_min", "training_max"})
        if (!f[key].isSeq() || f[key].size() != 2) throw std::runtime_error("학습 범위가 없다");
    for (int k = 0; k < 2; ++k) {
        minimum_[k] = double(f["training_min"][k]); maximum_[k] = double(f["training_max"][k]);
        if (!std::isfinite(minimum_[k]) || !std::isfinite(maximum_[k]) || minimum_[k] > maximum_[k]) throw std::runtime_error("학습 범위가 잘못됐다");
    }
    std::ifstream model(root / "model7.tsv");
    std::string line; std::getline(model, line);   // 머리줄
    for (int row = 0; row < 8; ++row) {
        if (!std::getline(model, line)) throw std::runtime_error("model7.tsv 의 행이 모자란다");
        std::istringstream s(line); std::string name; double mean, scale, wy, wt;
        if (!(s >> name >> mean >> scale >> wy >> wt)) throw std::runtime_error("model7.tsv 를 읽지 못했다");
        if (name != (row == 0 ? std::string("intercept") : std::string(kValueNames[row - 1]))) throw std::runtime_error("model7.tsv 의 값 순서가 다르다: " + name);
        if (!std::isfinite(mean) || !std::isfinite(scale) || !(scale > 0) || !std::isfinite(wy) || !std::isfinite(wt)) throw std::runtime_error("model7.tsv 에 잘못된 수가 있다");
        weights_[row] = {wy, wt};
        if (row > 0) { mean_[row - 1] = mean; scale_[row - 1] = scale; }
    }
}

Result Estimator::predict(const cv::Mat& image) const {
    Result r{};
    r.measurement = measurer_.measure(image);
    double y = weights_[0][0], t = weights_[0][1];
    for (int j = 0; j < 7; ++j) {
        const double z = (r.measurement.values[j] - mean_[j]) / scale_[j];
        y += weights_[j + 1][0] * z; t += weights_[j + 1][1] * z;
    }
    r.pose = {y, t};
    r.outside_training_range = y < minimum_[0] || y > maximum_[0] || t < minimum_[1] || t > maximum_[1];
    return r;
}
}
