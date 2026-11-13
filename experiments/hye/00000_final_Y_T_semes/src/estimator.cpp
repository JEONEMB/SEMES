#include "final00.hpp"
#include "control_internal.hpp"
#include "observer_internal.hpp"
#include "pin_observer.hpp"
#include "qr_ncc.hpp"
#include "final04.hpp"
#include "screw_gradient.hpp"
#include "site.hpp"
#include "reference.hpp"
#include <atomic>
#include <chrono>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <vector>
#include <limits>

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
    alignment_detail::Observer observer;   // 센터 사진 해시 검사와 바 48칸
    cv::Mat center;
    QrNcc qr;                              // QR 값 qr_method = "ncc" (우리 장비): QR 3개 NCC 포물선
    std::unique_ptr<legacy_control::Tracker> tracker;   // QR 값 qr_method = "track536" (SEMES): 536점 추적 평균
    std::vector<unsigned char> selected;
    final04_pin::Observer pins;
    ScrewGradient screws;
    std::vector<double> block_x;
    std::string method;                    // 실제로 쓰는 QR 방식
    // method: "" 이면 site.json 의 qr_method(없으면 track536), 아니면 "track536" 또는 "ncc"
    Impl(const fs::path& assets, const std::string& requested)
        : site_loaded((site::load(assets), true)), observer(assets),
          center(alignment_detail::read_gray(assets / "center.bmp")), qr(center), pins(center), screws(center), block_x(observer.block_centers()),
          method(requested.empty() ? site::get().qr_method : requested) {
        if (method != "track536" && method != "ncc") throw std::invalid_argument("QR 방식은 track536 또는 ncc 여야 한다: " + method);
        if (method == "track536") { tracker = std::make_unique<legacy_control::Tracker>(assets); selected = selected_points(assets); }
    }
};

Measurer::Measurer(const std::string& assets, const std::string& qr_method)
    : impl_(std::make_unique<Impl>(fs::u8path(assets), qr_method)), base_(assets), current_(assets), method_(qr_method) {}
const std::string& Measurer::qr_method() const { return impl_->method; }

ReferenceReport Measurer::set_reference(const cv::Mat& center) {
    static std::atomic<unsigned> counter{0};
    const auto stamp = std::chrono::system_clock::now().time_since_epoch().count();
    const fs::path dir = fs::temp_directory_path() / ("final00_reference_" + std::to_string(stamp) + "_" + std::to_string(counter++));
    ReferenceReport rep;
    try {
        rep = make_reference(fs::u8path(base_), center, dir, use_bars_);   // 언제나 원래 assets 에서 출발한다(이어 붙이지 않는다)
        auto next = std::make_unique<Impl>(dir, method_);     // site.json 을 새 값으로 읽는다. QR 방식은 그대로
        impl_ = std::move(next); current_ = dir.u8string();
    } catch (...) {
        site::load(fs::u8path(current_));                     // 이전 측정 기준의 화면 위치로 되돌린다
        throw;
    }
    return rep;
}
Measurer::~Measurer() = default;

Measurement Measurer::measure(const cv::Mat& image) const {
    const cv::Mat gray = to_gray(image);
    // QR 값: QR 3개 NCC 이동(정수 + 3점 포물선)의 가로 평균. 예전 536점 추적(약 29 ms, 초기화 약 0.4 초)을 대신한다(실험 20261113_1700).
    // SEMES 사진에서는 NCC 포물선의 잡음(0.015~0.022 px)이 536점 추적보다 커서 주 검증 탈락 0.1 → 2.6 장이 됐다. 그래서 site.json 의 qr_method 로 고른다.
    QrShift q{0, 0, 1};
    Measurement m{};
    if (impl_->tracker) {
        const auto tracks = impl_->tracker->measure(gray);
        m.valid_points = impl_->tracker->valid_selected(tracks);
        if (m.valid_points != 536) throw std::runtime_error("QR 고정 536점 중 유효하지 않은 점이 있다 (유효 " + std::to_string(m.valid_points) + ")");
        for (int p = 0; p < 630; ++p) if (impl_->selected[p]) { q.dx += tracks.displacement.at<float>(p, 0); q.dy += tracks.displacement.at<float>(p, 1); }
        q.dx /= 536.0; q.dy /= 536.0; q.min_ncc = *std::min_element(tracks.ncc.begin(), tracks.ncc.end());
    } else {
        q = impl_->qr.measure(gray);
        m.valid_points = 3;
    }
    // 바: 48칸 세로 이동만 잰다. 예전에는 observer.measure 로 값 7개에 쓰지 않는 QR NCC·ECC 호모그래피 3개까지 계산했다
    // (Windows 1스레드에서 약 700 ms 중 대부분). 바 값과 기울기 계산은 그대로라 결과가 같다.
    cv::Mat qr102 = cv::Mat::zeros(1, 102, CV_64F);   // final04::features 의 입력 형식(앞 54칸은 쓰지 않는 QR 값, 뒤 48칸은 바)
    double line_correlation = std::numeric_limits<double>::quiet_NaN();
    if (use_bars_) {   // 바를 쓰지 않는 모델(값 5개)에서는 재지 않는다: 바 기울기 값은 0
        cv::Mat lines; impl_->observer.measure_lines(gray, lines, line_correlation);
        lines.copyTo(qr102.colRange(54, 102));
    }
    // 나사 시작 위치: QR NCC 평균 이동 주변만 찾는다(예전에는 ±260 px 전체, 약 34 ms). 못 찾으면 전체에서 다시 찾는다.
    const auto pin = impl_->pins.measure(gray, cv::Point(int(std::lround(q.dx)), int(std::lround(q.dy))));
    const cv::Mat f66 = final04::features(qr102, pin.displacement, impl_->block_x);
    const cv::Mat screw = impl_->screws.measure(gray, pin.displacement);
    if (!cv::checkRange(screw)) throw std::runtime_error("나사 기울기 영상 맞춤에 실패했다");
    m.values = {q.dx, f66.at<double>(55), f66.at<double>(57), screw.at<double>(0), screw.at<double>(2), screw.at<double>(4), screw.at<double>(6)};
    m.qr_correlation = q.min_ncc;   // QR 3개 NCC 봉우리의 최솟값
    m.line_correlation = line_correlation;
    return m;
}

Estimator::Estimator(const std::string& assets, const std::string& qr_method) : measurer_(assets, qr_method) {
    const fs::path root = fs::u8path(assets);
    cv::FileStorage f((root / "provenance.json").u8string(), cv::FileStorage::READ);
    if (!f.isOpened()) throw std::runtime_error("provenance.json 이 없다");
    std::string id; f["model_id"] >> id;
    if (id.rfind("final00_huber7_", 0) != 0 || int(f["features"]) != 7) throw std::runtime_error("모델 식별자가 다르다");   // 장비별 모델 이름을 받는다(원본은 SEMES 702장 모델 하나만 받는다)
    // QR 방식마다 계수가 다르다. site.json 의 기본 방식이면 model7.tsv, 다른 방식이면 provenance 의 alt_models.<방식> 파일을 쓴다.
    std::string model_file = "model7.tsv";
    // 측정값 일치 점검식: 기본 방식은 health.tsv(provenance 의 health.tsv_sha256), 다른 방식은 alt_models.<방식>.health_file. 없으면 점검하지 않는다.
    std::string health_file, health_sha;
    if (!f["health.tsv_sha256"].empty()) { health_file = "health.tsv"; f["health.tsv_sha256"] >> health_sha; }
    if (measurer_.qr_method() != site::read(root).qr_method) {
        const auto alt = f["alt_models"][measurer_.qr_method()];
        if (alt.empty()) throw std::runtime_error("이 assets 에는 QR 방식 " + measurer_.qr_method() + " 의 계수가 없다");
        std::string file, sha; alt["file"] >> file; alt["sha256"] >> sha; alt["model_id"] >> id;
        if (file.empty() || sha.size() != 64 || alignment_detail::sha256(root / file) != sha) throw std::runtime_error("자료 해시가 다르다: " + file);
        model_file = file;
        health_file.clear(); health_sha.clear();
        if (!alt["health_file"].empty()) { alt["health_file"] >> health_file; alt["health_sha256"] >> health_sha; }
    }
    if (!health_file.empty()) {
        if (health_sha.size() != 64 || alignment_detail::sha256(root / health_file) != health_sha) throw std::runtime_error("자료 해시가 다르다: " + health_file);
        std::ifstream h(root / health_file); std::string line; bool header = false;
        while (std::getline(h, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (line.empty() || line[0] == '#') continue;
            std::istringstream s(line); std::string name; s >> name;
            if (name == "threshold") { s >> health_threshold_; continue; }
            if (name == "name") { header = true; continue; }
            HealthRow r{}; r.value = -1;
            for (int j = 0; j < 7; ++j) if (name == kValueNames[j]) r.value = j;
            for (double& c : r.c) s >> c;
            s >> r.scale;
            if (!header || r.value < 0 || !s || !(r.scale > 0)) throw std::runtime_error(health_file + " 를 읽지 못했다: " + line);
            health_.push_back(r);
        }
        if (health_.empty() || !(health_threshold_ > 0)) throw std::runtime_error(health_file + " 에 점검식이 없다");
    }
    model_id_ = id;
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
    std::ifstream model(root / model_file);
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
    // 바 기울기 두 줄의 가중치가 모두 0 이면 바를 쓰지 않는 모델(값 5개)이다. 바를 재지 않는다(바 측정 실패, 바 칸 찾기가 없어진다).
    if (weights_[2][0] == 0 && weights_[2][1] == 0 && weights_[3][0] == 0 && weights_[3][1] == 0) measurer_.set_use_bars(false);
}

std::array<double, 2> Estimator::raw(const Measurement& m) const {
    double y = weights_[0][0], t = weights_[0][1];
    for (int j = 0; j < 7; ++j) {
        const double z = (m.values[j] - mean_[j]) / scale_[j];
        y += weights_[j + 1][0] * z; t += weights_[j + 1][1] * z;
    }
    return {y, t};
}

std::array<double, 7> Estimator::residual(const Measurement& m) const {
    std::array<double, 7> r{};
    const auto p = raw(m);   // 측정 기준 보정·영점 빼기 전 출력(점검식을 이 출력으로 만들었다)
    const double y = p[0], t = p[1], term[6] = {1, y, t, y * y, y * t, t * t};
    for (const auto& h : health_) {
        double v = 0; for (int k = 0; k < 6; ++k) v += h.c[k] * term[k];
        r[h.value] = m.values[h.value] - v;
    }
    return r;
}

cv::Mat Estimator::average(const std::vector<cv::Mat>& centers) const {
    if (centers.empty()) throw std::invalid_argument("센터 사진이 없다");
    cv::Mat sum = cv::Mat::zeros(2160, 3840, CV_64F);
    for (const auto& c : centers) { cv::Mat g = to_gray(c), d; g.convertTo(d, CV_64F); sum += d; }
    cv::Mat mean8; sum.convertTo(mean8, CV_8U, 1.0 / double(centers.size()));   // 화소 평균(반올림)
    return mean8;
}

std::array<double, 2> Estimator::set_zero(const std::vector<cv::Mat>& centers) {
    const cv::Mat img = average(centers);
    const Measurement m = measurer_.measure(img);        // 측정이 실패하면 예외, 영점은 그대로
    const auto p = raw(m);
    zero_ = {p[0] + bridge_[0], p[1] + bridge_[1]};
    health_offset_ = residual(m);   // 그날 화면 상태(카메라 처짐 등으로 생긴 일정한 잔차)를 점검의 0 으로 삼는다
    return zero_;
}

ReferenceReport Estimator::set_reference(const std::vector<cv::Mat>& centers) {
    const cv::Mat img = average(centers);
    bool old_ok = false; std::array<double, 2> before{0, 0};
    try { const auto p = raw(measurer_.measure(img)); before = {p[0] + bridge_[0], p[1] + bridge_[1]}; old_ok = true; } catch (const std::exception&) {}
    ReferenceReport rep = measurer_.set_reference(img);   // 실패하면 예외, 이전 상태 그대로
    const Measurement m_after = measurer_.measure(img);
    const auto after = raw(m_after);
    health_offset_ = residual(m_after);   // 새 측정 기준에서 이 센터 사진의 잔차를 점검의 0 으로 삼는다
    if (old_ok) bridge_ = {before[0] - after[0], before[1] - after[1]};   // 같은 사진이 바꾸기 전과 같은 값이 되게
    else { bridge_ = {-after[0], -after[1]}; zero_ = {0, 0}; absolute_kept_ = false; }   // 절대 영점을 잃음: 이 사진이 (0, 0)
    rep.absolute_kept = absolute_kept_; rep.bridge_y = bridge_[0]; rep.bridge_t = bridge_[1];
    return rep;
}

Result Estimator::predict(const cv::Mat& image) const {
    Result r{};
    r.measurement = measurer_.measure(image);
    const auto p = raw(r.measurement);
    const double ay = p[0] + bridge_[0], at = p[1] + bridge_[1];   // 절대 영점 기준(측정 기준을 다시 잡았어도)
    r.pose = {ay - zero_[0], at - zero_[1]};
    r.outside_training_range = ay < minimum_[0] || ay > maximum_[0] || at < minimum_[1] || at > maximum_[1];
    if (!health_.empty()) {
        const auto res = residual(r.measurement);
        r.health = 0; r.health_threshold = health_threshold_;
        for (const auto& h : health_) {
            const double z = std::abs(res[h.value] - health_offset_[h.value]) / h.scale;
            if (z > r.health) { r.health = z; r.health_worst = h.value; }
        }
    }
    return r;
}
}
