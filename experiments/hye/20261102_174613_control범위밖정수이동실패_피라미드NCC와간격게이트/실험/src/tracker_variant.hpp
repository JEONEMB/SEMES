// 채택 control tracker.cpp 의 복사본. 정수 이동 단계만 전략으로 바꿨다.
// 정밀화(LK 15x15 정역 + 17x17 스플라인 GN 12회)와 유효 판정은 원본과 같은 코드다.
// 원본 모듈은 건드리지 않는다.
#pragma once
#include "control_internal.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <string>
#include <filesystem>

namespace variant {
namespace lc = legacy_control;
namespace fs = std::filesystem;
static const cv::Rect full_crop(1340, 800, 1130, 530);
static const cv::Rect qr[] = {cv::Rect(1552, 969, 184, 198), cv::Rect(1809, 971, 184, 198), cv::Rect(2067, 973, 185, 198)};

enum class Strategy { Single, Escalate, Pyramid };

struct Tracks {
    cv::Mat displacement, quality;
    std::vector<unsigned char> valid;
    std::array<double, 3> ncc{};
    std::array<cv::Point, 3> shift{};
    int shift_spread = 0;
    int template_matches = 0;   // matchTemplate 호출 수 (비용 대리값)
    double coarse_pixels = 0;   // 탐색한 화소 수 합계
    bool gate_ok = true;
    std::string gate_reason;
};

static std::vector<lc::PointModel> reference_models(const cv::Mat& template8, const std::vector<cv::Point2f>& points) {
    cv::Mat image; template8.convertTo(image, CV_64F); cv::GaussianBlur(image, image, cv::Size(), .8);
    cv::Mat gx(image.size(), CV_64F), gy(image.size(), CV_64F);
    for (int y = 0; y < image.rows; ++y) for (int x = 0; x < image.cols; ++x) {
        gx.at<double>(y, x) = x == 0 ? image.at<double>(y, 1) - image.at<double>(y, 0) : x == image.cols - 1 ? image.at<double>(y, x) - image.at<double>(y, x - 1) : .5 * (image.at<double>(y, x + 1) - image.at<double>(y, x - 1));
        gy.at<double>(y, x) = y == 0 ? image.at<double>(1, x) - image.at<double>(0, x) : y == image.rows - 1 ? image.at<double>(y, x) - image.at<double>(y - 1, x) : .5 * (image.at<double>(y + 1, x) - image.at<double>(y - 1, x));
    }
    auto tc = lc::nearest_prefilter(image), xc = lc::nearest_prefilter(gx), yc = lc::nearest_prefilter(gy);
    std::vector<lc::PointModel> out;
    for (auto point : points) {
        lc::PointModel model; model.point = point; model.reference = cv::Mat(289, 1, CV_64F);
        cv::Mat design(289, 4, CV_64F); int i = 0;
        for (int y = -8; y <= 8; ++y) for (int x = -8; x <= 8; ++x, ++i) {
            double xx = double(point.x) + x + 12, yy = double(point.y) + y + 12;
            double value = lc::spline_sample(tc, xx, yy);
            model.reference.at<double>(i) = value;
            auto d = design.ptr<double>(i);
            d[0] = lc::spline_sample(xc, xx, yy); d[1] = lc::spline_sample(yc, xx, yy); d[2] = value; d[3] = 1;
        }
        cv::invert(design, model.pinverse, cv::DECOMP_SVD); out.push_back(model);
    }
    return out;
}

class Tracker {
public:
    static constexpr double kMinNcc = .6;   // joint observer.cpp 와 같은 문턱
    static constexpr int kShiftSpread = 8;  // 세 QR 정수 이동의 허용 불일치 px
    static constexpr int kScale = 8;        // 피라미드 축소 배율
    static constexpr int kFineRadius = 12;  // 원본 해상도 정밀 탐색 반경
    // QR 3개는 257, 258 px 간격의 주기 배열이다. 한 주기만큼 밀면 QR1 이 QR2 자리에 맞으므로
    // QR 무늬만으로는 이동 s 와 s+257 을 구별할 수 없다. 그래서 거친 탐색을 한 주기 안으로 제한한다.
    // 이것은 구현 편의가 아니라 관측의 원리적 한계다. 더 넓히려면 QR 해독이 필요하다.
    static constexpr int kCoarseRadiusX = 250;  // < 257 (QR 간격)
    static constexpr int kCoarseRadiusY = 120;

    explicit Tracker(const fs::path& assets) {
        reference_ = lc::read_gray(assets / "center.bmp"); lc::checked_gray(reference_);
        cv::FileStorage file((assets / "points.json").u8string(), cv::FileStorage::READ);
        if (!file.isOpened()) throw std::runtime_error("points.json 을 열 수 없다");
        auto rows = file["points"];
        if (!rows.isSeq() || rows.size() != 630) throw std::runtime_error("고정 630점이 아니다");
        for (auto it = rows.begin(); it != rows.end(); ++it) {
            auto row = *it;
            double x = double(row[0]), y = double(row[1]);
            int k = int(row[2]), selected = int(row[3]);
            int id = int(points_.size());
            points_.emplace_back(x, y); regions_.push_back(k);
            selected_.push_back(static_cast<unsigned char>(selected));
            cv::Point2f crop(float(x - full_crop.x), float(y - full_crop.y));
            auto local = qr[k].tl() - full_crop.tl();
            panels_[k].points.emplace_back(crop.x - float(local.x), crop.y - float(local.y));
            panels_[k].ids.push_back(id);
        }
        if (std::count(selected_.begin(), selected_.end(), 1) != 536) throw std::runtime_error("선택 536점이 아니다");
        for (int k = 0; k < 3; ++k) {
            auto original = reference_(qr[k]).clone();
            cv::GaussianBlur(original, panels_[k].blurred8, cv::Size(), .8);
            panels_[k].models = reference_models(original, panels_[k].points);
            cv::resize(original, coarse_template_[k], cv::Size(), 1. / kScale, 1. / kScale, cv::INTER_AREA);
        }
        zero_ = measure(reference_, Strategy::Single).displacement.clone();
    }

    int valid_selected(const Tracks& t) const {
        int n = 0; for (int i = 0; i < 630; ++i) if (selected_[i] && t.valid[i]) ++n; return n;
    }

    Tracks measure(const cv::Mat& gray, Strategy strategy) const {
        lc::checked_gray(gray);
        Tracks out{cv::Mat::zeros(630, 2, CV_32F), cv::Mat::zeros(630, 5, CV_32F), std::vector<unsigned char>(630, 0)};
        auto ref = reference_(full_crop), cur = gray(full_crop);
        cv::Point coarse(0, 0);
        if (strategy == Strategy::Pyramid) coarse = coarse_pyramid(cur, out);
        for (int k = 0; k < 3; ++k) {
            auto b = qr[k]; b.x -= full_crop.x; b.y -= full_crop.y;
            switch (strategy) {
                case Strategy::Single:   match(cur, ref, b, k, out, 110, 22, {0, 0}); break;
                case Strategy::Escalate: escalate(cur, ref, b, k, out); break;
                case Strategy::Pyramid:  match(cur, ref, b, k, out, kFineRadius, kFineRadius, coarse); break;
            }
            refine(cur, b, k, out);
        }
        apply_gate(out);
        if (!cv::checkRange(out.displacement)) throw std::runtime_error("비유한 변위");
        return out;
    }

    const cv::Mat& zero() const { return zero_; }
    const std::vector<unsigned char>& selected() const { return selected_; }

private:
    void match(const cv::Mat& cur, const cv::Mat& ref, const cv::Rect& b, int k, Tracks& out,
               int pad_x, int pad_y, cv::Point center) const {
        cv::Rect search(b.x - pad_x + center.x, b.y - pad_y + center.y, b.width + 2 * pad_x, b.height + 2 * pad_y);
        search &= cv::Rect(0, 0, cur.cols, cur.rows);
        if (search.width <= b.width || search.height <= b.height) throw std::runtime_error("탐색창이 영상을 벗어났다");
        cv::Mat correlation; cv::matchTemplate(cur(search), ref(b), correlation, cv::TM_CCOEFF_NORMED);
        cv::Point location; cv::minMaxLoc(correlation, nullptr, &out.ncc[k], nullptr, &location);
        out.shift[k] = location + search.tl() - b.tl();
        ++out.template_matches;
        out.coarse_pixels += double(search.width) * search.height;
    }

    // S_B. joint observer.cpp:54 와 같은 단계 확장.
    void escalate(const cv::Mat& cur, const cv::Mat& ref, const cv::Rect& b, int k, Tracks& out) const {
        for (int pad : {110, 260, 512}) {
            match(cur, ref, b, k, out, pad, pad, {0, 0});
            if (out.ncc[k] >= kMinNcc) return;
        }
    }

    // S_C 1단계. 세 QR 의 1/8 NCC 지도를 공통 이동 좌표계로 옮겨 더한 뒤 봉우리 하나를 찾는다.
    // 판이 거의 강체로 움직이므로 세 QR 은 거의 같은 정수 이동을 공유한다. 더해서 찾으면
    // 일관성이 구조적으로 강제되고, QR 반복 무늬의 개별 가짜 봉우리는 서로 상쇄된다.
    cv::Point coarse_pyramid(const cv::Mat& cur, Tracks& out) const {
        cv::Mat small_cur; cv::resize(cur, small_cur, cv::Size(), 1. / kScale, 1. / kScale, cv::INTER_AREA);
        cv::Mat maps[3]; int ox[3], oy[3];
        for (int k = 0; k < 3; ++k) {
            auto b = qr[k]; b.x -= full_crop.x; b.y -= full_crop.y;
            cv::matchTemplate(small_cur, coarse_template_[k], maps[k], cv::TM_CCOEFF_NORMED);
            ox[k] = b.x / kScale; oy[k] = b.y / kScale;
            ++out.template_matches;
            out.coarse_pixels += double(small_cur.cols) * small_cur.rows;
        }
        // 이동 s 에 대한 점수 = 세 지도의 (기준위치 + s) 값 합
        int lo_x = -kCoarseRadiusX / kScale, hi_x = kCoarseRadiusX / kScale + 1;
        int lo_y = -kCoarseRadiusY / kScale, hi_y = kCoarseRadiusY / kScale + 1;
        for (int k = 0; k < 3; ++k) { lo_x = std::max(lo_x, -ox[k]); hi_x = std::min(hi_x, maps[k].cols - ox[k]); lo_y = std::max(lo_y, -oy[k]); hi_y = std::min(hi_y, maps[k].rows - oy[k]); }
        double best = -1e9; cv::Point peak(0, 0);
        for (int sy = lo_y; sy < hi_y; ++sy) for (int sx = lo_x; sx < hi_x; ++sx) {
            double total = 0;
            for (int k = 0; k < 3; ++k) total += maps[k].at<float>(sy + oy[k], sx + ox[k]);
            if (total > best) { best = total; peak = cv::Point(sx, sy); }
        }
        return cv::Point(peak.x * kScale, peak.y * kScale);
    }

    void apply_gate(Tracks& out) const {
        double lowest = *std::min_element(out.ncc.begin(), out.ncc.end());
        int spread = 0;
        for (int i = 0; i < 3; ++i) for (int j = i + 1; j < 3; ++j)
            spread = std::max(spread, std::max(std::abs(out.shift[i].x - out.shift[j].x),
                                               std::abs(out.shift[i].y - out.shift[j].y)));
        out.shift_spread = spread;
        if (spread > kShiftSpread) { out.gate_ok = false; out.gate_reason = "QR 간 정수 이동 불일치 " + std::to_string(spread) + " px"; }
        else if (lowest < kMinNcc) { out.gate_ok = false; out.gate_reason = "최소 NCC 낮음"; }
    }

    // 아래는 tracker.cpp 의 정밀화 단계를 그대로 옮긴 것이다.
    void refine(const cv::Mat& cur, const cv::Rect& b, int k, Tracks& out) const {
        cv::Rect target_box = b + out.shift[k];
        if ((target_box & cv::Rect(0, 0, cur.cols, cur.rows)) != target_box) throw std::runtime_error("QR 잘라내기가 영상 밖");
        cv::Mat target = cur(target_box).clone(), blurred;
        cv::GaussianBlur(target, blurred, cv::Size(), .8);
        const auto& panel = panels_[k];
        std::vector<cv::Point2f> next = panel.points, back = panel.points;
        std::vector<unsigned char> fwd, rev; std::vector<float> e1, e2;
        auto criteria = cv::TermCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 60, .0001);
        cv::calcOpticalFlowPyrLK(panel.blurred8, blurred, panel.points, next, fwd, e1, cv::Size(15, 15), 0, criteria, cv::OPTFLOW_USE_INITIAL_FLOW, 1e-5);
        cv::calcOpticalFlowPyrLK(blurred, panel.blurred8, next, back, rev, e2, cv::Size(15, 15), 0, criteria, cv::OPTFLOW_USE_INITIAL_FLOW, 1e-5);
        cv::Mat double_target; target.convertTo(double_target, CV_64F);
        cv::GaussianBlur(double_target, double_target, cv::Size(), .8);
        auto coefficients = lc::spline_coefficients(double_target);
        size_t n = panel.points.size();
        std::vector<cv::Point2d> d(n), initial(n), step(n);
        std::vector<cv::Mat> values(n), parameters(n);
        for (size_t i = 0; i < n; ++i) {
            auto delta = next[i] - panel.points[i];
            d[i] = initial[i] = cv::Point2d(delta.x, delta.y);
            values[i] = cv::Mat(289, 1, CV_64F);
        }
        for (int iteration = 0; iteration < 12; ++iteration) {
            double largest = 0;
            for (size_t i = 0; i < n; ++i) {
                auto point = panel.points[i]; int pixel = 0;
                for (int y = -8; y <= 8; ++y) for (int x = -8; x <= 8; ++x)
                    values[i].at<double>(pixel++) = lc::spline_sample(coefficients, double(point.x) + x + d[i].x, double(point.y) + y + d[i].y);
                parameters[i] = panel.models[i].pinverse * values[i];
                double gain = std::max(parameters[i].at<double>(2), .25);
                step[i] = {std::clamp(parameters[i].at<double>(0) / gain, -.5, .5), std::clamp(parameters[i].at<double>(1) / gain, -.5, .5)};
                d[i] -= step[i]; largest = std::max(largest, cv::norm(step[i]));
            }
            if (largest < 2e-5) break;
        }
        for (size_t i = 0; i < n; ++i) {
            int id = panel.ids[i];
            out.displacement.at<float>(id, 0) = float(d[i].x) + float(out.shift[k].x);
            out.displacement.at<float>(id, 1) = float(d[i].y) + float(out.shift[k].y);
            auto fb = back[i] - panel.points[i];
            double fb_error = std::sqrt(double(fb.x * fb.x + fb.y * fb.y)), last = cv::norm(step[i]);
            bool finite = std::isfinite(d[i].x) && std::isfinite(d[i].y);
            out.valid[id] = fwd[i] && rev[i] && fb_error < .5 && finite && cv::norm(d[i] - initial[i]) < 1.5 && last < .02;
            cv::Mat residual = values[i] - parameters[i].at<double>(2) * panel.models[i].reference - parameters[i].at<double>(3);
            auto q = out.quality.ptr<float>(id);
            q[0] = float(cv::norm(residual) / 17.); q[1] = float(last);
            q[2] = float(parameters[i].at<double>(2)); q[3] = float(parameters[i].at<double>(3)); q[4] = float(fb_error);
        }
    }

    cv::Mat reference_, zero_;
    std::vector<cv::Point2d> points_; std::vector<int> regions_;
    std::vector<unsigned char> selected_;
    lc::PanelModel panels_[3]; cv::Mat coarse_template_[3];
};
}
