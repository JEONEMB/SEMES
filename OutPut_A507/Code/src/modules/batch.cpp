// batch.cpp — 사진 여러 장 처리 구현
#include "ksh/batch.hpp"

#include "ksh/image_util.hpp"

#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <mutex>
#include <ostream>
#include <thread>

namespace ksh {

std::vector<PhotoFeatures> measure_photos(const std::vector<LabeledPhoto> &photos, const FeatureParams &params, int threads,
                                          std::ostream *progress) {
    std::vector<PhotoFeatures> out(photos.size());
    std::atomic<size_t> next{0}, done{0};
    std::mutex mu;
    auto worker = [&]() {
        for (;;) {
            const size_t i = next++;
            if (i >= photos.size()) return;
            PhotoFeatures &r = out[i];
            r.photo = photos[i];
            const cv::Mat img = read_image(photos[i].path, cv::IMREAD_GRAYSCALE);
            const auto t0 = std::chrono::steady_clock::now();
            if (img.empty()) r.features.message = "사진을 읽지 못함";
            else r.features = measure_features(to_gray8(img), params);
            r.elapsed_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            const size_t d = ++done;
            if (progress && (d % 100 == 0 || d == photos.size())) {
                std::lock_guard<std::mutex> lk(mu);
                *progress << "  " << d << " / " << photos.size() << "\n" << std::flush;
            }
        }
    };
    int n = threads > 0 ? threads : int(std::max(1u, std::thread::hardware_concurrency()));
    std::vector<std::thread> th;
    for (int k = 0; k < n; ++k) th.emplace_back(worker);
    for (auto &t : th) t.join();
    return out;
}

std::vector<FeatureSample> to_samples(const std::vector<PhotoFeatures> &measured) {
    std::vector<FeatureSample> v;
    for (const auto &m : measured) {
        if (!m.features.ok || m.photo.is_reference) continue;
        FeatureSample s;
        s.stem = m.photo.stem;
        s.y_mm = m.photo.y_mm;
        s.t_deg = m.photo.t_deg;
        s.s_px = m.features.s_px;
        s.d_px = m.features.d_px;
        v.push_back(s);
    }
    return v;
}

} // namespace ksh
