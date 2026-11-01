// batch.cpp — 사진 여러 장 처리 구현
#include "ksh/batch.hpp"

#include "ksh/image_util.hpp"

#include <opencv2/imgcodecs.hpp>

#include <chrono>
#include <ostream>

namespace ksh {

std::vector<PhotoFeatures> measure_photos(const std::vector<LabeledPhoto> &photos, const cv::Mat &ref32, const FeatureParams &params,
                                          std::ostream *progress) {
    std::vector<PhotoFeatures> out;
    for (const auto &p : photos) {
        PhotoFeatures r;
        r.photo = p;
        const cv::Mat img = cv::imread(p.path, cv::IMREAD_GRAYSCALE);
        const auto t0 = std::chrono::steady_clock::now();
        if (img.empty()) r.features.message = "사진을 읽지 못함";
        else if (img.size() != ref32.size()) r.features.message = "사진 크기가 기준과 다름";
        else r.features = measure_features(ref32, to_gray32(img), params);
        r.elapsed_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        out.push_back(r);
        if (progress && out.size() % 50 == 0) *progress << "  " << out.size() << " / " << photos.size() << " 장\n" << std::flush;
    }
    return out;
}

std::vector<PhotoEstimate> estimate_photos(const std::vector<LabeledPhoto> &photos, const std::map<int, cv::Mat> &ref32_by_set,
                                           const FeatureParams &params, const PoseModel &model, std::ostream *progress) {
    std::vector<PhotoEstimate> out;
    for (const auto &p : photos) {
        const auto ref = ref32_by_set.find(p.set);
        if (ref == ref32_by_set.end()) continue;
        PhotoEstimate r;
        r.photo = p;
        const cv::Mat img = cv::imread(p.path, cv::IMREAD_GRAYSCALE);
        r.estimate = estimate_pose(ref->second, img, params, model);
        out.push_back(r);
        if (progress && out.size() % 50 == 0) *progress << "  " << out.size() << " 장\n" << std::flush;
    }
    return out;
}

std::vector<FeatureSample> to_samples(const std::vector<PhotoFeatures> &measured) {
    std::vector<FeatureSample> v;
    for (const auto &m : measured) {
        if (!m.features.ok) continue;
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
