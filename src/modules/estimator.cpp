// estimator.cpp — 사진 한 장 → (Y, Θ) 구현
#include "ksh/estimator.hpp"

#include "ksh/image_util.hpp"

#include <chrono>

namespace ksh {

PoseEstimate estimate_pose(const cv::Mat &ref32, const cv::Mat &src, const FeatureParams &params, const PoseModel &model) {
    PoseEstimate out;
    const auto t0 = std::chrono::steady_clock::now();
    auto finish = [&](const std::string &msg) {
        out.message = msg;
        out.elapsed_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        return out;
    };
    if (ref32.empty()) return finish("기준 사진 없음");
    if (src.empty()) return finish("입력 사진이 비어 있음");
    if (src.size() != ref32.size()) return finish("입력 사진 크기가 기준 사진과 다름");

    out.features = measure_features(ref32, to_gray32(src), params);   // ①②③
    if (!out.features.ok) return finish(out.features.message);
    if (!invert_pose_model(model, out.features.s_px, out.features.d_px, out.y_mm, out.t_deg))   // ④
        return finish("모델 계수가 Y·Θ 를 구별하지 못함");
    out.ok = true;
    return finish("");
}

} // namespace ksh
