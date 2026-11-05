// estimator.cpp — 사진 한 장 → (Y, Θ) 구현
#include "ksh/estimator.hpp"

#include "ksh/image_util.hpp"

#include <chrono>

namespace ksh {

// 사진 한 장 → (Y, Θ)
//  입력 검사 → ①②③ 특징 측정(measure_features) → ④ 보정식 역산(invert_pose_model)
//  어느 단계든 실패하면 ok = false 와 이유(message)를 남기고 바로 돌아감. 처리 시간은 성공·실패 모두 기록
PoseEstimate estimate_pose(const cv::Mat &ref32, const cv::Mat &src, const FeatureParams &params, const PoseModel &model) {
    PoseEstimate out;
    const auto t0 = std::chrono::steady_clock::now();
    // 끝낼 때마다 메시지와 경과 시간을 함께 채우는 도우미
    auto finish = [&](const std::string &msg) {
        out.message = msg;
        out.elapsed_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        return out;
    };
    // 입력 검사 : 기준·현재 사진이 있고 크기가 같아야 ROI 좌표를 그대로 쓸 수 있음
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
