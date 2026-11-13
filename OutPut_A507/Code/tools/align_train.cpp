// =====================================================================================
// align_train.cpp — 보정식(training.output, 기본 model_s1d2.yml) 학습 + 정확도 평가 도구
//
//  사용 : align_train [config/alignment.yml]
//   1) training.train / training.test 의 세트마다 특징 s, d 를 잰다
//      (feature_csv 가 있으면 그 파일을 읽고, 없으면 image_dir 의 사진을 재서 result_dir/<이름>_features.csv 로 저장)
//   2) train 세트 전체로 s·d 식을 학습 (Θ 차수 = training.theta_order_s/d)
//   3) 채점 : 학습 사진 · Θ 순 블록 교차검증 · test 세트 (학습에 안 쓴 사진)
//   4) training.output 에 모델 저장, result_dir 에 보고서(report.md)와 사진별 결과 CSV
//  라벨 = 파일 이름 (dataset.cpp 의 parse_label). 센터 사진(center, center_after_NNNN …)은 학습·채점에서 뺌
// =====================================================================================
#include "ksh/align_config.hpp"
#include "ksh/batch.hpp"
#include "ksh/dataset.hpp"
#include "ksh/evaluation.hpp"

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace fs = std::filesystem;

namespace {

// 세트 하나의 표본 : CSV 가 있으면 읽고, 없으면 사진을 잼
bool collect(const ksh::TrainingSet &set, const ksh::AlignConfig &cfg, std::vector<ksh::FeatureSample> &out, std::ostream &log) {
    std::error_code ec;
    if (!set.feature_csv.empty() && fs::exists(fs::u8path(set.feature_csv), ec)) {
        if (!ksh::load_feature_csv(set.feature_csv, out)) { log << "  [" << set.name << "] 특징 CSV 형식 오류 : " << set.feature_csv << "\n"; return false; }
        log << "  [" << set.name << "] 특징 CSV " << out.size() << "장 : " << set.feature_csv << "\n";
        return true;
    }
    const auto photos = ksh::list_labeled_photos(set.image_dir);
    if (photos.empty()) { log << "  [" << set.name << "] 라벨 사진이 없음 : " << set.image_dir << "\n"; return false; }
    log << "  [" << set.name << "] 사진 " << photos.size() << "장 측정 : " << set.image_dir << "\n";
    const auto measured = ksh::measure_photos(photos, cfg.features, cfg.training.threads, &std::cout);
    int fail = 0;
    double ms = 0;
    for (const auto &m : measured) {
        ms += m.elapsed_ms;
        if (!m.features.ok) { ++fail; log << "    측정 실패 " << m.photo.stem << " : " << m.features.message << "\n"; }
    }
    log << "    실패 " << fail << "장, 평균 " << std::fixed << std::setprecision(1) << ms / std::max<size_t>(1, measured.size()) << " ms/장"
        << std::defaultfloat << "\n";
    out = ksh::to_samples(measured);
    const std::string csv = (fs::u8path(cfg.training.result_dir) / fs::u8path(set.name + "_features.csv")).u8string();
    if (ksh::save_feature_csv(csv, out)) log << "    특징 저장 : " << csv << "\n";
    return true;
}

// 모델로 예측
std::vector<ksh::PosePrediction> predict(const std::vector<ksh::FeatureSample> &samples, const ksh::PoseModel &m, const std::string &group) {
    std::vector<ksh::PosePrediction> v;
    for (const auto &s : samples) {
        ksh::PosePrediction p;
        p.stem = s.stem;
        p.group = group;
        p.y_mm = s.y_mm;
        p.t_deg = s.t_deg;
        p.ok = ksh::invert_pose_model(m, s.s_px, s.d_px, p.y_est_mm, p.t_est_deg);
        v.push_back(p);
    }
    return v;
}

} // namespace

int main(int argc, char **argv) {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif
    const std::string cfg_path = argc > 1 ? argv[1] : "config/alignment.yml";
    ksh::AlignConfig cfg;
    std::string error;
    if (!ksh::load_align_config(cfg_path, cfg, error, false)) { std::cerr << error << "\n"; return 1; }
    const ksh::TrainingConfig &T = cfg.training;
    if (T.train.empty() || T.output.empty()) { std::cerr << "training.train / training.output 이 설정에 없음\n"; return 1; }
    std::error_code ec;
    fs::create_directories(fs::u8path(T.result_dir), ec);

    std::ostringstream log;
    log << "# align_train 보고서\n\n설정 : " << cfg_path << "\n\n## 1. 특징 측정\n\n```\n";
    std::vector<ksh::FeatureSample> train;
    for (const auto &s : T.train) {
        std::vector<ksh::FeatureSample> v;
        if (!collect(s, cfg, v, log)) { std::cerr << log.str(); return 1; }
        train.insert(train.end(), v.begin(), v.end());
    }
    std::vector<std::pair<std::string, std::vector<ksh::FeatureSample>>> tests;
    for (const auto &s : T.test) {
        std::vector<ksh::FeatureSample> v;
        if (!collect(s, cfg, v, log)) { std::cerr << log.str(); return 1; }
        tests.push_back({s.name, v});
    }
    log << "```\n\n";

    // 학습
    ksh::PoseModel model;
    if (!ksh::fit_pose_model(train, T.theta_order_s, T.theta_order_d, model)) { std::cerr << "학습 실패 (표본 부족)\n"; return 1; }
    std::ostringstream note;
    note << "align_train : ";
    for (const auto &s : T.train) note << s.name << " ";
    note << "(" << train.size() << "장), s = QR 중심 x, d = s − 나사 x 평균, Θ 차수 s" << T.theta_order_s << " d" << T.theta_order_d;
    if (!ksh::save_pose_model(T.output, model, note.str(), int(train.size()))) { std::cerr << "모델 저장 실패 : " << T.output << "\n"; return 1; }

    // 채점
    std::vector<ksh::PosePrediction> all = predict(train, model, "학습 (학습에 쓴 사진)");
    const auto folds = ksh::block_folds(train, T.cv_blocks, T.cv_blocks);
    for (auto p : ksh::cross_validate(train, folds, T.theta_order_s, T.theta_order_d)) { p.group = "블록 교차검증 (Θ 순 " + std::to_string(T.cv_blocks) + "묶음)"; all.push_back(p); }
    for (const auto &t : tests) {
        auto v = predict(t.second, model, "검증 " + t.first + " (학습에 안 쓴 사진)");
        all.insert(all.end(), v.begin(), v.end());
    }
    const auto groups = ksh::summarize_groups(all, cfg.tolerance, cfg.spec_tolerance, 100.0);

    auto eq = [](const char *name, const ksh::FeatureModel &f) {
        std::ostringstream o;
        o << std::setprecision(8) << name << " = " << f.c_px << " + " << f.a_px_per_mm << "·Y";
        for (size_t k = 0; k < f.b_px_per_deg_k.size(); ++k) o << " + " << f.b_px_per_deg_k[k] << "·Θ" << (k ? (k == 1 ? "²" : "^" + std::to_string(k + 1)) : "");
        o << "   (σ " << f.sigma_px << " px)\n";
        return o.str();
    };
    log << "## 2. 보정식 (" << T.output << ")\n\n```\n" << eq("s", model.s) << eq("d", model.d)
        << "```\n\n## 3. 정확도\n\n" << ksh::summary_table(groups, cfg.tolerance, cfg.spec_tolerance) << "\n";
    const std::string rep = (fs::u8path(T.result_dir) / "report.md").u8string();
    const std::string csv = (fs::u8path(T.result_dir) / "predictions.csv").u8string();
    { std::ofstream o(fs::u8path(rep)); o << log.str(); }
    ksh::save_predictions_csv(csv, all, cfg.tolerance, cfg.spec_tolerance);
    std::cout << log.str() << "보고서 : " << rep << "\n사진별 : " << csv << "\n모델 : " << T.output << "\n";
    return 0;
}
