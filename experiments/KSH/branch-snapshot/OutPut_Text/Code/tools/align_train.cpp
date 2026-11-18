// =====================================================================================
// align_train.cpp — 보정식 학습 + 정확도 검증 도구 (교보재 텍스트 설정판)
//
//  사용 : align_train [config/alignment.yml]
//   1) training.train / training.test 의 세트마다 특징 s, d 를 잰다
//      feature_csv 가 있으면 그 파일을 읽고, 없으면 image_dir 의 사진을 재서 result_dir/<이름>_features.csv 로 저장
//      (센터 사진은 is_ref = 1 로 함께 저장 — 학습에는 안 쓰고 영점에만 씀)
//   2) 학습 : train 세트 전체로 기울기를 공통으로, 상수항은 세트마다 따로 (촬영 날마다 다른 영점 차이 흡수)
//      세트 균형 가중(balance_sets). Θ 차수 = theta_order_s / theta_order_d (기본 1 = 직선)
//   3) 채점
//      · 학습 세트 재대입 (세트 상수 사용)
//      · 세트 제외 검증 : 학습 세트가 2개 이상이고 zero_ref 가 있으면, 그 세트를 빼고 학습 → 그 세트 센터로 영점 → 채점
//      · test 세트 : zero_ref(같은 날 센터) 측정값 + center_to_encoder 로 영점을 잡고 채점 — 실제 사용과 같은 방식
//   4) training.output 에 모델 저장, result_dir 에 report.md · predictions.csv
// =====================================================================================
#include "ksh/align_config.hpp"
#include "ksh/batch.hpp"
#include "ksh/dataset.hpp"
#include "ksh/evaluation.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace fs = std::filesystem;

namespace {

struct SetData {
    ksh::TrainingSet spec;
    std::vector<ksh::FeatureSample> samples;   // 라벨 사진 (센터·제외 사진 빼고)
    std::vector<ksh::FeatureRow> refs;         // 센터 사진
};

// 세트 하나 : CSV 가 있으면 읽고, 없으면 사진을 잼
bool collect(const ksh::TrainingSet &set, const ksh::AlignConfig &cfg, SetData &out, std::ostream &log) {
    out.spec = set;
    std::vector<ksh::FeatureRow> rows;
    std::error_code ec;
    if (!set.feature_csv.empty() && fs::exists(fs::u8path(set.feature_csv), ec)) {
        if (!ksh::load_feature_csv(set.feature_csv, rows)) { log << "  [" << set.name << "] 특징 CSV 형식 오류 : " << set.feature_csv << "\n"; return false; }
        log << "  [" << set.name << "] 특징 CSV " << rows.size() << "줄 : " << set.feature_csv << "\n";
    } else {
        const auto photos = ksh::list_labeled_photos(set.image_dir);
        if (photos.empty()) { log << "  [" << set.name << "] 라벨 사진이 없음 : " << set.image_dir << "\n"; return false; }
        log << "  [" << set.name << "] 사진 " << photos.size() << "장 측정 : " << set.image_dir << "\n";
        const auto measured = ksh::measure_photos(photos, cfg.features, cfg.training.threads, &std::cout);
        int fail = 0;
        double ms = 0;
        for (const auto &m : measured) {
            ms += m.elapsed_ms;
            if (!m.features.ok) { ++fail; log << "    측정 실패 " << m.photo.stem << " : " << m.features.message << "\n"; continue; }
            ksh::FeatureRow r;
            r.sample.stem = m.photo.stem;
            r.sample.y_mm = m.photo.y_mm;
            r.sample.t_deg = m.photo.t_deg;
            r.sample.s_px = m.features.s_px;
            r.sample.d_px = m.features.d_px;
            r.is_reference = m.photo.is_reference;
            rows.push_back(r);
        }
        log << "    실패 " << fail << "장, 평균 " << std::fixed << std::setprecision(1) << ms / std::max<size_t>(1, measured.size()) << " ms/장"
            << std::defaultfloat << "\n";
        const std::string csv = (fs::u8path(cfg.training.result_dir) / fs::u8path(set.name + "_features.csv")).u8string();
        if (ksh::save_feature_csv(csv, rows)) log << "    특징 저장 : " << csv << "\n";
    }
    int excluded = 0;
    for (const auto &r : rows) {
        if (r.is_reference) { out.refs.push_back(r); continue; }
        if (std::find(set.exclude.begin(), set.exclude.end(), r.sample.stem) != set.exclude.end()) { ++excluded; continue; }
        out.samples.push_back(r.sample);
    }
    log << "    학습·채점 " << out.samples.size() << "장, 센터 " << out.refs.size() << "장, 제외 " << excluded << "장\n";
    return true;
}

std::vector<ksh::PosePrediction> predict(const std::vector<ksh::FeatureSample> &samples, const ksh::PoseModel &m, const ksh::ZeroOffset &z,
                                         const std::string &group) {
    std::vector<ksh::PosePrediction> v;
    for (const auto &s : samples) {
        ksh::PosePrediction p;
        p.stem = s.stem;
        p.group = group;
        p.y_mm = s.y_mm;
        p.t_deg = s.t_deg;
        p.ok = ksh::invert_pose_model(m, s.s_px, s.d_px, p.y_est_mm, p.t_est_deg);
        p.y_est_mm -= z.y_mm;
        p.t_est_deg -= z.t_deg;
        v.push_back(p);
    }
    return v;
}

std::string eq(const char *name, const ksh::FeatureModel &f) {
    std::ostringstream o;
    o << std::setprecision(8) << name << " = " << f.c_px << " + " << f.a_px_per_mm << "·Y";
    for (size_t k = 0; k < f.b_px_per_deg_k.size(); ++k) o << " + " << f.b_px_per_deg_k[k] << "·Θ" << (k ? "^" + std::to_string(k + 1) : "");
    o << "   (σ " << f.sigma_px << " px)\n";
    return o.str();
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
    std::vector<SetData> train, test;
    for (const auto &s : T.train) { SetData d; if (!collect(s, cfg, d, log)) { std::cerr << log.str(); return 1; } train.push_back(d); }
    for (const auto &s : T.test) { SetData d; if (!collect(s, cfg, d, log)) { std::cerr << log.str(); return 1; } test.push_back(d); }
    log << "```\n\n";
    // 센터 사진 (이름 → 특징) : 모든 세트에서 찾음
    std::map<std::string, ksh::FeatureSample> refs;
    for (auto *v : {&train, &test})
        for (const auto &d : *v)
            for (const auto &r : d.refs)
                if (!refs.count(r.sample.stem)) refs[r.sample.stem] = r.sample;

    // 학습
    std::vector<std::vector<ksh::FeatureSample>> sets;
    int n = 0;
    for (const auto &d : train) { sets.push_back(d.samples); n += int(d.samples.size()); }
    ksh::PoseModel model;
    std::vector<double> cs, cd;
    if (!ksh::fit_pose_model_sets(sets, T.theta_order_s, T.theta_order_d, T.balance_sets, model, &cs, &cd)) { std::cerr << "학습 실패 (표본 부족)\n"; return 1; }
    std::ostringstream note;
    note << "align_train : ";
    for (const auto &d : train) note << d.spec.name << " ";
    note << "(" << n << "장, 세트별 상수" << (T.balance_sets ? " + 세트 균형 가중" : "") << "), s = QR 중심 x, d = s − 홀 x 평균, Θ 차수 s" << T.theta_order_s
         << " d" << T.theta_order_d << ", c = 첫 세트(" << train.front().spec.name << ") 상수 — 사용 때는 영점(zero)을 뺌";
    if (!ksh::save_pose_model(T.output, model, note.str(), n)) { std::cerr << "모델 저장 실패 : " << T.output << "\n"; return 1; }

    // 채점
    std::vector<ksh::PosePrediction> all;
    std::ostringstream zlog;
    auto zero_of = [&](const ksh::PoseModel &m, const std::string &ref, ksh::ZeroOffset &z) {
        auto it = refs.find(ref);
        if (it == refs.end()) return false;
        if (!ksh::invert_pose_model(m, it->second.s_px, it->second.d_px, z.y_mm, z.t_deg)) return false;
        z.y_mm += cfg.center_to_encoder.y_mm;    // 실제 사용과 같게 : 영점 = 센터 측정값 + (센터 자세 − 엔코더 영점)
        z.t_deg += cfg.center_to_encoder.t_deg;
        return true;
    };
    for (size_t k = 0; k < train.size(); ++k) {
        ksh::PoseModel mk = model;
        mk.s.c_px = cs[k];
        mk.d.c_px = cd[k];
        auto v = predict(train[k].samples, mk, ksh::ZeroOffset(), "학습 " + train[k].spec.name + " (재대입, 세트 상수)");
        all.insert(all.end(), v.begin(), v.end());
    }
    if (train.size() >= 2)
        for (size_t k = 0; k < train.size(); ++k) {
            if (train[k].spec.zero_ref.empty()) continue;
            std::vector<std::vector<ksh::FeatureSample>> others;
            for (size_t j = 0; j < train.size(); ++j) if (j != k) others.push_back(train[j].samples);
            ksh::PoseModel mo;
            ksh::ZeroOffset z;
            if (!ksh::fit_pose_model_sets(others, T.theta_order_s, T.theta_order_d, T.balance_sets, mo) || !zero_of(mo, train[k].spec.zero_ref, z)) continue;
            auto v = predict(train[k].samples, mo, z, "세트 제외 검증 " + train[k].spec.name + " (나머지로 학습, 영점 " + train[k].spec.zero_ref + " + 엔코더 보정)");
            all.insert(all.end(), v.begin(), v.end());
        }
    for (const auto &d : test) {
        ksh::ZeroOffset z;
        if (d.spec.zero_ref.empty() || !zero_of(model, d.spec.zero_ref, z)) {
            zlog << "  [" << d.spec.name << "] zero_ref '" << d.spec.zero_ref << "' 센터 사진을 찾지 못해 채점 생략\n";
            continue;
        }
        zlog << std::fixed << std::setprecision(5) << "  [" << d.spec.name << "] 영점 " << d.spec.zero_ref << " : y " << z.y_mm << " mm, t " << z.t_deg << " deg\n"
             << std::defaultfloat;
        auto v = predict(d.samples, model, z, "검증 " + d.spec.name + " (학습에 안 쓴 사진, 영점 " + d.spec.zero_ref + " + 엔코더 보정)");
        all.insert(all.end(), v.begin(), v.end());
    }
    const auto groups = ksh::summarize_groups(all, cfg.tolerance, cfg.spec_tolerance, 100.0);

    log << "## 2. 보정식 (" << T.output << ")\n\n```\n" << eq("s", model.s) << eq("d", model.d) << "세트별 상수항 (c_s, c_d) :\n";
    for (size_t k = 0; k < train.size(); ++k) log << std::setprecision(8) << "  " << train[k].spec.name << " : " << cs[k] << ", " << cd[k] << "\n";
    log << "```\n\n## 3. 영점 (검증 세트)\n\n```\n" << zlog.str() << "```\n\n## 4. 정확도\n\n" << ksh::summary_table(groups, cfg.tolerance, cfg.spec_tolerance) << "\n";
    const std::string rep = (fs::u8path(T.result_dir) / "report.md").u8string();
    const std::string csv = (fs::u8path(T.result_dir) / "predictions.csv").u8string();
    { std::ofstream o(fs::u8path(rep)); o << log.str(); }
    ksh::save_predictions_csv(csv, all, cfg.tolerance, cfg.spec_tolerance);
    std::cout << log.str() << "보고서 : " << rep << "\n사진별 : " << csv << "\n모델 : " << T.output << "\n";
    return 0;
}
