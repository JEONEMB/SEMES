// =====================================================================================
// align_eval.cpp — 정확도 평가 도구
//
//  사용 (작업 폴더 = Alignment 프로젝트 폴더)
//   align_eval [--config 설정] [--model 모델파일] [--dir 사진폴더]
//       사진 평가 : 라벨이 붙은 사진 전체를 세트별 기준 사진으로 추정 → 묶음별 허용 범위 통과·RMS·최대·J·처리 시간
//       1차 = 학습에 쓴 데이터(참고), 2차 = 학습에 쓰지 않은 시험 데이터
//   align_eval --cv [--order-s N --order-d N | --orders all] [--features 특징CSV]
//       블록 교차검증 : 학습 사진(설정 training)을 이웃 구간 묶음으로 나눠 번갈아 빼고 학습·예측
//       --orders all : s·d 의 Θ 1~3차 9가지를 같은 특징으로 비교
//  결과 : results/eval_photos.csv·.md, results/eval_cv.csv·.md
// =====================================================================================
#include "ksh/align_config.hpp"
#include "ksh/batch.hpp"
#include "ksh/dataset.hpp"
#include "ksh/evaluation.hpp"
#include "ksh/image_util.hpp"
#include "ksh/pose_model.hpp"

#include <opencv2/imgcodecs.hpp>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#endif

namespace fs = std::filesystem;

namespace {

void write_text(const std::string &path, const std::string &text) {
    std::ofstream f(fs::u8path(path));
    f << "\xEF\xBB\xBF" << text;
}

std::string group_of(const ksh::LabeledPhoto &p, double near_t_deg) {
    if (p.set == 1) return "1차 (학습 데이터, 참고)";
    if (std::abs(p.t_deg) <= near_t_deg + 1e-9) {
        std::ostringstream s;
        s << "2차 목표 근처 (Θ ±" << near_t_deg << "° 이내)";
        return s.str();
    }
    std::ostringstream s;
    s << "2차 Θ = " << p.t_deg << "°";
    return s.str();
}

// ---------------------------------------------------------------- 사진 평가
int run_photos(const ksh::AlignConfig &cfg, const std::string &config_path, std::string dir) {
    const ksh::EvaluationConfig &ec = cfg.evaluation;
    if (dir.empty()) dir = ksh::resolve_path(config_path, ec.image_dir);
    const auto photos = ksh::list_labeled_photos(dir);
    std::map<int, cv::Mat> refs;
    for (const auto &rs : {std::make_pair(1, ec.reference_set1), std::make_pair(2, ec.reference_set2)}) {
        const ksh::LabeledPhoto *r = ksh::find_photo(photos, rs.second);
        if (!r) { std::cerr << "기준 사진 " << rs.second << " 이 " << dir << " 에 없음\n"; return 1; }
        refs[rs.first] = ksh::to_gray32(cv::imread(r->path, cv::IMREAD_GRAYSCALE));
    }
    std::vector<ksh::LabeledPhoto> targets;
    for (const auto &p : photos)
        if (!p.is_reference) targets.push_back(p);
    std::cout << "사진 " << targets.size() << " 장 평가 (" << dir << ")\n";
    const auto results = ksh::estimate_photos(targets, refs, cfg.features, cfg.model, &std::cout);

    std::vector<ksh::PosePrediction> preds;
    for (const auto &r : results) {
        ksh::PosePrediction p;
        p.stem = r.photo.stem;
        p.group = group_of(r.photo, ec.near_target_t_deg);
        p.y_mm = r.photo.y_mm;
        p.t_deg = r.photo.t_deg;
        p.ok = r.estimate.ok;
        p.y_est_mm = r.estimate.y_mm;
        p.t_est_deg = r.estimate.t_deg;
        p.elapsed_ms = r.estimate.elapsed_ms;
        preds.push_back(p);
    }
    std::vector<ksh::PosePrediction> set2;
    for (const auto &p : preds)
        if (p.group.rfind("2차", 0) == 0) set2.push_back(p);
    auto groups = ksh::summarize_groups(preds, cfg.tolerance, cfg.spec_tolerance, ec.score_cap);
    const auto all2 = ksh::summarize_groups(set2, cfg.tolerance, cfg.spec_tolerance, ec.score_cap, "2차 전체");
    groups.push_back(all2.back());

    std::ostringstream md;
    md << "# 사진 평가\n\n- 설정 : " << config_path << "\n- 모델 : " << (cfg.model_path.empty() ? "(설정 안 model:)" : cfg.model_path)
       << "\n- 사진 : " << dir << "\n\n" << ksh::summary_table(groups, cfg.tolerance, cfg.spec_tolerance);
    std::cout << "\n" << md.str();
    const std::string base = ec.result_dir + "/eval_photos";
    ksh::save_predictions_csv(base + ".csv", preds, cfg.tolerance, cfg.spec_tolerance);
    write_text(base + ".md", md.str());
    std::cout << "\n사진별 결과 : " << base << ".csv, 요약 : " << base << ".md\n";
    return 0;
}

// ---------------------------------------------------------------- 블록 교차검증
int run_cv(const ksh::AlignConfig &cfg, const std::string &config_path, const std::string &features_csv, int order_s, int order_d, bool all_orders) {
    const ksh::TrainingConfig &tc = cfg.training;
    const ksh::EvaluationConfig &ec = cfg.evaluation;
    std::vector<ksh::FeatureSample> samples;
    if (!features_csv.empty()) {
        if (!ksh::load_feature_csv(features_csv, samples)) { std::cerr << "특징 CSV 를 읽지 못함 : " << features_csv << "\n"; return 1; }
    } else {
        const std::string dir = ksh::resolve_path(config_path, tc.image_dir);
        const auto photos = ksh::list_labeled_photos(dir);
        const ksh::LabeledPhoto *ref = ksh::find_photo(photos, tc.reference);
        if (!ref) { std::cerr << "기준 사진 " << tc.reference << " 이 " << dir << " 에 없음\n"; return 1; }
        std::vector<ksh::LabeledPhoto> train;
        for (const auto &p : photos)
            if (ksh::is_training_photo(p, tc.set, tc.y_grid_mm)) train.push_back(p);
        std::cout << "학습 사진 " << train.size() << " 장 측정\n";
        const auto measured = ksh::measure_photos(train, ksh::to_gray32(cv::imread(ref->path, cv::IMREAD_GRAYSCALE)), cfg.features, &std::cout);
        samples = ksh::to_samples(measured);
        if (samples.size() != measured.size()) std::cout << "측정 실패 " << measured.size() - samples.size() << " 장은 교차검증에서 빠짐\n";
        ksh::save_feature_csv(ec.result_dir + "/cv_features.csv", samples);
    }
    const std::vector<int> folds = ksh::block_folds(samples, ec.cv_blocks_y, ec.cv_blocks_t);
    std::vector<std::pair<int, int>> forms;
    if (all_orders) {
        for (int a = 1; a <= 3; ++a)
            for (int b = 1; b <= 3; ++b) forms.push_back({a, b});
    } else {
        forms.push_back({order_s, order_d});
    }
    std::ostringstream md;
    md << "# 블록 교차검증\n\n- 표본 " << samples.size() << " 장, 묶음 " << ec.cv_blocks_y << " (Y 단독) + " << ec.cv_blocks_t
       << " (Θ 단독)\n- 목표 근처 = |Y| ≤ " << ec.cv_near_y_mm << " mm 이고 |Θ| ≤ " << ec.near_target_t_deg << "°\n\n";
    std::vector<ksh::PosePrediction> keep;
    for (const auto &f : forms) {
        std::vector<ksh::PosePrediction> preds = ksh::cross_validate(samples, folds, f.first, f.second);
        std::vector<ksh::PosePrediction> near_preds;
        for (const auto &p : preds)
            if (std::abs(p.y_mm) <= ec.cv_near_y_mm + 1e-9 && std::abs(p.t_deg) <= ec.near_target_t_deg + 1e-9) {
                ksh::PosePrediction q = p;
                q.group = "목표 근처";
                near_preds.push_back(q);
            }
        auto groups = ksh::summarize_groups(preds, cfg.tolerance, cfg.spec_tolerance, ec.score_cap, "전체");
        const auto ng = ksh::summarize_groups(near_preds, cfg.tolerance, cfg.spec_tolerance, ec.score_cap);
        groups.insert(groups.end(), ng.begin(), ng.end());
        md << "## s : Θ " << f.first << "차, d : Θ " << f.second << "차\n\n" << ksh::summary_table(groups, cfg.tolerance, cfg.spec_tolerance) << "\n";
        for (auto &p : preds) {
            p.group += " | s" + std::to_string(f.first) + "d" + std::to_string(f.second);
            keep.push_back(p);
        }
    }
    std::cout << "\n" << md.str();
    const std::string base = ec.result_dir + "/eval_cv";
    ksh::save_predictions_csv(base + ".csv", keep, cfg.tolerance, cfg.spec_tolerance);
    write_text(base + ".md", md.str());
    std::cout << "사진별 결과 : " << base << ".csv, 요약 : " << base << ".md\n";
    return 0;
}

} // namespace

int main(int argc, char **argv) {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif
    std::string config_path = "config/alignment.yml", model_path, dir, features_csv;
    bool cv_mode = false, all_orders = false;
    int order_s = -1, order_d = -1;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        const bool has = i + 1 < argc;
        if (a == "--config" && has) config_path = argv[++i];
        else if (a == "--model" && has) model_path = argv[++i];
        else if (a == "--dir" && has) dir = argv[++i];
        else if (a == "--cv") cv_mode = true;
        else if (a == "--order-s" && has) order_s = std::atoi(argv[++i]);
        else if (a == "--order-d" && has) order_d = std::atoi(argv[++i]);
        else if (a == "--orders" && has) all_orders = std::string(argv[++i]) == "all";
        else if (a == "--features" && has) features_csv = argv[++i];
        else { std::cerr << "알 수 없는 인자 : " << a << "\n"; return 2; }
    }
    ksh::AlignConfig cfg;
    std::string error;
    if (!ksh::load_align_config(config_path, cfg, error)) { std::cerr << error << "\n"; return 1; }
    if (!model_path.empty()) {
        if (!ksh::load_pose_model(model_path, cfg.model)) { std::cerr << "모델 파일을 읽지 못함 : " << model_path << "\n"; return 1; }
        cfg.model_path = model_path;
    }
    std::error_code ec;
    fs::create_directories(fs::u8path(cfg.evaluation.result_dir), ec);
    std::cout << "허용 범위 : ±" << cfg.tolerance.y_um << " µm · ±" << cfg.tolerance.t_deg << "° (과제 ±" << cfg.spec_tolerance.y_um << " µm · ±"
              << cfg.spec_tolerance.t_deg << "°)\n";
    if (cv_mode) {
        if (order_s < 0) order_s = int(cfg.model.s.b_px_per_deg_k.size());
        if (order_d < 0) order_d = int(cfg.model.d.b_px_per_deg_k.size());
        return run_cv(cfg, config_path, features_csv, order_s, order_d, all_orders);
    }
    return run_photos(cfg, config_path, dir);
}
