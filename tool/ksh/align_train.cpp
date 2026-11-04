// =====================================================================================
// align_train.cpp — 모델 학습 도구 : 라벨 사진 → 특징 s, d 측정 → 계수 학습 → 모델 파일 저장
//
//  사용 (작업 폴더 = Alignment 프로젝트 폴더)
//   align_train [--config 설정] [--order-s N] [--order-d N] [--out 모델파일] [--features 특징CSV]
//     --order-s/-d : Θ 차수 (기본 = 설정 training.theta_order_s/d)
//     --out        : 저장 경로 (기본 = 설정 training.output, 비어 있으면 results/model_s{N}d{N}.yml)
//     --features   : 이미 측정한 특징 CSV 를 다시 씀 (사진 측정 생략)
//  학습 사진 = 설정 training.set 의 단독 축 사진 (Y 단독은 training.y_grid_mm 배수만), 기준 = training.reference
//  학습만 한다. 정확도는 align_eval 로 따로 평가 (재대입 오차는 참고로만 출력)
// =====================================================================================
#include "ksh/align_config.hpp"
#include "ksh/batch.hpp"
#include "ksh/dataset.hpp"
#include "ksh/evaluation.hpp"
#include "ksh/image_util.hpp"
#include "ksh/pose_model.hpp"

#include <opencv2/imgcodecs.hpp>

#include <chrono>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#endif

namespace fs = std::filesystem;

namespace {

void print_feature(const char *name, const ksh::FeatureModel &m) {
    std::cout << "  " << name << " = " << std::setprecision(6) << m.c_px << " + " << m.a_px_per_mm << "·Y";
    for (size_t k = 0; k < m.b_px_per_deg_k.size(); ++k) std::cout << " + " << m.b_px_per_deg_k[k] << "·Θ^" << (k + 1);
    std::cout << "   (σ " << m.sigma_px << " px)\n";
}

} // namespace

int main(int argc, char **argv) {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif
    std::string config_path = "config/alignment.yml", out_path, features_csv;
    int order_s = -1, order_d = -1;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        const bool has = i + 1 < argc;
        if (a == "--config" && has) config_path = argv[++i];
        else if (a == "--order-s" && has) order_s = std::atoi(argv[++i]);
        else if (a == "--order-d" && has) order_d = std::atoi(argv[++i]);
        else if (a == "--out" && has) out_path = argv[++i];
        else if (a == "--features" && has) features_csv = argv[++i];
        else { std::cerr << "알 수 없는 인자 : " << a << "\n"; return 2; }
    }
    ksh::AlignConfig cfg;
    std::string error;
    if (!ksh::load_align_config(config_path, cfg, error, false)) { std::cerr << error << "\n"; return 1; }
    const ksh::TrainingConfig &tc = cfg.training;
    if (order_s < 0) order_s = tc.theta_order_s;
    if (order_d < 0) order_d = tc.theta_order_d;
    if (order_s < 1 || order_d < 1) { std::cerr << "Θ 차수는 1 이상\n"; return 2; }
    if (out_path.empty()) out_path = tc.output;
    if (out_path.empty()) out_path = cfg.evaluation.result_dir + "/model_s" + std::to_string(order_s) + "d" + std::to_string(order_d) + ".yml";
    std::error_code ec;
    fs::create_directories(fs::u8path(cfg.evaluation.result_dir), ec);

    // 1) 특징 표본
    std::vector<ksh::FeatureSample> samples;
    if (!features_csv.empty()) {
        if (!ksh::load_feature_csv(features_csv, samples)) { std::cerr << "특징 CSV 를 읽지 못함 : " << features_csv << "\n"; return 1; }
        std::cout << "특징 CSV 다시 사용 : " << features_csv << " (" << samples.size() << " 장)\n";
    } else {
        const std::string dir = ksh::resolve_path(config_path, tc.image_dir);
        const auto photos = ksh::list_labeled_photos(dir);
        const ksh::LabeledPhoto *ref = ksh::find_photo(photos, tc.reference);
        if (!ref) { std::cerr << "기준 사진 " << tc.reference << " 이 " << dir << " 에 없음\n"; return 1; }
        std::vector<ksh::LabeledPhoto> train;
        for (const auto &p : photos)
            if (ksh::is_training_photo(p, tc.set, tc.y_grid_mm)) train.push_back(p);
        std::cout << "학습 사진 " << train.size() << " 장 (세트 " << tc.set << ", Y 간격 " << tc.y_grid_mm << " mm 배수), 기준 = " << ref->stem << "\n";
        const cv::Mat ref32 = ksh::to_gray32(ksh::read_image(ref->path, cv::IMREAD_GRAYSCALE));
        const auto t0 = std::chrono::steady_clock::now();
        const auto measured = ksh::measure_photos(train, ref32, cfg.features, &std::cout);
        const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        samples = ksh::to_samples(measured);
        double ms = 0;
        for (const auto &m : measured) ms += m.elapsed_ms;
        std::cout << "측정 : 성공 " << samples.size() << " / " << measured.size() << " 장, 측정 평균 " << std::fixed << std::setprecision(1)
                  << (measured.empty() ? 0.0 : ms / measured.size()) << " ms/장 (사진 읽기 포함 전체 " << sec << " s)\n" << std::defaultfloat;
        for (const auto &m : measured)
            if (!m.features.ok) std::cout << "  실패 : " << m.photo.stem << " — " << m.features.message << "\n";
        const std::string csv = cfg.evaluation.result_dir + "/train_features.csv";
        if (ksh::save_feature_csv(csv, samples)) std::cout << "특징 저장 : " << csv << "\n";
    }

    // 2) 학습
    ksh::PoseModel model;
    if (!ksh::fit_pose_model(samples, order_s, order_d, model)) { std::cerr << "학습 실패 (표본 부족)\n"; return 1; }
    std::cout << "\n학습 결과 (s : Θ " << order_s << "차, d : Θ " << order_d << "차, 계수 " << (order_s + order_d + 4) << "개)\n";
    print_feature("s", model.s);
    print_feature("d", model.d);

    // 3) 재대입 오차 (학습 사진을 다시 예측 — 일반화 성능이 아님, 참고)
    std::vector<ksh::PosePrediction> fit_pred;
    for (const auto &s : samples) {
        ksh::PosePrediction p;
        p.stem = s.stem;
        p.group = "재대입 (참고)";
        p.y_mm = s.y_mm;
        p.t_deg = s.t_deg;
        p.ok = ksh::invert_pose_model(model, s.s_px, s.d_px, p.y_est_mm, p.t_est_deg);
        fit_pred.push_back(p);
    }
    std::cout << "\n" << ksh::summary_table(ksh::summarize_groups(fit_pred, cfg.tolerance, cfg.spec_tolerance, cfg.evaluation.score_cap),
                                            cfg.tolerance, cfg.spec_tolerance);

    // 4) 저장
    std::ostringstream note;
    note << "align_train : 세트 " << tc.set << " 단독 축 사진 (Y " << tc.y_grid_mm << " mm 배수), 기준 " << tc.reference << ", Fine = "
         << (cfg.features.fine.method == ksh::FineMethod::icgn ? std::string("icgn/") + ksh::kernel_name(cfg.features.fine.kernel) : std::string("opencv_ecc"))
         << ", σ " << cfg.features.fine.gauss_sigma_px;
    if (!ksh::save_pose_model(out_path, model, note.str(), int(samples.size()))) { std::cerr << "모델 저장 실패 : " << out_path << "\n"; return 1; }
    std::cout << "\n모델 저장 : " << out_path << "\n  → 사용하려면 config/alignment.yml 의 model_file 을 이 파일로 바꾼다\n";
    return 0;
}
