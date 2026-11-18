// evaluation.cpp — 정확도 평가 구현
#include "ksh/evaluation.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <sstream>

namespace ksh {

// 오차 = 추정 − 정답 (Y 는 µm 로)
double error_y_um(const PosePrediction &p) { return (p.y_est_mm - p.y_mm) * 1000.0; }
double error_t_deg(const PosePrediction &p) { return p.t_est_deg - p.t_deg; }

// 측정 성공이고 Y·Θ 오차가 모두 허용 범위 안
bool within_tolerance(const PosePrediction &p, const Tolerance &tol) {
    return p.ok && std::abs(error_y_um(p)) <= tol.y_um && std::abs(error_t_deg(p)) <= tol.t_deg;
}

// J = √ mean[ (eY/tolY)² + (eΘ/tolΘ)² ] — 항마다 cap 에서 자르고, 측정 실패는 두 항 모두 cap
double score_j(const std::vector<PosePrediction> &preds, const Tolerance &tol, double cap) {
    if (preds.empty()) return 0.0;
    double sum = 0.0;
    for (const auto &p : preds) {
        if (!p.ok) { sum += 2.0 * cap * cap; continue; }
        const double ny = std::min(cap, std::abs(error_y_um(p) / tol.y_um)), nt = std::min(cap, std::abs(error_t_deg(p) / tol.t_deg));
        sum += ny * ny + nt * nt;
    }
    return std::sqrt(sum / double(preds.size()));
}

namespace {
// 묶음 하나 요약 : 장 수, 성공 수, 허용 통과 수, RMS·최대 오차, 평균 시간, J
GroupSummary summarize(const std::string &name, const std::vector<const PosePrediction *> &v, const Tolerance &tol, const Tolerance &spec,
                       double cap) {
    GroupSummary g;
    g.group = name;
    g.n = int(v.size());
    std::vector<PosePrediction> copy;
    double sy = 0, st = 0;
    for (const PosePrediction *p : v) {
        copy.push_back(*p);
        g.mean_ms += p->elapsed_ms;
        if (!p->ok) continue;
        ++g.n_ok;
        const double ey = error_y_um(*p), et = error_t_deg(*p);
        sy += ey * ey; st += et * et;
        g.max_y_um = std::max(g.max_y_um, std::abs(ey));
        g.max_t_deg = std::max(g.max_t_deg, std::abs(et));
        if (within_tolerance(*p, tol)) ++g.pass;
        if (within_tolerance(*p, spec)) ++g.pass_spec;
    }
    if (g.n_ok) { g.rms_y_um = std::sqrt(sy / g.n_ok); g.rms_t_deg = std::sqrt(st / g.n_ok); }
    if (g.n) g.mean_ms /= g.n;
    g.j = score_j(copy, tol, cap);
    return g;
}
} // namespace

// group 이름별 요약 (처음 나온 순서), all_label 이 있으면 전체 줄 추가
std::vector<GroupSummary> summarize_groups(const std::vector<PosePrediction> &preds, const Tolerance &tol, const Tolerance &spec, double cap,
                                           const std::string &all_label) {
    std::vector<std::string> order;
    for (const auto &p : preds)
        if (std::find(order.begin(), order.end(), p.group) == order.end()) order.push_back(p.group);
    std::vector<GroupSummary> out;
    for (const auto &name : order) {
        std::vector<const PosePrediction *> v;
        for (const auto &p : preds)
            if (p.group == name) v.push_back(&p);
        out.push_back(summarize(name, v, tol, spec, cap));
    }
    if (!all_label.empty()) {
        std::vector<const PosePrediction *> v;
        for (const auto &p : preds) v.push_back(&p);
        out.push_back(summarize(all_label, v, tol, spec, cap));
    }
    return out;
}

// 블록 교차검증 묶음 번호 : Y 단독은 Y 순으로 n_y 묶음, Θ 단독은 Θ 순으로 n_t 묶음 (이웃 라벨끼리 같은 묶음)
std::vector<int> block_folds(const std::vector<FeatureSample> &samples, int n_y_blocks, int n_t_blocks) {
    std::vector<int> fold(samples.size(), 0), iy, it;
    for (size_t i = 0; i < samples.size(); ++i) (samples[i].t_deg == 0.0 ? iy : it).push_back(int(i));
    auto assign = [&](std::vector<int> &idx, int nb, int base, bool by_y) {
        std::sort(idx.begin(), idx.end(), [&](int a, int b) { return by_y ? samples[a].y_mm < samples[b].y_mm : samples[a].t_deg < samples[b].t_deg; });
        for (size_t j = 0; j < idx.size(); ++j) fold[idx[j]] = base + int(j * nb / idx.size());
    };
    assign(iy, n_y_blocks, 0, true);
    assign(it, n_t_blocks, n_y_blocks, false);
    return fold;
}

// 묶음마다 : 나머지로 학습 → 그 묶음 예측 (학습에 안 쓴 사진으로만 채점)
std::vector<PosePrediction> cross_validate(const std::vector<FeatureSample> &samples, const std::vector<int> &folds, int theta_order_s,
                                           int theta_order_d) {
    std::vector<PosePrediction> out(samples.size());
    const int nf = folds.empty() ? 0 : *std::max_element(folds.begin(), folds.end()) + 1;
    for (int f = 0; f < nf; ++f) {
        std::vector<FeatureSample> train;
        for (size_t i = 0; i < samples.size(); ++i)
            if (folds[i] != f) train.push_back(samples[i]);
        PoseModel m;
        const bool fitted = fit_pose_model(train, theta_order_s, theta_order_d, m);
        for (size_t i = 0; i < samples.size(); ++i) {
            if (folds[i] != f) continue;
            PosePrediction &p = out[i];
            p.stem = samples[i].stem;
            p.group = samples[i].t_deg == 0.0 ? "Y 단독" : "Θ 단독";
            p.y_mm = samples[i].y_mm;
            p.t_deg = samples[i].t_deg;
            p.ok = fitted && invert_pose_model(m, samples[i].s_px, samples[i].d_px, p.y_est_mm, p.t_est_deg);
        }
    }
    return out;
}

// 요약 표 (마크다운)
std::string summary_table(const std::vector<GroupSummary> &groups, const Tolerance &tol, const Tolerance &spec) {
    std::ostringstream s;
    s << std::fixed;
    s << "| 묶음 | 장 | 허용 안 (±" << std::setprecision(0) << tol.y_um << " µm · ±" << std::setprecision(3) << tol.t_deg << "°) | 과제 허용 안 (±"
      << std::setprecision(0) << spec.y_um << " µm · ±" << std::setprecision(2) << spec.t_deg << "°) | J | Θ RMS (°) | Θ 최대 (°) | Y RMS (µm) | Y 최대 (µm) | 시간 (ms/장) |\n";
    s << "|---|---|---|---|---|---|---|---|---|---|\n";
    for (const auto &g : groups) {
        s << "| " << g.group << " | " << g.n << " | " << g.pass << " / " << g.n << " | " << g.pass_spec << " / " << g.n << " | " << std::setprecision(3) << g.j
          << " | " << std::setprecision(4) << g.rms_t_deg << " | " << g.max_t_deg << " | " << std::setprecision(1) << g.rms_y_um << " | " << g.max_y_um
          << " | " << g.mean_ms << " |";
        if (g.n_ok < g.n) s << " 측정 실패 " << (g.n - g.n_ok);
        s << "\n";
    }
    return s.str();
}

// 사진별 결과 CSV (UTF-8 BOM)
bool save_predictions_csv(const std::string &path, const std::vector<PosePrediction> &preds, const Tolerance &tol, const Tolerance &spec) {
    std::ofstream f(std::filesystem::u8path(path));
    if (!f) return false;
    f << "\xEF\xBB\xBF" << "stem,group,y_mm,t_deg,y_est_mm,t_est_deg,err_y_um,err_t_deg,ok,in_tol,in_spec,ms\n" << std::setprecision(10);
    for (const auto &p : preds) {
        f << '"' << p.stem << "\",\"" << p.group << "\"," << p.y_mm << ',' << p.t_deg << ',';
        if (p.ok) f << p.y_est_mm << ',' << p.t_est_deg << ',' << error_y_um(p) << ',' << error_t_deg(p);
        else f << ",,,";
        f << ',' << int(p.ok) << ',' << int(within_tolerance(p, tol)) << ',' << int(within_tolerance(p, spec)) << ',' << p.elapsed_ms << '\n';
    }
    return true;
}

} // namespace ksh
