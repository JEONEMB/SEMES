// =====================================================================================
// sim_runner.cpp — 영점 맞춤 반복, 다회 시험, 합성 사진 검증
// =====================================================================================
#include "sim_runner.hpp"

#include "alignment.hpp"

#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>

namespace fs = std::filesystem;

namespace sim {

namespace {
double msSince(const std::chrono::steady_clock::time_point& t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}
std::string f2(const char* f, double a, double b = 0, double c = 0, double d = 0, double e = 0, double g = 0) {
    char buf[512];
    std::snprintf(buf, sizeof(buf), f, a, b, c, d, e, g);
    return buf;
}
double clampAbs(double v, double lim) { return std::max(-lim, std::min(lim, v)); }
} // namespace

std::string poseText(const Pose& p) { return f2("Y %+.4f mm, Θ %+.4f°", p.y_mm, p.t_deg); }

// =====================================================================================
bool Simulator::init(const std::string& configPath, std::string& message) {
    configPath_ = configPath;
    if (!cfg_.load(configPath)) {
        std::error_code ec;
        const fs::path parent = fs::path(configPath).parent_path();
        if (!parent.empty()) fs::create_directories(parent, ec);
        cfg_.save(configPath);
        message = "설정 파일이 없어 기본값으로 만듦 : " + configPath + "\n";
    }
    const std::string bad = cfg_.check();
    if (!bad.empty()) { message += "설정 오류 : " + bad; return false; }
    // 모델 설정 (거친 이동 창은 모델과 같은 값을 씀)
    if (!alignment_init(cfg_.alignment_config)) { message += "모델 설정을 읽지 못함 : " + cfg_.alignment_config; return false; }
    cv::FileStorage a(cfg_.alignment_config, cv::FileStorage::READ);
    if (a.isOpened()) {
        const cv::FileNode c = a["coarse"];
        coarseCenter_ = cv::Point(int(c["center_x_px"]), int(c["center_y_px"]));
        coarseWin_ = cv::Size(int(c["width_px"]), int(c["height_px"]));
    }
    if (world_.load(cfg_.world_dir, cfg_)) message += "가상 장면을 불러옴 : " + cfg_.world_dir;
    else message += "가상 장면이 없음 → 먼저 만들기 (--build 또는 [파노라마 생성])";
    return true;
}

bool Simulator::buildWorld(const Progress& progress, std::string& message) {
    World w;
    if (!w.build(cfg_, coarseCenter_, coarseWin_, progress, message)) return false;
    if (!w.save(cfg_.world_dir)) { message = "가상 장면 저장 실패 : " + cfg_.world_dir; return false; }
    world_ = w;
    world_.load(cfg_.world_dir, cfg_);   // 설정(잡음·상수항)을 반영해 다시 읽음
    message = "가상 장면을 만들어 저장함 : " + cfg_.world_dir;
    return true;
}

Pose Simulator::randomStart(cv::RNG& rng) const {
    Pose p;
    p.y_mm = rng.uniform(-cfg_.start_range_y_mm, cfg_.start_range_y_mm);
    p.t_deg = rng.uniform(-cfg_.start_range_t_deg, cfg_.start_range_t_deg);
    return p;
}

bool Simulator::inTol(double ey_mm, double et_deg, double ratio) const {
    return std::abs(ey_mm) * 1000.0 <= cfg_.tol_y_um * ratio && std::abs(et_deg) <= cfg_.tol_t_deg * ratio;
}

bool Simulator::teach(cv::RNG& rng) {
    if (!world_.ready()) return false;
    return alignment_set_reference(cfg_.teach_center_photo ? world_.renderReference(&rng) : world_.render(Pose(), &rng));
}

// =====================================================================================
bool Simulator::finished(const TrialResult& tr) {
    if (tr.steps.empty()) return false;
    const StepRecord& s = tr.steps.back();
    return !s.est_ok || s.stop || !tr.reason.empty();
}

bool Simulator::stepTrial(TrialResult& tr, cv::RNG& rng, cv::Mat* image) {
    if (finished(tr)) return false;
    // 이번 참 자세 = 시작 또는 (직전 참 자세 + 직전 실제 이동)
    Pose truth = tr.start;
    if (!tr.steps.empty()) {
        const StepRecord& p = tr.steps.back();
        truth.y_mm = p.truth.y_mm + p.applied.y_mm;
        truth.t_deg = p.truth.t_deg + p.applied.t_deg;
        ++tr.moves;
    }
    StepRecord s;
    s.k = int(tr.steps.size());
    s.truth = truth;
    s.truth_in_tol = inTol(truth.y_mm, truth.t_deg);
    auto t0 = std::chrono::steady_clock::now();
    cv::Mat img = world_.render(truth, &rng);
    s.ms_render = msSince(t0);
    t0 = std::chrono::steady_clock::now();
    const Alignment_t a = alignment(img);
    s.ms_align = msSince(t0);
    s.est_ok = alignment_last_info().ok;
    s.est.y_mm = a.y; s.est.t_deg = a.t;

    if (!s.est_ok) {
        tr.reason = "측정 실패 : " + alignment_last_info().message;
    } else if (inTol(a.y, a.t, cfg_.control_stop_ratio)) {
        s.stop = true;
        tr.stopped = true;
    } else if (tr.moves >= cfg_.control_max_moves) {
        tr.reason = "이동 한도(" + std::to_string(cfg_.control_max_moves) + "회) 초과";
    } else {
        // 보정 명령 = −추정, 한 번 최대 이동으로 자름
        s.cmd.y_mm = clampAbs(-a.y, cfg_.robot_max_step_y_mm);
        s.cmd.t_deg = clampAbs(-a.t, cfg_.robot_max_step_t_deg);
        // 로봇 오차 : 배율 오차 + 반복 정밀도
        s.applied.y_mm = s.cmd.y_mm * (1.0 + rng.gaussian(cfg_.robot_gain_error)) + rng.gaussian(cfg_.robot_repeat_y_um / 1000.0);
        s.applied.t_deg = s.cmd.t_deg * (1.0 + rng.gaussian(cfg_.robot_gain_error)) + rng.gaussian(cfg_.robot_repeat_t_deg);
    }
    tr.steps.push_back(s);
    if (image) *image = img;
    if (finished(tr)) {
        tr.final_truth = truth;
        tr.success = tr.stopped && inTol(truth.y_mm, truth.t_deg);
        if (tr.stopped && !tr.success) tr.reason = "멈췄지만 참값이 허용 범위 밖";
        return false;
    }
    return true;
}

TrialResult Simulator::runTrial(const Pose& start, cv::RNG& rng, const StepCallback& onStep, const std::atomic<bool>* cancel) {
    TrialResult tr;
    tr.start = start;
    cv::Mat img;
    for (;;) {
        if (cancel && *cancel) { tr.reason = "사용자 중단"; break; }
        const bool more = stepTrial(tr, rng, &img);
        if (onStep && !onStep(tr, img)) { if (more) tr.reason = "사용자 중단"; break; }
        if (!more) break;
    }
    if (!finished(tr) || tr.reason == "사용자 중단") tr.final_truth = tr.steps.empty() ? start : tr.steps.back().truth;
    return tr;
}

MonteSummary Simulator::monteCarlo(int n, unsigned int seed, const std::function<void(int, const TrialResult&)>& onTrial,
                                   const std::atomic<bool>* cancel) {
    MonteSummary m;
    m.moves_hist.assign(size_t(cfg_.control_max_moves) + 2, 0);
    cv::RNG rng(seed);
    double sy = 0, st = 0, msSum = 0;
    int steps = 0, movesSum = 0;
    for (int i = 0; i < n; ++i) {
        if (cancel && *cancel) break;
        const Pose start = randomStart(rng);
        teach(rng);   // 시험마다 새 기준 사진 (잡음이 다름)
        TrialResult t = runTrial(start, rng, nullptr, cancel);
        if (cancel && *cancel && !t.stopped) break;
        ++m.n;
        if (t.success) ++m.success;
        const int mv = t.stopped ? t.moves : cfg_.control_max_moves + 1;   // 끝 칸 = 실패
        m.moves_hist[size_t(std::min(mv, cfg_.control_max_moves + 1))]++;
        movesSum += t.moves;
        m.moves_max = std::max(m.moves_max, t.moves);
        const double ey = t.final_truth.y_mm * 1000.0, et = t.final_truth.t_deg;
        sy += ey * ey; st += et * et;
        m.final_y_max_um = std::max(m.final_y_max_um, std::abs(ey));
        m.final_t_max_deg = std::max(m.final_t_max_deg, std::abs(et));
        for (const auto& s : t.steps) { msSum += s.ms_render + s.ms_align; ++steps; }
        m.trials.push_back(t);
        if (onTrial) onTrial(i, t);
    }
    if (m.n > 0) {
        m.moves_mean = double(movesSum) / m.n;
        m.final_y_rms_um = std::sqrt(sy / m.n);
        m.final_t_rms_deg = std::sqrt(st / m.n);
    }
    if (steps > 0) m.ms_mean_per_step = msSum / steps;
    return m;
}

std::string Simulator::monteText(const MonteSummary& m) const {
    std::ostringstream o;
    o << "시험 " << m.n << "회 : 성공 " << m.success << f2(" (%.1f %%)\n", m.n ? 100.0 * m.success / m.n : 0.0);
    o << f2("이동 횟수 : 평균 %.2f, 최대 %.0f (한도 %.0f)\n", m.moves_mean, m.moves_max, cfg_.control_max_moves);
    o << f2("멈춘 뒤 참 오차 : Y RMS %.1f µm (최대 %.1f), Θ RMS %.4f° (최대 %.4f)\n", m.final_y_rms_um, m.final_y_max_um, m.final_t_rms_deg,
            m.final_t_max_deg);
    o << f2("한 단계(합성 + 추정) 평균 %.0f ms\n", m.ms_mean_per_step);
    o << "이동 횟수 분포 : ";
    for (size_t i = 0; i < m.moves_hist.size(); ++i)
        if (m.moves_hist[i]) o << (int(i) > cfg_.control_max_moves ? std::string("실패") : std::to_string(i) + "회") << " " << m.moves_hist[i] << "  ";
    o << "\n";
    return o.str();
}

// =====================================================================================
bool Simulator::writeTrialCsv(const TrialResult& t, const std::string& path) const {
    std::error_code ec;
    fs::create_directories(fs::path(path).parent_path(), ec);
    std::ofstream f(path);
    if (!f) return false;
    f << "\xEF\xBB\xBF" << "k,true_y_mm,true_t_deg,est_y_mm,est_t_deg,est_ok,cmd_y_mm,cmd_t_deg,applied_y_mm,applied_t_deg,err_y_um,err_t_deg,true_in_tol,stop,ms_render,ms_align\n";
    f.precision(8);
    for (const auto& s : t.steps)
        f << s.k << ',' << s.truth.y_mm << ',' << s.truth.t_deg << ',' << s.est.y_mm << ',' << s.est.t_deg << ',' << int(s.est_ok) << ','
          << s.cmd.y_mm << ',' << s.cmd.t_deg << ',' << s.applied.y_mm << ',' << s.applied.t_deg << ',' << s.truth.y_mm * 1000.0 << ','
          << s.truth.t_deg << ',' << int(s.truth_in_tol) << ',' << int(s.stop) << ',' << s.ms_render << ',' << s.ms_align << '\n';
    return true;
}

bool Simulator::writeMonteCsv(const MonteSummary& m, const std::string& path) const {
    std::error_code ec;
    fs::create_directories(fs::path(path).parent_path(), ec);
    std::ofstream f(path);
    if (!f) return false;
    f << "\xEF\xBB\xBF" << "trial,start_y_mm,start_t_deg,moves,stopped,success,final_err_y_um,final_err_t_deg,reason\n";
    f.precision(8);
    for (size_t i = 0; i < m.trials.size(); ++i) {
        const auto& t = m.trials[i];
        f << i << ',' << t.start.y_mm << ',' << t.start.t_deg << ',' << t.moves << ',' << int(t.stopped) << ',' << int(t.success) << ','
          << t.final_truth.y_mm * 1000.0 << ',' << t.final_truth.t_deg << ",\"" << t.reason << "\"\n";
    }
    return true;
}

// =====================================================================================
// 검증 : 같은 라벨의 실제 사진 vs 합성 사진 → 모델 추정값
// =====================================================================================
ValidationSummary Simulator::validate(const Progress& progress, const std::atomic<bool>* cancel) {
    ValidationSummary vs;
    const std::vector<Photo> all = scanPhotos(cfg_.image_dir);
    const Photo* r1 = findPhoto(all, "center");
    const Photo* r2 = findPhoto(all, "center1");
    if (!r1 || !r2 || !world_.ready()) { vs.table = "기준 사진(center, center1) 또는 가상 장면이 없음"; return vs; }
    const cv::Mat ref1 = cv::imread(r1->path, cv::IMREAD_GRAYSCALE), ref2 = cv::imread(r2->path, cv::IMREAD_GRAYSCALE);
    cv::RNG rng(cfg_.seed);
    const cv::Mat refSim = cfg_.teach_center_photo ? world_.renderReference(&rng) : world_.render(Pose(), &rng);

    std::vector<Photo> ph;
    for (const auto& p : all) if (!p.is_ref) ph.push_back(p);
    for (size_t i = 0; i < ph.size(); ++i) {
        if (cancel && *cancel) break;
        const Photo& p = ph[i];
        ValidationRow r;
        r.stem = p.stem; r.label = p.pose;
        if (p.set == 1) r.group = p.pose.y_mm == 0.0 ? "1차 Θ 스윕" : (firstSession(p) ? "1차 Y 스윕 (첫 측정)" : "1차 Y 스윕 (48분 뒤)");
        else r.group = std::abs(p.pose.t_deg) <= 0.2 + 1e-9 ? "2차 목표 근처" : (std::abs(std::abs(p.pose.t_deg) - 1.0) < 1e-9 ? "2차 Θ 1°" : "2차 Θ 2°");
        // 실제 사진 (같은 날 기준 사진)
        alignment_set_reference(p.set == 1 ? ref1 : ref2);
        cv::Mat img = cv::imread(p.path, cv::IMREAD_GRAYSCALE);
        Alignment_t a = alignment(img);
        r.ok_real = alignment_last_info().ok; r.est_real = {a.y, a.t};
        // 합성 사진 (합성 기준 사진)
        alignment_set_reference(refSim);
        cv::Mat sim = world_.render(p.pose, &rng);
        a = alignment(sim);
        r.ok_sim = alignment_last_info().ok; r.est_sim = {a.y, a.t};
        vs.rows.push_back(r);
        if (progress) progress(double(i + 1) / ph.size(), p.stem);
    }

    // 묶음별 요약
    std::map<std::string, std::vector<const ValidationRow*>> g;
    std::vector<std::string> order;
    for (const auto& r : vs.rows) { if (!g.count(r.group)) order.push_back(r.group); g[r.group].push_back(&r); }
    std::ostringstream o;
    o << "| 묶음 | 장 | 실제 허용 안 | 합성 허용 안 | 실제 Θ RMS (°) | 합성 Θ RMS (°) | 실제 Y RMS (µm) | 합성 Y RMS (µm) | 합성−실제 Θ 차 RMS (°) | 합성−실제 Y 차 RMS (µm) |\n";
    o << "|---|---|---|---|---|---|---|---|---|---|\n";
    for (const auto& name : order) {
        const auto& v = g[name];
        int pr = 0, ps = 0, nr = 0, ns = 0, nd = 0;
        double tr = 0, ts = 0, yr = 0, ys = 0, dt = 0, dy = 0;
        for (const auto* r : v) {
            if (r->ok_real) {
                const double ey = (r->est_real.y_mm - r->label.y_mm) * 1000, et = r->est_real.t_deg - r->label.t_deg;
                tr += et * et; yr += ey * ey; ++nr; if (inTol(ey / 1000, et)) ++pr;
            }
            if (r->ok_sim) {
                const double ey = (r->est_sim.y_mm - r->label.y_mm) * 1000, et = r->est_sim.t_deg - r->label.t_deg;
                ts += et * et; ys += ey * ey; ++ns; if (inTol(ey / 1000, et)) ++ps;
            }
            if (r->ok_real && r->ok_sim) {
                const double a = (r->est_sim.t_deg - r->est_real.t_deg), b = (r->est_sim.y_mm - r->est_real.y_mm) * 1000;
                dt += a * a; dy += b * b; ++nd;
            }
        }
        const auto rms = [](double s, int n) { return n ? std::sqrt(s / n) : 0.0; };
        o << "| " << name << " | " << v.size() << " | " << pr << " | " << ps << f2(" | %.4f | %.4f | %.1f | %.1f | %.4f | %.1f |\n", rms(tr, nr), rms(ts, ns),
                                                                                  rms(yr, nr), rms(ys, ns), rms(dt, nd), rms(dy, nd));
    }
    vs.table = o.str();
    return vs;
}

} // namespace sim
