// =====================================================================================
// sim_runner.cpp — 영점 맞춤 반복, 다회 시험, 합성 사진 검증
// =====================================================================================
#include "sim_runner.hpp"

#include "sim_paths.hpp"

#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <ctime>
#include <iterator>
#include <map>
#include <regex>
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
        const fs::path parent = u8p(configPath).parent_path();
        if (!parent.empty()) fs::create_directories(parent, ec);
        cfg_.save(configPath);
        message = "설정 파일이 없어 기본값으로 만듦 : " + configPath + "\n";
    }
    const std::string bad = cfg_.check();
    if (!bad.empty()) { message += "설정 오류 : " + bad; return false; }
    cfg_.resolvePaths();
    message += "데이터 폴더 : " + (cfg_.image_dir.empty() ? std::string("없음 (합성 검증·장면 만들기만 필요, 시험은 가능)") : cfg_.image_dir) + "\n";
    // 거친 이동 창 (가상 장면 만들 때) — Alignment 설정과 같은 값
    if (!cfg_.alignment_config.empty()) {
        cv::FileStorage a(cfg_.alignment_config, cv::FileStorage::READ);
        if (a.isOpened()) {
            const cv::FileNode c = a["coarse"];
            if (!c.empty()) {
                coarseCenter_ = cv::Point(int(c["center_x_px"]), int(c["center_y_px"]));
                coarseWin_ = cv::Size(int(c["width_px"]), int(c["height_px"]));
            }
        }
    }
    if (world_.load(cfg_.world_dir, cfg_)) {
        message += "가상 장면을 불러옴 : " + cfg_.world_dir + " (영점 사진 " + world_.referencePhoto() + ")\n";
        if (!referenceMatchesWorld())
            message += "주의 : 설정의 영점 사진(" + cfg_.reference_photo + ")과 장면의 영점 사진(" + world_.referencePhoto() +
                       ")이 다름 → [파노라마 생성] 으로 다시 만들어야 적용됨\n";
    }
    else message += "가상 장면이 없음 → 먼저 만들기 (--build 또는 [파노라마 생성], 데이터 필요)\n";
    std::string em;
    est_ = makeEstimator(cfg_, em);
    message += em;
    if (!est_ && cfg_.estimator_type == "plugin") {   // 플러그인을 못 읽으면 내장 추정기로 시작 (화면에서 다시 고를 수 있게)
        SimConfig k = cfg_;
        k.estimator_type = "ksh";
        std::string km;
        est_ = makeEstimator(k, km);
        if (est_) { cfg_.estimator_type = "ksh"; message += "\n→ 내장 추정기로 시작 : " + km; }
    }
    return est_ != nullptr;
}

bool Simulator::setEstimator(const std::string& type, const std::string& plugin, std::string& message) {
    SimConfig c = cfg_;
    c.estimator_type = type;
    if (type == "plugin") c.estimator_plugin = plugin;
    est_.reset();   // ksh 는 전역 상태라 새로 만들기 전에 이전 것을 지움
    std::unique_ptr<Estimator> e = makeEstimator(c, message);
    if (!e) {
        std::string m2;
        est_ = makeEstimator(cfg_, m2);   // 이전 추정기로 되돌림
        return false;
    }
    est_ = std::move(e);
    cfg_.estimator_type = c.estimator_type;
    cfg_.estimator_plugin = c.estimator_plugin;
    return true;
}

bool Simulator::buildWorld(const Progress& progress, std::string& message) {
    // 이전 장면 보관 (지우지 않고 world/_backup/날짜/ 로 복사)
    std::error_code ec;
    const fs::path wd = u8p(cfg_.world_dir);
    if (fs::exists(wd / "field.yml", ec)) {
        const std::time_t now = std::time(nullptr);
        std::tm tm{};
#ifdef _WIN32
        localtime_s(&tm, &now);
#else
        localtime_r(&now, &tm);
#endif
        char stamp[32];
        std::strftime(stamp, sizeof(stamp), "%Y%m%d_%H%M%S", &tm);
        const fs::path bk = wd / "_backup" / stamp;
        fs::create_directories(bk, ec);
        for (const auto& e : fs::directory_iterator(wd, ec))
            if (e.is_regular_file()) fs::copy_file(e.path(), bk / e.path().filename(), fs::copy_options::overwrite_existing, ec);
    }
    World w;
    if (!w.build(cfg_, coarseCenter_, coarseWin_, progress, message)) return false;
    if (!w.save(cfg_.world_dir)) { message = "가상 장면 저장 실패 : " + cfg_.world_dir; return false; }
    world_ = w;
    world_.load(cfg_.world_dir, cfg_);   // 설정(잡음·상수항)을 반영해 다시 읽음
    message = "가상 장면을 만들어 저장함 : " + cfg_.world_dir + " (영점 사진 " + world_.referencePhoto() +
              f2(", 1차 지령 좌표에서 Y %+.1f µm, Θ %+.4f°)", world_.referencePose().y_mm * 1000.0, world_.referencePose().t_deg);
    return true;
}

bool Simulator::setReferencePhoto(const std::string& stem, std::string& message) {
    if (stem.empty()) { message = "영점 사진 이름이 비어 있음"; return false; }
    if (!cfg_.image_dir.empty() && !findPhoto(scanPhotos(cfg_.image_dir), stem)) {
        message = "데이터 폴더에 " + stem + " 사진이 없음 : " + cfg_.image_dir;
        return false;
    }
    cfg_.reference_photo = stem;
    // 설정 파일에서 reference_photo 줄만 바꿈 (주석 유지). 없으면 paths 의 result_dir 줄 다음에 넣음
    std::ifstream in(u8p(configPath_), std::ios::binary);
    std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    in.close();
    const std::string nl = text.find("\r\n") != std::string::npos ? "\r\n" : "\n";
    const std::string line = "   reference_photo: \"" + stem + "\"";
    const std::regex re(R"((^|\n)([ \t]*)reference_photo[ \t]*:[^\r\n]*)");
    std::smatch m;
    if (std::regex_search(text, m, re)) {
        // 줄 끝 주석은 버리고 새 값 + 설명 주석
        text.replace(size_t(m.position(0)) + m[1].length(), size_t(m.length(0)) - m[1].length(),
                     m[2].str() + "reference_photo: \"" + stem + "\"   # 영점 사진 — 참 자세 (0, 0) = 이 사진 (바꾸면 가상 장면 다시 만들기)");
    } else {
        const size_t p = text.find("result_dir:");
        if (p == std::string::npos) { message = "설정 파일에 paths.result_dir 줄이 없어 영점 사진을 기록하지 못함"; return false; }
        const size_t e = text.find('\n', p);
        text.insert(e == std::string::npos ? text.size() : e + 1, line + "   # 영점 사진 — 참 자세 (0, 0) = 이 사진" + nl);
    }
    std::ofstream out(u8p(configPath_), std::ios::binary);
    out << text;
    if (!out) { message = "설정 파일을 쓰지 못함 : " + configPath_; return false; }
    message = "영점 사진을 " + stem + " 로 바꿈 (설정 저장) → 가상 장면을 다시 만들면 적용됨";
    return true;
}

std::vector<std::string> Simulator::referenceCandidates() const {
    std::vector<std::string> v;
    if (!cfg_.image_dir.empty())
        for (const auto& p : scanPhotos(cfg_.image_dir))
            if (p.is_ref && std::find(v.begin(), v.end(), p.stem) == v.end()) v.push_back(p.stem);
    for (const std::string& s : {world_.ready() ? world_.referencePhoto() : std::string(), cfg_.reference_photo})
        if (!s.empty() && std::find(v.begin(), v.end(), s) == v.end()) v.push_back(s);
    std::sort(v.begin(), v.end());
    return v;
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
    referenceNote_.clear();
    zero_ = Pose();
    zeroApplied_ = false;
    if (!world_.ready() || !est_) return false;
    const auto zeroImage = [&]() { return cfg_.teach_center_photo ? world_.renderReference(&rng) : world_.render(Pose(), &rng); };
    const bool ok = est_->setReference(zeroImage());
    if (!ok) referenceNote_ = "추정기가 기준 사진 교체를 지원하지 않음 → 추정기 자체 기준 사진 사용";
    // 영점 맞춤 : 참 자세 (0, 0) 사진(새 촬영 잡음)을 한 번 추정 → 그 값이 영점 차이
    if (cfg_.zero_calibration == 1 || (cfg_.zero_calibration == 2 && !ok)) {
        const EstimateResult z = est_->estimate(zeroImage());
        if (z.ok) {
            zero_.y_mm = z.y_mm; zero_.t_deg = z.t_deg;
            zeroApplied_ = true;
            referenceNote_ += std::string(referenceNote_.empty() ? "" : "\n") +
                              f2("영점 맞춤 : (0, 0) 사진의 추정 Y %+.1f µm, Θ %+.4f° → 이후 추정에서 뺌", z.y_mm * 1000.0, z.t_deg);
        } else {
            referenceNote_ += std::string(referenceNote_.empty() ? "" : "\n") + "영점 맞춤 실패 (영점 차이 0 으로 진행) : " + z.message;
        }
    }
    return ok;
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
    EstimateResult a = est_ ? est_->estimate(img) : EstimateResult();
    s.ms_align = msSince(t0);
    s.est_ok = a.ok;
    s.est_raw.y_mm = a.y_mm; s.est_raw.t_deg = a.t_deg;
    if (a.ok && zeroApplied_) { a.y_mm -= zero_.y_mm; a.t_deg -= zero_.t_deg; }   // 영점 맞춤
    s.est.y_mm = a.y_mm; s.est.t_deg = a.t_deg;
    if (tr.steps.empty()) { tr.zero = zero_; tr.zero_applied = zeroApplied_; }

    if (!s.est_ok) {
        tr.reason = "측정 실패 : " + (est_ ? a.message : std::string("추정기 없음"));
    } else if (inTol(a.y_mm, a.t_deg, cfg_.control_stop_ratio)) {
        s.stop = true;
        tr.stopped = true;
    } else if (tr.moves >= cfg_.control_max_moves) {
        tr.reason = "이동 한도(" + std::to_string(cfg_.control_max_moves) + "회) 초과";
    } else {
        // 보정 명령 = −추정, 한 번 최대 이동으로 자름
        s.cmd.y_mm = clampAbs(-a.y_mm, cfg_.robot_max_step_y_mm);
        s.cmd.t_deg = clampAbs(-a.t_deg, cfg_.robot_max_step_t_deg);
        // 로봇 오차 : 배율 오차(가정) + 반복 정밀도(실측 σ, 정규 분포)
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
    o << "추정기 : " << estimatorName() << "\n";
    o << "영점 맞춤 : " << (cfg_.zero_calibration == 0 ? "안 함" : cfg_.zero_calibration == 1 ? "항상" : "자동 (기준 사진 교체 미지원 추정기만)");
    {
        int nz = 0; double zy = 0, zt = 0;
        for (const auto& t : m.trials) if (t.zero_applied) { ++nz; zy += t.zero.y_mm; zt += t.zero.t_deg; }
        if (nz) o << f2(" — 적용 %.0f회, 평균 영점 차이 Y %+.1f µm, Θ %+.4f°", nz, zy / nz * 1000.0, zt / nz);
    }
    o << "\n";
    o << f2("로봇 오차 : 반복 정밀도 σ Y %.2f µm, Θ %.5f° (실측) · 배율 σ %.2f %% (가정)\n", cfg_.robot_repeat_y_um, cfg_.robot_repeat_t_deg,
            cfg_.robot_gain_error * 100.0);
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
    f << "\xEF\xBB\xBF" << "k,true_y_mm,true_t_deg,est_y_mm,est_t_deg,est_ok,cmd_y_mm,cmd_t_deg,applied_y_mm,applied_t_deg,err_y_um,err_t_deg,true_in_tol,stop,ms_render,ms_align,"
                             "raw_est_y_mm,raw_est_t_deg,zero_y_mm,zero_t_deg\n";
    f.precision(8);
    for (const auto& s : t.steps)
        f << s.k << ',' << s.truth.y_mm << ',' << s.truth.t_deg << ',' << s.est.y_mm << ',' << s.est.t_deg << ',' << int(s.est_ok) << ','
          << s.cmd.y_mm << ',' << s.cmd.t_deg << ',' << s.applied.y_mm << ',' << s.applied.t_deg << ',' << s.truth.y_mm * 1000.0 << ','
          << s.truth.t_deg << ',' << int(s.truth_in_tol) << ',' << int(s.stop) << ',' << s.ms_render << ',' << s.ms_align << ','
          << s.est_raw.y_mm << ',' << s.est_raw.t_deg << ',' << t.zero.y_mm << ',' << t.zero.t_deg << '\n';
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
    if (!est_) { vs.table = "추정기가 없음"; return vs; }
    const std::vector<Photo> all = scanPhotos(cfg_.image_dir);
    const Photo* r1 = findPhoto(all, "center");
    const Photo* r2 = findPhoto(all, "center1");
    if (!r1 || !r2 || !world_.ready()) { vs.table = "기준 사진(center, center1) 또는 가상 장면이 없음 — 데이터 폴더 : " + cfg_.image_dir; return vs; }
    const cv::Mat ref1 = readImage(r1->path, cv::IMREAD_GRAYSCALE), ref2 = readImage(r2->path, cv::IMREAD_GRAYSCALE);
    cv::RNG rng(cfg_.seed);
    const cv::Mat refSim = cfg_.teach_center_photo ? world_.renderReference(&rng) : world_.render(Pose(), &rng);

    std::vector<Photo> ph;
    for (const auto& p : all) if (!p.is_ref && (p.set != 3 || cfg_.validate_include_set3)) ph.push_back(p);
    bool refSupported = true;
    for (size_t i = 0; i < ph.size(); ++i) {
        if (cancel && *cancel) break;
        const Photo& p = ph[i];
        ValidationRow r;
        r.stem = p.stem; r.label = p.pose;
        if (p.set == 1) r.group = p.pose.y_mm == 0.0 ? "1차 Θ 스윕" : (firstSession(p) ? "1차 Y 스윕 (첫 측정)" : "1차 Y 스윕 (48분 뒤)");
        else if (p.set == 3) r.group = std::abs(p.pose.t_deg) <= 0.2 + 1e-9 ? "3차 Θ ≤ 0.2°" : "3차 Θ > 0.2°";
        else r.group = std::abs(p.pose.t_deg) <= 0.2 + 1e-9 ? "2차 목표 근처" : (std::abs(std::abs(p.pose.t_deg) - 1.0) < 1e-9 ? "2차 Θ 1°" : "2차 Θ 2°");
        // 실제 사진 (같은 날 기준 사진 : 1차 center, 2·3차 center1)
        refSupported = est_->setReference(p.set == 1 ? ref1 : ref2) && refSupported;
        cv::Mat img = readImage(p.path, cv::IMREAD_GRAYSCALE);
        EstimateResult a = est_->estimate(img);
        r.ok_real = a.ok; r.est_real = {a.y_mm, a.t_deg};
        // 합성 사진 (합성 기준 사진)
        est_->setReference(refSim);
        // 라벨은 1차 지령 좌표 → 참 자세(영점 사진 기준) = 라벨 − 영점 사진 자세. 추정값은 다시 1차 지령 좌표로 돌려 비교
        const Pose rp = world_.referencePose();
        cv::Mat sim = world_.render(Pose{p.pose.y_mm - rp.y_mm, p.pose.t_deg - rp.t_deg}, &rng);
        a = est_->estimate(sim);
        r.ok_sim = a.ok; r.est_sim = {a.y_mm + rp.y_mm, a.t_deg + rp.t_deg};
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
    vs.table = "추정기 : " + est_->name() + (refSupported ? "" : "  (기준 사진 교체 미지원 — 세트별 기준 사진을 바꾸지 못해 결과 해석 주의)") + "\n\n" + o.str();
    return vs;
}

} // namespace sim
