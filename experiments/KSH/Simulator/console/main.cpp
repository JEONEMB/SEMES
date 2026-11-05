// =====================================================================================
// main.cpp — 영점 맞춤 시뮬레이터 (콘솔 + OpenCV 창)
//
//  작업 폴더 = Simulator 폴더 (config/, world/, results/ 가 있는 곳 — .vcxproj.user 에 설정)
//
//  인자
//    (없음)              가상 장면이 없으면 만들고, 무작위 시작 자세로 한 번 시험 (창 표시)
//    --build             실제 사진으로 가상 장면(운동장 + 파노라마) 만들기 → world/
//    --validate          같은 라벨의 실제 사진 vs 합성 사진에서 모델 추정 비교 → results/validation.md
//    --run [Y Θ]         한 번 시험 (시작 자세 mm, °. 없으면 무작위) → results/trial_*/
//    --monte [N]         N 번 시험 통계 (없으면 설정값) → results/monte_*.csv/.png/.md
//    --gui               창으로 진행 표시 (--run 과 함께. 아무 키 = 다음, ESC = 창 닫고 계속)
//    --config 파일       설정 파일 (기본 config/simulator.yml)
// =====================================================================================
#include "../core/sim_draw.hpp"
#include "../core/sim_runner.hpp"

#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <chrono>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>   // 콘솔 한글 출력 (UTF-8)
#endif

namespace fs = std::filesystem;
using namespace sim;

namespace {

std::string stamp() {
    const std::time_t t = std::time(nullptr);
    std::tm tm{};
#ifdef _WIN32
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    char b[32];
    std::strftime(b, sizeof(b), "%Y%m%d_%H%M%S", &tm);
    return b;
}

// 한 화면 : 왼쪽 = 카메라 + 파노라마, 오른쪽 = 궤적 + 수렴
cv::Mat dashboard(const Simulator& s, const TrialResult& t, const cv::Mat& img) {
    const StepRecord* st = t.steps.empty() ? nullptr : &t.steps.back();
    const cv::Mat cam = drawCamera(img, st, &alignment_last_info(), 1100);
    const cv::Mat pano = drawPanorama(s.world(), &t, 1100);
    const cv::Mat plane = drawPlane(t, s.config(), cv::Size(760, 420));
    const cv::Mat conv = drawConvergence(t, s.config(), cv::Size(760, 300));
    const int H = std::max(cam.rows + pano.rows, plane.rows + conv.rows);
    cv::Mat d(H, cam.cols + plane.cols, CV_8UC3, cv::Scalar(250, 250, 250));
    cam.copyTo(d(cv::Rect(0, 0, cam.cols, cam.rows)));
    pano.copyTo(d(cv::Rect(0, cam.rows, pano.cols, pano.rows)));
    plane.copyTo(d(cv::Rect(cam.cols, 0, plane.cols, plane.rows)));
    conv.copyTo(d(cv::Rect(cam.cols, plane.rows, conv.cols, conv.rows)));
    return d;
}

void printStep(const StepRecord& s, const SimConfig& c) {
    std::printf("  %2d | 참 Y %+8.4f mm Θ %+8.4f° | 추정 ", s.k, s.truth.y_mm, s.truth.t_deg);
    if (s.est_ok) std::printf("Y %+8.4f Θ %+8.4f", s.est.y_mm, s.est.t_deg); else std::printf("%-21s", "실패");
    if (s.stop) std::printf(" | 멈춤 → 참값 %s", s.truth_in_tol ? "허용 안" : "허용 밖");
    else if (s.est_ok) std::printf(" | 이동 Y %+7.4f Θ %+7.4f", s.applied.y_mm, s.applied.t_deg);
    std::printf(" | %4.0f ms\n", s.ms_render + s.ms_align);
    (void)c;
}

int doBuild(Simulator& s) {
    std::cout << "가상 장면 만들기 (실제 사진 → 운동장 + 파노라마). 수 분 걸림\n";
    int last = -1;
    std::string msg;
    const auto t0 = std::chrono::steady_clock::now();
    const bool ok = s.buildWorld([&](double f, const std::string& m) {
        const int p = int(f * 100);
        if (p / 5 != last / 5) { last = p; std::cout << "  " << p << " % " << m << std::endl; }
    }, msg);
    std::cout << msg << " (" << std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() << " s)\n";
    if (!ok) return 1;
    std::cout << s.world().report() << "\n";
    fs::create_directories(s.config().result_dir);
    cv::imwrite(s.config().result_dir + "/field_map.png", drawField(s.world(), 1200));
    cv::imwrite(s.config().result_dir + "/panorama_view.png", drawPanorama(s.world(), nullptr, 1600));
    return 0;
}

int doValidate(Simulator& s) {
    std::cout << "검증 : 같은 라벨의 실제 사진과 합성 사진을 모델에 넣어 비교 (사진 수 × 2 회 추정)\n";
    int last = -1;
    const ValidationSummary v = s.validate([&](double f, const std::string& m) {
        const int p = int(f * 100);
        if (p / 10 != last / 10) { last = p; std::cout << "  " << p << " % " << m << std::endl; }
    });
    std::cout << "\n" << v.table << "\n";
    fs::create_directories(s.config().result_dir);
    std::ofstream md(s.config().result_dir + "/validation.md");
    md << "# 합성 사진 검증 — 같은 라벨에서 실제 사진 vs 합성 사진의 모델 추정\n\n"
       << "- 실제 : 같은 날 기준 사진 (1차 center, 2차 center1)\n- 합성 : 자세 (0,0) 합성 사진을 기준으로\n\n" << v.table;
    std::ofstream csv(s.config().result_dir + "/validation.csv");
    csv << "\xEF\xBB\xBF" << "stem,group,y_mm,t_deg,real_y_mm,real_t_deg,real_ok,sim_y_mm,sim_t_deg,sim_ok\n";
    csv.precision(8);
    for (const auto& r : v.rows)
        csv << '"' << r.stem << "\",\"" << r.group << "\"," << r.label.y_mm << ',' << r.label.t_deg << ',' << r.est_real.y_mm << ',' << r.est_real.t_deg << ','
            << int(r.ok_real) << ',' << r.est_sim.y_mm << ',' << r.est_sim.t_deg << ',' << int(r.ok_sim) << '\n';
    std::cout << "저장 : " << s.config().result_dir << "/validation.md, validation.csv\n";
    return 0;
}

int doRun(Simulator& s, bool hasStart, Pose start, bool gui) {
    cv::RNG rng(s.config().seed ^ unsigned(std::time(nullptr)));
    if (!hasStart) start = s.randomStart(rng);
    s.teach(rng);
    const std::string dir = s.config().result_dir + "/trial_" + stamp();
    fs::create_directories(dir);
    std::cout << "시험 : 시작 " << poseText(start) << "  (허용 Y ±" << s.config().tol_y_um << " µm, Θ ±" << s.config().tol_t_deg << "°, 이동 한도 "
              << s.config().control_max_moves << "회)\n";
    bool show = gui;
    const TrialResult t = s.runTrial(start, rng, [&](const TrialResult& tr, const cv::Mat& img) {
        const StepRecord& st = tr.steps.back();
        printStep(st, s.config());
        const cv::Mat d = dashboard(s, tr, img);
        char name[64];
        std::snprintf(name, sizeof(name), "/step_%02d.jpg", st.k);
        cv::imwrite(dir + name, d, {cv::IMWRITE_JPEG_QUALITY, 85});
        if (show) {
            cv::namedWindow("Simulator", cv::WINDOW_NORMAL);
            cv::imshow("Simulator", d);
            if (cv::waitKey(0) == 27) { show = false; cv::destroyAllWindows(); }
        }
        return true;
    });
    std::cout << (t.success ? "성공" : "실패") << " : 이동 " << t.moves << "회, 멈춘 뒤 참 오차 " << poseText(t.final_truth)
              << (t.reason.empty() ? "" : "  (" + t.reason + ")") << "\n";
    s.writeTrialCsv(t, dir + "/trial.csv");
    cv::imwrite(dir + "/plane.png", drawPlane(t, s.config(), cv::Size(1200, 600)));
    cv::imwrite(dir + "/convergence.png", drawConvergence(t, s.config(), cv::Size(900, 400)));
    std::cout << "저장 : " << dir << "\n";
    if (show) cv::destroyAllWindows();
    return t.success ? 0 : 2;
}

int doMonte(Simulator& s, int n) {
    std::cout << "다회 시험 " << n << "회 (시작 자세 : Y ±" << s.config().start_range_y_mm << " mm, Θ ±" << s.config().start_range_t_deg
              << "° 균일 분포, seed " << s.config().seed << ")\n";
    const MonteSummary m = s.monteCarlo(n, s.config().seed, [&](int i, const TrialResult& t) {
        std::printf("  %3d | 시작 Y %+7.3f Θ %+7.3f | 이동 %2d | %s | 참 오차 Y %+6.1f µm Θ %+7.4f°\n", i + 1, t.start.y_mm, t.start.t_deg, t.moves,
                    t.success ? "성공" : "실패", t.final_truth.y_mm * 1000.0, t.final_truth.t_deg);
        std::fflush(stdout);
    });
    const std::string txt = s.monteText(m);
    std::cout << "\n" << txt;
    const std::string base = s.config().result_dir + "/monte_" + stamp();
    s.writeMonteCsv(m, base + ".csv");
    cv::imwrite(base + ".png", drawMonte(m, s.config(), cv::Size(1200, 520)));
    std::ofstream(base + ".md") << "# 영점 맞춤 다회 시험\n\n```\n" << txt << "```\n";
    std::cout << "저장 : " << base << ".csv / .png / .md\n";
    return 0;
}

} // namespace

int main(int argc, char** argv) {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif
    std::string cfgPath = "config/simulator.yml", mode;
    bool gui = false, hasStart = false;
    Pose start;
    int n = -1;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--config" && i + 1 < argc) cfgPath = argv[++i];
        else if (a == "--gui") gui = true;
        else if (a == "--build" || a == "--validate" || a == "--run" || a == "--monte") mode = a;
        else if (mode == "--run" && i + 1 < argc) { start.y_mm = std::atof(argv[i]); start.t_deg = std::atof(argv[++i]); hasStart = true; }
        else if (mode == "--monte") n = std::atoi(argv[i]);
    }
    std::cout << "OpenCV " << CV_VERSION << " · 영점 맞춤 시뮬레이터\n";
    Simulator s;
    std::string msg;
    const bool ok = s.init(cfgPath, msg);
    std::cout << msg << "\n";
    if (!ok) return 1;

    if (mode == "--build") return doBuild(s);
    if (!s.worldReady()) {
        std::cout << "가상 장면이 없어 먼저 만듭니다.\n";
        if (doBuild(s)) return 1;
    }
    if (mode == "--validate") return doValidate(s);
    if (mode == "--monte") return doMonte(s, n > 0 ? n : s.config().monte_carlo_trials);
    if (mode.empty()) gui = true;
    return doRun(s, hasStart, start, gui);
}
