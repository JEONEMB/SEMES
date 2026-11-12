// capture_dataset — 알고리즘 학습·평가용 사진 자동 촬영 (라즈베리파이)
//
//  센터에서 정해진 간격으로 스테이지를 옮기며 찍고, SEMES 사진과 같은 폴더·파일 이름으로 저장한다.
//
//   sudo ./build/capture_dataset --plan y    -o ~/data/rig_1st --calib ../camera_calib/camera_calib.yml --center-here
//   sudo ./build/capture_dataset --plan set1 -o ~/data/rig_1st --calib ... --center 25.000 0.000
//   sudo ./build/capture_dataset --plan set2 -o ~/data/rig_2nd --calib ... --center-here
//   ./build/capture_dataset --plan set1 -o x --dry-run            장치 없이 계획만 출력
//
//  한 장 촬영 순서
//   1. 목표보다 접근 거리만큼 −쪽 → 목표 (늘 같은 방향에서 들어가 백래시 영향을 같게)
//   2. 흔들림이 멎을 때까지 기다린 뒤 촬영 (Camera::captureSettled). 안 멎으면 다시 시도
//   3. STATUS 로 실제 위치 확인
//   4. 3840x2160 BMP (알고리즘 입력) + 원본 PNG (<출력>_raw/) + 같은 이름 TXT + manifest.csv 한 줄
//
//  센터 사진은 시작할 때 찍고, --check-every 장마다·끝날 때 다시 찍어 처음 센터와 비교한다
//  (배경 전체 이동 = 카메라·배경이 움직임, 플레이트 창 이동 = 스테이지 반복 정밀도). 결과는 _check/
//
//  Ctrl-C: 지금 사진까지 저장하고 멈춘다. 같은 명령에 --resume 을 붙이면 없는 사진만 이어서 찍는다
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <exception>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "capture/plan.hpp"
#include "stage/stage.hpp"
#include "stage_camera/camera.hpp"

namespace fs = std::filesystem;
using namespace capture;
using Clock = std::chrono::steady_clock;

namespace {

std::atomic<bool> g_stop{false};

void onSigint(int)
{
    if (g_stop.exchange(true)) std::_Exit(130);  // 두 번째 Ctrl-C 는 바로 종료
}

void usage()
{
    std::fprintf(stderr,
                 "사용법: capture_dataset --plan set1|y|t|set2|grid -o 출력폴더 --calib camera_calib.yml\n"
                 "                        (--center-here | --center Y_MM T_DEG) [선택]\n"
                 "\n"
                 "  계획\n"
                 "    set1  Y 단독 + Θ 단독 (학습, center.bmp)      y / t : 그중 한 축만\n"
                 "    set2  Y·Θ 동시 이동, 자세마다 --repeat 장 (평가, center1.bmp)\n"
                 "    grid  Y·Θ 4사분면 격자, 'ㄹ'자 순서 (학습, center.bmp, Grid/\"Y+0.10 T-0.05 (1)\")\n"
                 "    --y-range MM (2.0)  --y-step MM (0.05)  --t-range DEG (1.0)  --t-step DEG (0.01)\n"
                 "    --grid-y-step MM (0.1)  --grid-t-step DEG (0.05)   grid 간격 (범위는 --y-range/--t-range)\n"
                 "    --poses FILE.csv    set2 자세 목록 \"y_mm,t_deg\" (기본 20자세)   --repeat N (3)\n"
                 "    --center-name NAME  센터 사진 이름 (기본 set2 = center1, 그 밖 = center)\n"
                 "\n"
                 "  스테이지\n"
                 "    --center-here       지금 위치를 센터로 (stagectl 로 맞춰 둔 뒤)\n"
                 "    --center Y T        HOME 기준 절대 좌표 (mm, °)\n"
                 "    --home none|y|t|both  시작 전 원점 복귀 (기본 none = 이미 되어 있어야 함)\n"
                 "    --approach-y MM (0.2)  --approach-t DEG (0.1)  접근 거리, 0 = 바로 이동\n"
                 "    --device PATH (/dev/robot_stage)\n"
                 "\n"
                 "  촬영\n"
                 "    --average N (1)  --center-average N (4)  --settle-retries N (2)\n"
                 "    --check-every N (100, 0 = 끝에만)  --plate-roi CX,CY,W,H (1901,1068,900,400)\n"
                 "    --exposure US  --gain G  --zoom Z  --settle-shift PX  --settle-frames N  --settle-timeout MS\n"
                 "    --no-raw  원본 저장 안 함\n"
                 "\n"
                 "  --resume   있는 사진은 건너뛰고 이어서 찍기\n"
                 "  --dry-run  장치 없이 계획만 출력\n");
}

std::string nowText()
{
    const std::time_t t = std::time(nullptr);
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", std::localtime(&t));
    return buf;
}

double sec(Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double>(b - a).count(); }

std::string fmtPos(const stage::Position &p)
{
    char buf[64];
    std::snprintf(buf, sizeof(buf), "Y %.3f mm, Θ %.3f°", p.yMm(), p.tDeg());
    return buf;
}

int32_t mustMilli(const char *flag, const std::string &s)
{
    int32_t v = 0;
    if (!parseMilli(s, v)) {
        std::fprintf(stderr, "%s 값을 읽을 수 없음 (소수 셋째 자리까지): %s\n", flag, s.c_str());
        std::exit(2);
    }
    return v;
}

void saveImage(const fs::path &path, const cv::Mat &img)
{
    fs::create_directories(path.parent_path());
    if (!cv::imwrite(path.string(), img)) throw std::runtime_error("저장 실패: " + path.string());
}

// a → b 이동량 (px). roi 가 비어 있으면 전체
cv::Point2d shiftBetween(const cv::Mat &a, const cv::Mat &b, const cv::Rect &roi, double *response)
{
    const cv::Rect r = roi.area() > 0 ? (roi & cv::Rect(0, 0, a.cols, a.rows)) : cv::Rect(0, 0, a.cols, a.rows);
    cv::Mat fa, fb, win;
    a(r).convertTo(fa, CV_32F);
    b(r).convertTo(fb, CV_32F);
    cv::createHanningWindow(win, fa.size(), CV_32F);
    return cv::phaseCorrelate(fa, fb, win, response);
}

struct Options {
    std::string plan, out, poses_file, center_name, device = "/dev/robot_stage", home = "none";
    int32_t y_range = 2000, y_step = 50, t_range = 1000, t_step = 10;
    int32_t grid_y_step = 100, grid_t_step = 50;
    int32_t approach_y = 200, approach_t = 100;
    int repeat = 3, average = 1, center_average = 4, settle_retries = 2, check_every = 100;
    bool center_here = false, have_center = false, raw = true, resume = false, dry_run = false;
    stage::Position center;
    cv::Rect plate_roi{1901 - 450, 1068 - 200, 900, 400};
    stage_camera::CameraOptions cam;
    bool have_calib = false;
};

// 한 장 기록 (TXT 와 manifest.csv 에 같은 값)
struct Record {
    std::string file;
    Rel rel;
    stage::Position target, reported;
    stage::DoneResult done = stage::DoneResult::Unknown;
    stage_camera::SettleResult settle;
    int tries = 0, frames_averaged = 1;
    stage_camera::FrameInfo info;
    double mean_gray = 0, move_s = 0, shot_s = 0;
    std::string time;
};

void writeTxt(const fs::path &path, const Record &r)
{
    std::ofstream f(path);
    f << "file: " << r.file << "\n"
      << "command_y_mm: " << fmtMilli(r.rel.y_um) << "\n"
      << "command_t_deg: " << fmtMilli(r.rel.t_mdeg) << "\n"
      << "target_abs_y_um: " << r.target.y_um << "\n"
      << "target_abs_t_mdeg: " << r.target.t_mdeg << "\n"
      << "status_abs_y_um: " << r.reported.y_um << "\n"
      << "status_abs_t_mdeg: " << r.reported.t_mdeg << "\n"
      << "move_result: " << stage::toString(r.done) << "\n"
      << "settled: " << (r.settle.settled ? 1 : 0) << "\n"
      << "settle_tries: " << r.tries << "\n"
      << "settle_frames: " << r.settle.frames << "\n"
      << "settle_waited_ms: " << r.settle.waited_ms << "\n"
      << "settle_last_shift_px: " << r.settle.last_shift_px << "\n"
      << "settle_max_shift_px: " << r.settle.max_shift_px << "\n"
      << "frames_averaged: " << r.frames_averaged << "\n"
      << "frame_sequence: " << r.info.sequence << "\n"
      << "sensor_timestamp_ns: " << r.info.sensor_timestamp_ns << "\n"
      << "exposure_us: " << r.info.exposure_us << "\n"
      << "analogue_gain: " << r.info.analogue_gain << "\n"
      << "lux: " << r.info.lux << "\n"
      << "mean_gray: " << r.mean_gray << "\n"
      << "time: " << r.time << "\n";
}

void appendManifest(const fs::path &path, const Record &r)
{
    const bool fresh = !fs::exists(path);
    std::ofstream f(path, std::ios::app);
    if (fresh)
        f << "file,command_y_mm,command_t_deg,target_y_um,target_t_mdeg,status_y_um,status_t_mdeg,move_result,"
             "settled,settle_tries,settle_waited_ms,settle_max_shift_px,exposure_us,analogue_gain,lux,mean_gray,"
             "move_s,shot_s,time\n";
    char buf[512];
    std::snprintf(buf, sizeof(buf), "\"%s\",%s,%s,%d,%d,%d,%d,%s,%d,%d,%d,%.3f,%d,%.3f,%.1f,%.2f,%.2f,%.2f,%s\n",
                  r.file.c_str(), fmtMilli(r.rel.y_um).c_str(), fmtMilli(r.rel.t_mdeg).c_str(), r.target.y_um,
                  r.target.t_mdeg, r.reported.y_um, r.reported.t_mdeg, stage::toString(r.done),
                  r.settle.settled ? 1 : 0, r.tries, r.settle.waited_ms, r.settle.max_shift_px, r.info.exposure_us,
                  r.info.analogue_gain, r.info.lux, r.mean_gray, r.move_s, r.shot_s, r.time.c_str());
    f << buf;
}

class Session {
public:
    Session(const Options &o, stage::Stage &st, stage_camera::Camera &cam, const stage::Limits &lim)
        : o_(o), st_(st), cam_(cam), lim_(lim), root_(o.out), raw_root_(o.out + "_raw"), check_(root_ / "_check")
    {
        fs::create_directories(check_);
    }

    stage::Position absOf(const Rel &r) const
    {
        return {o_.center.y_um + r.y_um, o_.center.t_mdeg + r.t_mdeg};
    }

    // 접근 위치를 거쳐 목표로. 실패하면 예외
    stage::MoveResult goTo(const stage::Position &target)
    {
        const bool t_active = !(lim_.t_min_mdeg == 0 && lim_.t_max_mdeg == 0);
        const stage::Position ap{o_.approach_y, t_active ? o_.approach_t : 0};
        if (ap.y_um != 0 || ap.t_mdeg != 0) {
            const stage::Position pre = lim_.clamp({target.y_um - ap.y_um, target.t_mdeg - ap.t_mdeg});
            const stage::MoveResult r = st_.moveTo(pre);
            if (!r.ok())
                throw std::runtime_error(std::string("접근 위치 이동 실패: ") + stage::toString(r.result) + " (" +
                                         fmtPos(r.pos) + ")");
        }
        const stage::MoveResult r = st_.moveTo(target);
        if (!r.ok())
            throw std::runtime_error(std::string("이동 실패: ") + stage::toString(r.result) + " (" + fmtPos(r.pos) +
                                     ")");
        return r;
    }

    // 이동 → 촬영 → 저장. 반환: 촬영한 센서 원본 아닌 3840x2160 사진
    cv::Mat shoot(const Rel &rel, const fs::path &rel_path, int average, bool record)
    {
        Record r;
        r.rel = rel;
        r.file = rel_path.generic_string();
        r.target = absOf(rel);

        const auto t0 = Clock::now();
        const stage::MoveResult mv = goTo(r.target);
        r.done = mv.result;
        const auto t1 = Clock::now();

        stage_camera::Shot s;
        for (r.tries = 1;; ++r.tries) {
            s = cam_.captureSettled(average);
            if (s.settle.settled || r.tries > o_.settle_retries) break;
            std::printf("    흔들림 안 멎음 (최대 %.2fpx) → 다시 (%d)\n", s.settle.max_shift_px, r.tries);
        }
        const auto t2 = Clock::now();
        if (!s.settle.settled) {
            ++unsettled_;
            std::printf("    주의: %d번 시도해도 안 멎음. 마지막 사진을 저장하고 TXT 에 settled 0 표시\n", r.tries);
        }

        r.reported = st_.status().pos;
        if (r.reported.y_um != r.target.y_um || r.reported.t_mdeg != r.target.t_mdeg) {
            ++mismatch_;
            std::printf("    주의: STATUS 위치 %s ≠ 목표 %s\n", fmtPos(r.reported).c_str(), fmtPos(r.target).c_str());
        }
        r.settle = s.settle;
        r.frames_averaged = s.frames_averaged;
        r.info = s.info;
        r.mean_gray = cv::mean(s.image)[0];
        r.move_s = sec(t0, t1);
        r.shot_s = sec(t1, t2);
        r.time = nowText();

        fs::path img = root_ / rel_path;
        img += ".bmp";
        saveImage(img, s.image);
        if (o_.raw) {
            fs::path raw = raw_root_ / rel_path;
            raw += ".png";
            saveImage(raw, s.raw);
        }
        fs::path txt = root_ / rel_path;
        txt += ".txt";
        writeTxt(txt, r);
        if (record) appendManifest(root_ / "manifest.csv", r);
        return s.image;
    }

    // 센터를 다시 찍어 처음 센터와 비교
    void checkCenter(const cv::Mat &ref, const std::string &tag)
    {
        const cv::Mat now = shoot({}, fs::path("_check") / ("center_" + tag), o_.center_average, false);
        double resp_all = 0, resp_plate = 0;
        const cv::Point2d all = shiftBetween(ref, now, {}, &resp_all);
        const cv::Point2d plate = shiftBetween(ref, now, o_.plate_roi, &resp_plate);
        const bool bad = std::hypot(all.x, all.y) > kWarnPx || std::hypot(plate.x, plate.y) > kWarnPx;
        char line[256];
        std::snprintf(line, sizeof(line),
                      "%s  %-10s 전체 (%+.3f, %+.3f) px  플레이트 (%+.3f, %+.3f) px  상관 %.2f/%.2f%s\n",
                      nowText().c_str(), tag.c_str(), all.x, all.y, plate.x, plate.y, resp_all, resp_plate,
                      bad ? "  ← 큼" : "");
        std::printf("  센터 확인 %s", line);
        std::ofstream(check_ / "drift.txt", std::ios::app) << line;
        if (bad) ++drift_warn_;
    }

    int unsettled_ = 0, mismatch_ = 0, drift_warn_ = 0;
    static constexpr double kWarnPx = 0.3;  // Y 허용 45µm ≈ 0.64px (14 px/mm) 의 절반

private:
    const Options &o_;
    stage::Stage &st_;
    stage_camera::Camera &cam_;
    const stage::Limits &lim_;
    fs::path root_, raw_root_, check_;
};

}  // namespace

int main(int argc, char **argv)
{
    Options o;
    std::string cmdline;
    for (int i = 0; i < argc; ++i) cmdline += (i ? " " : "") + std::string(argv[i]);

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto val = [&]() -> std::string {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "%s 뒤에 값이 필요함\n", a.c_str());
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "--plan") o.plan = val();
        else if (a == "-o") o.out = val();
        else if (a == "--calib") { o.cam.calib_path = val(); o.have_calib = true; }
        else if (a == "--center-here") o.center_here = true;
        else if (a == "--center") {
            o.center.y_um = mustMilli("--center", val());
            o.center.t_mdeg = mustMilli("--center", val());
            o.have_center = true;
        }
        else if (a == "--center-name") o.center_name = val();
        else if (a == "--home") o.home = val();
        else if (a == "--device") o.device = val();
        else if (a == "--y-range") o.y_range = mustMilli("--y-range", val());
        else if (a == "--y-step") o.y_step = mustMilli("--y-step", val());
        else if (a == "--t-range") o.t_range = mustMilli("--t-range", val());
        else if (a == "--t-step") o.t_step = mustMilli("--t-step", val());
        else if (a == "--grid-y-step") o.grid_y_step = mustMilli("--grid-y-step", val());
        else if (a == "--grid-t-step") o.grid_t_step = mustMilli("--grid-t-step", val());
        else if (a == "--poses") o.poses_file = val();
        else if (a == "--repeat") o.repeat = std::atoi(val().c_str());
        else if (a == "--approach-y") o.approach_y = mustMilli("--approach-y", val());
        else if (a == "--approach-t") o.approach_t = mustMilli("--approach-t", val());
        else if (a == "--average") o.average = std::atoi(val().c_str());
        else if (a == "--center-average") o.center_average = std::atoi(val().c_str());
        else if (a == "--settle-retries") o.settle_retries = std::atoi(val().c_str());
        else if (a == "--check-every") o.check_every = std::atoi(val().c_str());
        else if (a == "--plate-roi") {
            int cx, cy, w, h;
            if (std::sscanf(val().c_str(), "%d,%d,%d,%d", &cx, &cy, &w, &h) != 4) {
                std::fprintf(stderr, "--plate-roi 는 CX,CY,W,H\n");
                return 2;
            }
            o.plate_roi = {cx - w / 2, cy - h / 2, w, h};
        }
        else if (a == "--exposure") o.cam.exposure_us = std::atoi(val().c_str());
        else if (a == "--gain") o.cam.analogue_gain = static_cast<float>(std::atof(val().c_str()));
        else if (a == "--zoom") o.cam.zoom = std::atof(val().c_str());
        else if (a == "--settle-shift") o.cam.settle.max_shift_px = std::atof(val().c_str());
        else if (a == "--settle-frames") o.cam.settle.stable_frames = std::atoi(val().c_str());
        else if (a == "--settle-timeout") o.cam.settle.timeout_ms = std::atoi(val().c_str());
        else if (a == "--no-raw") o.raw = false;
        else if (a == "--resume") o.resume = true;
        else if (a == "--dry-run") o.dry_run = true;
        else if (a == "-h" || a == "--help") { usage(); return 0; }
        else {
            std::fprintf(stderr, "모르는 옵션: %s\n\n", a.c_str());
            usage();
            return 2;
        }
    }

    // ---- 계획
    std::vector<ShotSpec> shots;
    if (o.plan == "set1" || o.plan == "y") {
        const auto v = planY(o.y_range, o.y_step);
        shots.insert(shots.end(), v.begin(), v.end());
    }
    if (o.plan == "set1" || o.plan == "t") {
        const auto v = planT(o.t_range, o.t_step);
        shots.insert(shots.end(), v.begin(), v.end());
    }
    if (o.plan == "set2") {
        std::vector<Rel> poses = defaultMixedPoses();
        if (!o.poses_file.empty()) {
            std::ifstream f(o.poses_file);
            if (!f) {
                std::fprintf(stderr, "자세 파일을 열 수 없음: %s\n", o.poses_file.c_str());
                return 2;
            }
            std::stringstream ss;
            ss << f.rdbuf();
            std::string err;
            poses.clear();
            if (!parsePoseCsv(ss.str(), poses, err)) {
                std::fprintf(stderr, "%s: %s\n", o.poses_file.c_str(), err.c_str());
                return 2;
            }
        }
        shots = planMixed(poses, o.repeat);
    }
    if (o.plan == "grid") shots = planGrid(o.y_range, o.grid_y_step, o.t_range, o.grid_t_step);
    if (shots.empty() || o.out.empty() || o.repeat < 1 || o.average < 1 || o.center_average < 1) {
        usage();
        return 2;
    }
    if (o.plan != "grid") sortForTravel(shots);   // grid 는 planGrid 가 이미 'ㄹ'자 순서
    if (o.center_name.empty()) o.center_name = o.plan == "set2" ? "center1" : "center";

    Rel lo{0, 0}, hi{0, 0};
    for (const auto &s : shots) {
        lo.y_um = std::min(lo.y_um, s.rel.y_um);
        lo.t_mdeg = std::min(lo.t_mdeg, s.rel.t_mdeg);
        hi.y_um = std::max(hi.y_um, s.rel.y_um);
        hi.t_mdeg = std::max(hi.t_mdeg, s.rel.t_mdeg);
    }
    std::printf("계획 %s: 센터 1장 + %zu장 → %s/\n", o.plan.c_str(), shots.size(), o.out.c_str());
    std::printf("  범위 Y %s ~ %s mm, Θ %s ~ %s° (센터 기준)\n", fmtMilli(lo.y_um).c_str(), fmtMilli(hi.y_um).c_str(),
                fmtMilli(lo.t_mdeg).c_str(), fmtMilli(hi.t_mdeg).c_str());
    std::printf("  처음 %s/%s, 마지막 %s/%s\n", shots.front().dir.c_str(), shots.front().stem.c_str(),
                shots.back().dir.c_str(), shots.back().stem.c_str());
    if (o.dry_run) {
        for (const auto &s : shots) std::printf("  %s/%s.bmp\n", s.dir.c_str(), s.stem.c_str());
        return 0;
    }
    if (!o.have_calib || (o.center_here == o.have_center)) {
        std::fprintf(stderr, "--calib 과, --center-here 또는 --center 중 하나가 필요함\n");
        return 2;
    }

    const fs::path root(o.out);
    const fs::path center_bmp = root / (o.center_name + ".bmp");
    if (fs::exists(center_bmp) && !o.resume) {
        std::fprintf(stderr, "%s 가 이미 있음. 이어 찍으려면 --resume, 새로 찍으려면 다른 -o\n",
                     center_bmp.string().c_str());
        return 2;
    }

    std::signal(SIGINT, onSigint);
    try {
        // ---- 스테이지
        stage::Stage st(o.device);
        const stage::Version ver = st.ping();
        std::printf("STM32 펌웨어 %d.%d.%d\n", ver.fw_major, ver.fw_minor, ver.fw_patch);
        if (o.home != "none") {
            const stage::Axis ax = o.home == "t" ? stage::Axis::T : o.home == "both" ? stage::Axis::Both : stage::Axis::Y;
            std::printf("원점 복귀 (%s)...\n", o.home.c_str());
            const stage::MoveResult r = st.home(ax);
            if (!r.ok()) throw std::runtime_error(std::string("HOME 실패: ") + stage::toString(r.result));
        }
        const stage::Status s0 = st.status();
        const stage::Limits lim = st.limits();
        const bool uses_t = lo.t_mdeg != 0 || hi.t_mdeg != 0;
        const bool t_active = !(lim.t_min_mdeg == 0 && lim.t_max_mdeg == 0);
        if (!s0.yHomed()) throw std::runtime_error("Y 원점 복귀가 안 되어 있음 → --home y 또는 stagectl home");
        if (uses_t && !t_active) throw std::runtime_error("펌웨어에서 Θ 축이 비활성 (CONFIG Θ 0/0) → --plan y 만 가능");
        if (uses_t && !s0.tHomed()) throw std::runtime_error("Θ 원점 복귀가 안 되어 있음 → --home both");
        if (o.center_here) o.center = s0.pos;
        std::printf("센터 %s  (소프트 리밋 Y %.3f~%.3f mm, Θ %.3f~%.3f°)\n", fmtPos(o.center).c_str(),
                    lim.y_min_um / 1000.0, lim.y_max_um / 1000.0, lim.t_min_mdeg / 1000.0, lim.t_max_mdeg / 1000.0);
        for (const Rel &corner : {lo, hi}) {
            const stage::Position p{o.center.y_um + corner.y_um, o.center.t_mdeg + corner.t_mdeg};
            if (!lim.contains(p))
                throw std::runtime_error("계획 범위가 소프트 리밋 밖: " + fmtPos(p) + " → 센터나 --y-range 조정");
        }

        // ---- 카메라
        std::printf("카메라 준비...\n");
        stage_camera::Camera cam(o.cam);
        Session ses(o, st, cam, lim);

        {
            std::ofstream f(root / "_check" / "session.txt", std::ios::app);
            f << "---- " << nowText() << "\n"
              << "command: " << cmdline << "\n"
              << "firmware: " << ver.fw_major << "." << ver.fw_minor << "." << ver.fw_patch << "\n"
              << "center_abs_y_um: " << o.center.y_um << "\ncenter_abs_t_mdeg: " << o.center.t_mdeg << "\n"
              << "limits_y_um: " << lim.y_min_um << " " << lim.y_max_um << "\n"
              << "limits_t_mdeg: " << lim.t_min_mdeg << " " << lim.t_max_mdeg << "\n"
              << "approach: " << o.approach_y << " um, " << o.approach_t << " mdeg\n"
              << "calib: " << o.cam.calib_path << "\nzoom: " << o.cam.zoom << "\n"
              << "exposure_us: " << o.cam.exposure_us << "\nanalogue_gain: " << o.cam.analogue_gain << "\n"
              << "shots: " << shots.size() << "\n";
        }

        // ---- 센터 (기준 사진)
        cv::Mat ref;
        if (fs::exists(center_bmp)) {
            ref = cv::imread(center_bmp.string(), cv::IMREAD_GRAYSCALE);
            std::printf("센터 사진 있음 → 그대로 사용: %s\n", center_bmp.string().c_str());
        } else {
            std::printf("센터 촬영 (%d장 평균)...\n", o.center_average);
            ref = ses.shoot({}, o.center_name, o.center_average, true);
        }

        // ---- 본 촬영
        const auto start = Clock::now();
        int done = 0, skipped = 0, since_check = 0;
        for (size_t i = 0; i < shots.size(); ++i) {
            if (g_stop) break;
            const ShotSpec &sp = shots[i];
            const fs::path rel_path = fs::path(sp.dir) / sp.stem;
            fs::path bmp = root / rel_path;
            bmp += ".bmp";
            if (fs::exists(bmp)) {
                ++skipped;
                continue;
            }
            ses.shoot(sp.rel, rel_path, o.average, true);
            ++done;
            const double per = sec(start, Clock::now()) / done;
            const size_t left = shots.size() - i - 1;
            std::printf("[%zu/%zu] %s  (%.1f s/장, 남은 약 %.0f분)\n", i + 1, shots.size(),
                        rel_path.generic_string().c_str(), per, per * left / 60.0);
            std::fflush(stdout);
            if (o.check_every > 0 && ++since_check >= o.check_every && left > 0) {
                char tag[32];
                std::snprintf(tag, sizeof(tag), "after_%04zu", i + 1);
                ses.checkCenter(ref, tag);
                since_check = 0;
            }
        }

        if (g_stop) {
            std::printf("\n중단됨 (%d장 찍음). 이어서 찍으려면 같은 명령에 --resume\n", done);
            st.moveTo(o.center);
            return 130;
        }
        ses.checkCenter(ref, "end");
        std::printf("\n완료: %d장 촬영, %d장 건너뜀, %.1f분\n", done, skipped, sec(start, Clock::now()) / 60.0);
        if (ses.unsettled_) std::printf("  흔들림 안 멎은 사진 %d장 (TXT settled: 0)\n", ses.unsettled_);
        if (ses.mismatch_) std::printf("  STATUS 위치가 목표와 다른 사진 %d장\n", ses.mismatch_);
        if (ses.drift_warn_)
            std::printf("  센터가 %.1fpx 넘게 움직임 %d회 → _check/drift.txt 확인. 장비가 움직였으면 다시 찍기\n",
                        Session::kWarnPx, ses.drift_warn_);
        std::printf("  사진 %s/  원본 %s_raw/  기록 manifest.csv, _check/\n", o.out.c_str(), o.out.c_str());
        return 0;
    } catch (const stage::StageError &e) {
        std::fprintf(stderr, "스테이지 오류 (%s): %s\n", stage::toString(e.kind()), e.what());
        if (e.kind() == stage::StageError::Kind::HomeLost)
            std::fprintf(stderr, "원점을 잃음 (STM32 재부팅?). 원점이 바뀌었으니 이 폴더에 이어 찍지 말고 새로 찍기\n");
        return 1;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "오류: %s\n", e.what());
        return 1;
    }
}
