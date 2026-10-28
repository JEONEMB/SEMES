// =====================================================================================
// real_compare.cpp — 실제 사진으로 운동 모델 확인 (설명은 real_compare.hpp)
// =====================================================================================
#include "real_compare.hpp"
#include "plot_utils.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>

namespace fs = std::filesystem;

namespace mm {

// ---------------------------------------------------------------------------------- 목록
std::vector<ImageEntry> scanImages(const std::string& dir) {
    std::vector<ImageEntry> out;
    if (!fs::exists(dir)) return out;
    for (const auto& f : fs::recursive_directory_iterator(dir)) {
        if (!f.is_regular_file()) continue;
        std::string ext = f.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (ext != ".bmp" && ext != ".png") continue;
        ImageEntry e;
        e.path = f.path().string();
        e.stem = f.path().stem().string();
        if (!parseLabel(e.stem, e.pose, e.set_id)) continue;
        e.is_ref = e.stem.rfind("center", 0) == 0;
        out.push_back(e);
    }
    std::sort(out.begin(), out.end(), [](const ImageEntry& a, const ImageEntry& b) {
        if (a.set_id != b.set_id) return a.set_id < b.set_id;
        if (a.is_ref != b.is_ref) return a.is_ref;
        const int ka = a.pose.t_deg != 0 && a.pose.y_mm == 0, kb = b.pose.t_deg != 0 && b.pose.y_mm == 0;
        if (ka != kb) return ka < kb;
        if (a.pose.y_mm != b.pose.y_mm) return a.pose.y_mm < b.pose.y_mm;
        if (a.pose.t_deg != b.pose.t_deg) return a.pose.t_deg < b.pose.t_deg;
        return a.stem < b.stem;
    });
    return out;
}

const std::vector<Region>& defaultRegions() {
    // 출처 : src/KSH/Hypothesis_2/Center_check/landmarks.yml (center.bmp 좌표)
    static const std::vector<Region> R = {
        {"QR_L", 0, {1650, 1065}, {110, 110}},  {"QR_C", 0, {1907, 1065}, {110, 110}},
        {"QR_R", 0, {2165, 1065}, {110, 110}},  {"wall_U1", 1, {2483, 367}, {60, 60}},
        {"wall_U2", 1, {2483, 617}, {60, 60}},  {"wall_W1", 2, {967, 1357}, {60, 60}},
        {"wall_W2", 2, {2867, 1357}, {60, 60}}, {"wall_W3", 2, {1010, 1677}, {60, 60}},
        {"wall_W4", 2, {2820, 1683}, {60, 60}},
    };
    return R;
}

static const char* kGroupName[3] = {"plate", "wall_upper", "wall_lower"};

// ---------------------------------------------------------------------------------- 측정
namespace {

cv::Mat loadGray32(const std::string& path) {
    cv::Mat g = cv::imread(path, cv::IMREAD_GRAYSCALE), f;
    if (g.empty()) return f;
    g.convertTo(f, CV_32F);
    return f;
}

bool inside(const cv::Rect& r, const cv::Size& s) { return r.x >= 0 && r.y >= 0 && r.x + r.width <= s.width && r.y + r.height <= s.height; }

cv::Point iround(const cv::Point2d& p) { return cv::Point(int(std::lround(p.x)), int(std::lround(p.y))); }

// 기준 ref 의 영역 roi 가 cur 에서 얼마나 움직였는지 (guess 근처) — 위상 상관 + ECC
//  wide = true  : 영역 2 배 창의 위상 상관으로 guess 를 먼저 보정 (배경 벽 — 플레이트와 이동량이 수 px 다름)
//  wide = false : guess 에서 바로 ECC (QR — 이웃 QR·모듈 무늬가 반복돼 넓은 창은 엉뚱한 봉우리에 붙음)
bool measureShift(const cv::Mat& ref, const cv::Mat& cur, const cv::Rect& roi, const cv::Point2d& guess, bool wide,
                  cv::Point2d& shift, double& eccOut) {
    cv::Point2d g = guess;
    if (wide) {
        const int W = std::max(roi.width, roi.height) * 2;
        const cv::Point c(roi.x + roi.width / 2, roi.y + roi.height / 2);
        const cv::Rect A(c.x - W / 2, c.y - W / 2, W, W);
        const cv::Rect B = A + iround(guess);
        if (!inside(A, ref.size()) || !inside(B, cur.size())) return false;
        cv::Mat win;
        cv::createHanningWindow(win, A.size(), CV_32F);
        g += cv::phaseCorrelate(ref(A).clone(), cur(B).clone(), win);  // clone 필수 : 창을 쓰면 입력을 제자리에서 곱해 기준 영상이 망가짐
    }
    // 2) ECC 평행이동 (정밀)
    const cv::Point gi = iround(g);
    const cv::Rect R1 = roi + gi;
    if (!inside(R1, cur.size())) return false;
    cv::Mat w = cv::Mat::eye(2, 3, CV_32F);
    w.at<float>(0, 2) = float(g.x - gi.x);
    w.at<float>(1, 2) = float(g.y - gi.y);
    try {
        eccOut = cv::findTransformECC(ref(roi), cur(R1), w, cv::MOTION_TRANSLATION,
                                      cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 100, 1e-6), cv::noArray());
    } catch (const cv::Exception&) {
        return false;
    }
    shift = cv::Point2d(gi.x + w.at<float>(0, 2), gi.y + w.at<float>(1, 2));
    return true;
}

} // namespace

double RealComparer::regionDepth(int group) const {
    return group == 0 ? cam_.z_plate : (group == 1 ? cam_.z_wall_upper : cam_.z_bg);
}

const cv::Mat& RealComparer::reference(int set_id) {
    auto it = refs_.find(set_id);
    if (it != refs_.end()) return it->second;
    const std::string name = set_id == 1 ? "center" : "center1";
    std::string path;
    for (const auto& e : scanImages(dir_)) if (e.stem == name) { path = e.path; break; }
    refNames_[set_id] = name;
    refs_[set_id] = path.empty() ? cv::Mat() : loadGray32(path);
    if (path.empty()) std::cerr << "[경고] 기준 영상 " << name << " 을(를) 찾지 못했습니다\n";
    return refs_[set_id];
}

bool RealComparer::compare(const ImageEntry& e, CompareResult& out, cv::Mat* panel) {
    const cv::Mat& ref = reference(e.set_id);
    if (ref.empty()) return false;
    const cv::Mat cur = loadGray32(e.path);
    if (cur.empty() || cur.size() != ref.size()) return false;
    out = CompareResult{};
    out.e = e;
    out.ref = refNames_[e.set_id];

    // 거친 이동 : 플레이트 주변 900 × 400 위상 상관 (모델 값은 쓰지 않음)
    const cv::Rect P(int(cam_.plate_center.x) - 450, int(cam_.plate_center.y) - 200, 900, 400);
    cv::Mat win;
    cv::createHanningWindow(win, P.size(), CV_32F);
    out.coarse = cv::phaseCorrelate(ref(P).clone(), cur(P).clone(), win);  // clone : 위와 같은 이유

    for (const auto& reg : defaultRegions()) {
        RegionResult rr;
        const cv::Rect roi(int(reg.c.x) - reg.half.width, int(reg.c.y) - reg.half.height, 2 * reg.half.width, 2 * reg.half.height);
        rr.ok = measureShift(ref, cur, roi, out.coarse, reg.group != 0, rr.meas, rr.ecc);
        if (rr.ok && reg.group == 0 && cv::norm(rr.meas - out.coarse) > 15.0) rr.ok = false;  // QR 은 플레이트와 같이 움직여야 함
        rr.pred = cam_.displacement(reg.c, regionDepth(reg.group), e.pose);
        out.r.push_back(rr);
    }
    if (!panel) return true;

    // ------------------------------------------------------------ 비교 화면
    cv::Mat& cv_ = *panel;
    cv_ = cv::Mat(900, 1600, CV_8UC3, col::white);
    text(cv_, "[5] Real photo vs model  |  " + e.stem + "   (reference: " + out.ref + ")", cv::Point(20, 26), 0.6);

    // QR 띠 : 모델이 예측한 플레이트 이동만큼 따라가며 잘라 냄
    const cv::Point2d dp = cam_.displacement(cam_.plate_center, cam_.z_plate, e.pose);
    const cv::Rect band(int(cam_.plate_center.x - 420 + dp.x), int(cam_.plate_center.y - 150 + dp.y), 840, 300);
    if (inside(band, cur.size())) {
        // 모델 합성 : 띠의 화소마다 플레이트 깊이 평면으로 역사상 → 기준 영상에서 가져옴 (왜곡 포함, k1 = 0 이면 호모그래피와 같음)
        cv::Mat mapx(band.size(), CV_32F), mapy(band.size(), CV_32F), sim;
        for (int yy = 0; yy < band.height; ++yy)
            for (int xx = 0; xx < band.width; ++xx) {
                const cv::Point2d r0 = cam_.mapToReference(cv::Point2d(band.x + xx, band.y + yy), cam_.z_plate, e.pose);
                mapx.at<float>(yy, xx) = float(r0.x);
                mapy.at<float>(yy, xx) = float(r0.y);
            }
        cv::remap(ref, sim, mapx, mapy, cv::INTER_LINEAR);
        cv::Mat real = cur(band), diff = real - sim, diff64;
        diff.convertTo(diff64, CV_64F);
        auto show = [&](const cv::Mat& g, const std::string& t, int y) {
            cv::Mat u8, bgr, small;
            g.convertTo(u8, CV_8U);
            cv::cvtColor(u8, bgr, cv::COLOR_GRAY2BGR);
            cv::resize(bgr, small, cv::Size(714, 255), 0, 0, cv::INTER_AREA);
            small.copyTo(cv_(cv::Rect(20, y, 714, 255)));
            text(cv_, t, cv::Point(24, y - 6), 0.5);
        };
        show(real, "real photo (QR band)", 56);
        show(sim, "model: reference re-projected through the motion model", 336);
        cv::Mat d = signedColorMap(diff64, 40.0), ds;
        cv::resize(d, ds, cv::Size(714, 255), 0, 0, cv::INTER_AREA);
        ds.copyTo(cv_(cv::Rect(20, 616, 714, 255)));
        text(cv_, "real - model (gray levels, +-40)  : white = model matches", cv::Point(24, 610), 0.5);
    }

    // 오른쪽 : 영역별 이동량
    int y = 70; const int x = 770;
    auto line = [&](const std::string& t, const cv::Scalar& c = col::black) { text(cv_, t, cv::Point(x, y), 0.5, c); y += 23; };
    line(fmt("label : Y = %+.3f mm,  Theta = %+.3f deg", e.pose.y_mm, e.pose.t_deg), col::blue);
    line(fmt("coarse plate shift (phase corr.) : %+.2f, %+.2f px", out.coarse.x, out.coarse.y));
    y += 6;
    line("region       group        meas dx   model dx   resid   meas dy  model dy", col::blue);
    const auto& R = defaultRegions();
    for (size_t i = 0; i < R.size(); ++i) {
        const auto& rr = out.r[i];
        char b[200];
        if (rr.ok)
            std::snprintf(b, sizeof(b), "%-11s  %-11s %+8.3f  %+8.3f  %+6.3f  %+7.3f  %+7.3f", R[i].name.c_str(), kGroupName[R[i].group],
                          rr.meas.x, rr.pred.x, rr.meas.x - rr.pred.x, rr.meas.y, rr.pred.y);
        else
            std::snprintf(b, sizeof(b), "%-11s  %-11s   (not measured)", R[i].name.c_str(), kGroupName[R[i].group]);
        line(b, std::abs(rr.meas.x - rr.pred.x) > 0.5 && rr.ok ? col::red : col::black);
    }
    y += 10;
    line("How to read", col::purple);
    line("  meas  : shift measured in the photo (phase corr. + ECC)");
    line("  model : shift predicted by the motion model at the");
    line("          region depth (plate / upper wall / lower wall)");
    line("  resid : meas - model (px).  1 px = about 70 um on the plate");
    line("  plate vs wall difference under Y = depth parallax");
    line("  diff image: QR edges appear where the model is off");
    y += 6;
    line("Keys: n / p next / previous photo, s save, 1-4 model views", col::gray);
    return true;
}

// ---------------------------------------------------------------------------------- 일괄
namespace {

struct Fit { double c = 0, a = 0, b = 0, q = 0, sd = 0; int n = 0; };

// meas = c + a·Y + b·Θ + q·Θ²  (1차 세트)
Fit fitRegion(const std::vector<CompareResult>& res, size_t k) {
    std::vector<double> A, yv;
    for (const auto& r : res) {
        if (r.e.set_id != 1 || !r.r[k].ok) continue;
        const double Y = r.e.pose.y_mm, T = r.e.pose.t_deg;
        A.insert(A.end(), {1.0, Y, T, T * T});
        yv.push_back(r.r[k].meas.x);
    }
    Fit f;
    f.n = int(yv.size());
    if (f.n < 6) return f;
    const cv::Mat M(f.n, 4, CV_64F, A.data()), Y(f.n, 1, CV_64F, yv.data());
    cv::Mat beta;
    cv::solve(M, Y, beta, cv::DECOMP_SVD);
    f.c = beta.at<double>(0); f.a = beta.at<double>(1); f.b = beta.at<double>(2); f.q = beta.at<double>(3);
    const cv::Mat e = Y - M * beta;
    f.sd = std::sqrt(e.dot(e) / std::max(1, f.n - 4));
    return f;
}

struct ResidStat { double rms = 0, mean = 0; int n = 0; };
ResidStat residStat(const std::vector<CompareResult>& res, const CameraModel& cam, int set_id, int group) {
    ResidStat s;
    const auto& R = defaultRegions();
    for (const auto& r : res) {
        if (r.e.set_id != set_id || r.e.is_ref) continue;
        for (size_t k = 0; k < R.size(); ++k) {
            if (R[k].group != group || !r.r[k].ok) continue;
            const double z = group == 0 ? cam.z_plate : (group == 1 ? cam.z_wall_upper : cam.z_bg);
            const double e = r.r[k].meas.x - cam.displacement(R[k].c, z, r.e.pose).x;
            s.rms += e * e; s.mean += e; ++s.n;
        }
    }
    if (s.n) { s.rms = std::sqrt(s.rms / s.n); s.mean /= s.n; }
    return s;
}

} // namespace

void runBatch(const CameraModel& cam, const std::vector<ImageEntry>& entries, const std::string& imgDir, const std::string& outDir) {
    fs::create_directories(outDir);
    RealComparer cmp(cam, imgDir);
    std::vector<CompareResult> res;
    const auto& R = defaultRegions();
    int done = 0;
    for (const auto& e : entries) {
        if (e.stem == "center" || e.stem == "center1") continue;  // 기준 자체는 제외
        CompareResult r;
        if (cmp.compare(e, r)) res.push_back(r);
        if (++done % 25 == 0) std::cout << "  " << done << " / " << entries.size() << " 장 처리\n";
    }

    // ---- CSV
    std::ofstream csv(outDir + "/model_vs_real.csv");
    csv << "\xEF\xBB\xBF" << "file,set,ref,label_y_mm,label_t_deg,coarse_dx";
    for (const auto& reg : R) csv << "," << reg.name << "_meas_dx," << reg.name << "_model_dx," << reg.name << "_resid_dx," << reg.name << "_ecc";
    csv << "\n";
    for (const auto& r : res) {
        csv << r.e.stem << "," << r.e.set_id << "," << r.ref << "," << r.e.pose.y_mm << "," << r.e.pose.t_deg << "," << r.coarse.x;
        for (const auto& rr : r.r) {
            if (rr.ok) csv << "," << rr.meas.x << "," << rr.pred.x << "," << rr.meas.x - rr.pred.x << "," << rr.ecc;
            else csv << ",,,,";
        }
        csv << "\n";
    }

    // ---- 1차 세트로 영역별 감도 회귀 → 실효 모델
    std::ostringstream os;
    os << "운동 모델 vs 실제 사진 (" << res.size() << " 장)\n\n";
    os << "[1] 영역별 실측 감도 (1차 세트 회귀 : dx = c + a·Y + b·Θ + q·Θ²)\n";
    os << "  영역        그룹          a (px/mm)  b (px/°)   q (px/°²)  잔차σ (px)   장수\n";
    std::vector<Fit> fits;
    for (size_t k = 0; k < R.size(); ++k) {
        const Fit f = fitRegion(res, k);
        fits.push_back(f);
        char b[220];
        std::snprintf(b, sizeof(b), "  %-10s  %-12s %9.3f  %9.3f  %9.3f  %9.3f   %4d\n", R[k].name.c_str(), kGroupName[R[k].group],
                      f.a, f.b, f.q, f.sd, f.n);
        os << b;
    }
    // 영역 k 의 Θ 감도 : b_k ≈ c − m·a_k + h·x_k²
    //   c = fx·π/180 (순수 회전), m = c·axis/fx (축 어긋남), h = c·(1 + 3·k1) (화면 가장자리 2차 항 + 왜곡)
    //   x_k = 영역의 정규화 가로 위치. fx 를 모르므로 반복해 맞춤
    //   freeK1 = false : k1 = 0 고정 (h = c),  true : k1 도 맞춤
    //   주의 : 배경 벽은 깊이도 멀고 화면 가장자리에도 있어 m 과 h(k1) 가 서로 얽힘 → 매개변수 값은 유일하지 않음
    int used = 0;
    for (const auto& f : fits) used += f.n >= 6;
    auto fitModel = [&](bool freeK1, CameraModel& fitted) {
        fitted = cam;
        fitted.k1 = 0.0;
        double c = 0, m = 0, h = 0;
        for (int it = 0; it < 4; ++it) {
            std::vector<double> A, yv;
            for (size_t k = 0; k < R.size(); ++k) {
                if (fits[k].n < 6) continue;
                const double xk = (R[k].c.x - fitted.cx) / fitted.fx;
                if (freeK1) A.insert(A.end(), {1.0, -fits[k].a, xk * xk});
                else A.insert(A.end(), {1.0 + xk * xk, -fits[k].a});
                yv.push_back(fits[k].b);
            }
            const int nc = freeK1 ? 3 : 2;
            const cv::Mat M(int(yv.size()), nc, CV_64F, A.data()), Y(int(yv.size()), 1, CV_64F, yv.data());
            cv::Mat beta;
            cv::solve(M, Y, beta, cv::DECOMP_SVD);
            c = beta.at<double>(0); m = beta.at<double>(1); h = freeK1 ? beta.at<double>(2) : c;
            fitted.fx = c * 180.0 / CV_PI;
            fitted.fy = fitted.fx * 16.0 / 15.0;
            fitted.k1 = freeK1 ? (h / c - 1.0) / 3.0 : 0.0;
            fitted.axis_a = m * fitted.fx / c;
        }
        // 깊이 : a_k = fx·(1 + 3·k1·x_k²) / Z_k  → 그룹 평균
        auto meanZ = [&](int g) {
            double sz = 0; int n = 0;
            for (size_t k = 0; k < R.size(); ++k) {
                if (R[k].group != g || fits[k].n < 6) continue;
                const double xk = (R[k].c.x - fitted.cx) / fitted.fx;
                sz += fitted.fx * (1.0 + 3.0 * fitted.k1 * xk * xk) / fits[k].a; ++n;
            }
            return n ? sz / n : 0.0;
        };
        if (meanZ(0) > 0) fitted.z_plate = meanZ(0);
        if (meanZ(1) > 0) fitted.z_wall_upper = meanZ(1);
        if (meanZ(2) > 0) fitted.z_bg = meanZ(2);
    };
    CameraModel fitA = cam, fitB = cam;
    if (used >= 4) {
        fitModel(false, fitA);
        fitModel(true, fitB);
        os << "\n[2] 영역 감도로 다시 맞춘 실효 모델 (b_k = c − m·a_k + h·x_k²,  영역 " << used << " 개)\n";
        char b[900];
        std::snprintf(b, sizeof(b),
                      "                 제원 기반(현재)   맞춤 A (k1 = 0)   맞춤 B (k1 포함)\n"
                      "  fx (px)        %10.1f       %10.1f       %10.1f\n"
                      "  왜곡 k1        %10.4f       %10.4f       %10.4f\n"
                      "  플레이트 Z (mm) %10.1f       %10.1f       %10.1f\n"
                      "  위쪽 벽 Z (mm)  %10.1f       %10.1f       %10.1f\n"
                      "  아래 벽 Z (mm)  %10.1f       %10.1f       %10.1f\n"
                      "  회전축 a (mm)   %10.1f       %10.1f       %10.1f   (광학 중심 → 플레이트 쪽 +)\n",
                      cam.fx, fitA.fx, fitB.fx, cam.k1, fitA.k1, fitB.k1, cam.z_plate, fitA.z_plate, fitB.z_plate,
                      cam.z_wall_upper, fitA.z_wall_upper, fitB.z_wall_upper, cam.z_bg, fitA.z_bg, fitB.z_bg, cam.axis_a,
                      fitA.axis_a, fitB.axis_a);
        os << b;
        os << "  ※ 맞춤 B 는 잔차가 가장 작지만, 벽 영역이 '깊이가 멀다'와 '화면 가장자리'를 함께 가져 m·k1·fx 가 서로 얽힘\n"
              "    → 예측용(실효) 모델로만 쓰고, 개별 매개변수를 물리값으로 해석하지 말 것\n";
        fitA.save(outDir + "/camera_model_fitted_k1zero.yml");
        fitB.save(outDir + "/camera_model_fitted.yml");
    }

    // ---- 잔차 비교 (세 모델)
    os << "\n[3] 모델 예측 vs 실측 잔차 (dx, px) — 세트·영역 그룹별 RMS (평균)\n";
    os << "  세트  그룹          제원 기반           맞춤 A (k1 = 0)      맞춤 B (k1 포함)\n";
    for (int set = 1; set <= 2; ++set)
        for (int g = 0; g < 3; ++g) {
            const ResidStat a0 = residStat(res, cam, set, g), a1 = residStat(res, fitA, set, g), a2 = residStat(res, fitB, set, g);
            if (!a0.n) continue;
            char b[260];
            std::snprintf(b, sizeof(b), "  %d     %-12s  %6.3f (%+6.3f)    %6.3f (%+6.3f)    %6.3f (%+6.3f)   n=%d\n", set, kGroupName[g],
                          a0.rms, a0.mean, a1.rms, a1.mean, a2.rms, a2.mean, a0.n);
            os << b;
        }
    // 모델의 Θ² 항 (플레이트 중심) — 실측 q 와 비교
    auto plateX = [&](const CameraModel& c, double t) { return c.displacement(c.plate_center, c.z_plate, Pose{0, t}).x; };
    const double h = 0.5;
    os << "\n[4] 플레이트 Θ² 항 (px/°²) : 실측 평균 "
       << (fits[0].q + fits[1].q + fits[2].q) / 3.0 << ",  제원 모델 " << (plateX(cam, h) + plateX(cam, -h)) / (2 * h * h)
       << ",  맞춤 B " << (plateX(fitB, h) + plateX(fitB, -h)) / (2 * h * h) << "\n";
    os << "\n파일 : model_vs_real.csv (사진·영역별 측정·예측·잔차), camera_model_fitted.yml (맞춤 B), camera_model_fitted_k1zero.yml (맞춤 A) — config/camera_model.yml 로 복사하면 그 모델로 화면 표시\n";

    std::ofstream(outDir + "/model_vs_real_summary.txt") << "\xEF\xBB\xBF" << os.str();
    std::cout << "\n" << os.str();
}

} // namespace mm
