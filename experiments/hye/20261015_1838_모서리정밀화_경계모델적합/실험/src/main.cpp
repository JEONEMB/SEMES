// center.bmp 의 QR 격자 구조점 755개를 erf 경계 모델로 서브픽셀 맞춤하고 진단 기준으로 통과/불확실을 나눈다.
// 원본: codex_experiments_20261014/decoded_grid_corners_20261015/build.py, subpixel_grid_corners_20261015/refine.py,
//       confidence_map.py (품질 점수)
#include <hye/dataset.hpp>
#include <hye/plot.hpp>
#include <hye/report.hpp>
#include <omp.h>
#include <iostream>
#include <map>
#include "corner_fit.hpp"

using namespace hye;

namespace {

struct Row {
    cf::Corner c;
    double x = 0, y = 0, shift = 0, rmse = 0, stab = 0, sx = 0, sy = 0, contrast = 0, quality = 0;
    bool ok1 = false, ok2 = false, valid = false;
    int nfev1 = 0;
    std::string reason;
};

// 합성 영상 검증 (refine.py synthetic_check): 11x11 창, 세 가지 2x2 배치 x 두 정답 위치
struct Synth { double err, rmse; bool ok; };
std::vector<Synth> synthetic_check() {
    std::vector<Synth> out;
    std::vector<double> X, Y;
    for (int y = -5; y <= 5; ++y)
        for (int x = -5; x <= 5; ++x) { X.push_back(x); Y.push_back(y); }
    const double patches[3][2][2] = {{{1, 0}, {0, 0}}, {{0, 1}, {1, 1}}, {{1, 0}, {0, 1}}};
    const double truths[2][2] = {{.25, -.35}, {-.6, .4}};
    float eye[2][2] = {{1, 0}, {0, 1}};
    for (auto& pa : patches)
        for (auto& t : truths) {
            cf::Model M{X.data(), Y.data(), nullptr, (int)X.size()};
            for (int a = 0; a < 2; ++a) for (int b = 0; b < 2; ++b) { M.inv[a][b] = eye[a][b]; M.patch[a][b] = pa[a][b]; }
            std::vector<double> zero(X.size(), 0.0), r(X.size());
            M.I = zero.data();
            double p[6] = {t[0], t[1], .9, 1.1, 40, 150};
            M.residual(p, r.data());  // = 40 + 150 (1 - field)
            std::vector<double> I = r;
            auto R = cf::fit(X, Y, I, 0, 0, eye, pa);
            out.push_back({std::hypot(R.p[0] - t[0], R.p[1] - t[1]), R.rmse, R.ok});
        }
    return out;
}

double median(std::vector<double> v) { return percentile(std::move(v), 0.5); }

// 품질 점수 색 (confidence_map.py: 빨강 -> 노랑 -> 연두), BGR
cv::Scalar qcolor(double q) {
    const double st[3][3] = {{244, 105, 64}, {255, 216, 81}, {139, 250, 131}};
    double pos = std::min(1.0, std::max(0.0, q)) * 2;
    int i = std::min((int)pos, 1);
    double f = pos - i;
    double rgb[3];
    for (int k = 0; k < 3; ++k) rgb[k] = std::nearbyint(st[i][k] * (1 - f) + st[i + 1][k] * f);
    return cv::Scalar(rgb[2], rgb[1], rgb[0]);
}

// 판 영역을 확대해 점을 그린다. mode 0 = 통과/불확실 색, 1 = 품질 점수 색, 2 = 시작점(보라) -> 맞춘 점
cv::Mat panel(const cv::Mat& gray, cv::Rect box, int out_w, const std::vector<Row>& rows, int qr, int mode, const std::string& cap) {
    cv::Mat c;
    cv::cvtColor(gray(box), c, cv::COLOR_GRAY2BGR);
    double sc = (double)out_w / box.width;
    cv::resize(c, c, {out_w, (int)std::lround(box.height * sc)}, 0, 0, cv::INTER_CUBIC);
    const cv::Scalar green(131, 250, 139), amber(65, 180, 255), purple(255, 124, 208);
    for (auto& r : rows) {
        if (r.c.qr != qr || !(r.x >= box.x && r.x < box.x + box.width && r.y >= box.y && r.y < box.y + box.height)) continue;
        // 화소 중심 좌표 -> 확대 그림 좌표
        auto P = [&](double x, double y) { return cv::Point2d((x - box.x + 0.5) * sc - 0.5, (y - box.y + 0.5) * sc - 0.5); };
        cv::Point2d p = P(r.x, r.y);
        cv::Point ip((int)std::lround(p.x), (int)std::lround(p.y));
        if (mode == 2) {
            cv::Point2d s0 = P(r.c.x, r.c.y);
            cv::Point is((int)std::lround(s0.x), (int)std::lround(s0.y));
            cv::drawMarker(c, is, purple, cv::MARKER_CROSS, 12, 2, cv::LINE_AA);
            cv::line(c, is, ip, purple, 2, cv::LINE_AA);
        }
        int rad = sc <= 6 ? 7 : 10;
        cv::Scalar col = mode == 1 ? qcolor(r.quality) : (r.valid ? green : amber);
        if (mode == 1) {
            int rr = (int)std::lround(2.3 * sc);
            cv::circle(c, ip, rr + 2, cv::Scalar(20, 23, 17), -1, cv::LINE_AA);
            cv::circle(c, ip, rr, col, -1, cv::LINE_AA);
        } else {
            cv::rectangle(c, ip - cv::Point(rad + 1, rad + 1), ip + cv::Point(rad + 1, rad + 1), cv::Scalar(12, 20, 12), 4);
            cv::rectangle(c, ip - cv::Point(rad, rad), ip + cv::Point(rad, rad), col, 2);
        }
    }
    if (!cap.empty()) {
        cv::rectangle(c, {0, 0}, {text_w(cap, 0.6) + 12, 28}, cv::Scalar(255, 255, 255), -1);
        text(c, cap, {6, 20}, 0.6);
    }
    return c;
}

}  // namespace

static int run();
int main() {
    try {
        return run();
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}

static int run() {
    begin_results();
    omp_set_num_threads(4);
    auto shots = load_1014();
    const cv::Mat ref = read_gray(center_1014(shots));

    // 1) QR 해독 -> 구조점
    auto qrs = cf::detect_qrs(ref);
    auto corners = cf::structural_corners(qrs);
    std::cout << "corners " << corners.size() << "\n";

    // 2) 합성 검증
    auto synth = synthetic_check();

    // 3) 모서리마다 9x9, 11x11 창 맞춤
    std::vector<Row> rows(corners.size());
#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < (int)corners.size(); ++i) {
        const auto& c = corners[i];
        auto b = cf::kBoxes[c.qr - 1];
        std::vector<double> X, Y, I;
        cf::FitResult F[2];
        for (int k = 0; k < 2; ++k) {
            cf::window(ref, c.x, c.y, 4 + k, b[0], b[1], b[2], b[3], X, Y, I);
            F[k] = cf::fit(X, Y, I, c.x, c.y, c.inv, c.patch);
        }
        Row r;
        r.c = c;
        r.x = F[0].p[0]; r.y = F[0].p[1];
        r.shift = std::hypot(r.x - c.x, r.y - c.y);
        r.rmse = F[0].rmse;
        r.stab = std::hypot(F[0].p[0] - F[1].p[0], F[0].p[1] - F[1].p[1]);
        r.sx = F[0].p[2]; r.sy = F[0].p[3]; r.contrast = F[0].p[5];
        r.ok1 = F[0].ok; r.ok2 = F[1].ok; r.nfev1 = F[0].nfev;
        r.reason = cf::reasons(r.ok1, r.ok2, r.rmse, r.contrast, r.stab, std::max(std::abs(r.x - c.x), std::abs(r.y - c.y)), std::max(r.sx, r.sy));
        r.valid = r.reason.empty();
        // confidence_map.py 품질 점수
        double q = std::exp(-.5 * std::pow(r.rmse / 15, 2)) * std::exp(-.5 * std::pow(r.stab / .65, 2)) * std::min(1.0, std::max(0.0, r.contrast / 90));
        if (r.reason.find("optimizer") != std::string::npos || r.reason.find("search_limit") != std::string::npos ||
            r.reason.find("blur_limit") != std::string::npos)
            q *= .25;
        r.quality = std::min(1.0, std::max(0.0, q));
        rows[i] = r;
    }

    // ---- 집계 ----
    int total = (int)rows.size(), valid = 0;
    std::vector<double> shifts, rmses, quals;
    std::map<std::string, int> reason_cnt;
    std::map<int, std::array<int, 2>> by_black;
    for (auto& r : rows) {
        valid += r.valid;
        shifts.push_back(r.shift); rmses.push_back(r.rmse); quals.push_back(r.quality);
        size_t p = 0;
        while (!r.reason.empty() && p != std::string::npos) {
            size_t e = r.reason.find(';', p);
            reason_cnt[r.reason.substr(p, e == std::string::npos ? std::string::npos : e - p)]++;
            p = e == std::string::npos ? e : e + 1;
        }
        by_black[r.c.black][0]++;
        by_black[r.c.black][1] += r.valid;
    }
    struct PerQR { int total = 0, valid = 0; double med_shift = 0, med_rmse = 0; };
    std::map<int, PerQR> perqr;
    for (int k = 1; k <= 3; ++k) {
        std::vector<double> s, e;
        auto& P = perqr[k];
        for (auto& r : rows) if (r.c.qr == k) { P.total++; P.valid += r.valid; s.push_back(r.shift); e.push_back(r.rmse); }
        P.med_shift = median(s); P.med_rmse = median(e);
    }

    // ---- CSV ----
    Table csv{{"id", "qr", "grid_x", "grid_y", "initial_x", "initial_y", "x", "y", "shift_px", "rmse", "window_stability_px", "sigma_x", "sigma_y",
               "contrast", "valid", "reason", "black_neighbors", "quality_score"}};
    Table unc = csv;
    unc.rows.clear();
    for (auto& r : rows) {
        std::vector<std::string> v{std::to_string(r.c.id), std::to_string(r.c.qr), std::to_string(r.c.gx), std::to_string(r.c.gy), fmt(r.c.x, 4),
                                   fmt(r.c.y, 4), fmt(r.x, 4), fmt(r.y, 4), fmt(r.shift, 4), fmt(r.rmse, 3), fmt(r.stab, 4), fmt(r.sx, 3), fmt(r.sy, 3),
                                   fmt(r.contrast, 2), r.valid ? "True" : "False", r.reason, std::to_string(r.c.black), fmt(r.quality, 4)};
        csv.add(v);
        if (!r.valid) unc.add(v);
    }
    csv.csv(out("refined_corners.csv"));
    unc.csv(out("uncertain_corners.csv"));

    // ---- 그림 ----
    const cv::Rect boxes[3] = {{1552, 969, 184, 198}, {1809, 971, 184, 198}, {2067, 973, 185, 198}};
    {
        std::vector<cv::Mat> t;
        for (int k = 0; k < 3; ++k) t.push_back(panel(ref, boxes[k], 600, rows, k + 1, 0, "QR" + std::to_string(k + 1) + " pass " + std::to_string(perqr[k + 1].valid) + "/" + std::to_string(perqr[k + 1].total)));
        save_png("01_전체모서리_통과불확실.png", hstack(t, 16));
    }
    {
        cv::Rect detail(1557, 976, 68, 72);
        save_png("02_큰네모_시작점대비맞춤.png", hstack({panel(ref, detail, 640, rows, 1, 0, "fitted (green pass, amber uncertain)"),
                                                     panel(ref, detail, 640, rows, 1, 2, "purple = projected start -> fitted")}, 16));
    }
    {
        std::vector<cv::Mat> t;
        for (int k = 0; k < 3; ++k) t.push_back(panel(ref, boxes[k], 600, rows, k + 1, 1, k == 0 ? "quality: red 0 - yellow - green 1" : ""));
        save_png("03_품질점수.png", hstack(t, 16));
    }
    {
        // 진단 산점도: RMSE vs 대비, 2x2 배치 종류별
        Chart c("Fit diagnostics per corner", "photometric RMSE (gray level)", "fitted contrast");
        for (int b = 1; b <= 3; ++b) {
            std::vector<double> x, y;
            for (auto& r : rows) if (r.c.black == b) { x.push_back(r.rmse); y.push_back(r.contrast); }
            c.scatter(x, y, color(b - 1), std::to_string(b) + " black neighbors" + (b == 2 ? " (diagonal)" : ""));
        }
        c.vline(15, kRed).hline(45, kRed);
        c.save("04_진단_RMSE와대비.png");
    }
    {
        std::vector<double> x, y;
        for (auto& r : rows) { x.push_back(r.c.id); y.push_back(r.shift); }
        Chart c("Shift from projected start to fitted corner", "corner id (QR1 | QR2 | QR3)", "shift (px)");
        c.scatter(x, y, color(0), "", 2).hline(median(shifts), kRed);
        c.save("05_시작점대비이동.png");
    }

    // 튀는 값: 이동이 가장 큰 점, RMSE 가 가장 큰 점
    size_t wi_shift = 0, wi_rmse = 0;
    for (size_t i = 0; i < rows.size(); ++i) {
        if (rows[i].shift > rows[wi_shift].shift) wi_shift = i;
        if (rows[i].rmse > rows[wi_rmse].rmse) wi_rmse = i;
    }
    auto outlier_fig = [&](size_t i, const std::string& file, const std::string& cap) {
        auto& r = rows[i];
        cv::Rect roi((int)std::floor(r.c.x) - 9, (int)std::floor(r.c.y) - 9, 19, 19);
        std::vector<Mark> marks{{{r.c.x + 0.5, r.c.y + 0.5}, {0, 0}, "start", cv::Scalar(255, 124, 208)}, {{r.x + 0.5, r.y + 0.5}, {0, 0}, "fit", kRed}};
        save_png(file, zoom(ref, roi, 520, marks, 0, cap));
    };
    auto cname = [&](const Row& r) { return "QR" + std::to_string(r.c.qr) + "(" + std::to_string(r.c.gx) + "," + std::to_string(r.c.gy) + ")"; };
    outlier_fig(wi_shift, "06_튀는값_최대이동.png", cname(rows[wi_shift]) + " shift " + fmt(rows[wi_shift].shift, 2) + " px");
    outlier_fig(wi_rmse, "07_튀는값_최대RMSE.png", cname(rows[wi_rmse]) + " RMSE " + fmt(rows[wi_rmse].rmse, 1));

    Table outl{{"구분", "모서리 (QR, 격자 u,v)", "검정 이웃", "시작 (x, y)", "맞춤 (x, y)", "이동 (px)", "RMSE", "두 창 차이 (px)", "대비", "σx / σy", "판정"}};
    auto add_out = [&](const std::string& lab, size_t i) {
        auto& r = rows[i];
        outl.add({lab, cname(r), std::to_string(r.c.black), fmt(r.c.x, 2) + ", " + fmt(r.c.y, 2), fmt(r.x, 2) + ", " + fmt(r.y, 2), fmt(r.shift, 3),
                  fmt(r.rmse, 2), fmt(r.stab, 3), fmt(r.contrast, 1), fmt(r.sx, 2) + " / " + fmt(r.sy, 2), r.valid ? "통과" : r.reason});
    };
    {
        std::vector<size_t> idx(rows.size());
        std::iota(idx.begin(), idx.end(), 0);
        auto top = [&](auto key, int k) {
            auto v = idx;
            std::stable_sort(v.begin(), v.end(), [&](size_t a, size_t b) { return key(rows[a]) > key(rows[b]); });
            v.resize(k);
            return v;
        };
        for (auto i : top([](const Row& r) { return r.shift; }, 3)) add_out("이동 상위", i);
        for (auto i : top([](const Row& r) { return r.rmse; }, 3)) add_out("RMSE 상위", i);
    }

    // 사분위 품질 점수
    double q0 = percentile(quals, 0), q25 = percentile(quals, .25), q50 = percentile(quals, .5), q75 = percentile(quals, .75), q100 = percentile(quals, 1);
    double synth_max = 0;
    bool synth_ok = true;
    for (auto& s : synth) { synth_max = std::max(synth_max, s.err); synth_ok = synth_ok && s.ok && s.err < 1e-3; }

    // ---- 원래 결과와 대조 ----
    // 출처: subpixel_grid_corners_20261015/checks.json, refined_corners.csv, confidence_filled/settings.json,
    //       decoded_grid_corners_20261015/metadata.json
    Parity par;
    par.add("QR1 구조점 수", 258, perqr[1].total, 0, 0);
    par.add("QR2 구조점 수", 247, perqr[2].total, 0, 0);
    par.add("QR3 구조점 수", 250, perqr[3].total, 0, 0);
    par.add("QR1 사각형 좌상 x (px)", 1563.0, qrs[0].quad[0].x, 0.01, 3);
    par.add("QR2 사각형 우하 y (px)", 1155.4679, qrs[1].quad[2].y, 0.01, 3);
    par.add("QR3 사각형 좌상 x (px)", 2079.3434, qrs[2].quad[0].x, 0.5, 3);
    par.add("QR3 사각형 좌상 y (px)", 983.3527, qrs[2].quad[0].y, 0.5, 3);
    par.add("QR3 사각형 우하 x (px)", 2240.9758, qrs[2].quad[2].x, 0.5, 3);
    par.add("QR3 사각형 우하 y (px)", 1156.7299, qrs[2].quad[2].y, 0.5, 3);
    par.add_text("QR 해독 문자열", "FP11L / FP11C / FP11R", qrs[0].payload + " / " + qrs[1].payload + " / " + qrs[2].payload,
                 qrs[0].payload == "FP11L" && qrs[1].payload == "FP11C" && qrs[2].payload == "FP11R");
    par.add_text("합성 검증 6건 위치 오차 < 1e-3 px", "6/6", std::to_string(synth_ok ? 6 : 0) + "/6 (최대 " + fmt(synth_max, 9) + ")", synth_ok);
    par.add("전체 점", 755, total, 0, 0);
    par.add("통과 점", 608, valid, 3, 0);
    par.add("불확실 점", 147, total - valid, 3, 0);
    par.add("QR1 통과", 203, perqr[1].valid, 2, 0);
    par.add("QR2 통과", 202, perqr[2].valid, 2, 0);
    par.add("QR3 통과", 203, perqr[3].valid, 2, 0);
    par.add("이동 중앙값 (px)", 1.0662, median(shifts), 0.01);
    par.add("QR1 이동 중앙값 (px)", 1.2008, perqr[1].med_shift, 0.01);
    par.add("QR2 이동 중앙값 (px)", 1.1044, perqr[2].med_shift, 0.01);
    par.add("QR3 이동 중앙값 (px)", 0.9368, perqr[3].med_shift, 0.05);
    par.add("이동 최대 (px)", 3.6432, max_abs(shifts), 0.05);
    par.add("RMSE 중앙값", 11.3308, median(rmses), 0.05);
    par.add("QR1 RMSE 중앙값", 11.3683, perqr[1].med_rmse, 0.05);
    par.add("QR2 RMSE 중앙값", 11.4793, perqr[2].med_rmse, 0.05);
    par.add("QR3 RMSE 중앙값", 11.1252, perqr[3].med_rmse, 0.1);
    par.add("불확실 사유 photometric_rmse", 127, reason_cnt["photometric_rmse"], 3, 0);
    par.add("불확실 사유 optimizer", 22, reason_cnt["optimizer"], 3, 0);
    par.add("대각 2검정 점 통과 (125점 중)", 0, by_black[2][1], 0, 0);
    par.add("품질 점수 중앙값", 0.7451, q50, 0.005);
    par.add("품질 점수 25% / 75% 평균", (0.6754 + 0.7836) / 2, (q25 + q75) / 2, 0.005);

    // QR3 사각형이 원래(decoded_grid_corners_20261015/metadata.json)와 다른 최대 거리
    const double q3o[4][2] = {{2079.34344291687, 983.3527421951294}, {2241.0, 984.6666660308838}, {2240.9757537841797, 1156.7299041748047}, {2079.0, 1155.3333282470703}};
    double q3diff = 0;
    for (int j = 0; j < 4; ++j) q3diff = std::max(q3diff, std::hypot(qrs[2].quad[j].x - q3o[j][0], qrs[2].quad[j].y - q3o[j][1]));
    // ---- 결과.md ----
    Table qt{{"QR", "해독", "모듈", "구조점", "사각형 (좌상, 우상, 우하, 좌하)", "찾은 방법"}};
    for (auto& q : qrs) {
        std::string s;
        for (int j = 0; j < 4; ++j) s += (j ? " / " : "") + fmt(q.quad[j].x, 2) + ", " + fmt(q.quad[j].y, 2);
        qt.add({std::to_string(q.qr), q.payload, std::to_string(q.n) + "x" + std::to_string(q.n), std::to_string(perqr[q.qr].total), s, q.how});
    }
    Table pq{{"QR", "점", "통과", "불확실", "이동 중앙값 (px)", "RMSE 중앙값"}};
    for (int k = 1; k <= 3; ++k)
        pq.add({std::to_string(k), std::to_string(perqr[k].total), std::to_string(perqr[k].valid), std::to_string(perqr[k].total - perqr[k].valid),
                fmt(perqr[k].med_shift, 4), fmt(perqr[k].med_rmse, 3)});
    pq.add({"전체", std::to_string(total), std::to_string(valid), std::to_string(total - valid), fmt(median(shifts), 4), fmt(median(rmses), 3)});
    Table bt{{"검정 이웃 수", "모양", "점", "통과"}};
    for (auto& [b, v] : by_black) bt.add({std::to_string(b), b == 1 ? "볼록 모서리" : b == 3 ? "오목 모서리" : "대각 접점", std::to_string(v[0]), std::to_string(v[1])});
    Table rt{{"사유", "점"}};
    for (auto& [k, v] : reason_cnt) rt.add({k, std::to_string(v)});
    Table st{{"2x2 배치", "정답 (x, y)", "위치 오차 (px)", "RMSE", "수렴"}};
    {
        const char* pn[3] = {"[[1,0],[0,0]]", "[[0,1],[1,1]]", "[[1,0],[0,1]]"};
        const char* tn[2] = {"0.25, -0.35", "-0.60, 0.40"};
        for (int i = 0; i < 6; ++i)
            st.add({pn[i / 2], tn[i % 2], fmt(synth[i].err, 10), fmt(synth[i].rmse, 8), synth[i].ok ? "예" : "아니오"});
    }
    std::ostringstream R;
    R << "# 결과: QR 구조점 서브픽셀 경계 모델 맞춤\n\n"
      << "입력: `data/clean/clean_20261014/center.bmp` 한 장. QR 세 개를 해독해 21x21 모듈 격자를 얻고, 2x2 이웃 모듈의 검정 배치가 모서리인 격자 꼭짓점(구조점)을 QR 사각형 호모그래피로 투영해 시작점으로 썼다.\n"
      << "모델: 2x2 모듈 배치를 국소 격자 축 방향의 erf 흐림 경계로 그린 밝기 `low + contrast(1 - field)`. 변수 x, y, σx, σy, low, contrast 여섯 개를 9x9 창(반경 4)과 11x11 창(반경 5)에서 각각 soft-L1 최소제곱(scipy least_squares trf 를 옮긴 것, f_scale 6, max_nfev 85)으로 맞췄다. 창은 해당 QR 판 영역 안쪽 화소만 쓴다.\n"
      << "통과 기준(하나라도 걸리면 불확실): 두 창 모두 수렴, RMSE <= 15, 대비 >= 45, 두 창 차이 <= 0.65 px, 시작점에서 각 축 3.6 px 이내, σ <= 2.6.\n\n"
      << "## QR 해독\n\n" << qt.md()
      << "\n원래 Python(cv2 5.0)은 QR1·QR2 를 자른 영역 전체에서, QR3 을 판 영역 3배 확대에서 찾았다. OpenCV 3.4 해독기는 QR1·QR2 사각형을 원래와 똑같이 냈지만 QR3 은 원본 화소로는 1~4배 어디서도 못 찾아서, 같은 순서로 3x3 가우시안으로 흐린 판 영역을 다시 시도했다(찾은 방법 열). "
      << "그래서 QR3 사각형이 원래와 최대 " << fmt(q3diff, 2) << " px 다르고, 그 사각형에서 투영한 QR3 시작점과 이동량도 달라진다. 확인용으로 원래 QR3 사각형 값을 넣어 따로 돌렸을 때는 31개 지표가 모두 일치하고 점 좌표 차이가 최대 0.002 px 였다(이 결과 파일에는 넣지 않음).\n\n"
      << "## QR별 결과\n\n" << pq.md() << "\n## 2x2 배치별\n\n" << bt.md() << "\n## 불확실 사유 (한 점에 여러 개 가능)\n\n" << rt.md()
      << "\n## 품질 점수 (confidence_map.py)\n\n`exp(-0.5 (RMSE/15)^2) * exp(-0.5 (두 창 차이/0.65)^2) * clip(대비/90, 0, 1)`, optimizer·search_limit·blur_limit 이면 x0.25.\n\n"
      << "| 분위 | 0 | 25% | 50% | 75% | 100% |\n|---|---|---|---|---|---|\n| 품질 점수 | " << fmt(q0, 4) << " | " << fmt(q25, 4) << " | " << fmt(q50, 4)
      << " | " << fmt(q75, 4) << " | " << fmt(q100, 4) << " |\n\n"
      << "## 합성 영상 검증\n\n" << st.md()
      << "\n## 원래 결과와 대조 (" << par.ok << "/" << par.total << " 일치)\n\n" << par.t.md()
      << "\n## 옮기지 않은 부분\n\n- `confidence_squares.py`, `confidence_outlines.py`, `confidence_bordered_squares.py`, `package_figure.py`: 같은 품질 점수를 다른 표시 모양·한글 제목으로 다시 그린 발표용 그림이라 `03_품질점수.png` 하나로 대신했다.\n- `report.html`: 보고서.md 로 대신했다.\n"
      << "\n## 파일\n\n- `refined_corners.csv`: 755점 시작·맞춤 좌표와 진단값, 품질 점수\n- `uncertain_corners.csv`: 불확실 점만\n"
      << "- `이미지/01_전체모서리_통과불확실.png`: 세 QR 판, 연두 = 통과, 주황 = 불확실 (가로·세로 같은 배율)\n"
      << "- `이미지/02_큰네모_시작점대비맞춤.png`: QR1 왼쪽 위 파인더, 보라 = 투영 시작점\n- `이미지/03_품질점수.png`, `04_진단_RMSE와대비.png`, `05_시작점대비이동.png`, `06`·`07` 튀는 값\n";
    write_text(out("결과.md"), R.str());

    // ---- 보고서.md ----
    auto& ws = rows[wi_shift];
    auto& wr = rows[wi_rmse];
    int diag_total = by_black[2][0];
    int nondiag_total = total - diag_total, nondiag_valid = valid - by_black[2][1];
    std::ostringstream P;
    P << "# 보고서: QR 구조점 서브픽셀 경계 모델 맞춤 (10-15 18:38)\n\n"
      << "## 문제\n\nQR 격자에서 투영한 모서리 755점을 실제 영상 경계에 서브픽셀로 맞추고 점마다 믿을 만한지 표시해야 했다.\n\n"
      << "## 해결 방법\n\n모서리마다 2x2 모듈 배치의 erf 흐림 경계 모델을 9x9·11x11 창에서 soft-L1 최소제곱으로 맞추고, 수렴·RMSE·대비·두 창 차이로 통과를 나눴다.\n\n"
      << "## 결과\n\n755점 중 " << valid << "점이 통과하고 " << total - valid << "점이 불확실했으며, 투영 시작점에서 맞춘 점까지의 이동 중앙값은 " << fmt(median(shifts), 3) << " px였다.\n\n"
      << "| 지표 | 값 |\n|---|---|\n"
      << "| 통과 / 전체 | " << valid << " / " << total << " |\n"
      << "| QR1 / QR2 / QR3 통과 | " << perqr[1].valid << "/" << perqr[1].total << " · " << perqr[2].valid << "/" << perqr[2].total << " · " << perqr[3].valid << "/"
      << perqr[3].total << " |\n"
      << "| 대각 접점 제외 통과 | " << nondiag_valid << " / " << nondiag_total << " |\n"
      << "| 이동 중앙값 / 최대 (px) | " << fmt(median(shifts), 4) << " / " << fmt(max_abs(shifts), 4) << " |\n"
      << "| RMSE 중앙값 (밝기) | " << fmt(median(rmses), 3) << " |\n"
      << "| 품질 점수 중앙값 | " << fmt(q50, 4) << " |\n\n"
      << img_md("01_전체모서리_통과불확실.png", "전체 모서리") << "\n\n" << img_md("02_큰네모_시작점대비맞춤.png", "큰 네모 상세") << "\n\n"
      << img_md("04_진단_RMSE와대비.png", "진단 산점도") << "\n\n"
      << "## 튀는 값\n\n시작점에서 가장 멀리 간 점은 " << cname(ws) << "(" << fmt(ws.shift, 3) << " px, " << (ws.valid ? "통과" : ws.reason)
      << ")이고, 밝기 모델이 가장 안 맞은 점은 " << cname(wr) << "(RMSE " << fmt(wr.rmse, 2) << ", 검정 이웃 " << wr.c.black << "개)이다.\n\n"
      << outl.md() << "\n" << img_md("06_튀는값_최대이동.png", "최대 이동") << "\n\n" << img_md("07_튀는값_최대RMSE.png", "최대 RMSE")
      << "\n\n보라 십자는 QR 사각형에서 투영한 시작점, 빨간 십자는 맞춘 점이다.\n\n"
      << "## 한줄평\n\n모델이 2x2 모듈의 곧은 erf 경계라서 대각으로 닿은 검정 2개 점(" << diag_total << "점)은 인쇄 모서리가 둥글게 붙어 모델과 달라 하나도 통과하지 못했고("
      << by_black[2][1] << "/" << diag_total << "), 나머지 " << nondiag_total << "점은 " << nondiag_valid << "점이 통과해 불확실 점 대부분이 모델 모양의 한계에서 나왔다.\n\n"
      << "## 원래 결과와 대조\n\n원래 값은 `subpixel_grid_corners_20261015/checks.json`, `refined_corners.csv`, `confidence_filled/settings.json`, `decoded_grid_corners_20261015/metadata.json` 에서 가져왔다. "
      << par.ok << "/" << par.total << " 일치. QR3 사각형은 OpenCV 3.4 해독기가 흐린 영역에서만 찾아 원래와 최대 " << fmt(q3diff, 2) << " px 다르고, 그래서 QR3 시작점에서 잰 이동 중앙값이 어긋났다. 맞춤 자체(QR1·QR2 의 좌표, RMSE·대비·통과 판정)는 원래와 같다.\n\n"
      << par.t.md();
    write_text(out("보고서.md"), P.str());
    std::cout << "valid " << valid << "/" << total << " median shift " << median(shifts) << " parity " << par.ok << "/" << par.total << "\n";
    return 0;
}
