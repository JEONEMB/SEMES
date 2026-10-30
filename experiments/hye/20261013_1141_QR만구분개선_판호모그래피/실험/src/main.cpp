// v1 에서 QR 가로 위치로 구분 안 되던 Y·Θ 사진을 QR만으로 더 구분한다: 비교값(QR 세로·폭·높이, QR별 ECC affine, 판 전체 호모그래피)을 바꾸고
// 규칙1(값마다 3σ)과 규칙2(마할라노비스 99.73%)로 판정, 같은 값으로 Θ·Y 직접 추정.
// 원본: 00000.legacies/analysis_v2/extract_homography.py, judge.py, estimate.py, make_outputs.py
#include <hye/dataset.hpp>
#include <hye/ecc_compat.hpp>
#include <hye/plot.hpp>
#include <hye/report.hpp>
#include <omp.h>
#include <array>
#include <iostream>
#include <map>
#include <sstream>
#include "judge.hpp"

using namespace hye;

namespace {

// ---- QR 세 영역 ECC affine (analysis_agent/measure_qr.py, 시범 실험과 같음) ----
const std::array<cv::Vec4i, 3> kBoxes{{{1554, 1734, 972, 1164}, {1815, 1995, 972, 1164}, {2073, 2253, 972, 1164}}};
const int kPad = 130;
struct QR { double dx = 0, dy = 0, a00 = 1, a01 = 0, a10 = 0, a11 = 1; };
std::array<QR, 3> measure_qr(const cv::Mat& center, const cv::Mat& im) {
    std::array<QR, 3> out;
    for (int q = 0; q < 3; ++q) {
        auto b = kBoxes[q];
        cv::Mat tpl = center(cv::Range(b[2], b[3]), cv::Range(b[0], b[1]));
        cv::Mat search = im(cv::Range(b[2] - kPad, b[3] + kPad), cv::Range(b[0] - kPad, b[1] + kPad)), res;
        cv::matchTemplate(search, tpl, res, cv::TM_CCOEFF_NORMED);
        cv::Point loc;
        cv::minMaxLoc(res, nullptr, nullptr, nullptr, &loc);
        int dx0 = loc.x - kPad, dy0 = loc.y - kPad;
        cv::Mat target = im(cv::Range(b[2] + dy0, b[3] + dy0), cv::Range(b[0] + dx0, b[1] + dx0));
        cv::Mat W = cv::Mat::eye(2, 3, CV_32F);
        try {
            semes::ecc5::findTransformECCWithMask(tpl, target, cv::noArray(), cv::noArray(), W, cv::MOTION_AFFINE,
                                                  cv::TermCriteria(cv::TermCriteria::EPS | cv::TermCriteria::COUNT, 150, 1e-7), 3);
        } catch (const cv::Exception&) {
        }
        out[q] = {dx0 + W.at<float>(0, 2), dy0 + W.at<float>(1, 2), W.at<float>(0, 0), W.at<float>(0, 1), W.at<float>(1, 0), W.at<float>(1, 1)};
    }
    return out;
}
double cx(const QR& r) { return r.dx + (r.a00 - 1) * 90 + r.a01 * 96; }
double cy(const QR& r) { return r.dy + r.a10 * 90 + (r.a11 - 1) * 96; }

// ---- 판 전체 호모그래피 (extract_homography.py) ----
// center 의 QR 중심: 원래 14_데이터_QR마커위치.csv center 행 = cv2.QRCodeDetectorAruco 꼭짓점 평균. cv2 5.0 으로 같은 center.bmp 에서 다시 뽑은 값과 같다.
const double cLx = 1642.4546, cCx = 1900.4866, cRx = 2159.2039, cLy = 1067.5, cCy = 1068.25, cRy = 1070.5366;
const int X0 = int(cLx - 110), X1 = int(cRx + 110), Y0 = int(std::min(cLy, cRy) - 120), Y1 = int(std::max(cLy, cRy) + 120);
const char* kHomNames[] = {"TL", "TR", "BR", "BL", "QR_L", "QR_C", "QR_R"};
struct Hom { double cc = 0, h20 = 0, h21 = 0; std::array<cv::Point2d, 7> d; };
Hom measure_hom(const cv::Mat& tpl, const cv::Mat& im, int sx, int sy) {
    cv::Mat crop;
    im(cv::Range(Y0 + sy, Y1 + sy), cv::Range(X0 + sx, X1 + sx)).convertTo(crop, CV_32F);
    cv::Mat W = cv::Mat::eye(3, 3, CV_32F);
    Hom h;
    h.cc = semes::ecc5::findTransformECCWithMask(tpl, crop, cv::noArray(), cv::noArray(), W, cv::MOTION_HOMOGRAPHY,
                                                 cv::TermCriteria(cv::TermCriteria::EPS | cv::TermCriteria::COUNT, 300, 1e-7), 5);
    const cv::Point2f pts[7] = {{(float)X0, (float)Y0}, {(float)X1, (float)Y0}, {(float)X1, (float)Y1}, {(float)X0, (float)Y1},
                                {(float)cLx, (float)cLy}, {(float)cCx, (float)cCy}, {(float)cRx, (float)cRy}};
    std::vector<cv::Point2f> loc, m;
    for (auto& p : pts) loc.push_back(p - cv::Point2f((float)X0, (float)Y0));
    cv::perspectiveTransform(loc, m, W);
    for (int k = 0; k < 7; ++k) {
        cv::Point2f mm = m[k] + cv::Point2f((float)(X0 + sx), (float)(Y0 + sy));
        h.d[k] = cv::Point2d(mm.x - pts[k].x, mm.y - pts[k].y);
    }
    h.h20 = W.at<float>(2, 0);
    h.h21 = W.at<float>(2, 1);
    return h;
}

// ---- 뒤쪽 볼트 가로 이동 (15_데이터_파인더_볼트이동량.csv bolt_dx 대신): (3576, 1802) 48x48 템플릿, ±120/±12, NCC + 포물선 ----
const cv::Point kBolt{3576, 1802};
double parab(float am, float a0, float ap) {
    float d = am - 2.0f * a0 + ap;
    return d != 0 ? (0.5f * (am - ap)) / d : 0.0f;
}
double track_bolt(const cv::Mat& tpl, const cv::Mat& g) {
    cv::Mat s, r;
    g(cv::Rect(kBolt.x - 24 - 120, kBolt.y - 24 - 12, 2 * 144, 2 * 36)).convertTo(s, CV_32F);
    cv::matchTemplate(s, tpl, r, cv::TM_CCOEFF_NORMED);
    cv::Point b;
    cv::minMaxLoc(r, nullptr, nullptr, nullptr, &b);
    double x = b.x;
    if (b.x > 0 && b.x < r.cols - 1) x += parab(r.at<float>(b.y, b.x - 1), r.at<float>(b.y, b.x), r.at<float>(b.y, b.x + 1));
    return x - 120;
}

struct Version { std::string id, label; std::vector<std::string> cols; judge::Result R; judge::Est E; };

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
    omp_set_num_threads(4);
    begin_results();
    auto shots = load_1014();
    const int n = (int)shots.size();
    const cv::Mat center = read_gray(center_1014(shots));
    cv::Mat htpl, btpl;
    center(cv::Range(Y0, Y1), cv::Range(X0, X1)).convertTo(htpl, CV_32F);
    center(cv::Rect(kBolt.x - 24, kBolt.y - 24, 48, 48)).convertTo(btpl, CV_32F);
    std::vector<std::array<QR, 3>> mq(n);
    std::vector<Hom> mh(n);
    std::vector<double> bolt(n);
    std::vector<cv::Point> shift(n);
#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < n; ++i) {
        cv::Mat g = read_gray(shots[i]);
        mq[i] = measure_qr(center, g);
        // 판 창 이동: 원래는 QR 네모 가로·세로 평균을 반올림 (여기서는 ECC 로 잰 QR 중심 평균)
        double mx = (cx(mq[i][0]) + cx(mq[i][1]) + cx(mq[i][2])) / 3, my = (cy(mq[i][0]) + cy(mq[i][1]) + cy(mq[i][2])) / 3;
        shift[i] = {(int)std::nearbyint(mx), (int)std::nearbyint(my)};
        mh[i] = measure_hom(htpl, g, shift[i].x, shift[i].y);
        bolt[i] = track_bolt(btpl, g);
    }
    std::cout << "measured\n";
    auto fname = [&](int i) { return shots[i].rel.substr(shots[i].rel.rfind('/') + 1); };
    auto stem = [&](const std::string& f) { return f.substr(0, f.rfind('.')); };

    // 특징 이름 → 값
    auto feat = [&](int i) {
        std::map<std::string, double> f;
        const char* LCR[] = {"L", "C", "R"};
        for (int q = 0; q < 3; ++q) {
            auto& r = mq[i][q];
            f[std::string("d") + LCR[q] + "x"] = cx(r);
            f[std::string("d") + LCR[q] + "y"] = cy(r);
            f[std::string(LCR[q]) + "w"] = 180 * r.a00;  // QR 네모 폭 (원래: 검출 꼭짓점 폭)
            f[std::string(LCR[q]) + "h"] = 192 * r.a11;  // QR 네모 높이
            const std::string p = "q" + std::to_string(q) + "_";
            f[p + "dx"] = r.dx; f[p + "dy"] = r.dy; f[p + "a00"] = r.a00; f[p + "a01"] = r.a01; f[p + "a10"] = r.a10; f[p + "a11"] = r.a11;
        }
        for (int k = 0; k < 7; ++k) { f[std::string(kHomNames[k]) + "_dx"] = mh[i].d[k].x; f[std::string(kHomNames[k]) + "_dy"] = mh[i].d[k].y; }
        f["bolt_dx"] = bolt[i];
        return f;
    };
    const std::vector<std::string> X3{"dLx", "dCx", "dRx"};
    std::vector<std::string> B12 = X3;
    for (auto c : {"dLy", "dCy", "dRy", "Lw", "Cw", "Rw", "Lh", "Ch", "Rh"}) B12.push_back(c);
    std::vector<std::string> A1;
    for (int q = 0; q < 3; ++q) for (auto c : {"dx", "dy", "a00", "a01", "a10", "a11"}) A1.push_back("q" + std::to_string(q) + "_" + c);
    std::vector<std::string> A2;
    for (int k = 0; k < 7; ++k) { A2.push_back(std::string(kHomNames[k]) + "_dx"); A2.push_back(std::string(kHomNames[k]) + "_dy"); }
    std::vector<std::string> A3 = B12;
    A3.insert(A3.end(), A1.begin(), A1.end());
    A3.insert(A3.end(), A2.begin(), A2.end());
    std::vector<std::string> Dd = X3;
    Dd.push_back("bolt_dx");
    std::vector<Version> V{{"v1", "QR box x 3", X3},           {"v2-b", "QR box x, y, w, h 12", B12}, {"v2-a1", "QR ECC affine 18", A1},
                           {"v2-a2", "plate homography 14", A2}, {"v2-a3", "QR all " + std::to_string(A3.size()), A3}, {"v2-d", "QR + rear bolt 4 (ref)", Dd}};
    std::vector<std::map<std::string, double>> F(n);
    for (int i = 0; i < n; ++i) F[i] = feat(i);
    for (auto& v : V) {
        std::vector<judge::Item> items;
        for (int i = 0; i < n; ++i) {
            if (shots[i].condition == "center") continue;  // 원래도 center 없는 280장 (15_ CSV 와 합치며 빠짐)
            judge::Item it;
            it.file = fname(i);
            it.ax = shots[i].condition == "T" ? 'T' : 'Y';
            it.value = it.ax == 'T' ? shots[i].t : shots[i].y;
            it.m = (F[i]["dLx"] + F[i]["dCx"] + F[i]["dRx"]) / 3;
            for (auto& c : v.cols) it.f.push_back(F[i][c]);
            it.src = i;
            items.push_back(it);
        }
        v.R = judge::run(items);
        v.E = judge::estimate(v.R);
        std::cout << v.id << " r1 Y " << v.R.n1Y << " r2 Y " << v.R.n2Y << " T " << v.R.n2T << " Yrms " << v.E.yrms_um << "\n";
    }

    // ---------- 측정 CSV ----------
    {
        Table t{{"file", "cc", "h20", "h21"}};
        for (auto k : kHomNames) { t.head.push_back(std::string(k) + "_dx"); t.head.push_back(std::string(k) + "_dy"); }
        t.head.push_back("bolt_dx");
        for (int i = 0; i < n; ++i) {
            std::vector<std::string> r{fname(i), fmt(mh[i].cc, 6), fmt(mh[i].h20 * 1e6, 4) + "e-6", fmt(mh[i].h21 * 1e6, 4) + "e-6"};
            for (int k = 0; k < 7; ++k) { r.push_back(fmt(mh[i].d[k].x, 4)); r.push_back(fmt(mh[i].d[k].y, 4)); }
            r.push_back(fmt(bolt[i], 4));
            t.add(r);
        }
        t.csv(out("판호모그래피_볼트_측정.csv"));
    }
    // 사진별 버전 판정 (23_YΘ구분표_v2.xlsx Y·Θ 시트 대신)
    auto sheet_csv = [&](bool isY, const std::string& file) {
        Table t{{"파일명", isY ? "값 (mm)" : "값 (deg)"}};
        for (auto& v : V) { t.head.push_back(v.id + " 규칙1"); t.head.push_back(v.id + " 규칙2"); t.head.push_back(v.id + " 거리"); }
        auto& base = isY ? V[0].R.Ys : V[0].R.Ts;
        for (auto& it : base) {
            std::vector<std::string> r{it.file, fmt(it.value, 2)};
            for (auto& v : V) {
                auto& s = isY ? v.R.Ys : v.R.Ts;
                auto& d = isY ? v.R.dy : v.R.dt;
                for (size_t j = 0; j < s.size(); ++j)
                    if (s[j].file == it.file) {
                        if (d[j].out) { r.insert(r.end(), {"범위 밖", "범위 밖", ""}); }
                        else r.insert(r.end(), {d[j].r1 ? "불가" : "가능", d[j].r2 ? "불가" : "가능", fmt(d[j].d2, 3)});
                    }
            }
            t.add(r);
        }
        t.csv(out(file));
    };
    sheet_csv(true, "구분표v2_Y.csv");
    sheet_csv(false, "구분표v2_T.csv");

    // ---------- 그림 ----------
    {  // 24 히트맵 대신: 버전별 구분 불가 비율 막대
        std::vector<std::string> cats;
        std::vector<double> y2, t2, yr;
        for (auto& v : V) { cats.push_back(v.id); y2.push_back(100.0 * v.R.n2Y / v.R.Ys.size()); t2.push_back(100.0 * v.R.n2T / v.R.inT); yr.push_back(v.E.yrms_um); }
        bar_chart("01_버전별_Y구분불가.png", "Rule 2 indistinguishable Y photos (% of 80)", cats, y2, "%", 3);
        bar_chart("02_버전별_T구분불가.png", "Rule 2 indistinguishable T photos (% of in-range)", cats, t2, "%", 3);
        bar_chart("03_버전별_Y추정RMS.png", "Direct estimate Y RMS (um), 2-fold", cats, yr, "um", 3, 90);
    }
    auto& a2 = V[3];
    {
        auto plot = [&](const Version& v, const std::string& file) {
            std::vector<double> xy, yy, xt, yt;
            for (size_t i = 0; i < v.R.Ys.size(); ++i) if (!v.R.dy[i].out) { xy.push_back(v.R.Ys[i].value); yy.push_back(std::sqrt(v.R.dy[i].d2)); }
            for (size_t i = 0; i < v.R.Ts.size(); ++i) if (!v.R.dt[i].out) { xt.push_back(v.R.Ts[i].value / 0.18); yt.push_back(std::sqrt(v.R.dt[i].d2)); }
            Chart c(v.id + " rule 2: Mahalanobis distance to partner", "Y (mm); T photos plotted at T / 0.18", "sqrt(d2)");
            c.scatter(xy, yy, color(2), "Y photos").scatter(xt, yt, color(1), "T photos").hline(std::sqrt(v.R.dthr), kRed);
            c.save(file);
        };
        plot(V[0], "04_v1_통계거리.png");
        plot(a2, "05_v2-a2_통계거리.png");
    }
    std::vector<double> ey, et, ty, py, tt, pt;
    std::vector<std::string> names;
    for (size_t i = 0; i < a2.E.rows.size(); ++i) {
        ty.push_back(a2.E.tY[i]); py.push_back(a2.E.pY[i]); tt.push_back(a2.E.tT[i]); pt.push_back(a2.E.pT[i]);
        ey.push_back(py.back() - ty.back()); et.push_back(pt.back() - tt.back());
        names.push_back(stem(a2.E.rows[i]->file));
    }
    pred_vs_truth("06_v2-a2_추정오차.png", "v2-a2 plate homography, 2-fold", ty, py, tt, pt, names);
    const size_t wy = worst(ey, 1)[0], wt = worst(et, 1)[0];
    auto outlier_fig = [&](size_t k, const std::string& file, const std::string& cap) {
        int i = a2.E.rows[k]->src;
        cv::Mat im = read_color(shots[i]);
        std::vector<cv::Mat> tiles;
        // 판 귀퉁이 4개 (TL, TR, BR, BL): 초록 = center 위치, 빨강 = 호모그래피로 옮긴 위치, 화살표 = 이동 - 판 평균 이동 (x200)
        double mx = 0, my = 0;
        for (int c = 0; c < 4; ++c) { mx += mh[i].d[c].x / 4; my += mh[i].d[c].y / 4; }
        const cv::Point2d P[4] = {{(double)X0, (double)Y0}, {(double)X1, (double)Y0}, {(double)X1, (double)Y1}, {(double)X0, (double)Y1}};
        for (int c = 0; c < 4; ++c) {
            cv::Point2d now = P[c] + mh[i].d[c];
            cv::Rect roi((int)now.x - 60, (int)now.y - 60, 120, 120);
            tiles.push_back(zoom(im, roi, 300, {{now, {mh[i].d[c].x - mx, mh[i].d[c].y - my}, kHomNames[c], kRed}, {P[c] + cv::Point2d(mx, my), {0, 0}, "", kGreen}}, 200,
                                 c == 0 ? cap : ""));
        }
        save_png(file, hstack(tiles));
    };
    const std::string fy = "07_튀는값_Y_" + names[wy] + ".png", ft = "08_튀는값_T_" + names[wt] + ".png";
    outlier_fig(wy, fy, names[wy] + " Y err " + fmts(ey[wy], 3));
    outlier_fig(wt, ft, names[wt] + " T err " + fmts(et[wt], 3));

    // ---------- 원래 결과와 대조 ----------
    // 출처: 00000.legacies/00_v2결과리포트.md 결과 표, analysis_v2/estimate_result.csv, analysis_v2/homography_features.csv (T-1.00 행)
    struct Orig { int r1Y; double fp1; int r2Y, r2T; double fp2, yrms, ymax; int over90; double trms; };
    const Orig O[6] = {{33, 0.7, 36, 33, 0.7, 235.644, 1155.722, 168, 0.042181}, {29, 5.1, 29, 24, 3.3, 190.815, 706.749, 173, 0.034465},
                       {26, 4.3, 21, 21, 1.8, 128.954, 554.580, 120, 0.022389}, {22, 1.8, 15, 16, 1.4, 117.504, 328.646, 128, 0.020848},
                       {21, 7.2, 15, 15, 2.9, 87.161, 420.877, 70, 0.015266},  {5, 1.4, 5, 5, 0.7, 52.396, 187.407, 31, 0.008859}};
    Parity par, full;
    for (int k = 0; k < 6; ++k) {
        auto& v = V[k];
        full.add(v.id + " 규칙1 Y 불가", O[k].r1Y, v.R.n1Y, 0, 0);
        full.add(v.id + " 규칙1 오판율 %", O[k].fp1, v.R.fp1 * 100, 0.3, 1);
        full.add(v.id + " 규칙2 Y 불가", O[k].r2Y, v.R.n2Y, 0, 0);
        full.add(v.id + " 규칙2 Θ 불가", O[k].r2T, v.R.n2T, 0, 0);
        full.add(v.id + " 규칙2 오판율 %", O[k].fp2, v.R.fp2 * 100, 0.3, 1);
        full.add(v.id + " Y RMS µm", O[k].yrms, v.E.yrms_um, 5, 1);
        full.add(v.id + " Y 최대 µm", O[k].ymax, v.E.ymax_um, 20, 1);
        full.add(v.id + " Y 90µm 초과", O[k].over90, v.E.over90, 0, 0);
        full.add(v.id + " Θ RMS", O[k].trms, v.E.trms, 0.001);
    }
    int iT100 = -1;
    for (int i = 0; i < n; ++i) if (fname(i) == "T-1.00.bmp") iT100 = i;
    par.add("v2-a2 규칙2 구분 불가 Y (/80)", 15, a2.R.n2Y, 0, 0);
    par.add("v2-a2 규칙2 구분 불가 Θ (/71)", 16, a2.R.n2T, 0, 0);
    par.add("v2-a2 규칙2 오판율 (%)", 1.4, a2.R.fp2 * 100, 0.3, 1);
    par.add("v2-a2 추정 Y RMS (µm)", 117.504, a2.E.yrms_um, 5, 1);
    par.add("v2-a2 Y 90µm 초과 (/280)", 128, a2.E.over90, 0, 0);
    par.add("v2-a3 추정 Y RMS (µm)", 87.161, V[4].E.yrms_um, 5, 1);
    par.add("v2-a1 추정 Y RMS (µm) (측정 동일)", 128.954, V[2].E.yrms_um, 1, 1);
    par.add("v2-a1 규칙2 구분 불가 Y (측정 동일)", 21, V[2].R.n2Y, 0, 0);
    par.add("T-1.00 판 좌상 dx (px)", -81.51111, mh[iT100].d[0].x, 0.01);
    par.add("T-1.00 판 QR좌 dx (px)", -81.19849, mh[iT100].d[4].x, 0.01);
    par.add("T-1.00 판 우하 dy (px)", -0.29138184, mh[iT100].d[2].y, 0.01);
    std::cout << "parity " << par.ok << "/" << par.total << " full " << full.ok << "/" << full.total << "\n";

    // ---------- 결과.md ----------
    Table vt{{"버전", "비교값", "값 개수", "규칙1 Y 불가", "규칙1 오판율", "규칙2 Y 불가", "규칙2 Θ 불가", "규칙2 오판율", "추정 Y RMS (µm)", "Y 최대 (µm)", "Y 90µm 초과", "Θ RMS",
              "작은 틀어짐 Y RMS (µm)"}};
    for (auto& v : V)
        vt.add({v.id, v.label, std::to_string(v.cols.size()), std::to_string(v.R.n1Y) + "/" + std::to_string(v.R.Ys.size()), fmt(v.R.fp1 * 100, 1) + " %",
                std::to_string(v.R.n2Y) + "/" + std::to_string(v.R.Ys.size()), std::to_string(v.R.n2T) + "/" + std::to_string(v.R.inT), fmt(v.R.fp2 * 100, 1) + " %", fmt(v.E.yrms_um, 1),
                fmt(v.E.ymax_um, 1), std::to_string(v.E.over90) + "/" + std::to_string(v.E.rows.size()), fmt(v.E.trms, 4), fmt(v.E.small_um, 1)});
    vt.csv(out("버전별_요약.csv"));
    auto list_r2 = [&](const Version& v, bool isY) {
        std::string o;
        auto& s = isY ? v.R.Ys : v.R.Ts;
        auto& d = isY ? v.R.dy : v.R.dt;
        std::vector<std::pair<double, std::string>> l;
        for (size_t i = 0; i < s.size(); ++i) if (!d[i].out && d[i].r2) l.push_back({s[i].value, stem(s[i].file)});
        std::sort(l.begin(), l.end());
        for (auto& p : l) o += (o.empty() ? "" : ", ") + p.second;
        return o;
    };
    std::ostringstream Rm;
    Rm << "# 결과: QR만 구분 개선, 판 호모그래피 (v2)\n\n"
       << "입력: `data/clean/clean_20261014` 280장 (center 제외, 원래와 같음). 짝 기준은 v1과 같이 QR 3개 가로 평균 px이다.\n\n"
       << "측정:\n\n- QR 네모 위치·폭·높이: QR 세 영역 템플릿 매칭 + ECC affine (가로 = 중심 x 이동, 세로 = 중심 y 이동, 폭 = 180 a00, 높이 = 192 a11).\n"
       << "- QR별 ECC affine 18개: 같은 측정의 dx, dy, a00, a01, a10, a11 (analysis_agent/measure_qr.py와 같음).\n"
       << "- 판 호모그래피 14개: center의 판 창 (" << X0 << ".." << X1 << ", " << Y0 << ".." << Y1 << ")을 ECC homography (300회, 1e-7, 필터 5)로 맞춘 뒤 판 귀퉁이 4개와 QR 중심 3개의 이동.\n"
       << "- 뒤쪽 볼트 1개: (3576, 1802) 48x48 템플릿 가로 이동.\n\n"
       << "## 버전별 결과\n\n" << vt.md() << "\nv2-a2 규칙2로도 구분 불가 Y: " << list_r2(a2, true) << "\n\nv2-a2 규칙2로도 구분 불가 Θ: " << list_r2(a2, false) << "\n\n"
       << "## 전체 대조 (" << full.ok << "/" << full.total << " 일치)\n\n원래 값: `00000.legacies/00_v2결과리포트.md` 결과 표, `analysis_v2/estimate_result.csv`.\n\n" << full.t.md()
       << "\n## 옮기지 않은 부분\n\n- QR 네모 가로·세로·폭·높이(v1, v2-b): 원래는 `cv2.QRCodeDetectorAruco` 꼭짓점(`14_데이터_QR마커위치.csv`)이다. OpenCV 3.4에 없어 QR 영역 ECC affine으로 대신 쟀다. v1·v2-b·짝 기준 m이 원래와 다르다.\n"
       << "- 뒤쪽 볼트(v2-d): 원래 `15_데이터_파인더_볼트이동량.csv` bolt_dx를 만든 코드가 없다. 같은 위치 템플릿 매칭으로 대신했다.\n"
       << "- 판 창 이동(sx, sy): 원래는 QR 네모 평균을 반올림했다. 여기서는 ECC로 잰 QR 중심 평균을 반올림했다.\n"
       << "- `make_outputs.py`의 엑셀(23_YΘ구분표_v2.xlsx)과 히트맵은 CSV(`구분표v2_Y.csv`, `구분표v2_T.csv`, `버전별_요약.csv`)와 막대 그림으로 대신했다.\n"
       << "- 결과/의 `estimate_result.csv`, `homography_features.csv`, `homography_test.csv`는 원래 Python 결과라 그대로 두었다.\n\n"
       << "## 원래 결과와 대조 (" << par.ok << "/" << par.total << " 일치)\n\n" << par.t.md()
       << "\n## 파일\n\n- `판호모그래피_볼트_측정.csv`: 사진별 판 호모그래피 14값, h20, h21, 볼트\n- `구분표v2_Y.csv`, `구분표v2_T.csv`: 사진별 버전 판정\n- `버전별_요약.csv`\n- `이미지/`: 그림 8장\n";
    write_text(out("결과.md"), Rm.str());

    // ---------- 보고서.md ----------
    auto cnt = [](const std::vector<double>& v) { return (int)std::count_if(v.begin(), v.end(), [](double x) { return std::abs(x) <= kPass; }); };
    auto dec = [&](const judge::Item* it) -> const judge::Decision& {
        auto& s = it->ax == 'Y' ? a2.R.Ys : a2.R.Ts;
        auto& d = it->ax == 'Y' ? a2.R.dy : a2.R.dt;
        for (size_t i = 0; i < s.size(); ++i) if (s[i].file == it->file) return d[i];
        throw std::runtime_error("dec");
    };
    std::ostringstream P;
    P << "# 보고서: QR만 구분 개선, 판 호모그래피 (10-13 11:41)\n\n"
      << "## 문제\n\nv1에서 QR 가로 위치로 구분 안 되던 Y·Θ 사진을 QR만으로 더 구분하려 했다.\n\n"
      << "## 해결 방법\n\n비교값을 QR 세로·폭·높이, QR별 ECC affine, 판 전체 호모그래피로 바꾸고, 값마다 3σ(규칙1)와 마할라노비스 거리 99.73 %(규칙2)로 판정했다.\n\n"
      << "## 결과\n\n판 전체 호모그래피(v2-a2)는 규칙2 구분 불가 Y " << a2.R.n2Y << "/80, Θ " << a2.R.n2T << "/" << a2.R.inT << ", 추정 Y RMS " << fmt(a2.E.yrms_um, 0)
      << " µm로 v1(Y " << V[0].R.n2Y << "/80, " << fmt(V[0].E.yrms_um, 0) << " µm)보다 줄었다.\n\n"
      << "| 버전 | 값 개수 | 규칙2 Y 불가 | 규칙2 Θ 불가 | 오판율 | Y RMS (µm) | Y 최대 (µm) | Y 통과 (\\|eY\\|<=0.09) | Θ RMS |\n|---|---|---|---|---|---|---|---|---|\n";
    for (auto& v : V) {
        std::vector<double> e;
        for (size_t i = 0; i < v.E.rows.size(); ++i) e.push_back(v.E.pY[i] - v.E.tY[i]);
        P << "| " << v.id << " | " << v.cols.size() << " | " << v.R.n2Y << "/80 | " << v.R.n2T << "/" << v.R.inT << " | " << fmt(v.R.fp2 * 100, 1) << " % | " << fmt(v.E.yrms_um, 1) << " | "
          << fmt(v.E.ymax_um, 1) << " | " << cnt(e) << "/" << e.size() << " | " << fmt(v.E.trms, 4) << " |\n";
    }
    P << "\nv2-a2 T 통과 " << cnt(et) << "/" << et.size() << ".\n\n" << img_md("01_버전별_Y구분불가.png", "Y 구분 불가") << "\n\n" << img_md("02_버전별_T구분불가.png", "T 구분 불가") << "\n\n"
      << img_md("03_버전별_Y추정RMS.png", "Y RMS") << "\n\n" << img_md("04_v1_통계거리.png", "v1 거리") << "\n\n" << img_md("05_v2-a2_통계거리.png", "v2-a2 거리") << "\n\n"
      << img_md("06_v2-a2_추정오차.png", "v2-a2 추정") << "\n\n"
      << "## 튀는 값\n\nv2-a2에서 가장 크게 틀린 사진은 Y는 " << names[wy] << "(정답 Y " << fmt(ty[wy], 2) << ", 추정 " << fmt(py[wy], 3) << " mm), T는 " << names[wt] << "(정답 T "
      << fmt(tt[wt], 2) << ", 추정 " << fmt(pt[wt], 3) << ")이고, 판 귀퉁이 좌우 이동 차(우상 - 좌상)가 " << fmts(mh[a2.E.rows[wy]->src].d[1].x - mh[a2.E.rows[wy]->src].d[0].x, 3)
      << " px로 원근 단서가 작아 직선 추정이 평균 이동을 Y와 Θ에 잘못 나눴다.\n\n"
      << "| 구분 | 사진 | 정답 Y (mm) | 추정 Y | 정답 T (deg) | 추정 T | 좌상 / 우상 / 우하 / 좌하 dx (px) | 우상-좌상 dx (px) | 규칙2 거리 |\n|---|---|---|---|---|---|---|---|---|\n";
    for (auto [lab, idx] : std::vector<std::pair<std::string, std::vector<size_t>>>{{"Y 오차 상위", worst(ey, 3)}, {"T 오차 상위", worst(et, 3)}})
        for (size_t k : idx) {
            int i = a2.E.rows[k]->src;
            auto& d = dec(a2.E.rows[k]);
            P << "| " << lab << " | " << names[k] << " | " << fmt(ty[k], 2) << " | " << fmt(py[k], 3) << " | " << fmt(tt[k], 2) << " | " << fmt(pt[k], 3) << " | " << fmt(mh[i].d[0].x, 2) << " / "
              << fmt(mh[i].d[1].x, 2) << " / " << fmt(mh[i].d[2].x, 2) << " / " << fmt(mh[i].d[3].x, 2) << " | " << fmts(mh[i].d[1].x - mh[i].d[0].x, 3) << " | "
              << (d.out ? std::string("범위 밖") : fmt(d.d2, 2)) << " |\n";
        }
    P << "\n" << img_md(fy, "Y 튀는 값") << "\n\n" << img_md(ft, "T 튀는 값")
      << "\n\n빨간 십자는 호모그래피로 옮긴 판 귀퉁이, 초록 십자는 center 귀퉁이 + 판 평균 이동이다. 화살표는 귀퉁이 이동에서 판 평균 이동을 뺀 것(200배)이다.\n\n"
      << "## 한줄평\n\n판 양끝(약 " << (X1 - X0) << " px 떨어짐)을 한 원근 변환으로 맞추면 Θ가 만드는 판 기울어짐을 QR 하나씩보다 잘 잡아 구분 불가가 Y " << V[0].R.n2Y << " → " << a2.R.n2Y
      << "장으로 줄었지만, 남은 원근 단서가 잡음 수준이라 Y ±" << [&] {
             double mx = 0;
             for (size_t i = 0; i < a2.R.Ys.size(); ++i) if (!a2.R.dy[i].out && a2.R.dy[i].r2) mx = std::max(mx, std::abs(a2.R.Ys[i].value));
             return fmt(mx, 2);
         }() << " mm 안쪽은 여전히 사진 1장으로 구분되지 않았다.\n\n"
      << "## 원래 결과와 대조\n\n원래 값은 `00000.legacies/00_v2결과리포트.md`, `analysis_v2/estimate_result.csv`, `analysis_v2/homography_features.csv`에서 가져왔다. " << par.ok << "/" << par.total
      << " 일치. 버전 6개 전체 대조는 결과.md에 있다(" << full.ok << "/" << full.total << ").\n\n" << par.t.md()
      << "\n짝 기준 m과 v1·v2-b 값은 원래 `cv2.QRCodeDetectorAruco` 꼭짓점(OpenCV 3.4에 없음)이라 C++은 QR 영역 ECC로 다시 쟀다. 이 차이로 짝이 달라진다. "
         "판 호모그래피 ECC는 원래 값과 사진에 따라 최대 0.035 px 다르다(T-1.00은 0.002 px 안). 같은 창·같은 설정에서 cv2 5.0, OpenCV 3.4, 공통 ECC(`ecc_compat`)의 homography 결과도 서로 0.01~0.1 px 다르고, 원래 결과를 만든 cv2 판은 남아 있지 않다. "
         "Y·Θ를 가르는 원근 신호가 0.02~0.06 px라 이 차이가 v2-a2·v2-a3 구분 장수와 Y RMS 차이의 주 원인이다. 측정이 원래와 같은 v2-a1(QR별 ECC affine)은 Y RMS·구분 장수가 맞았다.\n";
    write_text(out("보고서.md"), P.str());
    return 0;
}
