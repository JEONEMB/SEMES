// A2: 정보 행렬과 약한 방향 (H01, H03).
//  J  = 1014 단일축 자료에서 맞춘 점별 감도 (d = a*Y + b*T + e), 1260x2 (+ 선 48x2)
//  Si = 1020/1029 반복에서 잰 잡음 공분산. 대각 근사를 쓰지 않는다 (A1: 유효 독립 차원 19)
//  F  = J^T Si^-1 J  (2x2). F^-1 이 추정 공분산이고, 그 큰 고유값 방향이 약한 방향이다.
#include "cache.hpp"
#include "noise.hpp"
#include "hye/plot.hpp"
#include <functional>
#include <iostream>
#include <set>

using namespace a0;
namespace {

// 판 가로 이동이 0 이 되는 방향: 14.25*dY + 80.04*dT = 0  ->  dY = -5.617*dT
const cv::Point2d kNullDir = [] {
    cv::Point2d v(-kTtoY, 1.0);
    return v / cv::norm(v);
}();

struct Fit {
    cv::Mat J;        // D x 2
    std::vector<double> resid;  // 차원별 적합 잔차 RMS
};

// shots 중 fit_ok 인 1014 단일축 사진으로 d = a*Y + b*T + e 를 맞춘다.
Fit fit_sensitivity(const Cache& c, const std::function<bool(const Shot&)>& ok, bool lines, double ylim, double tlim) {
    std::vector<int> ids;
    for (auto& s : c.shots)
        if (s.measured && s.set == "clean_20261014" && ok(s) && std::abs(s.y) <= ylim && std::abs(s.t) <= tlim) ids.push_back(s.idx);
    const int n = int(ids.size()), D = lines ? kLines : kPoints * 2;
    cv::Mat X(n, 3, CV_64F), Y(n, D, CV_64F);
    for (int i = 0; i < n; ++i) {
        X.at<double>(i, 0) = c.shots[ids[i]].y;
        X.at<double>(i, 1) = c.shots[ids[i]].t;
        X.at<double>(i, 2) = 1;
        if (lines) for (int b = 0; b < kLines; ++b) Y.at<double>(i, b) = c.line(ids[i])[b * 2];
        else for (int p = 0; p < kPoints; ++p) {
            auto d = c.dz(ids[i], p);
            Y.at<double>(i, p * 2) = d.x;
            Y.at<double>(i, p * 2 + 1) = d.y;
        }
    }
    cv::Mat B;
    cv::solve(X, Y, B, cv::DECOMP_SVD);   // 3 x D
    cv::Mat R = Y - X * B;
    Fit f;
    f.J = cv::Mat(D, 2, CV_64F);
    f.resid.assign(D, 0);
    for (int j = 0; j < D; ++j) {
        f.J.at<double>(j, 0) = B.at<double>(0, j);
        f.J.at<double>(j, 1) = B.at<double>(1, j);
        double s = 0;
        for (int i = 0; i < n; ++i) s += R.at<double>(i, j) * R.at<double>(i, j);
        f.resid[j] = std::sqrt(s / std::max(1, n));
    }
    return f;
}

struct Info {
    cv::Matx22d F, Finv;
    double sy = 0, st = 0;      // 예측 표준편차
    double weak_angle = 0;      // 약한 방향과 kNullDir 사이 각도 (deg)
    double weak_sigma = 0;      // 약한 방향의 표준편차 (Y mm 성분 기준 크기)
    cv::Point2d weak{0, 0};
    double cond = 0;
};

Info information(const cv::Mat& Jp, const InvSigma& Sp, const cv::Mat* Jl, const InvSigma* Sl) {
    cv::Mat F = Jp.t() * Sp.apply(Jp);
    if (Jl && Sl) F += Jl->t() * Sl->apply(*Jl);
    Info out;
    out.F = cv::Matx22d(F.at<double>(0, 0), F.at<double>(0, 1), F.at<double>(1, 0), F.at<double>(1, 1));
    out.Finv = out.F.inv();
    out.sy = std::sqrt(std::max(0.0, out.Finv(0, 0)));
    out.st = std::sqrt(std::max(0.0, out.Finv(1, 1)));
    cv::Mat ev, evec;
    cv::eigen(cv::Mat(out.Finv), ev, evec);   // 내림차순: 첫 행이 가장 큰 분산 = 약한 방향
    out.weak = {evec.at<double>(0, 0), evec.at<double>(0, 1)};
    out.weak_sigma = std::sqrt(std::max(0.0, ev.at<double>(0)));
    double cosang = std::abs(out.weak.dot(kNullDir)) / (cv::norm(out.weak) * cv::norm(kNullDir));
    out.weak_angle = std::acos(std::min(1.0, cosang)) * 180 / CV_PI;
    out.cond = ev.at<double>(1) > 0 ? ev.at<double>(0) / ev.at<double>(1) : 0;
    return out;
}

}  // namespace

int main() {
    try {
        const fs::path result = fs::u8path(EXPERIMENT_DIR);
        const fs::path results = result / "results";
        fs::create_directories(results);
        hye::begin_results();
        Cache c = load(cache_dir());

        // ---- 잡음 (fold 와 무관하게 1020/1029 반복 전체에서. A1 과 같은 정의) ----
        auto ns = measure_noise(c, [](const Shot&) { return true; });
        std::cout << "잡음 표본 " << ns.shots << "장 (묶음 " << ns.groups << ", 제외 " << ns.dropped << ")\n";

        std::vector<int> sel_dims, all_dims;
        for (int p = 0; p < kPoints; ++p)
            if (c.points[p].selected) { sel_dims.push_back(p * 2); sel_dims.push_back(p * 2 + 1); }
        for (int p = 0; p < kPoints; ++p) { all_dims.push_back(p * 2); all_dims.push_back(p * 2 + 1); }
        auto Sp_sel = build_inv_sigma(ns.point_rows, sel_dims);
        std::cout << "점 Sigma: 차원 " << Sp_sel.D << ", 표본 고유방향 " << Sp_sel.k
                  << ", 바닥 분산 " << Sp_sel.floor_var << "\n";
        std::vector<int> upper, lower, both;
        for (int b = 0; b < kLines; ++b) { both.push_back(b); (b < 24 ? upper : lower).push_back(b); }
        auto Sl_up = build_inv_sigma(ns.line_rows, upper);
        auto Sl_lo = build_inv_sigma(ns.line_rows, lower);
        auto Sl_both = build_inv_sigma(ns.line_rows, both);

        // ---- 감도 ----
        auto all_shots = [](const Shot&) { return true; };
        Fit Jp_all = fit_sensitivity(c, all_shots, false, 9, 9);
        Fit Jl_all = fit_sensitivity(c, all_shots, true, 9, 9);
        Fit Jp_small = fit_sensitivity(c, all_shots, false, 0.5, 0.5);

        Table sens;
        sens.head = {"point", "x", "y", "region", "selected", "dx/dY px per mm", "dy/dY", "dx/dT px per deg", "dy/dT", "적합 잔차 x px", "적합 잔차 y px"};
        for (int p = 0; p < kPoints; ++p)
            sens.add({std::to_string(p), num(c.points[p].x, 1), num(c.points[p].y, 1), std::to_string(c.points[p].region),
                      std::to_string(c.points[p].selected), num(Jp_all.J.at<double>(p * 2, 0), 4), num(Jp_all.J.at<double>(p * 2 + 1, 0), 4),
                      num(Jp_all.J.at<double>(p * 2, 1), 4), num(Jp_all.J.at<double>(p * 2 + 1, 1), 4),
                      num(Jp_all.resid[p * 2], 4), num(Jp_all.resid[p * 2 + 1], 4)});
        sens.csv(results / "sensitivity.csv");

        // 선형성: 전체 범위 대 좁은 범위
        Table lin;
        lin.head = {"항목", "전체 범위 (|Y|<=2, |T|<=1)", "좁은 범위 (|Y|<=0.5, |T|<=0.5)", "차이 %"};
        auto col_mean = [](const cv::Mat& J, int col) {
            double s = 0;
            for (int r = 0; r < J.rows; r += 2) s += J.at<double>(r, col);   // x 성분만
            return s / (J.rows / 2);
        };
        double a1 = col_mean(Jp_all.J, 0), a2 = col_mean(Jp_small.J, 0);
        double b1 = col_mean(Jp_all.J, 1), b2 = col_mean(Jp_small.J, 1);
        lin.add({"판 dx 감도 (px per Y mm)", num(a1, 4), num(a2, 4), num(100 * (a2 - a1) / a1, 2)});
        lin.add({"판 dx 감도 (px per T deg)", num(b1, 4), num(b2, 4), num(100 * (b2 - b1) / b1, 2)});
        lin.add({"팀 기준값", num(kPxPerMmY, 4) + " / " + num(kPxPerT, 4), "", ""});
        lin.add({"적합 잔차 중앙값 (px)", num(median(Jp_all.resid), 5), num(median(Jp_small.resid), 5), ""});
        lin.csv(results / "linearity.csv");

        // ---- 관측 조건별 정보 행렬 ----
        auto sub_rows = [&](const cv::Mat& J, const std::vector<int>& dims) {
            cv::Mat out(int(dims.size()), 2, CV_64F);
            for (size_t i = 0; i < dims.size(); ++i) { out.at<double>(int(i), 0) = J.at<double>(dims[i], 0); out.at<double>(int(i), 1) = J.at<double>(dims[i], 1); }
            return out;
        };
        cv::Mat Jp_sel = sub_rows(Jp_all.J, sel_dims);
        cv::Mat Jl_up = sub_rows(Jl_all.J, upper), Jl_lo = sub_rows(Jl_all.J, lower), Jl_bo = sub_rows(Jl_all.J, both);

        Table ft;
        ft.head = {"관측 조건", "F11 (Y)", "F22 (T)", "sigma_Y mm", "sigma_T deg", "sigma_Y / 0.045", "sigma_T / 0.045",
                   "약한 방향 sigma", "약한 방향 각도 deg", "조건수"};
        std::vector<std::pair<std::string, Info>> conds;
        conds.push_back({"QR만 (536점)", information(Jp_sel, Sp_sel, nullptr, nullptr)});
        conds.push_back({"QR + 위 선", information(Jp_sel, Sp_sel, &Jl_up, &Sl_up)});
        conds.push_back({"QR + 아래 선", information(Jp_sel, Sp_sel, &Jl_lo, &Sl_lo)});
        conds.push_back({"QR + 양쪽 선", information(Jp_sel, Sp_sel, &Jl_bo, &Sl_both)});
        for (auto& k : conds) {
            auto& f = k.second;
            ft.add({k.first, num(f.F(0, 0), 2), num(f.F(1, 1), 2), num(f.sy, 5), num(f.st, 5),
                    num(f.sy / kTol, 3), num(f.st / kTol, 3), num(f.weak_sigma, 5), num(f.weak_angle, 2), num(f.cond, 1)});
        }
        // 대각 근사와 비교 (얼마나 과대평가되는지)
        {
            InvSigma diag;
            diag.D = Sp_sel.D;
            diag.k = 1;
            diag.U = cv::Mat::zeros(diag.D, 1, CV_64F);
            diag.lam.assign(1, 1);
            std::vector<double> dv(diag.D);
            for (int j = 0; j < diag.D; ++j) {
                double s = 0;
                for (int i = 0; i < ns.point_rows.rows; ++i) {
                    double v = ns.point_rows.at<double>(i, sel_dims[j]);
                    s += v * v;
                }
                dv[j] = s / std::max(1, ns.point_rows.rows - 1);
            }
            diag.floor_var = median(dv);
            diag.lam[0] = diag.floor_var;
            auto f = information(Jp_sel, diag, nullptr, nullptr);
            ft.add({"QR만, 대각 근사 (참고)", num(f.F(0, 0), 2), num(f.F(1, 1), 2), num(f.sy, 5), num(f.st, 5),
                    num(f.sy / kTol, 3), num(f.st / kTol, 3), num(f.weak_sigma, 5), num(f.weak_angle, 2), num(f.cond, 1)});
        }
        // 아핀을 빼지 않은 잔차로도 낸다. 그 잔차에는 반복 촬영 사이의 실제 자세 차이가 들어 있다.
        // 아핀 제거는 J 가 사는 부분공간(판 전체 평행이동과 1차항)을 지우므로, 위의 값은 낙관적인 하한이다.
        Info raw_both;
        {
            auto nr = measure_noise(c, [](const Shot&) { return true; }, false);
            auto Spr = build_inv_sigma(nr.point_rows, sel_dims);
            auto Slr = build_inv_sigma(nr.line_rows, both);
            auto f1 = information(Jp_sel, Spr, nullptr, nullptr);
            raw_both = information(Jp_sel, Spr, &Jl_bo, &Slr);
            ft.add({"QR만, raw 잔차 (자세 차이 포함)", num(f1.F(0, 0), 2), num(f1.F(1, 1), 2), num(f1.sy, 5), num(f1.st, 5),
                    num(f1.sy / kTol, 3), num(f1.st / kTol, 3), num(f1.weak_sigma, 5), num(f1.weak_angle, 2), num(f1.cond, 1)});
            ft.add({"QR + 양쪽 선, raw 잔차", num(raw_both.F(0, 0), 2), num(raw_both.F(1, 1), 2), num(raw_both.sy, 5),
                    num(raw_both.st, 5), num(raw_both.sy / kTol, 3), num(raw_both.st / kTol, 3), num(raw_both.weak_sigma, 5),
                    num(raw_both.weak_angle, 2), num(raw_both.cond, 1)});
        }
        ft.csv(results / "fisher.csv");

        // ---- H03: 안쪽 대 바깥 ----
        Table io;
        io.head = {"점 집합", "점 수", "sigma_Y mm", "sigma_T deg", "약한 방향 sigma", "약한 방향 각도 deg"};
        {
            std::vector<std::pair<double, int>> rank;
            for (int p = 0; p < kPoints; ++p) {
                if (!c.points[p].selected) continue;
                int q = c.points[p].region;
                cv::Point2d ctr(kQrBox[q].x + kQrBox[q].width / 2.0, kQrBox[q].y + kQrBox[q].height / 2.0);
                rank.push_back({cv::norm(cv::Point2d(c.points[p].x, c.points[p].y) - ctr), p});
            }
            std::sort(rank.begin(), rank.end());
            const int half = int(rank.size()) / 2;
            auto make = [&](int from, int to, const std::string& name) {
                std::vector<int> dims;
                for (int i = from; i < to; ++i) { dims.push_back(rank[i].second * 2); dims.push_back(rank[i].second * 2 + 1); }
                auto S = build_inv_sigma(ns.point_rows, dims);
                auto J = sub_rows(Jp_all.J, dims);
                auto f = information(J, S, nullptr, nullptr);
                io.add({name, std::to_string((to - from)), num(f.sy, 5), num(f.st, 5), num(f.weak_sigma, 5), num(f.weak_angle, 2)});
                return f;
            };
            make(0, half, "안쪽 절반 (QR 중심에 가까움)");
            make(half, int(rank.size()), "바깥 절반 (QR 중심에서 멂)");
        }
        io.csv(results / "inner_outer.csv");

        // ---- Adiv 조건 대조 ----
        // Adiv 1989 번역.md 262~270행의 **9개** 조건을 그대로 옮겼다.
        // (2026-11-02 정정: 처음에 8개로 적고 '인접 흐름 벡터의 잡음 상관'을 빠뜨렸다. 그것이 A1 이 직접 잰 조건이다.)
        Table adiv;
        adiv.head = {"번호", "Adiv 1989 조건", "우리 실측", "해당?"};
        adiv.add({"1", "FOV 가 작다", "f=4492 px(A13) -> 가로 46.3도, 세로 26.9도. Adiv 예시는 60도", "해당"});
        adiv.add({"2", "국소성 인자 gamma 가 작다", "QR만 gamma_x 0.18 / gamma_y 0.09. **선을 더하면 0.77 / 0.53**. Adiv 예시는 1, 1/2, 1/4", "**강하게 해당 (선 없을 때)**"});
        adiv.add({"3", "평면이 기껏해야 적당히 기울어 있다", "QR 판이 카메라에 거의 정면", "해당"});
        adiv.add({"4", "물체가 멀리 있다", "Z=315 mm(A13). 병진/거리 비 r/d <= 2/315 = 0.006. Adiv 예시는 0.1", "**강하게 해당**"});
        adiv.add({"5", "평행이동의 절대값이 작다", "최소 명령 Y 0.05 mm = Z 의 0.016%", "**강하게 해당**"});
        adiv.add({"6", "영상 해상도가 낮다", "3840x2160. Adiv 예시는 128x128", "해당 아님 (유리)"});
        adiv.add({"7", "화소 단위 잡음이 높다", "점 측정 잡음 0.0087 px (A1)", "해당 아님 (유리)"});
        adiv.add({"8", "흐름장이 성기다", "536점이 영상 면적의 약 2%에 몰려 있다", "해당"});
        adiv.add({"9", "인접 흐름 벡터의 잡음이 강하게 상관", "**N_eff 18.9 / 536 = 3.5%** (A1)", "**강하게 해당**"});
        adiv.csv(results / "adiv.csv");

        // ---- 그림 ----
        cv::Mat center = hye::imread_u8((fs::u8path(REPO_DIR) / "data/clean/clean_20261014/center.bmp").u8string(), cv::IMREAD_GRAYSCALE);
        cv::Rect roi(kQrBox[0].x - 20, kQrBox[0].y - 30, kQrBox[2].x + kQrBox[2].width - kQrBox[0].x + 40, kQrBox[0].height + 60);
        {  // 21 감도장: Y 열과 T 열을 각각, 평균을 뺀 것도 함께
            std::vector<cv::Mat> tiles;
            for (int col = 0; col < 2; ++col) {
                cv::Point2d m(0, 0);
                for (int p = 0; p < kPoints; ++p) m += cv::Point2d(Jp_all.J.at<double>(p * 2, col), Jp_all.J.at<double>(p * 2 + 1, col));
                m *= 1.0 / kPoints;
                double big = 0;
                for (int p = 0; p < kPoints; ++p)
                    big = std::max(big, cv::norm(cv::Point2d(Jp_all.J.at<double>(p * 2, col), Jp_all.J.at<double>(p * 2 + 1, col)) - m));
                double g = big > 1e-12 ? 40 / big : 1;
                std::vector<hye::Mark> marks;
                for (int p = 0; p < kPoints; p += 7) {
                    cv::Point2d j(Jp_all.J.at<double>(p * 2, col), Jp_all.J.at<double>(p * 2 + 1, col));
                    marks.push_back({{c.points[p].x, c.points[p].y}, j - m, "", col ? hye::color(1) : hye::color(0)});
                }
                char cap[220];
                std::snprintf(cap, sizeof cap, "A2-21 sensitivity to %s minus its mean (%.3f,%.3f)  structure %.4f px per unit  arrow x%.0f",
                              col ? "T [px/deg]" : "Y [px/mm]", m.x, m.y, big, g);
                tiles.push_back(hye::zoom(center, roi, 1500, marks, g, cap));
            }
            hye::save_png("21_감도장_Y와T.png", hye::vstack(tiles));
        }
        {  // 22 오차 타원
            hye::Chart ch("A2-22 predicted error ellipse (1 sigma) in (Y,T)", "Y error [mm]", "T error [deg]");
            for (size_t k = 0; k < conds.size(); ++k) {
                auto& f = conds[k].second;
                std::vector<double> x, y;
                cv::Mat ev, evec;
                cv::eigen(cv::Mat(f.Finv), ev, evec);
                for (int i = 0; i <= 120; ++i) {
                    double a = i * 2 * CV_PI / 120;
                    double u = std::sqrt(std::max(0.0, ev.at<double>(0))) * std::cos(a);
                    double v = std::sqrt(std::max(0.0, ev.at<double>(1))) * std::sin(a);
                    x.push_back(u * evec.at<double>(0, 0) + v * evec.at<double>(1, 0));
                    y.push_back(u * evec.at<double>(0, 1) + v * evec.at<double>(1, 1));
                }
                ch.line(x, y, hye::color(int(k)), conds[k].first);
            }
            {  // dY = -5.617 dT 선
                double s = conds[0].second.weak_sigma * 1.5;
                std::vector<double> x = {-kNullDir.x * s, kNullDir.x * s}, y = {-kNullDir.y * s, kNullDir.y * s};
                ch.line(x, y, hye::kGray, "dY = -5.617 dT");
            }
            ch.equal_axes = true;
            ch.save("22_오차타원.png");
        }
        {  // 23 조건별 sigma (허용오차 단위)
            std::vector<std::string> cats;
            std::vector<double> vals;
            for (auto& k : conds) { cats.push_back(k.first + " sY"); vals.push_back(k.second.sy / kTol); }
            for (auto& k : conds) { cats.push_back(k.first + " sT"); vals.push_back(k.second.st / kTol); }
            hye::bar_chart("23_조건별시그마.png", "A2-23 predicted sigma in tolerance units (1.0 = +-0.045)", cats, vals, "sigma / 0.045", -1, 1.0 / 3);
        }
        {  // 24 안쪽/바깥
            std::vector<std::string> cats;
            std::vector<double> vals;
            for (auto& r : io.rows) { cats.push_back(r[0] + " weak"); vals.push_back(std::stod(r[4])); }
            hye::bar_chart("24_안쪽바깥.png", "A2-24 weak-direction sigma: inner vs outer points", cats, vals, "sigma");
        }
        {  // 25 선형성
            hye::Chart ch("A2-25 commanded vs measured plate translation (1014)", "command", "plate dx [px]");
            for (int k = 0; k < 2; ++k) {
                std::vector<double> x, y;
                for (auto& s : c.shots) {
                    if (s.set != "clean_20261014" || !s.measured) continue;
                    bool isY = std::abs(s.t) < 1e-9;
                    if (isY != (k == 0)) continue;
                    double m = 0;
                    int cnt = 0;
                    for (int p = 0; p < kPoints; ++p) if (c.points[p].selected) { m += c.dz(s.idx, p).x; ++cnt; }
                    x.push_back(k ? s.t : s.y);
                    y.push_back(cnt ? m / cnt : 0);
                }
                ch.scatter(x, y, hye::color(k), k ? "T [deg]" : "Y [mm]", 3);
            }
            ch.save("25_선형성.png");
        }

        // ---- 결과.md ----
        auto& qr = conds[0].second;
        auto& both_c = conds[3].second;
        bool info_short = both_c.weak_sigma > kTight;
        std::ostringstream o;
        o << "# A2 결과 — 정보 행렬과 약한 방향\n\n"
          << "`실험/src/main.cpp` 가 다시 쓴다. 해석은 [보고서](보고서.md). 계획은 [계획서](../계획서.md).\n\n"
          << "허용 절대오차 ±" << num(kTol, 3) << ", 판정용 '작다' 경계 " << num(kTight, 3) << ".\n"
          << "잡음 표본 " << ns.shots << "장, 점 공분산의 표본 고유방향 " << Sp_sel.k << "개 (차원 " << Sp_sel.D << ").\n\n"
          << "## 1. 감도 J 와 선형성\n\n" << lin.md() << "\n"
          << hye::img_md("25_선형성.png", "linearity") << "\n\n"
          << "점별 감도: [sensitivity.csv](results/sensitivity.csv)\n\n"
          << "## 2. Y 와 T 는 무엇이 다른가\n\n"
          << hye::img_md("21_감도장_Y와T.png", "sensitivity field") << "\n\n"
          << "두 그림 모두 **평균(판 전체 평행이동)을 뺀 나머지**다. 평균은 캡션에 적었다. "
          << "평행이동만 보면 두 축은 한 숫자로 합쳐지므로, 구별은 이 나머지 구조에서만 나온다.\n\n"
          << "## 3. 관측 조건별 정보 행렬\n\n" << ft.md() << "\n"
          << hye::img_md("23_조건별시그마.png", "sigma by condition") << "\n\n"
          << hye::img_md("22_오차타원.png", "error ellipse") << "\n\n"
          << "표의 마지막 두 줄은 **아핀을 빼지 않은 잔차**로 낸 것이다. 아핀 제거는 J 가 사는 부분공간(판 전체 평행이동과 1차항)을 "
          << "같이 지우므로, 위쪽 값들은 낙관적인 하한이다. raw 잔차에는 반복 촬영 사이에 판이 실제로 움직인 양(A1: dx SD 0.085 px)이 "
          << "들어 있어, 측정만의 하한이 아니라 **같은 명령을 반복했을 때 실제로 재현되는 범위**에 가깝다.\n\n"
          << "## 4. H03 안쪽 대 바깥\n\n" << io.md() << "\n"
          << hye::img_md("24_안쪽바깥.png", "inner vs outer") << "\n\n"
          << "## 5. Adiv 조건 대조\n\n" << adiv.md() << "\n"
          << "Adiv 1989 번역.md 262~270행의 **9개** 조건이다. 9개 중 **7개 해당**, 2개(해상도, 잡음)는 우리에게 유리하다.\n"
          << "(2026-11-02 정정: 처음에 8개로 적고 '인접 흐름 벡터의 잡음 상관'을 빠뜨렸다. 그것이 A1 이 직접 잰 조건이다.)\n\n"
          << "**2번 국소성 인자가 핵심이다.** Adiv 가 시험한 가장 나쁜 경우가 gamma = 1/4 인데, "
          << "QR 만 쓰면 gamma 가 0.09~0.18 로 **그보다 더 나쁘다**. 허용된 위·아래 가로선을 더하면 0.53~0.77 로 올라가 "
          << "Adiv 의 가장 좋은 경우(gamma = 1)에 가까워진다. A3 에서 선을 빼니 통과율이 95.7%에서 24.6%로 무너진 것이 이 수치로 설명된다.\n\n"
          << "## 6. 판정 기준 대비\n\n"
          << "| 조건 | 실측 | 판정 |\n|---|---|---|\n"
          << "| 약한 방향이 dY = -5.617 dT 와 10도 이내 | QR만 " << num(qr.weak_angle, 2) << "도, 양쪽 선 "
          << num(both_c.weak_angle, 2) << "도 | " << (both_c.weak_angle < 10 ? "**맞다**" : "다르다") << " |\n"
          << "| 약한 방향 sigma > " << num(kTight, 3) << " (정보 부족) | 양쪽 선 " << num(both_c.weak_sigma, 5) << " | "
          << (info_short ? "**정보 부족 = H01 성립**" : "정보는 충분. 편향 쪽을 먼저 본다") << " |\n"
          << "| sigma_Y, sigma_T 가 허용오차 대비 | Y " << num(both_c.sy / kTol, 3) << ", T " << num(both_c.st / kTol, 3) << " | |\n"
          << "| 같은 판정을 raw 잔차로 | 약한 방향 " << num(raw_both.weak_sigma, 5) << ", Y " << num(raw_both.sy / kTol, 3)
          << ", T " << num(raw_both.st / kTol, 3) << " | " << (raw_both.weak_sigma > kTight ? "**정보 부족**" : "충분") << " |\n";
        write_text(result / fs::u8path(u8"결과.md"), o.str());
        std::cout << "끝\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "오류: " << e.what() << "\n";
        return 1;
    }
}
