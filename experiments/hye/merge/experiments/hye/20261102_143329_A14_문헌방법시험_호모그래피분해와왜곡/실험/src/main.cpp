// A14: A13 이 뽑은 K 로 문헌 방법을 실제로 세운다.
//  1부 호모그래피 분해 (Malis 1999 2.5D, Collins 2014 IPPE): 판은 평면이므로 center -> 현재 호모그래피를
//       K 로 분해해 회전 R 과 평면거리 대비 병진 t/d 를 얻는다. 회귀 학습 없이 (Y, T) 가 나와야 한다.
//  2부 평면 자세 이중해 (H02): 분해가 주는 두 후보의 비용 차를 본다.
//  3부 방사 왜곡 (H07): k1 을 넣고 A13 물리 모형 잔차가 줄어드는지, u 의 y^2 항이 설명되는지 본다.
#include "cache.hpp"
#include "noise.hpp"
#include "model.hpp"
#include "hye/plot.hpp"
#include <functional>
#include <iostream>
#include <opencv2/calib3d.hpp>

using namespace a0;
namespace {

// A13 결과 (QR 536점 + 선 48블록 제약 모형)
const double kF = 4492.4, kCx = 1919.0, kCy = 1097.0, kZ = 315.3;
const double kDegPerT = 1.0187;   // T 명령 1단위의 물리 각도

cv::Mat K() {
    return (cv::Mat_<double>(3, 3) << kF, 0, kCx, 0, kF, kCy, 0, 0, 1);
}

// 방사 왜곡 보정: 왜곡된 좌표 -> 보정 좌표 (1차항만)
cv::Point2d undistort1(const cv::Point2d& p, double k1) {
    double x = (p.x - kCx) / kF, y = (p.y - kCy) / kF;
    double r2 = x * x + y * y;
    double s = 1 + k1 * r2;
    return {kCx + x * s * kF, kCy + y * s * kF};
}

struct Pose {
    double y = 0, t = 0;        // 채택 후보
    double y2 = 0, t2 = 0;      // 두 번째 후보
    double cost = 0, cost2 = 0;
    int n_sol = 0;
    bool ok = false;
};

// center -> 현재 호모그래피를 K 로 분해해 (Y mm, T deg) 를 낸다. 학습 없음.
Pose decompose(const Cache& c, int shot, double k1) {
    std::vector<cv::Point2f> ref, cur;
    for (int p = 0; p < kPoints; ++p) {
        if (!c.points[p].selected || !c.v(shot)[p]) continue;
        cv::Point2d a(c.points[p].x, c.points[p].y);
        auto d = c.dz(shot, p);
        cv::Point2d b(a.x + d.x, a.y + d.y);
        if (k1 != 0) { a = undistort1(a, k1); b = undistort1(b, k1); }
        ref.emplace_back(float(a.x), float(a.y));
        cur.emplace_back(float(b.x), float(b.y));
    }
    Pose out;
    if (ref.size() < 20) return out;
    cv::Mat H = cv::findHomography(ref, cur, 0);   // 전체 점 최소제곱
    if (H.empty()) return out;
    std::vector<cv::Mat> Rs, ts, ns;
    int n = cv::decomposeHomographyMat(H, K(), Rs, ts, ns);
    if (n <= 0) return out;
    out.n_sol = n;
    // 평면이 거의 정면이므로 법선이 광축에 가까운 해만 쓴다
    struct Cand { double y, t, score; };
    std::vector<Cand> cs;
    for (int i = 0; i < n; ++i) {
        double nz = std::abs(ns[i].at<double>(2));
        if (nz < 0.9) continue;
        cv::Mat R = Rs[i];
        // 세로축(카메라 Y축) 회전각: R 의 (0,2), (2,2) 성분
        double ang = std::atan2(R.at<double>(0, 2), R.at<double>(2, 2));
        double tx = ts[i].at<double>(0) * kZ;   // t 는 평면거리 d 로 정규화되어 있다
        cs.push_back({tx, ang * 180 / CV_PI, nz});
    }
    if (cs.empty()) return out;
    std::sort(cs.begin(), cs.end(), [](const Cand& a, const Cand& b) { return a.score > b.score; });
    out.ok = true;
    out.y = cs[0].y;
    out.t = cs[0].t;
    out.cost = cs[0].score;
    if (cs.size() > 1) { out.y2 = cs[1].y; out.t2 = cs[1].t; out.cost2 = cs[1].score; }
    return out;
}

}  // namespace

int main() {
    try {
        const fs::path repo = fs::u8path(REPO_DIR);
        const fs::path result = fs::u8path(EXPERIMENT_DIR);
        const fs::path results = result / "results";
        fs::create_directories(results);
        hye::begin_results();
        Cache c = load(cache_dir());

        // ---------- 1부: 호모그래피 분해 ----------
        std::vector<Pose> P(c.n());
        int ok = 0;
        for (auto& s : c.shots) {
            if (!s.measured) continue;
            P[s.idx] = decompose(c, s.idx, 0);
            if (P[s.idx].ok) ++ok;
        }
        std::cout << "호모그래피 분해 성공 " << ok << "/" << c.n() << "\n";

        // 물리 예측과의 관계: 학습 없이 바로 쓴 값의 기울기를 본다. 모형이 맞으면 1 에 가깝다.
        auto slope = [&](std::function<double(const Shot&)> truth, std::function<double(const Pose&)> est,
                         std::function<bool(const Shot&)> keep) {
            std::vector<double> X, Y;
            for (auto& s : c.shots) {
                if (!s.measured || !P[s.idx].ok || !keep(s)) continue;
                X.push_back(truth(s));
                Y.push_back(est(P[s.idx]));
            }
            double mx = mean(X), my = mean(Y), sxy = 0, sxx = 0;
            for (size_t i = 0; i < X.size(); ++i) { sxy += (X[i] - mx) * (Y[i] - my); sxx += (X[i] - mx) * (X[i] - mx); }
            double a = sxx > 0 ? sxy / sxx : 0, b = my - a * mx;
            std::vector<double> r;
            for (size_t i = 0; i < X.size(); ++i) r.push_back(Y[i] - (a * X[i] + b));
            return std::make_tuple(a, b, rms(r), int(X.size()));
        };
        Table sl;
        sl.head = {"대응", "자료", "기울기 (모형이 맞으면 1)", "절편", "잔차 RMS", "장수"};
        {
            auto in_range = [](const Shot& s) { return s.t <= 1.001; };
            auto yonly = [](const Shot& s) { return s.set == "clean_20261014" && std::abs(s.t) < 1e-9; };
            auto tonly = [](const Shot& s) { return s.set == "clean_20261014" && std::abs(s.y) < 1e-9; };
            auto g1 = slope([](const Shot& s) { return s.y; }, [](const Pose& p) { return p.y; }, yonly);
            sl.add({"명령 Y (mm) -> 분해 t_x*Z (mm)", "1014 Y 단독 80장", num(std::get<0>(g1), 4), num(std::get<1>(g1), 4),
                    num(std::get<2>(g1), 4), std::to_string(std::get<3>(g1))});
            auto g2 = slope([](const Shot& s) { return s.t * kDegPerT; }, [](const Pose& p) { return p.t; }, tonly);
            sl.add({"명령 T (deg) -> 분해 회전각 (deg)", "1014 T 단독 200장", num(std::get<0>(g2), 4), num(std::get<1>(g2), 4),
                    num(std::get<2>(g2), 4), std::to_string(std::get<3>(g2))});
            auto g3 = slope([](const Shot& s) { return s.y; }, [](const Pose& p) { return p.y; }, in_range);
            sl.add({"명령 Y -> 분해 t_x*Z (전체)", "범위 안 전체", num(std::get<0>(g3), 4), num(std::get<1>(g3), 4),
                    num(std::get<2>(g3), 4), std::to_string(std::get<3>(g3))});
            auto g4 = slope([](const Shot& s) { return s.t * kDegPerT; }, [](const Pose& p) { return p.t; }, in_range);
            sl.add({"명령 T (deg) -> 분해 회전각 (전체)", "범위 안 전체", num(std::get<0>(g4), 4), num(std::get<1>(g4), 4),
                    num(std::get<2>(g4), 4), std::to_string(std::get<3>(g4))});
        }
        sl.csv(results / "slope_vs_command.csv");

        // 성적: 분해값을 그대로 (부호·척도만 전체에서 한 번 맞춘 뒤) 썼을 때
        Table sc;
        sc.head = {"방법", "학습", "장수", "Y 평균", "Y 최대", "T 평균", "T 최대", "통과 ±0.045"};
        {
            // 전체에서 척도 a, 절편 b 를 한 번만 구한다 (fold 별 재학습이 아니다. 자유도 2개)
            auto in_range = [](const Shot& s) { return s.t <= 1.001; };
            auto gy = slope([](const Shot& s) { return s.y; }, [](const Pose& p) { return p.y; }, in_range);
            auto gt = slope([](const Shot& s) { return s.t; }, [](const Pose& p) { return p.t; }, in_range);
            std::vector<double> ey, et;
            int pass = 0, n2 = 0;
            Table pr;
            pr.head = {"shot", "set", "rel", "Y", "T", "분해Y", "분해T", "eY", "eT", "후보수", "2후보 Y", "2후보 T"};
            for (auto& s : c.shots) {
                if (!s.measured || !P[s.idx].ok || !in_range(s)) continue;
                double py = (P[s.idx].y - std::get<1>(gy)) / std::get<0>(gy);
                double pt = (P[s.idx].t - std::get<1>(gt)) / std::get<0>(gt);
                ey.push_back(py - s.y);
                et.push_back(pt - s.t);
                ++n2;
                if (std::abs(py - s.y) <= kTol && std::abs(pt - s.t) <= kTol) ++pass;
                pr.add({std::to_string(s.idx), s.set.substr(6), s.rel, num(s.y, 3), num(s.t, 3), num(py, 4), num(pt, 4),
                        num(py - s.y, 4), num(pt - s.t, 4), std::to_string(P[s.idx].n_sol), num(P[s.idx].y2, 4),
                        num(P[s.idx].t2, 4)});
            }
            pr.csv(results / "homography_predictions.csv");
            std::vector<double> ay, at;
            for (double v : ey) ay.push_back(std::abs(v));
            for (double v : et) at.push_back(std::abs(v));
            sc.add({"호모그래피 분해 (Malis/IPPE 계열)", "척도·절편 2개만", std::to_string(n2), num(mean(ay), 4),
                    num(max_abs(ey), 4), num(mean(at), 4), num(max_abs(et), 4), num(double(pass) / n2, 4)});
        }
        // 비교: A3 기준 모델 OOF
        {
            std::vector<std::vector<double>> f66(c.n());
            for (auto& s : c.shots) if (s.measured) f66[s.idx] = features(c, s.idx, true);
            auto oof = run_oof(c, f66, [](const Shot& s) { return s.fold_mixed; }, 4, 0);
            std::vector<unsigned char> keep(oof.shot.size(), 1);
            for (size_t i = 0; i < oof.shot.size(); ++i) if (oof.truth[i].y > 1.001) keep[i] = 0;
            auto s2 = score(oof, &keep);
            sc.add({"기준 모델 66d (대조)", "fold 마다 66x2 계수", std::to_string(s2.n), num(s2.mean_y, 4),
                    num(s2.max_y, 4), num(s2.mean_t, 4), num(s2.max_t, 4), num(s2.pass, 4)});
        }
        sc.csv(results / "scores.csv");

        // ---------- 2부: 평면 자세 이중해 (H02) ----------
        Table dd;
        dd.head = {"항목", "값"};
        {
            int multi = 0, tot = 0;
            std::vector<double> gap;
            for (auto& s : c.shots) {
                if (!s.measured || !P[s.idx].ok || s.t > 1.001) continue;
                ++tot;
                if (P[s.idx].cost2 > 0) { ++multi; gap.push_back(std::abs(P[s.idx].y - P[s.idx].y2)); }
            }
            dd.add({"분해 후보가 2개 이상인 사진", std::to_string(multi) + " / " + std::to_string(tot)});
            dd.add({"두 후보의 Y 차이 중앙값 (mm)", gap.empty() ? "" : num(median(gap), 4)});
            dd.add({"두 후보의 Y 차이 최대 (mm)", gap.empty() ? "" : num(max_abs(gap), 4)});
        }
        dd.csv(results / "double_solution.csv");

        // ---------- 3부: 방사 왜곡 (H07) ----------
        // A13 물리 모형을 k1 별로 다시 맞춰 잔차가 줄어드는지 본다.
        std::vector<int> ids;
        for (auto& s : c.shots) if (s.measured && s.set == "clean_20261014") ids.push_back(s.idx);
        auto fit_resid = [&](double k1) {
            const int n = int(ids.size());
            cv::Mat X(n, 3, CV_64F), Yp(n, kPoints * 2, CV_64F);
            for (int i = 0; i < n; ++i) {
                X.at<double>(i, 0) = c.shots[ids[i]].y;
                X.at<double>(i, 1) = c.shots[ids[i]].t;
                X.at<double>(i, 2) = 1;
                for (int p = 0; p < kPoints; ++p) {
                    cv::Point2d a(c.points[p].x, c.points[p].y);
                    auto d = c.dz(ids[i], p);
                    cv::Point2d b(a.x + d.x, a.y + d.y);
                    if (k1 != 0) { a = undistort1(a, k1); b = undistort1(b, k1); }
                    Yp.at<double>(i, p * 2) = b.x - a.x;
                    Yp.at<double>(i, p * 2 + 1) = b.y - a.y;
                }
            }
            cv::Mat B;
            cv::solve(X, Yp, B, cv::DECOMP_SVD);
            // 제약 모형 u = A + C x^2, v = C x y 를 선택점에 맞춘 잔차
            double s11 = 0, s12 = 0, s22 = 0, b1 = 0, b2 = 0;
            std::vector<std::pair<double, double>> xy;
            std::vector<std::pair<double, double>> uv;
            for (int p = 0; p < kPoints; ++p) {
                if (!c.points[p].selected) continue;
                cv::Point2d a = undistort1({c.points[p].x, c.points[p].y}, k1);
                double x = a.x - kCx, y = a.y - kCy;
                double u = B.at<double>(1, p * 2), v = B.at<double>(1, p * 2 + 1);
                double g = x * x, h = x * y;
                s11 += 1; s12 += g; s22 += g * g + h * h;
                b1 += u; b2 += g * u + h * v;
                xy.push_back({x, y});
                uv.push_back({u, v});
            }
            double det = s11 * s22 - s12 * s12;
            double A = (b1 * s22 - b2 * s12) / det, C = (s11 * b2 - s12 * b1) / det;
            double rss = 0;
            int n2 = 0;
            for (size_t i = 0; i < xy.size(); ++i) {
                double r1 = uv[i].first - (A + C * xy[i].first * xy[i].first);
                double r2 = uv[i].second - C * xy[i].first * xy[i].second;
                rss += r1 * r1 + r2 * r2;
                n2 += 2;
            }
            // u 의 y^2 항 (제약 없는 2차 적합에서)
            cv::Mat Aq(int(xy.size()), 6, CV_64F), Bq(int(xy.size()), 1, CV_64F), Cq;
            for (size_t i = 0; i < xy.size(); ++i) {
                double x = xy[i].first / 1000, y = xy[i].second / 1000;
                double bb[] = {1, x, y, x * x, x * y, y * y};
                for (int j = 0; j < 6; ++j) Aq.at<double>(int(i), j) = bb[j];
                Bq.at<double>(int(i)) = uv[i].first;
            }
            cv::solve(Aq, Bq, Cq, cv::DECOMP_SVD);
            struct R { double rms, A, C, y2, f; };
            return R{std::sqrt(rss / n2), A, C, Cq.at<double>(5), std::sqrt(std::abs(A / C))};
        };
        Table kt;
        kt.head = {"k1", "제약 모형 잔차 RMS (px)", "A (px/T)", "C", "u 의 y^2 계수", "유도 f (px)"};
        double best_k = 0, best_r = 1e300;
        for (double k1 : {-0.20, -0.15, -0.10, -0.05, -0.02, 0.0, 0.02, 0.05, 0.10, 0.15, 0.20}) {
            auto r = fit_resid(k1);
            kt.add({num(k1, 3), num(r.rms, 5), num(r.A, 4), num(r.C, 9), num(r.y2, 5), num(r.f, 1)});
            if (r.rms < best_r) { best_r = r.rms; best_k = k1; }
        }
        // 최적 근처 세분
        for (double k1 = best_k - 0.02; k1 <= best_k + 0.02 + 1e-9; k1 += 0.005) {
            auto r = fit_resid(k1);
            if (r.rms < best_r) { best_r = r.rms; best_k = k1; }
        }
        auto rb = fit_resid(best_k);
        kt.add({"최적 " + num(best_k, 4), num(rb.rms, 5), num(rb.A, 4), num(rb.C, 9), num(rb.y2, 5), num(rb.f, 1)});
        kt.csv(results / "distortion.csv");

        // ---------- 그림 ----------
        {
            hye::Chart ch("A14-141 homography decomposition vs command (Y)", "command Y [mm]", "decomposed t_x*Z [mm]");
            std::vector<double> x, y;
            for (auto& s : c.shots)
                if (s.measured && P[s.idx].ok && s.set == "clean_20261014" && std::abs(s.t) < 1e-9) { x.push_back(s.y); y.push_back(P[s.idx].y); }
            ch.scatter(x, y, hye::color(0), "", 3);
            ch.save("141_분해Y.png");
        }
        {
            hye::Chart ch("A14-142 homography decomposition vs command (T)", "command T x 1.0187 [deg]", "decomposed angle [deg]");
            std::vector<double> x, y;
            for (auto& s : c.shots)
                if (s.measured && P[s.idx].ok && s.set == "clean_20261014" && std::abs(s.y) < 1e-9) { x.push_back(s.t * kDegPerT); y.push_back(P[s.idx].t); }
            ch.scatter(x, y, hye::color(1), "", 3);
            ch.save("142_분해T.png");
        }
        {
            hye::Chart ch("A14-143 physical-model residual vs radial distortion k1", "k1", "residual RMS [px]");
            std::vector<double> x, y;
            for (double k1 = -0.25; k1 <= 0.25; k1 += 0.01) { x.push_back(k1); y.push_back(fit_resid(k1).rms); }
            ch.line(x, y, hye::color(0), "", true);
            ch.vline(best_k, hye::kRed);
            ch.save("143_왜곡.png");
        }

        std::ostringstream o;
        o << "# A14 결과 — A13 의 K 로 문헌 방법을 세운다\n\n"
          << "`실험/src/main.cpp` 가 다시 쓴다. 해석은 [보고서](보고서.md).\n\n"
          << "쓴 K: f = " << num(kF, 1) << " px, 주점 (" << num(kCx, 1) << ", " << num(kCy, 1) << "), 평면 거리 "
          << num(kZ, 1) << " mm, T 1단위 = " << num(kDegPerT, 4) << " deg. 전부 [A13](../../20261102_141140_A13_물리모형자기보정_f와주점식별/결과/보고서.md) 에서 나왔다.\n\n"
          << "## 1. 호모그래피 분해가 명령을 맞히는가\n\n" << sl.md() << "\n"
          << "**기울기가 1 이면 학습 없이 물리만으로 명령값이 복원된다는 뜻이다.**\n\n"
          << hye::img_md("141_분해Y.png", "Y") << "\n\n" << hye::img_md("142_분해T.png", "T") << "\n\n"
          << "## 2. 성적\n\n" << sc.md() << "\n"
          << "호모그래피 분해는 fold 마다 다시 학습하지 않는다. 전체에서 척도와 절편 2개만 맞췄다.\n\n"
          << "## 3. 평면 자세 이중해 (H02)\n\n" << dd.md() << "\n"
          << "## 4. 방사 왜곡 (H07)\n\n" << kt.md() << "\n"
          << hye::img_md("143_왜곡.png", "distortion") << "\n";
        write_text(result / fs::u8path(u8"결과.md"), o.str());
        std::cout << "끝\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "오류: " << e.what() << "\n";
        return 1;
    }
}
