// A5: 부화소 위상 편향 (H08).
//  1014 Y 80장(0.71 px 간격)과 T 200장(0.80 px 간격)에서 QR별 아핀을 뺀 잔차를 부화소 위상으로 묶어
//  1주기 성분의 진폭을 잰다. 이어서 기준 모델의 Y 추정 오차에 같은 주기가 남는지 본다.
#include "cache.hpp"
#include "noise.hpp"
#include "model.hpp"
#include "hye/plot.hpp"
#include <functional>
#include <iostream>

using namespace a0;
namespace {

const int kBin = 20;

struct Harmonic {
    double a1 = 0, b1 = 0, amp1 = 0, amp2 = 0, n = 0;
};

// r = c + a1 sin(2pi p) + b1 cos(2pi p) + a2 sin(4pi p) + b2 cos(4pi p)
Harmonic fit_harmonic(const std::vector<double>& phase, const std::vector<double>& r) {
    Harmonic h;
    const int n = int(r.size());
    if (n < 10) return h;
    cv::Mat A(n, 5, CV_64F), B(n, 1, CV_64F), C;
    for (int i = 0; i < n; ++i) {
        double p = 2 * CV_PI * phase[i];
        A.at<double>(i, 0) = 1;
        A.at<double>(i, 1) = std::sin(p);
        A.at<double>(i, 2) = std::cos(p);
        A.at<double>(i, 3) = std::sin(2 * p);
        A.at<double>(i, 4) = std::cos(2 * p);
        B.at<double>(i) = r[i];
    }
    cv::solve(A, B, C, cv::DECOMP_SVD);
    h.a1 = C.at<double>(1);
    h.b1 = C.at<double>(2);
    h.amp1 = std::hypot(C.at<double>(1), C.at<double>(2));
    h.amp2 = std::hypot(C.at<double>(3), C.at<double>(4));
    h.n = n;
    return h;
}

double frac(double v) {
    double f = v - std::floor(v);
    return f;
}

}  // namespace

int main() {
    try {
        const fs::path result = fs::u8path(EXPERIMENT_DIR);
        const fs::path results = result / "results";
        fs::create_directories(results);
        hye::begin_results();
        Cache c = load(cache_dir());

        // 측정 잡음 (A1 과 같은 값)
        auto ns = measure_noise(c, [](const Shot&) { return true; }, true);
        std::vector<double> pn;
        for (int i = 0; i < ns.point_rows.rows; ++i)
            for (int j = 0; j < ns.point_rows.cols; ++j) pn.push_back(ns.point_rows.at<double>(i, j));
        const double noise = rms(pn);

        // ---- 점 잔차와 위상 ----
        struct Axis { std::string name; std::function<bool(const Shot&)> pick; };
        std::vector<Axis> axes = {
            {"Y 80장 (0.71 px 간격)", [](const Shot& s) { return s.set == "clean_20261014" && std::abs(s.t) < 1e-9 && std::abs(s.y) > 1e-9; }},
            {"T 200장 (0.80 px 간격)", [](const Shot& s) { return s.set == "clean_20261014" && std::abs(s.y) < 1e-9 && std::abs(s.t) > 1e-9; }},
        };

        Table amp;
        amp.head = {"자료", "성분", "표본 수", "1주기 진폭 px", "2주기 진폭 px", "측정 잡음 대비", "잡음 px"};
        Table pb;
        pb.head = {"자료", "성분", "위상칸", "위상 중앙", "평균 잔차 px", "표본 수"};
        std::vector<std::pair<std::string, std::vector<std::pair<double, double>>>> curves;

        for (auto& ax : axes) {
            std::vector<double> phx, rx, phy, ry;
            for (auto& s : c.shots) {
                if (!s.measured || !ax.pick(s)) continue;
                std::vector<unsigned char> use(kPoints, 1);
                for (int p = 0; p < kPoints; ++p) use[p] = c.v(s.idx)[p];
                std::vector<cv::Point2d> r(kPoints);
                for (int p = 0; p < kPoints; ++p) r[p] = c.dz(s.idx, p);
                remove_qr_affine(c.points, r, use);
                for (int p = 0; p < kPoints; ++p) {
                    if (!use[p] || !c.points[p].selected) continue;
                    double gx = c.points[p].x + c.d(s.idx)[p * 2];
                    double gy = c.points[p].y + c.d(s.idx)[p * 2 + 1];
                    phx.push_back(frac(gx));
                    rx.push_back(r[p].x);
                    phy.push_back(frac(gy));
                    ry.push_back(r[p].y);
                }
            }
            struct Comp { std::string name; std::vector<double>* ph; std::vector<double>* r; };
            std::vector<Comp> comps = {{"x", &phx, &rx}, {"y", &phy, &ry}};
            for (auto& cm : comps) {
                auto h = fit_harmonic(*cm.ph, *cm.r);
                amp.add({ax.name, cm.name, num(h.n, 0), num(h.amp1, 5), num(h.amp2, 5),
                         num(h.amp1 / noise, 2), num(noise, 5)});
                // 위상칸 평균
                std::vector<double> sum(kBin, 0), cnt(kBin, 0);
                for (size_t i = 0; i < cm.ph->size(); ++i) {
                    int b = std::min(kBin - 1, int((*cm.ph)[i] * kBin));
                    sum[b] += (*cm.r)[i];
                    cnt[b] += 1;
                }
                std::vector<std::pair<double, double>> cur;
                for (int b = 0; b < kBin; ++b) {
                    double m = cnt[b] ? sum[b] / cnt[b] : 0;
                    pb.add({ax.name, cm.name, std::to_string(b), num((b + 0.5) / kBin, 3), num(m, 5), num(cnt[b], 0)});
                    cur.push_back({(b + 0.5) / kBin, m});
                }
                curves.push_back({ax.name + " " + cm.name, cur});
            }
        }
        amp.csv(results / "amplitude.csv");
        pb.csv(results / "phase_bias.csv");

        // ---- Y 추정 오차의 주기 성분 ----
        std::vector<std::vector<double>> feat(c.n());
        for (auto& s : c.shots) if (s.measured) feat[s.idx] = features(c, s.idx, true);
        auto oof = run_oof(c, feat, [](const Shot& s) { return s.fold_mixed; }, 4, 0);
        std::vector<double> ph_est, e_est, cmd;
        for (size_t i = 0; i < oof.shot.size(); ++i) {
            const Shot& s = c.shots[oof.shot[i]];
            if (s.set != "clean_20261014" || std::abs(s.t) > 1e-9) continue;
            // 판 전체 평균 위상
            double m = 0;
            int n2 = 0;
            for (int p = 0; p < kPoints; ++p)
                if (c.v(s.idx)[p] && c.points[p].selected) { m += c.points[p].x + c.d(s.idx)[p * 2]; ++n2; }
            if (!n2) continue;
            ph_est.push_back(frac(m / n2));
            e_est.push_back(oof.pred[i].x - oof.truth[i].x);
            cmd.push_back(s.y);
        }
        auto he = fit_harmonic(ph_est, e_est);
        bool point_bias = false;
        {
            double a = std::stod(amp.rows[0][3]);
            point_bias = a > 3 * noise;
        }
        bool h08 = he.amp1 > kTight;

        // ---- 그림 ----
        {
            hye::Chart ch("A5-51 point residual vs subpixel phase (mean per bin)", "subpixel phase", "mean residual [px]");
            for (size_t k = 0; k < curves.size(); ++k) {
                std::vector<double> x, y;
                for (auto& p : curves[k].second) { x.push_back(p.first); y.push_back(p.second); }
                ch.line(x, y, hye::color(int(k)), curves[k].first, true);
            }
            ch.hline(noise, hye::kGray);
            ch.hline(-noise, hye::kGray);
            ch.save("51_위상편향_점.png");
        }
        {
            hye::Chart ch("A5-52 Y estimate error vs mean subpixel phase (1014 Y shots)", "subpixel phase", "eY [mm]");
            ch.scatter(ph_est, e_est, hye::color(0), "OOF error", 4);
            std::vector<double> x, y;
            for (int i = 0; i <= 100; ++i) {
                double p = i / 100.0;
                x.push_back(p);
                y.push_back(he.a1 * std::sin(2 * CV_PI * p) + he.b1 * std::cos(2 * CV_PI * p));
            }
            ch.line(x, y, hye::kRed, "1-cycle fit");
            ch.hline(kTight, hye::kGray);
            ch.hline(-kTight, hye::kGray);
            ch.save("52_위상편향_Y추정.png");
        }
        {
            hye::Chart ch("A5-53 Y estimate error vs commanded Y (1014)", "Y command [mm]", "eY [mm]");
            ch.scatter(cmd, e_est, hye::color(1), "", 4);
            ch.hline(kTol, hye::kRed);
            ch.hline(-kTol, hye::kRed);
            ch.save("53_잔차주기_스펙트럼.png");
        }

        std::ostringstream o;
        o << "# A5 결과 — 부화소 위상 편향\n\n"
          << "`실험/src/main.cpp` 가 다시 쓴다. 해석은 [보고서](보고서.md). 계획은 [계획서](../계획서.md).\n\n"
          << "점 측정 잡음 RMS " << num(noise, 5) << " px (A1 과 같은 정의).\n\n"
          << "## 1. 점 잔차의 위상 의존\n\n" << amp.md() << "\n"
          << hye::img_md("51_위상편향_점.png", "phase bias") << "\n\n"
          << "회색 선은 측정 잡음 크기다. 위상칸별 값: [phase_bias.csv](results/phase_bias.csv)\n\n"
          << "## 2. Y 추정에 남는 주기 성분\n\n"
          << "| 값 | 크기 |\n|---|---|\n"
          << "| 1주기 진폭 (mm) | " << num(he.amp1, 5) << " |\n"
          << "| 2주기 진폭 (mm) | " << num(he.amp2, 5) << " |\n"
          << "| 허용오차 ±0.045 대비 | " << num(he.amp1 / kTol, 3) << " |\n"
          << "| 표본 | " << num(he.n, 0) << "장 |\n\n"
          << hye::img_md("52_위상편향_Y추정.png", "Y error vs phase") << "\n\n"
          << hye::img_md("53_잔차주기_스펙트럼.png", "Y error vs command") << "\n\n"
          << "## 3. 판정 기준 대비\n\n"
          << "| 조건 | 실측 | 판정 |\n|---|---|---|\n"
          << "| 점 잔차 1주기 진폭 > 잡음의 3배 | " << amp.rows[0][3] << " 대 " << num(3 * noise, 5) << " | "
          << (point_bias ? "**편향 존재**" : "없음") << " |\n"
          << "| Y 추정 주기 성분 > " << num(kTight, 3) << " mm | " << num(he.amp1, 5) << " | "
          << (h08 ? "**H08 성립**" : "기각. 위상 편향은 주원인이 아니다") << " |\n\n"
          << "## 4. 하지 않은 것\n\n"
          << "보간 차수·창 크기를 바꾼 재측정은 하지 않았다. 2026-10-29의 부화소 정합 비교 실험이 9가지 방법을 "
          << "이미 비교했고 채택한 NCC+LK15+GN스플라인이 가장 좋았다(사진 한 장 잡음 0.015 px). 그 결과를 인용한다.\n";
        write_text(result / fs::u8path(u8"결과.md"), o.str());
        std::cout << "끝\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "오류: " << e.what() << "\n";
        return 1;
    }
}
