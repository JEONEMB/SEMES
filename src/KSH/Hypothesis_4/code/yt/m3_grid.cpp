// =====================================================================================
// m3_grid.cpp — M3 구현 + 검증 (설명은 m3_grid.hpp)
// =====================================================================================
#include "m3_grid.hpp"
#include "m1_phase_ecc.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>

namespace yt {
namespace m3 {

cv::Mat moirePhase(const cv::Mat& img32, int p) {
    const int H = img32.rows, W = img32.cols;
    cv::Mat ph(H, W, CV_64F, cv::Scalar(std::nan("")));
    std::vector<double> S(W), C(W);
    for (int y = 0; y < H; ++y) {
        const float* r = img32.ptr<float>(y);
        std::fill(S.begin(), S.end(), 0.0);
        std::fill(C.begin(), C.end(), 0.0);
        for (int k = 0; k < p; ++k) {
            // 솎아 낸 점 (k, k+p, k+2p, ...) 사이를 선형 보간한 모아레 M_k(x)
            for (int x = 0; x < W; ++x) {
                int n = (x - k) >= 0 ? (x - k) / p : -1;
                const int xa = k + n * p, xb = xa + p;
                if (n < 0 || xb >= W) continue;
                const double t = double(x - xa) / p;
                const double m = (1 - t) * r[xa] + t * r[xb];
                S[x] += m * std::sin(2 * CV_PI * k / p);
                C[x] += m * std::cos(2 * CV_PI * k / p);
            }
        }
        double* o = ph.ptr<double>(y);
        for (int x = p; x < W - p; ++x) o[x] = std::atan2(-S[x], C[x]);
    }
    return ph;
}

MoireResult moireDisplacement(const cv::Mat& ref32, const cv::Mat& cur32, int p) {
    MoireResult r;
    const cv::Mat a = moirePhase(ref32, p), b = moirePhase(cur32, p);
    // 행·열마다 위상 차 (감기 −π..π), 열 평균 → 직선 맞춤 (기울기 = 변형률)
    const int W = a.cols, H = a.rows;
    std::vector<std::vector<double>> A;
    std::vector<double> yv;
    for (int x = 2 * p; x < W - 2 * p; ++x) {
        double sr = 0, si = 0;
        int n = 0;
        for (int y = 0; y < H; ++y) {
            const double pa = a.at<double>(y, x), pb = b.at<double>(y, x);
            if (!std::isfinite(pa) || !std::isfinite(pb)) continue;
            sr += std::cos(pb - pa); si += std::sin(pb - pa); ++n;
        }
        if (n < H / 2) continue;
        const double d = std::atan2(si, sr);
        A.push_back({1.0, double(x - W / 2)});
        yv.push_back(d);
    }
    if (yv.size() < 10) return r;
    const auto c = lstsq(A, yv);
    // 위상이 격자와 같은 방향으로 증가하는 정의 : 영상이 +u 밀리면 위상은 −2πu/p
    r.u = -c[0] * p / (2 * CV_PI);
    r.eps = -c[1] * p / (2 * CV_PI);
    r.ok = true;
    return r;
}

Periodicity periodicity(const cv::Mat& img32) {
    const int W = img32.cols, H = img32.rows;
    cv::Mat win(1, W, CV_64F);
    for (int x = 0; x < W; ++x) win.at<double>(0, x) = 0.5 - 0.5 * std::cos(2 * CV_PI * x / (W - 1));
    std::vector<double> P(W / 2 + 1, 0.0);
    for (int y = 0; y < H; ++y) {
        cv::Mat row, F;
        img32.row(y).convertTo(row, CV_64F);
        row -= cv::mean(row)[0];
        row = row.mul(win);
        cv::dft(row, F, cv::DFT_COMPLEX_OUTPUT);
        for (int k = 0; k <= W / 2; ++k) { const cv::Vec2d c = F.at<cv::Vec2d>(0, k); P[k] += c[0] * c[0] + c[1] * c[1]; }
    }
    // 주기 3 ~ W/4 px 구간
    const int k0 = 4, k1 = W / 3;
    std::vector<double> band(P.begin() + k0, P.begin() + k1 + 1);
    std::vector<double> tmp = band;
    std::nth_element(tmp.begin(), tmp.begin() + tmp.size() / 2, tmp.end());
    const double med = tmp[tmp.size() / 2];
    const auto it = std::max_element(band.begin(), band.end());
    Periodicity p;
    p.score = *it / std::max(med, 1e-12);
    p.period = double(W) / double(k0 + (it - band.begin()));
    return p;
}

// =====================================================================================
// 검증
// =====================================================================================
bool verify(const std::vector<Sample>& ds, const std::string& outDir) {
    Report rep(outDir, "M3 샘플링 모아레 / 격자법 — 검증 결과");
    const int p = 10;
    const int W = 420, H = 100;
    std::mt19937 rng(7);
    std::normal_distribution<double> noise(0.0, 0.7);   // 실측 촬영 잡음 σ ≈ 0.7
    auto grating = [&](double shift, double eps) {
        cv::Mat g(H, W, CV_32F);
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x) {
                const double X = W / 2.0 + (x - W / 2.0 - shift) / (1 + eps);
                g.at<float>(y, x) = float(128 + 60 * std::cos(2 * CV_PI * X / p + 0.3) + noise(rng));
            }
        return g;
    };

    rep.h("V3-1 합성 격자 (피치 10 px, 잡음 σ 0.7) — 알고리즘 구현 확인");
    {
        Csv csv(outDir + "/m3_synthetic_grating.csv");
        csv << "shift_true" << "eps_true" << "u_meas" << "eps_meas";
        csv.endl();
        const cv::Mat ref = grating(0, 0);
        std::vector<double> eu, ee;
        for (int i = 0; i <= 10; ++i) {
            const double sh = 0.1 * i;
            const MoireResult m = moireDisplacement(ref, grating(sh, 0), p);
            csv << sh << 0.0 << m.u << m.eps;
            csv.endl();
            eu.push_back(m.u - sh);
        }
        for (const double eps : {2e-4, 5e-4, 1e-3}) {
            const MoireResult m = moireDisplacement(ref, grating(0.2, eps), p);
            csv << 0.2 << eps << m.u << m.eps;
            csv.endl();
            ee.push_back(m.eps - eps);
        }
        const Stat a = stat(eu), b = stat(ee);
        rep.table({"항목", "RMS", "최대"}, {{"변위 오차 (px)", fmt("%.4f", a.rms), fmt("%.4f", a.maxAbs)},
                                          {"변형률 오차", fmt("%.2e", b.rms), fmt("%.2e", b.maxAbs)}});
        rep.check("격자에서 변위 측정 (피치의 1/100 이하)", a.rms <= 0.01, fmt("RMS %.4f px (≤ 0.01)", a.rms));
        rep.check("격자에서 변형률 측정", b.maxAbs <= 2e-5, fmt("최대 오차 %.2e (≤ 2e-5)", b.maxAbs));
    }

    rep.h("V3-2 적용 가능성 — QR 이 격자인가");
    const Sample* refS = findStem(ds, "center");
    if (!refS) { rep.check("기준 영상", false, "center 없음"); return false; }
    const cv::Mat ref = loadGray32(refS->path);
    const Periodicity gp = periodicity(grating(0, 0));
    std::vector<std::vector<std::string>> rows;
    rows.push_back({"합성 격자 (피치 10)", fmt("%.0f", gp.score), fmt("%.1f", gp.period)});
    double qrMax = 0;
    for (int q = 0; q < 3; ++q) {
        const Region& r = regions()[q];
        const Periodicity pq = periodicity(ref(cv::Rect(r.cx - 100, r.cy - 100, 200, 200)));
        qrMax = std::max(qrMax, pq.score);
        rows.push_back({r.name, fmt("%.1f", pq.score), fmt("%.1f", pq.period)});
    }
    rep.table({"영역", "주기성 점수 (봉우리/중앙값)", "봉우리 주기 (px)"}, rows);
    // 화면 전체 타일 탐색 : 격자처럼 쓸 수 있는 배경이 있는가
    {
        Csv csv(outDir + "/m3_tile_periodicity.csv");
        csv << "x" << "y" << "score" << "period";
        csv.endl();
        const int T = 128;
        std::vector<std::pair<double, cv::Point>> best;
        cv::Mat heat(ref.rows / 64, ref.cols / 64, CV_32F, cv::Scalar(0));
        for (int y = 0; y + T <= ref.rows; y += 64)
            for (int x = 0; x + T <= ref.cols; x += 64) {
                const Periodicity pt = periodicity(ref(cv::Rect(x, y, T, T)));
                csv << x << y << pt.score << pt.period;
                csv.endl();
                heat.at<float>(y / 64, x / 64) = float(std::log10(pt.score));
                best.push_back({pt.score, cv::Point(x + T / 2, y + T / 2)});
            }
        std::sort(best.begin(), best.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
        std::string top;
        for (int i = 0; i < 5 && i < int(best.size()); ++i) top += fmt("(%d, %d) %.0f  ", best[i].second.x, best[i].second.y, best[i].first);
        rep.line("화면 전체 128 px 타일 주기성 상위 5 (중심 x, y, 점수) : " + top);
        cv::Mat h8, hc;
        cv::normalize(heat, h8, 0, 255, cv::NORM_MINMAX);
        h8.convertTo(h8, CV_8U);
        cv::resize(h8, hc, cv::Size(), 8, 8, cv::INTER_NEAREST);
        cv::applyColorMap(hc, hc, cv::COLORMAP_JET);
        cv::imwrite(outDir + "/m3_tile_periodicity.png", hc);
    }
    const bool qrGrid = qrMax >= 0.1 * gp.score;
    rep.line(fmt("판정 기준 : 합성 격자 점수의 10 %% (%.0f) 이상이면 격자로 취급. QR 최대 %.1f → %s.", 0.1 * gp.score, qrMax,
                 qrGrid ? "적용 가능" : "적용 불가 (모듈 배치가 무작위라 단일 주기 봉우리가 없음)"));

    rep.h("V3-3 그래도 QR 에 적용하면 — 실제 사진에서 M1 과 비교");
    {
        const CsvTable m1 = readCsv(outDir + "/../M1_phase_ecc/m1_shifts.csv");
        std::vector<std::vector<std::string>> r2;
        std::vector<double> err;
        for (const char* st : {"T+0.01", "T+0.05", "T+0.10", "Y+0.05", "Y+0.25", "Y-0.10"}) {
            const Sample* s = findStem(ds, st);
            if (!s) continue;
            const cv::Mat cur = loadGray32(s->path);
            const Region& r = regions()[1];
            const double m1u = num(m1, st, "QR_C_dx");
            const int gi = std::isfinite(m1u) ? int(std::lround(m1u)) : 0;
            const cv::Rect a(r.cx - 100, r.cy - 60, 200, 120);
            // 모듈 2 칸 ≈ 17.8 px → 피치 18
            const MoireResult m = moireDisplacement(ref(a), cur(a + cv::Point(gi, 0)), 18);
            const double u = m.u + gi;
            err.push_back(u - m1u);
            r2.push_back({st, fmt("%+.3f", m1u), fmt("%+.3f", u), fmt("%+.3f", u - m1u)});
        }
        rep.table({"사진", "M1 이동 (px)", "모아레 이동 (px)", "차이 (px)"}, r2);
        const Stat e = stat(err);
        rep.line(fmt("QR 에 억지로 적용한 결과의 M1 대비 차이 RMS %.3f px — M1 의 합성 정확도(0.013 px)보다 %s.", e.rms, e.rms > 0.05 ? "훨씬 나쁨" : "비슷함"));
    }
    rep.line("결론 : 알고리즘은 격자에서 설계대로 동작하지만, 이 QR 은 격자가 아니어서 M3 는 트랙 A·B 에 쓰지 않는다. 위상으로 읽는 원리는 M4 Gabor 로 대신한다.");
    return rep.allOk();
}

} // namespace m3
} // namespace yt
