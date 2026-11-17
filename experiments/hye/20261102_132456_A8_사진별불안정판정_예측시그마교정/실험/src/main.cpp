// A8: 사진별 불안정 판정. 정답을 쓰지 않는 지표들이 실제 OOF 오차를 예측하는지 본다.
#include "cache.hpp"
#include "noise.hpp"
#include "model.hpp"
#include "hye/plot.hpp"
#include <functional>
#include <set>
#include <iostream>

using namespace a0;
namespace {

double spearman(std::vector<double> a, std::vector<double> b) {
    const int n = int(a.size());
    if (n < 5) return 0;
    auto rank = [&](std::vector<double> v) {
        std::vector<int> idx(v.size());
        for (size_t i = 0; i < idx.size(); ++i) idx[i] = int(i);
        std::sort(idx.begin(), idx.end(), [&](int x, int y) { return v[x] < v[y]; });
        std::vector<double> r(v.size());
        for (size_t i = 0; i < idx.size(); ++i) r[idx[i]] = double(i);
        return r;
    };
    auto ra = rank(a), rb = rank(b);
    double ma = mean(ra), mb = mean(rb), sa = 0, sb = 0, sc = 0;
    for (int i = 0; i < n; ++i) {
        sa += (ra[i] - ma) * (ra[i] - ma);
        sb += (rb[i] - mb) * (rb[i] - mb);
        sc += (ra[i] - ma) * (rb[i] - mb);
    }
    return (sa > 0 && sb > 0) ? sc / std::sqrt(sa * sb) : 0;
}

}  // namespace

int main() {
    try {
        const fs::path result = fs::u8path(EXPERIMENT_DIR);
        const fs::path results = result / "results";
        fs::create_directories(results);
        hye::begin_results();
        Cache c = load(cache_dir());

        // 기준 모델 OOF
        std::vector<std::vector<double>> f66(c.n());
        for (auto& s : c.shots) if (s.measured) f66[s.idx] = features(c, s.idx, true);
        auto oof = run_oof(c, f66, [](const Shot& s) { return s.fold_mixed; }, 4, 0);

        // 잡음과 감도 (A2 와 같은 정의, raw 잔차)
        auto ns = measure_noise(c, [](const Shot&) { return true; }, false);
        std::vector<int> dims;
        for (int p = 0; p < kPoints; ++p) if (c.points[p].selected) { dims.push_back(p * 2); dims.push_back(p * 2 + 1); }
        auto Sp = build_inv_sigma(ns.point_rows, dims);
        std::vector<int> ld;
        for (int b = 0; b < kLines; ++b) ld.push_back(b);
        auto Sl = build_inv_sigma(ns.line_rows, ld);

        // J (1014 전체에서. 진단 지표용이라 fold 를 나누지 않고, 그 사실을 보고서에 적는다)
        const int D = int(dims.size());
        std::vector<int> ids;
        for (auto& s : c.shots) if (s.measured && s.set == "clean_20261014") ids.push_back(s.idx);
        cv::Mat X(int(ids.size()), 3, CV_64F), Yp(int(ids.size()), D, CV_64F), Yl(int(ids.size()), kLines, CV_64F);
        for (size_t i = 0; i < ids.size(); ++i) {
            X.at<double>(int(i), 0) = c.shots[ids[i]].y;
            X.at<double>(int(i), 1) = c.shots[ids[i]].t;
            X.at<double>(int(i), 2) = 1;
            for (int j = 0; j < D; ++j) {
                auto d = c.dz(ids[i], dims[j] / 2);
                Yp.at<double>(int(i), j) = (dims[j] % 2 == 0) ? d.x : d.y;
            }
            for (int b = 0; b < kLines; ++b) Yl.at<double>(int(i), b) = c.line(ids[i])[b * 2];
        }
        cv::Mat Bp, Bl;
        cv::solve(X, Yp, Bp, cv::DECOMP_SVD);
        cv::solve(X, Yl, Bl, cv::DECOMP_SVD);
        cv::Mat Jp(D, 2, CV_64F), Jl(kLines, 2, CV_64F);
        for (int j = 0; j < D; ++j) { Jp.at<double>(j, 0) = Bp.at<double>(0, j); Jp.at<double>(j, 1) = Bp.at<double>(1, j); }
        for (int b = 0; b < kLines; ++b) { Jl.at<double>(b, 0) = Bl.at<double>(0, b); Jl.at<double>(b, 1) = Bl.at<double>(1, b); }
        cv::Mat SJp = Sp.apply(Jp), SJl = Sl.apply(Jl);
        cv::Mat F = Jp.t() * SJp + Jl.t() * SJl, Finv = F.inv();
        cv::Point2d nd(-kTtoY, 1.0);
        nd *= 1.0 / cv::norm(nd);
        const double sigma_weak = std::sqrt(std::max(0.0,
            nd.x * nd.x * Finv.at<double>(0, 0) + 2 * nd.x * nd.y * Finv.at<double>(0, 1) + nd.y * nd.y * Finv.at<double>(1, 1)));

        // ---- 사진별 지표 ----
        Table dg;
        dg.head = {"shot", "set", "rel", "유효점", "최소 NCC", "유효 선블록", "GN 잔차 중앙값", "GN 잔차 p95",
                   "물리모델 잔차", "예측 sigma_weak", "eY", "eT", "약한방향 e", "통과"};
        std::vector<double> err, phys, vpt, ncc, lvl, gnm, gnp, weak_e;
        for (size_t i = 0; i < oof.shot.size(); ++i) {
            int id = oof.shot[i];
            const Shot& s = c.shots[id];
            double ey = oof.pred[i].x - oof.truth[i].x, et = oof.pred[i].y - oof.truth[i].y;
            // 물리 모델 잔차
            cv::Mat dp(D, 1, CV_64F), dl(kLines, 1, CV_64F);
            for (int j = 0; j < D; ++j) {
                auto d = c.dz(id, dims[j] / 2);
                dp.at<double>(j) = ((dims[j] % 2 == 0) ? d.x : d.y) - Bp.at<double>(2, j);
            }
            for (int b = 0; b < kLines; ++b) dl.at<double>(b) = c.line(id)[b * 2] - Bl.at<double>(2, b);
            cv::Mat q = Finv * (SJp.t() * dp + SJl.t() * dl);
            cv::Mat rp = dp - Jp * q, rl = dl - Jl * q;
            cv::Mat wrp = Sp.apply(rp), wrl = Sl.apply(rl);
            double res = std::sqrt(std::max(0.0, rp.dot(wrp) + rl.dot(wrl)) / (D + kLines));
            std::vector<double> gq;
            for (int p = 0; p < kPoints; ++p) if (c.points[p].selected) gq.push_back(c.q(id)[p * kQuality]);
            double m = median(gq), p95 = quantile(gq, 0.95);
            double we = ey * nd.x + et * nd.y;
            dg.add({std::to_string(id), s.set.substr(6), s.rel, std::to_string(s.valid_selected), num(std::min({s.ncc[0], s.ncc[1], s.ncc[2]}), 4),
                    num([&] { int k = 0; for (int b = 0; b < kLines; ++b) k += c.line_ok(id)[b]; return double(k); }(), 0),
                    num(m, 5), num(p95, 5), num(res, 5), num(sigma_weak, 5), num(ey, 4), num(et, 4), num(we, 4),
                    (std::abs(ey) <= kTol && std::abs(et) <= kTol) ? "1" : "0"});
            err.push_back(std::max(std::abs(ey), std::abs(et)));
            weak_e.push_back(we);
            phys.push_back(res);
            vpt.push_back(-double(s.valid_selected));
            ncc.push_back(-std::min({s.ncc[0], s.ncc[1], s.ncc[2]}));
            lvl.push_back(-[&] { int k = 0; for (int b = 0; b < kLines; ++b) k += c.line_ok(id)[b]; return double(k); }());
            gnm.push_back(m);
            gnp.push_back(p95);
        }
        dg.csv(results / "diagnostics.csv");

        // ---- 상관과 적중률 ----
        auto hit_rate = [&](const std::vector<double>& ind, double frac) {
            const int n = int(ind.size()), k = std::max(1, int(n * frac));
            std::vector<int> a(n), b(n);
            for (int i = 0; i < n; ++i) a[i] = b[i] = i;
            std::sort(a.begin(), a.end(), [&](int x, int y) { return ind[x] > ind[y]; });
            std::sort(b.begin(), b.end(), [&](int x, int y) { return err[x] > err[y]; });
            std::set<int> top(b.begin(), b.begin() + k);
            int hit = 0;
            for (int i = 0; i < k; ++i) if (top.count(a[i])) ++hit;
            return double(hit) / k;
        };
        Table ct;
        ct.head = {"지표", "순위 상관 (최대오차)", "순위 상관 (약한방향 |e|)", "상위 10% 적중률", "상위 20% 적중률"};
        std::vector<double> abs_weak;
        for (double v : weak_e) abs_weak.push_back(std::abs(v));
        struct Ind { std::string name; std::vector<double>* v; };
        std::vector<Ind> inds = {{"물리 모델 잔차", &phys}, {"유효점 수 (적을수록)", &vpt}, {"최소 NCC (낮을수록)", &ncc},
                                 {"유효 선블록 (적을수록)", &lvl}, {"GN 잔차 중앙값", &gnm}, {"GN 잔차 p95", &gnp}};
        for (auto& k : inds)
            ct.add({k.name, num(spearman(*k.v, err), 3), num(spearman(*k.v, abs_weak), 3),
                    num(hit_rate(*k.v, 0.10), 3), num(hit_rate(*k.v, 0.20), 3)});
        ct.csv(results / "correlation.csv");

        // ---- z 분포 ----
        std::vector<double> z;
        for (double v : weak_e) z.push_back(v / sigma_weak);
        double zsd = sd(z);

        // ---- 그림 ----
        {
            hye::Chart ch("A8-81 physical-model residual vs actual max error", "physical residual", "max |error|");
            ch.scatter(phys, err, hye::color(0), "", 3);
            ch.hline(kTol, hye::kRed);
            ch.save("81_지표대오차.png");
        }
        {
            std::vector<double> bins(21, 0), xs(21);
            for (int i = 0; i < 21; ++i) xs[i] = -5 + i * 0.5;
            for (double v : z) {
                int b = std::min(20, std::max(0, int((v + 5) / 0.5)));
                bins[b] += 1;
            }
            hye::Chart ch("A8-82 z = weak-direction error / predicted sigma", "z", "count");
            ch.line(xs, bins, hye::color(0), "observed", true);
            ch.vline(-1, hye::kGray);
            ch.vline(1, hye::kGray);
            ch.save("82_z분포.png");
        }
        {
            std::vector<std::string> cats;
            std::vector<double> vals;
            for (double f : {0.05, 0.10, 0.20, 0.30}) {
                cats.push_back(num(f * 100, 0) + "%");
                vals.push_back(hit_rate(phys, f));
            }
            hye::bar_chart("83_선별적중.png", "A8-83 hit rate of physical-residual selection", cats, vals, "hit rate", -1, 0.3);
        }

        double best_corr = 0;
        std::string best_name;
        for (auto& k : inds) {
            double v = std::abs(spearman(*k.v, err));
            if (v > best_corr) { best_corr = v; best_name = k.name; }
        }
        std::ostringstream o;
        o << "# A8 결과 — 사진별 불안정 판정\n\n"
          << "`실험/src/main.cpp` 가 다시 쓴다. 해석은 [보고서](보고서.md). 계획은 [계획서](../계획서.md).\n\n"
          << "기준 모델(M0) 혼합 4-fold OOF 오차 " << oof.shot.size() << "장에 대해, 정답을 쓰지 않는 지표들이 "
          << "그 오차를 예측하는지 봤다.\n\n"
          << "## 1. 지표와 실제 오차\n\n" << ct.md() << "\n"
          << hye::img_md("81_지표대오차.png", "indicator vs error") << "\n\n"
          << hye::img_md("83_선별적중.png", "hit rate") << "\n\n"
          << "사진별 값: [diagnostics.csv](results/diagnostics.csv)\n\n"
          << "## 2. 예측 sigma 의 보정\n\n"
          << "| 값 | 크기 |\n|---|---|\n"
          << "| 약한 방향 예측 sigma | " << num(sigma_weak, 5) << " |\n"
          << "| 실제 약한 방향 오차 SD | " << num(sd(weak_e), 5) << " |\n"
          << "| **z 의 표준편차** | **" << num(zsd, 3) << "** |\n\n"
          << hye::img_md("82_z분포.png", "z distribution") << "\n\n"
          << "## 3. 판정 기준 대비\n\n"
          << "| 조건 | 실측 | 판정 |\n|---|---|---|\n"
          << "| z 의 SD 가 0.7~1.4 | " << num(zsd, 3) << " | " << (zsd >= 0.7 && zsd <= 1.4 ? "**쓸 수 있다**" : "쓸 수 없다") << " |\n"
          << "| 어떤 지표의 |순위 상관| > 0.5 | 최고 " << best_name << " " << num(best_corr, 3) << " | "
          << (best_corr > 0.5 ? "**쓸 수 있다**" : "쓸 수 없다") << " |\n"
          << "| 상위 10% 선별 적중률 > 0.3 | " << num(hit_rate(phys, 0.10), 3) << " | "
          << (hit_rate(phys, 0.10) > 0.3 ? "**조건부 탐색이 쓸모 있다**" : "쓸모 없다") << " |\n";
        write_text(result / fs::u8path(u8"결과.md"), o.str());
        std::cout << "끝\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "오류: " << e.what() << "\n";
        return 1;
    }
}
