// A6: 추정 모델 대조. 같은 측정·같은 분할에서 8가지 모델을 비교한다.
//  M0 OLS 66d / M1 Ridge / M2 백색화+Ridge / M3 물리 전방 역해 / M4 OLS+2차항 / M5 QR만 / M6 선만 / M7 전체점 PCA
#include "cache.hpp"
#include "noise.hpp"
#include "model.hpp"
#include "hye/plot.hpp"
#include <chrono>
#include <functional>
#include <iostream>
#include <set>

using namespace a0;
namespace {

using Feat = std::vector<std::vector<double>>;

// 내부 CV 로 ridge lambda 를 고른다. 학습 사진만 받는다. 묶음 단위로 나눈다.
double pick_lambda(const Cache& c, const Feat& feat, const std::vector<int>& train, const std::vector<double>& grid,
                   std::vector<std::pair<double, double>>* curve) {
    std::map<int, int> inner;
    {
        std::set<int> gs;
        for (int i : train) gs.insert(c.shots[i].group);
        int k = 0;
        for (int g : gs) inner[g] = (k++) % 3;
    }
    double best = grid[0], best_score = 1e300;
    for (double lam : grid) {
        double worst = 0;
        int n = 0, pass = 0;
        for (int f = 0; f < 3; ++f) {
            Feat X;
            std::vector<cv::Point2d> Y;
            for (int i : train) {
                if (inner[c.shots[i].group] == f) continue;
                X.push_back(feat[i]);
                Y.push_back({c.shots[i].y, c.shots[i].t});
            }
            if (X.empty()) continue;
            LinearModel m;
            m.fit(X, Y, lam);
            for (int i : train) {
                if (inner[c.shots[i].group] != f) continue;
                auto p = m.predict(feat[i]);
                double ey = p.x - c.shots[i].y, et = p.y - c.shots[i].t;
                worst = std::max(worst, std::max(std::abs(ey), std::abs(et)));
                ++n;
                if (std::abs(ey) <= kTol && std::abs(et) <= kTol) ++pass;
            }
        }
        // 선택 기준은 바깥 평가 기준과 같아야 한다. 통과율을 주로, 최대오차를 동점 처리로 쓴다.
        // (처음에는 최대오차를 주 기준으로 썼는데, 그러면 예측을 평균 쪽으로 심하게 수축시키는 큰 lambda 가 뽑힌다.
        //  최대오차는 작아지지만 통과율이 무너진다. 2026-11-02 결과를 본 뒤 고쳤다.)
        double score = -(n ? double(pass) / n : 0) * 1000.0 + worst;
        if (curve) curve->push_back({lam, worst});
        if (score < best_score) { best_score = score; best = lam; }
    }
    return best;
}

Oof run_ridge(const Cache& c, const Feat& feat, const std::function<int(const Shot&)>& fold_of, int n_fold,
              const std::vector<double>& grid, Table* lamtab, const std::string& name,
              std::vector<std::pair<double, double>>* curve) {
    Oof out;
    for (int f = 0; f < n_fold; ++f) {
        std::vector<int> train;
        for (auto& s : c.shots) if (s.measured && fold_of(s) != f) train.push_back(s.idx);
        double lam = grid.size() > 1 ? pick_lambda(c, feat, train, grid, f == 0 ? curve : nullptr) : grid[0];
        if (lamtab) lamtab->add({name, std::to_string(f), num(lam, 5)});
        Feat X;
        std::vector<cv::Point2d> Y;
        for (int i : train) { X.push_back(feat[i]); Y.push_back({c.shots[i].y, c.shots[i].t}); }
        LinearModel m;
        m.fit(X, Y, lam);
        for (auto& s : c.shots) {
            if (!s.measured || fold_of(s) != f) continue;
            out.shot.push_back(s.idx);
            out.truth.push_back({s.y, s.t});
            out.pred.push_back(m.predict(feat[s.idx]));
            out.fold.push_back(f);
        }
    }
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
        auto t0 = std::chrono::steady_clock::now();

        // ---- 특징들 ----
        Feat f66(c.n()), f18(c.n()), f48(c.n()), f2nd(c.n()), fpts(c.n());
        for (auto& s : c.shots) {
            if (!s.measured) continue;
            f66[s.idx] = features(c, s.idx, true);
            f18[s.idx].assign(f66[s.idx].begin(), f66[s.idx].begin() + kFeatQr);
            f48[s.idx].assign(f66[s.idx].begin() + kFeatQr, f66[s.idx].end());
            // 전체 점 1260차원
            fpts[s.idx].resize(kPoints * 2);
            for (int p = 0; p < kPoints; ++p) {
                auto d = c.dz(s.idx, p);
                fpts[s.idx][p * 2] = d.x;
                fpts[s.idx][p * 2 + 1] = d.y;
            }
        }
        auto feat_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / c.n();

        // 2차항: 66차원 중 분산이 큰 상위 8개의 제곱과 서로의 곱
        {
            std::vector<std::pair<double, int>> var;
            for (int j = 0; j < kFeat; ++j) {
                std::vector<double> v;
                for (auto& s : c.shots) if (s.measured) v.push_back(f66[s.idx][j]);
                var.push_back({sd(v), j});
            }
            std::sort(var.rbegin(), var.rend());
            std::vector<int> top;
            for (int k = 0; k < 8; ++k) top.push_back(var[k].second);
            for (auto& s : c.shots) {
                if (!s.measured) continue;
                f2nd[s.idx] = f66[s.idx];
                for (size_t a = 0; a < top.size(); ++a)
                    for (size_t b = a; b < top.size(); ++b)
                        f2nd[s.idx].push_back(f66[s.idx][top[a]] * f66[s.idx][top[b]]);
            }
        }

        // 전체 점 PCA 40
        Feat fpca(c.n());
        {
            std::vector<int> ids;
            for (auto& s : c.shots) if (s.measured) ids.push_back(s.idx);
            cv::Mat M(int(ids.size()), kPoints * 2, CV_64F);
            for (size_t i = 0; i < ids.size(); ++i)
                for (int j = 0; j < kPoints * 2; ++j) M.at<double>(int(i), j) = fpts[ids[i]][j];
            cv::PCA pca(M, cv::Mat(), cv::PCA::DATA_AS_ROW, 40);
            for (size_t i = 0; i < ids.size(); ++i) {
                cv::Mat p = pca.project(M.row(int(i)));
                fpca[ids[i]].assign(p.ptr<double>(), p.ptr<double>() + p.cols);
            }
        }

        // 백색화 특징: 반복에서 잰 특징 잡음 공분산으로 백색화
        Feat fwhite(c.n());
        {
            std::map<int, std::vector<int>> group;
            for (auto& s : c.shots) if (s.measured) group[s.group].push_back(s.idx);
            std::vector<std::vector<double>> rows;
            for (auto& g : group) {
                if (g.second.size() < 2) continue;
                std::vector<double> m(kFeat, 0);
                for (int i : g.second) for (int j = 0; j < kFeat; ++j) m[j] += f66[i][j];
                for (auto& v : m) v /= g.second.size();
                for (int i : g.second) {
                    std::vector<double> r(kFeat);
                    for (int j = 0; j < kFeat; ++j) r[j] = f66[i][j] - m[j];
                    rows.push_back(std::move(r));
                }
            }
            cv::Mat R(int(rows.size()), kFeat, CV_64F);
            for (int i = 0; i < R.rows; ++i) for (int j = 0; j < kFeat; ++j) R.at<double>(i, j) = rows[i][j];
            cv::Mat S = R.t() * R / std::max(1, R.rows - 1);
            cv::Mat ev, evec;
            cv::eigen(S, ev, evec);
            double fl = std::max(1e-12, ev.at<double>(kFeat - 1));
            cv::Mat Wm = cv::Mat::zeros(kFeat, kFeat, CV_64F);
            for (int i = 0; i < kFeat; ++i) {
                double l = std::max(ev.at<double>(i), fl);
                Wm += evec.row(i).t() * evec.row(i) / std::sqrt(l);
            }
            for (auto& s : c.shots) {
                if (!s.measured) continue;
                cv::Mat x(kFeat, 1, CV_64F);
                for (int j = 0; j < kFeat; ++j) x.at<double>(j) = f66[s.idx][j];
                cv::Mat z = Wm * x;
                fwhite[s.idx].assign(z.ptr<double>(), z.ptr<double>() + kFeat);
            }
        }

        // ---- M3 물리 전방 역해 ----
        // fold 의 학습 사진(1014 단일축)으로 J 를 세우고, 시험 사진의 관측을 가중 최소제곱으로 역해한다.
        auto physical_oof = [&](const std::function<int(const Shot&)>& fold_of, int n_fold) {
            Oof out;
            auto ns = measure_noise(c, [](const Shot&) { return true; }, false);  // raw 잔차 (A2 의 결론)
            std::vector<int> dims;
            for (int p = 0; p < kPoints; ++p) if (c.points[p].selected) { dims.push_back(p * 2); dims.push_back(p * 2 + 1); }
            auto Sp = build_inv_sigma(ns.point_rows, dims);
            std::vector<int> ld;
            for (int b = 0; b < kLines; ++b) ld.push_back(b);
            auto Sl = build_inv_sigma(ns.line_rows, ld);
            for (int f = 0; f < n_fold; ++f) {
                // J: 학습 fold 의 1014 사진에서
                std::vector<int> ids;
                for (auto& s : c.shots)
                    if (s.measured && s.set == "clean_20261014" && fold_of(s) != f) ids.push_back(s.idx);
                if (ids.size() < 10) continue;
                const int n = int(ids.size()), D = int(dims.size());
                cv::Mat X(n, 3, CV_64F), Yp(n, D, CV_64F), Yl(n, kLines, CV_64F);
                for (int i = 0; i < n; ++i) {
                    X.at<double>(i, 0) = c.shots[ids[i]].y;
                    X.at<double>(i, 1) = c.shots[ids[i]].t;
                    X.at<double>(i, 2) = 1;
                    for (int j = 0; j < D; ++j) {
                        int p = dims[j] / 2;
                        auto d = c.dz(ids[i], p);
                        Yp.at<double>(i, j) = (dims[j] % 2 == 0) ? d.x : d.y;
                    }
                    for (int b = 0; b < kLines; ++b) Yl.at<double>(i, b) = c.line(ids[i])[b * 2];
                }
                cv::Mat Bp, Bl;
                cv::solve(X, Yp, Bp, cv::DECOMP_SVD);
                cv::solve(X, Yl, Bl, cv::DECOMP_SVD);
                cv::Mat Jp(D, 2, CV_64F), Jl(kLines, 2, CV_64F);
                for (int j = 0; j < D; ++j) { Jp.at<double>(j, 0) = Bp.at<double>(0, j); Jp.at<double>(j, 1) = Bp.at<double>(1, j); }
                for (int b = 0; b < kLines; ++b) { Jl.at<double>(b, 0) = Bl.at<double>(0, b); Jl.at<double>(b, 1) = Bl.at<double>(1, b); }
                cv::Mat SJp = Sp.apply(Jp), SJl = Sl.apply(Jl);
                cv::Mat F = Jp.t() * SJp + Jl.t() * SJl;
                cv::Mat Finv = F.inv();
                for (auto& s : c.shots) {
                    if (!s.measured || fold_of(s) != f) continue;
                    cv::Mat dp(D, 1, CV_64F), dl(kLines, 1, CV_64F);
                    for (int j = 0; j < D; ++j) {
                        int p = dims[j] / 2;
                        auto d = c.dz(s.idx, p);
                        dp.at<double>(j) = ((dims[j] % 2 == 0) ? d.x : d.y) - Bp.at<double>(2, j);
                    }
                    for (int b = 0; b < kLines; ++b) dl.at<double>(b) = c.line(s.idx)[b * 2] - Bl.at<double>(2, b);
                    cv::Mat q = Finv * (SJp.t() * dp + SJl.t() * dl);
                    out.shot.push_back(s.idx);
                    out.truth.push_back({s.y, s.t});
                    out.pred.push_back({q.at<double>(0), q.at<double>(1)});
                    out.fold.push_back(f);
                }
            }
            return out;
        };

        // ---- 실행 ----
        auto mixed = [](const Shot& s) { return s.fold_mixed; };
        auto bydate = [](const Shot& s) { return s.fold_date; };
        std::vector<double> grid = {0, 0.01, 0.1, 1, 10, 100, 1000};
        std::vector<double> one = {0};
        Table lam;
        lam.head = {"모델", "fold", "lambda"};
        std::vector<std::pair<double, double>> curve;

        struct Entry { std::string id, name; Oof mixed, date; };
        std::vector<Entry> E;
        E.push_back({"M0", "OLS 66d (대조)", run_ridge(c, f66, mixed, 4, one, nullptr, "M0", nullptr),
                     run_ridge(c, f66, bydate, 3, one, nullptr, "M0", nullptr)});
        E.push_back({"M1", "Ridge 66d", run_ridge(c, f66, mixed, 4, grid, &lam, "M1", &curve),
                     run_ridge(c, f66, bydate, 3, grid, &lam, "M1", nullptr)});
        E.push_back({"M2", "백색화 + Ridge 66d", run_ridge(c, fwhite, mixed, 4, grid, &lam, "M2", nullptr),
                     run_ridge(c, fwhite, bydate, 3, grid, &lam, "M2", nullptr)});
        E.push_back({"M3", "물리 전방 역해 (J, raw Sigma)", physical_oof(mixed, 4), physical_oof(bydate, 3)});
        E.push_back({"M4", "OLS + 2차항", run_ridge(c, f2nd, mixed, 4, grid, &lam, "M4", nullptr),
                     run_ridge(c, f2nd, bydate, 3, grid, &lam, "M4", nullptr)});
        E.push_back({"M5", "QR만 18d", run_ridge(c, f18, mixed, 4, one, nullptr, "M5", nullptr),
                     run_ridge(c, f18, bydate, 3, one, nullptr, "M5", nullptr)});
        E.push_back({"M6", "선만 48d", run_ridge(c, f48, mixed, 4, one, nullptr, "M6", nullptr),
                     run_ridge(c, f48, bydate, 3, one, nullptr, "M6", nullptr)});
        E.push_back({"M7", "전체점 1260d -> PCA 40", run_ridge(c, fpca, mixed, 4, grid, &lam, "M7", nullptr),
                     run_ridge(c, fpca, bydate, 3, grid, &lam, "M7", nullptr)});
        lam.csv(results / "lambda.csv");

        auto keep_in_range = [&](const Oof& o) {
            std::vector<unsigned char> k(o.shot.size(), 1);
            for (size_t i = 0; i < o.shot.size(); ++i) if (o.truth[i].y > 1.001) k[i] = 0;
            return k;
        };

        Table st;
        st.head = {"ID", "모델", "분할", "장수", "Y 평균", "Y 최대", "T 평균", "T 최대", "통과 ±0.045", "참고 ±0.09", "최대오차 대조비"};
        Table perf;
        perf.head = {"experiment", "method", "split", "n", "mean_y", "max_y", "mean_t", "max_t", "pass_0045", "pass_009"};
        double base_max = 0;
        {
            auto k = keep_in_range(E[0].mixed);
            auto s = score(E[0].mixed, &k);
            base_max = std::max(s.max_y, s.max_t);
        }
        for (auto& e : E) {
            for (int d = 0; d < 2; ++d) {
                const Oof& o = d ? e.date : e.mixed;
                auto k = keep_in_range(o);
                auto s = score(o, &k);
                double mx = std::max(s.max_y, s.max_t);
                st.add({e.id, e.name, d ? "날짜 3-fold" : "혼합 4-fold", std::to_string(s.n), num(s.mean_y, 4),
                        num(s.max_y, 4), num(s.mean_t, 4), num(s.max_t, 4), num(s.pass, 4), num(s.pass09, 4),
                        d ? "" : num(mx / base_max, 3)});
                perf.add({"A6", e.id + " " + e.name, d ? "date3" : "mixed4", std::to_string(s.n), num(s.mean_y, 4),
                          num(s.max_y, 4), num(s.mean_t, 4), num(s.max_t, 4), num(s.pass, 4), num(s.pass09, 4)});
            }
        }
        st.csv(results / "model_scores.csv");
        perf.csv(results / "perf_rows.csv");

        // ---- 그림 ----
        {
            std::vector<std::string> cats;
            std::vector<double> v1, v2;
            for (auto& e : E) {
                cats.push_back(e.id);
                auto k1 = keep_in_range(e.mixed), k2 = keep_in_range(e.date);
                v1.push_back(score(e.mixed, &k1).pass);
                v2.push_back(score(e.date, &k2).pass);
            }
            hye::bar_chart("61_모델비교_통과율.png", "A6-61 pass rate at +-0.045 (mixed 4-fold)", cats, v1, "pass rate");
            hye::bar_chart("61b_모델비교_통과율_날짜.png", "A6-61b pass rate at +-0.045 (date 3-fold)", cats, v2, "pass rate");
        }
        {
            std::vector<std::string> cats;
            std::vector<double> v;
            for (auto& e : E) {
                cats.push_back(e.id);
                auto k = keep_in_range(e.mixed);
                auto s = score(e.mixed, &k);
                v.push_back(std::max(s.max_y, s.max_t));
            }
            hye::bar_chart("62_모델비교_최대오차.png", "A6-62 max abs error (mixed 4-fold)", cats, v, "max error", -1, kTol);
        }
        {
            hye::Chart ch("A6-63 |eY| sorted (mixed 4-fold)", "rank", "|eY| [mm]");
            for (size_t i = 0; i < E.size(); ++i) {
                auto k = keep_in_range(E[i].mixed);
                std::vector<double> v;
                for (size_t j = 0; j < E[i].mixed.shot.size(); ++j)
                    if (k[j]) v.push_back(std::abs(E[i].mixed.pred[j].x - E[i].mixed.truth[j].x));
                std::sort(v.begin(), v.end());
                std::vector<double> x(v.size());
                for (size_t j = 0; j < v.size(); ++j) x[j] = double(j);
                ch.line(x, v, hye::color(int(i)), E[i].id);
            }
            ch.hline(kTol, hye::kRed);
            ch.save("63_오차분포.png");
        }
        if (!curve.empty()) {
            hye::Chart ch("A6-64 inner-CV worst error vs ridge lambda (M1, fold 0)", "lambda", "worst |error|");
            std::vector<double> x, y;
            for (auto& p : curve) { x.push_back(p.first); y.push_back(p.second); }
            ch.line(x, y, hye::color(0), "", true);
            ch.save("64_람다곡선.png");
        }

        // ---- 결과.md ----
        std::ostringstream o;
        o << "# A6 결과 — 추정 모델 대조\n\n"
          << "`실험/src/main.cpp` 가 다시 쓴다. 해석은 [보고서](보고서.md). 계획은 [계획서](../계획서.md).\n\n"
          << "모두 같은 A0 캐시, 같은 분할을 쓴다. 특징 추출 + 예측 시간은 사진당 " << num(feat_ms, 3)
          << " ms 다. **점 측정 자체의 비용은 모델 간에 같고 A0 에서 쟀다(575장 15초).**\n\n"
          << "## 1. 모델별 성적\n\n" << st.md() << "\n"
          << hye::img_md("61_모델비교_통과율.png", "pass mixed") << "\n\n"
          << hye::img_md("61b_모델비교_통과율_날짜.png", "pass date") << "\n\n"
          << hye::img_md("62_모델비교_최대오차.png", "max error") << "\n\n"
          << hye::img_md("63_오차분포.png", "error distribution") << "\n\n"
          << "## 2. 정규화\n\n" << lam.md() << "\n"
          << hye::img_md("64_람다곡선.png", "lambda curve") << "\n\n"
          << "## 3. 판정 기준 대비\n\n"
          << "대조 M0 의 최대오차는 " << num(base_max, 4) << " 다. 20% 이상 줄이고 통과율이 떨어지지 않는 모델만 후보다.\n\n"
          << "| ID | 최대오차 대조비 | 통과율 | 판정 |\n|---|---|---|---|\n";
        {
            auto k0 = keep_in_range(E[0].mixed);
            double base_pass = score(E[0].mixed, &k0).pass;
            for (auto& e : E) {
                auto k = keep_in_range(e.mixed);
                auto s = score(e.mixed, &k);
                double mx = std::max(s.max_y, s.max_t);
                bool ok = mx <= base_max * 0.8 && s.pass >= base_pass;
                o << "| " << e.id << " | " << num(mx / base_max, 3) << " | " << num(s.pass, 4) << " | "
                  << (e.id == "M0" ? "대조" : (ok ? "**후보**" : "기각")) << " |\n";
            }
        }
        write_text(result / fs::u8path(u8"결과.md"), o.str());
        std::cout << "끝\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "오류: " << e.what() << "\n";
        return 1;
    }
}
