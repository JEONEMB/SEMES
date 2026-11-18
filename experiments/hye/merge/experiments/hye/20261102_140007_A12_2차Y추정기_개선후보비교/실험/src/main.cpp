// A12: 2차(Y 정렬) 추정기의 개선 후보 비교.
//  control() 은 QR 536점 P12 만 쓰고 뒷벽 선을 쓰지 않는다. 그래서 남은 T 가 있으면 5.617 배로 Y 가 틀어진다.
//  여기서는 "선을 써서 남은 T 를 스스로 처리하는 2차 추정기" 가 가능한지 본다.
#include "cache.hpp"
#include "noise.hpp"
#include "model.hpp"
#include "hye/plot.hpp"
#include <functional>
#include <iostream>
#include <set>

using namespace a0;
namespace {

// control() 과 같은 P12: 선택 536점에 [1,u,v,u^2,uv,v^2] 를 dx,dy 에 맞춘 12계수
std::vector<double> p12(const Cache& c, int shot) {
    std::vector<int> idx;
    for (int p = 0; p < kPoints; ++p) if (c.points[p].selected && c.v(shot)[p]) idx.push_back(p);
    std::vector<double> out(12, 0);
    if (idx.size() < 12) return out;
    cv::Mat X(int(idx.size()), 6, CV_64F), Y(int(idx.size()), 2, CV_64F), B;
    for (int k = 0; k < int(idx.size()); ++k) {
        double u = (c.points[idx[k]].x - 1906) / 486, v = (c.points[idx[k]].y - 1070) / 122;
        double b[] = {1, u, v, u * u, u * v, v * v};
        for (int j = 0; j < 6; ++j) X.at<double>(k, j) = b[j];
        auto d = c.dz(shot, idx[k]);
        Y.at<double>(k, 0) = d.x;
        Y.at<double>(k, 1) = d.y;
    }
    cv::solve(X, Y, B, cv::DECOMP_SVD);
    for (int j = 0; j < 6; ++j) { out[j] = B.at<double>(j, 0); out[6 + j] = B.at<double>(j, 1); }
    return out;
}

// 판 전체 가로 이동 (선택점 평균 dx)
double plate_dx(const Cache& c, int shot) {
    double s = 0;
    int n = 0;
    for (int p = 0; p < kPoints; ++p) if (c.points[p].selected && c.v(shot)[p]) { s += c.dz(shot, p).x; ++n; }
    return n ? s / n : 0;
}

struct Axis1 {
    cv::Mat w;
    std::vector<double> mu, sd_;
    int d = 0;
    void fit(const std::vector<std::vector<double>>& X, const std::vector<double>& y, double lambda) {
        d = int(X[0].size());
        const int n = int(X.size());
        mu.assign(d, 0); sd_.assign(d, 1);
        for (int j = 0; j < d; ++j) {
            double s = 0;
            for (int i = 0; i < n; ++i) s += X[i][j];
            mu[j] = s / n;
            double v = 0;
            for (int i = 0; i < n; ++i) v += (X[i][j] - mu[j]) * (X[i][j] - mu[j]);
            sd_[j] = std::sqrt(v / std::max(1, n - 1));
            if (sd_[j] < 1e-12) sd_[j] = 1;
        }
        cv::Mat A(n + d, d + 1, CV_64F, 0.0), B(n + d, 1, CV_64F, 0.0);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < d; ++j) A.at<double>(i, j) = (X[i][j] - mu[j]) / sd_[j];
            A.at<double>(i, d) = 1;
            B.at<double>(i) = y[i];
        }
        for (int j = 0; j < d; ++j) A.at<double>(n + j, j) = std::sqrt(std::max(lambda, 1e-9));
        cv::solve(A, B, w, cv::DECOMP_SVD);
    }
    double predict(const std::vector<double>& x) const {
        double a = w.at<double>(d);
        for (int j = 0; j < d; ++j) a += w.at<double>(j) * (x[j] - mu[j]) / sd_[j];
        return a;
    }
};

}  // namespace

int main() {
    try {
        const fs::path result = fs::u8path(EXPERIMENT_DIR);
        const fs::path results = result / "results";
        fs::create_directories(results);
        hye::begin_results();
        Cache c = load(cache_dir());

        // 특징들
        std::vector<std::vector<double>> F12(c.n()), FQR(c.n()), FLN(c.n()), F12L(c.n()), FQRL(c.n());
        std::vector<double> PDX(c.n(), 0);
        for (auto& s : c.shots) {
            if (!s.measured) continue;
            F12[s.idx] = p12(c, s.idx);
            auto f = features(c, s.idx, true);
            FQR[s.idx].assign(f.begin(), f.begin() + kFeatQr);
            FLN[s.idx].assign(f.begin() + kFeatQr, f.end());
            F12L[s.idx] = F12[s.idx];
            F12L[s.idx].insert(F12L[s.idx].end(), FLN[s.idx].begin(), FLN[s.idx].end());
            FQRL[s.idx] = f;
            PDX[s.idx] = plate_dx(c, s.idx);
        }

        const double kSmall = 0.0501;   // 2차 진입 영역으로 보는 |T| 범위
        auto in_small = [&](const Shot& s) { return std::abs(s.t) <= kSmall; };
        auto in_range = [&](const Shot& s) { return s.t <= 1.001; };

        struct Cand { std::string id, name; std::vector<double> pred; };
        std::vector<Cand> C;
        auto make = [&](const std::string& id, const std::string& name) {
            C.push_back({id, name, std::vector<double>(c.n(), std::numeric_limits<double>::quiet_NaN())});
            return C.size() - 1;
        };
        size_t i1 = make("C1", "P12 12d, |T|<=0.05 학습 (control 재현)");
        size_t i2 = make("C2", "P12 12d + 선 48d, |T|<=0.05 학습");
        size_t i3 = make("C3", "QR아핀 18d + 선 48d, |T|<=0.05 학습");
        size_t i4 = make("C4", "물리 보정: 선으로 T 빼고 판이동/14.25");
        size_t i5 = make("C5", "QR아핀 18d + 선 48d, 전체 T 학습");
        size_t i6 = make("C6", "P12 12d, 전체 T 학습");
        std::vector<double> D1(c.n(), 0), D2(c.n(), 0), D3(c.n(), 0);

        for (int f = 0; f < 4; ++f) {
            std::vector<int> tr_small, tr_all, te;
            for (auto& s : c.shots) {
                if (!s.measured) continue;
                if (s.fold_mixed == f) { te.push_back(s.idx); continue; }
                tr_all.push_back(s.idx);
                if (in_small(s)) tr_small.push_back(s.idx);
            }
            auto fit_y = [&](const std::vector<std::vector<double>>& FE, const std::vector<int>& tr, double lam) {
                std::vector<std::vector<double>> X;
                std::vector<double> y;
                for (int i : tr) { X.push_back(FE[i]); y.push_back(c.shots[i].y); }
                Axis1 m;
                m.fit(X, y, lam);
                return m;
            };
            auto m1 = fit_y(F12, tr_small, 1e-6);
            auto m2 = fit_y(F12L, tr_small, 1e-6);
            auto m3 = fit_y(FQRL, tr_small, 1e-6);
            auto m5 = fit_y(FQRL, tr_all, 1e-6);
            auto m6 = fit_y(F12, tr_all, 1e-6);
            // C4: 선 -> T 선형, 판 dx -> Y. 둘 다 학습 fold 전체에서 (물리량이라 넓은 범위가 낫다)
            Axis1 mT;
            {
                std::vector<std::vector<double>> X;
                std::vector<double> y;
                for (int i : tr_all) { X.push_back(FLN[i]); y.push_back(c.shots[i].t); }
                mT.fit(X, y, 1e-6);
            }
            double bY = 0, bT = 0, b0 = 0;
            {
                const int n = int(tr_all.size());
                cv::Mat X(n, 3, CV_64F), Y(n, 1, CV_64F), B;
                for (int i = 0; i < n; ++i) {
                    X.at<double>(i, 0) = c.shots[tr_all[i]].y;
                    X.at<double>(i, 1) = c.shots[tr_all[i]].t;
                    X.at<double>(i, 2) = 1;
                    Y.at<double>(i) = PDX[tr_all[i]];
                }
                cv::solve(X, Y, B, cv::DECOMP_SVD);
                bY = B.at<double>(0); bT = B.at<double>(1); b0 = B.at<double>(2);
            }
            // ---- 불안정 지표 (정답을 쓰지 않는다) ----
            // d1 |T_line|      : 선이 말하는 남은 T. 2차 진입 조건 위반을 바로 잡아낸다.
            // d2 |T_line-T_qr| : 선과 QR 이 서로 다른 T 를 말한다. 둘 중 하나가 틀렸다는 뜻.
            // d3 판 dx 불일치  : 두 축 모델이 판 가로 이동을 설명하지 못하고 남긴 양 (px)
            Axis1 mTq;
            {
                std::vector<std::vector<double>> X;
                std::vector<double> y;
                for (int i : tr_all) { X.push_back(FQR[i]); y.push_back(c.shots[i].t); }
                mTq.fit(X, y, 1e-6);
            }
            Axis1 mYq;
            {
                std::vector<std::vector<double>> X;
                std::vector<double> y;
                for (int i : tr_all) { X.push_back(FQRL[i]); y.push_back(c.shots[i].y); }
                mYq.fit(X, y, 1e-6);
            }
            for (int i : te) {
                C[i1].pred[i] = m1.predict(F12[i]);
                C[i2].pred[i] = m2.predict(F12L[i]);
                C[i3].pred[i] = m3.predict(FQRL[i]);
                C[i5].pred[i] = m5.predict(FQRL[i]);
                C[i6].pred[i] = m6.predict(F12[i]);
                double that = mT.predict(FLN[i]);
                C[i4].pred[i] = (PDX[i] - b0 - bT * that) / bY;
                D1[i] = std::abs(that);
                D2[i] = std::abs(that - mTq.predict(FQR[i]));
                D3[i] = std::abs(PDX[i] - b0 - bY * mYq.predict(FQRL[i]) - bT * that);
            }
        }

        // ---- A10 의 control() 실측을 함께 싣는다 ----
        std::vector<double> ctrl(c.n(), std::numeric_limits<double>::quiet_NaN());
        std::vector<double> jnt(c.n(), std::numeric_limits<double>::quiet_NaN());
        {
            fs::path p = fs::u8path(REPO_DIR) / fs::u8path(u8"experiments/hye/20261102_120000_A10_전달모듈대조_joint와control실측/결과/results/per_shot.csv");
            std::ifstream in(p);
            if (in) {
                std::string line;
                std::getline(in, line);
                auto head = split_csv(line);
                int ci = -1, cy = -1, cj = -1;
                for (size_t k = 0; k < head.size(); ++k) {
                    if (head[k] == "shot") ci = int(k);
                    if (head[k] == "control_Y") cy = int(k);
                    if (head[k] == "joint_Y") cj = int(k);
                }
                while (std::getline(in, line)) {
                    if (line.empty()) continue;
                    auto r = split_csv(line);
                    int id = std::stoi(r[ci]);
                    if (r[cy] != "FAIL" && !r[cy].empty()) ctrl[id] = std::stod(r[cy]);
                    if (r[cj] != "FAIL" && !r[cj].empty()) jnt[id] = std::stod(r[cj]);
                }
                std::cout << "A10 결과 병합 완료\n";
            } else std::cout << "A10 결과 없음 (control/joint 열은 비운다)\n";
        }

        // ---- 평가 ----
        struct Row { std::string id, name; const std::vector<double>* p; };
        std::vector<Row> all;
        all.push_back({"C0", "control() 전달 모듈 (QR만)", &ctrl});
        for (auto& k : C) all.push_back({k.id, k.name, &k.pred});
        all.push_back({"CJ", "joint() 전달 모듈 (참고)", &jnt});

        auto stat = [&](const std::vector<double>& pred, const std::function<bool(const Shot&)>& keep) {
            std::vector<double> e;
            for (auto& s : c.shots) {
                if (!s.measured || !in_range(s) || !keep(s)) continue;
                if (!std::isfinite(pred[s.idx])) continue;
                e.push_back(pred[s.idx] - s.y);
            }
            std::vector<double> a;
            for (double v : e) a.push_back(std::abs(v));
            int pass = 0;
            for (double v : a) if (v <= kTol) ++pass;
            return std::make_tuple(int(e.size()), mean(a), max_abs(e), e.empty() ? 0.0 : double(pass) / e.size(), sd(e));
        };

        Table st;
        st.head = {"ID", "2차 Y 추정기", "T=0 n", "T=0 |eY|평균", "T=0 최대", "|T|<=0.05 평균", "|T|<=0.05 최대",
                   "|T|<=0.05 통과", "전 범위 평균", "전 범위 최대"};
        for (auto& r : all) {
            auto z = stat(*r.p, [](const Shot& s) { return std::abs(s.t) < 1e-9; });
            auto sm = stat(*r.p, [&](const Shot& s) { return std::abs(s.t) <= kSmall; });
            auto ar = stat(*r.p, [](const Shot&) { return true; });
            st.add({r.id, r.name, std::to_string(std::get<0>(z)), num(std::get<1>(z), 4), num(std::get<2>(z), 4),
                    num(std::get<1>(sm), 4), num(std::get<2>(sm), 4), num(std::get<3>(sm), 4),
                    num(std::get<1>(ar), 4), num(std::get<2>(ar), 4)});
        }
        st.csv(results / "stage2_scores.csv");

        // 남은 T 구간별
        Table bt;
        bt.head = {"ID"};
        std::vector<std::pair<double, double>> bins = {{0, 0.001}, {0.001, 0.015}, {0.015, 0.035}, {0.035, 0.06}, {0.06, 0.15}, {0.15, 0.5}, {0.5, 1.01}};
        for (auto& b : bins) bt.head.push_back("|T| " + num(b.first, 3) + "~" + num(b.second, 3));
        for (auto& r : all) {
            std::vector<std::string> row{r.id};
            for (auto& b : bins) {
                auto z = stat(*r.p, [&](const Shot& s) { return std::abs(s.t) >= b.first && std::abs(s.t) < b.second; });
                row.push_back(std::get<0>(z) ? num(std::get<1>(z), 4) : "");
            }
            bt.add(row);
        }
        bt.csv(results / "by_residual_T.csv");

        // 반복 흩어짐 (|T| <= 0.05 묶음)
        Table rt;
        rt.head = {"ID", "묶음 수", "반복 SD 중앙값", "p95", "3SD(중앙) "};
        for (auto& r : all) {
            std::map<int, std::vector<double>> g;
            for (auto& s : c.shots)
                if (s.measured && std::abs(s.t) <= kSmall && std::isfinite((*r.p)[s.idx])) g[s.group].push_back((*r.p)[s.idx]);
            std::vector<double> sds;
            for (auto& k : g) if (k.second.size() >= 3) sds.push_back(sd(k.second));
            if (sds.empty()) { rt.add({r.id, "0", "", "", ""}); continue; }
            rt.add({r.id, std::to_string(sds.size()), num(median(sds), 5), num(quantile(sds, 0.95), 5), num(3 * median(sds), 4)});
        }
        rt.csv(results / "repeat_scatter.csv");

        // ---- 불안정 지표가 실제 오차를 잡아내는가 ----
        auto spearman = [](std::vector<double> a, std::vector<double> b) {
            const int n = int(a.size());
            if (n < 5) return 0.0;
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
            return (sa > 0 && sb > 0) ? sc / std::sqrt(sa * sb) : 0.0;
        };
        Table dg;
        dg.head = {"추정기", "지표", "순위 상관 |eY|", "상위10% 적중", "상위20% 적중"};
        struct Ind { std::string name; std::vector<double>* v; };
        std::vector<Ind> inds = {{"d1 |T_line| (남은 T)", &D1}, {"d2 |T_line - T_qr| (불일치)", &D2}, {"d3 판 dx 불일치", &D3}};
        auto gather = [&](const std::vector<double>& pred, std::vector<double>& err, std::vector<int>& ids) {
            for (auto& s : c.shots) {
                if (!s.measured || !in_range(s) || !std::isfinite(pred[s.idx])) continue;
                err.push_back(std::abs(pred[s.idx] - s.y));
                ids.push_back(s.idx);
            }
        };
        for (auto& r : all) {
            if (r.id != "C0" && r.id != "C1" && r.id != "C3") continue;
            std::vector<double> err;
            std::vector<int> ids;
            gather(*r.p, err, ids);
            for (auto& k : inds) {
                std::vector<double> ind;
                for (int i : ids) ind.push_back((*k.v)[i]);
                auto hit = [&](double frac) {
                    const int n = int(ind.size()), m = std::max(1, int(n * frac));
                    std::vector<int> a(n), b(n);
                    for (int i = 0; i < n; ++i) a[i] = b[i] = i;
                    std::sort(a.begin(), a.end(), [&](int x, int y) { return ind[x] > ind[y]; });
                    std::sort(b.begin(), b.end(), [&](int x, int y) { return err[x] > err[y]; });
                    std::set<int> top(b.begin(), b.begin() + m);
                    int h = 0;
                    for (int i = 0; i < m; ++i) if (top.count(a[i])) ++h;
                    return double(h) / m;
                };
                dg.add({r.id, k.name, num(spearman(ind, err), 3), num(hit(0.10), 3), num(hit(0.20), 3)});
            }
        }
        dg.csv(results / "instability_indicators.csv");

        // ---- 게이팅: d1 임계를 넘으면 2차를 하지 않고 1차로 되돌린다 ----
        Table gt;
        gt.head = {"추정기", "d1 임계 (deg)", "통과한 사진", "비율", "그중 |eY| 평균", "최대", "±0.045 통과"};
        for (auto& r : all) {
            if (r.id != "C0" && r.id != "C1" && r.id != "C3") continue;
            for (double th : {1.0, 0.05, 0.02, 0.01, 0.005, 0.0027}) {
                std::vector<double> e;
                int tot = 0;
                for (auto& s : c.shots) {
                    if (!s.measured || !in_range(s) || !std::isfinite((*r.p)[s.idx])) continue;
                    ++tot;
                    if (D1[s.idx] > th) continue;
                    e.push_back((*r.p)[s.idx] - s.y);
                }
                if (e.empty()) continue;
                std::vector<double> a;
                for (double v : e) a.push_back(std::abs(v));
                int pass = 0;
                for (double v : a) if (v <= kTol) ++pass;
                gt.add({r.id, num(th, 4), std::to_string(e.size()), num(double(e.size()) / tot, 3),
                        num(mean(a), 4), num(max_abs(e), 4), num(double(pass) / e.size(), 4)});
            }
        }
        gt.csv(results / "gating.csv");

        Table perf;
        perf.head = {"experiment", "method", "split", "n", "mean_y", "max_y", "mean_t", "max_t", "pass_0045", "pass_009"};
        for (auto& r : all) {
            auto sm = stat(*r.p, [&](const Shot& s) { return std::abs(s.t) <= kSmall; });
            perf.add({"A12", r.id + " " + r.name, "stage2 |T|<=0.05", std::to_string(std::get<0>(sm)),
                      num(std::get<1>(sm), 4), num(std::get<2>(sm), 4), "", "", num(std::get<3>(sm), 4), ""});
        }
        perf.csv(results / "perf_rows.csv");

        // ---- 그림 ----
        {
            hye::Chart ch("A12-121 stage-2 |eY| vs residual |T|", "|T| [deg]", "mean |eY| [mm]");
            int k = 0;
            for (auto& r : all) {
                std::vector<double> x, y;
                for (auto& b : bins) {
                    auto z = stat(*r.p, [&](const Shot& s) { return std::abs(s.t) >= b.first && std::abs(s.t) < b.second; });
                    if (!std::get<0>(z)) continue;
                    x.push_back(0.5 * (b.first + b.second));
                    y.push_back(std::get<1>(z));
                }
                ch.line(x, y, hye::color(k++), r.id, true);
            }
            ch.hline(kTol, hye::kRed);
            ch.save("121_잔여T별_Y오차.png");
        }
        {
            std::vector<std::string> cats;
            std::vector<double> v;
            for (auto& r : all) {
                auto sm = stat(*r.p, [&](const Shot& s) { return std::abs(s.t) <= kSmall; });
                cats.push_back(r.id);
                v.push_back(std::get<2>(sm));
            }
            hye::bar_chart("122_2차최대오차.png", "A12-122 stage-2 max |eY| for |T|<=0.05", cats, v, "max |eY| [mm]", -1, kTol);
        }
        {
            std::vector<std::string> cats;
            std::vector<double> v;
            for (auto& r : rt.rows) if (!r[2].empty()) { cats.push_back(r[0]); v.push_back(3 * std::stod(r[2])); }
            hye::bar_chart("123_반복3SD.png", "A12-123 stage-2 repeat 3SD (|T|<=0.05 groups)", cats, v, "3SD [mm]", -1, kTol);
        }

        std::ostringstream o;
        o << "# A12 결과 — 2차 Y 추정기 개선 후보\n\n"
          << "`실험/src/main.cpp` 가 다시 쓴다. 해석은 [보고서](보고서.md).\n\n"
          << "2차 진입 영역을 **|T| <= 0.05** 로 보고, 그 영역에서 학습한 Y 전용 추정기들을 비교한다. "
          << "C0(control)·CJ(joint)는 A10 의 실측을 가져왔다(재대입 성적). C1~C6 은 혼합 4-fold OOF 다.\n\n"
          << "## 1. 성적\n\n" << st.md() << "\n"
          << hye::img_md("122_2차최대오차.png", "max error") << "\n\n"
          << "## 2. 남은 T 가 커질 때\n\n" << bt.md() << "\n"
          << hye::img_md("121_잔여T별_Y오차.png", "vs residual T") << "\n\n"
          << "## 3. 반복 흩어짐\n\n" << rt.md() << "\n"
          << hye::img_md("123_반복3SD.png", "repeat 3SD") << "\n\n"
          << "## 4. 불안정 지표 (정답을 쓰지 않는다)\n\n" << dg.md() << "\n"
          << "무작위 적중률은 상위10% 선별에서 0.10, 상위20% 에서 0.20 이다.\n\n"
          << "## 5. 게이팅: d1 임계를 넘으면 2차를 하지 않는다\n\n" << gt.md() << "\n";
        write_text(result / fs::u8path(u8"결과.md"), o.str());
        std::cout << "끝\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "오류: " << e.what() << "\n";
        return 1;
    }
}
