// A11: 순차 추정(T 먼저, Y 나중)을 동시 추정과 비교한다.
//  규칙: 1차 T 정렬, 2차 Y 정렬. 2차는 "T가 0에 가깝다"는 전제를 쓸 수 있다.
//  여기서는 실제로 로봇을 움직여 T를 맞춘 뒤 다시 찍은 자료가 없으므로, **대수적으로** T 기여를 빼고 Y 를 추정한다.
//  따라서 이 결과는 실제 2단계 운용 성적이 아니라 "추정 구조만 순차로 바꿨을 때" 의 결과다.
#include "cache.hpp"
#include "model.hpp"
#include "hye/plot.hpp"
#include <functional>
#include <iostream>

using namespace a0;
namespace {

// 한 축만 맞추는 선형 모델
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
        cv::Mat A(n + (lambda > 0 ? d : 0), d + 1, CV_64F, 0.0), B(A.rows, 1, CV_64F, 0.0);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < d; ++j) A.at<double>(i, j) = (X[i][j] - mu[j]) / sd_[j];
            A.at<double>(i, d) = 1;
            B.at<double>(i) = y[i];
        }
        if (lambda > 0) for (int j = 0; j < d; ++j) A.at<double>(n + j, j) = std::sqrt(lambda);
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

        std::vector<std::vector<double>> f66(c.n()), f48(c.n());
        for (auto& s : c.shots) if (s.measured) {
            f66[s.idx] = features(c, s.idx, true);
            f48[s.idx].assign(f66[s.idx].begin() + kFeatQr, f66[s.idx].end());
        }

        auto fold_of = [](const Shot& s) { return s.fold_mixed; };
        const int NF = 4;

        struct Out { std::vector<int> shot; std::vector<cv::Point2d> truth, pred; std::vector<double> t_err; };
        auto blank = [&] { Out o; return o; };
        Out S0 = blank(), S1 = blank(), S2 = blank(), S3 = blank();

        for (int f = 0; f < NF; ++f) {
            std::vector<int> train, test;
            for (auto& s : c.shots) {
                if (!s.measured) continue;
                (fold_of(s) == f ? test : train).push_back(s.idx);
            }
            // --- S0 동시 추정 (대조) ---
            {
                std::vector<std::vector<double>> X;
                std::vector<cv::Point2d> Y;
                for (int i : train) { X.push_back(f66[i]); Y.push_back({c.shots[i].y, c.shots[i].t}); }
                LinearModel m;
                m.fit(X, Y, 0);
                for (int i : test) {
                    S0.shot.push_back(i);
                    S0.truth.push_back({c.shots[i].y, c.shots[i].t});
                    S0.pred.push_back(m.predict(f66[i]));
                    S0.t_err.push_back(0);
                }
            }
            // --- 특징의 축 기여 J_f: 학습 사진에서 f = Jy*Y + Jt*T + f0 ---
            cv::Mat Bf;
            {
                const int n = int(train.size());
                cv::Mat X(n, 3, CV_64F), Yf(n, kFeat, CV_64F);
                for (int i = 0; i < n; ++i) {
                    X.at<double>(i, 0) = c.shots[train[i]].y;
                    X.at<double>(i, 1) = c.shots[train[i]].t;
                    X.at<double>(i, 2) = 1;
                    for (int j = 0; j < kFeat; ++j) Yf.at<double>(i, j) = f66[train[i]][j];
                }
                cv::solve(X, Yf, Bf, cv::DECOMP_SVD);
            }
            auto strip_t = [&](const std::vector<double>& x, double that) {
                std::vector<double> z(kFeat);
                for (int j = 0; j < kFeat; ++j) z[j] = x[j] - Bf.at<double>(1, j) * that;
                return z;
            };
            auto strip_y = [&](const std::vector<double>& x, double yhat) {
                std::vector<double> z(kFeat);
                for (int j = 0; j < kFeat; ++j) z[j] = x[j] - Bf.at<double>(0, j) * yhat;
                return z;
            };

            // --- 1단계 T 추정기 (전체 특징) ---
            Axis1 mt;
            {
                std::vector<std::vector<double>> X;
                std::vector<double> y;
                for (int i : train) { X.push_back(f66[i]); y.push_back(c.shots[i].t); }
                mt.fit(X, y, 0);
            }
            // --- 1단계 Y 추정기 (S3 역순용) ---
            Axis1 my_first;
            {
                std::vector<std::vector<double>> X;
                std::vector<double> y;
                for (int i : train) { X.push_back(f66[i]); y.push_back(c.shots[i].y); }
                my_first.fit(X, y, 0);
            }
            // --- 2단계 Y 추정기 (a) 전체 학습 자료의 T 제거 잔차로 학습 ---
            Axis1 my_all;
            {
                std::vector<std::vector<double>> X;
                std::vector<double> y;
                for (int i : train) { X.push_back(strip_t(f66[i], c.shots[i].t)); y.push_back(c.shots[i].y); }
                my_all.fit(X, y, 0);
            }
            // --- 2단계 Y 추정기 (b) T=0 자료만으로 학습 (규칙의 "T≈0 전제") ---
            Axis1 my_t0;
            {
                std::vector<std::vector<double>> X;
                std::vector<double> y;
                for (int i : train) {
                    if (std::abs(c.shots[i].t) > 1e-9) continue;
                    X.push_back(f66[i]);
                    y.push_back(c.shots[i].y);
                }
                // T=0 학습 사진이 fold 당 약 60장인데 특징이 66차원이라 보통 최소제곱으로는 풀리지 않는다.
                // S2 에만 작은 ridge 를 건다. 다른 구조는 lambda=0 으로 두어 A3·A6 과 비교 가능하게 한다.
                my_t0.fit(X, y, 0.01);
                if (X.size() < 20) throw std::runtime_error("T=0 학습 사진이 너무 적다");
            }
            // --- 2단계 T 추정기 (S3 역순용): Y 제거 잔차 ---
            Axis1 mt_second;
            {
                std::vector<std::vector<double>> X;
                std::vector<double> y;
                for (int i : train) { X.push_back(strip_y(f66[i], c.shots[i].y)); y.push_back(c.shots[i].t); }
                mt_second.fit(X, y, 0);
            }

            for (int i : test) {
                double that = mt.predict(f66[i]);
                auto fr = strip_t(f66[i], that);
                double y_all = my_all.predict(fr);
                double y_t0 = my_t0.predict(fr);
                double yfirst = my_first.predict(f66[i]);
                double t_second = mt_second.predict(strip_y(f66[i], yfirst));
                cv::Point2d truth(c.shots[i].y, c.shots[i].t);
                S1.shot.push_back(i); S1.truth.push_back(truth); S1.pred.push_back({y_all, that}); S1.t_err.push_back(that - truth.y);
                S2.shot.push_back(i); S2.truth.push_back(truth); S2.pred.push_back({y_t0, that}); S2.t_err.push_back(that - truth.y);
                S3.shot.push_back(i); S3.truth.push_back(truth); S3.pred.push_back({yfirst, t_second}); S3.t_err.push_back(t_second - truth.y);
            }
        }

        auto summarize = [&](const Out& o, const std::string& name, Table& st, Table& perf) {
            std::vector<double> ey, et;
            int n = 0, pass = 0, pass09 = 0;
            for (size_t i = 0; i < o.shot.size(); ++i) {
                if (o.truth[i].y > 1.001) continue;   // T+2.00 범위 밖
                double a = o.pred[i].x - o.truth[i].x, b = o.pred[i].y - o.truth[i].y;
                ey.push_back(a); et.push_back(b);
                ++n;
                if (std::abs(a) <= kTol && std::abs(b) <= kTol) ++pass;
                if (std::abs(a) <= 0.09 && std::abs(b) <= 0.09) ++pass09;
            }
            std::vector<double> ay, at;
            for (double v : ey) ay.push_back(std::abs(v));
            for (double v : et) at.push_back(std::abs(v));
            st.add({name, std::to_string(n), num(mean(ay), 4), num(max_abs(ey), 4), num(mean(at), 4), num(max_abs(et), 4),
                    num(n ? double(pass) / n : 0, 4), num(n ? double(pass09) / n : 0, 4)});
            perf.add({"A11", name, "mixed4", std::to_string(n), num(mean(ay), 4), num(max_abs(ey), 4), num(mean(at), 4),
                      num(max_abs(et), 4), num(n ? double(pass) / n : 0, 4), num(n ? double(pass09) / n : 0, 4)});
            return std::make_pair(ey, et);
        };

        Table st;
        st.head = {"추정 구조", "장수", "Y 평균", "Y 최대", "T 평균", "T 최대", "통과 ±0.045", "참고 ±0.09"};
        Table perf;
        perf.head = {"experiment", "method", "split", "n", "mean_y", "max_y", "mean_t", "max_t", "pass_0045", "pass_009"};
        auto e0 = summarize(S0, "S0 동시 추정 (대조)", st, perf);
        auto e1 = summarize(S1, "S1 순차 T→Y, 2단계 전체 자료 학습", st, perf);
        auto e2 = summarize(S2, "S2 순차 T→Y, 2단계 T=0 자료만 학습", st, perf);
        auto e3 = summarize(S3, "S3 순차 Y→T (역순)", st, perf);
        st.csv(results / "sequential_scores.csv");
        perf.csv(results / "perf_rows.csv");

        // ---- 1단계 오차가 2단계로 얼마나 번지는가 ----
        Table prop;
        prop.head = {"구조", "1단계 오차 SD", "2단계 오차 SD", "두 오차의 상관", "전파 기울기 (2단계/1단계)"};
        auto propagate = [&](const Out& o, const std::string& name, bool t_first) {
            std::vector<double> e1v, e2v;
            for (size_t i = 0; i < o.shot.size(); ++i) {
                if (o.truth[i].y > 1.001) continue;
                double ey = o.pred[i].x - o.truth[i].x, et = o.pred[i].y - o.truth[i].y;
                e1v.push_back(t_first ? et : ey);
                e2v.push_back(t_first ? ey : et);
            }
            double m1 = mean(e1v), m2 = mean(e2v), s1 = 0, s2 = 0, sc = 0;
            for (size_t i = 0; i < e1v.size(); ++i) {
                s1 += (e1v[i] - m1) * (e1v[i] - m1);
                s2 += (e2v[i] - m2) * (e2v[i] - m2);
                sc += (e1v[i] - m1) * (e2v[i] - m2);
            }
            double r = (s1 > 0 && s2 > 0) ? sc / std::sqrt(s1 * s2) : 0;
            prop.add({name, num(sd(e1v), 5), num(sd(e2v), 5), num(r, 3), num(s1 > 0 ? sc / s1 : 0, 3)});
        };
        propagate(S1, "S1 (T 먼저)", true);
        propagate(S2, "S2 (T 먼저, T=0 학습)", true);
        propagate(S3, "S3 (Y 먼저)", false);
        prop.csv(results / "propagation.csv");

        // ---- 그림 ----
        {
            std::vector<std::string> cats;
            std::vector<double> v, mx;
            for (auto& r : st.rows) { cats.push_back(r[0]); v.push_back(std::stod(r[6])); mx.push_back(std::stod(r[3])); }
            hye::bar_chart("111_순차대동시_통과율.png", "A11-111 pass rate at +-0.045", cats, v, "pass rate");
            hye::bar_chart("112_순차대동시_최대오차.png", "A11-112 max |eY|", cats, mx, "max |eY| [mm]", -1, kTol);
        }
        {
            hye::Chart ch("A11-113 stage-1 T error vs stage-2 Y error (S2)", "eT (stage 1) [deg]", "eY (stage 2) [mm]");
            std::vector<double> x, y;
            for (size_t i = 0; i < S2.shot.size(); ++i) {
                if (S2.truth[i].y > 1.001) continue;
                x.push_back(S2.pred[i].y - S2.truth[i].y);
                y.push_back(S2.pred[i].x - S2.truth[i].x);
            }
            ch.scatter(x, y, hye::color(0), "", 3);
            ch.hline(kTol, hye::kRed); ch.hline(-kTol, hye::kRed);
            ch.save("113_오차전파.png");
        }
        {
            hye::Chart ch("A11-114 |eY| sorted", "rank", "|eY| [mm]");
            int k = 0;
            for (const Out* o : {&S0, &S1, &S2, &S3}) {
                std::vector<double> v;
                for (size_t i = 0; i < o->shot.size(); ++i)
                    if (o->truth[i].y <= 1.001) v.push_back(std::abs(o->pred[i].x - o->truth[i].x));
                std::sort(v.begin(), v.end());
                std::vector<double> x(v.size());
                for (size_t i = 0; i < v.size(); ++i) x[i] = double(i);
                const char* nm[] = {"S0 joint", "S1 T->Y all", "S2 T->Y T0-only", "S3 Y->T"};
                ch.line(x, v, hye::color(k), nm[k]);
                ++k;
            }
            ch.hline(kTol, hye::kRed);
            ch.save("114_오차분포.png");
        }

        std::ostringstream o;
        o << "# A11 결과 — 순차 추정 (T 먼저, Y 나중)\n\n"
          << "`실험/src/main.cpp` 가 다시 쓴다. 해석은 [보고서](보고서.md).\n\n"
          << "**중요.** 실제로 T 를 맞춘 뒤 다시 찍은 자료가 없다. 그래서 1단계 추정값으로 **특징에서 T 기여를 대수적으로 빼고** "
          << "2단계 Y 를 추정했다. 실제 2단계 운용에서는 로봇이 움직이고 다시 찍으므로 백래시·정지 오차가 더 붙는다. "
          << "이 결과는 **추정 구조만 순차로 바꿨을 때**의 상한이다.\n\n"
          << "## 1. 네 가지 구조\n\n"
          << "| ID | 구조 | 2단계 학습 자료 |\n|---|---|---|\n"
          << "| S0 | Y·T 동시 추정 (대조) | 전체 |\n"
          << "| S1 | T 먼저 → Y | 전체 학습 사진에서 T 기여를 뺀 잔차 |\n"
          << "| S2 | T 먼저 → Y | **T=0 사진만** (규칙의 'T≈0 전제') |\n"
          << "| S3 | Y 먼저 → T (역순) | 전체 학습 사진에서 Y 기여를 뺀 잔차 |\n\n"
          << "## 2. 성적 (범위 안, T ≤ 1.00)\n\n" << st.md() << "\n"
          << hye::img_md("111_순차대동시_통과율.png", "pass rate") << "\n\n"
          << hye::img_md("112_순차대동시_최대오차.png", "max error") << "\n\n"
          << hye::img_md("114_오차분포.png", "distribution") << "\n\n"
          << "## 3. 1단계 오차가 2단계로 번지는 정도\n\n" << prop.md() << "\n"
          << hye::img_md("113_오차전파.png", "propagation") << "\n\n"
          << "전파 기울기는 1단계 오차 1단위가 2단계 오차 몇 단위를 만드는지다. "
          << "감도 비로 보면 T 1 단위는 Y 약 " << num(kTtoY, 2) << " mm 에 해당하므로, 기울기가 그 값에 가까우면 "
          << "**1단계 오차가 그대로 2단계로 넘어간다**는 뜻이다.\n";
        write_text(result / fs::u8path(u8"결과.md"), o.str());
        std::cout << "끝\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "오류: " << e.what() << "\n";
        return 1;
    }
}
