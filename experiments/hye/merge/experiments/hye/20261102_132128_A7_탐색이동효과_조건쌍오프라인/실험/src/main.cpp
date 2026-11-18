// A7: 알려진 탐색 이동의 효과 (오프라인 조합, H16).
//  단일 추정기 f(i) -> (Y,T) 와 짝 추정기 [f(i), f(j)-f(i), dcmd] -> (Y,T) 를 같은 분할에서 비교한다.
//  실제로 움직이고 다시 찍은 자료가 아니다. 서로 다른 시점의 정지 사진을 짝지은 것이다.
#include "cache.hpp"
#include "model.hpp"
#include "hye/plot.hpp"
#include <functional>
#include <iostream>
#include <set>

using namespace a0;

int main() {
    try {
        const fs::path result = fs::u8path(EXPERIMENT_DIR);
        const fs::path results = result / "results";
        fs::create_directories(results);
        hye::begin_results();
        Cache c = load(cache_dir());

        std::vector<std::vector<double>> f66(c.n());
        for (auto& s : c.shots) if (s.measured) f66[s.idx] = features(c, s.idx, true);

        // 약한 방향 단위벡터
        cv::Point2d nd(-kTtoY, 1.0);
        nd *= 1.0 / cv::norm(nd);

        // ---- A7 전용 분할 ----
        // A3 의 혼합 4-fold 는 이웃 T 조건을 서로 다른 fold 로 보낸다(크기 계층 배정). 그러면 짝의 두 사진이
        // 학습과 시험으로 갈려 짝 추정기를 평가할 수 없다(실제로 273쌍 중 7쌍만 남았다).
        // 그래서 여기서는 **이웃 조건 4개를 한 덩어리로 묶어** fold 를 준다. 짝은 같은 fold 안에 들어간다.
        std::map<int, int> a7fold;
        {
            std::map<int, std::vector<int>> group;
            for (auto& s : c.shots) if (s.measured) group[s.group].push_back(s.idx);
            std::map<std::pair<std::string, long long>, std::vector<std::pair<double, int>>> by;
            for (auto& g : group) {
                const Shot& s = c.shots[g.second[0]];
                by[{s.set, llround(s.y * 1000)}].push_back({s.t, g.first});
            }
            for (auto& row : by) {
                auto v = row.second;
                std::sort(v.begin(), v.end());
                for (size_t k = 0; k < v.size(); ++k) a7fold[v[k].second] = int((k / 4) % 4);
            }
        }

        // ---- 짝 만들기: 같은 세트, 같은 Y, T 가 다른 이웃 조건 (1029 중심) ----
        struct Pair { int i, j; double dy, dt; };
        std::vector<Pair> pairs;
        {
            // 조건 대표 = 각 묶음의 첫 사진
            std::map<int, std::vector<int>> group;
            for (auto& s : c.shots) if (s.measured) group[s.group].push_back(s.idx);
            // 같은 세트, 같은 Y 인 묶음을 T 로 정렬해 이웃끼리 짝지음
            for (auto set : {"clean_20261029", "clean_20261020"}) {
                std::map<long long, std::vector<std::pair<double, int>>> by_y;
                for (auto& g : group) {
                    const Shot& s = c.shots[g.second[0]];
                    if (s.set != set) continue;
                    by_y[llround(s.y * 1000)].push_back({s.t, g.first});
                }
                for (auto& row : by_y) {
                    auto v = row.second;
                    std::sort(v.begin(), v.end());
                    for (size_t k = 0; k + 1 < v.size(); ++k)
                        for (int step = 1; step <= 3 && k + step < v.size(); ++step) {
                            int gi = v[k].second, gj = v[k + step].second;
                            // 각 묶음의 첫 사진끼리
                            Pair p{group[gi][0], group[gj][0], 0, v[k + step].first - v[k].first};
                            pairs.push_back(p);
                        }
                }
            }
            // 같은 T, Y 가 다른 짝 (1020)
            std::map<long long, std::vector<std::pair<double, int>>> by_t;
            for (auto& g : group) {
                const Shot& s = c.shots[g.second[0]];
                if (s.set != "clean_20261020") continue;
                by_t[llround(s.t * 1000)].push_back({s.y, g.first});
            }
            for (auto& row : by_t) {
                auto v = row.second;
                std::sort(v.begin(), v.end());
                for (size_t k = 0; k + 1 < v.size(); ++k) {
                    int gi = v[k].second, gj = v[k + 1].second;
                    pairs.push_back({group[gi][0], group[gj][0], v[k + 1].first - v[k].first, 0});
                }
            }
        }
        std::cout << "짝 " << pairs.size() << "개\n";

        // ---- 두 추정기를 짝 단위로 평가 ----
        // fold 는 i 사진의 fold 를 쓰되, j 가 같은 fold 가 아니면 그 짝을 버린다 (학습·시험 섞임 방지)
        struct Row { int shot; cv::Point2d truth; std::vector<double> single, pairf; int fold; };
        std::vector<Row> rows;
        for (auto& p : pairs) {
            const Shot& si = c.shots[p.i];
            const Shot& sj = c.shots[p.j];
            if (a7fold[si.group] != a7fold[sj.group]) continue;
            Row r;
            r.shot = p.i;
            r.truth = {si.y, si.t};
            r.fold = a7fold[si.group];
            r.single = f66[p.i];
            r.pairf = f66[p.i];
            for (int k = 0; k < kFeat; ++k) r.pairf.push_back(f66[p.j][k] - f66[p.i][k]);
            r.pairf.push_back(p.dy);
            r.pairf.push_back(p.dt);
            rows.push_back(std::move(r));
        }
        std::cout << "같은 fold 짝 " << rows.size() << "개\n";

        auto run = [&](bool use_pair) {
            std::vector<cv::Point2d> truth, pred;
            std::vector<int> shot;
            for (int f = 0; f < 4; ++f) {
                std::vector<std::vector<double>> X;
                std::vector<cv::Point2d> Y;
                for (auto& r : rows) {
                    if (r.fold == f) continue;
                    X.push_back(use_pair ? r.pairf : r.single);
                    Y.push_back(r.truth);
                }
                if (X.empty()) continue;
                LinearModel m;
                // 짝 특징은 134차원인데 fold 하나의 학습 짝이 약 112개라 보통 최소제곱으로는 풀리지 않는다.
                // 두 추정기에 **같은** 작은 ridge 를 걸어 공정하게 비교한다.
                m.fit(X, Y, 0.01);
                for (auto& r : rows) {
                    if (r.fold != f) continue;
                    truth.push_back(r.truth);
                    pred.push_back(m.predict(use_pair ? r.pairf : r.single));
                    shot.push_back(r.shot);
                }
            }
            return std::make_tuple(truth, pred, shot);
        };
        auto [t1, p1, s1] = run(false);
        auto [t2, p2, s2] = run(true);

        auto stats = [&](const std::vector<cv::Point2d>& t, const std::vector<cv::Point2d>& p) {
            std::vector<double> ey, et, weak;
            int pass = 0;
            for (size_t i = 0; i < t.size(); ++i) {
                double a = p[i].x - t[i].x, b = p[i].y - t[i].y;
                ey.push_back(a);
                et.push_back(b);
                weak.push_back(a * nd.x + b * nd.y);
                if (std::abs(a) <= kTol && std::abs(b) <= kTol) ++pass;
            }
            struct S { double my, mxy, mt, mxt, ws, pass; int n; };
            std::vector<double> ay, at;
            for (double v : ey) ay.push_back(std::abs(v));
            for (double v : et) at.push_back(std::abs(v));
            return S{mean(ay), max_abs(ey), mean(at), max_abs(et), sd(weak), t.empty() ? 0 : double(pass) / t.size(), int(t.size())};
        };
        auto a = stats(t1, p1), b = stats(t2, p2);

        Table st;
        st.head = {"추정기", "장수", "Y 평균", "Y 최대", "T 평균", "T 최대", "약한 방향 SD", "통과 ±0.045"};
        st.add({"단일 사진", std::to_string(a.n), num(a.my, 4), num(a.mxy, 4), num(a.mt, 4), num(a.mxt, 4), num(a.ws, 5), num(a.pass, 4)});
        st.add({"짝 (알려진 이동 1회)", std::to_string(b.n), num(b.my, 4), num(b.mxy, 4), num(b.mt, 4), num(b.mxt, 4), num(b.ws, 5), num(b.pass, 4)});
        st.csv(results / "pair_scores.csv");

        Table perf;
        perf.head = {"experiment", "method", "split", "n", "mean_y", "max_y", "mean_t", "max_t", "pass_0045", "pass_009"};
        perf.add({"A7", "single image (66d)", "mixed4 pairs", std::to_string(a.n), num(a.my, 4), num(a.mxy, 4), num(a.mt, 4), num(a.mxt, 4), num(a.pass, 4), ""});
        perf.add({"A7", "image pair + known move (134d)", "mixed4 pairs", std::to_string(b.n), num(b.my, 4), num(b.mxy, 4), num(b.mt, 4), num(b.mxt, 4), num(b.pass, 4), ""});
        perf.csv(results / "perf_rows.csv");

        // 명령 차 크기별
        Table bd;
        bd.head = {"|dT|", "짝 수", "단일 약한 방향 SD", "짝 약한 방향 SD", "비"};
        {
            std::map<long long, std::vector<size_t>> by;
            for (size_t i = 0; i < rows.size(); ++i) {
                // s2 와 rows 의 순서가 같지 않으므로 shot 으로 맞춘다
            }
            // shot -> index
            std::map<int, size_t> i1, i2;
            for (size_t i = 0; i < s1.size(); ++i) i1[s1[i]] = i;
            for (size_t i = 0; i < s2.size(); ++i) i2[s2[i]] = i;
            std::map<long long, std::pair<std::vector<double>, std::vector<double>>> acc;
            for (size_t r = 0; r < rows.size(); ++r) {
                int sh = rows[r].shot;
                if (!i1.count(sh) || !i2.count(sh)) continue;
                double dt = rows[r].pairf[rows[r].pairf.size() - 1];
                long long key = llround(std::abs(dt) * 100);
                auto e1 = p1[i1[sh]] - t1[i1[sh]];
                auto e2 = p2[i2[sh]] - t2[i2[sh]];
                acc[key].first.push_back(e1.x * nd.x + e1.y * nd.y);
                acc[key].second.push_back(e2.x * nd.x + e2.y * nd.y);
            }
            for (auto& k : acc) {
                double s_1 = sd(k.second.first), s_2 = sd(k.second.second);
                bd.add({num(k.first / 100.0, 2), std::to_string(k.second.first.size()), num(s_1, 5), num(s_2, 5),
                        num(s_1 > 0 ? s_2 / s_1 : 0, 3)});
            }
        }
        bd.csv(results / "by_delta.csv");

        // ---- 그림 ----
        {
            hye::Chart ch("A7-71 OOF error along weak direction", "rank", "weak-direction error");
            for (int k = 0; k < 2; ++k) {
                const auto& t = k ? t2 : t1;
                const auto& p = k ? p2 : p1;
                std::vector<double> v;
                for (size_t i = 0; i < t.size(); ++i) {
                    auto e = p[i] - t[i];
                    v.push_back(e.x * nd.x + e.y * nd.y);
                }
                std::sort(v.begin(), v.end());
                std::vector<double> x(v.size());
                for (size_t i = 0; i < v.size(); ++i) x[i] = double(i);
                ch.line(x, v, hye::color(k), k ? "pair + known move" : "single image");
            }
            ch.save("71_단일대짝_오차.png");
        }
        {
            std::vector<std::string> cats;
            std::vector<double> vals;
            for (auto& r : bd.rows) { cats.push_back("dT " + r[0]); vals.push_back(std::stod(r[4])); }
            if (!cats.empty()) hye::bar_chart("72_명령차별효과.png", "A7-72 pair/single weak-direction SD ratio (lower is better)", cats, vals, "ratio", -1, 0.5);
        }

        bool h16 = b.ws <= a.ws * 0.5;
        std::ostringstream o;
        o << "# A7 결과 — 알려진 탐색 이동의 효과 (오프라인)\n\n"
          << "`실험/src/main.cpp` 가 다시 쓴다. 해석은 [보고서](보고서.md). 계획은 [계획서](../계획서.md).\n\n"
          << "**이것은 오프라인 조합이다.** 서로 다른 시점의 정지 사진을 짝지은 것이며, 실제로 움직이고 다시 찍은 자료가 아니다. "
          << "백래시·정지 오차·드리프트가 빠져 있다.\n\n"
          << "짝 " << pairs.size() << "개를 만들고, 두 사진이 같은 fold 인 " << rows.size() << "개만 썼다.\n\n"
          << "## 1. 단일 대 짝\n\n" << st.md() << "\n"
          << hye::img_md("71_단일대짝_오차.png", "single vs pair") << "\n\n"
          << "## 2. 명령 차 크기별\n\n" << bd.md() << "\n"
          << hye::img_md("72_명령차별효과.png", "by delta") << "\n\n"
          << "## 3. 판정 기준 대비\n\n"
          << "| 조건 | 실측 | 판정 |\n|---|---|---|\n"
          << "| 짝의 약한 방향 SD가 단일의 절반 이하 | " << num(b.ws, 5) << " 대 " << num(a.ws, 5)
          << " (비 " << num(a.ws > 0 ? b.ws / a.ws : 0, 3) << ") | " << (h16 ? "**H16 성립**" : "기각 또는 부분 효과") << " |\n";
        write_text(result / fs::u8path(u8"결과.md"), o.str());
        std::cout << "끝\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "오류: " << e.what() << "\n";
        return 1;
    }
}
