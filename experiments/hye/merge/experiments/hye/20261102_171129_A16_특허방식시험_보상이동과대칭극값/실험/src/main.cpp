// A16: 원인분석 특허 두 건의 방식을 실제로 돌린다.
//
//  1부 [PT-01 Cognex US9247684] 보상 이동으로 "가상 회전 중심" 만들기
//      특허: 회전 중심이 시야 밖이면 궤적이 짧은 원호라 중심 오차가 1/tan(dθ/2) 배로 커진다.
//            시야 안에 설정 회전 중심을 두고 재측정해 보정한다.
//      우리 2축 판: "T 회전 + 보상 Y 이동"을 함께 주면 판이 제자리에 머문다. 그 보상비가 곧 지렛대(등가 반경)다.
//      **영점 시험(null test)이라 단일축 적합보다 정밀하다.** 세션 안의 조건 쌍만 써서 세션 드리프트도 뺀다.
//
//  2부 [PT-04 Kawasaki US11749547] 대칭·극값으로 축 찾기
//      특허: 매개변수 적합 없이 "같은 값이 나오는 두 위치의 가운데"와 "최댓값"만으로 회전 중심을 찾는다.
//      우리 판: 선 블록의 세로 응답 v = C(x-x0)(y-y0) 는 x = x0 에서 **부호가 바뀐다**.
//               줄마다 영교차점을 찾으면 x0 를, 두 줄의 기울기 비로 y0 를 **모형 적합 없이** 얻는다.
//      A13 의 최소제곱 결과(1919.0, 1097.0)와 비교한다.
#include "cache.hpp"
#include "noise.hpp"
#include "hye/plot.hpp"
#include <functional>
#include <iostream>
#include <set>

using namespace a0;
namespace {

struct Block { double x, y; bool upper; };
std::vector<Block> read_blocks(const fs::path& p) {
    cv::FileStorage f(p.u8string(), cv::FileStorage::READ);
    if (!f.isOpened()) throw std::runtime_error("line_protocol.json 없음");
    std::vector<Block> out;
    for (auto it = f["blocks"].begin(); it != f["blocks"].end(); ++it) {
        auto n = *it;
        auto b = n["read_box"];
        out.push_back({0.5 * (double(int(b[0])) + double(int(b[2]))),
                       0.5 * (double(int(b[1])) + double(int(b[3]))),
                       std::string(n["line"]) == "upper"});
    }
    return out;
}
cv::Point2d plate(const Cache& c, int shot) {
    cv::Point2d m(0, 0);
    int n = 0;
    for (int p = 0; p < kPoints; ++p)
        if (c.points[p].selected && c.v(shot)[p]) { m += c.dz(shot, p); ++n; }
    return n ? m * (1.0 / n) : cv::Point2d(0, 0);
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
        auto blocks = read_blocks(repo / fs::u8path(u8"experiments/hye/20261028_1258_final_Y_T_joint/assets/line_protocol.json"));

        // ================= 1부: 보상 이동 영점 시험 (PT-01) =================
        // 세션 안의 조건 쌍 (i, j) 마다 명령 차 (dY, dT) 와 실측 판 이동 차를 모은다.
        struct Cond { std::string set; double y, t; std::vector<int> ids; };
        std::map<int, Cond> cond;
        for (auto& s : c.shots) {
            if (!s.measured || s.t > 1.001) continue;
            auto& k = cond[s.group];
            k.set = s.set; k.y = s.y; k.t = s.t;
            k.ids.push_back(s.idx);
        }
        struct Pair { std::string set; double dy, dt, dx_meas, dy_meas; };
        std::vector<Pair> P;
        for (auto a = cond.begin(); a != cond.end(); ++a)
            for (auto b = std::next(a); b != cond.end(); ++b) {
                if (a->second.set != b->second.set) continue;   // 세션 안에서만 (드리프트 제거)
                double dy = b->second.y - a->second.y, dt = b->second.t - a->second.t;
                if (std::abs(dy) < 1e-9 && std::abs(dt) < 1e-9) continue;
                cv::Point2d pa(0, 0), pb(0, 0);
                for (int i : a->second.ids) pa += plate(c, i);
                for (int i : b->second.ids) pb += plate(c, i);
                pa *= 1.0 / a->second.ids.size();
                pb *= 1.0 / b->second.ids.size();
                P.push_back({a->second.set, dy, dt, pb.x - pa.x, pb.y - pa.y});
            }
        std::cout << u8"세션 안 조건 쌍 " << P.size() << u8"개\n";

        // 세트별로 d(판 dx) = a*dY + b*dT 를 맞춘다. 영점 방향은 b/a.
        Table lv;
        lv.head = {u8"자료", u8"쌍 수", u8"a (px per Y mm)", u8"b (px per T)", u8"지렛대 b/a (mm per T)",
                   u8"등가 반경 (mm)", u8"잔차 RMS px"};
        auto fit_pairs = [&](const std::string& set, const std::string& label) {
            std::vector<const Pair*> v;
            for (auto& p : P) if (set.empty() || p.set == set) v.push_back(&p);
            if (v.size() < 5) return;
            cv::Mat X(int(v.size()), 2, CV_64F), Y(int(v.size()), 1, CV_64F), B;
            for (size_t i = 0; i < v.size(); ++i) {
                X.at<double>(int(i), 0) = v[i]->dy;
                X.at<double>(int(i), 1) = v[i]->dt;
                Y.at<double>(int(i)) = v[i]->dx_meas;
            }
            cv::solve(X, Y, B, cv::DECOMP_SVD);
            cv::Mat R = Y - X * B;
            double rms_ = 0;
            for (int i = 0; i < R.rows; ++i) rms_ += R.at<double>(i) * R.at<double>(i);
            rms_ = std::sqrt(rms_ / R.rows);
            double a = B.at<double>(0), b = B.at<double>(1), lever = b / a;
            lv.add({label, std::to_string(v.size()), num(a, 4), num(b, 4), num(lever, 4),
                    num(lever / (1.0187 * CV_PI / 180), 1), num(rms_, 4)});
        };
        fit_pairs("clean_20261014", u8"1014 (단일축만)");
        fit_pairs("clean_20261020", u8"1020 (혼합 19조건)");
        fit_pairs("clean_20261029", u8"1029 (Y 고정)");
        fit_pairs("", u8"전체");
        lv.csv(results / "lever_from_pairs.csv");

        // 보상에 가까운 쌍: dY ≈ -5.617 dT 인 쌍에서 판이 실제로 안 움직이는가
        Table nl;
        nl.head = {u8"자료", u8"dY", u8"dT", u8"보상 예측 dY", u8"보상 어긋남", u8"실측 판 dx", u8"설명되는가"};
        {
            std::vector<const Pair*> v;
            for (auto& p : P) if (std::abs(p.dt) > 1e-9) v.push_back(&p);
            std::sort(v.begin(), v.end(), [](const Pair* x, const Pair* y) {
                return std::abs(x->dy + kTtoY * x->dt) < std::abs(y->dy + kTtoY * y->dt);
            });
            for (size_t i = 0; i < std::min<size_t>(10, v.size()); ++i) {
                double want = -kTtoY * v[i]->dt;
                nl.add({v[i]->set.substr(6), num(v[i]->dy, 3), num(v[i]->dt, 3), num(want, 4),
                        num(v[i]->dy - want, 4), num(v[i]->dx_meas, 4),
                        num(14.249 * v[i]->dy + 80.06 * v[i]->dt, 4)});
            }
        }
        nl.csv(results / "null_pairs.csv");

        // Cognex 의 1/tan(dθ/2): T 범위가 넓어질 때 중심 오차가 얼마나 줄어드는가
        Table cg;
        cg.head = {u8"T 범위 (명령단위)", u8"물리 각도 (deg)", u8"1/tan(각/2)", u8"현재 대비 중심 오차"};
        for (double r : {1.0, 2.0, 5.0, 10.0, 22.1}) {
            double deg = r * 1.0187, f = 1.0 / std::tan(deg * CV_PI / 360);
            double base = 1.0 / std::tan(1.0187 * CV_PI / 360);
            cg.add({num(r, 1), num(deg, 3), num(f, 1), num(f / base, 3)});
        }
        cg.csv(results / "cognex_scaling.csv");

        // ================= 2부: 대칭·극값으로 축 찾기 (PT-04) =================
        // 선 블록의 T 감도를 블록마다 내고, 줄마다 x 에 대한 직선을 맞춰 영교차점을 찾는다.
        std::vector<int> ids14;
        for (auto& s : c.shots) if (s.measured && s.set == "clean_20261014") ids14.push_back(s.idx);
        const int n14 = int(ids14.size());
        cv::Mat X(n14, 3, CV_64F), YL(n14, kLines, CV_64F);
        for (int i = 0; i < n14; ++i) {
            X.at<double>(i, 0) = c.shots[ids14[i]].y;
            X.at<double>(i, 1) = c.shots[ids14[i]].t;
            X.at<double>(i, 2) = 1;
            for (int b = 0; b < kLines; ++b) YL.at<double>(i, b) = c.line(ids14[i])[b * 2];
        }
        cv::Mat BL;
        cv::solve(X, YL, BL, cv::DECOMP_SVD);
        std::vector<double> aT(kLines);
        for (int b = 0; b < kLines; ++b) aT[b] = BL.at<double>(1, b);

        Table zc;
        zc.head = {u8"줄", u8"블록 수", u8"기울기 (px/deg per px)", u8"절편", u8"영교차 x0", u8"적합 잔차"};
        double x0_up = 0, x0_lo = 0, slope_up = 0, slope_lo = 0, y_up = 0, y_lo = 0;
        for (int up = 1; up >= 0; --up) {
            std::vector<double> xs, ys;
            double ymean = 0;
            for (int b = 0; b < kLines; ++b)
                if (blocks[b].upper == (up == 1)) { xs.push_back(blocks[b].x); ys.push_back(aT[b]); ymean += blocks[b].y; }
            ymean /= xs.size();
            double mx = mean(xs), my = mean(ys), sxy = 0, sxx = 0;
            for (size_t i = 0; i < xs.size(); ++i) { sxy += (xs[i] - mx) * (ys[i] - my); sxx += (xs[i] - mx) * (xs[i] - mx); }
            double m = sxy / sxx, b0 = my - m * mx, x0 = -b0 / m;
            std::vector<double> r;
            for (size_t i = 0; i < xs.size(); ++i) r.push_back(ys[i] - (b0 + m * xs[i]));
            zc.add({up ? u8"위 선" : u8"아래 선", std::to_string(xs.size()), num(m, 8), num(b0, 5), num(x0, 1), num(rms(r), 5)});
            if (up) { x0_up = x0; slope_up = m; y_up = ymean; } else { x0_lo = x0; slope_lo = m; y_lo = ymean; }
        }
        zc.csv(results / "zero_crossing.csv");

        // 두 줄의 기울기 비로 y0: slope ∝ (y - y0) 이므로 slope_up/slope_lo = (y_up - y0)/(y_lo - y0)
        double k = slope_up / slope_lo;
        double y0_sym = (y_up - k * y_lo) / (1 - k);
        double x0_sym = 0.5 * (x0_up + x0_lo);

        Table cmp;
        cmp.head = {u8"값", u8"대칭·극값 (PT-04, 적합 없음)", u8"최소제곱 (A13)", u8"차이", u8"영상 중심"};
        cmp.add({u8"주점 x0", num(x0_sym, 1), "1919.0", num(x0_sym - 1919.0, 1), "1920"});
        cmp.add({u8"주점 y0", num(y0_sym, 1), "1097.0", num(y0_sym - 1097.0, 1), "1080"});
        cmp.add({u8"위 선 영교차", num(x0_up, 1), "-", "-", "-"});
        cmp.add({u8"아래 선 영교차", num(x0_lo, 1), "-", "-", "-"});
        cmp.add({u8"두 줄 기울기 비", num(k, 4), "-", "-", u8"(y_up-y0)/(y_lo-y0)"});
        cmp.csv(results / "symmetry_vs_leastsq.csv");

        // ================= 그림 =================
        {
            hye::Chart ch("A16-161 pair test: commanded vs measured plate dx", "14.249 dY + 80.06 dT [px]", "measured d(plate dx) [px]");
            for (int k2 = 0; k2 < 3; ++k2) {
                const char* sets[] = {"clean_20261014", "clean_20261020", "clean_20261029"};
                std::vector<double> x, y;
                for (auto& p : P) if (p.set == sets[k2]) { x.push_back(14.249 * p.dy + 80.06 * p.dt); y.push_back(p.dx_meas); }
                ch.scatter(x, y, hye::color(k2), std::string(sets[k2]).substr(6), 3);
            }
            ch.save("161_쌍시험.png");
        }
        {
            hye::Chart ch("A16-162 line T-sensitivity vs x, zero crossing", "block x [px]", "dv/dT [px per unit]");
            for (int up = 1; up >= 0; --up) {
                std::vector<double> x, y;
                for (int b = 0; b < kLines; ++b) if (blocks[b].upper == (up == 1)) { x.push_back(blocks[b].x); y.push_back(aT[b]); }
                ch.scatter(x, y, hye::color(up), up ? "upper" : "lower", 5);
            }
            ch.hline(0, hye::kGray);
            ch.vline(x0_sym, hye::kRed);
            ch.vline(1919.0, hye::color(2));
            ch.save("162_영교차.png");
        }
        {
            std::vector<std::string> cats;
            std::vector<double> v;
            for (auto& r : cg.rows) { cats.push_back(r[0] + u8"단위"); v.push_back(std::stod(r[3])); }
            hye::bar_chart("163_T범위효과.png", "A16-163 center-estimate error vs T range (Cognex 1/tan)", cats, v, "relative error");
        }

        std::ostringstream o;
        o << u8"# A16 결과 — 원인분석 특허 두 건을 실제로 돌린다\n\n"
          << u8"`실험/src/main.cpp` 가 다시 쓴다. 해석은 [보고서](보고서.md).\n\n"
          << u8"## 1부. 보상 이동 영점 시험 (PT-01 Cognex)\n\n"
          << u8"세션 안의 조건 쌍 " << P.size() << u8"개를 썼다. 쌍을 쓰면 세션 영점 드리프트가 빠진다.\n\n"
          << u8"### 지렛대 (등가 회전 반경)\n\n" << lv.md() << "\n"
          << hye::img_md("161_쌍시험.png", "pair test") << "\n\n"
          << u8"### 보상에 가장 가까운 쌍\n\n" << nl.md() << "\n"
          << u8"### Cognex 의 1/tan(각/2) — T 범위를 넓히면\n\n" << cg.md() << "\n"
          << hye::img_md("163_T범위효과.png", "T range") << "\n\n"
          << u8"## 2부. 대칭·극값으로 축 찾기 (PT-04 Kawasaki)\n\n"
          << u8"선 블록의 T 감도가 x 에 대해 직선이고 x0 에서 부호가 바뀐다. **모형 적합 없이** 영교차점만 찾는다.\n\n"
          << zc.md() << "\n" << cmp.md() << "\n"
          << hye::img_md("162_영교차.png", "zero crossing") << "\n";
        write_text(result / fs::u8path(u8"결과.md"), o.str());
        std::cout << u8"끝\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << u8"오류: " << e.what() << "\n";
        return 1;
    }
}
