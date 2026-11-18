// A1: 반복 잡음, 점 간 상관(H11), 세션 영점(H12).
//  raw 잔차  = 묶음 평균을 뺀 점 변위           (측정 잡음 + 반복 사이 실제 자세 차이)
//  affine 잔차 = raw 에서 QR별 아핀 1개를 더 뺀 것 (측정 잡음만)
//  N_eff = (tr S)^2 / ||S||_F^2  (S = R R^T, 잔차 행렬의 그람 행렬. 1260x1260 공분산을 만들지 않는다)
#include "cache.hpp"
#include "hye/plot.hpp"
#include <functional>
#include <set>
#include <iostream>

using namespace a0;
namespace {

const cv::Rect kQr[3] = {cv::Rect(1552, 969, 184, 198), cv::Rect(1809, 971, 184, 198), cv::Rect(2067, 973, 185, 198)};
const double kOutlier = 1.0;  // 아핀 잔차가 이보다 크면 빼고 다시 맞춘다 (px)

// r(630,2) 에서 QR별 아핀 d = B[1,u,v] 를 빼고 남긴다. use 가 0 인 점은 건드리지 않고 그대로 둔다.
void remove_affine(const std::vector<Point>& pts, std::vector<cv::Point2d>& r, const std::vector<unsigned char>& use) {
    for (int q = 0; q < 3; ++q) {
        std::vector<int> idx;
        for (int i = 0; i < kPoints; ++i)
            if (pts[i].region == q && use[i] && std::isfinite(r[i].x) && std::isfinite(r[i].y)) idx.push_back(i);
        if (idx.size() < 6) continue;
        cv::Point2d c(kQr[q].x + kQr[q].width / 2.0, kQr[q].y + kQr[q].height / 2.0);
        auto fit = [&](const std::vector<int>& rows) {
            cv::Mat X(int(rows.size()), 3, CV_64F), Y(int(rows.size()), 2, CV_64F), B;
            for (int k = 0; k < int(rows.size()); ++k) {
                X.at<double>(k, 0) = 1;
                X.at<double>(k, 1) = (pts[rows[k]].x - c.x) / 100;
                X.at<double>(k, 2) = (pts[rows[k]].y - c.y) / 100;
                Y.at<double>(k, 0) = r[rows[k]].x;
                Y.at<double>(k, 1) = r[rows[k]].y;
            }
            cv::solve(X, Y, B, cv::DECOMP_SVD);
            return B;
        };
        auto model = [&](const cv::Mat& B, int i) {
            double u = (pts[i].x - c.x) / 100, v = (pts[i].y - c.y) / 100;
            return cv::Point2d(B.at<double>(0, 0) + B.at<double>(1, 0) * u + B.at<double>(2, 0) * v,
                               B.at<double>(0, 1) + B.at<double>(1, 1) * u + B.at<double>(2, 1) * v);
        };
        cv::Mat B = fit(idx);
        std::vector<int> in;
        for (int i : idx) if (cv::norm(r[i] - model(B, i)) <= kOutlier) in.push_back(i);
        if (in.size() >= 6 && in.size() < idx.size()) B = fit(in);
        for (int i : idx) r[i] -= model(B, i);
    }
}

// 잔차 행렬의 그람 행렬 S 로부터 N_eff 와 최대 고유값 비율을 낸다.
struct Spread {
    double tr = 0, fro2 = 0, neff = 0, top = 0;
};
Spread spread_from_rows(const std::vector<std::vector<double>>& R) {
    const int N = int(R.size());
    Spread s;
    if (N < 2) return s;
    cv::Mat S(N, N, CV_64F, 0.0);
    for (int i = 0; i < N; ++i)
        for (int j = i; j < N; ++j) {
            double v = 0;
            for (size_t k = 0; k < R[i].size(); ++k) v += R[i][k] * R[j][k];
            S.at<double>(i, j) = S.at<double>(j, i) = v;
        }
    for (int i = 0; i < N; ++i) s.tr += S.at<double>(i, i);
    for (int i = 0; i < N; ++i) for (int j = 0; j < N; ++j) s.fro2 += S.at<double>(i, j) * S.at<double>(i, j);
    s.neff = s.fro2 > 0 ? s.tr * s.tr / s.fro2 : 0;
    // 거듭제곱 반복으로 최대 고유값
    cv::Mat v(N, 1, CV_64F, 1.0 / std::sqrt(double(N)));
    double lam = 0;
    for (int it = 0; it < 60; ++it) {
        cv::Mat w = S * v;
        lam = cv::norm(w);
        if (lam < 1e-30) break;
        v = w / lam;
    }
    s.top = s.tr > 0 ? lam / s.tr : 0;
    return s;
}

}  // namespace

int main() {
    try {
        const fs::path result = fs::u8path(EXPERIMENT_DIR);
        const fs::path results = result / "results";
        fs::create_directories(results);
        hye::begin_results();
        Cache c = load(cache_dir());
        std::cout << "캐시 " << c.n() << "장\n";

        // ---- 묶음 ----
        std::map<int, std::vector<int>> group;
        for (auto& s : c.shots) if (s.measured) group[s.group].push_back(s.idx);
        std::vector<int> repeat_groups;
        for (auto& g : group) if (g.second.size() >= 2) repeat_groups.push_back(g.first);
        std::sort(repeat_groups.begin(), repeat_groups.end());
        std::cout << "반복 묶음 " << repeat_groups.size() << "개\n";

        // ---- 잔차 ----
        struct Res {
            int shot, group;
            std::string set;
            double t, y;
            std::vector<cv::Point2d> raw, aff;
            std::vector<unsigned char> use;
            bool full = false;  // 630점이 모두 유효한 묶음인가 (T+2.00 은 아니다)
        };
        std::vector<Res> res;
        Table gt;
        gt.head = {"group", "set", "y", "t", "장수", "raw RMS px", "affine RMS px", "한 장 잡음 sigma px",
                   "raw 공통모드 비율", "affine 공통모드 비율"};
        for (int g : repeat_groups) {
            auto& ids = group[g];
            const int m = int(ids.size());
            // 모든 반복에서 유효한 점만 쓴다
            std::vector<unsigned char> use(kPoints, 1);
            for (int i : ids) for (int p = 0; p < kPoints; ++p) if (!c.v(i)[p]) use[p] = 0;
            std::vector<cv::Point2d> mean(kPoints, {0, 0});
            for (int i : ids) for (int p = 0; p < kPoints; ++p) mean[p] += c.dz(i, p);
            for (auto& v : mean) v *= 1.0 / m;
            std::vector<double> rr, ra;
            double cm_raw = 0, cm_aff = 0, en_raw = 0, en_aff = 0;
            for (int i : ids) {
                Res r;
                r.shot = i; r.group = g; r.set = c.shots[i].set;
                r.y = c.shots[i].y; r.t = c.shots[i].t; r.use = use;
                r.full = std::count(use.begin(), use.end(), 1) == kPoints;
                r.raw.assign(kPoints, {0, 0});
                for (int p = 0; p < kPoints; ++p) r.raw[p] = c.dz(i, p) - mean[p];
                r.aff = r.raw;
                remove_affine(c.points, r.aff, use);
                cv::Point2d mr(0, 0), ma(0, 0);
                int cnt = 0;
                for (int p = 0; p < kPoints; ++p) {
                    if (!use[p]) continue;
                    ++cnt;
                    rr.push_back(r.raw[p].x); rr.push_back(r.raw[p].y);
                    ra.push_back(r.aff[p].x); ra.push_back(r.aff[p].y);
                    mr += r.raw[p]; ma += r.aff[p];
                    en_raw += r.raw[p].dot(r.raw[p]);
                    en_aff += r.aff[p].dot(r.aff[p]);
                }
                if (cnt) { mr *= 1.0 / cnt; ma *= 1.0 / cnt; }
                cm_raw += cnt * mr.dot(mr);
                cm_aff += cnt * ma.dot(ma);
                res.push_back(std::move(r));
            }
            double scale = std::sqrt(double(m) / (m - 1));
            gt.add({std::to_string(g), c.shots[ids[0]].set.substr(6), num(c.shots[ids[0]].y, 2), num(c.shots[ids[0]].t, 2),
                    std::to_string(m), num(rms(rr), 4), num(rms(ra), 4), num(rms(ra) * scale, 4),
                    num(en_raw > 0 ? cm_raw / en_raw : 0, 3), num(en_aff > 0 ? cm_aff / en_aff : 0, 3)});
        }
        gt.csv(results / "noise_by_group.csv");

        // ---- 점별 sigma ----
        Table pst;
        pst.head = {"point", "x", "y", "region", "selected", "n", "sigma_x_px", "sigma_y_px", "sigma_px"};
        std::vector<double> sig(kPoints, 0);
        for (int p = 0; p < kPoints; ++p) {
            std::vector<double> xs, ys;
            for (auto& r : res) if (r.use[p]) { xs.push_back(r.aff[p].x); ys.push_back(r.aff[p].y); }
            double sx = rms(xs), sy = rms(ys);
            // m=3 묶음 평균을 뺐으므로 한 장의 잡음으로 되돌린다
            double k = std::sqrt(3.0 / 2.0);
            sx *= k; sy *= k;
            sig[p] = std::sqrt(0.5 * (sx * sx + sy * sy));
            pst.add({std::to_string(p), num(c.points[p].x, 1), num(c.points[p].y, 1), std::to_string(c.points[p].region),
                     std::to_string(c.points[p].selected), std::to_string(xs.size()), num(sx, 5), num(sy, 5), num(sig[p], 5)});
        }
        pst.csv(results / "sigma_by_point.csv");

        // ---- N_eff ----
        // 모든 잔차 장에서 공통으로 유효한 점만 쓴다. 장마다 길이가 다르면 그람 행렬이 틀린다.
        // 630점이 모두 유효한 묶음만 쓴다. 범위 밖(T+2.00)에서 점이 깨진 묶음은 따로 센다.
        std::vector<unsigned char> common(kPoints, 1);
        for (auto& r : res) if (r.full) for (int p = 0; p < kPoints; ++p) if (!r.use[p]) common[p] = 0;
        int dropped_partial = 0;
        std::set<int> partial_group;
        for (auto& r : res) if (!r.full) { ++dropped_partial; partial_group.insert(r.group); }
        std::cout << "점이 깨진 장 " << dropped_partial << " (묶음 " << partial_group.size() << ")" << std::endl;
        int common_n = int(std::count(common.begin(), common.end(), 1));
        std::cout << "공통 유효점 " << common_n << "/630\n";
        // 한 장이라도 아핀 잔차가 큰 묶음은 따로 세고, N_eff 는 그 묶음을 뺀 집합에서도 낸다.
        Table out_t;
        out_t.head = {"group", "set", "y", "t", "한 장 잡음 sigma px"};
        std::set<int> outlier_group;
        for (auto& r : gt.rows) if (std::stod(r[7]) > 0.05) { outlier_group.insert(std::stoi(r[0])); out_t.add({r[0], r[1], r[2], r[3], r[7]}); }
        out_t.csv(results / "outlier_groups.csv");

        Table nt;
        nt.head = {"점 집합", "명목 차원", "잔차 장수", "tr S", "N_eff", "N_eff / 명목", "최대 고유값 비율"};
        auto rows_for = [&](std::function<bool(int)> keep, bool affine, bool drop_outlier) {
            std::vector<std::vector<double>> R;
            for (auto& r : res) {
                if (!r.full) continue;
                if (drop_outlier && outlier_group.count(r.group)) continue;
                std::vector<double> v;
                for (int p = 0; p < kPoints; ++p) {
                    if (!common[p] || !keep(p)) continue;
                    v.push_back(affine ? r.aff[p].x : r.raw[p].x);
                    v.push_back(affine ? r.aff[p].y : r.raw[p].y);
                }
                R.push_back(std::move(v));
            }
            return R;
        };
        struct SetDef { std::string name; std::function<bool(int)> keep; int nominal; };
        std::vector<SetDef> sets = {
            {"630점 전부", [&](int p) { return true; }, kPoints},
            {"선택 536점", [&](int p) { return c.points[p].selected == 1; }, 536},
            {"QR1", [&](int p) { return c.points[p].region == 0; }, 0},
            {"QR2", [&](int p) { return c.points[p].region == 1; }, 0},
            {"QR3", [&](int p) { return c.points[p].region == 2; }, 0},
        };
        for (auto& d : sets) {
            int nom = 0;
            for (int p = 0; p < kPoints; ++p) if (common[p] && d.keep(p)) ++nom;
            for (int drop = 0; drop < 2; ++drop) {
                auto R = rows_for(d.keep, true, drop == 1);
                auto s = spread_from_rows(R);
                nt.add({d.name + (drop ? " (affine, 이상 묶음 제외)" : " (affine 잔차)"), std::to_string(nom),
                        std::to_string(R.size()), num(s.tr, 4), num(s.neff, 2), num(nom ? s.neff / nom : 0, 4), num(s.top, 3)});
            }
        }
        {
            auto R = rows_for([&](int p) { return c.points[p].selected == 1; }, false, false);
            auto s = spread_from_rows(R);
            int nom = 0;
            for (int p = 0; p < kPoints; ++p) if (common[p] && c.points[p].selected) ++nom;
            nt.add({"선택 536점 (raw 잔차)", std::to_string(nom), std::to_string(R.size()), num(s.tr, 4), num(s.neff, 2),
                    num(nom ? s.neff / nom : 0, 4), num(s.top, 3)});
        }
        nt.csv(results / "neff.csv");

        // ---- 세션 영점 ----
        // 판 전체 평행이동 성분(점 평균 dx, dy)을 사진마다 구해 둔다.
        auto plate = [&](int i) {
            cv::Point2d m(0, 0);
            int cnt = 0;
            for (int p = 0; p < kPoints; ++p) if (c.v(i)[p] && c.points[p].selected) { m += c.dz(i, p); ++cnt; }
            return cnt ? m * (1.0 / cnt) : cv::Point2d(0, 0);
        };
        auto group_ids = [&](const std::string& set, double y, double t) {
            std::vector<int> out;
            for (auto& s : c.shots)
                if (s.set == set && std::abs(s.y - y) < 1e-9 && std::abs(s.t - t) < 1e-9 && s.measured) out.push_back(s.idx);
            return out;
        };
        auto group_plate = [&](const std::vector<int>& ids) {
            cv::Point2d m(0, 0);
            for (int i : ids) m += plate(i);
            return ids.empty() ? cv::Point2d(0, 0) : m * (1.0 / ids.size());
        };

        // 세션 내 반복 SD (판 평행이동 dx, dy 각각)
        std::vector<double> within, within_y;
        for (int g : repeat_groups) {
            std::vector<double> dx, dy;
            for (int i : group[g]) { dx.push_back(plate(i).x); dy.push_back(plate(i).y); }
            if (std::isfinite(sd(dx))) within.push_back(sd(dx));
            if (std::isfinite(sd(dy))) within_y.push_back(sd(dy));
        }
        const double within_sd = median(within), within_sd_y = median(within_y);

        Table zt;
        zt.head = {"비교", "판 dx 차이 px", "판 dy 차이 px", "Y 환산 mm", "T 환산", "dx SD 배수", "dy SD 배수"};
        auto add_zero = [&](const std::string& label, cv::Point2d diff) {
            zt.add({label, num(diff.x, 4), num(diff.y, 4), num(diff.x / kPxPerMmY, 4), num(diff.x / kPxPerT, 4),
                    num(within_sd > 0 ? std::abs(diff.x) / within_sd : 0, 2),
                    num(within_sd_y > 0 ? std::abs(diff.y) / within_sd_y : 0, 2)});
        };
        auto c1020 = group_ids("clean_20261020", 0.0, 0.0);
        if (!c1020.empty()) add_zero("1014 center -> 1020 기준 위치 3장", group_plate(c1020));
        {
            auto a = group_ids("clean_20261020", 0.05, 0.10);
            auto b1 = group_ids("clean_20261029", 0.05, 0.09);
            auto b2 = group_ids("clean_20261029", 0.05, 0.11);
            if (!a.empty() && !b1.empty() && !b2.empty()) {
                cv::Point2d interp = (group_plate(b1) + group_plate(b2)) * 0.5;
                add_zero("1020 Y+0.05 T+0.10 -> 1029 T+0.09/0.11 내삽", group_plate(a) - interp);
            }
        }
        zt.csv(results / "session_zero.csv");

        // ---- 그림 ----
        cv::Mat center = hye::imread_u8((fs::u8path(REPO_DIR) / "data/clean/clean_20261014/center.bmp").u8string(), cv::IMREAD_GRAYSCALE);
        {  // 11 점별 잡음 지도
            double lo = quantile(std::vector<double>(sig.begin(), sig.end()), 0.05);
            double hi = quantile(std::vector<double>(sig.begin(), sig.end()), 0.95);
            std::vector<cv::Mat> tiles;
            for (int q = 0; q < 3; ++q) {
                std::vector<hye::Mark> marks;
                for (int p = 0; p < kPoints; ++p) {
                    if (c.points[p].region != q) continue;
                    double u = hi > lo ? std::clamp((sig[p] - lo) / (hi - lo), 0.0, 1.0) : 0.5;
                    marks.push_back({{c.points[p].x, c.points[p].y}, {0, 0}, "", cv::Scalar(255 * (1 - u), 60, 255 * u)});
                }
                cv::Rect roi = kQr[q];
                roi.x -= 14; roi.y -= 14; roi.width += 28; roi.height += 28;
                char cap[120];
                std::snprintf(cap, sizeof cap, "A1-11 QR%d point noise sigma  blue=%.4f px  red=%.4f px", q + 1, lo, hi);
                tiles.push_back(hye::zoom(center, roi, 480, marks, 0, cap));
            }
            hye::save_png("11_점별잡음지도.png", hye::hstack(tiles));
        }
        {  // 12 N_eff
            std::vector<std::string> cats;
            std::vector<double> vals;
            for (auto& r : nt.rows) { cats.push_back(r[0]); vals.push_back(std::stod(r[4])); }
            hye::bar_chart("12_유효점수.png", "A1-12 effective independent dimensions (N_eff)", cats, vals, "N_eff", -1, 536 * 0.1);
        }
        {  // 13 공통 모드
            std::vector<double> raw, aff;
            for (auto& r : gt.rows) { raw.push_back(std::stod(r[8])); aff.push_back(std::stod(r[9])); }
            hye::Chart ch("A1-13 common-mode share of residual variance (per group)", "group index", "share");
            std::vector<double> x(raw.size());
            for (size_t i = 0; i < x.size(); ++i) x[i] = double(i);
            ch.scatter(x, raw, hye::color(0), "raw residual", 3);
            ch.scatter(x, aff, hye::color(1), "after per-QR affine", 3);
            ch.hline(0.5, hye::kRed);
            ch.save("13_공통모드.png");
        }
        {  // 14 날짜 영점 변위장
            std::vector<cv::Mat> tiles;
            auto draw = [&](const std::vector<int>& ids, const std::string& cap) {
                if (ids.empty()) return;
                std::vector<cv::Point2d> d(kPoints, {0, 0});
                for (int i : ids) for (int p = 0; p < kPoints; ++p) d[p] += c.dz(i, p);
                for (auto& v : d) v *= 1.0 / ids.size();
                cv::Point2d m(0, 0);
                for (int p = 0; p < kPoints; ++p) m += d[p];
                m *= 1.0 / kPoints;
                double big = 0;
                for (int p = 0; p < kPoints; ++p) big = std::max(big, cv::norm(d[p]));
                double g = big > 1e-9 ? 40 / big : 1;
                std::vector<hye::Mark> marks;
                for (int p = 0; p < kPoints; p += 7) marks.push_back({{c.points[p].x, c.points[p].y}, d[p], "", hye::kRed});
                cv::Rect roi(kQr[0].x - 20, kQr[0].y - 30, kQr[2].x + kQr[2].width - kQr[0].x + 40, kQr[0].height + 60);
                char t[200];
                std::snprintf(t, sizeof t, "A1-14 %s  mean dx=%.3f px (Y %.4f mm / T %.4f)  arrow x%.1f",
                              cap.c_str(), m.x, m.x / kPxPerMmY, m.x / kPxPerT, g);
                tiles.push_back(hye::zoom(center, roi, 1500, marks, g, t));
            };
            draw(c1020, "1020 center minus 1014 center");
            if (!tiles.empty()) hye::save_png("14_날짜영점_변위장.png", hye::vstack(tiles));
        }
        {  // 15 묶음별 잡음 대 T 크기
            hye::Chart ch("A1-15 per-group measurement noise vs commanded T", "T command [deg]", "single-shot sigma [px]");
            for (int k = 0; k < 2; ++k) {
                const char* set = k ? "clean_20261029" : "clean_20261020";
                std::vector<double> x, y;
                for (auto& r : gt.rows) if (r[1] == std::string(set).substr(6)) { x.push_back(std::stod(r[3])); y.push_back(std::stod(r[7])); }
                ch.scatter(x, y, hye::color(k), std::string(set).substr(6), 4);
            }
            ch.save("15_묶음별잡음.png");
        }

        // ---- 결과.md ----
        std::vector<double> allsig(sig.begin(), sig.end());
        std::vector<double> gsig;
        for (auto& r : gt.rows) gsig.push_back(std::stod(r[7]));
        std::vector<double> graw;
        for (auto& r : gt.rows) graw.push_back(std::stod(r[5]));
        std::vector<double> gcm;
        for (auto& r : gt.rows) gcm.push_back(std::stod(r[9]));

        double neff536_all = std::stod(nt.rows[2][4]);   // 선택 536점, affine 잔차, 전체 묶음
        double neff536 = std::stod(nt.rows[3][4]);       // 같은 것, 이상 묶음 제외 (주 판정값)
        double nom536 = std::stod(nt.rows[3][1]);
        bool h11_neff = neff536 < nom536 * 0.10;
        std::ostringstream o;
        o << "# A1 결과 — 반복 잡음, 점 간 상관, 세션 영점\n\n"
          << "`실험/src/main.cpp` 가 다시 쓴다. 해석과 판정은 [보고서](보고서.md). 계획은 [계획서](../계획서.md).\n\n"
          << "허용 절대오차 ±" << num(kTol, 3) << ", 판정용 '작다' 경계 " << num(kTight, 3) << " (2026-11-02 기준).\n\n"
          << "## 1. 반복 잡음\n\n"
          << "반복 2장 이상인 묶음 " << repeat_groups.size() << "개, 잔차 " << res.size() << "장.\n\n"
          << "| 값 | 중앙값 | p95 | 최대 |\n|---|---|---|---|\n"
          << "| raw 잔차 RMS (px) | " << num(median(graw), 4) << " | " << num(quantile(graw, .95), 4) << " | " << num(max_abs(graw), 4) << " |\n"
          << "| 한 장 측정 잡음 sigma (px) | " << num(median(gsig), 4) << " | " << num(quantile(gsig, .95), 4) << " | " << num(max_abs(gsig), 4) << " |\n\n"
          << "raw 와 affine 의 차이는 **반복 촬영 사이에 판이 실제로 움직인 양**이다. 측정 잡음이 아니다.\n\n"
          << "묶음별 값: [noise_by_group.csv](results/noise_by_group.csv)\n\n"
          << hye::img_md("15_묶음별잡음.png", "noise vs T") << "\n\n"
          << "## 2. 점별 잡음\n\n"
          << "| 값 | 중앙값 | p05 | p95 |\n|---|---|---|---|\n"
          << "| 점별 sigma (px) | " << num(median(allsig), 5) << " | " << num(quantile(allsig, .05), 5) << " | " << num(quantile(allsig, .95), 5) << " |\n\n"
          << hye::img_md("11_점별잡음지도.png", "point noise map") << "\n\n"
          << "## 3. 유효 독립 차원 (H11)\n\n" << nt.md() << "\n"
          << hye::img_md("12_유효점수.png", "N_eff") << "\n\n"
          << "빨간 선은 명목 536점의 10% 다. **N_eff 가 이 선 아래면 H11 성립 조건 하나를 만족한다.**\n\n"
          << "이상 묶음(한 장 잡음 > 0.05 px): " << outlier_group.size() << "개, [outlier_groups.csv](results/outlier_groups.csv)\n\n"
          << "N_eff 는 630점이 모두 유효한 묶음에서만 냈다. 점이 깨진 장 " << dropped_partial << "장(묶음 " << partial_group.size()
          << "개)은 전부 **T+2.00** 이고, 이는 1014 학습 범위 ±1.00 밖이다. 범위 밖 실패로 따로 센다.\n\n"
          << "## 4. 공통 모드\n\n"
          << "| 잔차 | 공통 모드 비율 중앙값 |\n|---|---|\n";
        {
            std::vector<double> rawcm;
            for (auto& r : gt.rows) rawcm.push_back(std::stod(r[8]));
            o << "| raw | " << num(median(rawcm), 3) << " |\n| affine 뺀 뒤 | " << num(median(gcm), 3) << " |\n\n";
        }
        o << "**계획서의 '공통 모드 > 50%' 판정 기준은 이 자료에서 쓸 수 없다.** 이유는 두 가지다.\n"
          << "raw 잔차의 공통 모드는 반복 촬영 사이에 판이 실제로 움직인 양이라 측정 잡음의 상관이 아니다. "
          << "affine 을 뺀 잔차는 QR별 아핀에 상수항이 들어 있어 공통 모드가 정의상 0 이 된다. "
          << "그래서 H11 판정은 N_eff 하나로만 한다. (판정 기준 변경: 2026-11-02, 결과를 본 뒤)\n\n"
          << hye::img_md("13_공통모드.png", "common mode") << "\n\n"
          << "## 5. 세션 영점 (H12)\n\n" << zt.md() << "\n"
          << "세션 내 반복 SD 중앙값은 판 dx " << num(within_sd, 4) << " px, dy " << num(within_sd_y, 4) << " px 다. "
          << "측정 잡음(" << num(median(gsig), 4) << " px)보다 훨씬 크므로, 반복 3장 사이에 판이 실제로 움직였다는 뜻이다.\n\n"
          << hye::img_md("14_날짜영점_변위장.png", "session zero") << "\n\n"
          << "Y 환산과 T 환산을 **둘 다** 적었다. 판 가로 이동만 보면 같은 dx 가 두 해석을 모두 허용하기 때문이다. "
          << "둘 중 하나로 단정하지 않는다.\n\n"
          << "## 6. 판정 기준 대비\n\n"
          << "| 가설 | 조건 | 실측 | 결과 |\n|---|---|---|---|\n"
          << "| H11 | N_eff(선택점, 이상 묶음 제외) < 명목의 10% = " << num(nom536 * 0.1, 1) << " | " << num(neff536, 2)
          << " (전체 묶음일 때 " << num(neff536_all, 2) << ") | " << (h11_neff ? "**성립**" : "기각") << " |\n"
          << "| H11 | 공통 모드 > 0.5 | 쓸 수 없음 (4절) | 판정에서 제외 |\n";
        for (auto& r : zt.rows) {
            double ey = std::abs(std::stod(r[3])), et = std::abs(std::stod(r[4])), k = std::stod(r[5]);
            bool ok = k > 3 && (ey > kTight || et > kTight);
            o << "| H12 | " << r[0] << ": dx SD 3배 초과 그리고 환산 > " << num(kTight, 3) << " | " << num(k, 2)
              << "배, Y " << num(ey, 4) << " / T " << num(et, 4) << " | " << (ok ? "**성립**" : "기각") << " |\n";
        }
        o << "\n## 7. 이 실험이 답하지 못한 것\n\n"
          << "- 반복 3장 사이에 로봇을 다시 움직였는지 기록이 없다. raw 잔차의 대부분은 측정이 아니라 판의 실제 움직임이지만, "
          << "그것이 재현 오차인지 드리프트인지 이 자료로는 나눌 수 없다. B1 촬영이 필요하다.\n"
          << "- 1014 에는 반복 촬영이 없어 세 날짜의 잡음을 같은 방식으로 비교하지 못했다.\n"
          << "- 날짜 간 비교는 두 경로 모두 서로 다른 명령이나 내삽을 거친다. 같은 명령의 날짜 간 직접 비교 자료가 없다.\n";
        write_text(result / fs::u8path(u8"결과.md"), o.str());
        std::cout << "끝\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "오류: " << e.what() << "\n";
        return 1;
    }
}
