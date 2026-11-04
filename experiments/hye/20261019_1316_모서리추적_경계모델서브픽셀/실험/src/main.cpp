// QR 격자 모서리 755점을 center 에서 ID 로 고정하고 281장 모두에서 같은 경계 모델로 서브픽셀 측정한 뒤,
// 한 칸 이동량과 점별·사진 공통 흔들림을 계산한다.
// 원본: analysis_grid_corners_20261019/track_corners.py, corner_steps.py (RULE=valid/nondiag/all), viz_trials.py
#include <hye/dataset.hpp>
#include <hye/plot.hpp>
#include <hye/report.hpp>
#include <omp.h>
#include <iostream>
#include <map>
#include <set>
#include "track.hpp"

using namespace hye;

namespace {

const std::vector<std::string> kTrials = {"Y 0.1 간격 +", "Y 0.1 간격 -", "Y 0.05 홀수 +", "Y 0.05 홀수 -", "Θ +", "Θ -"};
const std::vector<std::string> kTrialEn = {"Y0.1+", "Y0.1-", "Y0.05odd+", "Y0.05odd-", "T+", "T-"};

std::string base_name(const std::string& rel) { return rel.substr(rel.rfind('/') + 1); }
std::string stem(const std::string& f) { return f.substr(0, f.size() - 4); }

int trial_of(const Shot& s) {
    if (s.condition == "center") return -1;
    std::string f = base_name(s.rel);
    double v = std::stod(f.substr(1, f.size() - 5));
    bool pos = v > 0;
    if (f[0] == 'T') return pos ? 4 : 5;
    bool tenth = (long long)std::nearbyint(std::abs(v) * 100) % 10 == 0;
    return tenth ? (pos ? 0 : 1) : (pos ? 2 : 3);
}
double value_of(const Shot& s) {
    if (s.condition == "center") return 0;
    std::string f = base_name(s.rel);
    return std::stod(f.substr(1, f.size() - 5));
}

double sd(const std::vector<double>& v, int ddof) {
    double m = mean(v), s = 0;
    for (double x : v) s += (x - m) * (x - m);
    return std::sqrt(s / (v.size() - ddof));
}
double median(std::vector<double> v) { return percentile(std::move(v), 0.5); }

// numpy.polyfit(v, y, 2) 후 polyval 의 남은 값
std::vector<double> quad_resid(const std::vector<double>& v, const std::vector<double>& y) {
    cv::Mat A((int)v.size(), 3, CV_64F), B((int)v.size(), 1, CV_64F), C;
    for (int i = 0; i < A.rows; ++i) {
        A.at<double>(i, 0) = v[i] * v[i]; A.at<double>(i, 1) = v[i]; A.at<double>(i, 2) = 1;
        B.at<double>(i) = y[i];
    }
    cv::solve(A, B, C, cv::DECOMP_SVD);
    std::vector<double> r(v.size());
    for (size_t i = 0; i < v.size(); ++i) r[i] = y[i] - (C.at<double>(0) * v[i] * v[i] + C.at<double>(1) * v[i] + C.at<double>(2));
    return r;
}

struct Step {
    int trial;
    size_t from, to;  // shots 인덱스
    double px, spread;
    int n;
    bool first;
    double qr_px[3];
};
struct StepStats { int count; double mean, std, min, max, spread_med; };
struct Noise { int trial; int comp; double common; int npts; double total_med, point_med, point_p90, point_max; };

}  // namespace

static int run();
int main() {
    try {
        return run();
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}

static int run() {
    begin_results();
    omp_set_num_threads(4);
    auto shots = load_1014();
    const size_t NS = shots.size();
    tk::Tracker T;
    T.init(read_gray(center_1014(shots)));
    const size_t NC = T.corners.size();
    std::cout << "corners " << NC << " photos " << NS << "\n";

    // ---- 281장 추적 ----
    std::vector<std::vector<tk::Meas>> M(NS);
    int done = 0;
#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < (int)NS; ++i) {
        cv::Mat img = read_gray(shots[i]);
        M[i] = T.track(img);
#pragma omp critical
        {
            if (++done % 40 == 0) std::cout << done << "/" << NS << "\n";
        }
    }
    size_t ci = 0;
    for (size_t i = 0; i < NS; ++i) if (shots[i].condition == "center") ci = i;
    std::vector<int> trial(NS);
    std::vector<double> val(NS);
    for (size_t i = 0; i < NS; ++i) { trial[i] = trial_of(shots[i]); val[i] = value_of(shots[i]); }

    // 전체 측정 CSV (크다: 결과/out/ 은 git 제외)
    {
        fs::create_directories(out("out"));
        Table t{{"file", "id", "qr", "grid_x", "grid_y", "black_neighbors", "x", "y", "rmse", "window_stability_px", "contrast", "valid", "reason"}};
        for (size_t i = 0; i < NS; ++i)
            for (size_t k = 0; k < NC; ++k) {
                auto& c = T.corners[k];
                auto& m = M[i][k];
                t.add({base_name(shots[i].rel), std::to_string(k), std::to_string(c.qr), std::to_string(c.gx), std::to_string(c.gy), std::to_string(c.black),
                       fmt(m.x, 5), fmt(m.y, 5), fmt(m.rmse, 4), fmt(m.stab, 5), fmt(m.contrast, 3), m.valid ? "True" : "False", m.reason});
            }
        t.csv(out("out") / "corners_281.csv");
    }

    // ---- 사진별·trial별 통과 (viz_trials.py) ----
    struct TS { int photos, always_ok, sometimes_bad, always_bad, pmin, pmax; double rate; };
    std::vector<TS> ts(6);
    std::vector<std::vector<size_t>> tfiles(6);
    for (size_t i = 0; i < NS; ++i) if (trial[i] >= 0) tfiles[trial[i]].push_back(i);
    for (auto& f : tfiles) std::stable_sort(f.begin(), f.end(), [&](size_t a, size_t b) { return std::abs(val[a]) < std::abs(val[b]); });
    std::vector<int> per_photo(NS, 0);
    for (size_t i = 0; i < NS; ++i) for (auto& m : M[i]) per_photo[i] += m.valid;
    std::vector<int> nbad_all(NC, 0);  // 280장 중 불확실 횟수
    for (int t = 0; t < 6; ++t) {
        auto& S = ts[t];
        S = {(int)tfiles[t].size(), 0, 0, 0, 1 << 30, 0, 0};
        long long vsum = 0;
        for (size_t k = 0; k < NC; ++k) {
            int nb = 0;
            for (size_t i : tfiles[t]) nb += !M[i][k].valid;
            nbad_all[k] += nb;
            S.always_ok += nb == 0;
            S.sometimes_bad += nb > 0 && nb < S.photos;
            S.always_bad += nb == S.photos;
        }
        for (size_t i : tfiles[t]) { vsum += per_photo[i]; S.pmin = std::min(S.pmin, per_photo[i]); S.pmax = std::max(S.pmax, per_photo[i]); }
        S.rate = std::nearbyint(100.0 * vsum / ((double)S.photos * NC) * 100) / 100;
    }
    std::map<std::string, long long> reason_cnt;
    long long total_valid = 0;
    std::map<int, std::array<long long, 2>> by_black;
    for (size_t i = 0; i < NS; ++i)
        for (size_t k = 0; k < NC; ++k) {
            auto& m = M[i][k];
            total_valid += m.valid;
            by_black[T.corners[k].black][0]++;
            by_black[T.corners[k].black][1] += m.valid;
            size_t p = 0;
            while (!m.reason.empty() && p != std::string::npos) {
                size_t e = m.reason.find(';', p);
                reason_cnt[m.reason.substr(p, e == std::string::npos ? std::string::npos : e - p)]++;
                p = e == std::string::npos ? e : e + 1;
            }
        }

    // ---- 한 칸 이동량과 흔들림 (corner_steps.py), 규칙 3가지 ----
    const std::vector<std::string> rules = {"valid", "nondiag", "all"};
    std::map<std::string, std::vector<Step>> steps;
    std::map<std::string, std::map<std::string, StepStats>> sstat;   // rule -> axis
    std::map<std::string, std::vector<Noise>> noise;
    for (auto& rule : rules) {
        auto ok = [&](size_t i, size_t k) {
            if (rule == "valid") return M[i][k].valid && M[ci][k].valid;
            if (rule == "nondiag") return T.corners[k].black != 2;
            return true;
        };
        auto& S = steps[rule];
        for (int t = 0; t < 6; ++t) {
            double sgn = val[tfiles[t][0]] > 0 ? 1 : -1;
            size_t prev = ci;
            for (size_t j = 0; j < tfiles[t].size(); ++j) {
                size_t f = tfiles[t][j];
                std::vector<double> d, dq[3];
                for (size_t k = 0; k < NC; ++k)
                    if (ok(f, k) && ok(prev, k)) {
                        double v = (M[f][k].x - M[prev][k].x) * sgn;
                        d.push_back(v);
                        dq[T.corners[k].qr - 1].push_back(v);
                    }
                Step s{t, prev, f, mean(d), sd(d, 1), (int)d.size(), j == 0, {mean(dq[0]), mean(dq[1]), mean(dq[2])}};
                S.push_back(s);
                prev = f;
            }
        }
        for (std::string ax : {"Y", "T"}) {
            std::vector<double> px, spread;
            for (auto& s : S) if (!s.first && ((s.trial >= 4) == (ax == "T"))) { px.push_back(s.px); spread.push_back(s.spread); }
            sstat[rule][ax] = {(int)px.size(), mean(px), sd(px, 1), *std::min_element(px.begin(), px.end()), *std::max_element(px.begin(), px.end()), median(spread)};
        }
        // 흔들림
        auto& NZ = noise[rule];
        for (int t = 0; t < 6; ++t) {
            std::vector<size_t> ids;
            for (size_t k = 0; k < NC; ++k) {
                bool all = true;
                for (size_t i : tfiles[t]) all = all && ok(i, k);
                if (all) ids.push_back(k);
            }
            std::vector<double> v;
            for (size_t i : tfiles[t]) v.push_back(val[i]);
            size_t n = v.size();
            for (int comp = 0; comp < 2; ++comp) {
                std::vector<std::vector<double>> res(ids.size());
                for (size_t j = 0; j < ids.size(); ++j) {
                    size_t k = ids[j];
                    std::vector<double> y;
                    for (size_t i : tfiles[t]) y.push_back(comp == 0 ? M[i][k].x - M[ci][k].x : M[i][k].y - M[ci][k].y);
                    res[j] = quad_resid(v, y);
                }
                std::vector<double> common(n, 0.0);
                for (size_t r = 0; r < n; ++r) {
                    for (size_t j = 0; j < ids.size(); ++j) common[r] += res[j][r];
                    common[r] /= ids.size();
                }
                std::vector<double> tot, pt;
                for (size_t j = 0; j < ids.size(); ++j) {
                    std::vector<double> own(n);
                    for (size_t r = 0; r < n; ++r) own[r] = res[j][r] - common[r];
                    tot.push_back(sd(res[j], 3));
                    pt.push_back(sd(own, 3));
                }
                NZ.push_back({t, comp, sd(common, 3), (int)ids.size(), median(tot), median(pt), percentile(pt, .9), *std::max_element(pt.begin(), pt.end())});
            }
        }
    }

    // CSV
    for (auto& rule : rules) {
        Table t{{"trial", "axis", "step_from", "step_to", "px", "spread_px", "n_points", "kind", "qr1_px", "qr2_px", "qr3_px"}};
        for (auto& s : steps[rule])
            t.add({kTrials[s.trial], s.trial >= 4 ? "Θ" : "Y", stem(base_name(shots[s.from].rel)), stem(base_name(shots[s.to].rel)), fmt(s.px, 5), fmt(s.spread, 5),
                   std::to_string(s.n), s.first ? "first" : "step", fmt(s.qr_px[0], 5), fmt(s.qr_px[1], 5), fmt(s.qr_px[2], 5)});
        t.csv(out("corner_step_px_" + rule + ".csv"));
        Table p{{"trial", "comp", "sigma_common", "n_points", "sigma_total_median", "sigma_point_median", "sigma_point_p90", "sigma_point_max"}};
        for (auto& z : noise[rule])
            p.add({kTrials[z.trial], z.comp ? "dy" : "dx", fmt(z.common, 5), std::to_string(z.npts), fmt(z.total_med, 5), fmt(z.point_med, 5), fmt(z.point_p90, 5), fmt(z.point_max, 5)});
        p.csv(out("noise_sigma_" + rule + ".csv"));
    }
    Table tsum{{"trial", "사진", "늘 통과", "가끔 불확실", "늘 불확실", "통과율 (%)", "사진당 통과 최소", "최대"}};
    for (int t = 0; t < 6; ++t)
        tsum.add({kTrials[t], std::to_string(ts[t].photos), std::to_string(ts[t].always_ok), std::to_string(ts[t].sometimes_bad), std::to_string(ts[t].always_bad),
                  fmt(ts[t].rate, 2), std::to_string(ts[t].pmin), std::to_string(ts[t].pmax)});
    tsum.csv(out("trial_summary.csv"));

    // ---- 그림 ----
    {
        Chart c("Passed corners per photo (of 755)", "|command| (Y mm or T deg x 2)", "passed corners");
        for (int t = 0; t < 6; ++t) {
            std::vector<double> x, y;
            for (size_t i : tfiles[t]) { x.push_back(std::abs(val[i]) * (t >= 4 ? 2 : 1)); y.push_back(per_photo[i]); }
            c.line(x, y, color(t), kTrialEn[t], true);
        }
        c.hline(per_photo[ci], kRed);
        c.save("01_사진별_통과점수.png");
    }
    {
        // center 위에 280장 중 불확실 횟수로 색: 연두 = 늘 통과, 주황 = 가끔, 빨강 = 늘 불확실
        cv::Rect roi(1545, 962, 717, 218);
        cv::Mat c;
        cv::cvtColor(T.center(roi), c, cv::COLOR_GRAY2BGR);
        const double K = 2.5;
        cv::resize(c, c, cv::Size(), K, K, cv::INTER_CUBIC);
        for (size_t k = 0; k < NC; ++k) {
            cv::Point p((int)std::lround((T.cx[k] - roi.x + 0.5) * K - 0.5), (int)std::lround((T.cy[k] - roi.y + 0.5) * K - 0.5));
            cv::Scalar col = nbad_all[k] == 0 ? cv::Scalar(74, 163, 22) : nbad_all[k] == 280 ? kRed : cv::Scalar(12, 88, 234);
            cv::rectangle(c, p - cv::Point(7, 7), p + cv::Point(7, 7), cv::Scalar(42, 23, 15), 2);
            cv::rectangle(c, p - cv::Point(5, 5), p + cv::Point(5, 5), col, 2);
        }
        cv::rectangle(c, {0, 0}, {text_w("green: pass in all 280, orange: sometimes, red: never", 0.6) + 12, 28}, cv::Scalar(255, 255, 255), -1);
        text(c, "green: pass in all 280, orange: sometimes, red: never", {6, 20}, 0.6);
        save_png("02_통과지도_280장.png", c);
    }
    {
        std::vector<cv::Mat> tiles;
        for (std::string ax : {"Y", "T"}) {
            auto& st = sstat["valid"][ax];
            Chart c(std::string("One step shift (valid corners) - ") + (ax == "Y" ? "Y, step 0.1 or 0.05 mm" : "T, step 0.01 deg"), "|command| at step end",
                    "mean corner x shift (px)");
            for (int t = (ax == "Y" ? 0 : 4); t < (ax == "Y" ? 4 : 6); ++t) {
                std::vector<double> x, y;
                for (auto& s : steps["valid"]) if (s.trial == t && !s.first) { x.push_back(std::abs(val[s.to])); y.push_back(s.px); }
                c.line(x, y, color(t), kTrialEn[t], true);
            }
            c.hline(st.mean, kGray).hline(st.mean + 3 * st.std, kRed).hline(st.mean - 3 * st.std, kRed);
            tiles.push_back(c.render());
        }
        save_png("03_한칸이동량.png", vstack(tiles));
    }
    {
        Chart c("Jitter after quadratic fit per corner (valid rule)", "trial (1 Y0.1+ 2 Y0.1- 3 Y0.05odd+ 4 Y0.05odd- 5 T+ 6 T-)", "sigma (px)");
        std::vector<double> x, pdx, pdy, cdx, cdy;
        for (auto& z : noise["valid"]) {
            if (z.comp == 0) { x.push_back(z.trial + 1); pdx.push_back(z.point_med); cdx.push_back(z.common); }
            else { pdy.push_back(z.point_med); cdy.push_back(z.common); }
        }
        c.line(x, pdx, color(0), "own sigma median dx", true).line(x, pdy, color(2), "own sigma median dy", true);
        c.line(x, cdx, color(3), "photo common sigma dx", true).line(x, cdy, color(4), "photo common sigma dy", true);
        c.save("04_흔들림_점고유와사진공통.png");
    }

    // 튀는 값: valid 규칙 한 칸 이동량에서 축 평균과 가장 먼 칸 (Y, Θ 각각)
    auto worst_step = [&](const std::string& ax) {
        auto& st = sstat["valid"][ax];
        const Step* w = nullptr;
        for (auto& s : steps["valid"])
            if (!s.first && ((s.trial >= 4) == (ax == "T")) && (!w || std::abs(s.px - st.mean) > std::abs(w->px - st.mean))) w = &s;
        return w;
    };
    const Step* wy = worst_step("Y");
    const Step* wt = worst_step("T");
    auto step_fig = [&](const Step* s, const std::string& file) {
        cv::Mat im = read_color(shots[s->to]);
        std::vector<cv::Mat> tiles;
        double sgn = val[s->to] > 0 ? 1 : -1;
        for (int q = 0; q < 3; ++q) {
            std::vector<Mark> marks;
            double minx = 1e9, miny = 1e9;
            for (size_t k = 0; k < NC; ++k) {
                if (T.corners[k].qr != q + 1) continue;
                minx = std::min(minx, M[s->to][k].x); miny = std::min(miny, M[s->to][k].y);
            }
            cv::Rect roi((int)minx - 4, (int)miny - 4, 64, 64);
            for (size_t k = 0; k < NC; ++k) {
                if (T.corners[k].qr != q + 1 || !(M[s->to][k].valid && M[s->from][k].valid && M[ci][k].valid)) continue;
                auto& a = M[s->from][k];
                auto& b = M[s->to][k];
                if (!roi.contains(cv::Point((int)b.x, (int)b.y))) continue;
                marks.push_back({{a.x + 0.5, a.y + 0.5}, {0, 0}, "", kGreen});
                marks.push_back({{b.x + 0.5, b.y + 0.5}, {0, 0}, "", kRed});
            }
            tiles.push_back(zoom(im, roi, 420, marks, 0,
                                 "QR" + std::to_string(q + 1) + " step " + fmt(s->qr_px[q], 3) + " px"));
        }
        cv::Mat all = hstack(tiles);
        std::string cap = stem(base_name(shots[s->from].rel)) + " -> " + stem(base_name(shots[s->to].rel)) + ": green = before, red = after (sign " + (sgn > 0 ? "+" : "-") + ")";
        cv::Mat head(30, all.cols, CV_8UC3, cv::Scalar(255, 255, 255));
        text(head, cap, {6, 21}, 0.6);
        save_png(file, vstack({head, all}, 0));
    };
    std::string wyn = stem(base_name(shots[wy->to].rel)), wtn = stem(base_name(shots[wt->to].rel));
    step_fig(wy, "05_튀는값_Y_" + wyn + ".png");
    step_fig(wt, "06_튀는값_T_" + wtn + ".png");
    Table outl{{"축", "trial", "칸 (이전 -> 이번)", "정답 칸", "한 칸 이동 (px)", "축 평균 (px)", "평균에서 (σ 배)", "QR1 / QR2 / QR3 이동 (px)", "점 수", "점끼리 σ (px)"}};
    for (auto [ax, s] : std::vector<std::pair<std::string, const Step*>>{{"Y", wy}, {"T", wt}}) {
        auto& st = sstat["valid"][ax];
        outl.add({ax == "Y" ? "Y" : "Θ", kTrials[s->trial], stem(base_name(shots[s->from].rel)) + " -> " + stem(base_name(shots[s->to].rel)),
                  fmt(std::abs(val[s->to] - val[s->from]), 2) + (ax == "Y" ? " mm" : " deg"), fmt(s->px, 3), fmt(st.mean, 4), fmts((s->px - st.mean) / st.std, 1),
                  fmt(s->qr_px[0], 3) + " / " + fmt(s->qr_px[1], 3) + " / " + fmt(s->qr_px[2], 3), std::to_string(s->n), fmt(s->spread, 3)});
    }
    // 3σ 밖 칸 목록
    Table out3{{"규칙", "trial", "칸", "한 칸 이동 (px)"}};
    for (auto& rule : rules)
        for (std::string ax : {"Y", "T"}) {
            auto& st = sstat[rule][ax];
            for (auto& s : steps[rule])
                if (!s.first && ((s.trial >= 4) == (ax == "T")) && std::abs(s.px - st.mean) >= 3 * st.std)
                    out3.add({rule, kTrials[s.trial], stem(base_name(shots[s.from].rel)) + " -> " + stem(base_name(shots[s.to].rel)), fmt(s.px, 3)});
        }

    // ---- 원래 결과와 대조 ----
    // 출처: analysis_grid_corners_20261019/out/trial_summary.csv, out/log_valid.txt, log_nondiag.txt, log_all.txt,
    //       out/corners_280.csv(center 통과 597, 전체 통과 170404/212155, 사유별 횟수)
    Parity par;
    const int o_ok[6] = {486, 491, 513, 501, 365, 390}, o_bad[6] = {125, 125, 125, 126, 125, 125}, o_min[6] = {593, 591, 606, 593, 595, 595},
              o_max[6] = {609, 614, 618, 615, 621, 615};
    const double o_rate[6] = {79.75, 79.69, 81.0, 80.32, 80.7, 80.05};
    for (int t = 0; t < 6; ++t) par.add(kTrials[t] + " 통과율 (%)", o_rate[t], ts[t].rate, 0.1, 2);
    for (int t = 0; t < 6; ++t) par.add(kTrials[t] + " 늘 통과 점", o_ok[t], ts[t].always_ok, 6, 0);
    for (int t = 0; t < 6; ++t) par.add(kTrials[t] + " 늘 불확실 점", o_bad[t], ts[t].always_bad, 1, 0);
    for (int t = 0; t < 6; ++t) par.add(kTrials[t] + " 사진당 통과 최소/최대 합", o_min[t] + o_max[t], ts[t].pmin + ts[t].pmax, 4, 0);
    par.add("center 통과 점", 597, per_photo[ci], 2, 0);
    par.add("전체 점-사진 통과", 170404, (double)total_valid, 200, 0);
    par.add("사유 photometric_rmse (점-사진)", 35674, (double)reason_cnt["photometric_rmse"], 150, 0);
    par.add("사유 window_instability (점-사진)", 21184, (double)reason_cnt["window_instability"], 150, 0);
    par.add("사유 optimizer (점-사진)", 6770, (double)reason_cnt["optimizer"], 100, 0);
    // 한 칸 이동량 (log_*.txt)
    const std::map<std::string, std::array<double, 4>> o_step = {{"valid", {1.4357, 0.0227, 0.8044, 0.0260}}, {"nondiag", {1.4358, 0.0228, 0.8044, 0.0260}},
                                                                {"all", {1.4355, 0.0267, 0.8044, 0.0288}}};
    for (auto& rule : rules) {
        auto& o = o_step.at(rule);
        par.add(rule + " 한 칸 Y 평균 (px)", o[0], sstat[rule]["Y"].mean, 0.001);
        par.add(rule + " 한 칸 Y 표준편차 (px)", o[1], sstat[rule]["Y"].std, 0.001);
        par.add(rule + " 한 칸 Θ 평균 (px)", o[2], sstat[rule]["T"].mean, 0.001);
        par.add(rule + " 한 칸 Θ 표준편차 (px)", o[3], sstat[rule]["T"].std, 0.001);
    }
    par.add("valid 칸 안 점끼리 σ 중앙값 Y", 0.0756, sstat["valid"]["Y"].spread_med, 0.002);
    par.add("valid 칸 안 점끼리 σ 중앙값 Θ", 0.0668, sstat["valid"]["T"].spread_med, 0.002);
    {
        // 첫 칸 (valid): Y+0.10, Y-0.10, Y+0.05, Y-0.05, T+0.01, T-0.01
        const double o_first[6] = {0.942, 1.344, 0.303, 1.112, 0.164, 1.062};
        const int o_n[6] = {573, 586, 585, 586, 579, 580};
        for (auto& s : steps["valid"])
            if (s.first) {
                par.add("첫 칸 center -> " + stem(base_name(shots[s.to].rel)) + " (px)", o_first[s.trial], s.px, 0.003, 3);
                par.add("첫 칸 center -> " + stem(base_name(shots[s.to].rel)) + " 점 수", o_n[s.trial], s.n, 5, 0);
            }
    }
    {
        // 사진 공통 흔들림 σ (valid, log_valid.txt), 점 수
        const double o_cdx[6] = {0.0150, 0.0140, 0.0141, 0.0286, 0.0324, 0.0286}, o_cdy[6] = {0.0095, 0.0117, 0.0080, 0.0155, 0.0123, 0.0119};
        const double o_pdx[6] = {0.0462, 0.0453, 0.0475, 0.0465, 0.0461, 0.0475};
        const int o_np[6] = {479, 485, 500, 493, 359, 388};
        for (auto& z : noise["valid"]) {
            if (z.comp == 0) {
                par.add(kTrials[z.trial] + " 사진 공통 σ dx", o_cdx[z.trial], z.common, 0.001);
                par.add(kTrials[z.trial] + " 점 고유 σ 중앙값 dx", o_pdx[z.trial], z.point_med, 0.001);
                par.add(kTrials[z.trial] + " 흔들림 계산 점 수", o_np[z.trial], z.npts, 8, 0);
            } else {
                par.add(kTrials[z.trial] + " 사진 공통 σ dy", o_cdy[z.trial], z.common, 0.001);
            }
        }
    }

    // ---- 결과.md ----
    Table stt{{"규칙", "축", "칸 수", "평균 (px)", "표준편차", "최소", "최대", "칸 안 점끼리 σ 중앙값"}};
    for (auto& rule : rules)
        for (std::string ax : {"Y", "T"}) {
            auto& s = sstat[rule][ax];
            stt.add({rule, ax == "Y" ? "Y" : "Θ", std::to_string(s.count), fmt(s.mean, 4), fmt(s.std, 4), fmt(s.min, 4), fmt(s.max, 4), fmt(s.spread_med, 4)});
        }
    Table ftab{{"규칙", "첫 칸", "이동 (px)", "점 수"}};
    for (auto& rule : rules)
        for (auto& s : steps[rule]) if (s.first) ftab.add({rule, "center -> " + stem(base_name(shots[s.to].rel)), fmt(s.px, 3), std::to_string(s.n)});
    Table nt{{"규칙", "trial", "성분", "점 수", "점 남은 값 σ 중앙값", "점 고유 σ 중앙값", "점 고유 σ 90%", "점 고유 σ 최대", "사진 공통 σ"}};
    for (auto& rule : rules)
        for (auto& z : noise[rule])
            nt.add({rule, kTrials[z.trial], z.comp ? "dy" : "dx", std::to_string(z.npts), fmt(z.total_med, 4), fmt(z.point_med, 4), fmt(z.point_p90, 4), fmt(z.point_max, 4),
                    fmt(z.common, 4)});
    Table rt{{"사유", "점-사진 수"}};
    for (auto& [k, v] : reason_cnt) rt.add({k, std::to_string(v)});
    Table bt{{"검정 이웃 수", "점-사진", "통과율 (%)"}};
    for (auto& [b, v] : by_black) bt.add({std::to_string(b), std::to_string(v[0]), fmt(100.0 * v[1] / v[0], 2)});
    std::string q3how = T.qrs[2].how;

    std::ostringstream R;
    R << "# 결과: 모서리 755점 281장 추적\n\n"
      << "입력: `data/clean/clean_20261014` 281장 (center 1, Y 80, T 200). center 에서 QR 세 개를 해독해 구조점 755개를 만들고 경계 모델로 맞춘 위치를 ID별 기준으로 삼았다(10-15 실험과 같은 방법).\n"
      << "사진마다 QR별 템플릿 매칭(±130 px)으로 정수 이동을 구해 시작점을 옮기고, 9x9 창 맞춤을 창이 더 안 움직일 때까지(최대 3번) 다시 한 뒤 11x11 창으로 한 번 더 맞춰 두 창 차이를 진단에 썼다. 통과 기준은 10-15 실험과 같다.\n"
      << "trial: Y 는 0.1 mm 간격(±0.1~2.0)과 0.05 홀수(±0.05~1.95)를 부호별로 나눠 4개(각 20장), Θ 는 부호별 2개(각 100장).\n"
      << "규칙: valid = 그 사진과 center 에서 모두 통과한 점, nondiag = 대각 접점(검정 이웃 2) 125점만 뺀 630점, all = 755점 전부.\n"
      << "한 칸 이동량 = 같은 trial 에서 |명령값| 순으로 이웃한 두 사진의 점별 x 차이 평균(바깥쪽 +). 흔들림 = trial 의 모든 사진에서 규칙을 만족한 점마다 이동량을 명령값 2차 곡선으로 맞춘 남은 값, 그 사진의 모든 점 평균을 사진 공통, 나머지를 점 고유로 나눔(ddof 3).\n\n"
      << "QR3 은 OpenCV 3.4 해독기로 찾은 방법이 `" << q3how << "` 라서 원래(cv2 5.0)와 시작 사각형이 조금 다르다(10-15 실험 결과.md 참고). 기준 위치는 영상에 맞춘 값이라 영향이 작다.\n\n"
      << "## trial별 통과\n\n" << tsum.md() << "\ncenter 통과 " << per_photo[ci] << " / " << NC << ", 전체 점-사진 통과 " << total_valid << " / " << NS * NC << ".\n\n"
      << "### 불확실 사유 (점-사진, 한 번에 여러 개 가능)\n\n" << rt.md() << "\n### 검정 이웃 수별\n\n" << bt.md()
      << "\n## 한 칸 이동량\n\n" << stt.md() << "\n### 첫 칸 (center -> 첫 사진)\n\n" << ftab.md() << "\n### 평균에서 3σ 밖 칸\n\n" << out3.md()
      << "\n## 흔들림 σ (px)\n\n" << nt.md()
      << "\n## 원래 결과와 대조 (" << par.ok << "/" << par.total << " 일치)\n\n" << par.t.md()
      << "\n## 옮기지 않은 부분\n\n- corner_steps.py 의 'ECC 와 비교'(presentation_eda_20261019/img/step_px.csv)와 ecc_noise.py(14_데이터_QR마커위치.csv): 다른 실험의 ECC 결과 파일을 읽는 비교라 옮기지 않았다.\n"
      << "- add_*_slides.py, corner_step_figure.py, noise_figures.py, viz_trials.py 의 발표용 그림·슬라이드: 같은 수치를 `이미지/` 그림 4장으로 대신했다.\n"
      << "\n## 파일\n\n- `out/corners_281.csv`: 281장 x 755점 측정값 (크기 때문에 out/ 에 둠, git 제외)\n- `corner_step_px_<규칙>.csv`: 칸별 이동량(QR별 평균 포함)\n"
      << "- `noise_sigma_<규칙>.csv`: trial별 흔들림 요약\n- `trial_summary.csv`: trial별 통과\n- `이미지/`: 01 사진별 통과 점 수, 02 통과 지도, 03 한 칸 이동량, 04 흔들림, 05·06 튀는 값\n";
    write_text(out("결과.md"), R.str());

    // ---- 보고서.md ----
    double rmin = 1e9, rmax = -1e9;
    for (auto& s : ts) { rmin = std::min(rmin, s.rate); rmax = std::max(rmax, s.rate); }
    auto cz = [&](int t, int comp) { for (auto& z : noise["valid"]) if (z.trial == t && z.comp == comp) return z.common; return kNaN; };
    double pmin_dx = 1e9, pmax_dx = 0;
    for (auto& z : noise["valid"]) if (z.comp == 0) { pmin_dx = std::min(pmin_dx, z.point_med); pmax_dx = std::max(pmax_dx, z.point_med); }
    auto& sy = sstat["valid"]["Y"];
    auto& sth = sstat["valid"]["T"];
    std::ostringstream P;
    P << "# 보고서: 모서리 755점 281장 추적 (10-19 13:16)\n\n"
      << "## 문제\n\nQR 격자 모서리 755점을 281장 모두에서 같은 ID로 서브픽셀 측정할 수 있는지, 어느 점이 불안정한지 확인해야 했다.\n\n"
      << "## 해결 방법\n\ncenter 에서 맞춘 모서리 위치를 ID로 고정하고, 사진마다 QR별 정수 이동으로 시작점을 옮겨 같은 경계 모델을 맞춘 뒤 한 칸 이동량과 흔들림을 계산했다.\n\n"
      << "## 결과\n\ntrial별 통과율은 " << fmt(rmin, 2) << "~" << fmt(rmax, 2) << "%이고, 통과한 점으로 잰 한 칸 이동은 Y " << fmt(sy.mean, 4) << " px, Θ " << fmt(sth.mean, 4)
      << " px, 사진 공통 흔들림(dx)은 Y " << fmt(std::min({cz(0, 0), cz(1, 0), cz(2, 0), cz(3, 0)}), 4) << "~" << fmt(std::max({cz(0, 0), cz(1, 0), cz(2, 0), cz(3, 0)}), 4) << " px, Θ "
      << fmt(std::min(cz(4, 0), cz(5, 0)), 4) << "~" << fmt(std::max(cz(4, 0), cz(5, 0)), 4) << " px였다.\n\n"
      << "| 지표 | 값 |\n|---|---|\n"
      << "| 점-사진 통과 / 전체 | " << total_valid << " / " << NS * NC << " (" << fmt(100.0 * total_valid / (NS * NC), 2) << "%) |\n"
      << "| center 통과 | " << per_photo[ci] << " / " << NC << " |\n"
      << "| 늘 불확실 점 (trial마다) | " << ts[0].always_bad << " · " << ts[1].always_bad << " · " << ts[2].always_bad << " · " << ts[3].always_bad << " · " << ts[4].always_bad
      << " · " << ts[5].always_bad << " |\n"
      << "| 한 칸 Y 평균 ± 표준편차 (px, 76칸) | " << fmt(sy.mean, 4) << " ± " << fmt(sy.std, 4) << " |\n"
      << "| 한 칸 Θ 평균 ± 표준편차 (px, 198칸) | " << fmt(sth.mean, 4) << " ± " << fmt(sth.std, 4) << " |\n"
      << "| 점 고유 흔들림 σ 중앙값 dx (trial 6개) | " << fmt(pmin_dx, 4) << " ~ " << fmt(pmax_dx, 4) << " |\n\n"
      << tsum.md() << "\n" << img_md("01_사진별_통과점수.png", "사진별 통과") << "\n\n" << img_md("02_통과지도_280장.png", "통과 지도") << "\n\n"
      << img_md("03_한칸이동량.png", "한 칸 이동량") << "\n\n" << img_md("04_흔들림_점고유와사진공통.png", "흔들림") << "\n\n"
      << "## 튀는 값\n\n한 칸 이동이 평균에서 가장 먼 칸은 Y 는 " << stem(base_name(shots[wy->from].rel)) << " -> " << wyn << "(" << fmt(wy->px, 3) << " px, 평균의 "
      << fmts((wy->px - sy.mean) / sy.std, 1) << "σ), Θ 는 " << stem(base_name(shots[wt->from].rel)) << " -> " << wtn << "(" << fmt(wt->px, 3) << " px, "
      << fmts((wt->px - sth.mean) / sth.std, 1) << "σ)이고, 세 QR 이 같은 양만큼 움직여서(QR별 차이 " << fmt(std::max({std::abs(wy->qr_px[0] - wy->qr_px[1]), std::abs(wy->qr_px[1] - wy->qr_px[2]), std::abs(wy->qr_px[0] - wy->qr_px[2])}), 3)
      << " px) 측정이 아니라 그 칸에서 판이 실제로 덜·더 움직인 것으로 보인다.\n\n"
      << outl.md() << "\n" << img_md("05_튀는값_Y_" + wyn + ".png", "Y 튀는 값") << "\n\n" << img_md("06_튀는값_T_" + wtn + ".png", "T 튀는 값")
      << "\n\n초록 십자는 이전 사진, 빨간 십자는 이번 사진에서 맞춘 모서리다.\n\n"
      << "## 한줄평\n\n점마다 2x2 곧은 경계 모델을 맞추는 방식이라 대각 접점 125점은 280장 내내 불확실해서 통과율이 80% 근처에 묶였지만, 통과한 점끼리 평균하면 한 칸 이동이 Y "
      << fmt(sy.mean, 4) << "±" << fmt(sy.std, 4) << " px로 흔들림(사진 공통 dx 약 " << fmt(cz(0, 0), 3) << " px)보다 훨씬 커서 칸마다 이동을 뚜렷이 구분했다.\n\n"
      << "## 원래 결과와 대조\n\n원래 값은 `analysis_grid_corners_20261019/out/trial_summary.csv`, `out/log_valid.txt`, `log_nondiag.txt`, `log_all.txt`, `out/corners_280.csv` 에서 가져왔다. "
      << par.ok << "/" << par.total << " 일치.\n\n" << par.t.md();
    write_text(out("보고서.md"), P.str());
    std::cout << "parity " << par.ok << "/" << par.total << "\n";
    return 0;
}
