// T-0.05 오분류: 6후보 GLS 와 v7b 2단계를 같은 QR205 측정·같은 119 분할로 비교 (원본: codex_experiments_20261014/t005_v7b_20261015/experiment.py,
// 측정 measurement/spread.py gn17_spread14, 6후보 zero_experiment.py, 분할 gls_comparison_20261014/compare.py)
#include <hye/dataset.hpp>
#include <hye/plot.hpp>
#include <hye/report.hpp>
#include <omp.h>
#include <iostream>
#include <map>
#include <set>
#include "qr205.hpp"
#include "sixcand.hpp"

using namespace hye;

namespace {

// v7b 2단계 (t005 experiment.py Stage/TwoStage): 방해 무늬는 화이트닝에서 투영으로 뺀다
struct Stage {
    bool quad, free_, local;
    cv::Mat U, W, Q;
    std::map<std::string, std::array<cv::Mat, 3>> poly;  // s, c, b (12 x 1)
    void fit(const std::vector<six::Meta>& m, const std::vector<std::vector<double>>& X, const std::vector<int>& tr, int center) {
        std::vector<cv::Mat> res;
        for (std::string a : {"Y", "T"}) {
            std::vector<int> ids;
            for (int i : tr)
                if (m[i].axis == a && (!local || std::abs(m[i].value) <= (a == "Y" ? 0.6 : 0.12))) ids.push_back(i);
            ids.push_back(center);
            int nc = quad ? (free_ ? 3 : 2) : 1;
            cv::Mat A((int)ids.size(), nc, CV_64F), Y((int)ids.size(), 12, CV_64F);
            for (int r = 0; r < A.rows; ++r) {
                double v = m[ids[r]].value;
                A.at<double>(r, 0) = v;
                if (quad) A.at<double>(r, 1) = v * v;
                if (quad && free_) A.at<double>(r, 2) = 1;
                for (int j = 0; j < 12; ++j) Y.at<double>(r, j) = X[ids[r]][j];
            }
            cv::Mat co = six::lstsq(A, Y);
            cv::Mat z = cv::Mat::zeros(12, 1, CV_64F);
            poly[a] = {co.row(0).t(), quad ? cv::Mat(co.row(1).t()) : z.clone(), quad && free_ ? cv::Mat(co.row(nc - 1).t()) : z.clone()};
            cv::Mat R = Y - A * co;
            for (int r = 0; r < A.rows; ++r) if (ids[r] != center) res.push_back(R.row(r));
        }
        cv::Mat R;
        cv::vconcat(res, R);
        cv::Mat cov = R.t() * R / std::max(R.rows - 6, 1) + 1e-10 * cv::Mat::eye(12, 12, CV_64F);
        W = six::chol_inv_root(cov);
        Q = six::project_out(W, U);
    }
    std::pair<double, double> solve(const std::vector<double>& x, const std::string& a) const {
        cv::Mat f(12, 1, CV_64F);
        for (int j = 0; j < 12; ++j) f.at<double>(j) = x[j];
        auto& p = poly.at(a);
        cv::Mat z = Q * (f - p[2]), sw = Q * p[0], cw = Q * p[1];
        double v = 0;
        for (int it = 0; it < 8; ++it) {
            cv::Mat r = z - v * sw - v * v * cw, j = sw + 2 * v * cw;
            v += j.dot(r) / std::max(j.dot(j), 1e-30);
        }
        cv::Mat r = z - v * sw - v * v * cw;
        return {v, r.dot(r)};
    }
};

std::string stem(const std::string& f) { return f.substr(0, f.rfind('.')); }

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
    auto m = six::manifest(shots);
    const int N = (int)m.size(), C0 = 0;  // m[0] = center
    cv::Mat ref = qr::read_crop(shots[m[0].shot]);
    qr::Spread ext(ref, 14);
    auto gpts = ext.global_points();
    int nreg[3] = {};
    for (int r : ext.regions) nreg[r]++;
    std::cout << "points " << gpts.size() << " " << nreg[0] << "/" << nreg[1] << "/" << nreg[2] << "\n";

    std::vector<std::vector<cv::Point2d>> Draw(N);
    std::vector<int> nvalid(N);
#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < N; ++i) {
        auto o = ext.extract(qr::read_crop(shots[m[i].shot]));
        Draw[i] = o.D;
        nvalid[i] = (int)std::count(o.valid.begin(), o.valid.end(), true);
    }
    int allvalid = 0;
    for (int v : nvalid) allvalid += v == (int)gpts.size();
    std::vector<std::vector<cv::Point2d>> D(N, std::vector<cv::Point2d>(gpts.size()));
    for (int i = 0; i < N; ++i)
        for (size_t k = 0; k < gpts.size(); ++k) D[i][k] = Draw[i][k] - Draw[C0][k];
    std::cout << "tracked, all-valid images " << allvalid << "\n";

    qr::P12 p12(gpts);
    std::vector<std::vector<double>> X(N);
    for (int i = 0; i < N; ++i) X[i] = p12(D[i]);
    // v7b 방해 무늬: 점 무게중심 기준 세로 이동·화면 회전을 12값으로
    cv::Mat U(12, 2, CV_64F);
    {
        double mx = 0, my = 0;
        for (auto& p : gpts) { mx += p.x; my += p.y; }
        mx /= gpts.size(); my /= gpts.size();
        for (int j = 0; j < 2; ++j) {
            std::vector<cv::Point2d> f;
            for (auto& p : gpts) f.push_back(j == 0 ? cv::Point2d(0, 0.05) : cv::Point2d(-(p.y - my) / 100 * 0.05, (p.x - mx) / 100 * 0.05));
            auto v = p12(f);
            for (int r = 0; r < 12; ++r) U.at<double>(r, j) = v[r] / 0.05;
        }
    }
    std::vector<double> lab(N);
    for (int i = 0; i < N; ++i) lab[i] = m[i].value;
    int target = -1;
    for (int i = 0; i < N; ++i) if (m[i].file == "T-0.05.bmp") target = i;

    auto splits = six::protocols(m);
    uint64_t fp = six::fingerprint(splits, m);
    std::vector<std::string> methods{"gn_six", "gn_v7b"};
    std::map<std::string, std::vector<six::Pred>> P;  // method|protocol -> 예측
    std::vector<std::string> prot_order;
    struct TRow { std::string protocol, fold, method, pred_axis; double pred, cY, cT, lr; };
    std::vector<TRow> trow;
    std::vector<std::vector<six::Cand>> six_target_cands;
    for (auto& sp : splits) {
        if (prot_order.empty() || prot_order.back() != sp.protocol) prot_order.push_back(sp.protocol);
        auto sm = six::fit(m, X, sp.tr, lab, "inferred_session", true);
        Stage A{false, false, true, U}, V{true, true, false, U};
        A.fit(m, X, sp.tr, C0);
        V.fit(m, X, sp.tr, C0);
        for (int i : sp.te) {
            std::vector<six::Cand> cands;
            auto p = six::predict_one(m, X[i], i, sm, "inferred_session", i == target ? &cands : nullptr);
            P["gn_six|" + sp.protocol].push_back(p);
            auto ay = A.solve(X[i], "Y"), at = A.solve(X[i], "T");
            std::string pa = ay.second < at.second ? "Y" : "T";
            double v = (pa == "Y" ? V.solve(X[i], "Y") : V.solve(X[i], "T")).first;
            six::Pred q{i, m[i].axis, pa, pa, m[i].value, v, std::min(ay.second, at.second), ay.second, at.second};
            P["gn_v7b|" + sp.protocol].push_back(q);
            if (i == target) {
                trow.push_back({sp.protocol, sp.fold, "gn_six", p.pred_axis, p.pred, p.costY, p.costT,
                                std::isfinite(p.costY) && std::isfinite(p.costT) ? std::log(p.costY / p.costT) : kNaN});
                trow.push_back({sp.protocol, sp.fold, "gn_v7b", pa, v, ay.second, at.second, std::log(ay.second / at.second)});
                if (sp.protocol == "full280") six_target_cands.push_back(cands);
            }
        }
    }

    // ---- 지표 ----
    Table mt{{"방법", "프로토콜", "맞음", "실패", "Y 축 틀림", "T 축 틀림", "Y 평균 / 최대 |오차| (mm)", "T 평균 / 최대 |오차| (deg)"}};
    std::map<std::string, six::Score> SC;
    for (auto& me : methods)
        for (auto& pr : prot_order) {
            auto s = six::score(P[me + "|" + pr]);
            SC[me + "|" + pr] = s;
            if (pr.rfind("repeated5", 0) == 0) continue;
            mt.add({me, pr, std::to_string(s.correct), std::to_string(s.failed), std::to_string(s.Ywrong), std::to_string(s.Twrong),
                    fmt(s.YMAE, 4) + " / " + fmt(s.Ymax, 4), fmt(s.TMAE, 5) + " / " + fmt(s.Tmax, 5)});
        }
    struct Rep { int fail = 0, zero = 0, t005 = 0, maxfail = 0; std::map<std::string, int> freq; };
    std::map<std::string, Rep> REP;
    Table rp{{"seed", "gn_six 실패", "gn_v7b 실패", "T-0.05 gn_six 판정", "T-0.05 gn_v7b 판정", "T-0.05 gn_six log(비용Y/비용T)", "T-0.05 gn_v7b log(비용Y/비용T)"}};
    std::vector<double> sx, sl6, sl7;
    for (int seed = 0; seed < 20; ++seed) {
        std::string pr = "repeated5_seed" + std::to_string(seed);
        std::vector<std::string> row{std::to_string(seed)};
        for (auto& me : methods) {
            auto& r = REP[me];
            auto s = SC[me + "|" + pr];
            r.fail += s.failed; r.zero += s.failed == 0; r.maxfail = std::max(r.maxfail, s.failed);
            for (auto& p : P[me + "|" + pr]) {
                if (!six::pass09(p)) r.freq[m[p.i].file]++;
                if (p.i == target && p.pred_axis != "T") r.t005++;
            }
            row.push_back(std::to_string(s.failed));
        }
        for (auto& t : trow) if (t.protocol == pr && t.method == "gn_six") { row.push_back(t.pred_axis); sx.push_back(seed); sl6.push_back(t.lr); }
        for (auto& t : trow) if (t.protocol == pr && t.method == "gn_v7b") { row.push_back(t.pred_axis); sl7.push_back(t.lr); }
        row.push_back(fmts(sl6.back(), 4));
        row.push_back(fmts(sl7.back(), 4));
        rp.add(row);
    }
    Table freq{{"방법", "사진", "20회 반복 중 실패 횟수"}};
    std::map<std::string, std::vector<std::pair<int, std::string>>> top;
    for (auto& me : methods) {
        std::vector<std::pair<int, std::string>> v;
        for (auto& kv : REP[me].freq) v.push_back({-kv.second, kv.first});
        std::sort(v.begin(), v.end());
        for (size_t k = 0; k < v.size() && k < 8; ++k) freq.add({me, stem(v[k].second), std::to_string(-v[k].first)});
        top[me] = v;
    }

    // 원래 결과 대조
    // 원래 값: t005_v7b_20261015/metrics.csv, predictions.csv, T005_predictions_and_costs.csv, splits.json(지문), measurement/gn17_spread14.npz
    Parity par;
    par.add_text("분할 119개 시험 목록 지문 (splits.json)", "0x1fb43e12f1df4935", [&] { char b[32]; std::snprintf(b, sizeof b, "0x%016llx", (unsigned long long)fp); return std::string(b); }(),
                 fp == 0x1fb43e12f1df4935ull);
    par.add("QR 점 수", 205, (double)gpts.size(), 0, 0);
    par.add("QR 좌 / 중 / 우 점 수 (좌)", 69, nreg[0], 0, 0);
    par.add("QR 중 점 수", 66, nreg[1], 0, 0);
    par.add("QR 우 점 수", 70, nreg[2], 0, 0);
    par.add("281장 모두 유효", 281, allvalid, 0, 0);
    auto meanD = [&](const std::string& f, int c) {
        for (int i = 0; i < N; ++i) if (m[i].file == f) { double s = 0; for (auto& d : D[i]) s += c ? d.y : d.x; return s / D[i].size(); }
        return kNaN;
    };
    par.add("T+1.00 평균 dx (px)", 79.49491298, meanD("T+1.00.bmp", 0), 0.0005, 5);
    par.add("Y+2.00 평균 dx (px)", 28.17641113, meanD("Y+2.00.bmp", 0), 0.0005, 5);
    par.add("T-0.05 평균 dx (px)", -4.24331559, meanD("T-0.05.bmp", 0), 0.0005, 5);
    par.add("T-0.05 평균 dy (px)", 0.06623138, meanD("T-0.05.bmp", 1), 0.0005, 5);
    par.add("Y-0.05 평균 dx (px)", -1.11028343, meanD("Y-0.05.bmp", 0), 0.0005, 5);
    struct O { const char* me; const char* pr; int corr, fail, yw, tw; double ymae, ymax, tmae; };
    const O orig[] = {{"gn_six", "full280", 280, 0, 0, 0, 0.001613, 0.036241, 0.000367}, {"gn_v7b", "full280", 274, 6, 4, 2, 0.011375, 0.027529, 0.001685},
                      {"gn_six", "interleaved", 280, 0, 0, 0, 0.002826, 0.037107, 0.000412}, {"gn_v7b", "interleaved", 274, 6, 4, 2, 0.012123, 0.028550, 0.001722},
                      {"gn_six", "stratified5", 280, 0, 0, 0, 0.002213, 0.036530, 0.000398}, {"gn_v7b", "stratified5", 273, 7, 5, 2, 0.012015, 0.030783, 0.001763},
                      {"gn_six", "blocked_within_session5", 280, 0, 0, 0, 0.002367, 0.036196, 0.000538}, {"gn_v7b", "blocked_within_session5", 271, 9, 7, 2, 0.011980, 0.028928, 0.002256},
                      {"gn_six", "session_out_missing_origin", 58, 222, 22, 200, 0.023247, 0.041594, kNaN}, {"gn_v7b", "session_out_missing_origin", 271, 9, 5, 4, 0.023335, 0.052859, 0.008293}};
    for (auto& o : orig) {
        auto& s = SC[std::string(o.me) + "|" + o.pr];
        std::string k = std::string(o.me) + " " + o.pr;
        par.add(k + " 맞음", o.corr, s.correct, 0, 0);
        par.add(k + " 실패", o.fail, s.failed, 0, 0);
        par.add(k + " Y/T 축 틀림 (Y)", o.yw, s.Ywrong, 0, 0);
        par.add(k + " T 축 틀림", o.tw, s.Twrong, 0, 0);
        par.add(k + " Y 평균 |오차| (mm)", o.ymae, s.YMAE, 0.0003, 5);
        par.add(k + " Y 최대 |오차| (mm)", o.ymax, s.Ymax, 0.001, 5);
        if (std::isfinite(o.tmae)) par.add(k + " T 평균 |오차| (deg)", o.tmae, s.TMAE, 0.00005, 6);
    }
    par.add("반복 20회 실패 합 gn_six (/5600)", 13, REP["gn_six"].fail, 0, 0);
    par.add("반복 20회 실패 합 gn_v7b (/5600)", 154, REP["gn_v7b"].fail, 0, 0);
    par.add("반복 20회 중 전부 맞음 gn_six", 9, REP["gn_six"].zero, 0, 0);
    par.add("반복 20회 중 전부 맞음 gn_v7b", 0, REP["gn_v7b"].zero, 0, 0);
    par.add("T-0.05 오분류 gn_six (/20)", 11, REP["gn_six"].t005, 0, 0);
    par.add("T-0.05 오분류 gn_v7b (/20)", 20, REP["gn_v7b"].t005, 0, 0);
    par.add("T-0.05 gn_six log 비용비 평균 (반복)", -0.008329, mean(sl6), 0.01);
    par.add("T-0.05 gn_v7b log 비용비 평균 (반복)", -0.450676, mean(sl7), 0.01);
    double lr_full6 = kNaN, lr_full7 = kNaN;
    for (auto& t : trow) if (t.protocol == "full280") (t.method == "gn_six" ? lr_full6 : lr_full7) = t.lr;
    par.add("T-0.05 full280 gn_six log 비용비", 0.115985, lr_full6, 0.01);
    par.add("T-0.05 full280 gn_v7b log 비용비", -0.294910, lr_full7, 0.01);

    // ---- 그림 ----
    {
        Chart c("T-0.05: log(cost_Y / cost_T) over 20 repeated 5-fold splits", "repeat seed (7000 + k)", "log cost ratio (<0 -> judged Y)");
        c.line(sx, sl6, color(0), "gn_six (6 origin candidates)", true).line(sx, sl7, color(1), "gn_v7b (two-stage)", true).hline(0, kRed);
        c.save("01_T-0.05_비용비_반복20회.png");
    }
    {
        std::vector<std::string> cats;
        std::vector<double> v;
        for (auto& pr : prot_order) {
            if (pr.rfind("repeated5", 0) == 0) continue;
            for (auto& me : methods) { cats.push_back(me.substr(3) + " " + (pr == "full280" ? "full" : pr.substr(0, 5))); v.push_back(SC[me + "|" + pr].failed); }
        }
        bar_chart("02_프로토콜별_실패장수.png", "Failures (of 280) by protocol", cats, v, "failed images");
    }
    {
        std::vector<double> x, y6, y7;
        for (int seed = 0; seed < 20; ++seed) {
            x.push_back(seed);
            y6.push_back(SC["gn_six|repeated5_seed" + std::to_string(seed)].failed);
            y7.push_back(SC["gn_v7b|repeated5_seed" + std::to_string(seed)].failed);
        }
        Chart c("Failures per repeated 5-fold split (of 280)", "repeat seed (7000 + k)", "failed images");
        c.line(x, y6, color(0), "gn_six", true).line(x, y7, color(1), "gn_v7b", true);
        c.save("03_반복분할별_실패.png");
    }
    // 튀는 값 사진: QR 세 개 확대. 초록 = center 점, 빨강 = 이 사진 점, 화살표 = (점 이동 - 판 평균 이동) x 200
    auto photo = [&](int i, const std::string& file, const std::string& cap) {
        cv::Mat im = read_color(shots[m[i].shot]);
        double mx = 0, my = 0;
        for (auto& d : D[i]) { mx += d.x; my += d.y; }
        mx /= D[i].size(); my /= D[i].size();
        std::vector<cv::Mat> tiles;
        for (int q = 0; q < 3; ++q) {
            std::vector<Mark> mk;
            for (size_t k = 0; k < gpts.size(); ++k)
                if (ext.regions[k] == q) mk.push_back({gpts[k] + D[i][k], {D[i][k].x - mx, D[i][k].y - my}, "", kRed});
            const int* b = qr::BOXES[q];
            cv::Rect2d rf(b[0], b[1], b[2] - b[0], b[3] - b[1]), nw(rf.x + mx, rf.y + my, rf.width, rf.height);
            cv::Rect roi = cv::Rect(rf | nw) + cv::Size(30, 30) - cv::Point(15, 15);
            tiles.push_back(zoom(im, roi, 460, mk, 200, q == 0 ? cap : "", {{rf, kGreen, "center"}, {nw, kRed, ""}}));
        }
        save_png(file, hstack(tiles));
    };
    // Y 튀는 값: gn_v7b 반복에서 가장 자주 실패한 Y 사진
    int yi = -1, yfail = 0;
    for (auto& kv : top["gn_v7b"]) {
        for (int i = 0; i < N; ++i) if (m[i].file == kv.second && m[i].axis == "Y") { yi = i; yfail = -kv.first; }
        if (yi >= 0) break;
    }
    std::string fT = "04_튀는값_T_T-0.05.png", fY = "05_튀는값_Y_" + stem(m[yi].file) + ".png";
    photo(target, fT, "T-0.05 mean dx " + fmt(meanD("T-0.05.bmp", 0), 3) + " px, six wrong " + std::to_string(REP["gn_six"].t005) + "/20");
    photo(yi, fY, stem(m[yi].file) + " v7b failed " + std::to_string(yfail) + "/20");

    // T-0.05 후보 비용 (full280, 6후보)
    Table cand{{"후보 구간", "축", "값", "비용", "부호 허용"}};
    if (!six_target_cands.empty())
        for (auto& c : six_target_cands[0]) cand.add({c.branch, c.axis, fmts(c.v, 4), fmt(c.cost, 3), c.allowed ? "예" : "아니오"});
    Table outl{{"구분", "사진", "정답", "full280 gn_six 판정 / 값", "full280 gn_v7b 판정 / 값", "반복 실패 gn_six / gn_v7b (/20)", "판 평균 이동 dx / dy (px)", "b1x / b3x (px)"}};
    for (int i : {target, yi}) {
        std::string s6, s7;
        for (auto& p : P["gn_six|full280"]) if (p.i == i) s6 = p.pred_axis + " " + fmts(p.pred, 4);
        for (auto& p : P["gn_v7b|full280"]) if (p.i == i) s7 = p.pred_axis + " " + fmts(p.pred, 4);
        outl.add({i == target ? "T 튀는 값" : "Y 튀는 값", stem(m[i].file), m[i].axis + " " + fmts(m[i].value, 2), s6, s7,
                  std::to_string(REP["gn_six"].freq[m[i].file]) + " / " + std::to_string(REP["gn_v7b"].freq[m[i].file]),
                  fmts(meanD(m[i].file, 0), 3) + " / " + fmts(meanD(m[i].file, 1), 3), fmts(X[i][2], 4) + " / " + fmts(X[i][6], 4)});
    }

    // CSV
    {
        Table pr{{"method", "protocol", "file", "axis", "value", "pred_axis", "pred", "branch", "cost_Y", "cost_T", "pass09"}};
        for (auto& me : methods)
            for (auto& p_ : prot_order)
                for (auto& p : P[me + "|" + p_])
                    pr.add({me, p_, m[p.i].file, p.axis, fmt(p.value, 2), p.pred_axis, fmt(p.pred, 6), p.branch, fmt(p.costY, 5), fmt(p.costT, 5), six::pass09(p) ? "1" : "0"});
        pr.csv(out("predictions.csv"));
        Table tc{{"protocol", "fold", "method", "pred_axis", "pred", "cost_Y", "cost_T", "log_cost_ratio"}};
        for (auto& t : trow) tc.add({t.protocol, t.fold, t.method, t.pred_axis, fmt(t.pred, 6), fmt(t.cY, 5), fmt(t.cT, 5), fmt(t.lr, 5)});
        tc.csv(out("T005_predictions_and_costs.csv"));
        Table ft{{"file", "Y", "T", "mean_dx", "mean_dy", "b0x", "b0y", "b1x", "b1y", "b2x", "b2y", "b3x", "b3y", "b4x", "b4y", "b5x", "b5y"}};
        for (int i = 0; i < N; ++i) {
            std::vector<std::string> r{m[i].file, fmt(m[i].axis == "Y" ? m[i].value : 0, 2), fmt(m[i].axis == "T" ? m[i].value : 0, 2), fmt(meanD(m[i].file, 0), 5), fmt(meanD(m[i].file, 1), 5)};
            for (double v : X[i]) r.push_back(fmt(v, 6));
            ft.add(r);
        }
        ft.csv(out("qr205_p12_features.csv"));
    }

    auto& r6 = REP["gn_six"];
    auto& r7 = REP["gn_v7b"];
    // ---- 결과.md ----
    std::ostringstream R;
    R << "# 결과: T-0.05 오분류, 6후보 GLS vs v7b 2단계\n\n"
      << "입력: `data/clean/clean_20261014` 281장. 측정은 QR205 (center 기준 고정 ID 점 " << gpts.size() << "개, QR 상자마다 잘라 NCC -> LK15 -> GN17)이고 모든 사진에서 다시 쟀다.\n"
      << "특징: 점 이동을 판 2차 변형 12값으로 요약. 정답은 파일명 명령값.\n\n"
      << "- gn_six: 촬영 구간 6개(Y1±, Y2±, T±)마다 0점을 따로 둔 2차 모델, 축·구간 잔차 상관 GLS, 후보 값 부호가 구간 부호와 맞는 것 중 비용 최소.\n"
      << "- gn_v7b: 축 판정 = |Y|<=0.6, |T|<=0.12 사진의 1차 모델(절편 0, center 포함), 값 = 전체 2차 모델(절편 자유). 세로 이동·화면 회전을 화이트닝에서 투영으로 뺀다.\n\n"
      << "분할 25 프로토콜 119개(full280 재대입, interleaved 2, stratified5, blocked_within_session5, StratifiedKFold 반복 20회 seed 7000~7019, session_out 6)는 원래 splits.json과 같다.\n\n"
      << "## 프로토콜별 지표\n\n" << mt.md()
      << "\n## 무작위 5분할 반복 20회\n\n"
      << "| 방법 | 실패 합 (/5600) | 전부 맞은 반복 | 반복당 최대 실패 | T-0.05 오분류 (/20) |\n|---|---|---|---|---|\n"
      << "| gn_six | " << r6.fail << " | " << r6.zero << " | " << r6.maxfail << " | " << r6.t005 << " |\n"
      << "| gn_v7b | " << r7.fail << " | " << r7.zero << " | " << r7.maxfail << " | " << r7.t005 << " |\n\n"
      << rp.md() << "\n## 반복에서 자주 실패한 사진\n\n" << freq.md()
      << "\n## T-0.05의 6후보 비용 (full280)\n\n" << cand.md()
      << "\n## 원래 결과와 대조 (" << par.ok << "/" << par.total << " 일치)\n\n" << par.t.md()
      << "\n## 옮기지 않은 부분\n\n- pixel_six, pixel_v7b: 직전 실험(pixel_weighted_gn_20261015)의 분할별 픽셀 잡음 곡선으로 가중한 GN 측정. 잡음 곡선 학습 전체가 필요해 옮기지 않았다.\n"
      << "- analyze.py의 백색화 성분별 기여·교차 진단 그림(03~05), verify.py 감사, package.py.\n\n"
      << "## 파일\n\n- `predictions.csv`: 방법 x 프로토콜 사진별 판정\n- `T005_predictions_and_costs.csv`: T-0.05의 분할별 비용\n- `qr205_p12_features.csv`: 사진별 평균 이동과 12값\n";
    write_text(out("결과.md"), R.str());

    // ---- 보고서.md ----
    auto& f6 = SC["gn_six|full280"];
    auto& f7 = SC["gn_v7b|full280"];
    std::ostringstream Pm;
    Pm << "# 보고서: T-0.05 오분류 원인과 v7b 2단계 비교 (10-14 16:16)\n\n"
       << "## 문제\n\n반복 분할에서 T-0.05가 자주 Y로 오분류되는 원인을 찾고 v7b 2단계 판정과 비교해야 했다.\n\n"
       << "## 해결 방법\n\n같은 QR205 측정과 같은 119개 분할에서 6후보 GLS와 v7b 2단계를 돌리고, T-0.05의 Y/T 후보 비용을 분할마다 기록했다.\n\n"
       << "## 결과\n\nT-0.05는 반복 20회 중 6후보에서 " << r6.t005 << "번, v7b에서 " << r7.t005 << "번 Y로 판정됐고, 반복 전체 실패도 " << r6.fail << " 대 " << r7.fail << "장(/5600)으로 6후보가 적었다.\n\n"
       << "| 지표 | gn_six | gn_v7b |\n|---|---|---|\n"
       << "| T-0.05 오분류 (/20) | " << r6.t005 << " | " << r7.t005 << " |\n"
       << "| 반복 20회 실패 합 (/5600) | " << r6.fail << " | " << r7.fail << " |\n"
       << "| 전부 맞은 반복 (/20) | " << r6.zero << " | " << r7.zero << " |\n"
       << "| full280 실패 (/280) | " << f6.failed << " | " << f7.failed << " |\n"
       << "| full280 Y 평균 / 최대 \\|오차\\| (mm) | " << fmt(f6.YMAE, 4) << " / " << fmt(f6.Ymax, 4) << " | " << fmt(f7.YMAE, 4) << " / " << fmt(f7.Ymax, 4) << " |\n"
       << "| full280 T 평균 / 최대 \\|오차\\| (deg) | " << fmt(f6.TMAE, 5) << " / " << fmt(f6.Tmax, 5) << " | " << fmt(f7.TMAE, 5) << " / " << fmt(f7.Tmax, 5) << " |\n"
       << "| T-0.05 log(비용Y/비용T) 반복 평균 | " << fmts(mean(sl6), 3) << " | " << fmts(mean(sl7), 3) << " |\n\n"
       << img_md("01_T-0.05_비용비_반복20회.png", "T-0.05 비용비") << "\n\n" << img_md("02_프로토콜별_실패장수.png", "프로토콜별 실패") << "\n\n"
       << img_md("03_반복분할별_실패.png", "반복 분할별 실패") << "\n\n"
       << "## 튀는 값\n\nT-0.05는 판 평균 이동 " << fmt(meanD("T-0.05.bmp", 0), 3) << " px로 Y " << fmts(meanD("T-0.05.bmp", 0) / 14.25, 2) << "mm(14.25 px/mm)와 같은 크기라 6후보에서는 Y·T 비용비가 0 근처(반복 평균 " << fmts(mean(sl6), 3)
       << ")에서 분할마다 뒤집히고, v7b는 비용비가 항상 음수(" << fmts(*std::max_element(sl7.begin(), sl7.end()), 3) << " 이하)라 매번 Y가 됐다.\n\n"
       << outl.md() << "\n" << img_md(fY, "Y 튀는 값") << "\n\n" << img_md(fT, "T 튀는 값")
       << "\n\n초록 상자는 center의 QR 상자, 빨간 상자는 판 평균 이동만큼 옮긴 상자다. 빨간 십자는 이 사진의 점 위치, 화살표는 (점 이동 - 판 평균 이동) x 200이다.\n\n"
       << "## 한줄평\n\n6후보는 촬영 구간마다 0점을 따로 맞춰 T-0.05의 작은 휨을 T 구간 0점에 붙일 수 있어 비용비가 0 근처(" << fmts(mean(sl6), 3)
       << ")에 머물렀고, v7b는 0점을 center 하나로 고정해 그 0점 차이가 Y 쪽 비용을 낮춰서 T-0.05를 20회 모두 Y로 판정했다.\n\n"
       << "## 원래 결과와 대조\n\n원래 값은 `t005_v7b_20261015/metrics.csv`, `predictions.csv`, `T005_predictions_and_costs.csv`, `splits.json`과 `measurement/gn17_spread14.npz`에서 가져왔다. "
       << par.ok << "/" << par.total << " 일치.\n\n" << par.t.md();
    write_text(out("보고서.md"), Pm.str());
    std::cout << "parity " << par.ok << "/" << par.total << "\n";
    return 0;
}
