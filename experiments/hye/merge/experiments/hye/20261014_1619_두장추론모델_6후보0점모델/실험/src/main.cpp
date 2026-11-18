// 센터·현재 두 장 추론 모델: QR205 점 -> 판 2차 변형 12값 -> 촬영 구간 6개 0점 후보 GLS
// (원본: codex_experiments_20261014/validate_calibrated_runs.py, zero_experiment.py, infer.py, measurement/spread.py)
#include <hye/dataset.hpp>
#include <hye/plot.hpp>
#include <hye/report.hpp>
#include <omp.h>
#include <iostream>
#include <map>
#include "qr205.hpp"
#include "sixcand.hpp"

using namespace hye;

namespace {

std::string stem(const std::string& f) { return f.substr(0, f.rfind('.')); }

// zero_experiment.correct_labels: 학습 사진만으로 축별 0점 보정 라벨 ('session' 구간 절편, 'global' 축 절편)
std::vector<double> correct_labels(const std::vector<six::Meta>& m, const std::vector<double>& x, const std::vector<int>& tr, const std::string& mode) {
    std::vector<double> val(m.size());
    for (size_t i = 0; i < m.size(); ++i) val[i] = m[i].value;
    for (std::string a : {"Y", "T"}) {
        std::vector<int> ii;
        for (int i : tr) if (m[i].axis == a) ii.push_back(i);
        std::vector<std::string> ses;
        for (int i : ii) ses.push_back(m[i].session);
        std::sort(ses.begin(), ses.end());
        ses.erase(std::unique(ses.begin(), ses.end()), ses.end());
        int nc = mode == "session" ? 1 + (int)ses.size() : 2;
        cv::Mat A((int)ii.size(), nc, CV_64F, 0.0), y((int)ii.size(), 1, CV_64F);
        for (int r = 0; r < A.rows; ++r) {
            A.at<double>(r, 0) = m[ii[r]].value;
            if (mode == "session") A.at<double>(r, 1 + (int)(std::find(ses.begin(), ses.end(), m[ii[r]].session) - ses.begin())) = 1;
            else A.at<double>(r, 1) = 1;
            y.at<double>(r) = x[ii[r]];
        }
        cv::Mat co = six::lstsq(A, y);
        for (int r = 0; r < A.rows; ++r) {
            double off = 0;
            for (int c = 1; c < nc; ++c) off += A.at<double>(r, c) * co.at<double>(c);
            val[ii[r]] = m[ii[r]].value + off / co.at<double>(0);
        }
    }
    return val;
}

// oracle_zero: 시험 사진의 정답 구간 절편을 두 축 모두에 준다 (진단용)
six::Pred predict_oracle(const std::vector<six::Meta>& m, const std::vector<double>& x, int i, const six::SixModel& md) {
    cv::Mat base;
    std::map<std::string, const six::Branch*> ax;
    for (auto& b : md.br) {
        if (b.name == m[i].session) base = b.base;
        if (!ax.count(b.axis)) ax[b.axis] = &b;
    }
    six::Pred best{i, m[i].axis, "", "", m[i].value, 0, std::numeric_limits<double>::infinity(), 0, 0};
    for (std::string a : {"Y", "T"}) {
        six::Branch b{a, a, base, ax[a]->s, ax[a]->c};
        auto c = six::solve_branch(b, md.W, x, false);
        (a == "Y" ? best.costY : best.costT) = c.cost;
        if (c.cost < best.cost) { best.cost = c.cost; best.pred_axis = a; best.branch = a; best.pred = c.v; }
    }
    return best;
}

struct Vec { double ymax = 0, tmax = 0; };
Vec vector_max(const std::vector<six::Pred>& P) {
    Vec v;
    for (auto& p : P) {
        double tY = p.axis == "Y" ? p.value : 0, tT = p.axis == "T" ? p.value : 0, qY = p.pred_axis == "Y" ? p.pred : 0, qT = p.pred_axis == "T" ? p.pred : 0;
        v.ymax = std::max(v.ymax, std::abs(qY - tY));
        v.tmax = std::max(v.tmax, std::abs(qT - tT));
    }
    return v;
}

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
    const int N = (int)m.size();
    cv::Mat ref = qr::read_crop(shots[m[0].shot]);
    qr::Spread ext(ref, 14);
    auto gpts = ext.global_points();
    std::vector<std::vector<cv::Point2d>> Draw(N);
#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < N; ++i) Draw[i] = ext.extract(qr::read_crop(shots[m[i].shot])).D;
    std::vector<std::vector<cv::Point2d>> D(N, std::vector<cv::Point2d>(gpts.size()));
    std::vector<double> xm(N, 0);
    for (int i = 0; i < N; ++i) {
        for (size_t k = 0; k < gpts.size(); ++k) { D[i][k] = Draw[i][k] - Draw[0][k]; xm[i] += D[i][k].x; }
        xm[i] /= gpts.size();
    }
    qr::P12 p12(gpts);
    std::vector<std::vector<double>> X(N);
    for (int i = 0; i < N; ++i) X[i] = p12(D[i]);
    cv::Mat U(12, 2, CV_64F);  // Z.load: 세로 이동, (1906, 1070) 기준 회전
    {
        for (int j = 0; j < 2; ++j) {
            std::vector<cv::Point2d> f;
            for (auto& p : gpts) f.push_back(j == 0 ? cv::Point2d(0, 1) : cv::Point2d(-(p.y - 1070) / 100, (p.x - 1906) / 100));
            auto v = p12(f);
            for (int r = 0; r < 12; ++r) U.at<double>(r, j) = v[r];
        }
    }
    std::vector<double> lab(N);
    for (int i = 0; i < N; ++i) lab[i] = m[i].value;
    std::cout << "measured " << gpts.size() << " points\n";

    // 구간별 0점 절편 (session_zero_estimates)
    Table zt{{"구간", "장수", "기울기 (px/단위)", "0점 절편 (px)", "등가 이동량", "절편 표준오차 (px)"}};
    std::map<std::string, std::array<double, 3>> ZE;
    for (std::string s : {"T+", "T-", "Y1+", "Y1-", "Y2+", "Y2-"}) {
        std::vector<int> ii;
        for (int i = 0; i < N; ++i) if (m[i].session == s) ii.push_back(i);
        cv::Mat A((int)ii.size(), 2, CV_64F), y((int)ii.size(), 1, CV_64F);
        for (int r = 0; r < A.rows; ++r) { A.at<double>(r, 0) = 1; A.at<double>(r, 1) = m[ii[r]].value; y.at<double>(r) = xm[ii[r]]; }
        cv::Mat p = six::lstsq(A, y), res = y - A * p;
        double se = std::sqrt(res.dot(res) / (A.rows - 2) * cv::Mat((A.t() * A).inv()).at<double>(0, 0));
        ZE[s] = {p.at<double>(1), p.at<double>(0), p.at<double>(0) / p.at<double>(1)};
        zt.add({s, std::to_string(ii.size()), fmt(p.at<double>(1), 3), fmts(p.at<double>(0), 4), fmts(ZE[s][2], 6), fmt(se, 4)});
    }

    // ---- 스트레스 검증 (validate_calibrated_runs): 6후보, 2차, 방해 무늬 없음 ----
    auto splits = six::protocols(m);
    std::vector<std::string> cvs;
    std::map<std::string, std::vector<six::Pred>> SP;
    for (auto& sp : splits) {
        if (sp.protocol == "full280" || sp.protocol == "session_out_missing_origin") continue;
        if (cvs.empty() || cvs.back() != sp.protocol) cvs.push_back(sp.protocol);
        auto md = six::fit(m, X, sp.tr, lab, "inferred_session", true);
        for (int i : sp.te) SP[sp.protocol].push_back(six::predict_one(m, X[i], i, md, "inferred_session"));
    }
    Table st{{"검증", "Y 축+부호", "T 축+부호", "축 맞힌 Y 최대 |오차|", "축 맞힌 T 최대 |오차|", "출력 벡터 Y / T 최대 |오차|", "틀린 사진"}};
    std::map<std::string, six::Score> SS;
    std::map<std::string, Vec> SV;
    int rep_zero = 0, rep_fail = 0;
    std::map<std::string, int> wrongfreq;
    std::vector<double> sx, sf;
    for (auto& c : cvs) {
        auto s = six::score(SP[c]);
        auto v = vector_max(SP[c]);
        SS[c] = s; SV[c] = v;
        std::string wl;
        for (auto& p : SP[c])
            if (!(p.axis == p.pred_axis && ((p.value > 0) == (p.pred > 0)))) {
                wl += (wl.empty() ? "" : ", ") + stem(m[p.i].file);
                if (c.rfind("repeated", 0) == 0) wrongfreq[m[p.i].file]++;
            }
        if (c.rfind("repeated", 0) == 0) {
            rep_zero += s.correct == 280;
            rep_fail += 280 - s.correct;
            sx.push_back(std::stoi(c.substr(14)));
            sf.push_back(280 - s.correct);
        }
        st.add({c, std::to_string(s.Ycorrect) + "/80", std::to_string(s.Tcorrect) + "/200", fmt(s.Ymax, 6), fmt(s.Tmax, 6), fmt(v.ymax, 6) + " / " + fmt(v.tmax, 6), wl.empty() ? "-" : wl});
    }

    // ---- 0점 처리 비교 (zero_experiment, 촬영 시각이 필요한 drift 제외) ----
    struct ZC { std::string name, mode, cor; };
    std::vector<ZC> zcs{{"raw_center", "center", ""}, {"raw_free", "free", ""}, {"center_relative_session", "center", "session"},
                        {"center_relative_global", "center", "global"}, {"inferred_session", "inferred_session", ""}, {"oracle_zero", "oracle_zero", ""}};
    struct ZR { std::string cv, cfg; bool nu, quad; six::Score s; };
    std::vector<ZR> zres;
    for (std::string cv : {"interleaved", "stratified5", "session_out_missing_origin"})
        for (auto& zc : zcs) {
            if (cv == "session_out_missing_origin" && zc.mode == "oracle_zero") continue;
            for (bool nu : {false, true})
                for (bool quad : {false, true}) {
                    std::vector<six::Pred> P;
                    for (auto& sp : splits) {
                        if (sp.protocol != cv) continue;
                        auto lb = zc.cor.empty() ? lab : correct_labels(m, xm, sp.tr, zc.cor);
                        auto md = six::fit(m, X, sp.tr, lb, zc.mode, quad, nu ? &U : nullptr);
                        for (int i : sp.te) P.push_back(zc.mode == "oracle_zero" ? predict_oracle(m, X[i], i, md) : six::predict_one(m, X[i], i, md, zc.mode));
                    }
                    zres.push_back({cv, zc.name, nu, quad, six::score(P)});
                }
        }
    Table za{{"0점 처리", "방해 무늬 제거", "2차", "Y 축+부호", "T 축+부호", "축 맞힌 Y 최대 |오차|", "축 맞힌 T 최대 |오차|"}};
    for (auto& z : zres)
        if (z.cv == "interleaved")
            za.add({z.cfg, z.nu ? "예" : "아니오", z.quad ? "예" : "아니오", std::to_string(z.s.Ycorrect) + "/80", std::to_string(z.s.Tcorrect) + "/200", fmt(z.s.Ymax, 6), fmt(z.s.Tmax, 6)});
    Table zb{{"검증", "방해 무늬 제거", "2차", "Y 축+부호", "T 축+부호"}};
    for (auto& z : zres)
        if (z.cv != "interleaved" && z.cfg == "inferred_session")
            zb.add({z.cv, z.nu ? "예" : "아니오", z.quad ? "예" : "아니오", std::to_string(z.s.Ycorrect) + "/80", std::to_string(z.s.Tcorrect) + "/200"});

    // ---- 두 장 추론 (infer.py calibrated-runs): 280장 전체로 맞춘 모델 + 사진 두 장에서 새로 측정 ----
    std::vector<int> ids;
    for (int i = 1; i < N; ++i) ids.push_back(i);
    auto full = six::fit(m, X, ids, lab, "inferred_session", true);
    Table it{{"현재 사진", "정답", "판정 축", "값", "선택 후보", "최소 비용", "다른 축 최소 비용", "정규화 비용차", "배치 측정과 값 차이"}};
    std::vector<std::string> demo{"T-0.05.bmp", "T+0.01.bmp", "Y-0.05.bmp", "Y+0.05.bmp", "T+1.00.bmp", "Y-2.00.bmp"};
    double max_batch_diff = 0;
    std::map<std::string, std::pair<std::string, double>> INF;
    for (auto& f : demo) {
        int i = -1;
        for (int k = 0; k < N; ++k) if (m[k].file == f) i = k;
        qr::Spread pe(qr::read_crop(shots[m[0].shot]), 14);  // 두 사진만 입력: center 로 새 추출기
        auto o = pe.extract(qr::read_crop(shots[m[i].shot]));
        auto x = qr::P12(pe.global_points())(o.D);  // infer.py: zero_feature(= 학습 center 행, 0) 만 뺀다
        std::vector<six::Cand> cands;
        auto p = six::predict_one(m, x, i, full, "inferred_session", &cands);
        double other = std::numeric_limits<double>::infinity();
        for (auto& c : cands) if (c.axis != p.pred_axis && c.allowed) other = std::min(other, c.cost);
        auto pb = six::predict_one(m, X[i], i, full, "inferred_session");
        max_batch_diff = std::max(max_batch_diff, std::abs(pb.pred - p.pred));
        INF[f] = {p.pred_axis, p.pred};
        it.add({stem(f), m[i].axis + " " + fmts(m[i].value, 2), p.pred_axis, fmts(p.pred, 5), p.branch, fmt(p.cost, 3), fmt(other, 3), fmt((other - p.cost) / (other + p.cost + 1e-12), 4),
                fmt(std::abs(pb.pred - p.pred), 6)});
    }

    // ---- 그림 ----
    {
        std::vector<double> ty, py, tt, pt;
        std::vector<std::string> nm;
        for (auto& p : SP["interleaved"]) {
            ty.push_back(p.axis == "Y" ? p.value : 0); tt.push_back(p.axis == "T" ? p.value : 0);
            py.push_back(p.pred_axis == "Y" ? p.pred : 0); pt.push_back(p.pred_axis == "T" ? p.pred : 0);
            nm.push_back(stem(m[p.i].file));
        }
        pred_vs_truth("01_짝홀_예측오차.png", "6-origin GLS, interleaved", ty, py, tt, pt, nm);
    }
    {
        Chart c("Repeated stratified 5-fold (20 seeds): misjudged images", "seed (7000 + k)", "wrong axis or sign (of 280)");
        c.line(sx, sf, color(1), "", true);
        c.save("02_반복분할_실패.png");
    }
    {
        std::vector<std::string> cats;
        std::vector<double> v;
        for (std::string s : {"T+", "T-", "Y1+", "Y1-", "Y2+", "Y2-"}) { cats.push_back(s); v.push_back(std::abs(ZE[s][2]) * (s[0] == 'T' ? 1000 : 1000)); }
        bar_chart("03_구간별_0점절편.png", "|zero intercept| per run (T: 0.001 deg, Y: um)", cats, v, "|offset|");
    }
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
    std::vector<std::pair<int, std::string>> wf;
    for (auto& kv : wrongfreq) wf.push_back({-kv.second, kv.first});
    std::sort(wf.begin(), wf.end());
    int ti = -1, yi = -1;
    for (auto& w : wf)
        for (int k = 0; k < N; ++k)
            if (m[k].file == w.second) { if (m[k].axis == "T" && ti < 0) ti = k; if (m[k].axis == "Y" && yi < 0) yi = k; }
    std::string fT = "05_튀는값_T_" + stem(m[ti].file) + ".png", fY = "04_튀는값_Y_" + stem(m[yi].file) + ".png";
    photo(ti, fT, stem(m[ti].file) + " wrong " + std::to_string(wrongfreq[m[ti].file]) + "/20 repeats");
    photo(yi, fY, stem(m[yi].file) + " wrong " + std::to_string(wrongfreq[m[yi].file]) + "/20 repeats");
    Table wt{{"사진", "정답", "20회 중 오분류", "full280 모델 판정 / 값", "판 평균 이동 dx (px)", "b1x / b3x (px)", "가장 가까운 다른 축 후보와 비용차"}};
    for (auto& w : wf) {
        int i = -1;
        for (int k = 0; k < N; ++k) if (m[k].file == w.second) i = k;
        std::vector<six::Cand> cands;
        auto p = six::predict_one(m, X[i], i, full, "inferred_session", &cands);
        double other = std::numeric_limits<double>::infinity();
        for (auto& c : cands) if (c.axis != p.pred_axis && c.allowed) other = std::min(other, c.cost);
        wt.add({stem(w.second), m[i].axis + " " + fmts(m[i].value, 2), std::to_string(-w.first), p.pred_axis + " " + fmts(p.pred, 4), fmts(xm[i], 3),
                fmts(X[i][2], 4) + " / " + fmts(X[i][6], 4), fmt(other - p.cost, 3)});
    }

    // CSV
    {
        Table pr{{"cv", "file", "axis", "value", "pred_axis", "pred", "branch", "cost"}};
        for (auto& c : cvs) for (auto& p : SP[c]) pr.add({c, m[p.i].file, p.axis, fmt(p.value, 2), p.pred_axis, fmt(p.pred, 6), p.branch, fmt(p.cost, 5)});
        pr.csv(out("stress_predictions.csv"));
        Table zz{{"cv", "config", "nuisance", "quadratic", "Y_axis_sign_correct", "T_axis_sign_correct", "Y_matched_max", "T_matched_max"}};
        for (auto& z : zres) zz.add({z.cv, z.cfg, z.nu ? "1" : "0", z.quad ? "1" : "0", std::to_string(z.s.Ycorrect), std::to_string(z.s.Tcorrect), fmt(z.s.Ymax, 6), fmt(z.s.Tmax, 6)});
        zz.csv(out("zero_ablation.csv"));
    }

    // ---- 원래 결과와 대조 ----
    // 원래 값: calibrated_runs/stress_metrics.csv, zero_gn17_spread14/zero_ablation.csv, RESULTS.md 5절 표(session_zero_estimates)
    Parity par;
    par.add("QR 점 수", 205, (double)gpts.size(), 0, 0);
    const double sto[3][4] = {{0.037107, 0.003119, 1, 1}, {0.036530, 0.003098, 1, 1}, {0.036196, 0.003675, 1, 1}};
    const char* stn[3] = {"interleaved", "stratified5", "blocked_within_session5"};
    for (int k = 0; k < 3; ++k) {
        auto& s = SS[stn[k]];
        par.add(std::string(stn[k]) + " Y 축+부호 맞음 (/80)", 80, s.Ycorrect, 0, 0);
        par.add(std::string(stn[k]) + " T 축+부호 맞음 (/200)", 200, s.Tcorrect, 0, 0);
        par.add(std::string(stn[k]) + " Y 최대 |오차|", sto[k][0], s.Ymax, 0.0005, 6);
        par.add(std::string(stn[k]) + " T 최대 |오차|", sto[k][1], s.Tmax, 0.00005, 6);
    }
    par.add("반복 20회 중 전부 맞음", 9, rep_zero, 0, 0);
    par.add("반복 20회 오분류 합", 13, rep_fail, 0, 0);
    par.add("T-0.05 오분류 횟수 (/20)", 11, wrongfreq["T-0.05.bmp"], 0, 0);
    par.add("T+0.01 오분류 횟수 (/20)", 1, wrongfreq["T+0.01.bmp"], 0, 0);
    par.add("Y-0.05 오분류 횟수 (/20)", 1, wrongfreq["Y-0.05.bmp"], 0, 0);
    double vy = 0, vt = 0;
    for (auto& c : cvs) if (c.rfind("repeated", 0) == 0) { vy = std::max(vy, SV[c].ymax); vt = std::max(vt, SV[c].tmax); }
    par.add("반복 출력 벡터 Y 최대 |오차|", 0.267574, vy, 0.001, 6);
    par.add("반복 출력 벡터 T 최대 |오차|", 0.05, vt, 0.0001, 6);
    const double zo[6][2] = {{80.296, -0.8349}, {80.778, -0.1922}, {14.360, -0.5392}, {14.393, 0.1104}, {14.389, -0.4501}, {14.362, -0.4291}};
    int zk = 0;
    for (std::string s : {"T+", "T-", "Y1+", "Y1-", "Y2+", "Y2-"}) {
        par.add(s + " 기울기 (px/단위)", zo[zk][0], ZE[s][0], 0.002, 3);
        par.add(s + " 0점 절편 (px)", zo[zk][1], ZE[s][1], 0.0005, 4);
        ++zk;
    }
    struct ZO { const char* cfg; bool nu, quad; double ya, ta, ym, tm; };
    const ZO zor[] = {{"raw_center", 0, 0, .925, .995, .033377, .009936}, {"raw_center", 0, 1, .9375, .995, .029319, .009235}, {"raw_center", 1, 0, .9375, .995, .035438, .010915},
                      {"raw_center", 1, 1, .9375, .995, .036577, .009252}, {"raw_free", 0, 0, .925, .985, .040511, .004676}, {"raw_free", 0, 1, .9375, .995, .038576, .004529},
                      {"raw_free", 1, 0, .9375, .985, .030362, .005164}, {"raw_free", 1, 1, .95, .985, .029323, .005073},
                      {"center_relative_session", 0, 0, .9375, .995, .042660, .013598}, {"center_relative_session", 0, 1, .95, .995, .042791, .012945},
                      {"center_relative_session", 1, 0, .9375, .995, .045345, .013535}, {"center_relative_session", 1, 1, .9375, .995, .043491, .012841},
                      {"center_relative_global", 0, 0, .9375, .995, .055050, .012112}, {"center_relative_global", 0, 1, .95, .995, .052076, .012419},
                      {"center_relative_global", 1, 0, .9375, .99, .056223, .011487}, {"center_relative_global", 1, 1, .9375, .995, .053667, .012268},
                      {"inferred_session", 0, 0, 1, 1, .035407, .003234}, {"inferred_session", 0, 1, 1, 1, .037107, .003119},
                      {"inferred_session", 1, 0, .95, .98, .037710, .002197}, {"inferred_session", 1, 1, .95, .99, .037318, .003132},
                      {"oracle_zero", 0, 0, .9625, .995, .010481, .003234}, {"oracle_zero", 0, 1, .9625, .995, .010573, .003119},
                      {"oracle_zero", 1, 0, .9625, .995, .009256, .003104}, {"oracle_zero", 1, 1, .9625, .995, .009485, .003132}};
    for (auto& o : zor)
        for (auto& z : zres)
            if (z.cv == "interleaved" && z.cfg == o.cfg && z.nu == o.nu && z.quad == o.quad) {
                std::string k = std::string("짝홀 ") + o.cfg + (o.nu ? " 방해제거" : "") + (o.quad ? " 2차" : " 1차");
                par.add(k + " Y 축+부호 (/80)", o.ya * 80, z.s.Ycorrect, 0, 0);
                par.add(k + " T 축+부호 (/200)", o.ta * 200, z.s.Tcorrect, 0, 0);
                par.add(k + " Y 최대 |오차|", o.ym, z.s.Ymax, 0.0005, 6);
            }
    const double zso[8][2] = {{1, .995}, {1, 1}, {.9375, .97}, {.95, .99}, {.7625, 0}, {.725, 0}, {.9, 0}, {.85, 0}};
    int q = 0;
    for (auto& z : zres)
        if (z.cv != "interleaved" && z.cfg == "inferred_session") {
            std::string k = z.cv.substr(0, 11) + " 6후보" + (z.nu ? " 방해제거" : "") + (z.quad ? " 2차" : " 1차");
            par.add(k + " Y 축+부호 (/80)", zso[q][0] * 80, z.s.Ycorrect, 0, 0);
            par.add(k + " T 축+부호 (/200)", zso[q][1] * 200, z.s.Tcorrect, 0, 0);
            ++q;
        }
    par.add("두 장 추론 = 배치 측정 (값 최대 차이)", 0, max_batch_diff, 1e-6, 7);

    auto& si = SS["interleaved"];
    // ---- 결과.md ----
    std::ostringstream R;
    R << "# 결과: 두 장 추론 모델 (QR205 + 6개 0점 후보)\n\n"
      << "입력: `data/clean/clean_20261014` 281장. center와 현재 사진 두 장에서 QR 상자마다 점 " << gpts.size() << "개를 추적(NCC -> LK15 -> GN17)하고, 판 2차 변형 12값을 특징으로 쓴다.\n"
      << "모델: 축마다 2차 곡선 + 촬영 구간 6개(T±, Y1±, Y2±)의 0점 절편. 후보마다 값을 가우스-뉴턴으로 풀고, 구간 부호와 맞는 후보 중 잔차 상관 GLS 비용이 가장 작은 것을 고른다. 정답은 파일명 명령값.\n\n"
      << "## 스트레스 검증 (validate_calibrated_runs)\n\n" << st.md()
      << "\n## 반복 분할에서 틀린 사진\n\n" << wt.md()
      << "\n## 구간별 0점 절편 (평균 가로 이동 ~ 명령값 직선)\n\n" << zt.md()
      << "\n## 0점 처리 비교 (짝홀, zero_experiment)\n\n" << za.md() << "\n" << zb.md()
      << "\n`oracle_zero`는 시험 사진의 정답 구간 절편을 준 진단이라 두 장 추론 성적이 아니다.\n\n"
      << "## 두 장 추론 (infer.py calibrated-runs, 280장 전체로 맞춘 모델)\n\n" << it.md()
      << "\n두 장 추론은 center와 현재 사진만 받아 새 추출기로 다시 측정한다. 배치 측정으로 푼 값과 최대 " << fmt(max_batch_diff, 7) << " 차이.\n\n"
      << "## 원래 결과와 대조 (" << par.ok << "/" << par.total << " 일치)\n\n" << par.t.md()
      << "\n## 옮기지 않은 부분\n\n- zero_experiment의 `center_relative_drift`(촬영 시각·접근 방향 드리프트 보정): data/clean에 촬영 시각이 없다.\n"
      << "- lattice_experiment.py(명령 격자 prior), infer.py의 general 모드(modeling/deploy_model.pkl, 별도 실험 폴더 20261014_1621_일반모델선택), make_report.py, package_reference.py, verify_delivery.py, reproduce_measurements.py의 direct_alignment.\n\n"
      << "## 파일\n\n- `stress_predictions.csv`: 검증별 사진 판정\n- `zero_ablation.csv`: 0점 처리 비교 전체\n";
    write_text(out("결과.md"), R.str());

    // ---- 보고서.md ----
    std::ostringstream P;
    P << "# 보고서: 센터·현재 두 장 추론, 6개 0점 후보 모델 (10-14 16:19)\n\n"
      << "## 문제\n\n센터·현재 두 장만으로 Y/T를 내는 205점·P12·6후보 0점 모델을 만들고 검증해야 했다.\n\n"
      << "## 해결 방법\n\nQR 점 " << gpts.size() << "개 이동을 12값으로 요약하고 촬영 구간 6개의 0점을 후보로 둔 GLS 판정기를 여러 분할로 검증한 뒤, 두 사진 입력 추론 함수로 묶었다.\n\n"
      << "## 결과\n\n짝홀·층화·연속구간에서 Y " << si.Ycorrect << "/80, T " << si.Tcorrect << "/200으로 모두 맞혔으나, 무작위 5분할 20회 중 전부 맞힌 것은 " << rep_zero << "회였다.\n\n"
      << "| 지표 | 값 |\n|---|---|\n"
      << "| 짝홀 / 층화 / 연속구간 Y 맞음 | " << SS["interleaved"].Ycorrect << " / " << SS["stratified5"].Ycorrect << " / " << SS["blocked_within_session5"].Ycorrect << " (/80) |\n"
      << "| 짝홀 / 층화 / 연속구간 T 맞음 | " << SS["interleaved"].Tcorrect << " / " << SS["stratified5"].Tcorrect << " / " << SS["blocked_within_session5"].Tcorrect << " (/200) |\n"
      << "| 세 검증 Y 최대 \\|오차\\| (mm) | " << fmt(std::max({SS["interleaved"].Ymax, SS["stratified5"].Ymax, SS["blocked_within_session5"].Ymax}), 6) << " |\n"
      << "| 세 검증 T 최대 \\|오차\\| (deg) | " << fmt(std::max({SS["interleaved"].Tmax, SS["stratified5"].Tmax, SS["blocked_within_session5"].Tmax}), 6) << " |\n"
      << "| 무작위 5분할 20회 전부 맞음 | " << rep_zero << " / 20 |\n"
      << "| 반복 20회 오분류 합 (/5600) | " << rep_fail << " |\n\n"
      << img_md("01_짝홀_예측오차.png", "짝홀 예측 오차") << "\n\n" << img_md("02_반복분할_실패.png", "반복 분할 실패") << "\n\n" << img_md("03_구간별_0점절편.png", "0점 절편") << "\n\n"
      << "## 튀는 값\n\n반복 분할에서 가장 자주 틀린 사진은 " << stem(m[ti].file) << "(20회 중 " << wrongfreq[m[ti].file] << "회)이고, Y에서는 " << stem(m[yi].file) << "(" << wrongfreq[m[yi].file]
      << "회)이다. 둘 다 가장 작은 몇 스텝이라 다른 축 후보와 비용차가 작다.\n\n" << wt.md() << "\n" << img_md(fY, "Y 튀는 값") << "\n\n" << img_md(fT, "T 튀는 값")
      << "\n\n초록 상자는 center의 QR 상자, 빨간 상자는 판 평균 이동만큼 옮긴 상자다. 빨간 십자는 이 사진의 점 위치, 화살표는 (점 이동 - 판 평균 이동) x 200이다.\n\n"
      << "## 한줄평\n\n6후보 모델은 기존 6개 촬영 구간의 0점(T+ " << fmts(ZE["T+"][1], 3) << " px, T- " << fmts(ZE["T-"][1], 3) << " px 등)을 외워 짝홀 0점 처리 비교에서 유일하게 Y " << si.Ycorrect
      << "/80을 냈지만, 학습 분할이 바뀌면 그 절편 추정도 흔들려 T-0.05처럼 다른 축 후보와 비용차가 작은 사진이 20회 중 " << wrongfreq[m[ti].file] << "회 뒤집혔다.\n\n"
      << "## 원래 결과와 대조\n\n원래 값은 `codex_experiments_20261014/calibrated_runs/stress_metrics.csv`, `zero_gn17_spread14/zero_ablation.csv`, `RESULTS.md` 5절 표에서 가져왔다. "
      << par.ok << "/" << par.total << " 일치.\n\n" << par.t.md();
    write_text(out("보고서.md"), P.str());
    std::cout << "parity " << par.ok << "/" << par.total << "\n";
    return 0;
}
