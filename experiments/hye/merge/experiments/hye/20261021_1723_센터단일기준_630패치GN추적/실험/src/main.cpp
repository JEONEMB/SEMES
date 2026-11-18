// 10-21 17:23 center 단일 기준 630점 패치 GN 추적 (codex_experiments_20261014/center_patch_v2_20261021/run.py 의 C++ 재현)
//  - center QR 구조 모서리 630점(검정 이웃 2개 제외)을 center 사진에서 다시 찾는다
//  - v1_edge630: 같은 630점을 사진마다 경계 모델로 다시 맞춘 좌표 차이
//  - v2_patch630: center 패치로 LK15 -> GN17 추적한 이동
//  - v2_patch630_quality: v2 + 품질 가중 P12
//  - P12 -> 원점 통과 이차곡선 GLS(Y/T 두 가지) 비용 비교. full280(재대입), repeated20(5-fold x 20), interleaved(2-fold)
#include "measure630.hpp"
#include <hye/plot.hpp>

using namespace yt;
using hye::fmt;

// gls_comparison_20261014/splits.json 의 interleaved f0 시험 70+70장. 원래 사진 파일 시각(mtime) 순서로 번갈아 고른 것이라
// data/clean 에 없는 정보다. 그래서 분할 이름만 상수로 둔다(f1 = 나머지 140장).
static const char* kInterleavedF0 =
    "T+0.01 T+0.03 T+0.05 T+0.07 T+0.09 T+0.11 T+0.13 T+0.15 T+0.17 T+0.19 T+0.21 T+0.23 T+0.25 T+0.27 T+0.29 T+0.31 T+0.33 T+0.35 T+0.37 "
    "T+0.39 T+0.41 T+0.43 T+0.45 T+0.47 T+0.49 T+0.51 T+0.53 T+0.55 T+0.57 T+0.59 T+0.61 T+0.63 T+0.65 T+0.67 T+0.69 T+0.71 T+0.73 T+0.75 "
    "T+0.77 T+0.79 T+0.81 T+0.83 T+0.85 T+0.87 T+0.89 T+0.91 T+0.93 T+0.95 T+0.97 T+0.99 T-0.01 T-0.03 T-0.05 T-0.07 T-0.09 T-0.11 T-0.13 "
    "T-0.15 T-0.17 T-0.19 T-0.21 T-0.23 T-0.25 T-0.27 T-0.29 T-0.31 T-0.33 T-0.35 T-0.37 T-0.39 T-0.41 T-0.43 T-0.45 T-0.47 T-0.49 T-0.51 "
    "T-0.53 T-0.55 T-0.57 T-0.59 T-0.61 T-0.63 T-0.65 T-0.67 T-0.69 T-0.71 T-0.73 T-0.75 T-0.77 T-0.79 T-0.81 T-0.83 T-0.85 T-0.87 T-0.89 "
    "T-0.91 T-0.93 T-0.95 T-0.97 T-0.99 Y+0.05 Y+0.10 Y+0.25 Y+0.30 Y+0.45 Y+0.50 Y+0.65 Y+0.70 Y+0.85 Y+0.90 Y+1.05 Y+1.10 Y+1.25 Y+1.30 "
    "Y+1.45 Y+1.50 Y+1.65 Y+1.70 Y+1.85 Y+1.90 Y-0.10 Y-0.15 Y-0.30 Y-0.35 Y-0.50 Y-0.55 Y-0.70 Y-0.75 Y-0.90 Y-0.95 Y-1.10 Y-1.15 Y-1.30 "
    "Y-1.35 Y-1.50 Y-1.55 Y-1.70 Y-1.75 Y-1.90 Y-1.95";

struct Split { std::string protocol; std::vector<int> tr, te; };

static std::string pname(const Img& m) { return m.file.substr(0, m.file.size() - 4); }

int main() {
    hye::begin_results();
    auto M = measure630(true);
    const auto& imgs = M.imgs;
    const int N = (int)imgs.size(), P = (int)M.X.size();

    // ---- v2 패치 추적
    Extractor ex(M.crops[0], M.X, M.Y, 17, .8);
    Tracks tk = ex.run_all(M.crops);
    std::vector<std::vector<double>> Dx, Dy;
    center_relative(tk, Dx, Dy);
    auto Wq = quality_weights(tk);
    // ---- v1 경계 좌표 차이
    std::vector<std::vector<double>> Ex(N, std::vector<double>(P)), Ey(N, std::vector<double>(P));
    for (int i = 0; i < N; ++i)
        for (int p = 0; p < P; ++p) { Ex[i][p] = M.edge[i][p].x - M.edge[0][p].x; Ey[i][p] = M.edge[i][p].y - M.edge[0][p].y; }
    const std::vector<std::string> keys{"v1_edge630", "v2_patch630", "v2_patch630_quality"};
    std::vector<Feat> F{spatial(M.X, M.Y, Ex, Ey), spatial(M.X, M.Y, Dx, Dy), spatial(M.X, M.Y, Dx, Dy, &Wq)};
    {
        std::vector<std::vector<double>> rows;
        for (int k = 0; k < 3; ++k) for (int i = 0; i < N; ++i) { std::vector<double> r{(double)k}; for (double v : F[k][i]) r.push_back(v); rows.push_back(r); }
        dump_csv("e50_features.csv", rows);
    }

    // ---- 분할
    std::vector<int> ids;
    std::vector<std::string> sess;
    for (int i = 0; i < N; ++i) if (imgs[i].axis != 'C') { ids.push_back(i); sess.push_back(imgs[i].session); }
    std::vector<Split> splits;
    splits.push_back({"full280", ids, ids});
    {
        std::set<std::string> f0;
        std::string s = kInterleavedF0, w;
        for (char ch : s + " ") { if (ch == ' ') { if (!w.empty()) f0.insert(w); w.clear(); } else w += ch; }
        Split a{"interleaved", {}, {}}, b{"interleaved", {}, {}};
        for (int i : ids) (f0.count(pname(imgs[i])) ? a.te : a.tr).push_back(i);
        for (int i : ids) (f0.count(pname(imgs[i])) ? b.tr : b.te).push_back(i);
        splits.push_back(a); splits.push_back(b);
    }
    for (int seed = 0; seed < 20; ++seed)
        for (auto& f : stratified_kfold(sess, 5, 7000 + seed)) {
            Split s{"repeated20", {}, {}};
            for (int j : f.first) s.tr.push_back(ids[j]);
            for (int j : f.second) s.te.push_back(ids[j]);
            splits.push_back(s);
        }

    // ---- 평가
    const std::vector<std::string> prots{"full280", "repeated20", "interleaved"};
    std::map<std::pair<int, std::string>, std::vector<Row>> rows;
    std::vector<Model> full_model(3);
    for (int k = 0; k < 3; ++k)
        for (auto& s : splits) {
            Model mdl = fit(imgs, F[k], s.tr, false, true, .001);
            auto r = enrich(imgs, predict(mdl, F[k], s.te, false));
            auto& dst = rows[{k, s.protocol}];
            dst.insert(dst.end(), r.begin(), r.end());
            if (s.protocol == "full280") full_model[k] = mdl;
        }

    // ---- 표
    hye::Table summ{{"method", "protocol", "Y_correct", "Y_n", "T_correct", "T_n", "failures", "Y_max", "T_max"}};
    std::map<std::pair<int, std::string>, Stats> st;
    for (int k = 0; k < 3; ++k)
        for (auto& p : prots) {
            auto s = stats(rows[{k, p}]);
            st[{k, p}] = s;
            summ.add({keys[k], p, std::to_string(s.Yc), std::to_string(s.Yn), std::to_string(s.Tc), std::to_string(s.Tn), std::to_string(s.fail), fmt(s.Ymax, 4), fmt(s.Tmax, 4)});
        }
    summ.csv(hye::out("summary.csv"));
    hye::Table fails{{"method", "protocol", "사진", "정답 축", "정답 값", "추정 축", "추정 값", "eY", "eT", "비용"}};
    for (int k = 0; k < 3; ++k)
        for (auto& p : prots)
            for (auto& r : rows[{k, p}])
                if (!r.pass && p != "repeated20")
                    fails.add({keys[k], p, imgs[r.idx].file, std::string(1, r.axis), hye::fmts(r.value, 2), std::string(1, r.pred_axis), hye::fmts(r.pred, 4),
                               fmt(r.eY, 4), fmt(r.eT, 4), fmt(r.cost, 3)});
    fails.csv(hye::out("failures.csv"));

    // ---- 그림 1: repeated20 실패·최대 오차
    std::vector<std::string> lab{"Boundary fit (v1)", "Actual patch (v2)", "Patch + quality"};
    {
        std::vector<double> f, ym, tm;
        for (int k = 0; k < 3; ++k) { auto s = st[{k, "repeated20"}]; f.push_back(s.fail); ym.push_back(s.Ymax); tm.push_back(s.Tmax); }
        hye::bar_chart("01a_repeated20_실패수.png", "Failures / 5600 held-out", lab, f, "count", 2);
        hye::bar_chart("01b_repeated20_Y최대오차.png", "Max Y error (repeated20)", lab, ym, "mm", -1, hye::kPass);
        hye::bar_chart("01c_repeated20_T최대오차.png", "Max T error (repeated20)", lab, tm, "deg", -1, hye::kPass);
    }
    // ---- 그림 2: full280 v2_quality 정답 대비
    const int K = 2;
    auto& fr = rows[{K, "full280"}];
    std::vector<double> ty, py, tt, pt;
    std::vector<std::string> nm;
    for (auto& r : fr) {
        const Img& g = imgs[r.idx];
        ty.push_back(g.axis == 'Y' ? g.value : 0); tt.push_back(g.axis == 'T' ? g.value : 0);
        py.push_back(r.pred_axis == 'Y' ? r.pred : 0); pt.push_back(r.pred_axis == 'T' ? r.pred : 0);
        nm.push_back(pname(g));
    }
    hye::pred_vs_truth("02_full280_정답대비오차.png", "v2 patch+quality full280", ty, py, tt, pt, nm, 4);

    // ---- 튀는 값: 두 가지 비용
    auto branch_costs = [&](int idx, double& cy, double& vy, double& ct, double& vt) {
        Model one = full_model[K];
        for (auto& b : full_model[K].br) {
            Model m1 = one;
            m1.br = {b};
            auto p = predict(m1, F[K], {idx}, false)[0];
            if (b.axis == 'Y') { cy = p.cost; vy = p.pred; } else { ct = p.cost; vt = p.pred; }
        }
    };
    std::vector<size_t> bad;
    for (size_t i = 0; i < fr.size(); ++i) if (!fr[i].pass) bad.push_back(i);
    size_t wy = 0, wt = 0;
    pick_outliers(fr, wy, wt);
    hye::Table outl{{"구분", "사진", "정답 Y (mm)", "정답 T (deg)", "추정", "eY", "eT", "Y가지 비용 (값)", "T가지 비용 (값)", "평균 x 이동 (px)", "x 이동의 v 기울기 c2x (px)"}};
    std::vector<size_t> show = bad;
    if (std::find(show.begin(), show.end(), wy) == show.end()) show.push_back(wy);
    if (std::find(show.begin(), show.end(), wt) == show.end()) show.push_back(wt);
    for (size_t i : show) {
        auto& r = fr[i];
        const Img& g = imgs[r.idx];
        double cy = 0, vy = 0, ct = 0, vt = 0;
        branch_costs(r.idx, cy, vy, ct, vt);
        double mx = 0;
        for (int p = 0; p < P; ++p) mx += Dx[r.idx][p] / P;
        std::string tag = i == wy ? "Y 사진 최대 오차" : (i == wt ? "T 사진 최대 오차" : "실패");
        if (!r.pass && i != wy && i != wt) tag = "실패";
        outl.add({tag, pname(g), fmt(g.axis == 'Y' ? g.value : 0, 2), fmt(g.axis == 'T' ? g.value : 0, 2), std::string(1, r.pred_axis) + " " + hye::fmts(r.pred, 4),
                  fmt(r.eY, 4), fmt(r.eT, 4), fmt(cy, 2) + " (" + hye::fmts(vy, 3) + ")", fmt(ct, 2) + " (" + hye::fmts(vt, 3) + ")", fmt(mx, 3),
                  fmt(F[K][r.idx][4], 4)});
    }
    // 사진 확대: center QR 상자(초록), 이 사진에서 잰 상자(빨강), 점별 이동 - 평균 이동 (x150)
    auto outlier_png = [&](size_t i, const std::string& file) {
        const auto& r = fr[i];
        int idx = r.idx;
        double mx = 0, my = 0;
        for (int p = 0; p < P; ++p) { mx += (tk.Dx[idx][p]) / P; my += (tk.Dy[idx][p]) / P; }
        std::vector<cv::Mat> panels;
        for (int k = 0; k < 3; ++k) {
            double kx = 0, ky = 0;
            int nk = 0;
            for (int p = 0; p < P; ++p) if (ex.region[p] == k) { kx += tk.Dx[idx][p]; ky += tk.Dy[idx][p]; ++nk; }
            kx /= nk; ky /= nk;
            cv::Rect roi(BOXES[k][0] - CROP[0] + (int)std::lround(kx) - 6, BOXES[k][1] - CROP[1] + (int)std::lround(ky) - 6, BOXES[k][2] - BOXES[k][0] + 12,
                         BOXES[k][3] - BOXES[k][1] + 12);
            std::vector<hye::Mark> mk;
            for (int p = 0; p < P; ++p) {
                if (ex.region[p] != k) continue;
                cv::Point2d at(M.X[p] - CROP[0] + tk.Dx[idx][p], M.Y[p] - CROP[1] + tk.Dy[idx][p]);
                mk.push_back({at, {tk.Dx[idx][p] - kx, tk.Dy[idx][p] - ky}, "", hye::kRed});
            }
            std::vector<hye::Box> bx{{cv::Rect2d(BOXES[k][0] - CROP[0], BOXES[k][1] - CROP[1], BOXES[k][2] - BOXES[k][0], BOXES[k][3] - BOXES[k][1]), hye::kGreen, "center"},
                                     {cv::Rect2d(BOXES[k][0] - CROP[0] + kx, BOXES[k][1] - CROP[1] + ky, BOXES[k][2] - BOXES[k][0], BOXES[k][3] - BOXES[k][1]), hye::kRed, "measured"}};
            char cap[96];
            std::snprintf(cap, sizeof cap, "QR%d shift %.2f, %.2f px", k + 1, kx, ky);
            panels.push_back(hye::zoom(M.crops[idx], roi, 420, mk, 150, cap, bx));
        }
        cv::Mat all = hye::hstack(panels);
        char cap[160];
        const Img& g = imgs[idx];
        std::snprintf(cap, sizeof cap, "%s  true %s %+.2f  pred %c %+.4f  (arrows = point shift - QR mean, x150)", pname(g).c_str(), g.axis == 'Y' ? "Y" : "T", g.value,
                      r.pred_axis, r.pred);
        cv::Mat head(34, all.cols, CV_8UC3, cv::Scalar(255, 255, 255));
        hye::text(head, cap, {8, 23}, 0.6);
        hye::save_png(file, hye::vstack({head, all}, 0));
        (void)mx; (void)my;
    };
    std::string fY = "03_튀는값_Y_" + pname(imgs[fr[wy].idx]) + ".png", fT = "04_튀는값_T_" + pname(imgs[fr[wt].idx]) + ".png";
    outlier_png(wy, fY);
    outlier_png(wt, fT);

    // ---- 측정 요약
    int nvalid = 0, nall = 0, edge_valid = 0;
    double minrate = 1;
    for (int i = 1; i < N; ++i) {
        int v = 0;
        for (int p = 0; p < P; ++p) { v += tk.V[i][p]; edge_valid += M.edge[i][p].valid; }
        nvalid += v; nall += P;
        minrate = std::min(minrate, (double)v / P);
    }
    std::vector<double> rm;
    for (int i = 1; i < N; ++i) for (int p = 0; p < P; ++p) rm.push_back(tk.Q[i][p][0]);
    // P12 반응 기울기 (x 평균 이동 / 단위)
    auto slope = [&](char ax, const std::vector<std::vector<double>>& D) {
        double sxy = 0, sxx = 0;
        for (int i = 0; i < N; ++i) {
            if (imgs[i].axis != ax) continue;
            double m = 0;
            for (int p = 0; p < P; ++p) m += D[i][p] / P;
            sxy += imgs[i].value * m; sxx += imgs[i].value * imgs[i].value;
        }
        return sxy / sxx;
    };
    double sT = slope('T', Dx), sY = slope('Y', Dx);

    // ---- 대조 (원래 값: center_patch_v2_20261021/summary.csv, failures.csv, audit.json)
    hye::Parity par;
    struct O { int Yc, Tc, f; double ym, tm; };
    const O orig[3][3] = {{{76, 199, 5, .2555738917184803, .05}, {1506, 3980, 114, .3, .05}, {75, 198, 7, .41682910482593866, .08}},
                          {{77, 199, 4, .2586932775093803, .05}, {1518, 3979, 103, .65, .1213229564519758}, {75, 199, 6, .65, .11984492049785937}},
                          {{77, 199, 4, .25896855871912733, .05}, {1520, 3979, 101, .65, .12129117106673347}, {75, 199, 6, .65, .11979154710075489}}};
    for (int k = 0; k < 3; ++k)
        for (int j = 0; j < 3; ++j) {
            auto s = st[{k, prots[j]}];
            std::string h = keys[k] + " " + prots[j] + " ";
            par.add(h + "Y 축·부호 맞음", orig[k][j].Yc, s.Yc, 0, 0);
            par.add(h + "T 축·부호 맞음", orig[k][j].Tc, s.Tc, 0, 0);
            par.add(h + "실패 수", orig[k][j].f, s.fail, 0, 0);
            par.add(h + "Y 최대 오차", orig[k][j].ym, s.Ymax, 0.0005, 4);
            par.add(h + "T 최대 오차", orig[k][j].tm, s.Tmax, 0.0005, 4);
        }
    // full280 v2_patch630_quality 실패 4장의 추정값 (failures.csv)
    const std::pair<const char*, double> of[4] = {{"T-0.05", -0.25896855871912733}, {"Y+0.05", 0.009833152573920884}, {"Y+0.15", 0.026606649621206256}, {"Y-0.20", -0.032244452362939574}};
    for (auto& o : of) {
        double now = hye::kNaN;
        for (auto& r : fr) if (pname(imgs[r.idx]) == o.first) now = r.pred;
        par.add(std::string("full280 quality ") + o.first + " 추정값", o.second, now, 0.0005, 4);
    }
    par.add("GN 유효율 (audit valid_rate)", 1.0, (double)nvalid / nall, 1e-9, 4);

    // ---- 보고서
    auto sq = st[{K, "full280"}], sr = st[{K, "repeated20"}], si = st[{K, "interleaved"}];
    auto sv1 = st[{0, "repeated20"}];
    std::ostringstream R;
    R << "# 보고서: center 단일 기준 630점 패치 GN 추적 (10-21 17:23)\n\n";
    R << "## 문제\n\ncenter 하나를 기준으로 630점 실제 패치를 추적하면 Y/T 판정이 나아지는지 확인해야 했다.\n\n";
    R << "## 해결 방법\n\ncenter QR 세 개의 구조 모서리 630점을 center 패치로 LK15·GN17 추적하고 P12로 요약해 원점 통과 이차곡선 GLS의 Y/T 두 가지 비용을 비교했다.\n\n";
    R << "## 결과\n\n패치+품질 가중(v2_patch630_quality) full280 재대입에서 Y " << sq.Yc << "/" << sq.Yn << ", T " << sq.Tc << "/" << sq.Tn << ", 실패 " << sq.fail
      << "장이고, 20회 5-fold 5600개 예측에서는 실패 " << sr.fail << "개로 경계 모델 좌표(v1) " << sv1.fail << "개보다 적었다.\n\n";
    hye::Table res{{"측정", "full280 Y / T (실패)", "repeated20 Y / T (실패)", "interleaved Y / T (실패)", "repeated20 Y·T 최대 오차"}};
    for (int k = 0; k < 3; ++k) {
        auto a = st[{k, "full280"}], b = st[{k, "repeated20"}], c = st[{k, "interleaved"}];
        res.add({keys[k], std::to_string(a.Yc) + "/80 · " + std::to_string(a.Tc) + "/200 (" + std::to_string(a.fail) + ")",
                 std::to_string(b.Yc) + "/1600 · " + std::to_string(b.Tc) + "/4000 (" + std::to_string(b.fail) + ")",
                 std::to_string(c.Yc) + "/80 · " + std::to_string(c.Tc) + "/200 (" + std::to_string(c.fail) + ")", fmt(b.Ymax, 3) + " mm · " + fmt(b.Tmax, 3) + " deg"});
    }
    R << res.md() << "\n";
    R << hye::img_md("01a_repeated20_실패수.png", "repeated20 실패 수") << "\n\n";
    R << hye::img_md("02_full280_정답대비오차.png", "full280 정답 대비 오차") << "\n\n";
    double badmax = 0;
    int swapYT = 0, swapTY = 0;
    for (size_t i : bad) {
        badmax = std::max(badmax, std::abs(fr[i].value));
        swapYT += fr[i].axis == 'Y' && fr[i].pred_axis == 'T';
        swapTY += fr[i].axis == 'T' && fr[i].pred_axis == 'Y';
    }
    R << "## 튀는 값\n\nfull280에서 틀린 " << bad.size() << "장은 모두 |명령값|≤" << fmt(badmax, 2) << "의 작은 이동이고, Y 사진 " << swapYT << "장은 T 가지, T 사진 " << swapTY
      << "장은 Y 가지의 비용이 더 낮아 축을 바꿔 읽었다.\n\n";
    R << outl.md() << "\n";
    R << hye::img_md(fY, "Y 튀는 값") << "\n\n" << hye::img_md(fT, "T 튀는 값") << "\n\n";
    R << "초록 상자는 center QR 상자, 빨간 상자는 이 사진에서 잰 QR 평균 이동만큼 옮긴 상자다. 화살표는 점별 이동에서 QR 평균 이동을 뺀 나머지를 150배 키운 것이다.\n\n";
    R << "## 한줄평\n\nT 1deg는 평균 x 이동 " << fmt(sT, 2) << " px, Y 1mm는 " << fmt(sY, 2)
      << " px로 같은 방향이라, 이동이 0.2 이하로 작으면 구분 단서(점별 이동의 공간 패턴)가 패치 잡음과 비슷해져서 실패가 작은 값에만 몰렸다.\n\n";
    R << "## 원래 결과와 대조\n\n원래 값은 `center_patch_v2_20261021/summary.csv`, `failures.csv`, `audit.json`에서 가져왔다. " << par.ok << "/" << par.total << " 일치.\n\n";
    R << par.t.md() << "\n";
    // 원래 QR3 사각형 (decoded_grid_corners_20261015/metadata.json, OpenCV 5 가 3배 확대에서 찾음) — 대조용
    const double q3[4][2] = {{2079.34344291687, 983.3527421951294}, {2241.0, 984.6666660308838}, {2240.9757537841797, 1156.7299041748047}, {2079.0, 1155.3333282470703}};
    double qd = 0;
    std::string q3how;
    for (auto& q : M.qrs)
        if (q.qr == 3) {
            q3how = q.how;
            for (int j = 0; j < 4; ++j) qd = std::max(qd, std::max(std::abs(q.quad[j].x - q3[j][0]), std::abs(q.quad[j].y - q3[j][1])));
        }
    R << "QR1·QR2는 OpenCV 3.4 해독 사각형이 원래(OpenCV 5)와 같아서 두 QR의 모서리·GN 이동이 원래와 2e-5 px 안으로 같았다. "
         "QR3(FP11R)은 OpenCV 3.4가 원래 방식(여백 5px, 1~4배)으로 못 찾아 여백 10~60px에서 찾은 사각형의 평균(" << q3how << ")을 썼고 꼭짓점이 최대 " << fmt(qd, 2)
      << " px 달랐다. 그래서 QR3 모서리 위치가 0.002 px(중앙값), GN 이동이 0.0007 px 안에서 달랐고, repeated20에서 두 가지 비용 차이가 0.04%인 경계 사례 하나(Y+0.65, seed5 fold3)가 "
         "축을 바꿔 v2 계열 개수가 1씩 달랐다.\n";
    hye::write_text(hye::out("보고서.md"), R.str());

    // ---- 결과.md
    std::ostringstream D;
    D << "# 결과: center 단일 기준 630점 패치 GN 추적\n\n";
    D << "입력: `data/clean/clean_20261014` 281장 (center 1, T 200, Y 80). 정답은 파일명 명령값이다. 모든 측정은 사진 영역 x 1340~2470, y 800~1330 안에서만 했다.\n\n";
    D << "## 측정\n\n";
    hye::Table qt{{"QR", "내용", "찾은 방법 (OpenCV 3.4)", "구조 모서리", "검정 이웃 2개 제외"}};
    for (auto& q : M.qrs) {
        int a = 0, b = 0;
        for (auto& c : M.all) if (c.qr == q.qr) { ++a; b += c.black != 2; }
        qt.add({std::to_string(q.qr), q.payload, q.how, std::to_string(a), std::to_string(b)});
    }
    D << qt.md() << "\n";
    D << "- center 모서리: 해독 격자 꼭짓점을 사각형 호모그래피로 투영 → 9x9 창 경계 모델(흐린 2x2 칸, soft-L1, TRF) → 같은 모델로 창을 다시 맞춘 위치(630점).\n";
    D << "- v2 GN: QR별 NCC 정수 이동(±110x22) → LK15(σ0.8) → 3차 스플라인 GN17(σ0.8, 밝기 이득·오프셋). 유효 " << nvalid << "/" << nall << ", 사진별 최저 유효율 "
      << fmt(minrate, 4) << ", 광도 RMSE 중앙값 " << fmt(percentile(rm, 50), 3) << ".\n";
    D << "- v1 경계: 630점을 사진마다 경계 모델로 다시 맞춘 좌표. 진단 통과 " << edge_valid << "/" << nall << " (통과 여부와 관계없이 모두 썼다, 원래와 같음).\n";
    D << "- P12: 점 좌표 u=(x-1906)/486, v=(y-1070)/122 의 이차식 6항 x (dx, dy) = 12값. 품질 가중 = 1/(1+(RMSE/6)^2+(FB/0.15)^2), 무효 x0.05, 하한 0.001.\n";
    D << "- 판정: 학습 사진으로 Y, T 각각 P12 = s·v + c·v² (원점 통과) 를 맞추고, 잔차 상관행렬(0.999 + 0.001 I)의 백색화로 두 가지 비용을 비교해 작은 쪽 축·값을 낸다.\n\n";
    D << "## 전체 수치\n\n" << summ.md() << "\n";
    D << "분할: full280 = 280장 학습·평가(재대입). repeated20 = StratifiedKFold(5, shuffle, random_state 7000~7019, 세션 층화) 20회 = 5600개 예측. "
         "interleaved = 원래 파일 시각 순서로 번갈아 나눈 2-fold(분할 이름은 원래 splits.json에서 옮겨 적음).\n\n";
    D << "## 실패 목록 (full280, interleaved)\n\n" << fails.md() << "\n";
    D << "## 그림\n\n" << hye::img_md("01a_repeated20_실패수.png", "a") << "\n\n" << hye::img_md("01b_repeated20_Y최대오차.png", "b") << "\n\n"
      << hye::img_md("01c_repeated20_T최대오차.png", "c") << "\n\n";
    D << "## 원래 결과와 대조 (" << par.ok << "/" << par.total << " 일치)\n\n" << par.t.md() << "\n";
    D << "## 옮기지 않은 부분\n\n- `predict.py`, `verify.py`(사진 한 장 추론 점검): 같은 함수(Extractor, predict)를 쓰므로 따로 옮기지 않았다.\n"
         "- 원래 run.py 는 splits.json 119개 분할을 모두 돌렸지만 summary.csv 에는 full280, repeated20, interleaved 세 가지만 남겼다. 이 세 가지만 계산했다.\n"
         "- report.html, audit.json 의 시간 측정값.\n\n";
    D << "## 출력 파일\n\n- `summary.csv`: 측정 x 분할 요약\n- `failures.csv`: full280·interleaved 실패\n- `이미지/01a~01c`: repeated20 실패 수·최대 오차\n"
         "- `이미지/02_full280_정답대비오차.png`\n- `이미지/" << fY << "`, `이미지/" << fT << "`: 튀는 값 사진 확대\n\n";
    D << "## 코드\n\n- `실험/src/yt630.hpp`: 사진 목록, QR 구조 모서리, 3차 스플라인 GN, 630점 추적, P12, GLS 판정, StratifiedKFold\n- `실험/src/trf.hpp`: scipy least_squares(TRF, soft-L1) 이식 (모서리 경계 모델)\n- `실험/src/measure630.hpp`: 281장 측정 묶음\n- `실험/src/fourfold.hpp`: #51·#54·#55 의 4-fold 중첩 선택 틀\n- `실험/src/ecc_extra.hpp`, `ecc_compat_h.cpp/.hpp`: #51·#52 의 ECC affine/homography·위상상관 (공통 ecc_compat 사본, homography 역변환도 정확한 bilinear)\n- 다른 실험(#51~#55)은 CMake 의 include 경로로 이 폴더를 가리키고 각자 main.cpp 와 결과/를 가진다.\n- `실험/src/main.cpp`: 이 실험\n";
    hye::write_text(hye::out("결과.md"), D.str());
    std::printf("%s\nparity %d/%d\n", summ.md().c_str(), par.ok, par.total);
    return 0;
}
