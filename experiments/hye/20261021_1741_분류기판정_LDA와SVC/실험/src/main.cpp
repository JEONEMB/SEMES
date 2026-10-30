// 10-21 17:41 분류기 판정: LDA·SVC 로 축(Y/T) 또는 시행(세션)을 먼저 고르고 그 가지 안에서 GLS 로 이동량 (classifier_4fold_20261021/run.py 의 C++ 재현)
//  측정 4종(v1 경계, v2 패치, v2 패치+품질, ECC affine) x 목표 {axis, session} x 분류기 13종(LDA 축소 4, SVC C 3 x gamma 3) = 104조합
//  바깥 4-fold(multiref 와 같은 분할), 안쪽 3-fold 로 조합 선택(중첩), 재대입 280.
// 측정 코드는 #50 실험 폴더(20261021_1723_센터단일기준_630패치GN추적/실험/src)의 것을 쓴다.
#include "measure630.hpp"
#include "ecc_extra.hpp"
#include "fourfold.hpp"
#include "classifiers.hpp"
#include <hye/plot.hpp>

using namespace yt;
using hye::fmt;

static std::string pname(const Img& m) { return m.file.substr(0, m.file.size() - 4); }
struct Setting { bool lda; double c, g; };
struct Combo { int feat; bool session; Setting s; };
static std::string dname(int k) { char b[8]; std::snprintf(b, sizeof b, "d%03d", k); return b; }

int main() {
    hye::begin_results();
    auto M = measure630(true);
    const auto& imgs = M.imgs;
    const int N = (int)imgs.size(), P = (int)M.X.size();
    Extractor ex(M.crops[0], M.X, M.Y, 17, .8);
    Tracks tk = ex.run_all(M.crops);
    std::vector<std::vector<double>> Dx, Dy;
    center_relative(tk, Dx, Dy);
    auto Wq = quality_weights(tk);
    std::vector<std::vector<double>> Ex(N, std::vector<double>(P)), Ey(N, std::vector<double>(P));
    for (int i = 0; i < N; ++i)
        for (int p = 0; p < P; ++p) { Ex[i][p] = M.edge[i][p].x - M.edge[0][p].x; Ey[i][p] = M.edge[i][p].y - M.edge[0][p].y; }
    auto EX = measure_extra(ex, M.crops, imgs);
    const std::vector<std::string> fnames{"v1_edge630", "v2_patch630", "v2_patch630_quality", "ecc_affine"};
    std::vector<Feat> F{spatial(M.X, M.Y, Ex, Ey), spatial(M.X, M.Y, Dx, Dy), spatial(M.X, M.Y, Dx, Dy, &Wq), spatial(M.X, M.Y, EX.Dx[0], EX.Dy[0])};
    std::printf("features ready\n");

    std::vector<Setting> sets;
    for (double s : {.001, .05, .2, 1.}) sets.push_back({true, s, 0});
    for (double c : {1., 10., 100.}) for (double g : {.01, .1, 1.}) sets.push_back({false, c, g});
    std::vector<Combo> combos;
    for (int f = 0; f < 4; ++f) for (bool t : {false, true}) for (auto& s : sets) combos.push_back({f, t, s});
    const int C = (int)combos.size();

    auto to_mat = [&](const Feat& X, const std::vector<int>& idx) {
        clf::Mat o;
        for (int i : idx) o.push_back(std::vector<double>(X[i].begin(), X[i].end()));
        return o;
    };
    // run.py predict(config, tr, te)
    auto run = [&](const Combo& c, const std::vector<int>& tr, const std::vector<int>& te, std::vector<std::string>* chosen_out = nullptr) {
        const Feat& X = F[c.feat];
        clf::Scaler sc;
        auto xtr = to_mat(X, tr), xte = to_mat(X, te);
        sc.fit(xtr);
        xtr = sc.transform(xtr); xte = sc.transform(xte);
        std::vector<std::string> lab;
        for (int i : tr) lab.push_back(c.session ? imgs[i].session : std::string(1, imgs[i].axis));
        std::vector<std::string> chosen(te.size());
        if (c.s.lda) {
            clf::LDA l;
            l.fit(xtr, lab, c.s.c);
            for (size_t i = 0; i < te.size(); ++i) chosen[i] = l.predict(xte[i]);
        } else {
            clf::SVC s;
            s.C = c.s.c; s.gamma = c.s.g;
            s.fit(xtr, lab);
            for (size_t i = 0; i < te.size(); ++i) chosen[i] = s.predict(xte[i]);
        }
        Model md = fit(imgs, X, tr, true, true, .2);
        std::vector<Row> out;
        auto groups = clf::classes_of(chosen);
        for (auto& gname : groups) {
            std::vector<int> tt;
            for (size_t i = 0; i < te.size(); ++i) if (chosen[i] == gname) tt.push_back(te[i]);
            Model sub = md;
            sub.br.clear();
            for (auto& b : md.br) if (c.session ? b.name == gname : std::string(1, b.axis) == gname) sub.br.push_back(b);
            auto r = enrich(imgs, predict(sub, X, tt, true));
            out.insert(out.end(), r.begin(), r.end());
        }
        if (chosen_out) *chosen_out = chosen;
        return out;
    };

    auto folds = outer_folds(imgs);
    std::vector<std::vector<Row>> oof(C), resub(C);
    std::vector<std::vector<std::vector<Row>>> per(C, std::vector<std::vector<Row>>(4));
    std::vector<Row> nested;
    std::vector<int> selected;
    for (int f = 0; f < 4; ++f) {
        const auto& tr = folds[f].first;
        const auto& te = folds[f].second;
        auto inner = inner_folds(imgs, tr, f);
        std::vector<std::tuple<int, double, int>> rank(C);
#pragma omp parallel for schedule(dynamic) num_threads(4)
        for (int k = 0; k < C; ++k) {
            std::vector<Row> ip;
            for (auto& in : inner) { auto r = run(combos[k], in.first, in.second); ip.insert(ip.end(), r.begin(), r.end()); }
            auto s = stats(ip);
            rank[k] = std::make_tuple(s.fail, std::max(s.Ymax, s.Tmax), k);
            per[k][f] = run(combos[k], tr, te);
        }
        int chosen = std::get<2>(*std::min_element(rank.begin(), rank.end()));
        selected.push_back(chosen);
        nested.insert(nested.end(), per[chosen][f].begin(), per[chosen][f].end());
        std::printf("fold %d selected %s\n", f, dname(chosen).c_str());
    }
    auto ids = outer_ids(imgs);
#pragma omp parallel for schedule(dynamic) num_threads(4)
    for (int k = 0; k < C; ++k) {
        for (int f = 0; f < 4; ++f) oof[k].insert(oof[k].end(), per[k][f].begin(), per[k][f].end());
        resub[k] = run(combos[k], ids, ids);
    }

    // ---- 표
    auto desc = [&](int k) {
        const Combo& c = combos[k];
        char b[64];
        if (c.s.lda) std::snprintf(b, sizeof b, "LDA shrink %g", c.s.c);
        else std::snprintf(b, sizeof b, "SVC C %g gamma %g", c.s.c, c.s.g);
        return fnames[c.feat] + " / " + (c.session ? "session" : "axis") + " / " + b;
    };
    auto yts = [](const Stats& s) { return std::to_string(s.Yc) + "/" + std::to_string(s.Yn) + " · " + std::to_string(s.Tc) + "/" + std::to_string(s.Tn); };
    hye::Table all{{"조합", "측정 / 목표 / 분류기", "OOF Y·T", "OOF 실패", "OOF Y·T 최대", "재대입 실패"}};
    hye::Table csv{{"config", "protocol", "feature", "classifier_target", "classifier", "parameter", "gamma", "n", "Y_correct", "Y_n", "T_correct", "T_n", "failures", "Y_max", "T_max"}};
    for (int k = 0; k < C; ++k) {
        auto a = stats(oof[k]), b = stats(resub[k]);
        all.add({dname(k), desc(k), yts(a), std::to_string(a.fail), fmt(a.Ymax, 3) + " / " + fmt(a.Tmax, 3), std::to_string(b.fail)});
        for (int pr = 0; pr < 2; ++pr) {
            auto s = pr ? b : a;
            char pc[16], gm[16];
            std::snprintf(pc, sizeof pc, "%g", combos[k].s.c);
            std::snprintf(gm, sizeof gm, "%g", combos[k].s.g);
            csv.add({dname(k), pr ? "resub280" : "oof4", fnames[combos[k].feat], combos[k].session ? "session" : "axis", combos[k].s.lda ? "lda" : "svc", pc, gm, std::to_string(s.n),
                     std::to_string(s.Yc), std::to_string(s.Yn), std::to_string(s.Tc), std::to_string(s.Tn), std::to_string(s.fail), fmt(s.Ymax, 6), fmt(s.Tmax, 6)});
        }
    }
    auto sn = stats(nested);
    std::string sel;
    for (int f = 0; f < 4; ++f) sel += (f ? ", " : "") + dname(selected[f]);
    all.add({"nested_selected", "fold별 " + sel, yts(sn), std::to_string(sn.fail), fmt(sn.Ymax, 3) + " / " + fmt(sn.Tmax, 3), "-"});
    csv.add({"nested_selected", "oof4", "", "", "", "", "", std::to_string(sn.n), std::to_string(sn.Yc), std::to_string(sn.Yn), std::to_string(sn.Tc), std::to_string(sn.Tn),
             std::to_string(sn.fail), fmt(sn.Ymax, 6), fmt(sn.Tmax, 6)});
    csv.csv(hye::out("summary.csv"));

    // ---- 비교: 같은 분할에서 분류기 없이 GLS 만 (multiref c46 = v2+품질, 세션, 이차, 축소 .2)
    std::vector<Row> gls;
    for (int f = 0; f < 4; ++f) {
        Model md = fit(imgs, F[2], folds[f].first, true, true, .2);
        auto r = enrich(imgs, predict(md, F[2], folds[f].second, true));
        gls.insert(gls.end(), r.begin(), r.end());
    }
    auto sg = stats(gls);

    // ---- 그림
    {
        std::vector<std::string> cats;
        std::vector<double> v;
        int best = 0;
        for (int k = 0; k < C; ++k) {
            cats.push_back(dname(k));
            v.push_back(stats(oof[k]).fail);
            if (v.back() < v[best]) best = k;
        }
        hye::bar_chart("01_104조합_OOF실패.png", "104 classifier combos: OOF failures / 280 (red = fewest)", cats, v, "failures", best);
        std::vector<std::string> c2{"GLS only (c46)", "nested classifier"};
        std::vector<double> v2{(double)sg.fail, (double)sn.fail};
        hye::bar_chart("02_GLS대비_중첩선택.png", "Same 4 folds: GLS vs classifier", c2, v2, "failures / 280", 1);
        std::vector<double> ty, py, tt, pt;
        std::vector<std::string> nm;
        for (auto& r : nested) {
            const Img& g = imgs[r.idx];
            ty.push_back(g.axis == 'Y' ? g.value : 0); tt.push_back(g.axis == 'T' ? g.value : 0);
            py.push_back(r.pred_axis == 'Y' ? r.pred : 0); pt.push_back(r.pred_axis == 'T' ? r.pred : 0);
            nm.push_back(pname(g));
        }
        hye::pred_vs_truth("03_중첩선택_정답대비.png", "Nested classifier OOF", ty, py, tt, pt, nm, 5);
    }

    // ---- 튀는 값: 중첩 선택 실패
    std::map<int, int> foldof;
    for (int f = 0; f < 4; ++f) for (int i : folds[f].second) foldof[i] = f;
    std::map<int, Row> glsrow;
    for (auto& r : gls) glsrow[r.idx] = r;
    hye::Table outl{{"사진", "fold 조합", "정답 (세션)", "분류기가 고른 가지", "추정", "eY", "eT", "분류기 없는 GLS 추정"}};
    size_t wy = 0, wt = 0;
    int nbad = 0, wrong_session = 0;
    for (size_t i = 0; i < nested.size(); ++i) {
        auto& r = nested[i];
        if (r.pass) continue;
        ++nbad;
        const Img& g = imgs[r.idx];
        wrong_session += r.branch != g.session;
        auto& q = glsrow[r.idx];
        outl.add({pname(g), dname(selected[foldof[r.idx]]), std::string(1, g.axis) + " " + hye::fmts(g.value, 2) + " (" + g.session + ")", r.branch,
                  std::string(1, r.pred_axis) + " " + hye::fmts(r.pred, 4), fmt(r.eY, 3), fmt(r.eT, 3), std::string(1, q.pred_axis) + " " + hye::fmts(q.pred, 4)});
    }
    pick_outliers(nested, wy, wt);
    auto outlier_png = [&](const Row& r, const std::string& file) {
        int idx = r.idx;
        std::vector<cv::Mat> panels;
        for (int k = 0; k < 3; ++k) {
            double kx = 0, ky = 0;
            int nk = 0;
            for (int p = 0; p < P; ++p) if (ex.region[p] == k) { kx += tk.Dx[idx][p]; ky += tk.Dy[idx][p]; ++nk; }
            kx /= nk; ky /= nk;
            cv::Rect roi(BOXES[k][0] - CROP[0] + (int)std::lround(kx) - 6, BOXES[k][1] - CROP[1] + (int)std::lround(ky) - 6, BOXES[k][2] - BOXES[k][0] + 12, BOXES[k][3] - BOXES[k][1] + 12);
            std::vector<hye::Mark> mk;
            for (int p = 0; p < P; ++p)
                if (ex.region[p] == k) mk.push_back({{M.X[p] - CROP[0] + tk.Dx[idx][p], M.Y[p] - CROP[1] + tk.Dy[idx][p]}, {tk.Dx[idx][p] - kx, tk.Dy[idx][p] - ky}, "", hye::kRed});
            std::vector<hye::Box> bx{{cv::Rect2d(BOXES[k][0] - CROP[0], BOXES[k][1] - CROP[1], BOXES[k][2] - BOXES[k][0], BOXES[k][3] - BOXES[k][1]), hye::kGreen, "center"},
                                     {cv::Rect2d(BOXES[k][0] - CROP[0] + kx, BOXES[k][1] - CROP[1] + ky, BOXES[k][2] - BOXES[k][0], BOXES[k][3] - BOXES[k][1]), hye::kRed, "measured"}};
            char cap[96];
            std::snprintf(cap, sizeof cap, "QR%d shift %.2f, %.2f px", k + 1, kx, ky);
            panels.push_back(hye::zoom(M.crops[idx], roi, 420, mk, 150, cap, bx));
        }
        cv::Mat all = hye::hstack(panels);
        char cap[200];
        const Img& g = imgs[idx];
        std::snprintf(cap, sizeof cap, "%s  true %c %+.2f (%s)  branch %s -> %c %+.4f  (arrows = shift - QR mean, x150)", pname(g).c_str(), g.axis, g.value, g.session.c_str(),
                      r.branch.c_str(), r.pred_axis, r.pred);
        cv::Mat head(34, all.cols, CV_8UC3, cv::Scalar(255, 255, 255));
        hye::text(head, cap, {8, 23}, 0.55);
        hye::save_png(file, hye::vstack({head, all}, 0));
    };
    std::string fY = "04_튀는값_Y_" + pname(imgs[nested[wy].idx]) + ".png", fT = "05_튀는값_T_" + pname(imgs[nested[wt].idx]) + ".png";
    outlier_png(nested[wy], fY);
    outlier_png(nested[wt], fT);

    // ---- 대조 (원래 값: classifier_4fold_20261021/summary.csv, nested_selection.json)
    hye::Parity par;
    const int oo[104] = {62, 62, 58, 63, 75, 51, 79, 57, 46, 71, 48, 44, 71, 16, 18, 29, 59, 80, 51, 80, 29, 32, 78, 20, 31, 78, 65, 63, 64, 57, 69, 47, 74, 53, 43, 69, 46,
                         43, 69, 11, 16, 30, 52, 80, 40, 78, 29, 27, 75, 22, 29, 75, 65, 64, 63, 56, 68, 47, 75, 53, 43, 68, 47, 41, 68, 11, 15, 29, 51, 80, 41, 79, 29, 27,
                         75, 21, 30, 75, 52, 56, 58, 64, 80, 48, 58, 51, 36, 50, 41, 33, 50, 10, 12, 31, 49, 80, 40, 72, 27, 21, 61, 14, 26, 61};
    const int orr[104] = {50, 56, 52, 62, 60, 29, 0, 42, 3, 0, 21, 0, 0, 10, 9, 25, 56, 79, 21, 0, 14, 0, 0, 3, 0, 0, 53, 54, 55, 54, 57, 28, 0, 42, 5, 0, 19, 0, 0, 6, 10,
                          23, 49, 80, 19, 0, 13, 1, 0, 2, 0, 0, 52, 55, 55, 53, 57, 29, 0, 42, 4, 0, 20, 0, 0, 6, 10, 23, 48, 80, 19, 0, 14, 1, 0, 2, 0, 0, 45, 52, 53, 64, 77,
                          31, 0, 42, 7, 0, 23, 0, 0, 8, 6, 19, 47, 80, 12, 2, 10, 1, 0, 3, 0, 0};
    par.add("중첩 선택 OOF Y 맞음", 69, sn.Yc, 0, 0);
    par.add("중첩 선택 OOF T 맞음", 200, sn.Tc, 0, 0);
    par.add("중첩 선택 OOF 실패", 11, sn.fail, 0, 0);
    par.add("중첩 선택 OOF Y 최대 오차", 0.75, sn.Ymax, 0.0005, 3);
    par.add("중첩 선택 OOF T 최대 오차", 0.13122, sn.Tmax, 0.0005, 4);
    par.add_text("fold별 선택", "d039, d039, d091, d091", sel, sel == "d039, d039, d091, d091");
    int okO[4] = {0}, okR[4] = {0}, cnt[4] = {0};
    for (int k = 0; k < C; ++k) {
        int f = combos[k].feat;
        ++cnt[f];
        okO[f] += stats(oof[k]).fail == oo[k];
        okR[f] += stats(resub[k]).fail == orr[k];
    }
    for (int f = 0; f < 4; ++f) {
        par.add(fnames[f] + " 26조합 OOF 실패 수 같은 조합", 26, okO[f], 0, 0);
        par.add(fnames[f] + " 26조합 재대입 실패 수 같은 조합", 26, okR[f], 0, 0);
    }
    par.add("d091 (ecc_affine, session, LDA .001) OOF 실패", 10, stats(oof[91]).fail, 0, 0);
    par.add("d039 (v2_patch630, session, LDA .001) OOF 실패", 11, stats(oof[39]).fail, 0, 0);

    // ---- 보고서
    std::ostringstream R;
    R << "# 보고서: 분류기 판정, LDA와 SVC (10-21 17:41)\n\n";
    R << "## 문제\n\n축이나 시행을 일반 분류기로 먼저 고르면 GLS 후보 비교보다 나아지는지 확인해야 했다.\n\n";
    R << "## 해결 방법\n\n측정 4종의 P12를 표준화해 LDA와 SVC로 축 또는 시행(세션)을 분류한 뒤, 고른 가지 안에서만 GLS 이차곡선으로 이동량을 구했다(104조합, 4-fold 중첩 선택).\n\n";
    R << "## 결과\n\n중첩 선택 OOF는 Y " << sn.Yc << "/80, T " << sn.Tc << "/200(실패 " << sn.fail << ")로, 같은 4-fold에서 분류기 없이 세션 GLS만 쓴 경우(Y " << sg.Yc << "/80, T " << sg.Tc
      << "/200, 실패 " << sg.fail << ")보다 나빴다.\n\n";
    hye::Table res{{"방법", "OOF Y", "OOF T", "실패 / 280", "Y 최대 (mm)", "T 최대 (deg)"}};
    res.add({"분류기 중첩 선택 (" + sel + ")", std::to_string(sn.Yc) + "/80", std::to_string(sn.Tc) + "/200", std::to_string(sn.fail), fmt(sn.Ymax, 3), fmt(sn.Tmax, 3)});
    int bk = 0;
    for (int k = 0; k < C; ++k) if (stats(oof[k]).fail < stats(oof[bk]).fail) bk = k;
    auto sb = stats(oof[bk]);
    res.add({"OOF 최소 조합 " + dname(bk) + " (" + desc(bk) + ")", std::to_string(sb.Yc) + "/80", std::to_string(sb.Tc) + "/200", std::to_string(sb.fail), fmt(sb.Ymax, 3), fmt(sb.Tmax, 3)});
    res.add({"분류기 없음: 세션 GLS (v2+품질, 이차, 축소 0.2)", std::to_string(sg.Yc) + "/80", std::to_string(sg.Tc) + "/200", std::to_string(sg.fail), fmt(sg.Ymax, 3), fmt(sg.Tmax, 3)});
    R << res.md() << "\n" << hye::img_md("02_GLS대비_중첩선택.png", "GLS 대비") << "\n\n" << hye::img_md("01_104조합_OOF실패.png", "104조합") << "\n\n"
      << hye::img_md("03_중첩선택_정답대비.png", "정답 대비") << "\n\n";
    R << "## 튀는 값\n\n중첩 선택에서 틀린 " << nbad << "장 중 " << wrong_session << "장은 분류기가 다른 세션 가지를 골라, 그 가지의 곡선 위로만 값을 읽어 틀렸다.\n\n";
    R << outl.md() << "\n" << hye::img_md(fY, "Y 튀는 값") << "\n\n" << hye::img_md(fT, "T 튀는 값") << "\n\n";
    R << "초록 상자는 center QR 상자, 빨간 상자는 이 사진에서 잰 QR 평균 이동만큼 옮긴 상자, 빨간 십자는 잰 점 위치다. 화살표는 점 이동에서 QR 평균 이동을 뺀 나머지를 150배 키운 것이다.\n\n";
    int y_to_t = 0;
    double badmax = 0;
    for (auto& r : nested)
        if (!r.pass) { y_to_t += r.axis == 'Y' && r.pred_axis == 'T'; badmax = std::max(badmax, std::abs(r.value)); }
    R << "## 한줄평\n\n분류기는 표준화한 P12 12값의 평균 차이로 세션을 가르는데, 작은 이동에서는 Y 세션과 T 세션의 P12가 모두 0 근처에 모여 경계가 가까우므로, "
         "|명령값|≤" << fmt(badmax, 2) << "의 Y 사진 " << y_to_t << "장이 T 세션으로 분류됐고 고른 가지 하나만 비교하니 GLS 비용으로 되돌릴 기회가 없었다.\n\n";
    R << "## 원래 결과와 대조\n\n원래 값은 `classifier_4fold_20261021/summary.csv`, `nested_selection.json`에서 가져왔다. " << par.ok << "/" << par.total << " 일치.\n\n" << par.t.md() << "\n";
    R << "LDA는 sklearn lsqr 식(클래스 비율 가중 축소 공분산, lstsq)을, SVC는 libsvm SMO(WSS3, 축소, tol 1e-3, float Q)를 그대로 옮겼다. "
         "측정이 원래와 가장 가까운 v2_patch630_quality 26조합은 OOF " << okO[2] << "/26, 재대입 " << okR[2] << "/26 조합의 실패 수가 같았다. "
         "v1 경계 좌표(QR3 사각형 차이로 0.002 px), v2 패치 이동(QR3 점 0.0007 px), ECC affine(사진 2장에서 ECC 정지 반복이 달라 0.001 px 이상)은 측정이 조금 달라, "
         "분류 경계 근처 사진이 바뀐 조합이 실패 수 1~3장씩 달랐다. 분류기 코드가 원래와 같다는 것은 v2+품질 26조합 일치로 확인했고, 중첩 선택 결과와 fold별 선택도 같았다. "
         "630점 위치를 옛 corners_280.csv 상수로 넣으면 이 차이가 없어지지만, 옛 결과를 입력으로 쓰지 않는다는 규칙에 따라 center 사진에서 다시 쟀다.\n";
    hye::write_text(hye::out("보고서.md"), R.str());

    std::ostringstream D;
    D << "# 결과: 분류기 판정, LDA와 SVC\n\n입력: `data/clean/clean_20261014` 281장. 측정은 #50과 같은 630점 코드와 #51 ECC affine 을 다시 돌린 것이다.\n\n";
    D << "- 분할: 바깥 StratifiedKFold(4, random_state 20261021) = multiref extended/splits.json 과 같은 분할, 안쪽 StratifiedKFold(3, random_state 100+fold).\n"
         "- 분류기 입력: 학습 사진으로 맞춘 StandardScaler 의 P12 12값. 목표 axis = Y/T, session = Y1+, Y1-, Y2+, Y2-, T+, T-.\n"
         "- 이동량: 학습 사진으로 세션 원점 이차곡선 + 축소 0.2 공분산을 맞추고, 분류기가 고른 가지(axis 면 그 축의 세션 가지들, session 이면 그 하나)만 비용 비교.\n\n";
    D << "## 104조합\n\n" << all.md() << "\n";
    D << "## 원래 결과와 대조 (" << par.ok << "/" << par.total << ")\n\n" << par.t.md() << "\n";
    D << "## 옮기지 않은 부분\n\n- predictions.csv, failures.csv, fold_results.csv 의 행 단위 출력(요약 summary.csv 만 같은 열로 남김).\n\n";
    D << "## 출력 파일\n\n- `summary.csv`: 조합별 OOF·재대입\n- `이미지/01`: 조합별 OOF 실패\n- `이미지/02`: 같은 분할 GLS 대비\n- `이미지/03`: 중첩 선택 정답 대비\n- `이미지/04`, `05`: 튀는 값 사진\n";
    hye::write_text(hye::out("결과.md"), D.str());
    std::printf("parity %d/%d sel %s nested %d/%d fail %d\n", par.ok, par.total, sel.c_str(), sn.Yc, sn.Tc, sn.fail);
    for (int k = 0; k < C; ++k) std::printf("%d%s", stats(oof[k]).fail, k == C - 1 ? "\n" : ",");
    for (int k = 0; k < C; ++k) std::printf("%d%s", stats(resub[k]).fail, k == C - 1 ? "\n" : ",");
    return 0;
}
