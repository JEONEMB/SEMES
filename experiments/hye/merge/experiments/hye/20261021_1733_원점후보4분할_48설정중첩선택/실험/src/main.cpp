// 10-21 17:33 원점 후보 4-fold, 48설정 중첩 선택 (codex_experiments_20261014/multiref_4fold_20261021/experiment.py, measure_extra.py 의 C++ 재현)
//  기본: 측정 3종(v1 경계, v2 패치, v2 패치+품질) x 원점 {center 하나, 세션별 원점 후보} x {선형, 이차} x 공분산 축소 4종 = 48설정
//  구조영역(--regions): v2 이동을 finder / data / interior 점으로 나눠 P12
//  확장(--extended): ECC affine, ECC homography, 위상상관 3종 추가 (96설정)
// 측정 코드는 #50 실험 폴더(20261021_1723_센터단일기준_630패치GN추적/실험/src)의 것을 쓴다.
#include "measure630.hpp"
#include "ecc_extra.hpp"
#include "fourfold.hpp"
#include <hye/plot.hpp>

using namespace yt;
using hye::fmt;

static std::string pname(const Img& m) { return m.file.substr(0, m.file.size() - 4); }
static std::string sel_list(const FoldEval& E) {
    std::string s;
    for (size_t i = 0; i < E.selected.size(); ++i) s += (i ? ", " : "") + cfg_name(E.selected[i]);
    return s;
}
static std::string yt_str(const Stats& s) { return std::to_string(s.Yc) + "/" + std::to_string(s.Yn) + " · " + std::to_string(s.Tc) + "/" + std::to_string(s.Tn); }

static hye::Table summary_table(const FoldEval& E) {
    hye::Table t{{"설정", "측정 / 원점 / 곡선 / 축소", "OOF Y·T", "OOF 실패", "OOF Y·T 최대", "재대입 Y·T", "재대입 실패"}};
    for (size_t k = 0; k < E.cfgs.size(); ++k) {
        auto a = stats(E.oof[k]), b = stats(E.resub[k]);
        t.add({cfg_name((int)k), cfg_desc(E, (int)k), yt_str(a), std::to_string(a.fail), fmt(a.Ymax, 3) + " / " + fmt(a.Tmax, 3), yt_str(b), std::to_string(b.fail)});
    }
    auto n = stats(E.nested);
    t.add({"nested_selected", "fold별 " + sel_list(E), yt_str(n), std::to_string(n.fail), fmt(n.Ymax, 3) + " / " + fmt(n.Tmax, 3), "-", "-"});
    return t;
}
static void summary_csv(const FoldEval& E, const std::string& file) {
    hye::Table t{{"config", "protocol", "feature", "mode", "quadratic", "shrinkage", "n", "Y_correct", "Y_n", "T_correct", "T_n", "failures", "Y_max", "T_max"}};
    auto add = [&](const std::string& c, const std::string& p, const Stats& s, int k) {
        std::vector<std::string> r{c, p};
        if (k >= 0) {
            const Cfg& g = E.cfgs[k];
            char sh[16]; std::snprintf(sh, sizeof sh, "%g", g.shrink);
            r.insert(r.end(), {E.fnames[g.feat], g.sess ? "inferred_session" : "center", g.quad ? "True" : "False", sh});
        } else r.insert(r.end(), {"", "", "", ""});
        r.insert(r.end(), {std::to_string(s.n), std::to_string(s.Yc), std::to_string(s.Yn), std::to_string(s.Tc), std::to_string(s.Tn), std::to_string(s.fail), fmt(s.Ymax, 6), fmt(s.Tmax, 6)});
        t.add(r);
    };
    for (size_t k = 0; k < E.cfgs.size(); ++k) { add(cfg_name((int)k), "oof4", stats(E.oof[k]), (int)k); add(cfg_name((int)k), "resub280", stats(E.resub[k]), (int)k); }
    add("nested_selected", "oof4", stats(E.nested), -1);
    t.csv(hye::out(file));
}

int main(int argc, char** argv) {
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
    std::vector<std::string> base_names{"v1_edge630", "v2_patch630", "v2_patch630_quality"};
    std::vector<Feat> base{spatial(M.X, M.Y, Ex, Ey), spatial(M.X, M.Y, Dx, Dy), spatial(M.X, M.Y, Dx, Dy, &Wq)};
    std::printf("features ready\n");

    // ---- 기본 48설정
    FoldEval B = four_fold(imgs, base_names, base);
    std::printf("base nested fail %d\n", stats(B.nested).fail);

    // ---- 구조영역 (finder / data / interior)
    std::vector<std::string> reg_names{"finder", "data", "interior"};
    std::vector<Feat> reg;
    std::vector<int> reg_count;
    for (int r = 0; r < 3; ++r) {
        std::vector<int> sel;
        for (int p = 0; p < P; ++p) {
            int gx = M.gxs[p], gy = M.gys[p];
            bool finder = (gx <= 7 && gy <= 7) || (gx >= 14 && gy <= 7) || (gx <= 7 && gy >= 14);
            bool interior = gx > 0 && gx < 21 && gy > 0 && gy < 21;
            if ((r == 0 && finder) || (r == 1 && !finder) || (r == 2 && interior)) sel.push_back(p);
        }
        reg_count.push_back((int)sel.size());
        std::vector<double> x, y;
        std::vector<std::vector<double>> dx(N), dy(N);
        for (int p : sel) { x.push_back(M.X[p]); y.push_back(M.Y[p]); }
        for (int i = 0; i < N; ++i) for (int p : sel) { dx[i].push_back(Dx[i][p]); dy[i].push_back(Dy[i][p]); }
        reg.push_back(spatial(x, y, dx, dy));
    }
    FoldEval G = four_fold(imgs, reg_names, reg);
    std::printf("regions nested fail %d\n", stats(G.nested).fail);

    // ---- 확장: ECC affine / homography / 위상상관
    auto EX = measure_extra(ex, M.crops, imgs);
    std::vector<std::string> ext_names = base_names;
    std::vector<Feat> ext = base;
    const char* en[3] = {"ecc_affine", "ecc_homography", "phase"};
    for (int t = 0; t < 3; ++t) { ext_names.push_back(en[t]); ext.push_back(spatial(M.X, M.Y, EX.Dx[t], EX.Dy[t])); }
    FoldEval X = four_fold(imgs, ext_names, ext);
    std::printf("extended nested fail %d\n", stats(X.nested).fail);
    {
        std::vector<std::vector<double>> rows;
        for (int t = 3; t < 6; ++t) for (int i = 0; i < N; ++i) { std::vector<double> r{(double)t}; for (double v : ext[t][i]) r.push_back(v); rows.push_back(r); }
        dump_csv("e51_extra_features.csv", rows);
    }

    summary_csv(B, "summary.csv");
    summary_csv(G, "summary_structural_regions.csv");
    summary_csv(X, "summary_extended.csv");

    // ---- 그림
    auto bars = [&](const FoldEval& E, const std::string& file, const std::string& title) {
        std::vector<std::string> cats;
        std::vector<double> v;
        int best = 0;
        for (size_t k = 0; k < E.cfgs.size(); ++k) {
            cats.push_back(cfg_name((int)k));
            v.push_back(stats(E.oof[k]).fail);
            if (v.back() < v[best]) best = (int)k;
        }
        hye::bar_chart(file, title, cats, v, "OOF failures / 280", best);
    };
    bars(B, "01_기본48설정_OOF실패.png", "Base 48 settings: OOF failures (red = fewest)");
    bars(G, "02_구조영역48설정_OOF실패.png", "Structural regions 48 settings: OOF failures");
    bars(X, "03_확장96설정_OOF실패.png", "Extended 96 settings: OOF failures");
    auto pv = [&](const std::vector<Row>& rs, const std::string& file, const std::string& title) {
        std::vector<double> ty, py, tt, pt;
        std::vector<std::string> nm;
        for (auto& r : rs) {
            const Img& g = imgs[r.idx];
            ty.push_back(g.axis == 'Y' ? g.value : 0); tt.push_back(g.axis == 'T' ? g.value : 0);
            py.push_back(r.pred_axis == 'Y' ? r.pred : 0); pt.push_back(r.pred_axis == 'T' ? r.pred : 0);
            nm.push_back(pname(g));
        }
        hye::pred_vs_truth(file, title, ty, py, tt, pt, nm, 4);
    };
    pv(B.nested, "04_기본_중첩선택_정답대비.png", "Base nested OOF");
    int c45 = 45;
    pv(G.oof[c45], "05_구조영역_c45_정답대비.png", "Interior c45 OOF");

    // ---- 튀는 값 (기본 중첩 선택)
    auto& nr = B.nested;
    size_t wy = 0, wt = 0;
    pick_outliers(nr, wy, wt);
    hye::Table outl{{"구분", "사진", "fold 설정", "정답", "추정 (가지)", "eY", "eT", "c45 interior 추정", "평균 x 이동 (px)"}};
    std::map<int, Row> c45row;
    for (auto& r : G.oof[c45]) c45row[r.idx] = r;
    std::map<int, int> foldof;
    for (int f = 0; f < 4; ++f) for (int i : B.fold_test[f]) foldof[i] = f;
    int nbad = 0;
    for (size_t i = 0; i < nr.size(); ++i) {
        auto& r = nr[i];
        if (r.pass && i != wy && i != wt) continue;
        nbad += !r.pass;
        const Img& g = imgs[r.idx];
        double mx = 0;
        for (int p = 0; p < P; ++p) mx += Dx[r.idx][p] / P;
        auto& c = c45row[r.idx];
        outl.add({!r.pass ? "실패" : "최대 오차", pname(g), cfg_name(B.selected[foldof[r.idx]]), std::string(1, g.axis) + " " + hye::fmts(g.value, 2),
                  std::string(1, r.pred_axis) + " " + hye::fmts(r.pred, 4) + " (" + r.branch + ")", fmt(r.eY, 4), fmt(r.eT, 4),
                  std::string(1, c.pred_axis) + " " + hye::fmts(c.pred, 4), fmt(mx, 3)});
    }
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
            for (int p = 0; p < P; ++p) {
                if (ex.region[p] != k) continue;
                int gx = M.gxs[p], gy = M.gys[p];
                bool interior = gx > 0 && gx < 21 && gy > 0 && gy < 21;
                mk.push_back({{M.X[p] - CROP[0] + tk.Dx[idx][p], M.Y[p] - CROP[1] + tk.Dy[idx][p]}, {tk.Dx[idx][p] - kx, tk.Dy[idx][p] - ky}, "", interior ? hye::kRed : hye::kGray});
            }
            std::vector<hye::Box> bx{{cv::Rect2d(BOXES[k][0] - CROP[0], BOXES[k][1] - CROP[1], BOXES[k][2] - BOXES[k][0], BOXES[k][3] - BOXES[k][1]), hye::kGreen, "center"},
                                     {cv::Rect2d(BOXES[k][0] - CROP[0] + kx, BOXES[k][1] - CROP[1] + ky, BOXES[k][2] - BOXES[k][0], BOXES[k][3] - BOXES[k][1]), hye::kRed, "measured"}};
            char cap[96];
            std::snprintf(cap, sizeof cap, "QR%d shift %.2f, %.2f px", k + 1, kx, ky);
            panels.push_back(hye::zoom(M.crops[idx], roi, 420, mk, 150, cap, bx));
        }
        cv::Mat all = hye::hstack(panels);
        char cap[200];
        const Img& g = imgs[idx];
        std::snprintf(cap, sizeof cap, "%s  true %c %+.2f  pred %c %+.4f  (red = interior points, gray = border; arrows = shift - QR mean, x150)", pname(g).c_str(), g.axis, g.value,
                      r.pred_axis, r.pred);
        cv::Mat head(34, all.cols, CV_8UC3, cv::Scalar(255, 255, 255));
        hye::text(head, cap, {8, 23}, 0.55);
        hye::save_png(file, hye::vstack({head, all}, 0));
    };
    std::string fY = "06_튀는값_Y_" + pname(imgs[nr[wy].idx]) + ".png", fT = "07_튀는값_T_" + pname(imgs[nr[wt].idx]) + ".png";
    outlier_png(nr[wy], fY);
    outlier_png(nr[wt], fT);

    // ---- 대조 (원래 값: multiref_4fold_20261021/summary.csv, nested_selection.json, structural_regions/, extended/)
    hye::Parity par;
    const int oB[48] = {5, 5, 5, 8, 5, 5, 5, 9, 5, 5, 3, 7, 4, 4, 4, 5, 5, 6, 5, 9, 5, 5, 6, 10, 2, 2, 2, 2, 1, 1, 1, 1, 5, 6, 6, 9, 5, 5, 6, 10, 2, 2, 2, 2, 1, 1, 1, 1};
    const int oBr[48] = {5, 5, 5, 8, 5, 5, 5, 8, 3, 3, 4, 5, 1, 1, 1, 1, 4, 5, 5, 9, 4, 4, 5, 10, 1, 2, 2, 3, 1, 1, 1, 1, 4, 5, 5, 9, 4, 4, 5, 10, 1, 2, 2, 3, 1, 1, 1, 1};
    const int oG[48] = {7, 7, 8, 11, 7, 7, 7, 11, 5, 5, 5, 4, 5, 5, 5, 4, 7, 7, 6, 8, 6, 5, 6, 8, 3, 3, 3, 5, 1, 1, 1, 1, 6, 6, 6, 9, 5, 6, 6, 10, 1, 1, 1, 5, 0, 0, 1, 2};
    const int oX[48] = {5, 5, 4, 5, 5, 5, 5, 5, 3, 4, 4, 4, 3, 3, 2, 1, 5, 5, 5, 6, 5, 5, 5, 6, 3, 3, 3, 7, 2, 2, 2, 2, 26, 27, 24, 24, 26, 26, 24, 25, 12, 12, 12, 22, 13, 12, 13, 22};
    auto nb = stats(B.nested), ng = stats(G.nested), nx = stats(X.nested), s45 = stats(G.oof[c45]);
    par.add("기본 중첩 OOF Y 맞음", 78, nb.Yc, 0, 0);
    par.add("기본 중첩 OOF T 맞음", 198, nb.Tc, 0, 0);
    par.add("기본 중첩 OOF 실패", 4, nb.fail, 0, 0);
    par.add("기본 중첩 OOF Y 최대 오차", 0.266156239, nb.Ymax, 0.0005, 4);
    par.add_text("기본 fold별 선택", "c31, c24, c20, c46", sel_list(B), sel_list(B) == "c31, c24, c20, c46");
    int okB = 0, okBr = 0, okG = 0, okX = 0;
    for (int k = 0; k < 48; ++k) {
        okB += stats(B.oof[k]).fail == oB[k];
        okBr += stats(B.resub[k]).fail == oBr[k];
        okG += stats(G.oof[k]).fail == oG[k];
        okX += stats(X.oof[48 + k]).fail == oX[k];
    }
    par.add("기본 48설정 OOF 실패 수 같은 설정", 48, okB, 0, 0);
    par.add("기본 48설정 재대입 실패 수 같은 설정", 48, okBr, 0, 0);
    par.add("구조영역 중첩 OOF Y 맞음", 80, ng.Yc, 0, 0);
    par.add("구조영역 중첩 OOF T 맞음", 199, ng.Tc, 0, 0);
    par.add("구조영역 중첩 OOF 실패", 1, ng.fail, 0, 0);
    par.add_text("구조영역 fold별 선택", "c28, c45, c46, c24", sel_list(G), sel_list(G) == "c28, c45, c46, c24");
    par.add("구조영역 c45(interior) OOF Y 맞음", 80, s45.Yc, 0, 0);
    par.add("구조영역 c45(interior) OOF T 맞음", 200, s45.Tc, 0, 0);
    par.add("구조영역 c45 OOF Y 최대 오차", 0.036550, s45.Ymax, 0.0005, 4);
    par.add("구조영역 c45 OOF T 최대 오차", 0.002993, s45.Tmax, 0.0005, 4);
    par.add("구조영역 48설정 OOF 실패 수 같은 설정", 48, okG, 0, 0);
    par.add("구조영역 interior 점 수", 0, 0, 0, 0);
    par.t.rows.pop_back(); par.ok--; par.total--;
    par.add("확장 중첩 OOF Y 맞음", 80, nx.Yc, 0, 0);
    par.add("확장 중첩 OOF T 맞음", 196, nx.Tc, 0, 0);
    par.add("확장 중첩 OOF 실패", 4, nx.fail, 0, 0);
    par.add_text("확장 fold별 선택", "c31, c72, c61, c60", sel_list(X), sel_list(X) == "c31, c72, c61, c60");
    par.add("확장 ECC·위상 48설정 OOF 실패 수 같은 설정", 48, okX, 0, 0);
    par.add("ECC 실패로 대략 이동만 쓴 사진 (measurement_audit)", 0, (double)EX.fallbacks.size(), 0, 0);

    // ---- 보고서
    int bestB = 0;
    for (int k = 0; k < 48; ++k) if (stats(B.oof[k]).fail < stats(B.oof[bestB]).fail) bestB = k;
    std::ostringstream R;
    R << "# 보고서: 원점 후보 4-fold, 48설정 중첩 선택 (10-21 17:33)\n\n";
    R << "## 문제\n\n측정·원점·곡선 차수·공분산 축소를 고정 4-fold에서 공정하게 비교하고 중첩 선택 성적을 구해야 했다.\n\n";
    R << "## 해결 방법\n\n48설정을 4-fold OOF와 재대입으로 평가하고 바깥 학습 210장 안의 3-fold로 설정을 골랐으며, 구조영역(finder·data·interior)과 ECC·위상상관 측정을 같은 틀로 돌렸다.\n\n";
    R << "## 결과\n\n기본 48설정의 중첩 선택 OOF는 Y " << nb.Yc << "/80, T " << nb.Tc << "/200(실패 " << nb.fail << ")이었고, 구조영역 c45(interior, 세션 원점, 이차, 축소 0.05)는 OOF Y "
      << s45.Yc << "/80, T " << s45.Tc << "/200으로 다 맞았다.\n\n";
    hye::Table res{{"묶음", "설정 수", "중첩 선택 OOF Y·T (실패)", "fold별 선택", "OOF 실패 최소 설정"}};
    auto bestof = [&](const FoldEval& E) {
        int b = 0;
        for (size_t k = 0; k < E.cfgs.size(); ++k) if (stats(E.oof[k]).fail < stats(E.oof[b]).fail) b = (int)k;
        auto s = stats(E.oof[b]);
        return cfg_name(b) + " " + cfg_desc(E, b) + " : " + yt_str(s) + " (" + std::to_string(s.fail) + ")";
    };
    res.add({"기본 (v1, v2, v2+품질)", "48", yt_str(nb) + " (" + std::to_string(nb.fail) + ")", sel_list(B), bestof(B)});
    res.add({"구조영역 (finder " + std::to_string(reg_count[0]) + ", data " + std::to_string(reg_count[1]) + ", interior " + std::to_string(reg_count[2]) + "점)", "48",
             yt_str(ng) + " (" + std::to_string(ng.fail) + ")", sel_list(G), bestof(G)});
    res.add({"확장 (+ECC affine, ECC homography, 위상상관)", "96", yt_str(nx) + " (" + std::to_string(nx.fail) + ")", sel_list(X), bestof(X)});
    R << res.md() << "\n";
    R << hye::img_md("01_기본48설정_OOF실패.png", "기본 48설정") << "\n\n" << hye::img_md("02_구조영역48설정_OOF실패.png", "구조영역 48설정") << "\n\n"
      << hye::img_md("04_기본_중첩선택_정답대비.png", "기본 중첩 선택") << "\n\n";
    R << "## 튀는 값\n\n기본 중첩 선택에서 틀린 " << nbad << "장은 모두 |값|≤0.20의 작은 이동이고, 같은 사진을 interior 점만 쓴 c45는 모두 맞혔다.\n\n";
    R << outl.md() << "\n" << hye::img_md(fY, "Y 튀는 값") << "\n\n" << hye::img_md(fT, "T 튀는 값") << "\n\n";
    R << "빨간 점은 interior(격자 0·21 경계를 뺀) 점, 회색 점은 QR 바깥 경계의 점이다. 화살표는 점 이동에서 QR 평균 이동을 뺀 나머지를 150배 키운 것이다.\n\n";
    auto sb = stats(B.oof[bestB]);
    R << "## 한줄평\n\n세션 원점 후보는 시행마다 다른 0점 어긋남을 가지로 흡수해 v2 계열 OOF 실패를 " << stats(B.oof[20]).fail << "→" << sb.fail
      << "장으로 줄였지만, QR 가장자리 점을 섞으면 작은 이동에서 축이 바뀌었고 interior 점만 쓰면 실패가 " << s45.fail << "장이 됐다.\n\n";
    R << "## 원래 결과와 대조\n\n원래 값은 `multiref_4fold_20261021/summary.csv`, `nested_selection.json`, `structural_regions/`, `extended/`, `measurement_audit.json`에서 가져왔다. " << par.ok
      << "/" << par.total << " 일치.\n\n" << par.t.md() << "\n";
    R << "측정은 #50과 같다. QR3 해독 사각형만 OpenCV 3.4에서 다르게 찾아 v1 경계 좌표가 0.002 px(중앙값) 다르고, 패치 이동은 0.0007 px 안이다. "
         "이 차이로 기본 48설정 중 한 설정의 OOF 실패 수가 1장 달랐다. ECC affine 은 사진 2장의 한 QR에서 정지 반복이 달라 0.001 px 이상 달랐고, "
         "그 때문에 확장 묶음 넷째 fold의 안쪽 순위가 바뀌어 c60 대신 c62(같은 ECC affine·세션·이차, 축소만 다름)를 골랐지만 중첩 선택 성적은 같았다. "
         "ECC homography 는 공통 ecc_compat 이 OpenCV 3.4 warpPerspective(좌표 양자화)를 써서 cv2 5와 0.03 px까지 달랐기 때문에, #50 폴더의 사본(ecc_compat_h.cpp)에서 투영 역변환도 정확한 bilinear 로 바꿔 썼다.\n";
    hye::write_text(hye::out("보고서.md"), R.str());

    std::ostringstream D;
    D << "# 결과: 원점 후보 4-fold, 48설정 중첩 선택\n\n";
    D << "입력: `data/clean/clean_20261014` 281장. 측정은 #50(`20261021_1723_센터단일기준_630패치GN추적`)과 같은 630점 코드를 다시 돌린 것이다.\n\n";
    D << "- 바깥 분할: StratifiedKFold(4, shuffle, random_state 20261021, 세션 층화), 시험 70장 x 4.\n- 안쪽 분할: 바깥 학습 210장에서 StratifiedKFold(3, random_state 100+fold). 순위 = (실패 수, max(Y 최대, T 최대), 설정 번호).\n"
         "- 원점 center: Y, T 두 가지. 원점 inferred_session: Y1+, Y1-, Y2+, Y2-, T+, T- 여섯 가지(학습 세션별 절편), 가지 부호와 다른 추정은 버린다.\n"
         "- 공분산: 잔차 상관 C 를 (1-s)C + sI 로 축소한 뒤 백색화.\n\n";
    D << "## 기본 48설정\n\n" << summary_table(B).md() << "\n";
    D << "## 구조영역 48설정 (finder " << reg_count[0] << ", data " << reg_count[1] << ", interior " << reg_count[2] << "점)\n\n" << summary_table(G).md() << "\n";
    D << "## 확장 96설정\n\n" << summary_table(X).md() << "\n";
    D << "ECC 실패로 대략 이동만 쓴 사진: " << EX.fallbacks.size() << "장.\n\n";
    D << hye::img_md("03_확장96설정_OOF실패.png", "확장") << "\n\n" << hye::img_md("05_구조영역_c45_정답대비.png", "c45") << "\n\n";
    D << "## 원래 결과와 대조 (" << par.ok << "/" << par.total << ")\n\n" << par.t.md() << "\n";
    D << "## 옮기지 않은 부분\n\n- `--local`(QR별 국소 1·2차식), `--subset`(center 고유값 간격 부분집합), `--spatial`(Huber 공간 적합), `--screws`(나사 점 추가) 하위 실험. 타임라인 결과(기본 중첩 선택, 구조영역 c45)에 쓰이지 않았다.\n"
         "- 설정별 모델 JSON, predictions.csv, report.html.\n\n";
    D << "## 출력 파일\n\n- `summary.csv`, `summary_structural_regions.csv`, `summary_extended.csv`: 설정별 OOF·재대입 (원래 summary.csv 와 같은 열)\n- `이미지/01~03`: 설정별 OOF 실패 막대\n"
         "- `이미지/04`, `05`: 정답 대비 오차\n- `이미지/06`, `07`: 튀는 값 사진 확대\n";
    hye::write_text(hye::out("결과.md"), D.str());
    std::printf("parity %d/%d  base %s  regions %s  ext %s\n", par.ok, par.total, sel_list(B).c_str(), sel_list(G).c_str(), sel_list(X).c_str());
    for (int k = 0; k < 48; ++k) std::printf("%d%s", stats(B.oof[k]).fail, k == 47 ? "\n" : ",");
    for (int k = 0; k < 48; ++k) std::printf("%d%s", stats(G.oof[k]).fail, k == 47 ? "\n" : ",");
    for (int k = 48; k < 96; ++k) std::printf("%d%s", stats(X.oof[k]).fail, k == 95 ? "\n" : ",");
    (void)argc; (void)argv;
    return 0;
}
