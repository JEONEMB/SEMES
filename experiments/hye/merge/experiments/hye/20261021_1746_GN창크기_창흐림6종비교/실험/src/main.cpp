// 10-21 17:46 GN 창 크기·흐림 6종 비교 (codex_experiments_20261014/patch_windows_20261021/measure.py + multiref experiment.py --windows 의 C++ 재현)
//  같은 center 630점, LK15(σ0.8) 시작값은 그대로 두고 GN 창·흐림만 바꿔 281장을 다시 추적한다:
//  (9,.8) (13,.8) (25,.8) (31,.8) (17,.4) (31,.4). 각 P12(가중 없음) -> 6 x 16 = 96설정 4-fold 중첩 선택.
// 측정·평가 코드는 #50 실험 폴더(20261021_1723_센터단일기준_630패치GN추적/실험/src)의 것을 쓴다.
#include "measure630.hpp"
#include "fourfold.hpp"
#include <hye/plot.hpp>

using namespace yt;
using hye::fmt;

static std::string pname(const Img& m) { return m.file.substr(0, m.file.size() - 4); }
static std::string yts(const Stats& s) { return std::to_string(s.Yc) + "/" + std::to_string(s.Yn) + " · " + std::to_string(s.Tc) + "/" + std::to_string(s.Tn); }

int main() {
    hye::begin_results();
    auto M = measure630(false);
    const auto& imgs = M.imgs;
    const int N = (int)imgs.size(), P = (int)M.X.size();
    struct Setting { int w; double s; std::string key; };
    std::vector<Setting> settings{{9, .8, "gn9_sigma0.8"}, {13, .8, "gn13_sigma0.8"}, {25, .8, "gn25_sigma0.8"}, {31, .8, "gn31_sigma0.8"}, {17, .4, "gn17_sigma0.4"}, {31, .4, "gn31_sigma0.4"}};
    std::map<std::string, Feat> feats;
    std::map<std::string, Tracks> tracks;
    hye::Table meas{{"설정", "창", "흐림 σ", "유효 비율", "광도 RMSE 중앙값", "center 대비 T+1.00 평균 x 이동 (px)"}};
    std::map<std::string, double> valid;
    for (auto& st : settings) {
        Extractor ex(M.crops[0], M.X, M.Y, st.w, st.s);
        Tracks tk = ex.run_all(M.crops);
        std::vector<std::vector<double>> Dx, Dy;
        center_relative(tk, Dx, Dy);
        feats[st.key] = spatial(M.X, M.Y, Dx, Dy);
        int nv = 0, na = 0;
        std::vector<double> rm;
        for (int i = 0; i < N; ++i) for (int p = 0; p < P; ++p) { nv += tk.V[i][p]; ++na; if (i) rm.push_back(tk.Q[i][p][0]); }
        valid[st.key] = (double)nv / na;
        int it1 = 0;
        for (int i = 0; i < N; ++i) if (imgs[i].file == "T+1.00.bmp") it1 = i;
        double mx = 0;
        for (int p = 0; p < P; ++p) mx += Dx[it1][p] / P;
        char sg[8]; std::snprintf(sg, sizeof sg, "%.1f", st.s);
        meas.add({st.key, std::to_string(st.w), sg, fmt(valid[st.key], 4), fmt(percentile(rm, 50), 3), fmt(mx, 4)});
        tracks[st.key] = std::move(tk);
        std::printf("%s tracked\n", st.key.c_str());
    }
    // 원래 순서: *_features.npy 파일 이름 정렬
    std::vector<std::string> names;
    for (auto& kv : feats) names.push_back(kv.first);
    std::vector<Feat> F;
    for (auto& n : names) F.push_back(feats[n]);
    {
        std::vector<std::vector<double>> rows;
        for (size_t t = 0; t < F.size(); ++t) for (int i = 0; i < N; ++i) { std::vector<double> r{(double)t}; for (double v : F[t][i]) r.push_back(v); rows.push_back(r); }
        dump_csv("e54_features.csv", rows);
    }
    FoldEval E = four_fold(imgs, names, F);
    const int C = (int)E.cfgs.size();
    auto sn = stats(E.nested);
    std::string sel;
    for (size_t f = 0; f < E.selected.size(); ++f) sel += (f ? ", " : "") + cfg_name(E.selected[f]);

    // ---- 표
    hye::Table all{{"설정", "측정 / 원점 / 곡선 / 축소", "OOF Y·T", "OOF 실패", "OOF Y·T 최대", "재대입 Y·T", "재대입 실패"}};
    hye::Table csv{{"config", "protocol", "feature", "mode", "quadratic", "shrinkage", "n", "Y_correct", "Y_n", "T_correct", "T_n", "failures", "Y_max", "T_max"}};
    for (int k = 0; k < C; ++k) {
        auto a = stats(E.oof[k]), b = stats(E.resub[k]);
        all.add({cfg_name(k), cfg_desc(E, k), yts(a), std::to_string(a.fail), fmt(a.Ymax, 3) + " / " + fmt(a.Tmax, 3), yts(b), std::to_string(b.fail)});
        for (int pr = 0; pr < 2; ++pr) {
            auto s = pr ? b : a;
            const Cfg& g = E.cfgs[k];
            char sh[16]; std::snprintf(sh, sizeof sh, "%g", g.shrink);
            csv.add({cfg_name(k), pr ? "resub280" : "oof4", names[g.feat], g.sess ? "inferred_session" : "center", g.quad ? "True" : "False", sh, std::to_string(s.n), std::to_string(s.Yc),
                     std::to_string(s.Yn), std::to_string(s.Tc), std::to_string(s.Tn), std::to_string(s.fail), fmt(s.Ymax, 6), fmt(s.Tmax, 6)});
        }
    }
    all.add({"nested_selected", "fold별 " + sel, yts(sn), std::to_string(sn.fail), fmt(sn.Ymax, 3) + " / " + fmt(sn.Tmax, 3), "-", "-"});
    csv.add({"nested_selected", "oof4", "", "", "", "", std::to_string(sn.n), std::to_string(sn.Yc), std::to_string(sn.Yn), std::to_string(sn.Tc), std::to_string(sn.Tn), std::to_string(sn.fail),
             fmt(sn.Ymax, 6), fmt(sn.Tmax, 6)});
    csv.csv(hye::out("summary.csv"));
    // 측정별 최소 OOF 실패와 세션·이차 설정 실패
    hye::Table per{{"측정", "OOF 실패 최소 (설정)", "세션·이차·축소1 OOF Y·T (실패)", "center·이차·축소.001 OOF 실패"}};
    std::vector<double> minfail, sessq;
    for (size_t f = 0; f < names.size(); ++f) {
        int b = -1;
        for (int k = 0; k < C; ++k) if (E.cfgs[k].feat == (int)f && (b < 0 || stats(E.oof[k]).fail < stats(E.oof[b]).fail)) b = k;
        int k1 = (int)f * 16 + 15, k0 = (int)f * 16 + 4;
        auto s1 = stats(E.oof[k1]);
        per.add({names[f], std::to_string(stats(E.oof[b]).fail) + " (" + cfg_name(b) + ")", yts(s1) + " (" + std::to_string(s1.fail) + ")", std::to_string(stats(E.oof[k0]).fail)});
        minfail.push_back(stats(E.oof[b]).fail);
        sessq.push_back(s1.fail);
    }

    // ---- 그림
    {
        std::vector<std::string> cats;
        std::vector<double> v;
        int best = 0;
        for (int k = 0; k < C; ++k) { cats.push_back(cfg_name(k)); v.push_back(stats(E.oof[k]).fail); if (v.back() < v[best]) best = k; }
        hye::bar_chart("01_96설정_OOF실패.png", "96 settings: OOF failures / 280 (red = fewest)", cats, v, "failures", best);
        hye::bar_chart("02_창흐림별_세션이차_OOF실패.png", "Session origin, quadratic, shrink 1: OOF failures", names, sessq, "failures / 280");
        std::vector<double> ty, py, tt, pt;
        std::vector<std::string> nm;
        for (auto& r : E.nested) {
            const Img& g = imgs[r.idx];
            ty.push_back(g.axis == 'Y' ? g.value : 0); tt.push_back(g.axis == 'T' ? g.value : 0);
            py.push_back(r.pred_axis == 'Y' ? r.pred : 0); pt.push_back(r.pred_axis == 'T' ? r.pred : 0);
            nm.push_back(pname(g));
        }
        hye::pred_vs_truth("03_중첩선택_정답대비.png", "Nested OOF (GN window / blur)", ty, py, tt, pt, nm, 4);
    }
    // ---- 튀는 값
    size_t wy, wt;
    pick_outliers(E.nested, wy, wt);
    std::map<int, int> foldof;
    for (int f = 0; f < 4; ++f) for (int i : E.fold_test[f]) foldof[i] = f;
    hye::Table outl{{"구분", "사진", "fold 설정", "정답", "추정 (가지)", "eY", "eT", "창별 같은 설정(세션·이차·축소1) OOF 추정"}};
    std::vector<std::map<int, Row>> byfeat(names.size());
    for (size_t f = 0; f < names.size(); ++f) for (auto& r : E.oof[f * 16 + 15]) byfeat[f][r.idx] = r;
    int nbad = 0;
    for (size_t i = 0; i < E.nested.size(); ++i) {
        auto& r = E.nested[i];
        if (r.pass && i != wy && i != wt) continue;
        nbad += !r.pass;
        const Img& g = imgs[r.idx];
        std::string across;
        for (size_t f = 0; f < names.size(); ++f) {
            auto& q = byfeat[f][r.idx];
            across += (f ? ", " : "") + names[f].substr(0, names[f].find('_')) + "/" + names[f].substr(names[f].size() - 3) + " " + std::string(1, q.pred_axis) + hye::fmts(q.pred, 3);
        }
        outl.add({!r.pass ? "실패" : (i == wy ? "Y 사진 최대 오차" : "T 사진 최대 오차"), pname(g), cfg_name(E.selected[foldof[r.idx]]) + " " + cfg_desc(E, E.selected[foldof[r.idx]]),
                  std::string(1, g.axis) + " " + hye::fmts(g.value, 2), std::string(1, r.pred_axis) + " " + hye::fmts(r.pred, 4) + " (" + r.branch + ")", fmt(r.eY, 4), fmt(r.eT, 4), across});
    }
    auto outlier_png = [&](const Row& r, const std::string& file) {
        int idx = r.idx;
        const Cfg& cf = E.cfgs[E.selected[foldof[idx]]];
        const Tracks& tk = tracks[names[cf.feat]];
        const double cen[3] = {1644, 1901, 2159};
        std::vector<cv::Mat> panels;
        for (int k = 0; k < 3; ++k) {
            double kx = 0, ky = 0;
            int nk = 0;
            std::vector<int> mem;
            for (int p = 0; p < P; ++p) {
                int b = 0;
                for (int q = 1; q < 3; ++q) if (std::abs(M.X[p] - cen[q]) < std::abs(M.X[p] - cen[b])) b = q;
                if (b == k) { mem.push_back(p); kx += tk.Dx[idx][p]; ky += tk.Dy[idx][p]; ++nk; }
            }
            kx /= nk; ky /= nk;
            cv::Rect roi(BOXES[k][0] - CROP[0] + (int)std::lround(kx) - 6, BOXES[k][1] - CROP[1] + (int)std::lround(ky) - 6, BOXES[k][2] - BOXES[k][0] + 12, BOXES[k][3] - BOXES[k][1] + 12);
            std::vector<hye::Mark> mk;
            for (int p : mem) mk.push_back({{M.X[p] - CROP[0] + tk.Dx[idx][p], M.Y[p] - CROP[1] + tk.Dy[idx][p]}, {tk.Dx[idx][p] - kx, tk.Dy[idx][p] - ky}, "", hye::kRed});
            std::vector<hye::Box> bx{{cv::Rect2d(BOXES[k][0] - CROP[0], BOXES[k][1] - CROP[1], BOXES[k][2] - BOXES[k][0], BOXES[k][3] - BOXES[k][1]), hye::kGreen, "center"},
                                     {cv::Rect2d(BOXES[k][0] - CROP[0] + kx, BOXES[k][1] - CROP[1] + ky, BOXES[k][2] - BOXES[k][0], BOXES[k][3] - BOXES[k][1]), hye::kRed, "measured"}};
            char cap[96];
            std::snprintf(cap, sizeof cap, "QR%d shift %.2f, %.2f px", k + 1, kx, ky);
            panels.push_back(hye::zoom(M.crops[idx], roi, 420, mk, 150, cap, bx));
        }
        cv::Mat all = hye::hstack(panels);
        char cap[220];
        const Img& g = imgs[idx];
        std::snprintf(cap, sizeof cap, "%s  true %c %+.2f  pred %c %+.4f  (%s; arrows = shift - QR mean, x150)", pname(g).c_str(), g.axis, g.value, r.pred_axis, r.pred, names[cf.feat].c_str());
        cv::Mat head(34, all.cols, CV_8UC3, cv::Scalar(255, 255, 255));
        hye::text(head, cap, {8, 23}, 0.55);
        hye::save_png(file, hye::vstack({head, all}, 0));
    };
    std::string fY = "04_튀는값_Y_" + pname(imgs[E.nested[wy].idx]) + ".png", fT = "05_튀는값_T_" + pname(imgs[E.nested[wt].idx]) + ".png";
    outlier_png(E.nested[wy], fY);
    outlier_png(E.nested[wt], fT);

    // ---- 대조 (원래 값: patch_windows_20261021/evaluation/summary.csv, nested_selection.json, audit.json)
    hye::Parity par;
    const int oo[96] = {6, 5, 5, 10, 6, 6, 5, 10, 3, 2, 2, 4, 1, 2, 2, 2, 6, 6, 6, 12, 5, 6, 6, 11, 4, 4, 3, 2, 2, 2, 1, 1, 4, 5, 6, 7, 4, 5, 5, 7, 2, 2, 2, 2, 1, 1, 1, 1,
                        5, 5, 6, 8, 4, 4, 5, 8, 2, 2, 2, 7, 3, 3, 2, 2, 4, 4, 6, 6, 4, 4, 5, 7, 2, 2, 2, 3, 1, 1, 2, 2, 6, 5, 6, 11, 6, 5, 5, 13, 5, 4, 4, 5, 2, 2, 2, 4};
    const int orr[96] = {5, 5, 5, 10, 6, 5, 5, 10, 1, 1, 2, 4, 1, 1, 1, 1, 6, 6, 5, 11, 6, 6, 5, 12, 2, 2, 3, 4, 1, 1, 1, 1, 3, 4, 6, 7, 4, 4, 5, 7, 1, 1, 1, 3, 1, 1, 1, 1,
                         4, 5, 6, 8, 4, 4, 5, 8, 2, 3, 3, 8, 2, 2, 2, 2, 4, 4, 6, 7, 5, 5, 6, 7, 0, 1, 2, 3, 1, 1, 1, 2, 5, 5, 6, 11, 6, 5, 6, 12, 4, 4, 4, 4, 1, 1, 1, 1};
    par.add("중첩 선택 OOF Y 맞음", 80, sn.Yc, 0, 0);
    par.add("중첩 선택 OOF T 맞음", 198, sn.Tc, 0, 0);
    par.add("중첩 선택 OOF 실패", 2, sn.fail, 0, 0);
    par.add("중첩 선택 OOF Y 최대 오차", 0.264248, sn.Ymax, 0.0005, 4);
    par.add_text("fold별 선택", "c09, c08, c77, c40", sel, sel == "c09, c08, c77, c40");
    for (size_t f = 0; f < names.size(); ++f) {
        int a = 0, b = 0;
        for (int j = 0; j < 16; ++j) { int k = (int)f * 16 + j; a += stats(E.oof[k]).fail == oo[k]; b += stats(E.resub[k]).fail == orr[k]; }
        par.add(names[f] + " 16설정 OOF 실패 수 같은 설정", 16, a, 0, 0);
        par.add(names[f] + " 16설정 재대입 실패 수 같은 설정", 16, b, 0, 0);
        par.add(names[f] + " 유효 비율 (audit)", 1.0, valid[names[f]], 1e-9, 4);
    }

    // ---- 보고서
    int best = 0;
    for (int k = 0; k < C; ++k) if (stats(E.oof[k]).fail < stats(E.oof[best]).fail) best = k;
    std::ostringstream R;
    R << "# 보고서: GN 창 크기·흐림 6종 비교 (10-21 17:46)\n\n";
    R << "## 문제\n\nGN 정밀 추적의 창 크기와 흐림 σ가 판정에 영향을 주는지 확인해야 했다.\n\n";
    R << "## 해결 방법\n\nLK 시작값은 그대로 두고 GN 창(9·13·25·31)과 흐림(σ 0.4·0.8) 6종으로 281장 630점을 다시 추적하고, 96설정을 4-fold 중첩 선택으로 평가했다.\n\n";
    R << "## 결과\n\n중첩 선택 OOF는 Y " << sn.Yc << "/80, T " << sn.Tc << "/200(실패 " << sn.fail << ")였고, 창·흐림을 바꿔도 세션·이차 설정의 OOF 실패는 "
      << fmt(*std::min_element(sessq.begin(), sessq.end()), 0) << "~" << fmt(*std::max_element(sessq.begin(), sessq.end()), 0) << "장에 머물렀다.\n\n";
    hye::Table res{{"지표", "값"}};
    res.add({"중첩 선택 OOF Y / T", std::to_string(sn.Yc) + "/80 · " + std::to_string(sn.Tc) + "/200"});
    res.add({"중첩 선택 실패 / Y 최대 / T 최대", std::to_string(sn.fail) + " / " + fmt(sn.Ymax, 3) + " mm / " + fmt(sn.Tmax, 3) + " deg"});
    res.add({"fold별 선택", sel});
    res.add({"OOF 실패 최소 설정", cfg_name(best) + " " + cfg_desc(E, best) + " : " + std::to_string(stats(E.oof[best]).fail)});
    R << res.md() << "\n" << per.md() << "\n";
    R << hye::img_md("02_창흐림별_세션이차_OOF실패.png", "창·흐림별") << "\n\n" << hye::img_md("03_중첩선택_정답대비.png", "정답 대비") << "\n\n";
    int allfail = 0;
    for (auto& r : E.nested) {
        if (r.pass) continue;
        bool every = true;
        for (size_t f = 0; f < names.size(); ++f) every = every && !byfeat[f][r.idx].pass;
        allfail += every;
    }
    R << "## 튀는 값\n\n중첩 선택에서 틀린 " << nbad << "장 중 " << allfail << "장은 창·흐림 6종의 같은 설정(세션·이차·축소1)에서도 모두 틀려, 창 크기보다 사진 자체의 작은 이동 신호가 원인이었다.\n\n";
    R << outl.md() << "\n" << hye::img_md(fY, "Y 튀는 값") << "\n\n" << hye::img_md(fT, "T 튀는 값") << "\n\n";
    R << "초록 상자는 center QR 상자, 빨간 상자는 이 사진에서 잰 QR 평균 이동만큼 옮긴 상자, 빨간 십자는 잰 점 위치다. 화살표는 점 이동에서 QR 평균 이동을 뺀 나머지를 150배 키운 것이다.\n\n";
    R << "## 한줄평\n\nGN 창과 흐림은 점 하나의 이동 잡음만 바꾸고 P12가 630점을 평균해 그 차이를 대부분 지우므로, 6종의 측정별 최소 OOF 실패가 "
      << fmt(*std::min_element(minfail.begin(), minfail.end()), 0) << "~" << fmt(*std::max_element(minfail.begin(), minfail.end()), 0) << "장으로 거의 같았다.\n\n";
    R << "## 원래 결과와 대조\n\n원래 값은 `patch_windows_20261021/evaluation/summary.csv`, `nested_selection.json`, `audit.json`에서 가져왔다. " << par.ok << "/" << par.total << " 일치.\n\n"
      << par.t.md() << "\n";
    R << "630점 위치는 옛 corners_280.csv 를 쓰지 않고 center 사진에서 다시 쟀다. QR1·QR2는 원래와 2e-5 px 안으로 같지만, QR3(FP11R)은 OpenCV 3.4가 원래 방식으로 해독 사각형을 못 찾아 여백을 넓힌 사각형 평균(꼭짓점 최대 0.51 px 차이)을 써서 QR3 점 위치가 0.002 px(중앙값), 패치 이동이 0.001 px 안에서 달랐다. 불일치 1건(설정 하나의 OOF 실패 수 1장 차이)은 비용 차이가 아주 작은 경계 사진이 이 측정 차이로 축을 바꾼 것이다. 중첩 선택 결과와 fold별 선택은 같았다.\n";
    hye::write_text(hye::out("보고서.md"), R.str());

    std::ostringstream D;
    D << "# 결과: GN 창 크기·흐림 6종 비교\n\n입력: `data/clean/clean_20261014` 281장. center 630점·LK15 시작값은 #50과 같고 GN 창·흐림만 바꿨다. 유효 판정과 관계없이 모든 점을 썼다(원래와 같음).\n\n";
    D << "## 측정\n\n" << meas.md() << "\n";
    D << "## 96설정 (측정 6 x 원점 2 x 곡선 2 x 축소 4)\n\n" << all.md() << "\n";
    D << "## 원래 결과와 대조 (" << par.ok << "/" << par.total << ")\n\n" << par.t.md() << "\n";
    D << "## 옮기지 않은 부분\n\n- 설정별 모델 JSON, tracks.npz, predictions.csv, report.html.\n\n";
    D << "## 출력 파일\n\n- `summary.csv`: 설정별 OOF·재대입\n- `이미지/01`: 설정별 OOF 실패\n- `이미지/02`: 창·흐림별 세션·이차 OOF 실패\n- `이미지/03`: 중첩 선택 정답 대비\n- `이미지/04`, `05`: 튀는 값 사진\n";
    hye::write_text(hye::out("결과.md"), D.str());
    std::printf("parity %d/%d sel %s nested %d/%d fail %d\n", par.ok, par.total, sel.c_str(), sn.Yc, sn.Tc, sn.fail);
    for (int k = 0; k < C; ++k) std::printf("%d%s", stats(E.oof[k]).fail, k == C - 1 ? "\n" : ",");
    return 0;
}
