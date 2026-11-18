// 10-21 4-fold 실험군 공통 보고: 설정별 요약표, 내부 선택 OOF 그림, 튀는 값 사진 확대, 보고서.md / 결과.md 틀.
#pragma once
#include <hye/dataset.hpp>  // plot.hpp 가 kPass 를 쓴다
#include <hye/plot.hpp>
#include <hye/report.hpp>
#include <functional>
#include <sstream>
#include "g8_model.hpp"

namespace g8 {

inline std::string stem(const std::string& f) { return f.substr(0, f.rfind('.')); }
inline std::string esc(const std::string& s) {  // 마크다운 표 안의 |
    std::string o;
    for (char c : s) { if (c == '|') o += "\\|"; else o += c; }
    return o;
}
inline std::string cfg_name(char p, int k) {
    char b[16];
    std::snprintf(b, sizeof b, "%c%03d", p, k);
    return b;
}

// 사진 한 장의 세 QR 확대: 초록 = 기준 QR 상자, 빨강 = 평균 이동만큼 옮긴 상자, 십자 = 측정 위치, 화살표 = (점 이동 - QR 평균 이동) x gain
inline cv::Mat qr_zoom(const hye::Shot& shot, const Measured& M, int n, const std::string& caption, double gain = 400) {
    cv::Mat im = hye::read_color(shot);
    auto D = rel_D(M.tr.at("gn17"));
    std::vector<cv::Mat> tiles;
    for (int k = 0; k < 3; ++k) {
        double mx = 0, my = 0;
        int c = 0;
        for (int i = 0; i < kN630; ++i) if (kPoints630[i].qr == k + 1) { mx += D[n][i].x; my += D[n][i].y; ++c; }
        mx /= c; my /= c;
        cv::Rect2d ref(kBox[k][0], kBox[k][1], kBox[k][2] - kBox[k][0], kBox[k][3] - kBox[k][1]);
        cv::Rect2d now(ref.x + mx, ref.y + my, ref.width, ref.height);
        cv::Rect roi = cv::Rect(ref | now) + cv::Size(60, 60) - cv::Point(30, 30);
        std::vector<hye::Mark> marks;
        int j = 0;
        for (int i = 0; i < kN630; ++i) {
            if (kPoints630[i].qr != k + 1 || (j++ % 3)) continue;
            hye::Mark mk;
            mk.p = {kPoints630[i].x + D[n][i].x, kPoints630[i].y + D[n][i].y};
            mk.d = {D[n][i].x - mx, D[n][i].y - my};
            marks.push_back(mk);
        }
        tiles.push_back(hye::zoom(im, roi, 460, marks, gain, k == 0 ? caption : "",
                                  {{ref, hye::kGreen, ""}, {now, hye::kRed, "QR" + std::to_string(k + 1) + " dx " + hye::fmt(mx, 2) + " dy " + hye::fmt(my, 2)}}));
    }
    return hye::hstack(tiles);
}

inline double mean_dx(const Measured& M, int n) {
    auto& T = M.tr.at("gn17");
    double s = 0;
    for (int i = 0; i < kN630; ++i) s += (double)T[n].D[i].x - (double)T[0].D[i].x;
    return s / kN630;
}

inline int wrong_axis(const std::vector<Pred>& P) {
    int c = 0;
    for (auto& p : P) c += !p.pass && p.axis != p.pred_axis;
    return c;
}
// 실패 사진 이름 목록과 (정답 후보 비용 - 고른 후보 비용) 최솟값
inline std::string fail_names(const Manifest& m, const std::vector<Pred>& P) {
    std::string s;
    for (auto& p : P) if (!p.pass) s += (s.empty() ? "" : ", ") + stem(m.it[p.i].file);
    return s.empty() ? "없음" : s;
}
inline double min_fail_gap(const std::vector<Pred>& P) {
    double g = std::numeric_limits<double>::infinity();
    for (auto& p : P) if (!p.pass) g = std::min(g, p.true_c - p.cost);
    return g;
}

struct ExpInfo {
    std::string title, when, problem, solution, prefix, source;
    std::vector<std::string> cfg_desc;  // 설정 설명
    std::vector<std::string> cfg_head;  // 설정 열 이름
    std::vector<std::vector<std::string>> cfg_cols;
};

inline hye::Table config_table(const ExpInfo& E, const Nested& N) {
    std::vector<std::string> h = {"설정"};
    h.insert(h.end(), E.cfg_head.begin(), E.cfg_head.end());
    for (auto s : {"OOF Y", "OOF T", "OOF 실패", "OOF Y 최대", "OOF T 최대", "재대입 실패", "재대입 Y 최대", "재대입 T 최대"}) h.push_back(s);
    hye::Table t{h};
    for (size_t k = 0; k < N.oof.size(); ++k) {
        auto o = stats(N.oof[k]), r = stats(N.resub[k]);
        std::vector<std::string> row = {cfg_name(E.prefix[0], (int)k)};
        row.insert(row.end(), E.cfg_cols[k].begin(), E.cfg_cols[k].end());
        for (auto s : {std::to_string(o.Yc) + "/" + std::to_string(o.Yn), std::to_string(o.Tc) + "/" + std::to_string(o.Tn), std::to_string(o.fail),
                       hye::fmt(o.Ymax, 4), hye::fmt(o.Tmax, 4), std::to_string(r.fail), hye::fmt(r.Ymax, 4), hye::fmt(r.Tmax, 4)})
            row.push_back(s);
        t.add(row);
    }
    return t;
}

inline hye::Table pred_table(const Manifest& m, const std::vector<Pred>& P, const std::vector<int>& fold) {
    hye::Table t{{"file", "axis", "value", "session", "fold", "pred_axis", "pred", "branch", "cost", "correct", "eY", "eT", "pass"}};
    for (size_t r = 0; r < P.size(); ++r) {
        auto& p = P[r];
        t.add({m.it[p.i].file, std::string(1, p.axis), hye::fmt(p.value, 2), p.session, std::to_string(fold[r]), std::string(1, p.pred_axis), hye::fmt(p.pred, 6),
               p.branch, hye::fmt(p.cost, 4), p.correct ? "1" : "0", hye::fmt(p.eY, 6), hye::fmt(p.eT, 6), p.pass ? "1" : "0"});
    }
    return t;
}

// 공통 실행·보고. oneliner(N, best) 는 한줄평 문장, extra_result 는 결과.md 에 덧붙일 글
struct Report {
    ExpInfo E;
    const std::vector<hye::Shot>* shots;
    const Measured* M;
    const Nested* N;
    hye::Parity par;
    std::string oneliner, extra_result, notported, method_note;

    int best_oof() const {
        int b = 0;
        for (int k = 1; k < (int)N->oof.size(); ++k) {
            auto s = stats(N->oof[k]), t = stats(N->oof[b]);
            if (std::make_tuple(s.fail, s.Ymax, k) < std::make_tuple(t.fail, t.Ymax, b)) b = k;
        }
        return b;
    }

    void figures() {
        const auto& m = M->m;
        std::vector<double> ty, py, tt, pt;
        std::vector<std::string> names;
        for (auto& p : N->nested) {
            ty.push_back(p.axis == 'Y' ? p.value : 0);
            tt.push_back(p.axis == 'T' ? p.value : 0);
            py.push_back(p.pred_axis == 'Y' ? p.pred : 0);
            pt.push_back(p.pred_axis == 'T' ? p.pred : 0);
            names.push_back(stem(m.it[p.i].file));
        }
        hye::pred_vs_truth("01_내부선택OOF_오차.png", "Nested OOF 280", ty, py, tt, pt, names, 4);
        std::vector<double> x, fo, fr;
        for (size_t k = 0; k < N->oof.size(); ++k) {
            x.push_back((double)k);
            fo.push_back(stats(N->oof[k]).fail);
            fr.push_back(stats(N->resub[k]).fail);
        }
        hye::Chart c("Failures per config (280 photos)", "config index", "failures");
        c.line(x, fo, hye::color(0), "4-fold OOF", true).line(x, fr, hye::color(1), "resubstitution", true);
        for (auto& s : N->sel) c.vline(s.chosen, hye::kGray);
        c.save("02_설정별_실패수.png");
    }

    void write() {
        const auto& m = M->m;
        const auto& sh = *shots;
        figures();
        Stats ns = stats(N->nested);
        int bk = best_oof();
        Stats bs = stats(N->oof[bk]);
        int resub_zero = 0;
        for (auto& r : N->resub) resub_zero += stats(r).fail == 0;
        // 튀는 값: 실패 전부 + Y·T 최대 오차
        std::vector<int> pick;
        for (int r = 0; r < (int)N->nested.size(); ++r) if (!N->nested[r].pass) pick.push_back(r);
        int wy = 0, wt = 0;
        for (int r = 0; r < (int)N->nested.size(); ++r) {
            if (N->nested[r].eY > N->nested[wy].eY) wy = r;
            if (N->nested[r].eT > N->nested[wt].eT) wt = r;
        }
        // Y 튀는 값 사진은 정답 축이 Y 인 사진 중 가장 큰 Y 오차, T 는 정답 축 T 중 가장 큰 T 오차
        int wyY = -1, wtT = -1;
        for (int r = 0; r < (int)N->nested.size(); ++r) {
            auto& p = N->nested[r];
            if (p.axis == 'Y' && (wyY < 0 || p.eY > N->nested[wyY].eY)) wyY = r;
            if (p.axis == 'T' && (wtT < 0 || p.eT > N->nested[wtT].eT)) wtT = r;
        }
        for (int r : {wy, wt, wyY, wtT}) if (std::find(pick.begin(), pick.end(), r) == pick.end()) pick.push_back(r);
        hye::Table out{{"사진", "fold", "정답 Y (mm)", "정답 T (deg)", "추정 축·값", "eY", "eT", "고른 후보 (비용)", "정답 후보의 값 (비용)", "630점 평균 dx (px)"}};
        for (int r : pick) {
            auto& p = N->nested[r];
            std::string truth_cost = hye::fmt(p.true_v, 4) + " (" + (std::isfinite(p.true_c) ? hye::fmt(p.true_c, 2) : std::string("inf")) + ")";
            out.add({stem(m.it[p.i].file), std::to_string(N->nested_fold[r]), hye::fmt(p.axis == 'Y' ? p.value : 0, 2), hye::fmt(p.axis == 'T' ? p.value : 0, 2),
                     std::string(1, p.pred_axis) + " " + hye::fmts(p.pred, 4), hye::fmt(p.eY, 4), hye::fmt(p.eT, 4), p.branch + " (" + hye::fmt(p.cost, 2) + ")",
                     p.session + " " + truth_cost, hye::fmts(mean_dx(*M, p.i), 3)});
        }
        auto& pY = N->nested[wyY];
        auto& pT = N->nested[wtT];
        std::string fy = "03_튀는값_Y_" + stem(m.it[pY.i].file) + ".png", ft = "04_튀는값_T_" + stem(m.it[pT.i].file) + ".png";
        hye::save_png(fy, qr_zoom(sh[m.it[pY.i].shot], *M, pY.i, stem(m.it[pY.i].file) + " pred " + std::string(1, pY.pred_axis) + " " + hye::fmts(pY.pred, 3) + " eY " + hye::fmt(pY.eY, 3)));
        hye::save_png(ft, qr_zoom(sh[m.it[pT.i].shot], *M, pT.i, stem(m.it[pT.i].file) + " pred " + std::string(1, pT.pred_axis) + " " + hye::fmts(pT.pred, 3) + " eT " + hye::fmt(pT.eT, 3)));

        // CSV
        config_table(E, *N).csv(hye::out("summary.csv"));
        pred_table(m, N->nested, N->nested_fold).csv(hye::out("predictions_nested.csv"));

        hye::Table selt{{"fold", "고른 설정", "안쪽 실패", "안쪽 최대오차", "2위", "3위", "바깥 시험 Y", "바깥 시험 T", "바깥 실패"}};
        for (auto& s : N->sel) {
            std::vector<Pred> fp;
            for (size_t r = 0; r < N->nested.size(); ++r) if (N->nested_fold[r] == s.fold) fp.push_back(N->nested[r]);
            auto fs = stats(fp);
            auto nm = [&](int j) { return cfg_name(E.prefix[0], std::get<2>(s.rank[j])) + " (" + std::to_string(std::get<0>(s.rank[j])) + ", " + hye::fmt(std::get<1>(s.rank[j]), 4) + ")"; };
            selt.add({std::to_string(s.fold), cfg_name(E.prefix[0], s.chosen) + " " + E.cfg_desc[s.chosen], std::to_string(std::get<0>(s.rank[0])), hye::fmt(std::get<1>(s.rank[0]), 4),
                      nm(1), nm(2), std::to_string(fs.Yc) + "/" + std::to_string(fs.Yn), std::to_string(fs.Tc) + "/" + std::to_string(fs.Tn), std::to_string(fs.fail)});
        }

        std::ostringstream R;
        R << "# 결과: " << E.title << "\n\n"
          << "입력: `data/clean/clean_20261014` 281장 (center 1, T 200, Y 80). center.bmp 의 630 구조점에서 QR마다 패치를 잘라 LK15로 시작하고 GN으로 정밀화한 이동을 다시 측정했다 (옛 측정 파일을 읽지 않는다).\n\n"
          << method_note << "\n\n"
          << "분할: 바깥 4-fold (StratifiedKFold 4, 시행 6개로 층화, seed 20261021, 학습 210 / 시험 70), 안쪽 3-fold (seed 100+fold). 안쪽 성적(실패 수, Y·T 최대 오차, 설정 번호 순)이 가장 좋은 설정을 바깥 시험에 쓴 것이 내부 선택 OOF다.\n\n"
          << "## 내부 선택 OOF\n\n| 지표 | 값 |\n|---|---|\n"
          << "| Y 축·부호 정답 | " << ns.Yc << " / " << ns.Yn << " |\n| T 축·부호 정답 | " << ns.Tc << " / " << ns.Tn << " |\n| 통과 실패 | " << ns.fail << " / " << ns.n << " |\n"
          << "| Y 최대 \\|오차\\| (mm) | " << hye::fmt(ns.Ymax, 6) << " |\n| T 최대 \\|오차\\| (deg) | " << hye::fmt(ns.Tmax, 6) << " |\n\n"
          << "## fold별 선택\n\n" << selt.md() << "\n"
          << "## 설정별 성적 (" << N->oof.size() << "개)\n\n" << config_table(E, *N).md() << "\n"
          << extra_result
          << "## 원래 결과와 대조 (" << par.ok << "/" << par.total << " 일치)\n\n" << par.t.md() << "\n"
          << "## 옮기지 않은 부분\n\n" << notported << "\n\n"
          << "## 출력 파일\n\n- `summary.csv`: 설정별 OOF·재대입 성적\n- `predictions_nested.csv`: 내부 선택 OOF 280장 예측\n- `이미지/01_내부선택OOF_오차.png`, `02_설정별_실패수.png`, `03_튀는값_Y_*.png`, `04_튀는값_T_*.png`\n";
        hye::write_text(hye::out("결과.md"), R.str());

        std::ostringstream P;
        P << "# 보고서: " << E.title << " (" << E.when << ")\n\n"
          << "## 문제\n\n" << E.problem << "\n\n## 해결 방법\n\n" << E.solution << "\n\n"
          << "## 결과\n\n내부 선택 OOF에서 Y " << ns.Yc << "/" << ns.Yn << ", T " << ns.Tc << "/" << ns.Tn << ", 통과 실패 " << ns.fail << "장이었다.\n\n"
          << "| 지표 | 값 |\n|---|---|\n"
          << "| 내부 선택 OOF 통과 | " << (ns.n - ns.fail) << " / " << ns.n << " |\n"
          << "| 내부 선택 OOF Y / T 축·부호 정답 | " << ns.Yc << "/" << ns.Yn << " · " << ns.Tc << "/" << ns.Tn << " |\n"
          << "| 내부 선택 OOF Y / T 최대 \\|오차\\| | " << hye::fmt(ns.Ymax, 4) << " mm / " << hye::fmt(ns.Tmax, 4) << " deg |\n"
          << "| fold별 고른 설정 | ";
        for (auto& s : N->sel) P << (s.fold ? ", " : "") << cfg_name(E.prefix[0], s.chosen);
        P << " |\n"
          << "| 사후 최고 단일 설정 OOF | " << cfg_name(E.prefix[0], bk) << " " << esc(E.cfg_desc[bk]) << ": 실패 " << bs.fail << ", Y 최대 " << hye::fmt(bs.Ymax, 4) << " |\n"
          << "| 재대입 실패 0 설정 수 | " << resub_zero << " / " << N->resub.size() << " |\n"
          << "| 측정 유효율 (gn17, 280장 x 630점) | " << hye::fmt(valid_rate(M->tr.at("gn17")) * 100, 2) << "% |\n\n"
          << hye::img_md("01_내부선택OOF_오차.png", "내부 선택 OOF 오차") << "\n\n" << hye::img_md("02_설정별_실패수.png", "설정별 실패 수") << "\n\n"
          << "회색 세로선은 fold마다 안쪽 3-fold가 고른 설정이다.\n\n"
          << "## 튀는 값\n\n내부 선택 OOF에서 가장 크게 틀린 사진은 Y 오차 " << stem(m.it[N->nested[wy].i].file) << "(" << hye::fmt(N->nested[wy].eY, 3) << " mm), T 오차 "
          << stem(m.it[N->nested[wt].i].file) << "(" << hye::fmt(N->nested[wt].eT, 3) << " deg)이고, 실패 " << ns.fail << "장 중 " << wrong_axis(N->nested)
          << "장은 다른 축 후보의 비용이 정답 시행 후보보다 낮아 축을 바꿔 읽은 경우다.\n\n"
          << out.md() << "\n"
          << hye::img_md(fy, "Y 튀는 값") << "\n\n" << hye::img_md(ft, "T 튀는 값") << "\n\n"
          << "초록 상자는 기준 사진의 QR 영역, 빨간 상자는 630점 평균 이동만큼 옮긴 영역이다. 십자는 측정 위치(3점마다 1개), 화살표는 그 점의 이동에서 QR 평균 이동을 뺀 나머지를 400배 한 것이다.\n\n"
          << "## 한줄평\n\n" << oneliner << "\n\n"
          << "## 원래 결과와 대조\n\n원래 값은 `" << E.source << "`의 `summary.csv`, `nested_selection.json`, `predictions.csv`에서 가져왔다. " << par.ok << "/" << par.total << " 일치.\n\n" << par.t.md();
        hye::write_text(hye::out("보고서.md"), P.str());
    }
};

}  // namespace g8
