// QR630 + 판 나사 2개 고정 설정(c12) 원본 재측정 검증: P12 12계수 + 나사 평균 dx/dy 4값 = 16차원, 6원점 이차곡선 GLS(축소 .001).
// 원본: codex_experiments_20261014/release_screw_fallback_20261021/{pipeline.py, verify_raw.py, report.py} (저장소 실험/ 과 같음)
//       나사 특징점: measurement/measure.py feature_set (regions 3,4), 나사 추적: measurement/extract_pair.py PairExtractor('gn17', include_screws)
// 측정·판정 공통부: 20261021_1753_후보별잡음_가우스우도/실험/src
#include "g8_report.hpp"
#include <iostream>

using namespace g8;

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
    omp_set_num_threads(4);
    hye::begin_results();
    auto shots = hye::load_1014();
    auto M = measure_all(shots, {{"gn17", 17, 0.8, 0, 6}}, true);
    const auto& m = M.m;
    // pipeline.p12 는 float32 로 저장된 crop 좌표 + CROP 을 쓴다
    std::vector<cv::Point2d> p32;
    for (auto& p : M.pts) p32.push_back({(double)(float)(p.x - kCX) + kCX, (double)(float)(p.y - kCY) + kCY});
    cv::Mat Q = spatial(p32, rel_D(M.tr["gn17"]));
    auto SD = rel_D(M.screw);
    cv::Mat X(m.n(), 16, CV_64F);
    int nsc[2] = {0, 0};
    for (int n = 0; n < m.n(); ++n) {
        Q.row(n).copyTo(X.row(n).colRange(0, 12));
        double s[4] = {0, 0, 0, 0};
        int c[2] = {0, 0};
        for (size_t i = 0; i < SD[n].size(); ++i) {
            int r = M.screw_set.region[i] - 3;
            s[2 * r] += SD[n][i].x;
            s[2 * r + 1] += SD[n][i].y;
            ++c[r];
        }
        for (int r = 0; r < 2; ++r) { X.at<double>(n, 12 + 2 * r) = s[2 * r] / c[r]; X.at<double>(n, 13 + 2 * r) = s[2 * r + 1] / c[r]; nsc[r] = c[r]; }
    }
    int qr_valid_all = 1, screw_valid_all = 1;
    for (int n = 0; n < m.n(); ++n) {
        for (char v : M.tr["gn17"][n].valid) qr_valid_all &= v;
        for (char v : M.screw[n].valid) screw_valid_all &= v;
    }

    // 고정 설정 c12: inferred_session, 이차, 축소 .001. 바깥 4-fold 학습으로 맞춘 모델로 시험, 전체 280 모델로 재대입
    auto folds = outer_folds(m);
    std::vector<Pred> oof;
    std::vector<int> oof_fold;
    std::vector<Stats> fs;
    for (int f = 0; f < 4; ++f) {
        auto p = zpredict(m, X, folds[f].te, mfit(m, X, folds[f].tr, true, 0.001));
        fs.push_back(stats(p));
        oof.insert(oof.end(), p.begin(), p.end());
        oof_fold.insert(oof_fold.end(), p.size(), f);
    }
    GLSModel full = mfit(m, X, m.ids, true, 0.001);
    auto resub = zpredict(m, X, m.ids, full);
    Stats so = stats(oof), sr = stats(resub);
    // T-0.05 한 장 예시 (pipeline.Predictor.predict 와 같은 전체 모델)
    int i005 = 0;
    for (int i = 0; i < m.n(); ++i) if (m.it[i].file == "T-0.05.bmp") i005 = i;
    auto ex005 = zpredict(m, X, {i005}, full)[0];
    // QR 만(P12 12차원) 같은 설정과 비교
    cv::Mat X12 = X.colRange(0, 12).clone();
    std::vector<Pred> oof12;
    for (int f = 0; f < 4; ++f) {
        auto p = zpredict(m, X12, folds[f].te, mfit(m, X12, folds[f].tr, true, 0.001));
        oof12.insert(oof12.end(), p.begin(), p.end());
    }
    Stats s12 = stats(oof12);

    // ---- 그림 ----
    std::vector<double> ty, py, tt, pt;
    std::vector<std::string> names;
    for (auto& p : oof) {
        ty.push_back(p.axis == 'Y' ? p.value : 0);
        tt.push_back(p.axis == 'T' ? p.value : 0);
        py.push_back(p.pred_axis == 'Y' ? p.pred : 0);
        pt.push_back(p.pred_axis == 'T' ? p.pred : 0);
        names.push_back(stem(m.it[p.i].file));
    }
    hye::pred_vs_truth("01_raw_oof4_오차.png", "QR630+screws raw OOF", ty, py, tt, pt, names, 3);
    {
        std::vector<std::string> cats;
        std::vector<double> vals;
        for (int f = 0; f < 4; ++f) {
            cats.push_back("fold" + std::to_string(f) + " Y");
            vals.push_back(fs[f].Ymax);
            cats.push_back("fold" + std::to_string(f) + " T");
            vals.push_back(fs[f].Tmax);
        }
        hye::bar_chart("02_fold별_최대오차.png", "Max |error| per fold (tolerance 0.09)", cats, vals, "max |error|", -1, 0.09);
    }
    int wy = 0, wt = 0;
    for (int r = 0; r < (int)oof.size(); ++r) {
        if (oof[r].eY > oof[wy].eY) wy = r;
        if (oof[r].eT > oof[wt].eT) wt = r;
    }
    auto screw_tiles = [&](int n) {
        cv::Mat im = hye::read_color(shots[m.it[n].shot]);
        std::vector<cv::Mat> t;
        for (int r = 0; r < 2; ++r) {
            std::vector<hye::Mark> mk;
            for (size_t i = 0; i < M.screw_set.pts.size(); ++i) {
                if (M.screw_set.region[i] != r + 3) continue;
                hye::Mark k;
                k.p = {M.screw_set.pts[i].x + kCX + SD[n][i].x, M.screw_set.pts[i].y + kCY + SD[n][i].y};
                mk.push_back(k);
                hye::Mark g;
                g.p = {M.screw_set.pts[i].x + kCX, M.screw_set.pts[i].y + kCY};
                g.c = hye::kGreen;
                mk.push_back(g);
            }
            cv::Rect roi(kScrew[r][0] - 45, kScrew[r][1] - 45, 90, 90);
            t.push_back(hye::zoom(im, roi, 300, mk, 0, "screw" + std::to_string(r + 1) + " dx " + hye::fmt(X.at<double>(n, 12 + 2 * r), 2)));
        }
        return t;
    };
    auto fig = [&](int r, const std::string& file) {
        auto& p = oof[r];
        std::string cap = stem(m.it[p.i].file) + " pred " + std::string(1, p.pred_axis) + " " + hye::fmts(p.pred, 4);
        cv::Mat a = qr_zoom(shots[m.it[p.i].shot], M, p.i, cap);
        auto st = screw_tiles(p.i);
        cv::Mat b = hye::hstack(st);
        hye::save_png(file, hye::vstack({a, b}));
    };
    std::string fy = "03_튀는값_Y_" + stem(m.it[oof[wy].i].file) + ".png", ft = "04_튀는값_T_" + stem(m.it[oof[wt].i].file) + ".png";
    fig(wy, fy);
    fig(wt, ft);

    // ---- 표 ----
    hye::Table foldt{{"fold", "시험 장수", "Y 정답", "T 정답", "실패", "Y 최대 |오차|", "T 최대 |오차|"}};
    for (int f = 0; f < 4; ++f)
        foldt.add({std::to_string(f), std::to_string(fs[f].n), std::to_string(fs[f].Yc) + "/" + std::to_string(fs[f].Yn), std::to_string(fs[f].Tc) + "/" + std::to_string(fs[f].Tn),
                   std::to_string(fs[f].fail), hye::fmt(fs[f].Ymax, 6), hye::fmt(fs[f].Tmax, 6)});
    foldt.add({"재대입 280", std::to_string(sr.n), std::to_string(sr.Yc) + "/" + std::to_string(sr.Yn), std::to_string(sr.Tc) + "/" + std::to_string(sr.Tn), std::to_string(sr.fail),
               hye::fmt(sr.Ymax, 6), hye::fmt(sr.Tmax, 6)});
    hye::Table outl{{"구분", "사진", "fold", "정답 Y (mm)", "정답 T (deg)", "추정 축·값", "eY", "eT", "고른 후보 (비용)", "나사1 dx / 나사2 dx (px)", "630점 평균 dx (px)"}};
    auto addrow = [&](const std::string& lab, int r) {
        auto& p = oof[r];
        outl.add({lab, stem(m.it[p.i].file), std::to_string(oof_fold[r]), hye::fmt(p.axis == 'Y' ? p.value : 0, 2), hye::fmt(p.axis == 'T' ? p.value : 0, 2),
                  std::string(1, p.pred_axis) + " " + hye::fmts(p.pred, 4), hye::fmt(p.eY, 4), hye::fmt(p.eT, 4), p.branch + " (" + hye::fmt(p.cost, 2) + ")",
                  hye::fmts(X.at<double>(p.i, 12), 3) + " / " + hye::fmts(X.at<double>(p.i, 14), 3), hye::fmts(mean_dx(M, p.i), 3)});
    };
    {
        std::vector<double> ey, et;
        for (auto& p : oof) { ey.push_back(p.eY); et.push_back(p.eT); }
        for (auto r : hye::worst(ey, 3)) addrow("Y 오차 상위", (int)r);
        for (auto r : hye::worst(et, 3)) addrow("T 오차 상위", (int)r);
    }
    pred_table(m, oof, oof_fold).csv(hye::out("predictions_raw_oof4.csv"));
    {
        hye::Table ft16{{"file", "c0x", "c0y", "c1x", "c1y", "c2x", "c2y", "c3x", "c3y", "c4x", "c4y", "c5x", "c5y", "screw1_dx", "screw1_dy", "screw2_dx", "screw2_dy"}};
        for (int n = 0; n < m.n(); ++n) {
            std::vector<std::string> r = {m.it[n].file};
            for (int j = 0; j < 16; ++j) r.push_back(hye::fmt(X.at<double>(n, j), 8));
            ft16.add(r);
        }
        ft16.csv(hye::out("features16.csv"));
    }

    // ---- 대조 ----  원래 값: release_screw_fallback_20261021/raw_summary.csv, raw_fold_results.csv, audit.json
    hye::Parity par;
    par.add("raw_oof4 Y 정답", 80, so.Yc, 0, 0);
    par.add("raw_oof4 T 정답", 200, so.Tc, 0, 0);
    par.add("raw_oof4 실패", 0, so.fail, 0, 0);
    par.add("raw_oof4 Y 최대 오차 (mm)", 0.03745912103396665, so.Ymax, 1e-4, 6);
    par.add("raw_oof4 T 최대 오차 (deg)", 0.0029508104247339666, so.Tmax, 1e-5, 6);
    const double fy0[4] = {0.03745912103396665, 0.033430595103237426, 0.010236603858697846, 0.0027820800346185504};
    const double ft0[4] = {0.0012095844573083644, 0.0029508104247339666, 0.0013430042942713522, 0.002284314700068544};
    for (int f = 0; f < 4; ++f) {
        par.add("fold" + std::to_string(f) + " Y 최대 오차", fy0[f], fs[f].Ymax, 1e-4, 6);
        par.add("fold" + std::to_string(f) + " T 최대 오차", ft0[f], fs[f].Tmax, 1e-5, 6);
    }
    par.add("raw_resub280 실패", 0, sr.fail, 0, 0);
    par.add("raw_resub280 Y 최대 오차", 0.008636280964149812, sr.Ymax, 1e-4, 6);
    par.add("raw_resub280 T 최대 오차", 0.0027923672862009193, sr.Tmax, 1e-5, 6);
    par.add_text("나사 점 수 (나사1 / 나사2)", "17 / 22", std::to_string(nsc[0]) + " / " + std::to_string(nsc[1]), nsc[0] == 17 && nsc[1] == 22);
    par.add_text("모든 QR·나사 점 유효", "true", qr_valid_all && screw_valid_all ? "true" : "false", qr_valid_all && screw_valid_all);
    par.add("T-0.05 예시 추정 T", -0.049810694748573324, ex005.pred_axis == 'T' ? ex005.pred : hye::kNaN, 1e-5, 6);

    // ---- 결과.md ----
    std::ostringstream R;
    R << "# 결과: QR630 + 나사 2개 고정 설정 원본 재측정 검증\n\n"
      << "입력: `data/clean/clean_20261014` 281장 (center 1, T 200, Y 80). 옛 특징 파일(npz)과 모델(json)을 읽지 않고 사진에서 다시 쟀다.\n\n"
      << "- QR: center.bmp 630 구조점, QR마다 잘라 LK15(흐림 0.8) 시작 -> GN17(3차 B-스플라인, 밝기 gain/offset). 630점 이동의 P12 12계수.\n"
      << "- 나사: center.bmp crop의 나사 두 개 둘레 반지름 21 px에서 Shi-Tomasi(최대 60, 품질 0.01, 간격 4) + cornerSubPix로 " << nsc[0] << " + " << nsc[1]
      << "점을 골라, 63x63 템플릿 매칭 -> LK31(흐림 0.8) -> GN17. 나사마다 평균 dx, dy.\n"
      << "- 판정: 16차원, 6원점 후보(T+, T-, Y1+, Y1-, Y2+, Y2-), 축 공유 이차곡선, 공분산 상관 축소 0.001 GLS (multiref c12 고정 설정).\n"
      << "- 분할: StratifiedKFold(4, seed 20261021) 바깥 4-fold 학습 210 / 시험 70, 그리고 전체 280 재대입.\n\n"
      << "## fold별 성적\n\n" << foldt.md() << "\n"
      << "## QR만(P12 12차원, 같은 설정) 비교\n\n| 특징 | OOF Y | OOF T | 실패 | Y 최대 | T 최대 |\n|---|---|---|---|---|---|\n"
      << "| QR630 + 나사 (16) | " << so.Yc << "/80 | " << so.Tc << "/200 | " << so.fail << " | " << hye::fmt(so.Ymax, 4) << " | " << hye::fmt(so.Tmax, 4) << " |\n"
      << "| QR630 만 (12) | " << s12.Yc << "/80 | " << s12.Tc << "/200 | " << s12.fail << " | " << hye::fmt(s12.Ymax, 4) << " | " << hye::fmt(s12.Tmax, 4) << " |\n\n"
      << "## T-0.05 한 장 예시 (전체 280 모델)\n\n축 " << ex005.pred_axis << ", 값 " << hye::fmts(ex005.pred, 6) << ", 고른 후보 " << ex005.branch << " (비용 " << hye::fmt(ex005.cost, 3) << ")\n\n"
      << "## 원래 결과와 대조 (" << par.ok << "/" << par.total << " 일치)\n\n" << par.t.md() << "\n"
      << "## 옮기지 않은 부분\n\n"
      << "- center 630점 위치(QR 디코딩 격자 + erf 경계 모델 맞춤)는 옮기지 않고 #56 실험의 `points630.hpp` 고정 ID 목록을 쓴다. 나사 39점은 C++에서 다시 골랐다.\n"
      << "- 원본 verify_raw.py 는 바깥 fold 모델을 옛 캐시 특징(gn17_screws_mean_features.npy)으로 맞추고 새로 잰 특징으로 시험했다. C++ 는 캐시를 쓰지 않으므로 새로 잰 특징으로 맞추고 시험한다(원본 기록상 두 특징 차 최대 1.2e-6).\n"
      << "- 재대입 모델도 옛 c12_model.json 대신 새로 잰 280장으로 다시 맞췄다. 기준 사진 SHA-256 확인, 모델 json 저장, HTML 보고서는 옮기지 않았다.\n\n"
      << "## 출력 파일\n\n- `features16.csv`: 281장 16차원 특징 (P12 12 + 나사 4)\n- `predictions_raw_oof4.csv`: 바깥 4-fold 시험 280장 예측\n"
      << "- `이미지/01_raw_oof4_오차.png`, `02_fold별_최대오차.png`, `03_튀는값_Y_*.png`, `04_튀는값_T_*.png`\n";
    hye::write_text(hye::out("결과.md"), R.str());

    // ---- 보고서.md ----
    auto& pY = oof[wy];
    auto& pT = oof[wt];
    std::ostringstream P;
    P << "# 보고서: QR630 + 나사 2개 고정 설정 검증 (10-21 18:04)\n\n"
      << "## 문제\n\nQR만으로 280/280이 안 나오던 시점에 판 나사 2개를 더한 고정 설정이 원본 재실행에서도 통과하는지 확인해야 했다.\n\n"
      << "## 해결 방법\n\nP12 12계수와 나사 2개 평균 이동 4값을 합친 16차원, 6원점 이차곡선 GLS(c12) 설정을 고정하고 원본 281장을 다시 측정해 4-fold와 재대입으로 검증했다.\n\n"
      << "## 결과\n\n바깥 4-fold 시험에서 Y " << so.Yc << "/" << so.Yn << ", T " << so.Tc << "/" << so.Tn << ", 최대 오차 Y " << hye::fmt(so.Ymax, 4) << " mm, T " << hye::fmt(so.Tmax, 4)
      << " deg로 " << (so.fail == 0 ? "모두 통과했다." : "실패가 남았다.") << "\n\n"
      << "| 지표 | 값 |\n|---|---|\n"
      << "| raw_oof4 통과 | " << (so.n - so.fail) << " / " << so.n << " |\n"
      << "| raw_oof4 Y / T 최대 \\|오차\\| | " << hye::fmt(so.Ymax, 4) << " mm / " << hye::fmt(so.Tmax, 4) << " deg |\n"
      << "| 재대입 280 통과 | " << (sr.n - sr.fail) << " / " << sr.n << " |\n"
      << "| 재대입 Y / T 최대 \\|오차\\| | " << hye::fmt(sr.Ymax, 4) << " mm / " << hye::fmt(sr.Tmax, 4) << " deg |\n"
      << "| 같은 설정 QR만(12차원) OOF 실패 | " << s12.fail << " / 280 |\n"
      << "| 점 유효 (QR 630 x 281, 나사 " << (nsc[0] + nsc[1]) << " x 281) | " << (qr_valid_all && screw_valid_all ? "모두 유효" : "일부 무효") << " |\n\n"
      << hye::img_md("01_raw_oof4_오차.png", "raw OOF 오차") << "\n\n" << hye::img_md("02_fold별_최대오차.png", "fold별 최대 오차") << "\n\n"
      << "## 튀는 값\n\n가장 큰 오차는 Y " << stem(m.it[pY.i].file) << "(" << hye::fmt(pY.eY, 4) << " mm, " << pY.branch << " 후보로 읽음), T " << stem(m.it[pT.i].file) << "("
      << hye::fmt(pT.eT, 4) << " deg)이고 모두 통과 기준 0.09 안이다.\n\n"
      << outl.md() << "\n" << hye::img_md(fy, "Y 튀는 값") << "\n\n" << hye::img_md(ft, "T 튀는 값") << "\n\n"
      << "위 줄: 초록 상자는 기준 QR 영역, 빨간 상자는 630점 평균 이동만큼 옮긴 영역, 화살표는 점 이동에서 QR 평균을 뺀 나머지 x400. 아래 줄: 나사 확대, 초록 십자는 center 위치, 빨간 십자는 이 사진에서 잰 위치.\n\n"
      << "## 한줄평\n\n나사는 QR과 다른 깊이에 있어 Y 이동과 T 회전에 다르게 반응하므로 나사 평균 이동 4값이 QR P12만으로는 비슷하던 축 후보 비용을 갈라 주고, 그래서 같은 설정의 QR만 OOF 실패 "
      << s12.fail << "장이 나사를 더하면 " << so.fail << "장이 되고 최대 오차도 Y " << hye::fmt(so.Ymax, 3) << " mm로 줄었지만, 많은 설정을 본 뒤 고른 고정 설정이라 새 촬영 일반화는 확인되지 않았다.\n\n"
      << "## 원래 결과와 대조\n\n원래 값은 `codex_experiments_20261014/release_screw_fallback_20261021`의 `raw_summary.csv`, `raw_fold_results.csv`, `audit.json`에서 가져왔다. "
      << par.ok << "/" << par.total << " 일치.\n\n" << par.t.md();
    hye::write_text(hye::out("보고서.md"), P.str());
    std::cout << "parity " << par.ok << "/" << par.total << "\n";
    return 0;
}
