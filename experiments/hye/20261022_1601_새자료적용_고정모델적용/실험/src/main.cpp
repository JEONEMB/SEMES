// 고정 모델을 20261014·20261020 341장에 그대로 적용 (원본: codex_experiments_20261014/evaluate_archives_20261022/evaluate.py)
//   원래는 저장된 joblib 모델 두 개(개선 비선형 fixedfeature_model, 좌표 선형 coordinate_model)를 불러와 썼다.
//   여기서는 옛 모델 파일을 읽지 않고, 같은 설정으로 20261014 280장 사진에서 C++ 로 다시 학습해 '고정 모델'로 쓴다.
//     개선 비선형 = P12 + ExtraTrees(80, leaf 1) + 잡음 증강, 좌표 선형 = 1260 좌표 gls_full alpha 0.01 (저장 모델 설정)
//   측정: 10-14 center.bmp 기준 630점, QR 대략 위치 ±130px 템플릿 매칭 (NCC < 0.6 이면 측정 실패, motion_web structural.py 와 같음)
#include "joint_models.hpp"
#include "numeric.hpp"
#include "struct630.hpp"
#include <hye/dataset.hpp>
#include <hye/plot.hpp>
#include <hye/report.hpp>
#include <iostream>
#include <set>

using namespace hye;
using cv::Mat;

namespace {
struct Rec {
    std::string archive, file, rel, scope;
    double ty, tt;
    int shot;
    bool ok = false, identical = false;
    double ncc = 0;
    Mat x12, raw;
};
struct Res { double py = kNaN, pt = kNaN; };
struct Sum { int total = 0, ok = 0, passed = 0; double ymae = 0, ymax = 0, tmae = 0, tmax = 0; };
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
    auto s14 = load_1014();
    auto s20 = load_set("clean_20261020");
    const auto& cshot = center_1014(s14);
    const cv::Mat center = s630::read_gray(cshot.path);
    auto all = s630::structural_corners(s630::decode_qr(center));
    s630::refine_center(center, all);
    auto cs = s630::non_diagonal(all);
    const int P = (int)cs.size();
    std::vector<Rec> recs;
    std::vector<const Shot*> sp;
    for (auto& s : s14) {
        Rec r{"20261014", s.rel.substr(s.rel.rfind('/') + 1), s.rel, s.condition == "center" ? "center" : "single", s.y, s.t};
        recs.push_back(r);
        sp.push_back(&s);
    }
    for (auto& s : s20) {
        Rec r{"20261020", s.rel.substr(s.rel.rfind('/') + 1), s.rel, s.condition == "center" ? "center" : "mixed", s.y, s.t};
        recs.push_back(r);
        sp.push_back(&s);
    }
    const int N = (int)recs.size();
    std::vector<s630::Track> tracks(N);
    s630::Track tc = s630::track(center, center, cs);
    std::vector<cv::Point2d> pts(P);
    for (int j = 0; j < P; ++j) pts[j] = {tc.obs[j].x, tc.obs[j].y};
#pragma omp parallel for schedule(dynamic) num_threads(4)
    for (int k = 0; k < N; ++k) {
        cv::Mat im = s630::read_gray(sp[k]->path);
        auto& r = recs[k];
        r.identical = im.size() == center.size() && cv::countNonZero(im != center) == 0;
        tracks[k] = s630::track(im, center, cs);
        r.ncc = std::min({tracks[k].score[0], tracks[k].score[1], tracks[k].score[2]});
        r.ok = r.ncc >= 0.6;
        if (!r.ok) continue;
        std::vector<cv::Point2d> d(P);
        r.raw = Mat(1, 2 * P, CV_64F);
        for (int j = 0; j < P; ++j) {
            d[j] = {tracks[k].obs[j].x - pts[j].x, tracks[k].obs[j].y - pts[j].y};
            r.raw.at<double>(0, 2 * j) = d[j].x;
            r.raw.at<double>(0, 2 * j + 1) = d[j].y;
        }
        auto f = s630::p12(pts, d);
        r.x12 = Mat(1, 12, CV_64F);
        for (int c = 0; c < 12; ++c) r.x12.at<double>(0, c) = f[c];
    }
    std::cout << "measured " << N << "\n";
    // ---- 고정 모델 학습: 20261014 의 center 를 뺀 280장 ----
    Mat X12(0, 12, CV_64F), RAW(0, 2 * P, CV_64F), Y(0, 2, CV_64F);
    for (auto& r : recs)
        if (r.archive == "20261014" && r.scope != "center") {
            X12.push_back(r.x12);
            RAW.push_back(r.raw);
            Mat yy = (cv::Mat_<double>(1, 2) << r.ty, r.tt);
            Y.push_back(yy);
        }
    jm::Forest forest;
    forest.seed = 20261022 + 4;  // #70 의 전체 학습(fold 4)과 같은 시드
    forest.fit(X12, Y);
    jm::GlsFull gls;
    gls.alpha = 0.01;
    gls.fit(RAW, Y);
    // ---- 적용 ----
    const std::vector<std::string> models{"improved_nonlinear", "coordinate_linear"};
    std::vector<std::array<Res, 2>> res(N);
    for (int k = 0; k < N; ++k) {
        auto& r = recs[k];
        if (!r.ok) continue;
        Mat p0 = forest.predict(r.x12), p1 = gls.predict(r.raw);
        res[k][0] = {p0.at<double>(0, 0), p0.at<double>(0, 1)};
        res[k][1] = {p1.at<double>(0, 0), p1.at<double>(0, 1)};
        if (r.identical) res[k][0] = res[k][1] = {0, 0};
    }
    std::map<std::tuple<std::string, std::string, std::string>, Sum> sums;
    Table pred{{"archive", "file", "scope", "true_Y", "true_T", "model", "status", "pred_Y", "pred_T", "eY", "eT", "both_within09", "min_ncc"}};
    for (int k = 0; k < N; ++k) {
        auto& r = recs[k];
        for (int m = 0; m < 2; ++m) {
            for (auto sub : {std::string("all"), r.scope}) {
                auto& s = sums[{r.archive, models[m], sub}];
                s.total++;
                if (!r.ok) continue;
                double eY = std::abs(res[k][m].py - r.ty), eT = std::abs(res[k][m].pt - r.tt);
                s.ok++;
                s.passed += eY <= .09 + 1e-12 && eT <= .09 + 1e-12;
                s.ymae += eY; s.tmae += eT; s.ymax = std::max(s.ymax, eY); s.tmax = std::max(s.tmax, eT);
            }
            double eY = std::abs(res[k][m].py - r.ty), eT = std::abs(res[k][m].pt - r.tt);
            pred.add({r.archive, r.rel, r.scope, fmt(r.ty, 2), fmt(r.tt, 2), models[m], r.ok ? "ok" : "measurement_failed", r.ok ? fmt(res[k][m].py, 6) : "", r.ok ? fmt(res[k][m].pt, 6) : "",
                      r.ok ? fmt(eY, 6) : "", r.ok ? fmt(eT, 6) : "", r.ok && eY <= .09 + 1e-12 && eT <= .09 + 1e-12 ? "True" : "False", fmt(r.ncc, 4)});
        }
    }
    pred.csv(out("predictions.csv"));
    Table sumT{{"자료", "모델", "부분", "전체", "추정 성공", "추정 실패", "두 축 <=0.09 통과", "Y 평균 / 최대 |오차| (성공분)", "T 평균 / 최대 |오차| (성공분)"}};
    for (auto& [key, s] : sums)
        sumT.add({std::get<0>(key), std::get<1>(key), std::get<2>(key), std::to_string(s.total), std::to_string(s.ok), std::to_string(s.total - s.ok), std::to_string(s.passed),
                  s.ok ? fmt(s.ymae / s.ok, 4) + " / " + fmt(s.ymax, 4) : "-", s.ok ? fmt(s.tmae / s.ok, 4) + " / " + fmt(s.tmax, 4) : "-"});
    auto S = [&](const std::string& a, const std::string& m, const std::string& sub) { return sums[{a, m, sub}]; };
    auto n20 = S("20261020", "improved_nonlinear", "all"), c20 = S("20261020", "coordinate_linear", "all");
    auto n14 = S("20261014", "improved_nonlinear", "all"), c14 = S("20261014", "coordinate_linear", "all");
    auto ncen = S("20261020", "improved_nonlinear", "center");
    // 조건별 (20261020, 주모델)
    Table condT{{"명령 (Y, T)", "장수", "추정 성공", "통과", "평균 추정 Y", "평균 추정 T", "최소 NCC"}};
    {
        std::map<std::pair<double, double>, std::vector<int>> g;
        for (int k = 0; k < N; ++k) if (recs[k].archive == "20261020") g[{recs[k].ty, recs[k].tt}].push_back(k);
        for (auto& [c, ks] : g) {
            int ok = 0, pass = 0;
            double sy = 0, st = 0, mn = 1;
            for (int k : ks) {
                mn = std::min(mn, recs[k].ncc);
                if (!recs[k].ok) continue;
                ++ok; sy += res[k][0].py; st += res[k][0].pt;
                pass += std::abs(res[k][0].py - c.first) <= .09 + 1e-12 && std::abs(res[k][0].pt - c.second) <= .09 + 1e-12;
            }
            condT.add({fmt(c.first, 2) + ", " + fmt(c.second, 2), std::to_string(ks.size()), std::to_string(ok), std::to_string(pass), ok ? fmt(sy / ok, 4) : "-", ok ? fmt(st / ok, 4) : "-", fmt(mn, 3)});
        }
    }

    // ---- 그림 ----
    std::vector<double> ty, tt, py, pt, ey, et;
    std::vector<std::string> nm;
    std::vector<int> kk;
    for (int k = 0; k < N; ++k)
        if (recs[k].archive == "20261020" && recs[k].ok) {
            ty.push_back(recs[k].ty); tt.push_back(recs[k].tt); py.push_back(res[k][0].py); pt.push_back(res[k][0].pt);
            ey.push_back(py.back() - ty.back()); et.push_back(pt.back() - tt.back());
            std::string n = recs[k].file.substr(0, recs[k].file.size() - 4);
            for (auto& ch : n) if (ch == ' ') ch = '_';
            nm.push_back(n);
            kk.push_back(k);
        }
    pred_vs_truth("01_20261020_비선형모델_오차.png", "10-20 frozen trees+noise", ty, py, tt, pt, nm, 3);
    {
        std::vector<double> a, b;
        for (int k : kk) { a.push_back(res[k][1].py); b.push_back(res[k][1].pt); }
        pred_vs_truth("02_20261020_좌표선형_오차.png", "10-20 frozen coordinate GLS", ty, a, tt, b, nm, 3);
    }
    {
        Chart c("10-20 mixed: commanded vs estimated (trees+noise)", "Y (mm)", "T (deg)");
        std::vector<double> cy, ct;
        for (int k = 0; k < N; ++k) if (recs[k].archive == "20261020") { cy.push_back(recs[k].ty); ct.push_back(recs[k].tt); }
        c.scatter(cy, ct, kGray, "command", 6).scatter(py, pt, color(3), "estimate", 3);
        c.save("03_명령과추정_YT평면.png");
    }
    auto outlier_fig = [&](int k, const std::string& file, const std::string& cap) {
        cv::Mat im = s630::read_gray(sp[k]->path);
        std::vector<cv::Mat> tiles;
        for (int q = 0; q < 3; ++q) {
            auto b = s630::kBoxes[q];
            auto sh = tracks[k].shift[q];
            cv::Rect roi(b[0] + std::min(0, sh.x) - 15, b[1] + std::min(0, sh.y) - 15, b[2] - b[0] + std::abs(sh.x) + 30, b[3] - b[1] + std::abs(sh.y) + 30);
            std::vector<Mark> mk;
            if (recs[k].ok)
                for (int j = 0; j < P; ++j) if (cs[j].qr == q + 1 && j % 4 == 0) mk.push_back({{tracks[k].obs[j].x, tracks[k].obs[j].y}, {}, "", kRed});
            tiles.push_back(zoom(im, roi, 420, mk, 0, q == 0 ? cap : "",
                                 {{cv::Rect2d(b[0], b[1], b[2] - b[0], b[3] - b[1]), kGreen, "ref"},
                                  {cv::Rect2d(b[0] + sh.x, b[1] + sh.y, b[2] - b[0], b[3] - b[1]), kRed, "NCC " + fmt(tracks[k].score[q], 2)}}));
        }
        save_png(file, hstack(tiles));
    };
    size_t wy = worst(ey, 1)[0], wt = worst(et, 1)[0];
    std::string fy = "04_튀는값_Y_" + nm[wy] + ".png", ft = "05_튀는값_T_" + nm[wt] + ".png";
    outlier_fig(kk[wy], fy, nm[wy] + " Y err " + fmts(ey[wy], 3) + " mm");
    outlier_fig(kk[wt], ft, nm[wt] + " T err " + fmts(et[wt], 3) + " deg");
    int kf = -1;
    for (int k = 0; k < N; ++k) if (!recs[k].ok) { kf = k; break; }
    std::string ff;
    if (kf >= 0) {
        std::string n = recs[kf].file.substr(0, recs[kf].file.size() - 4);
        for (auto& ch : n) if (ch == ' ') ch = '_';
        ff = "06_측정실패_" + n + ".png";
        outlier_fig(kf, ff, n + " min NCC " + fmt(recs[kf].ncc, 3));
    }
    Table outl{{"구분", "사진", "정답 Y", "추정 Y", "정답 T", "추정 T", "좌표 선형 추정 (Y / T)", "P12 c0x 평균 수평 이동 (px)", "최소 NCC"}};
    for (auto [lab, idx] : std::vector<std::pair<std::string, std::vector<size_t>>>{{"Y 오차 상위", worst(ey, 4)}, {"T 오차 상위", worst(et, 3)}})
        for (size_t q : idx) {
            int k = kk[q];
            outl.add({lab, recs[k].rel, fmt(ty[q], 2), fmt(py[q], 3), fmt(tt[q], 2), fmt(pt[q], 3), fmt(res[k][1].py, 3) + " / " + fmt(res[k][1].pt, 3),
                      fmt(recs[k].x12.at<double>(0, 0), 2), fmt(recs[k].ncc, 3)});
        }
    if (kf >= 0) outl.add({"측정 실패", recs[kf].rel, fmt(recs[kf].ty, 2), "-", fmt(recs[kf].tt, 2), "-", "-", "-", fmt(recs[kf].ncc, 3)});
    double cen_min = 1e9, cen_max = -1e9;
    for (int k = 0; k < N; ++k)
        if (recs[k].archive == "20261020" && recs[k].scope == "center" && recs[k].ok) { cen_min = std::min(cen_min, res[k][0].py); cen_max = std::max(cen_max, res[k][0].py); }
    // 혼합 사진에서 추정 Y 가 0 인 비율 (트리는 학습한 단일 축 정답만 섞을 수 있다)
    int mixed_ok = 0, mixed_axis = 0;
    for (int k = 0; k < N; ++k)
        if (recs[k].archive == "20261020" && recs[k].scope == "mixed" && recs[k].ok) { ++mixed_ok; mixed_axis += std::abs(res[k][0].py) < 1e-9 || std::abs(res[k][0].pt) < 1e-9; }

    // ---- 원래 결과와 대조 (evaluate_archives_20261022/summary.csv) ----
    Parity par;
    par.add("20261020 측정 실패 (NCC<0.6)", 12, n20.total - n20.ok, 0, 0);
    par.add("20261014 측정 실패", 0, n14.total - n14.ok, 0, 0);
    par.add("20261014 비선형 통과 (재대입)", 281, n14.passed, 0, 0);
    par.add("20261014 좌표 선형 통과 (재대입)", 281, c14.passed, 3, 0);
    par.add("20261014 좌표 선형 Y 최대 |오차| (mm)", 0.008503, c14.ymax, 0.004);
    par.add("20261020 비선형 통과 / 60", 14, n20.passed, 4, 0);
    par.add("20261020 비선형 Y 평균 |오차| (mm)", 0.223385, n20.ymae / n20.ok, 0.03);
    par.add("20261020 비선형 Y 최대 |오차| (mm)", 1.0, n20.ymax, 0.05);
    par.add("20261020 비선형 T 평균 |오차| (deg)", 0.018904, n20.tmae / n20.ok, 0.005);
    par.add("20261020 비선형 T 최대 |오차| (deg)", 0.075125, n20.tmax, 0.03);
    par.add("20261020 비선형 center 3장 통과", 0, ncen.passed, 0, 0);
    par.add("20261020 좌표 선형 통과 / 60", 12, c20.passed, 3, 0);
    par.add("20261020 좌표 선형 Y 평균 |오차| (mm)", 0.195854, c20.ymae / c20.ok, 0.01);
    par.add("20261020 좌표 선형 Y 최대 |오차| (mm)", 0.542356, c20.ymax, 0.03);
    par.add("20261020 좌표 선형 T 평균 |오차| (deg)", 0.036035, c20.tmae / c20.ok, 0.003);
    par.add("20261020 좌표 선형 T 최대 |오차| (deg)", 0.111038, c20.tmax, 0.01);

    std::ostringstream R;
    R << "# 결과: 고정 모델을 20261014·20261020 341장에 적용\n\n"
      << "입력: `data/clean/clean_20261014` 281장 + `data/clean/clean_20261020` 60장. 기준은 모두 10-14 center.bmp(새 묶음의 center1~3 으로 바꾸지 않음). 정답은 파일명 명령값.\n"
      << "측정: center.bmp 에서 고정한 630점을 사진마다 흐린 경계 모델로 맞춘다. QR 대략 위치는 ±130px 템플릿 매칭이고 세 QR 중 최소 NCC < 0.6 이면 측정 실패로 둔다(motion_web structural.py 와 같음).\n"
      << "모델: 옛 joblib 파일을 읽지 않고 같은 설정으로 20261014 280장 사진에서 C++ 로 다시 학습했다. 개선 비선형 = P12 + ExtraTrees(80그루, leaf 1) + 잡음 증강 3벌, "
      << "좌표 선형 = 1260 좌표 gls_full alpha 0.01(저장된 coordinate_model 의 설정). 20261020 은 학습에 쓰지 않았다. 사진 픽셀이 center.bmp 와 같으면 (0,0).\n\n"
      << "## 요약\n\n" << sumT.md()
      << "\n## 20261020 명령 조건별 (개선 비선형)\n\n" << condT.md()
      << "\n## 원래 결과와 대조 (" << par.ok << "/" << par.total << " 일치)\n\n" << par.t.md()
      << "\n트리 모델은 sklearn·numpy 난수열을 같게 만들 수 없어(#70 과 같은 이유) 통과 ±4장, 평균 오차 ±0.03 mm 를 일치로 본다. 20261014 비선형 재대입은 leaf 1 트리가 학습 사진 정답을 그대로 내므로 정확히 같아야 한다.\n"
      << "\n## 옮기지 않은 부분\n\n- 원래 측정 캐시(motion_web runs/…/result_*.json) 바이트 검증, source_audit, robot_corrections.csv(= -추정값), report.html 의 사진별 필터 표.\n"
      << "\n## 파일\n\n- `predictions.csv`: 341장 x 모델 2개 추정·오차\n- `이미지/`: 그림\n";
    write_text(out("결과.md"), R.str());

    std::ostringstream Pp;
    Pp << "# 보고서: 새 자료에 고정 모델 적용 (10-22 16:01)\n\n"
       << "## 문제\n\n저장된 개선 모델을 20261014·20261020 두 묶음에 그대로 적용하면 Y·T가 어떻게 나오는지 확인해야 했다.\n\n"
       << "## 해결 방법\n\n개선 비선형 모델과 좌표 선형 모델을 20261014 280장으로 한 번 학습해 고정하고, 재학습 없이 341장에 적용했다(옛 모델 파일 대신 같은 설정으로 사진에서 C++ 재학습).\n\n"
       << "## 결과\n\n학습하지 않은 20261020 60장은 비선형 " << n20.passed << "/60, 좌표 선형 " << c20.passed << "/60만 통과했고 T+2.0 " << n20.total - n20.ok << "장은 측정에 실패했다.\n\n"
       << "| 지표 | 개선 비선형 | 좌표 선형 |\n|---|---|---|\n"
       << "| 20261014 통과 / 281 (재대입) | " << n14.passed << " | " << c14.passed << " |\n"
       << "| 20261020 통과 / 60 | " << n20.passed << " | " << c20.passed << " |\n"
       << "| 20261020 측정 실패 | " << n20.total - n20.ok << " | " << c20.total - c20.ok << " |\n"
       << "| 20261020 Y 평균 / 최대 \\|오차\\| (mm, 성공분) | " << fmt(n20.ymae / n20.ok, 4) << " / " << fmt(n20.ymax, 4) << " | " << fmt(c20.ymae / c20.ok, 4) << " / " << fmt(c20.ymax, 4) << " |\n"
       << "| 20261020 T 평균 / 최대 \\|오차\\| (deg, 성공분) | " << fmt(n20.tmae / n20.ok, 4) << " / " << fmt(n20.tmax, 4) << " | " << fmt(c20.tmae / c20.ok, 4) << " / " << fmt(c20.tmax, 4) << " |\n"
       << "| 20261020 center 3장 추정 Y 범위 (mm) | " << fmt(cen_min, 3) << " ~ " << fmt(cen_max, 3) << " | - |\n\n"
       << img_md("01_20261020_비선형모델_오차.png", "비선형 오차") << "\n\n" << img_md("02_20261020_좌표선형_오차.png", "좌표 선형 오차") << "\n\n"
       << img_md("03_명령과추정_YT평면.png", "YT 평면") << "\n\n"
       << "## 튀는 값\n\n20261020 에서 가장 크게 틀린 사진은 Y는 " << nm[wy] << "(정답 Y " << fmt(ty[wy], 2) << ", 추정 " << fmt(py[wy], 3) << " mm), T는 " << nm[wt] << "(정답 T " << fmt(tt[wt], 2)
       << ", 추정 " << fmt(pt[wt], 3) << ")이고, 혼합 자세인데 단일 축 사진으로만 학습한 트리가 한 축 정답만 냈다.\n\n"
       << outl.md() << "\n" << img_md(fy, "Y 튀는 값") << "\n\n" << img_md(ft, "T 튀는 값");
    if (!ff.empty()) Pp << "\n\n" << img_md(ff, "측정 실패");
    Pp << "\n\n초록 상자는 기준 사진의 QR 영역, 빨간 상자는 ±130px 템플릿 매칭이 찾은 위치(NCC 표시), 빨간 십자는 맞춘 모서리(4개 중 1개)다. T+2.0 은 수평 이동이 약 160px 라 ±130px 탐색 밖이다.\n\n"
       << "## 한줄평\n\n개선 비선형 모델은 20261014 단일 축 사진만 외운 leaf 1 트리라서 학습 사진은 " << n14.passed << "/281 그대로 맞히지만, 혼합 자세 " << mixed_ok << "장 중 " << mixed_axis
       << "장에서 한 축을 0으로 내 새 묶음은 " << n20.passed << "/60만 통과했다.\n\n"
       << "## 원래 결과와 대조\n\n원래 값은 `codex_experiments_20261014/evaluate_archives_20261022/summary.csv`에서 가져왔다. " << par.ok << "/" << par.total
       << " 일치. 원래는 저장된 joblib 모델을 불러왔고 여기서는 같은 설정으로 사진에서 다시 학습한 모델이라, 트리의 난수열 차이만큼 20261020 추정이 달라진다.\n\n" << par.t.md();
    write_text(out("보고서.md"), Pp.str());
    std::cout << "10-20 trees " << n20.passed << " coord " << c20.passed << " failed " << n20.total - n20.ok << " parity " << par.ok << "/" << par.total << "\n";
    return 0;
}
