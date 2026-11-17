// 280장 전수 고정 모델 재실행 (원본: codex_experiments_20261014/full280/evaluate_all.py, infer.py, validate_calibrated_runs.py)
// 고정 모델 = calibrated_runs/model.json. 여기서는 그 파일을 읽지 않고 같은 절차(gn17_spread14 측정 -> 6후보 0점 2차 모델, 280장 전부로 학습)로 다시 만든다.
// 추론은 center 와 현재 사진 두 장만 받고, 정답은 예측이 끝난 뒤에만 본다.
#include <hye/dataset.hpp>
#include <hye/plot.hpp>
#include <hye/report.hpp>
#include <omp.h>
#include <iostream>
#include "spread.hpp"

using namespace hye;

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
    // 원래 순서: center, Theta 파일명 정렬, Y 파일명 정렬
    struct Row { cx::Item it; const Shot* s; };
    std::vector<Row> rows;
    for (auto& s : shots) {
        std::string f = s.rel.substr(s.rel.find('/') == std::string::npos ? 0 : s.rel.find('/') + 1);
        char a = s.condition == "center" ? 'C' : s.condition == "T" ? 'T' : 'Y';
        double v = a == 'T' ? s.t : a == 'Y' ? s.y : 0;
        rows.push_back({{f, a, v, cx::session_of(a, v)}, &s});
    }
    std::sort(rows.begin(), rows.end(), [](const Row& p, const Row& q) {
        int ka = p.it.axis == 'C' ? 0 : p.it.axis == 'T' ? 1 : 2, kb = q.it.axis == 'C' ? 0 : q.it.axis == 'T' ? 1 : 2;
        return ka != kb ? ka < kb : p.it.file < q.it.file;
    });
    const int NI = (int)rows.size();
    std::vector<cx::Item> m;
    for (auto& r : rows) m.push_back(r.it);
    int ic = 0;
    for (int i = 0; i < NI; ++i) if (m[i].axis == 'C') ic = i;

    cv::Mat ref = cx::crop(read_gray(*rows[ic].s));
    cx::Extractor ex(ref, 14);
    const int NP = (int)ex.P.p.size();
    int cnt[3] = {};
    for (int r : ex.P.region) ++cnt[r];
    std::vector<cv::Point2d> pts;
    for (int i = 0; i < NP; ++i) pts.push_back(ex.global(i));
    std::cout << "points " << NP << " (" << cnt[0] << "/" << cnt[1] << "/" << cnt[2] << ")\n";

    std::vector<cx::Extractor::Result> res(NI);
#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < NI; ++i) res[i] = ex.extract(cx::crop(read_gray(*rows[i].s)));
    int invalid = 0;
    for (auto& r : res) for (bool v : r.valid) invalid += !v;
    std::cout << "tracked, invalid " << invalid << "\n";

    // 학습: center 기준으로 뺀 이동의 12값, 280장 전부, 6후보 0점(구간 원-핫) + 2차
    auto to_d = [&](const std::vector<cv::Point2f>& D, const std::vector<cv::Point2f>* sub) {
        std::vector<cv::Point2d> d(NP);
        for (int k = 0; k < NP; ++k) d[k] = cv::Point2d(D[k].x, D[k].y) - (sub ? cv::Point2d((*sub)[k].x, (*sub)[k].y) : cv::Point2d(0, 0));
        return d;
    };
    std::vector<std::vector<double>> Xtr(NI);
    for (int i = 0; i < NI; ++i) Xtr[i] = cx::p12(pts, to_d(res[i].D, &res[ic].D));
    std::vector<int> ids;
    for (int i = 0; i < NI; ++i) if (i != ic) ids.push_back(i);
    cx::Model model = cx::fit(m, Xtr, ids, "inferred_session", true);
    const std::vector<double> zero = Xtr[ic];

    // 추론: 두 장만. 현재 사진의 유효 점만으로 12값을 맞추고 zero_feature 를 뺀다 (infer.run)
    struct Ev { int i; cx::Pred p; double ty, tt, py, pt, ey, et, margin; bool ok, pass; int valid; std::vector<double> X; };
    std::vector<Ev> ev;
    for (int i : ids) {
        auto& r = res[i];
        int nv = 0, per[3] = {};
        for (int k = 0; k < NP; ++k) if (r.valid[k]) { ++nv; ++per[ex.P.region[k]]; }
        Ev e{};
        e.i = i;
        e.valid = nv;
        if ((double)nv / NP < 0.95 || per[0] < 20 || per[1] < 20 || per[2] < 20) throw std::runtime_error("insufficient valid points " + m[i].file);
        auto X = cx::p12(pts, to_d(r.D, nullptr), &r.valid);
        for (int k = 0; k < 12; ++k) X[k] -= zero[k];
        e.X = X;
        e.p = cx::predict_one(X, model, true);
        e.margin = (e.p.second_cost - e.p.cost) / (e.p.second_cost + e.p.cost + 1e-12);
        // 정답은 예측 뒤에만 본다
        e.ty = m[i].axis == 'Y' ? m[i].value : 0;
        e.tt = m[i].axis == 'T' ? m[i].value : 0;
        e.py = e.p.axis == 'Y' ? e.p.value : 0;
        e.pt = e.p.axis == 'T' ? e.p.value : 0;
        e.ey = e.py - e.ty;
        e.et = e.pt - e.tt;
        e.ok = e.p.axis == m[i].axis && (e.p.value > 0 ? 1 : e.p.value < 0 ? -1 : 0) == (m[i].value > 0 ? 1 : -1);
        e.pass = e.ok && std::abs(e.ey) <= kPass && std::abs(e.et) <= kPass;
        ev.push_back(e);
    }
    // 정렬: 원래 image_id 순서(manifest: T 값 오름차순, Y 값 오름차순)
    std::stable_sort(ev.begin(), ev.end(), [&](const Ev& a, const Ev& b) {
        int ka = m[a.i].axis == 'Y', kb = m[b.i].axis == 'Y';
        return ka != kb ? ka < kb : m[a.i].value < m[b.i].value;
    });

    Table pred{{"image_id", "file", "true_axis", "true_value", "pred_axis", "pred_value", "pred_Y", "pred_T", "abs_error_Y", "abs_error_T", "branch",
                "axis_sign_correct", "passes", "valid_points", "normalized_cost_margin"}};
    std::vector<double> ty, py, tt, pt, eY, eT;
    std::vector<std::string> names;
    int n_ok = 0, n_pass = 0;
    std::map<char, std::vector<double>> abs_err;
    std::map<char, int> axis_ok, axis_n;
    for (size_t k = 0; k < ev.size(); ++k) {
        auto& e = ev[k];
        auto& it = m[e.i];
        pred.add({std::to_string(k + 1), it.file, std::string(1, it.axis), fmt(it.value, 2), std::string(1, e.p.axis), fmt(e.p.value, 10), fmt(e.py, 10), fmt(e.pt, 10),
                  fmt(std::abs(e.ey), 10), fmt(std::abs(e.et), 10), e.p.branch, e.ok ? "True" : "False", e.pass ? "True" : "False", std::to_string(e.valid), fmt(e.margin, 6)});
        ty.push_back(e.ty); py.push_back(e.py); tt.push_back(e.tt); pt.push_back(e.pt);
        eY.push_back(e.ey); eT.push_back(e.et);
        names.push_back(it.file.substr(0, it.file.size() - 4));
        n_ok += e.ok;
        n_pass += e.pass;
        abs_err[it.axis].push_back(it.axis == 'Y' ? std::abs(e.ey) : std::abs(e.et));
        axis_ok[it.axis] += e.ok;
        ++axis_n[it.axis];
    }
    pred.csv(out("predictions.csv"));
    size_t wy = worst(eY, 1)[0], wt = worst(eT, 1)[0];
    double maxY = max_abs(eY), maxT = max_abs(eT);
    double maeY = mean(abs_err['Y']), maeT = mean(abs_err['T']);

    // ---- 그림 ----
    pred_vs_truth("01_280장_예측오차.png", "Full 280 (resubstitution)", ty, py, tt, pt, names);
    {
        cv::Mat c;
        cv::cvtColor(ref(cv::Range(150, 390), cv::Range(190, 940)), c, cv::COLOR_GRAY2BGR);
        cv::resize(c, c, cv::Size(), 1.6, 1.6, cv::INTER_NEAREST);
        for (int k = 0; k < 3; ++k) {
            cv::Rect b((int)std::lround((cx::BOXES[k][0] - cx::CROP[0] - 190) * 1.6), (int)std::lround((cx::BOXES[k][1] - cx::CROP[1] - 150) * 1.6),
                       (int)std::lround((cx::BOXES[k][2] - cx::BOXES[k][0]) * 1.6), (int)std::lround((cx::BOXES[k][3] - cx::BOXES[k][1]) * 1.6));
            cv::rectangle(c, b, kGreen, 2);
        }
        for (int i = 0; i < NP; ++i)
            cv::circle(c, cv::Point((int)std::lround((ex.P.p[i].x - 190) * 1.6), (int)std::lround((ex.P.p[i].y - 150) * 1.6)), 3, color(ex.P.region[i] * 2 + 1), -1, cv::LINE_AA);
        text(c, "center.bmp: " + std::to_string(NP) + " points (green = QR boxes)", {8, 24}, 0.6, kBlack, 2);
        save_png("02_특징점205.png", c);
    }
    {
        std::vector<double> v[2], mg[2];
        for (auto& e : ev) { int a = m[e.i].axis == 'T'; v[a].push_back(m[e.i].value); mg[a].push_back(e.margin); }
        Chart c("Normalized cost margin (other axis vs chosen)", "true value (Y mm / T deg)", "margin");
        c.scatter(v[0], mg[0], color(2), "Y").scatter(v[1], mg[1], color(1), "T").hline(0);
        c.save("03_비용여유.png");
    }
    // 튀는 값 그림: 세 QR 상자. 초록 = center 위치, 빨강 = 이 사진의 정수 이동 + 점 평균 이동
    auto outlier_fig = [&](const Ev& e, const std::string& file, const std::string& cap) {
        cv::Mat im = read_color(*rows[e.i].s);
        std::vector<cv::Mat> tiles;
        for (int k = 0; k < 3; ++k) {
            cv::Point2d md(0, 0);
            int n = 0;
            for (int j = 0; j < NP; ++j) if (ex.P.region[j] == k) { md += cv::Point2d(res[e.i].D[j].x, res[e.i].D[j].y); ++n; }
            md /= (double)n;
            cv::Rect2d refb(cx::BOXES[k][0], cx::BOXES[k][1], cx::BOXES[k][2] - cx::BOXES[k][0], cx::BOXES[k][3] - cx::BOXES[k][1]);
            cv::Rect2d now(refb.x + md.x, refb.y + md.y, refb.width, refb.height);
            cv::Rect roi = cv::Rect(refb | now) + cv::Size(40, 40) - cv::Point(20, 20);
            std::vector<Mark> mk;
            for (int j = 0; j < NP; ++j)
                if (ex.P.region[j] == k) {
                    cv::Point2d d(res[e.i].D[j].x, res[e.i].D[j].y);
                    mk.push_back({pts[j] + d, (d - md), "", kRed});
                }
            tiles.push_back(zoom(im, roi, 460, mk, 200, k == 0 ? cap : "",
                                 {{refb, kGreen, ""}, {now, kRed, "QR" + std::to_string(k + 1) + " mean (" + fmts(md.x, 2) + ", " + fmts(md.y, 3) + ")"}}));
        }
        save_png(file, hstack(tiles));
    };
    std::string fy = "04_튀는값_Y_" + names[wy] + ".png", ft = "05_튀는값_T_" + names[wt] + ".png";
    outlier_fig(ev[wy], fy, names[wy] + " Y err " + fmts(eY[wy], 4) + " mm");
    outlier_fig(ev[wt], ft, names[wt] + " T err " + fmts(eT[wt], 4) + " deg");

    Table outl{{"구분", "사진", "정답 Y (mm)", "정답 T (deg)", "추정 Y", "추정 T", "오차", "고른 후보", "비용여유", "평균 가로 이동 c0x (px)", "가로 기울기 c1x (px)", "세로 c0y (px)"}};
    for (auto [lab, idx] : std::vector<std::pair<std::string, std::vector<size_t>>>{{"Y 오차 상위", worst(eY, 5)}, {"T 오차 상위", worst(eT, 3)}})
        for (size_t k : idx) {
            auto& e = ev[k];
            bool isY = lab[0] == 'Y';
            outl.add({lab, names[k], fmt(e.ty, 2), fmt(e.tt, 2), fmt(e.py, 4), fmt(e.pt, 4), fmts(isY ? eY[k] : eT[k], 4), e.p.branch, fmt(e.margin, 3),
                      fmt(e.X[0], 3), fmt(e.X[2], 4), fmt(e.X[1], 4)});
        }
    // 같은 구간 이웃과 비교: Y 최대 오차 사진의 앞뒤 단계
    Table nb{{"사진", "정답 Y (mm)", "추정 Y (mm)", "오차 (mm)", "c0x (px)", "고른 후보"}};
    {
        auto& w = ev[wy];
        for (auto& e : ev)
            if (m[e.i].session == m[w.i].session && std::abs(m[e.i].value - m[w.i].value) <= 0.21)
                nb.add({m[e.i].file, fmt(e.ty, 2), fmt(e.py, 4), fmts(e.ey, 4), fmt(e.X[0], 3), e.p.branch});
    }

    // ---- 원래 결과와 대조 (full280/summary.json, predictions.csv, measurement/gn17_spread14_metadata.json, calibrated_runs/model.json) ----
    Parity par;
    par.add("특징점 수", 205, NP, 0, 0);
    par.add("QR1 / QR2 / QR3 점 수 (69/66/70)", 696670, cnt[0] * 10000.0 + cnt[1] * 100 + cnt[2], 0, 0);
    par.add("유효하지 않은 점 (281장 x 205점)", 0, invalid, 0, 0);
    par.add("축·부호 정답 (/280)", 280, n_ok, 0, 0);
    par.add("통과 (/280)", 280, n_pass, 0, 0);
    par.add("Y 최대 |오차| (mm)", 0.03624133380489092, maxY, 0.001, 6);
    par.add("T 최대 |오차| (deg)", 0.002864345823411685, maxT, 0.0002, 6);
    par.add("Y 평균 |오차| (mm)", 0.0016134512773842834, maeY, 0.0002, 6);
    par.add("T 평균 |오차| (deg)", 0.00036654332440734364, maeT, 0.00005, 6);
    par.add_text("Y 최대 오차 사진", "Y-0.80.bmp", m[ev[wy].i].file, m[ev[wy].i].file == "Y-0.80.bmp");
    par.add_text("T 최대 오차 사진", "T+0.01.bmp", m[ev[wt].i].file, m[ev[wt].i].file == "T+0.01.bmp");
    auto pv = [&](const std::string& f) { for (auto& e : ev) if (m[e.i].file == f) return e; throw std::runtime_error(f); };
    const std::vector<std::tuple<std::string, double, double>> sample = {
        {"Y-0.80.bmp", -0.7637586661951091, 0.7578918184918929}, {"T+0.01.bmp", 0.012864345823411685, 0.12048916949491213},
        {"T-0.01.bmp", -0.01013885900587407, 0.3077531943767499}, {"Y+0.05.bmp", 0.05076487069748834, 0.7261622240607284},
        {"Y-0.05.bmp", -0.048022267405550524, 0.22189360292991822}, {"Y+2.00.bmp", 1.998853859179789, 0.9798754864507321},
        {"T-1.00.bmp", -0.9998403573364588, 0.9939368452740414}};
    for (auto& [f, v, mg] : sample) {
        auto e = pv(f);
        par.add(f + " 추정값", v, e.p.value, std::abs(v) > 0.5 ? 0.001 : 0.0003, 6);
        par.add(f + " 비용여유", mg, e.margin, 0.01, 4);
    }
    {
        double mn = 1e9;
        std::string mf;
        for (auto& e : ev) if (e.margin < mn) { mn = e.margin; mf = m[e.i].file; }
        par.add("최소 비용여유", 0.0579298705306503, mn, 0.01, 4);
        par.add_text("최소 비용여유 사진", "T-0.05.bmp", mf, mf == "T-0.05.bmp");
    }
    // model.json: Y 기울기 첫 값, T 기울기 첫 값, W[0][0]
    double sY = 0, sT = 0;
    for (auto& [n, b] : model.branches) (b.axis == 'Y' ? sY : sT) = b.s[0];
    par.add("모델 Y 기울기 c0x (px/mm)", 14.384046932582397, sY, 0.01, 4);
    par.add("모델 T 기울기 c0x (px/deg)", 80.37405859438309, sT, 0.02, 4);
    par.add("모델 W[0][0]", 26.500114831744874, model.W.at<double>(0, 0), 0.05, 4);

    // ---- 결과.md ----
    Table axes{{"축", "장수", "축·부호 정답", "통과", "평균 |오차|", "최대 |오차|", "최대 오차 사진"}};
    for (char a : {'Y', 'T'}) {
        auto& v = abs_err[a];
        size_t w = std::max_element(v.begin(), v.end()) - v.begin();
        int npass = 0;
        std::string wf;
        int k = 0;
        for (auto& e : ev)
            if (m[e.i].axis == a) { npass += e.pass; if (k++ == (int)w) wf = m[e.i].file; }
        axes.add({std::string(1, a), std::to_string(axis_n[a]), std::to_string(axis_ok[a]), std::to_string(npass), fmt(mean(v), 6), fmt(v[w], 6), wf});
    }
    Table ses{{"구간", "장수", "평균 |오차|", "최대 |오차|", "고른 후보가 같은 구간인 장수"}};
    for (auto s : {"Y1+", "Y1-", "Y2+", "Y2-", "T+", "T-"}) {
        std::vector<double> v;
        int same = 0;
        for (auto& e : ev)
            if (m[e.i].session == s) { v.push_back(m[e.i].axis == 'Y' ? std::abs(e.ey) : std::abs(e.et)); same += e.p.branch == s; }
        ses.add({s, std::to_string(v.size()), fmt(mean(v), 6), fmt(max_abs(v), 6), std::to_string(same)});
    }
    std::ostringstream R;
    R << "# 결과: 280장 전수 고정 모델 재실행\n\n"
      << "입력: `data/clean/clean_20261014` 281장 (center 1, T 200, Y 80). 측정은 gn17_spread14 (center 에서 QR 세 상자마다 Shi-Tomasi 간격 14 → " << NP
      << "점, 상자 전체 NCC ±110x22 → LK 15px σ0.8 → GN17 3차 스플라인 + 밝기 이득·오프셋). 사진마다 점 이동을 판 좌표 2차식(6항 x dx/dy = 12값)으로 요약했다.\n\n"
      << "모델: 원래 `calibrated_runs/model.json`을 읽지 않고 같은 절차로 다시 학습했다. 축마다 12값 ~ [v, v², 구간 원-핫 6개] 최소제곱, 잔차 상관으로 백색화, 후보 6개(Y1±, Y2±, T±)마다 v 를 가우스-뉴턴 8회로 풀고 부호가 맞는 후보 중 비용 최소를 고른다. "
      << "학습은 280장 전부라서 이 평가는 학습 데이터 재대입이다.\n\n"
      << "추론은 center 와 현재 사진만 받고 유효 점만으로 12값을 맞춘다. 정답은 예측 뒤에만 비교한다.\n\n"
      << "## 축별\n\n" << axes.md() << "\n## 구간별\n\n" << ses.md()
      << "\n## 원래 결과와 대조 (" << par.ok << "/" << par.total << " 일치)\n\n" << par.t.md()
      << "\n## 옮기지 않은 부분\n\n- 원래 `evaluate_all.py`는 사진을 작업 폴더에 복사한 뒤 `infer.run`을 4병렬로 불렀다. 여기서는 같은 함수 흐름을 한 프로그램 안에서 부른다(파일 해시 확인 생략).\n"
      << "- `visuals/`(QR 특징점 이동 20장 그림·zip)와 `종합/` 아래 발표·설명 자료는 옮기지 않았다.\n\n"
      << "## 파일\n\n- `predictions.csv`: 280장 예측 (원래 predictions.csv 와 같은 순서)\n- `이미지/`: 그림 5장\n";
    write_text(out("결과.md"), R.str());

    // ---- 보고서.md ----
    // 두 후보의 0점 차이 (c0x 기준, 단위 mm 또는 deg): 고른 후보로 읽으면 참 후보보다 이만큼 옮겨진다
    auto branch_off = [&](const std::string& a, const std::string& b) {
        const cx::Branch *A = nullptr, *Bb = nullptr;
        for (auto& nb2 : model.branches) { if (nb2.first == a) A = &nb2.second; if (nb2.first == b) Bb = &nb2.second; }
        return (A && Bb) ? (Bb->base[0] - A->base[0]) / A->s[0] : kNaN;
    };
    auto& EY = ev[wy];
    auto& ET = ev[wt];
    std::ostringstream P;
    P << "# 보고서: 280장 전수 고정 모델 재실행 (10-14 17:17)\n\n"
      << "## 문제\n\n고정된 배포 모델을 센터와 현재 사진 두 장만 넣어 280장 전체에 돌렸을 때 결과를 확인하려 했다.\n\n"
      << "## 해결 방법\n\n6후보 0점 2차 모델을 280장으로 다시 학습해 고정하고, 사진마다 center 와 그 사진만 넣어 예측한 뒤 정답과 비교했다.\n\n"
      << "## 결과\n\n" << n_pass << "/280장이 통과했고 최대 오차는 Y " << fmt(maxY, 4) << " mm, T " << fmt(maxT, 4) << " deg지만, 학습에 쓴 사진을 다시 맞힌 값이라 새 사진 성능은 아니다.\n\n"
      << "| 지표 | 값 |\n|---|---|\n"
      << "| 축·부호 정답 | " << n_ok << " / 280 |\n| 통과 (축·부호 + 두 축 \\|오차\\| <= 0.09) | " << n_pass << " / 280 |\n"
      << "| Y 평균 / 최대 \\|오차\\| (mm) | " << fmt(maeY, 5) << " / " << fmt(maxY, 5) << " |\n"
      << "| T 평균 / 최대 \\|오차\\| (deg) | " << fmt(maeT, 5) << " / " << fmt(maxT, 5) << " |\n"
      << "| 최소 비용여유 | " << fmt(std::min_element(ev.begin(), ev.end(), [](const Ev& a, const Ev& b) { return a.margin < b.margin; })->margin, 3) << " |\n\n"
      << img_md("01_280장_예측오차.png", "예측 오차") << "\n\n" << img_md("02_특징점205.png", "특징점") << "\n\n" << img_md("03_비용여유.png", "비용 여유") << "\n\n"
      << "## 튀는 값\n\nY는 " << m[EY.i].file << "(정답 " << fmt(EY.ty, 2) << ", 추정 " << fmt(EY.py, 4) << " mm)가 가장 크게 틀렸고, " << m[EY.i].session << " 구간 사진인데 " << EY.p.branch
      << " 후보를 골라 두 후보의 0점 차이 " << fmts(branch_off(EY.p.branch, m[EY.i].session), 4) << " mm가 그대로 오차가 됐다(평균 이동 c0x " << fmt(EY.X[0], 3) << " px 자체는 이웃 사진과 같은 간격이다). T는 " << m[ET.i].file << "(정답 " << fmt(ET.tt, 2) << ", 추정 " << fmt(ET.pt, 4)
      << " deg)이고 오차가 허용치의 " << fmt(std::abs(eT[wt]) / kPass * 100, 1) << "%라 문제 되지 않는다.\n\n"
      << outl.md() << "\n" << m[EY.i].file << " 주변 같은 구간(" << m[EY.i].session << ") 사진:\n\n" << nb.md() << "\n"
      << img_md(fy, "Y 튀는 값") << "\n\n" << img_md(ft, "T 튀는 값")
      << "\n\n초록 상자는 center 의 QR 상자, 빨간 상자는 이 사진에서 잰 점 평균 이동만큼 옮긴 상자, 빨간 십자는 측정한 점, 화살표는 점 이동에서 QR 평균을 뺀 나머지 x200이다.\n\n"
      << "## 한줄평\n\n이 모델은 사진마다 구간 6개 중 하나의 0점과 2차 곡선을 고르는 방식이라 학습 사진에서는 구간 곡선이 사진을 거의 그대로 지나가서 Y 평균 오차 " << fmt(maeY, 4)
      << " mm까지 맞지만, 새 촬영처럼 0점이 다른 사진에서는 이 정확도를 보장하지 않는다.\n\n"
      << "## 원래 결과와 대조\n\n원래 값은 `codex_experiments_20261014/full280/summary.json`, `full280/predictions.csv`, `measurement/gn17_spread14_metadata.json`, `calibrated_runs/model.json`에서 가져왔다. "
      << par.ok << "/" << par.total << " 일치.\n\n" << par.t.md();
    write_text(out("보고서.md"), P.str());
    std::cout << "pass " << n_pass << " maxY " << maxY << " maxT " << maxT << " parity " << par.ok << "/" << par.total << "\n";
    return 0;
}
