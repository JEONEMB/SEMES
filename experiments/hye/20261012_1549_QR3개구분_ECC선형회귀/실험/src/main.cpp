// QR 세 영역 ECC 정합 -> 평균 수평 이동·가운데 굽음 두 특징 -> 선형 회귀로 Y·T 추정 (원본: analysis_agent/measure_qr.py, evaluate.py, evaluate_theta_only.py)
#include <hye/dataset.hpp>
#include <hye/ecc_compat.hpp>
#include <hye/plot.hpp>
#include <hye/report.hpp>
#include <opencv2/video/tracking.hpp>
#include <array>
#include <map>
#include <tuple>
#include <iostream>

using namespace hye;

namespace {

// 기준 사진의 QR 영역 (x0, x1, y0, y1)
const std::array<cv::Vec4i, 3> kBoxes{{{1554, 1734, 972, 1164}, {1815, 1995, 972, 1164}, {2073, 2253, 972, 1164}}};
const int kPad = 130;

struct QR {
    double dx = 0, dy = 0, a00 = 1, a01 = 0, a10 = 0, a11 = 1, cc = kNaN, match = kNaN;
    double cx() const { return dx + (a00 - 1) * 90 + a01 * 96; }  // QR 중심의 수평 이동 (px)
    double cy() const { return dy + a10 * 90 + (a11 - 1) * 96; }
};

std::array<QR, 3> measure(const cv::Mat& center, const cv::Mat& im) {
    std::array<QR, 3> out;
    for (int q = 0; q < 3; ++q) {
        auto b = kBoxes[q];
        cv::Mat tpl = center(cv::Range(b[2], b[3]), cv::Range(b[0], b[1]));
        cv::Mat search = im(cv::Range(b[2] - kPad, b[3] + kPad), cv::Range(b[0] - kPad, b[1] + kPad));
        cv::Mat res;
        cv::matchTemplate(search, tpl, res, cv::TM_CCOEFF_NORMED);
        double score;
        cv::Point loc;
        cv::minMaxLoc(res, nullptr, &score, nullptr, &loc);
        int dx0 = loc.x - kPad, dy0 = loc.y - kPad;
        cv::Mat target = im(cv::Range(b[2] + dy0, b[3] + dy0), cv::Range(b[0] + dx0, b[1] + dx0));
        cv::Mat W = cv::Mat::eye(2, 3, CV_32F);
        double cc = kNaN;
        try {
            cc = semes::ecc5::findTransformECCWithMask(tpl, target, cv::noArray(), cv::noArray(), W, cv::MOTION_AFFINE,
                                                      cv::TermCriteria(cv::TermCriteria::EPS | cv::TermCriteria::COUNT, 150, 1e-7), 3);
        } catch (const cv::Exception&) {
        }
        auto& r = out[q];
        r.dx = dx0 + W.at<float>(0, 2);
        r.dy = dy0 + W.at<float>(1, 2);
        r.a00 = W.at<float>(0, 0); r.a01 = W.at<float>(0, 1);
        r.a10 = W.at<float>(1, 0); r.a11 = W.at<float>(1, 1);
        r.cc = cc;
        r.match = score;
    }
    return out;
}

// 최소제곱 [1, x...] -> 여러 목표. Ridge(alpha=0)+표준화와 같은 해
cv::Mat ols(const std::vector<std::vector<double>>& X, const std::vector<std::vector<double>>& Y) {
    cv::Mat A((int)X.size(), (int)X[0].size() + 1, CV_64F), B((int)Y.size(), (int)Y[0].size(), CV_64F);
    for (int i = 0; i < A.rows; ++i) {
        A.at<double>(i, 0) = 1;
        for (int j = 0; j < (int)X[i].size(); ++j) A.at<double>(i, j + 1) = X[i][j];
        for (int j = 0; j < B.cols; ++j) B.at<double>(i, j) = Y[i][j];
    }
    cv::Mat C;
    cv::solve(A, B, C, cv::DECOMP_SVD);
    return C;
}

std::pair<double, double> polyfit1(const std::vector<double>& x, const std::vector<double>& y) {
    double mx = mean(x), my = mean(y), sxy = 0, sxx = 0;
    for (size_t i = 0; i < x.size(); ++i) { sxy += (x[i] - mx) * (y[i] - my); sxx += (x[i] - mx) * (x[i] - mx); }
    double k = sxy / sxx;
    return {k, my - k * mx};
}

double sd1(const std::vector<double>& v) {  // pandas std (ddof=1)
    double m = mean(v), s = 0;
    for (double x : v) s += (x - m) * (x - m);
    return std::sqrt(s / (v.size() - 1));
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
    auto shots = load_1014();
    const cv::Mat center = read_gray(center_1014(shots));
    std::vector<std::array<QR, 3>> meas(shots.size());
#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < (int)shots.size(); ++i) meas[i] = measure(center, read_gray(shots[i]));
    std::cout << "measured " << shots.size() << "\n";

    // 원시 측정 저장
    Table raw{{"file", "Y", "T", "q", "dx", "dy", "a00", "a01", "a10", "a11", "cc", "match"}};
    for (size_t i = 0; i < shots.size(); ++i)
        for (int q = 0; q < 3; ++q) {
            auto& r = meas[i][q];
            raw.add({shots[i].rel, fmt(shots[i].y, 2), fmt(shots[i].t, 2), std::to_string(q), fmt(r.dx, 5), fmt(r.dy, 5), fmt(r.a00, 6),
                     fmt(r.a01, 6), fmt(r.a10, 6), fmt(r.a11, 6), fmt(r.cc, 6), fmt(r.match, 6)});
        }
    raw.csv(out("qr_measurements.csv"));

    // 특징: 평균 수평 이동, 가운데 굽음
    struct Row { size_t i; bool isT; double val, shift, bend, x[3], y[3]; int step; };
    std::vector<Row> d;
    for (size_t i = 0; i < shots.size(); ++i) {
        auto& s = shots[i];
        if (s.condition == "center") continue;
        Row r{i, s.condition == "T", s.condition == "T" ? s.t : s.y};
        for (int q = 0; q < 3; ++q) { r.x[q] = meas[i][q].cx(); r.y[q] = meas[i][q].cy(); }
        r.shift = (r.x[0] + r.x[1] + r.x[2]) / 3;
        r.bend = r.x[1] - (r.x[0] + r.x[2]) / 2;
        r.step = (int)std::lround(std::abs(r.val) * (r.isT ? 100 : 20));
        d.push_back(r);
    }
    std::vector<std::vector<double>> Xtr, Ytr;
    for (auto& r : d)
        if (r.step % 2 == 1) { Xtr.push_back({r.shift, r.bend}); Ytr.push_back({r.isT ? r.val : 0, r.isT ? 0 : r.val}); }
    cv::Mat C = ols(Xtr, Ytr);

    struct Ev { const Row* r; double tt, ty, pt, py; };
    std::vector<Ev> ev;
    for (auto& r : d) {
        if (r.step % 2 != 0) continue;
        double pt = C.at<double>(0, 0) + C.at<double>(1, 0) * r.shift + C.at<double>(2, 0) * r.bend;
        double py = C.at<double>(0, 1) + C.at<double>(1, 1) * r.shift + C.at<double>(2, 1) * r.bend;
        ev.push_back({&r, r.isT ? r.val : 0, r.isT ? 0 : r.val, pt, py});
    }
    auto mae = [&](double maxshift, int n_out[1]) {
        std::vector<double> et, ey;
        for (auto& e : ev)
            if (std::abs(e.r->shift) <= maxshift) { et.push_back(std::abs(e.pt - e.tt)); ey.push_back(std::abs(e.py - e.ty)); }
        n_out[0] = (int)et.size();
        return std::make_pair(mean(et), mean(ey));
    };
    Table held{{"file", "true Y", "pred Y", "true T", "pred T", "shift_x", "middle_bend_x"}};
    std::vector<double> ty, py, tt, pt, ey, et;
    std::vector<std::string> names;
    for (auto& e : ev) {
        held.add({shots[e.r->i].rel, fmt(e.ty, 2), fmt(e.py, 4), fmt(e.tt, 2), fmt(e.pt, 4), fmt(e.r->shift, 3), fmt(e.r->bend, 4)});
        ty.push_back(e.ty); py.push_back(e.py); tt.push_back(e.tt); pt.push_back(e.pt);
        ey.push_back(e.py - e.ty); et.push_back(e.pt - e.tt);
        auto nm = shots[e.r->i].rel.substr(shots[e.r->i].rel.find('/') + 1);
        names.push_back(nm.substr(0, nm.rfind('.')));
    }
    held.csv(out("heldout_predictions.csv"));

    Table bands{{"범위 (평균 수평 이동)", "검증 장수", "T 평균 |오차| (deg)", "Y 평균 |오차| (mm)"}};
    std::map<double, std::array<double, 3>> band_val;
    for (double ms : {2.0, 5.0, 10.0, 15.0, 30.0, 80.0, 1e9}) {
        int n[1];
        auto m = mae(ms, n);
        band_val[ms] = {(double)n[0], m.first, m.second};
        bands.add({ms > 1e8 ? "전체" : "|shift| <= " + fmt(ms, 0) + " px", std::to_string(n[0]), fmt(m.first, 4), fmt(m.second, 4)});
    }

    // 축별 기울기
    Table slopes{{"축", "특징", "기울기 (px / 단위)", "절편 (px)", "선형 잔차 표준편차 (px)"}};
    std::map<std::string, double> sl;
    for (bool isT : {true, false})
        for (int f = 0; f < 2; ++f) {
            std::vector<double> x, y;
            for (auto& r : d) if (r.isT == isT) { x.push_back(r.val); y.push_back(f ? r.bend : r.shift); }
            auto [k, b] = polyfit1(x, y);
            std::vector<double> res;
            for (size_t i = 0; i < x.size(); ++i) res.push_back(y[i] - (k * x[i] + b));
            std::string key = std::string(isT ? "T" : "Y") + (f ? "_bend" : "_shift");
            sl[key] = k;
            sl[key + "_sd"] = sd1(res);
            slopes.add({isT ? "T" : "Y", f ? "middle_bend_x" : "shift_x", fmt(k, 5), fmt(b, 4), fmt(sd1(res), 5)});
        }

    // Y 고정 가정 T 단독: 홀수 단계로 원점 통과 기울기, 짝수 단계 검증
    double sxy = 0, sxx = 0;
    for (auto& r : d) if (r.isT && r.step % 2 == 1) { sxy += r.shift * r.val; sxx += r.shift * r.shift; }
    double kT = sxy / sxx;
    // T 스캔 굽음 직선(기울기·절편)에서 벗어난 양
    double kb = 0, bb = 0;
    {
        std::vector<double> x, y;
        for (auto& r : d) if (r.isT) { x.push_back(r.val); y.push_back(r.bend); }
        std::tie(kb, bb) = polyfit1(x, y);
    }
    auto bend_dev = [&](const Row& r) { return r.bend - (kb * (r.isT ? r.val : 0) + bb); };
    Table tonly{{"범위", "검증 장수", "T 평균 |오차| (deg)", "T 최대 |오차| (deg)", "QR 이동 환산 Y 평균 (mm)"}};
    std::map<std::string, std::array<double, 3>> to_val;
    std::vector<double> to_t, to_e;
    for (auto [lab, lim] : std::vector<std::pair<std::string, double>>{{"전체", 9}, {"|T| <= 0.25", 0.25}, {"|T| <= 0.10", 0.10}}) {
        std::vector<double> e;
        for (auto& r : d)
            if (r.isT && r.step % 2 == 0 && std::abs(r.val) <= lim + 1e-9) {
                e.push_back(std::abs(kT * r.shift - r.val));
                if (lim > 1) { to_t.push_back(r.val); to_e.push_back(kT * r.shift - r.val); }
            }
        to_val[lab] = {(double)e.size(), mean(e), max_abs(e)};
        tonly.add({lab, std::to_string(e.size()), fmt(mean(e), 4), fmt(max_abs(e), 4), fmt(mean(e) * 80.06 / 14.25, 4)});
    }

    // ---- 그림 ----
    {
        std::vector<double> xs[2], ys[2];
        for (auto& r : d) { xs[r.isT].push_back(r.shift); ys[r.isT].push_back(r.bend); }
        Chart c("Features: mean QR shift vs middle QR bend", "mean QR horizontal shift (px)", "middle QR bend (px)");
        c.scatter(xs[1], ys[1], color(1), "T scan").scatter(xs[0], ys[0], color(2), "Y scan").hline(0);
        c.save("01_특징량_이동과굽음.png");
    }
    pred_vs_truth("02_검증140장_예측오차.png", "Held-out 140", ty, py, tt, pt, names);
    {
        Chart c("T only (Y fixed): held-out T error", "true T (deg)", "pred - true (deg)");
        c.scatter(to_t, to_e, color(1)).hline(0).hline(0.01, kRed).hline(-0.01, kRed);
        c.save("03_T단독_오차.png");
    }
    // 튀는 값: Y, T 각각 최대 오차 사진의 세 QR. 초록 = 기준 사진의 QR 영역, 빨강 = 이 사진에서 잰 QR 영역
    auto outlier_fig = [&](size_t k, const std::string& file, const std::string& cap) {
        auto& e = ev[k];
        cv::Mat im = read_color(shots[e.r->i]);
        std::vector<cv::Mat> tiles;
        for (int q = 0; q < 3; ++q) {
            auto b = kBoxes[q];
            cv::Rect2d ref(b[0], b[2], b[1] - b[0], b[3] - b[2]);
            cv::Rect2d now(ref.x + e.r->x[q], ref.y + e.r->y[q], ref.width, ref.height);
            cv::Rect roi = cv::Rect(ref | now) + cv::Size(60, 60) - cv::Point(30, 30);
            tiles.push_back(zoom(im, roi, 420, {}, 0, q == 0 ? cap : "",
                                 {{ref, kGreen, ""}, {now, kRed, "q" + std::to_string(q) + " dx " + fmt(e.r->x[q], 2) + " dy " + fmt(e.r->y[q], 2)}}));
        }
        save_png(file, hstack(tiles));
    };
    size_t wy = worst(ey, 1)[0], wt = worst(et, 1)[0];
    {
        std::vector<double> x, y;
        for (auto& r : d) if (r.isT) { x.push_back(r.val); y.push_back(bend_dev(r)); }
        Chart c("T scan: middle bend minus straight line", "true T (deg)", "bend deviation (px)");
        c.scatter(x, y, color(1)).hline(0).hline(0.0024, kRed).hline(-0.0024, kRed);
        for (size_t k : worst(ey, 3)) if (ev[k].r->isT) c.note(ev[k].r->val, bend_dev(*ev[k].r), names[k]);
        c.save("06_굽음_직선편차.png");
    }
    outlier_fig(wy, "04_튀는값_Y_" + names[wy] + ".png", names[wy] + " Y err " + fmts(ey[wy], 3) + " mm");
    outlier_fig(wt, "05_튀는값_T_" + names[wt] + ".png", names[wt] + " T err " + fmts(et[wt], 3) + " deg");

    // 튀는 값 표
    Table outl{{"구분", "사진", "정답 Y (mm)", "추정 Y", "정답 T (deg)", "추정 T", "q0 / q1 / q2 수평 이동 (px)", "가운데 굽음 (px)", "굽음의 T 직선 대비 편차 (px)"}};
    for (auto [lab, idx] : std::vector<std::pair<std::string, std::vector<size_t>>>{{"Y 오차 상위", worst(ey, 5)}, {"T 오차 상위", worst(et, 3)}})
        for (size_t k : idx) {
            auto& e = ev[k];
            outl.add({lab, names[k], fmt(e.ty, 2), fmt(e.py, 3), fmt(e.tt, 2), fmt(e.pt, 3),
                      fmt(e.r->x[0], 2) + " / " + fmt(e.r->x[1], 2) + " / " + fmt(e.r->x[2], 2), fmt(e.r->bend, 3), fmts(bend_dev(*e.r), 4)});
        }
    // 작은 Y 의 부호 오류
    Table sign{{"사진", "정답 Y (mm)", "추정 Y (mm)", "평균 수평 이동 (px)", "가운데 굽음 (px)"}};
    int sign_err = 0;
    for (auto& e : ev)
        if (e.ty != 0 && std::abs(e.ty) <= 0.5 && e.py * e.ty < 0) {
            ++sign_err;
            sign.add({shots[e.r->i].rel, fmt(e.ty, 2), fmt(e.py, 4), fmt(e.r->shift, 3), fmt(e.r->bend, 4)});
        }

    // ---- 원래 결과와 대조 ----
    Parity par;
    par.add("검증 140장 T 평균 |오차|", 0.022545, band_val[1e9][1], 0.0005);
    par.add("검증 140장 Y 평균 |오차| (mm)", 0.128048, band_val[1e9][2], 0.002);
    par.add("|shift|<=10px 검증 장수", 26, band_val[10][0], 0, 0);
    par.add("|shift|<=10px T 평균 |오차|", 0.0154, band_val[10][1], 0.0005);
    par.add("|shift|<=10px Y 평균 |오차| (mm)", 0.0943, band_val[10][2], 0.001);
    par.add("T 스캔 평균 이동 기울기 (px/deg)", 80.06, sl["T_shift"], 0.01, 2);
    par.add("Y 스캔 평균 이동 기울기 (px/mm)", 14.25, sl["Y_shift"], 0.01, 2);
    par.add("T 스캔 굽음 기울기 (px/deg)", -0.2377, sl["T_bend"], 0.0005);
    par.add("Y 스캔 굽음 기울기 (px/mm)", 0.00788, sl["Y_bend"], 0.0002, 5);
    par.add("T 스캔 굽음 잔차 표준편차 (px)", 0.0088, sl["T_bend_sd"], 0.0002);
    par.add("Y 스캔 굽음 잔차 표준편차 (px)", 0.0056, sl["Y_bend_sd"], 0.0002);
    par.add("T 단독 검증 100장 평균 |오차|", 0.007978, to_val["전체"][1], 0.0002);
    par.add("T 단독 |T|<=0.10 평균 |오차|", 0.006182, to_val["|T| <= 0.10"][1], 0.0002);

    // ---- 결과.md ----
    std::ostringstream R;
    R << "# 결과: QR 세 영역 ECC + 선형 회귀\n\n"
      << "입력: `data/clean/clean_20261014` 281장 (center 1, T 200, Y 80). 기준 사진 center.bmp의 QR 세 영역(180x192 px)을 템플릿 매칭 후 ECC affine으로 정합했다.\n"
      << "학습: 파일명 단계(T는 0.01, Y는 0.05 단위)가 홀수인 140장, 검증: 짝수인 140장. 모델은 [1, 평균 수평 이동, 가운데 굽음]의 최소제곱이다.\n\n"
      << "## 검증 오차\n\n" << bands.md() << "\n## 축별 반응 기울기\n\n" << slopes.md()
      << "\n## T 단독 (Y 고정 가정, 원점 통과 기울기 " << fmt(kT, 6) << " deg/px)\n\n" << tonly.md()
      << "\n## 작은 Y(|Y|<=0.5mm)에서 부호가 뒤집힌 사진 " << sign_err << "장\n\n" << sign.md()
      << "\n## 원래 결과와 대조 (" << par.ok << "/" << par.total << " 일치)\n\n" << par.t.md()
      << "\n## 파일\n\n- `qr_measurements.csv`: 사진별 QR 세 영역 정합값\n- `heldout_predictions.csv`: 검증 140장 예측\n";
    write_text(out("결과.md"), R.str());

    // ---- 보고서.md ----
    auto& wyE = ev[wy];
    auto& wtE = ev[wt];
    std::ostringstream P;
    P << "# 보고서: QR 세 영역 ECC + 선형 회귀 (10-12 15:49)\n\n"
      << "## 문제\n\nQR 세 개의 이동만으로 Y 이동과 T 회전을 구분할 수 있는지 확인해야 했다.\n\n"
      << "## 해결 방법\n\n세 QR 영역을 기준 사진에 ECC affine으로 정합해 평균 수평 이동과 가운데 QR의 굽음을 재고, 홀수 단계 140장으로 선형 회귀를 맞췄다.\n\n"
      << "## 결과\n\n짝수 단계 검증 140장에서 T 평균 오차 " << fmt(band_val[1e9][1], 4) << " deg, Y 평균 오차 " << fmt(band_val[1e9][2], 3)
      << " mm로, 통과 기준 0.09를 Y가 평균부터 넘었다.\n\n"
      << "| 지표 | 값 |\n|---|---|\n"
      << "| 검증 장수 | " << (int)band_val[1e9][0] << " |\n"
      << "| T 평균 / 최대 \\|오차\\| (deg) | " << fmt(mean(absv(et)), 4) << " / " << fmt(max_abs(et), 4) << " |\n"
      << "| Y 평균 / 최대 \\|오차\\| (mm) | " << fmt(mean(absv(ey)), 4) << " / " << fmt(max_abs(ey), 4) << " |\n"
      << "| Y 통과 (\\|eY\\|<=0.09) | " << std::count_if(ey.begin(), ey.end(), [](double v) { return std::abs(v) <= kPass; }) << " / " << ey.size() << " |\n"
      << "| T 통과 (\\|eT\\|<=0.09) | " << std::count_if(et.begin(), et.end(), [](double v) { return std::abs(v) <= kPass; }) << " / " << et.size() << " |\n"
      << "| T 단독(Y 고정 가정) 평균 \\|오차\\| | " << fmt(to_val["전체"][1], 4) << " deg |\n\n"
      << img_md("01_특징량_이동과굽음.png", "특징량") << "\n\n" << img_md("02_검증140장_예측오차.png", "예측 오차") << "\n\n"
      << "## 튀는 값\n\n가장 크게 틀린 사진은 Y는 " << names[wy] << "(정답 Y " << fmt(wyE.ty, 2) << ", 추정 " << fmt(wyE.py, 3) << " mm), T는 " << names[wt]
      << "(정답 T " << fmt(wtE.tt, 2) << ", 추정 " << fmt(wtE.pt, 3) << " deg)이고, 가운데 굽음이 T 직선에서 " << fmts(bend_dev(*wyE.r), 4)
      << " px 벗어나 T 0.01이 만드는 굽음 신호(0.0024 px)의 약 " << fmt(std::abs(bend_dev(*wyE.r)) / 0.0024, 0) << "배라서 회귀가 그 차이를 Y로 읽었다.\n\n"
      << outl.md() << "\n" << img_md("04_튀는값_Y_" + names[wy] + ".png", "Y 튀는 값") << "\n\n" << img_md("05_튀는값_T_" + names[wt] + ".png", "T 튀는 값")
      << "\n\n" << img_md("06_굽음_직선편차.png", "굽음 직선 편차")
      << "\n\n초록 상자는 기준 사진의 QR 영역, 빨간 상자는 이 사진에서 잰 QR 영역이다. 아래 그림의 빨간 선은 T 0.01이 만드는 굽음 신호 ±0.0024 px다. 작은 Y(|Y|<=0.5mm)에서 부호가 뒤집힌 사진은 " << sign_err << "장이다.\n\n"
      << sign.md() << "\n"
      << "## 한줄평\n\nY 1mm는 평균 이동 " << fmt(sl["Y_shift"], 2) << " px, T 1deg는 " << fmt(sl["T_shift"], 2)
      << " px로 두 축이 같은 방향의 평행 이동을 만들고, 구분 단서인 가운데 굽음은 T 0.01당 " << fmt(std::abs(sl["T_bend"]) / 100, 4)
      << " px로 잔차 " << fmt(sl["T_bend_sd"], 4) << " px보다 작아서, 선형 회귀가 평균 이동을 두 축에 나눠 주는 과정에서 Y가 크게 틀렸다.\n\n"
      << "## 원래 결과와 대조\n\n원래 값은 `analysis_agent/README.md`와 당시 `heldout_predictions.csv`, `theta_only_holdout.csv`에서 가져왔다. "
      << par.ok << "/" << par.total << " 일치.\n\n" << par.t.md();
    write_text(out("보고서.md"), P.str());
    std::cout << "parity " << par.ok << "/" << par.total << "\n";
    return par.ok == par.total ? 0 : 2;
}
