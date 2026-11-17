// v3: QR만 방법을 가우스-뉴턴으로 정밀화. 판 원근 변환 특징(Q5b) x C2~C6 과 G1(QR 판 픽셀 직접 가우스-뉴턴)을
// 같은 짝/홀 2겹 교차검증과 같은 판정(규칙3: 추정 Θ가 짝의 Θ에서 3σ 넘게 떨어지면 구분 가능)으로 비교한다.
// 원본: 00000.legacies/analysis_v3/run_v3.py, gn_features.py, gn_pixel.py, run_g1.py (common.py = v1_data/grid_repro.py)
#include <hye/dataset.hpp>
#include <hye/ecc_compat.hpp>
#include <hye/plot.hpp>
#include <hye/report.hpp>
#include <omp.h>
#include <array>
#include <iostream>
#include <map>
#include <sstream>
#include "g1.hpp"
#include "grid.hpp"

using namespace hye;
using grid::M;

namespace {

// ---- QR 세 영역 ECC affine: 짝 기준 m (QR 3개 가로 평균) 과 G1 출발점 ----
const std::array<cv::Vec4i, 3> kBoxes{{{1554, 1734, 972, 1164}, {1815, 1995, 972, 1164}, {2073, 2253, 972, 1164}}};
const int kPad = 130;
std::array<cv::Point2d, 3> measure_qr(const cv::Mat& center, const cv::Mat& im) {
    std::array<cv::Point2d, 3> out;
    for (int q = 0; q < 3; ++q) {
        auto b = kBoxes[q];
        cv::Mat tpl = center(cv::Range(b[2], b[3]), cv::Range(b[0], b[1]));
        cv::Mat search = im(cv::Range(b[2] - kPad, b[3] + kPad), cv::Range(b[0] - kPad, b[1] + kPad)), res;
        cv::matchTemplate(search, tpl, res, cv::TM_CCOEFF_NORMED);
        cv::Point loc;
        cv::minMaxLoc(res, nullptr, nullptr, nullptr, &loc);
        int dx0 = loc.x - kPad, dy0 = loc.y - kPad;
        cv::Mat target = im(cv::Range(b[2] + dy0, b[3] + dy0), cv::Range(b[0] + dx0, b[1] + dx0));
        cv::Mat W = cv::Mat::eye(2, 3, CV_32F);
        try {
            semes::ecc5::findTransformECCWithMask(tpl, target, cv::noArray(), cv::noArray(), W, cv::MOTION_AFFINE,
                                                  cv::TermCriteria(cv::TermCriteria::EPS | cv::TermCriteria::COUNT, 150, 1e-7), 3);
        } catch (const cv::Exception&) {
        }
        double dx = dx0 + W.at<float>(0, 2), dy = dy0 + W.at<float>(1, 2);
        out[q] = {dx + (W.at<float>(0, 0) - 1) * 90 + W.at<float>(0, 1) * 96, dy + W.at<float>(1, 0) * 90 + (W.at<float>(1, 1) - 1) * 96};
    }
    return out;
}

// ---- 판 호모그래피 (analysis_v2/extract_homography.py) ----
// center QR 중심 = cv2.QRCodeDetectorAruco 꼭짓점 평균 (14_데이터_QR마커위치.csv center 행, cv2 5.0 으로 다시 뽑아도 같음)
const double cLx = 1642.4546, cCx = 1900.4866, cRx = 2159.2039, cLy = 1067.5, cCy = 1068.25, cRy = 1070.5366;
const int X0 = int(cLx - 110), X1 = int(cRx + 110), Y0 = int(std::min(cLy, cRy) - 120), Y1 = int(std::max(cLy, cRy) + 120);
std::array<cv::Point2d, 7> measure_hom(const cv::Mat& tpl, const cv::Mat& im, int sx, int sy) {
    cv::Mat crop;
    im(cv::Range(Y0 + sy, Y1 + sy), cv::Range(X0 + sx, X1 + sx)).convertTo(crop, CV_32F);
    cv::Mat W = cv::Mat::eye(3, 3, CV_32F);
    semes::ecc5::findTransformECCWithMask(tpl, crop, cv::noArray(), cv::noArray(), W, cv::MOTION_HOMOGRAPHY,
                                          cv::TermCriteria(cv::TermCriteria::EPS | cv::TermCriteria::COUNT, 300, 1e-7), 5);
    const cv::Point2f pts[7] = {{(float)X0, (float)Y0}, {(float)X1, (float)Y0}, {(float)X1, (float)Y1}, {(float)X0, (float)Y1},
                                {(float)cLx, (float)cLy}, {(float)cCx, (float)cCy}, {(float)cRx, (float)cRy}};
    std::vector<cv::Point2f> loc, m;
    for (auto& p : pts) loc.push_back(p - cv::Point2f((float)X0, (float)Y0));
    cv::perspectiveTransform(loc, m, W);
    std::array<cv::Point2d, 7> d;
    for (int k = 0; k < 7; ++k) {
        cv::Point2f mm = m[k] + cv::Point2f((float)(X0 + sx), (float)(Y0 + sy));
        d[k] = cv::Point2d(mm.x - pts[k].x, mm.y - pts[k].y);
    }
    return d;
}

// ---- C6: 곡선 모델 f = W^T [Θ, Y, Θ², ΘY, Y², 1] 의 GLS 가중 감쇠 가우스-뉴턴 (gn_features.py), 출발점 = C5 ----
M basis(double t, double y) { return (cv::Mat_<double>(1, 6) << t, y, t * t, t * y, y * y, 1); }
M a_gn(const M& F, const M& X, const std::vector<int>& tr, const std::vector<int>& te) {
    M Xtr = grid::rows_of(X, tr), Ftr = grid::rows_of(F, tr), Bs;
    for (int i = 0; i < Xtr.rows; ++i) Bs.push_back(basis(Xtr.at<double>(i, 0), Xtr.at<double>(i, 1)));
    M W = grid::lstsq(Bs, Ftr), R = Ftr - Bs * W;
    M mu;
    cv::reduce(R, mu, 0, cv::REDUCE_AVG);
    M Rc = R - cv::repeat(mu, R.rows, 1), S = Rc.t() * Rc / (R.rows - 1);
    double jit = cv::trace(S)[0] / S.rows * 1e-2;
    for (int j = 0; j < S.rows; ++j) S.at<double>(j, j) += jit;
    M Si = S.inv(cv::DECOMP_LU);
    M P = grid::predict(5, F, X, tr, te), out(P.rows, 2, CV_64F);
    for (size_t s = 0; s < te.size(); ++s) {
        M f = F.row(te[s]);
        auto cost = [&](double t, double y) { M r = f - basis(t, y) * W; return M(r * Si * r.t()).at<double>(0); };
        double t = P.at<double>((int)s, 0), y = P.at<double>((int)s, 1), lam = 1e-6, c = cost(t, y);
        for (int k = 0; k < 100; ++k) {
            M r = f - basis(t, y) * W;
            M db = (cv::Mat_<double>(6, 2) << 1, 0, 0, 1, 2 * t, 0, y, t, 0, 2 * y, 0, 0);
            M Jm = (db.t() * W).t();
            M A = Jm.t() * Si * Jm, g = Jm.t() * Si * r.t(), st;
            M Ad = A.clone();
            for (int j = 0; j < 2; ++j) Ad.at<double>(j, j) += lam * A.at<double>(j, j);
            cv::solve(Ad, g, st, cv::DECOMP_LU);
            double c2 = cost(t + st.at<double>(0), y + st.at<double>(1));
            if (c2 < c) {
                t += st.at<double>(0); y += st.at<double>(1); c = c2; lam = std::max(lam / 10, 1e-9);
                if (std::max(std::abs(st.at<double>(0)), std::abs(st.at<double>(1))) < 1e-9) break;
            } else {
                lam *= 10;
                if (lam > 1e8) break;
            }
        }
        out.at<double>((int)s, 0) = t;
        out.at<double>((int)s, 1) = y;
    }
    return out;
}

struct Method { std::string feat, alg; M est; double trms, tmax, ymean, yrms, ymax, y009, y005, sT, fp; int nY, nT; std::vector<bool> can; };

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
    omp_set_num_threads(4);
    begin_results();
    // 281행: T -1.00..+1.00 (가운데 center) 다음 Y -2.00..+2.00 (center 중복 제거) — v1 결과표와 같은 순서·분할
    auto shots = load_1014();
    const Shot& cs = center_1014(shots);
    std::vector<const Shot*> Ts, Ys;
    for (auto& s : shots) if (&s != &cs) (s.condition == "T" ? Ts : Ys).push_back(&s);
    Ts.push_back(&cs);
    std::stable_sort(Ts.begin(), Ts.end(), [](auto a, auto b) { return a->t < b->t; });
    std::stable_sort(Ys.begin(), Ys.end(), [](auto a, auto b) { return a->y < b->y; });
    std::vector<const Shot*> rows = Ts;
    rows.insert(rows.end(), Ys.begin(), Ys.end());
    const int n = (int)rows.size();
    M X(n, 2, CV_64F);
    std::vector<std::string> names;
    for (int i = 0; i < n; ++i) {
        X.at<double>(i, 0) = rows[i]->condition == "T" ? rows[i]->t : 0;
        X.at<double>(i, 1) = rows[i]->condition == "Y" ? rows[i]->y : 0;
        auto r = rows[i]->rel.substr(rows[i]->rel.rfind('/') + 1);
        names.push_back(r.substr(0, r.rfind('.')));
    }
    std::vector<int> fa, fb;
    for (int i = 0; i < n; ++i) (i % 2 == 0 ? fa : fb).push_back(i);
    const std::vector<std::pair<std::vector<int>, std::vector<int>>> folds{{fa, fb}, {fb, fa}};

    // ---------- 측정: QR ECC (m), 판 호모그래피 ----------
    const cv::Mat center = read_gray(cs);
    cv::Mat htpl;
    center(cv::Range(Y0, Y1), cv::Range(X0, X1)).convertTo(htpl, CV_32F);
    std::vector<double> m(n), mdy(n);
    std::vector<std::array<cv::Point2d, 7>> hom(n);
#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < n; ++i) {
        cv::Mat g = read_gray(*rows[i]);
        auto q = measure_qr(center, g);
        m[i] = (q[0].x + q[1].x + q[2].x) / 3;
        mdy[i] = (q[0].y + q[1].y + q[2].y) / 3;
        hom[i] = measure_hom(htpl, g, (int)std::nearbyint(m[i]), (int)std::nearbyint(mdy[i]));
    }
    std::cout << "measured\n";
    M Q5b(n, 14, CV_64F), CD(n, 8, CV_64F);
    for (int i = 0; i < n; ++i)
        for (int k = 0; k < 7; ++k) {
            Q5b.at<double>(i, 2 * k) = hom[i][k].x;
            Q5b.at<double>(i, 2 * k + 1) = hom[i][k].y;
            if (k < 4) { CD.at<double>(i, 2 * k) = hom[i][k].x; CD.at<double>(i, 2 * k + 1) = hom[i][k].y; }
        }

    // ---------- 규칙3 짝: 같은 m 의 상대 축 Θ ----------
    std::vector<bool> isY(n), isT(n), isC(n), inr(n);
    std::vector<double> twinT(n, kNaN);
    std::vector<std::pair<double, double>> Tmv, Ymv;
    for (int i = 0; i < n; ++i) {
        double t = X.at<double>(i, 0), y = X.at<double>(i, 1);
        isY[i] = t == 0 && y != 0; isT[i] = y == 0 && t != 0; isC[i] = t == 0 && y == 0;
    }
    for (int i = 0; i < n; ++i) if (isT[i]) Tmv.push_back({m[i], X.at<double>(i, 0)});
    for (int i = 0; i < n; ++i) if (isC[i]) Tmv.push_back({m[i], 0.0});
    for (int i = 0; i < n; ++i) if (isY[i]) Ymv.push_back({m[i], 0.0});
    for (int i = 0; i < n; ++i) if (isC[i]) Ymv.push_back({m[i], 0.0});
    std::sort(Tmv.begin(), Tmv.end());
    std::sort(Ymv.begin(), Ymv.end());
    {
        std::vector<double> xp, fp;
        for (auto& p : Tmv) { xp.push_back(p.first); fp.push_back(p.second); }
        for (int i = 0; i < n; ++i) {
            if (isY[i]) {
                double x = m[i];
                if (x <= xp.front()) twinT[i] = fp.front();
                else if (x >= xp.back()) twinT[i] = fp.back();
                else {
                    size_t j = std::upper_bound(xp.begin(), xp.end(), x) - xp.begin();
                    twinT[i] = fp[j - 1] + (x - xp[j - 1]) / (xp[j] - xp[j - 1]) * (fp[j] - fp[j - 1]);
                }
            }
            inr[i] = isT[i] && m[i] >= Ymv.front().first && m[i] <= Ymv.back().first;
            if (inr[i]) twinT[i] = 0;
        }
    }
    auto evaluate = [&](const std::string& f, const std::string& a, const M& est) {
        Method r{f, a, est};
        M e = est - X;
        std::vector<double> eT, eY, sT;
        for (int i = 0; i < n; ++i) { eT.push_back(std::abs(e.at<double>(i, 0))); eY.push_back(std::abs(e.at<double>(i, 1))); sT.push_back(e.at<double>(i, 0)); }
        double mT = mean(sT), v = 0;
        for (double x : sT) v += (x - mT) * (x - mT);
        r.sT = std::sqrt(v / n);
        r.trms = rms(eT); r.tmax = max_abs(eT); r.ymean = mean(eY); r.yrms = rms(eY); r.ymax = max_abs(eY);
        int a9 = 0, a5 = 0, fpc = 0;
        r.nY = r.nT = 0;
        r.can.resize(n);
        for (int i = 0; i < n; ++i) {
            a9 += eY[i] <= 0.09; a5 += eY[i] <= 0.05; fpc += eT[i] > 3 * r.sT;
            r.can[i] = std::abs(est.at<double>(i, 0) - twinT[i]) > 3 * r.sT;  // NaN 이면 false
            r.nY += isY[i] && !r.can[i];
            r.nT += inr[i] && !r.can[i];
        }
        r.y009 = 100.0 * a9 / n; r.y005 = 100.0 * a5 / n; r.fp = (double)fpc / n;
        return r;
    };
    std::vector<Method> methods;
    const char* AN[] = {"", "", "C2 OLS", "C3 ridge", "C4 Jacobian LS", "C5 Jacobian GLS", "C6 Gauss-Newton (quadratic)"};
    for (int alg = 2; alg <= 6; ++alg) {
        M est(n, 2, CV_64F);
        for (auto& f : folds) {
            M r = alg == 6 ? a_gn(Q5b, X, f.first, f.second) : grid::predict(alg, Q5b, X, f.first, f.second);
            for (size_t i = 0; i < f.second.size(); ++i) r.row((int)i).copyTo(est.row(f.second[i]));
        }
        methods.push_back(evaluate("Q5b plate homography 14", AN[alg], est));
        std::cout << AN[alg] << " Y RMS " << methods.back().yrms << "\n";
    }

    // ---------- G1 ----------
    g1::Engine eng(X0, X1, Y0, Y1);
    double lin_err = 0;
    for (auto nm : {"T-1.00", "T+1.00", "Y-2.00", "Y+2.00", "T+0.50"}) {
        int i = (int)(std::find(names.begin(), names.end(), nm) - names.begin());
        double d[8];
        for (int k = 0; k < 8; ++k) d[k] = CD.at<double>(i, k);
        lin_err = std::max(lin_err, eng.check_linear(d));
    }
    M g1est(n, 2, CV_64F, cv::Scalar(kNaN)), g1nui(n, 3, CV_64F, cv::Scalar(kNaN));
    std::vector<double> g1cost(n, kNaN);
    std::vector<int> g1it(n, 0);
    for (int fo = 0; fo < 2; ++fo) {
        auto& tr = folds[fo].first;
        auto& te = folds[fo].second;
        M A = grid::ones_col(grid::rows_of(X, tr)), B = grid::lstsq(A, grid::rows_of(CD, tr));  // 3 x 8
        std::vector<cv::Mat> part(4);
        for (auto& p : part) p = cv::Mat::zeros(eng.HH, eng.WW, CV_64F);
#pragma omp parallel for schedule(static)
        for (int t = 0; t < 4; ++t)
            for (size_t s = t; s < tr.size(); s += 4) {
                int i = tr[s];
                double d[8];
                for (int k = 0; k < 8; ++k) d[k] = CD.at<double>(i, k);
                cv::Mat mx, my, w;
                eng.lin_map(d, mx, my);
                g1::warp(eng.prepare(read_gray(*rows[i]), false).I, mx, my).convertTo(w, CV_64F);
                part[t] += w;
            }
        cv::Mat acc = part[0] + part[1] + part[2] + part[3], Tm;  // 고정 순서 합 (스레드 순서와 무관)
        acc /= (double)tr.size();
        acc.convertTo(Tm, CV_32F);
        std::cout << "fold " << fo << " template\n";
#pragma omp parallel for schedule(dynamic)
        for (int s = 0; s < (int)te.size(); ++s) {
            int i = te[s];
            auto img = eng.prepare(read_gray(*rows[i]));
            auto fit = eng.solve_multi(img, B, m[i], Tm);
            g1est.at<double>(i, 0) = fit.p[0];
            g1est.at<double>(i, 1) = fit.p[1];
            for (int j = 0; j < 3; ++j) g1nui.at<double>(i, j) = fit.p[2 + j];
            g1cost[i] = fit.cost;
            g1it[i] = fit.it;
        }
        std::cout << "fold " << fo << " done\n";
    }
    methods.push_back(evaluate("G1 QR plate pixels", "G1 pixel Gauss-Newton (T, Y + 3 nuisance)", g1est));
    auto& G = methods.back();
    auto& c6 = methods[4];
    std::cout << "G1 Y RMS " << G.yrms << " nY " << G.nY << " nT " << G.nT << "\n";

    // ---------- 사진별 CSV ----------
    {
        Table t{{"file", "true_T", "true_Y", "m_px", "twin_T", "G1_T", "G1_Y", "nuis_dy", "nuis_rot", "nuis_scale", "cost", "iter", "G1_구분", "Q5b_C6_T", "Q5b_C6_Y"}};
        for (int i = 0; i < n; ++i)
            t.add({names[i], fmt(X.at<double>(i, 0), 2), fmt(X.at<double>(i, 1), 2), fmt(m[i], 4), fmt(twinT[i], 4), fmt(g1est.at<double>(i, 0), 5), fmt(g1est.at<double>(i, 1), 5),
                   fmt(g1nui.at<double>(i, 0), 5), fmt(g1nui.at<double>(i, 1), 5), fmt(g1nui.at<double>(i, 2), 5), fmt(g1cost[i], 5), std::to_string(g1it[i]),
                   isY[i] || inr[i] ? (G.can[i] ? "가능" : "불가") : "-", fmt(c6.est.at<double>(i, 0), 5), fmt(c6.est.at<double>(i, 1), 5)});
        t.csv(out("사진별_G1_Q5b.csv"));
        Table c{{"file"}};
        for (auto k : {"TL", "TR", "BR", "BL", "QR_L", "QR_C", "QR_R"}) { c.head.push_back(std::string(k) + "_dx"); c.head.push_back(std::string(k) + "_dy"); }
        for (int i = 0; i < n; ++i) {
            std::vector<std::string> r{names[i]};
            for (int k = 0; k < 14; ++k) r.push_back(fmt(Q5b.at<double>(i, k), 4));
            c.add(r);
        }
        c.csv(out("판호모그래피_측정.csv"));
    }

    // ---------- 그림 ----------
    std::vector<double> ty, py, tt, pt, ey, et;
    for (int i = 0; i < n; ++i) {
        ty.push_back(X.at<double>(i, 1)); tt.push_back(X.at<double>(i, 0)); py.push_back(g1est.at<double>(i, 1)); pt.push_back(g1est.at<double>(i, 0));
        ey.push_back(py.back() - ty.back()); et.push_back(pt.back() - tt.back());
    }
    pred_vs_truth("01_G1_추정오차.png", "G1 pixel GN, 2-fold", ty, py, tt, pt, names);
    {
        std::vector<std::string> cats;
        std::vector<double> yv, nv;
        for (auto& mt : methods) {
            std::string a = mt.alg.substr(0, 2);
            cats.push_back(a == "G1" ? "G1 pixel" : "Q5b " + a);
            yv.push_back(mt.yrms * 1000);
            nv.push_back(mt.nY);
        }
        bar_chart("02_방법별_Y오차RMS.png", "Y RMS error by method (um)", cats, yv, "um", (int)methods.size() - 1, 90);
        bar_chart("03_방법별_구분불가Y.png", "Indistinguishable Y photos (rule 3, of 80)", cats, nv, "photos", (int)methods.size() - 1);
    }
    {
        std::vector<double> x[2], y[2];
        for (int i = 0; i < n; ++i)
            if (isY[i]) { bool c = G.can[i]; x[c].push_back(X.at<double>(i, 1)); y[c].push_back(std::abs(g1est.at<double>(i, 0) - twinT[i]) / G.sT); }
        Chart c("G1 rule 3: |est T - partner T| / sigma_T (Y photos)", "true Y (mm)", "distance / sigma");
        c.scatter(x[0], y[0], kRed, "indistinguishable").scatter(x[1], y[1], color(0), "distinguishable").hline(3, kRed);
        c.save("04_G1_규칙3_Y사진.png");
    }
    {
        std::vector<double> x, y0, y1, y2;
        for (int i = 0; i < n; ++i) if (!std::isnan(g1cost[i])) { x.push_back(X.at<double>(i, 0) * 2 + X.at<double>(i, 1)); y0.push_back(g1nui.at<double>(i, 0)); y1.push_back(g1nui.at<double>(i, 1)); y2.push_back(g1nui.at<double>(i, 2)); }
        Chart c("G1 nuisance parameters", "2 T + Y (scan position)", "value");
        c.scatter(x, y0, color(0), "vertical shift (px)").scatter(x, y1, color(1), "rotation (px per 1000 px)").scatter(x, y2, color(2), "scale (px per 1000 px)").hline(0);
        c.save("05_G1_방해요소.png");
    }
    const size_t wy = worst(ey, 1)[0], wt = worst(et, 1)[0];
    auto outlier_fig = [&](size_t i, const std::string& file, const std::string& cap) {
        cv::Mat im = read_color(*rows[i]);
        const cv::Point2d P[4] = {{(double)X0, (double)Y0}, {(double)X1, (double)Y0}, {(double)X1, (double)Y1}, {(double)X0, (double)Y1}};
        const char* nm[4] = {"TL", "TR", "BR", "BL"};
        double mx = 0, my = 0;
        for (int c = 0; c < 4; ++c) { mx += CD.at<double>((int)i, 2 * c) / 4; my += CD.at<double>((int)i, 2 * c + 1) / 4; }
        std::vector<cv::Mat> tiles;
        for (int c = 0; c < 4; ++c) {
            cv::Point2d d(CD.at<double>((int)i, 2 * c), CD.at<double>((int)i, 2 * c + 1)), now = P[c] + d;
            cv::Rect roi((int)now.x - 60, (int)now.y - 60, 120, 120);
            tiles.push_back(zoom(im, roi, 300, {{now, {d.x - mx, d.y - my}, nm[c], kRed}, {P[c] + cv::Point2d(mx, my), {0, 0}, "", kGreen}}, 200, c == 0 ? cap : ""));
        }
        save_png(file, hstack(tiles));
    };
    const std::string fy = "06_튀는값_Y_" + names[wy] + ".png", ft = "07_튀는값_T_" + names[wt] + ".png";
    outlier_fig(wy, fy, names[wy] + " G1 Y err " + fmts(ey[wy], 3) + " mm");
    outlier_fig(wt, ft, names[wt] + " G1 T err " + fmts(et[wt], 4));

    // ---------- 원래 결과와 대조 ----------
    // 출처: 00000.legacies/analysis_v3/v3_results.csv (G1, Q5b 행), 00_v3결과리포트.md 결과 표
    Parity par, full;
    par.add("G1 Y 오차 RMS (µm)", 91.0, G.yrms * 1000, 5, 1);
    par.add("G1 Y 0.09 mm 안 (%)", 69.395, G.y009, 3, 1);
    par.add("G1 구분 불가 Y (/80)", 12, G.nY, 0, 0);
    par.add("G1 구분 불가 Θ", 8, G.nT, 0, 0);
    par.add("G1 Θ RMS", 0.0165, G.trms, 0.001);
    par.add("G1 Y 최대 (mm)", 0.2732, G.ymax, 0.02);
    par.add("Q5b × C6 Y RMS (µm)", 119.0, c6.yrms * 1000, 5, 1);
    par.add("Q5b × C6 구분 불가 Y / Θ", 13 * 100 + 15, c6.nY * 100 + c6.nT, 0, 0);
    const std::vector<std::tuple<std::string, double, double, double, double, int, int>> O{
        // 알고리즘, T_RMS, Y_RMS, Y_최대, Y_009안, Y불가, Θ불가
        {"C2", 0.0224, 0.1236, 0.3628, 51.9573, 17, 15}, {"C3", 0.0215, 0.1225, 0.3044, 49.4662, 12, 13}, {"C4", 0.0978, 0.5510, 1.6495, 18.8612, 80, 53},
        {"C5", 0.0225, 0.1261, 0.3501, 49.1103, 13, 15}, {"C6", 0.0214, 0.1190, 0.3264, 51.2456, 13, 15}, {"G1", 0.0165, 0.0910, 0.2732, 69.3950, 12, 8}};
    for (size_t k = 0; k < O.size(); ++k) {
        auto& [a, t, y, ym, y9, nY, nT] = O[k];
        auto& mt = methods[k];
        std::string lab = (a == "G1" ? "G1" : "Q5b × " + a);
        full.add(lab + " Θ RMS", t, mt.trms, 0.001);
        full.add(lab + " Y RMS (mm)", y, mt.yrms, 0.005);
        full.add(lab + " Y 최대 (mm)", ym, mt.ymax, 0.03);
        full.add(lab + " Y 0.09 안 (%)", y9, mt.y009, 3, 1);
        full.add(lab + " 구분 불가 Y", nY, mt.nY, 0, 0);
        full.add(lab + " 구분 불가 Θ", nT, mt.nT, 0, 0);
    }
    std::cout << "parity " << par.ok << "/" << par.total << " full " << full.ok << "/" << full.total << "\n";

    // ---------- 결과.md ----------
    Table mtab{{"특징", "알고리즘", "Θ RMS", "Θ 최대", "Y 평균 (mm)", "Y RMS (mm)", "Y 최대 (mm)", "Y 0.09 안 %", "Y 0.05 안 %", "구분 불가 Y", "구분 불가 Θ", "오판율", "σΘ"}};
    for (auto& mt : methods)
        mtab.add({mt.feat, mt.alg, fmt(mt.trms, 4), fmt(mt.tmax, 4), fmt(mt.ymean, 4), fmt(mt.yrms, 4), fmt(mt.ymax, 4), fmt(mt.y009, 1), fmt(mt.y005, 1), std::to_string(mt.nY),
                  std::to_string(mt.nT), fmt(mt.fp * 100, 2) + " %", fmt(mt.sT, 4)});
    std::string undY;
    {
        std::vector<std::pair<double, std::string>> l;
        for (int i = 0; i < n; ++i) if (isY[i] && !G.can[i]) l.push_back({X.at<double>(i, 1), names[i]});
        std::sort(l.begin(), l.end());
        for (auto& p : l) undY += (undY.empty() ? "" : ", ") + p.second;
    }
    double edgeY = 0;
    for (int i = 0; i < n; ++i) if (isY[i] && !G.can[i]) edgeY = std::max(edgeY, std::abs(X.at<double>(i, 1)));
    std::ostringstream Rm;
    Rm << "# 결과: QR만 정밀화, 픽셀 직접 가우스-뉴턴 (v3)\n\n"
       << "입력: `data/clean/clean_20261014` 281장, v1 결과표와 같은 순서(T 묶음 + Y 묶음, center 1장)·짝/홀 2겹 교차검증.\n\n"
       << "측정: 판 창 (" << X0 << ".." << X1 << ", " << Y0 << ".." << Y1 << ") ECC homography로 판 귀퉁이 4개·QR 중심 3개 이동(Q5b 14값, G1의 귀퉁이 8값), QR 세 영역 ECC affine으로 짝 기준 m.\n\n"
       << "G1: 학습 사진 귀퉁이 이동을 [Θ, Y, 1]로 선형 회귀(B), 학습 사진을 center 좌표로 되돌린 평균을 기준 영상으로, 판 영역 " << eng.WW << "x" << eng.HH
       << " 픽셀을 (Θ, Y, 세로 이동, 회전, 크기) + 밝기 a, b로 감쇠 가우스-뉴턴(최대 60회, 출발점 2곳). 보간은 3차 B-스플라인. 선형화 좌표 오차 최대 " << fmt(lin_err, 5) << " px.\n\n"
       << "판정(규칙3): 추정 Θ가 짝(같은 m의 상대 축)의 Θ에서 3σΘ 넘게 떨어지면 구분 가능.\n\n"
       << "## 방법별 결과\n\n" << mtab.md() << "\nG1 구분 불가 Y: " << undY << "\n\n"
       << "## 전체 대조 (" << full.ok << "/" << full.total << " 일치)\n\n원래 값: `00000.legacies/analysis_v3/v3_results.csv`.\n\n" << full.t.md()
       << "\n## 옮기지 않은 부분\n\n"
       << "- v1 결과표 재현(조합 13개 x C1~C5)은 20261012_1524 실험에서 옮겼다. 여기서는 Q5b와 G1만 다시 계산했다. QR 밖 조합(O1~O6)·Q1~Q7의 C6, 짝 기준 8가지 비교(PPT 슬라이드 5~7), 엑셀(25_YΘ구분표_v3.xlsx)·PPT·연결도 그림은 옮기지 않았다.\n"
       << "- 짝 기준 m: 원래는 `cv2.QRCodeDetectorAruco` QR 네모 위치(OpenCV 3.4에 없음). QR 영역 ECC 중심 이동으로 대신했다. G1 출발점도 이 m을 쓴다.\n"
       << "- 판 호모그래피: 원래 `analysis_v2/homography_features.csv` 값과 사진에 따라 최대 약 0.035 px 다르다(20261013_1141 실험 참고). G1의 귀퉁이 모델 B와 기준 영상이 이 값으로 만들어진다.\n\n"
       << "## 원래 결과와 대조 (" << par.ok << "/" << par.total << " 일치)\n\n" << par.t.md()
       << "\n## 파일\n\n- `사진별_G1_Q5b.csv`: 사진별 G1 추정·방해 요소·비용·반복 수·판정, Q5b C6 추정\n- `판호모그래피_측정.csv`\n- `이미지/`: 그림 7장\n";
    write_text(out("결과.md"), Rm.str());

    // ---------- 보고서.md ----------
    auto cnt = [](const std::vector<double>& v) { return (int)std::count_if(v.begin(), v.end(), [](double x) { return std::abs(x) <= kPass; }); };
    std::ostringstream P;
    P << "# 보고서: QR만 정밀화, 픽셀 직접 가우스-뉴턴 (10-13 13:05)\n\n"
      << "## 문제\n\n전날 결과를 같은 조건으로 재현하고 가우스-뉴턴으로 QR만 방법의 구분 성능을 높이려 했다.\n\n"
      << "## 해결 방법\n\n판 원근 변환 특징(Q5b)에 곡선 모델 가우스-뉴턴(C6)을 더하고, QR 판 픽셀 전부로 Θ·Y와 방해 요소 3개를 직접 푸는 G1을 같은 2겹 교차검증·규칙3으로 비교했다.\n\n"
      << "## 결과\n\nG1이 Y 오차 RMS " << fmt(G.yrms * 1000, 0) << " µm, 구분 불가 Y " << G.nY << " / Θ " << G.nT << "로 Q5b 특징 방법(최고 " << [&] {
             double b = 1e9;
             for (size_t k = 0; k + 1 < methods.size(); ++k) b = std::min(b, methods[k].yrms);
             return fmt(b * 1000, 0);
         }() << " µm)보다 좋았다.\n\n"
      << "| 방법 | Y RMS (µm) | Y 최대 (µm) | Y 0.09 안 | Y 통과 (\\|eY\\|<=0.09) | T 통과 (\\|eT\\|<=0.09) | Θ RMS | 구분 불가 Y / Θ |\n|---|---|---|---|---|---|---|---|\n";
    for (auto& mt : methods) {
        std::vector<double> e1, e2;
        for (int i = 0; i < n; ++i) { e1.push_back(mt.est.at<double>(i, 1) - X.at<double>(i, 1)); e2.push_back(mt.est.at<double>(i, 0) - X.at<double>(i, 0)); }
        P << "| " << (mt.alg.substr(0, 2) == "G1" ? std::string("G1 픽셀 직접 GN") : "Q5b + " + mt.alg) << " | " << fmt(mt.yrms * 1000, 1) << " | " << fmt(mt.ymax * 1000, 1) << " | "
          << fmt(mt.y009, 1) << " % | " << cnt(e1) << " / " << n << " | " << cnt(e2) << " / " << n << " | " << fmt(mt.trms, 4) << " | " << mt.nY << " / " << mt.nT << " |\n";
    }
    P << "\n" << img_md("01_G1_추정오차.png", "G1 추정") << "\n\n" << img_md("02_방법별_Y오차RMS.png", "Y RMS") << "\n\n" << img_md("03_방법별_구분불가Y.png", "구분 불가") << "\n\n"
      << img_md("04_G1_규칙3_Y사진.png", "규칙3") << "\n\n" << img_md("05_G1_방해요소.png", "방해 요소") << "\n\n"
      << "## 튀는 값\n\nG1에서 가장 크게 틀린 사진은 Y는 " << names[wy] << "(정답 Y " << fmt(ty[wy], 2) << ", 추정 " << fmt(py[wy], 3) << " mm), T는 " << names[wt] << "(정답 T " << fmt(tt[wt], 2)
      << ", 추정 " << fmt(pt[wt], 4) << ")이고, 방해 요소(세로 " << fmts(g1nui.at<double>((int)wy, 0), 3) << " px, 회전 " << fmts(g1nui.at<double>((int)wy, 1), 3) << ", 크기 "
      << fmts(g1nui.at<double>((int)wy, 2), 3) << ")가 판 기울어짐과 일부 섞여 Θ·Y 몫이 바뀌었다.\n\n"
      << "| 구분 | 사진 | 정답 Y (mm) | 추정 Y | 정답 T (deg) | 추정 T | 세로 / 회전 / 크기 | 비용 | 반복 | 우상-좌상 dx (px) |\n|---|---|---|---|---|---|---|---|---|---|\n";
    for (auto [lab, idx] : std::vector<std::pair<std::string, std::vector<size_t>>>{{"Y 오차 상위", worst(ey, 3)}, {"T 오차 상위", worst(et, 3)}})
        for (size_t k : idx)
            P << "| " << lab << " | " << names[k] << " | " << fmt(ty[k], 2) << " | " << fmt(py[k], 3) << " | " << fmt(tt[k], 2) << " | " << fmt(pt[k], 4) << " | "
              << fmts(g1nui.at<double>((int)k, 0), 3) << " / " << fmts(g1nui.at<double>((int)k, 1), 3) << " / " << fmts(g1nui.at<double>((int)k, 2), 3) << " | " << fmt(g1cost[k], 3) << " | "
              << g1it[k] << " | " << fmts(CD.at<double>((int)k, 2) - CD.at<double>((int)k, 0), 3) << " |\n";
    P << "\n" << img_md(fy, "Y 튀는 값") << "\n\n" << img_md(ft, "T 튀는 값")
      << "\n\n빨간 십자는 호모그래피로 옮긴 판 귀퉁이, 초록 십자는 center 귀퉁이 + 판 평균 이동, 화살표는 귀퉁이 이동 - 평균 이동(200배)이다.\n\n"
      << "## 한줄평\n\nG1은 판 픽셀 " << eng.WW * eng.HH << "개 전부로 원근 기울어짐을 재고 방해 변형 3개를 따로 풀어 특징 몇 개보다 잡음이 작아서 Y RMS가 " << fmt(G.yrms * 1000, 0)
      << " µm까지 내려갔지만, Y ±" << fmt(edgeY, 2) << " mm 안쪽은 원근 신호가 잡음 3σ(σΘ " << fmt(G.sT, 4) << ")보다 작아 사진 1장으로 구분되지 않았다.\n\n"
      << "## 원래 결과와 대조\n\n원래 값은 `00000.legacies/analysis_v3/v3_results.csv`와 `00_v3결과리포트.md`에서 가져왔다. " << par.ok << "/" << par.total
      << " 일치. Q5b C2~C6·G1 전체 대조는 결과.md에 있다(" << full.ok << "/" << full.total << ").\n\n" << par.t.md()
      << "\n입력 측정 두 가지가 원래와 다르다. 판 호모그래피 ECC 값이 사진에 따라 최대 약 0.035 px 다르고(원래 cv2 판 없음, 20261013_1141 실험), 짝 기준 m은 `cv2.QRCodeDetectorAruco` 대신 QR 영역 ECC로 쟀다. "
         "Y·Θ를 가르는 원근 신호가 0.02~0.06 px라 이 차이가 그대로 구분 장수와 RMS 차이로 나온다. G1 계산식(선형화 좌표, 3차 스플라인, 방해 3개, 평균 기준 영상, 출발점 2곳)은 원본과 같다.\n";
    write_text(out("보고서.md"), P.str());
    return 0;
}
