// 특징점 84개(QR 밖 포함)와 QR 내부 특징점 300개를 템플릿 매칭으로 추적해 시차 조합(A·B·C·B2)과 QR만 방법을 비교한다.
// 원본: 00000.legacies/analysis_v3/v1_data/track_all.py, qr_only.py, combos.py, stable.py, grid.py
#include <hye/dataset.hpp>
#include <hye/plot.hpp>
#include <hye/report.hpp>
#include <omp.h>
#include <array>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <tuple>

using namespace hye;

namespace {

// ---------- 입력: 원래 tracks.npz 와 같은 282행 순서 (T -1.00..+1.00 사이에 center, Y -2.00..+2.00 사이에 center) ----------
struct Row {
    const Shot* s;
    double t, y;   // 정답 [Θ, Y]
    bool isT;      // T 묶음 (center 포함)
    std::string name;
};

// PIL Image.convert("L") 과 같은 회색 변환 (ITU-R 601-2, 16비트 고정소수점)
cv::Mat pil_gray(const cv::Mat& bgr) {
    cv::Mat g(bgr.rows, bgr.cols, CV_8U);
    for (int y = 0; y < bgr.rows; ++y) {
        const uchar* p = bgr.ptr<uchar>(y);
        uchar* q = g.ptr<uchar>(y);
        for (int x = 0; x < bgr.cols; ++x, p += 3) q[x] = (uchar)((p[2] * 19595 + p[1] * 38470 + p[0] * 7471 + 0x8000) >> 16);
    }
    return g;
}

// 포물선 부화소 보간. NumPy 2 와 같게 float32 로 계산한다
float sub(const float* a, int n, int stride, int i) {
    if (0 < i && i < n - 1) {
        float am = a[(i - 1) * stride], a0 = a[i * stride], ap = a[(i + 1) * stride];
        float d = am - 2.0f * a0 + ap;
        float c = d != 0 ? (0.5f * (am - ap)) / d : 0.0f;
        return (float)i + c;
    }
    return (float)i;
}

struct Tracker {
    int HP, SXR, SYR;
    std::vector<cv::Point2d> pts;
    bool f32pts;  // qr_only: pts 가 float32 라 (xi - x) 도 float32
    std::vector<cv::Mat> tpl;
    void init(const cv::Mat& C) {
        for (auto& p : pts) {
            int xi = (int)std::nearbyint(p.x), yi = (int)std::nearbyint(p.y);
            cv::Mat t;
            C(cv::Rect(xi - HP, yi - HP, 2 * HP, 2 * HP)).convertTo(t, CV_32F);
            tpl.push_back(t);
        }
    }
    // dx, dy, score
    void track(const cv::Mat& img, std::vector<std::array<double, 3>>& out) const {
        out.resize(pts.size());
        for (size_t k = 0; k < pts.size(); ++k) {
            double x = pts[k].x, y = pts[k].y;
            int xi = (int)std::nearbyint(x), yi = (int)std::nearbyint(y);
            cv::Mat s, r;
            img(cv::Rect(xi - HP - SXR, yi - HP - SYR, 2 * (HP + SXR), 2 * (HP + SYR))).convertTo(s, CV_32F);
            cv::matchTemplate(s, tpl[k], r, cv::TM_CCOEFF_NORMED);
            double mx;
            cv::Point b;
            cv::minMaxLoc(r, nullptr, &mx, nullptr, &b);
            float sx = sub(r.ptr<float>(b.y), r.cols, 1, b.x) - (float)SXR;
            float sy = sub(r.ptr<float>(0) + b.x, r.rows, (int)(r.step1()), b.y) - (float)SYR;
            if (f32pts) {
                out[k] = {(double)(sx + (float)(xi - (float)x)), (double)(sy + (float)(yi - (float)y)), mx};
            } else {
                out[k] = {(double)(float)((double)sx + (xi - x)), (double)(float)((double)sy + (yi - y)), (double)(float)mx};
            }
        }
    }
};

// ---------- 선형대수 (numpy lstsq 대응) ----------
using M = cv::Mat;
M lstsq(const M& A, const M& B) {
    M X;
    cv::solve(A, B, X, cv::DECOMP_SVD);
    return X;
}
M rows_of(const M& F, const std::vector<int>& r) {
    M o((int)r.size(), F.cols, CV_64F);
    for (size_t i = 0; i < r.size(); ++i) F.row(r[i]).copyTo(o.row((int)i));
    return o;
}
M cols_of(const M& F, const std::vector<int>& c) {
    M o(F.rows, (int)c.size(), CV_64F);
    for (size_t j = 0; j < c.size(); ++j) F.col(c[j]).copyTo(o.col((int)j));
    return o;
}
M ones_col(const M& A) {
    M o;
    cv::hconcat(A, M::ones(A.rows, 1, CV_64F), o);
    return o;
}
double rmsc(const M& e, int c) {
    double s = 0;
    for (int i = 0; i < e.rows; ++i) s += e.at<double>(i, c) * e.at<double>(i, c);
    return std::sqrt(s / e.rows);
}
double cond2(const M& J) {
    M w;
    cv::SVD::compute(J, w);
    return w.at<double>(0) / w.at<double>(w.rows - 1);
}

// combos.py / stable.py 의 평가: 특징점마다 dx = J [Θ, Y] (절편 없음, center = 0 고정)를 학습 묶음에서 맞추고, 시험 묶음을 J 로 거꾸로 푼다
struct Eval {
    double t, y, cond;
    M est;  // 시험 행 [Θ, Y] 추정
};
Eval evaluate(const M& D, const M& X, const std::vector<int>& cols, const std::vector<int>& tr, const std::vector<int>& te,
              const std::vector<double>& w = {}) {
    M Dc = cols_of(D, cols);
    M J = lstsq(rows_of(X, tr), rows_of(Dc, tr)).t();  // m x 2
    M Dte = rows_of(Dc, te);
    if (!w.empty())
        for (int j = 0; j < J.rows; ++j) { J.row(j) *= w[j]; Dte.col(j) *= w[j]; }
    M est = lstsq(J, Dte.t()).t();
    M e = est - rows_of(X, te);
    return {rmsc(e, 0), rmsc(e, 1), cond2(J), est};
}

// grid.py 의 알고리즘 C2~C5 (절편 학습, 학습 묶음으로 표준화)
M std_pop(const M& F) {
    M mu, sd(1, F.cols, CV_64F);
    cv::reduce(F, mu, 0, cv::REDUCE_AVG);
    for (int j = 0; j < F.cols; ++j) {
        double s = 0;
        for (int i = 0; i < F.rows; ++i) s += std::pow(F.at<double>(i, j) - mu.at<double>(j), 2);
        sd.at<double>(j) = std::sqrt(s / F.rows);
    }
    M o;
    cv::vconcat(mu, sd, o);
    return o;
}
M standardize(const M& F, const M& ms) {
    M o = F.clone();
    for (int i = 0; i < o.rows; ++i)
        for (int j = 0; j < o.cols; ++j) o.at<double>(i, j) = (o.at<double>(i, j) - ms.at<double>(0, j)) / (ms.at<double>(1, j) + 1e-12);
    return o;
}
M ridge_fit(const M& A1, const M& y, double al) {
    M G = A1.t() * A1;
    for (int j = 0; j + 1 < G.cols; ++j) G.at<double>(j, j) += al;
    M W;
    cv::solve(G, A1.t() * y, W, cv::DECOMP_LU);
    return W;
}
M predict(int alg, const M& F, const M& X, const std::vector<int>& tr, const std::vector<int>& te, double* alpha = nullptr) {
    M Ftr = rows_of(F, tr), Fte = rows_of(F, te), Xtr = rows_of(X, tr);
    if (alg == 2 || alg == 3) {
        M ms = std_pop(Ftr);
        M A = ones_col(standardize(Ftr, ms)), B = ones_col(standardize(Fte, ms));
        if (alg == 2) {
            if (A.cols > A.rows * 0.8) return M();
            return B * lstsq(A, Xtr);
        }
        double best = 1e300, ba = 0;
        for (double al : {1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0}) {
            double s = 0;
            int c = 0;
            for (int p = 0; p < 2; ++p) {
                std::vector<int> in, ou;
                for (int i = 0; i < A.rows; ++i) (i % 2 == p ? ou : in).push_back(i);
                M e = rows_of(A, ou) * ridge_fit(rows_of(A, in), rows_of(Xtr, in), al) - rows_of(Xtr, ou);
                for (int i = 0; i < e.rows; ++i) { s += std::pow(e.at<double>(i, 0) * 80, 2) + std::pow(e.at<double>(i, 1) * 14.3, 2); c += 2; }
            }
            double sc = std::sqrt(s / c);
            if (sc < best) { best = sc; ba = al; }
        }
        if (alpha) *alpha = ba;
        return B * ridge_fit(A, Xtr, ba);
    }
    if (F.cols < 2) return M();
    M A = ones_col(Xtr), Bc = lstsq(A, Ftr);
    M J = Bc.rowRange(0, 2).t(), b = Bc.row(2), R = Ftr - A * Bc;
    M Y = Fte - cv::repeat(b, Fte.rows, 1);
    if (alg == 4) {
        M ms = std_pop(R);
        M Jn = J.clone(), Yn = Y.clone();
        for (int j = 0; j < J.rows; ++j) { double sd = ms.at<double>(1, j) + 1e-12; Jn.row(j) /= sd; Yn.col(j) /= sd; }
        return lstsq(Jn, Yn.t()).t();
    }
    M mu;
    cv::reduce(R, mu, 0, cv::REDUCE_AVG);
    M Rc = R - cv::repeat(mu, R.rows, 1);
    M S = Rc.t() * Rc / (R.rows - 1);
    double jit = cv::trace(S)[0] / S.rows * 1e-2;
    for (int j = 0; j < S.rows; ++j) S.at<double>(j, j) += jit;
    M Si = S.inv(cv::DECOMP_LU), Mp;
    cv::solve(J.t() * Si * J, J.t() * Si, Mp, cv::DECOMP_LU);
    return Y * Mp.t();
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
    omp_set_num_threads(4);
    begin_results();
    auto shots = load_1014();
    const Shot& cs = center_1014(shots);
    std::vector<const Shot*> Ts, Ys;
    for (auto& s : shots) {
        if (&s == &cs) continue;
        (s.condition == "T" ? Ts : Ys).push_back(&s);
    }
    Ts.push_back(&cs);
    Ys.push_back(&cs);
    std::stable_sort(Ts.begin(), Ts.end(), [](auto a, auto b) { return a->t < b->t; });
    std::stable_sort(Ys.begin(), Ys.end(), [](auto a, auto b) { return a->y < b->y; });
    std::vector<Row> rows;
    auto nm = [](const Shot* s) { auto r = s->rel.substr(s->rel.rfind('/') + 1); return r.substr(0, r.rfind('.')); };
    for (auto s : Ts) rows.push_back({s, s->t, 0.0, true, s == &cs ? "center" : nm(s)});
    for (auto s : Ys) rows.push_back({s, 0.0, s->y, false, s == &cs ? "center" : nm(s)});
    const int n = (int)rows.size();  // 282 (center 2번)
    M X(n, 2, CV_64F);
    for (int i = 0; i < n; ++i) { X.at<double>(i, 0) = rows[i].t; X.at<double>(i, 1) = rows[i].y; }

    // ---------- 특징점 검출 (center 한 장) ----------
    cv::Mat Cc = read_color(cs), C = pil_gray(Cc);
    Tracker ta{24, 120, 12};
    {
        std::vector<cv::Point2f> p;
        cv::goodFeaturesToTrack(C, p, 400, 0.01, 90, cv::noArray(), 9);
        ta.pts.push_back({1846.7, 1010.3});  // 파인더 패턴은 0번으로 고정 (track_all.py)
        for (auto& q : p)
            if (24 + 120 + 2 < q.x && q.x < C.cols - 24 - 120 - 2 && 24 + 12 + 2 < q.y && q.y < C.rows - 24 - 12 - 2) ta.pts.push_back({q.x, q.y});
        ta.f32pts = false;
        ta.init(C);
    }
    // QR 세 개 영역. 원본은 cv2.QRCodeDetectorAruco().detectMulti(center) (OpenCV 3.4 에 없음) → cv2 5.0 에서 같은 center.bmp 로 뽑은 꼭짓점을 상수로 둔다
    const std::vector<std::vector<cv::Point2f>> qr_poly{{{1821.f, 982.f}, {1982.f, 983.f}, {1978.9464f, 1154.f}, {1820.f, 1154.f}},
                                                        {{1563.f, 981.f}, {1725.f, 983.f}, {1718.8182f, 1153.f}, {1563.f, 1153.f}},
                                                        {{2080.f, 983.f}, {2241.f, 985.f}, {2237.8154f, 1160.1467f}, {2078.f, 1154.f}}};
    Tracker tq{10, 110, 6};
    {
        cv::Mat mask = cv::Mat::zeros(C.size(), CV_8U);
        for (auto& p : qr_poly) {
            std::vector<cv::Point> ip;
            for (auto& q : p) ip.push_back({(int)q.x, (int)q.y});
            cv::fillConvexPoly(mask, ip, 255);
        }
        std::vector<cv::Point2f> p;
        cv::goodFeaturesToTrack(C, p, 300, 0.01, 12, mask, 5);
        for (auto& q : p) tq.pts.push_back({q.x, q.y});
        tq.f32pts = true;
        tq.init(C);
    }
    const int N = (int)ta.pts.size(), NQ = (int)tq.pts.size();
    std::cout << "points " << N << " qr " << NQ << "\n";

    // ---------- 추적 (282행, center 는 한 번만 재서 복사) ----------
    std::vector<std::vector<std::array<double, 3>>> MA(n), MQ(n);
#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < n; ++i) {
        if (rows[i].s == &cs && !rows[i].isT) continue;
        cv::Mat g = pil_gray(read_color(*rows[i].s));
        ta.track(g, MA[i]);
        tq.track(g, MQ[i]);
    }
    for (int i = 0; i < n; ++i)
        if (MA[i].empty())
            for (int j = 0; j < n; ++j)
                if (rows[j].s == rows[i].s && !MA[j].empty()) { MA[i] = MA[j]; MQ[i] = MQ[j]; }
    std::cout << "tracked\n";
    M D(n, N, CV_64F), S(n, N, CV_64F), DQ(n, NQ, CV_64F), SQ(n, NQ, CV_64F);
    for (int i = 0; i < n; ++i) {
        for (int k = 0; k < N; ++k) { D.at<double>(i, k) = MA[i][k][0]; S.at<double>(i, k) = MA[i][k][2]; }
        for (int k = 0; k < NQ; ++k) { DQ.at<double>(i, k) = MQ[i][k][0]; SQ.at<double>(i, k) = MQ[i][k][2]; }
    }
    {
        Table t{{"row", "file", "true_T", "true_Y", "point", "x", "y", "dx", "dy", "score"}};
        for (int i = 0; i < n; ++i)
            for (int k = 0; k < N; ++k)
                t.add({std::to_string(i), rows[i].name, fmt(rows[i].t, 2), fmt(rows[i].y, 2), std::to_string(k), fmt(ta.pts[k].x, 1), fmt(ta.pts[k].y, 1),
                       fmt(MA[i][k][0], 5), fmt(MA[i][k][1], 5), fmt(MA[i][k][2], 5)});
        t.csv(out("tracks_84.csv"));
    }

    // ---------- combos.py: 특징점별 선형 모델, 좋은 점, 쌍·전체·greedy ----------
    struct Info { double a, b, ratio, sig, maxres, minscore; };
    auto fit_info = [&](const M& Dm, const M& Sm, int k) {
        M ab = lstsq(X, Dm.col(k));
        double a = ab.at<double>(0), b = ab.at<double>(1), s2 = 0, mr = 0, ms = 1e9;
        for (int i = 0; i < n; ++i) {
            double r = Dm.at<double>(i, k) - (a * X.at<double>(i, 0) + b * X.at<double>(i, 1));
            s2 += r * r; mr = std::max(mr, std::abs(r)); ms = std::min(ms, Sm.at<double>(i, k));
        }
        return Info{a, b, b / a, std::sqrt(s2 / n), mr, ms};
    };
    std::vector<Info> info;
    std::vector<int> good;
    for (int k = 0; k < N; ++k) {
        info.push_back(fit_info(D, S, k));
        if (info[k].maxres < 2.0 && info[k].minscore > 0.8) good.push_back(k);
    }
    std::vector<int> ev_tr, ev_te, fit4, sel4, fin4;
    for (int i = 0; i < n; ++i) {
        (i % 2 == 0 ? ev_tr : ev_te).push_back(i);
        if (i % 4 < 2) fit4.push_back(i); else if (i % 4 == 2) sel4.push_back(i); else fin4.push_back(i);
    }
    auto score = [&](const std::vector<int>& cols, const std::vector<double>& w = {}) { return evaluate(D, X, cols, ev_tr, ev_te, w); };
    int bolt = good[0];
    for (int k : good)
        if (std::hypot(ta.pts[k].x - 3576, ta.pts[k].y - 1802) < std::hypot(ta.pts[bolt].x - 3576, ta.pts[bolt].y - 1802)) bolt = k;
    std::vector<std::tuple<double, double, double, int, int>> pairs;
    for (size_t a = 0; a < good.size(); ++a)
        for (size_t b = a + 1; b < good.size(); ++b) {
            auto e = score({good[a], good[b]});
            pairs.emplace_back(e.y, e.t, e.cond, good[a], good[b]);
        }
    std::sort(pairs.begin(), pairs.end());
    std::vector<int> greedy{std::get<3>(pairs[0]), std::get<4>(pairs[0])};
    std::vector<std::array<double, 3>> ghist{{2, std::get<1>(pairs[0]), std::get<0>(pairs[0])}};
    {
        std::vector<int> rest;
        for (int k : good) if (k != greedy[0] && k != greedy[1]) rest.push_back(k);
        while (!rest.empty() && greedy.size() < 30) {
            size_t bi = 0;
            double by = 1e300;
            for (size_t r = 0; r < rest.size(); ++r) {
                auto s2 = greedy;
                s2.push_back(rest[r]);
                double y = score(s2).y;
                if (y < by) { by = y; bi = r; }
            }
            greedy.push_back(rest[bi]);
            rest.erase(rest.begin() + bi);
            auto e = score(greedy);
            ghist.push_back({(double)greedy.size(), e.t, e.y});
        }
    }
    size_t gbest = 0;
    for (size_t h = 1; h < ghist.size(); ++h) if (ghist[h][2] < ghist[gbest][2]) gbest = h;
    std::vector<double> wsig;
    for (int k : good) wsig.push_back(1 / info[k].sig);

    // ---------- stable.py: 위치마다 바뀌는 것(라벨, QR 데이터, 케이블) 제외 ----------
    const std::vector<std::pair<std::string, cv::Vec4d>> EXCL{{"label", {1740, 1195, 2060, 1360}}, {"QR data", {1560, 975, 2250, 1160}}, {"cable", {1840, 0, 2260, 560}}};
    auto why = [&](int k) -> std::string {
        if (k == 0) return "";
        for (auto& e : EXCL)
            if (e.second[0] < ta.pts[k].x && ta.pts[k].x < e.second[2] && e.second[1] < ta.pts[k].y && ta.pts[k].y < e.second[3]) return e.first;
        return "";
    };
    std::vector<int> ok, excl;
    for (int k : good) (why(k).empty() ? ok : excl).push_back(k);
    std::vector<std::tuple<double, int, int>> spairs;
    for (size_t a = 0; a < ok.size(); ++a)
        for (size_t b = a + 1; b < ok.size(); ++b) spairs.emplace_back(evaluate(D, X, {ok[a], ok[b]}, fit4, sel4).y, ok[a], ok[b]);
    std::sort(spairs.begin(), spairs.end());
    const std::vector<int> B2{std::get<1>(spairs[0]), std::get<2>(spairs[0])};

    // 대표 조합 (짝수 학습 / 홀수 시험, center = 0 고정)
    struct Combo { std::string id, label; std::vector<int> cols; Eval e; };
    std::vector<Combo> combos{{"A", "finder + wall bolt", {0, bolt}, {}},
                              {"B", "ADDRESS label + bracket hook", {1, 46}, {}},
                              {"C", "all good points", good, {}},
                              {"B2", "bracket top + bottom (no label)", B2, {}},
                              {"ok", "unchanging points", ok, {}}};
    for (auto& c : combos) c.e = score(c.cols);
    Eval ok_fin = evaluate(D, X, ok, fit4, fin4), b2_fin = evaluate(D, X, B2, fit4, fin4);
    Eval c_w = score(good, wsig);

    // ---------- qr_only.py: QR 내부 특징점 이동량만 ----------
    std::vector<Info> qinfo;
    std::vector<int> qgood;
    for (int k = 0; k < NQ; ++k) {
        qinfo.push_back(fit_info(DQ, SQ, k));
        if (qinfo[k].maxres < 2 && qinfo[k].minscore > 0.8) qgood.push_back(k);
    }
    double qr_min = 1e9, qr_max = -1e9;
    for (int k : qgood) { qr_min = std::min(qr_min, qinfo[k].ratio); qr_max = std::max(qr_max, qinfo[k].ratio); }
    Eval qe = evaluate(DQ, X, qgood, ev_tr, ev_te);

    // ---------- grid.py: 중복 center 제거 281행, 짝/홀 2겹, 절편 학습, C2~C5 ----------
    std::vector<int> keep;
    {
        bool seen = false;
        for (int i = 0; i < n; ++i) {
            if (rows[i].name == "center") { if (seen) continue; seen = true; }
            keep.push_back(i);
        }
    }
    const int n2 = (int)keep.size();
    M X2 = rows_of(X, keep), D2 = rows_of(D, keep), DQ2 = rows_of(DQ, keep);
    struct GridRow { std::string feat, alg; double tm, tr, tx, ym, yr, yx, yin, yT, tY; M est; };
    std::vector<GridRow> grid;
    std::vector<std::pair<std::string, M>> FEAT{{"Q7 QR inner points dx", cols_of(DQ2, qgood)},
                                                {"O2 combo A finder+wall bolt", cols_of(D2, {0, bolt})},
                                                {"O3 combo B2", cols_of(D2, B2)},
                                                {"O4 unchanging points", cols_of(D2, ok)},
                                                {"O5 all good points", cols_of(D2, good)}};
    const char* ALG[] = {"", "", "C2 OLS", "C3 ridge", "C4 Jacobian LS", "C5 Jacobian GLS"};
    std::vector<int> fa, fb;
    for (int i = 0; i < n2; ++i) (i % 2 == 0 ? fa : fb).push_back(i);
    for (auto& f : FEAT)
        for (int alg = 2; alg <= 5; ++alg) {
            M est(n2, 2, CV_64F);
            bool good_run = true;
            for (int fold = 0; fold < 2; ++fold) {
                auto& tr = fold ? fb : fa;
                auto& te = fold ? fa : fb;
                M r = predict(alg, f.second, X2, tr, te);
                if (r.empty()) { good_run = false; break; }
                for (size_t i = 0; i < te.size(); ++i) r.row((int)i).copyTo(est.row(te[i]));
            }
            if (!good_run) continue;
            M e = est - X2;
            std::vector<double> eT, eY, yT, tY;
            int in = 0;
            for (int i = 0; i < n2; ++i) {
                eT.push_back(std::abs(e.at<double>(i, 0)));
                eY.push_back(std::abs(e.at<double>(i, 1)));
                in += eY.back() <= 0.05;
                bool isT = X2.at<double>(i, 1) == 0, isY = X2.at<double>(i, 0) == 0;
                if (isY && !isT) yT.push_back(e.at<double>(i, 0));
                if (isT && !isY) tY.push_back(e.at<double>(i, 1));
            }
            grid.push_back({f.first, ALG[alg], mean(eT), rms(eT), max_abs(eT), mean(eY), rms(eY), max_abs(eY), 100.0 * in / n2, rms(yT), rms(tY), est});
        }
    auto grid_at = [&](const std::string& f, const std::string& a) -> GridRow& {
        for (auto& g : grid) if (g.feat.substr(0, 2) == f && g.alg.substr(0, 2) == a) return g;
        throw std::runtime_error("grid 없음 " + f + a);
    };

    // ---------- 튀는 값: B2 짝/홀 시험 행 ----------
    auto& b2 = combos[3].e;
    std::vector<double> ty, py, tt, pt, ey, et;
    std::vector<std::string> names;
    M Jb2 = lstsq(rows_of(X, ev_tr), rows_of(cols_of(D, B2), ev_tr)).t();
    for (size_t i = 0; i < ev_te.size(); ++i) {
        int r = ev_te[i];
        tt.push_back(X.at<double>(r, 0)); ty.push_back(X.at<double>(r, 1));
        pt.push_back(b2.est.at<double>((int)i, 0)); py.push_back(b2.est.at<double>((int)i, 1));
        et.push_back(pt.back() - tt.back()); ey.push_back(py.back() - ty.back());
        names.push_back(rows[r].name);
    }
    auto resid = [&](int r, int j) { return D.at<double>(r, B2[j]) - (Jb2.at<double>(j, 0) * X.at<double>(r, 0) + Jb2.at<double>(j, 1) * X.at<double>(r, 1)); };
    size_t wy = worst(ey, 1)[0], wt = worst(et, 1)[0];
    Table outl{{"구분", "사진", "정답 Y (mm)", "추정 Y", "정답 T (deg)", "추정 T", "#" + std::to_string(B2[0]) + " / #" + std::to_string(B2[1]) + " dx (px)",
                "모델 대비 잔차 (px)"}};
    for (auto [lab, idx] : std::vector<std::pair<std::string, std::vector<size_t>>>{{"Y 오차 상위", worst(ey, 3)}, {"T 오차 상위", worst(et, 3)}})
        for (size_t k : idx) {
            int r = ev_te[k];
            outl.add({lab, names[k], fmt(ty[k], 2), fmt(py[k], 4), fmt(tt[k], 2), fmt(pt[k], 4), fmt(D.at<double>(r, B2[0]), 3) + " / " + fmt(D.at<double>(r, B2[1]), 3),
                      fmts(resid(r, 0), 3) + " / " + fmts(resid(r, 1), 3)});
        }

    // ---------- 그림 ----------
    {  // 01 특징점 지도: 색 = b/a (Y 1mm 이동 / Θ 1 이동, 거리 단서), 회색 X = 제외, 노란 원 = B2
        cv::Mat m;
        cv::resize(Cc, m, {1920, 1080}, 0, 0, cv::INTER_AREA);
        double rmin = 1e9, rmax = -1e9;
        for (int k : good) { rmin = std::min(rmin, info[k].ratio); rmax = std::max(rmax, info[k].ratio); }
        for (auto& e : EXCL) cv::rectangle(m, {(int)(e.second[0] / 2), (int)(e.second[1] / 2)}, {(int)(e.second[2] / 2), (int)(e.second[3] / 2)}, kRed, 2);
        for (int k = 0; k < N; ++k) {
            cv::Point p((int)std::lround(ta.pts[k].x / 2), (int)std::lround(ta.pts[k].y / 2));
            bool g = std::find(good.begin(), good.end(), k) != good.end();
            if (!g) { cv::drawMarker(m, p, kGray, cv::MARKER_TILTED_CROSS, 12, 2); continue; }
            double f = (info[k].ratio - rmin) / (rmax - rmin + 1e-12);
            cv::Scalar c(255 * (1 - f), 60, 255 * f);  // 파랑(작은 b/a) → 빨강(큰 b/a)
            bool ex = !why(k).empty();
            if (ex) cv::drawMarker(m, p, c, cv::MARKER_TILTED_CROSS, 14, 3);
            else cv::circle(m, p, 8, c, 3, cv::LINE_AA);
        }
        for (int k : B2) {
            cv::Point p((int)std::lround(ta.pts[k].x / 2), (int)std::lround(ta.pts[k].y / 2));
            cv::circle(m, p, 18, {0, 230, 255}, 4, cv::LINE_AA);
            text(m, "#" + std::to_string(k), p + cv::Point(22, -10), 1.0, {0, 230, 255}, 2);
        }
        text(m, "color = Y/T response ratio b/a (blue " + fmt(rmin, 3) + " .. red " + fmt(rmax, 3) + "), X = excluded/unstable, yellow = B2", {20, 40}, 0.9, {255, 255, 255}, 2);
        save_png("01_특징점지도_비율색.png", m);
    }
    {  // 02 b/a 비율 vs 화면 y
        std::vector<double> x0, y0, x1, y1;
        for (int k : good) (why(k).empty() ? x0 : x1).push_back(ta.pts[k].y), (why(k).empty() ? y0 : y1).push_back(info[k].ratio);
        std::vector<double> xq, yq;
        for (int k : qgood) { xq.push_back(tq.pts[k].y); yq.push_back(qinfo[k].ratio); }
        Chart c("Parallax cue: Y response / T response per point", "image y of point (px)", "b/a (px per mm / px per deg)");
        c.scatter(xq, yq, kGray, "QR inner points", 2).scatter(x0, y0, color(0), "unchanging points").scatter(x1, y1, color(1), "label / QR data / cable");
        for (int k : B2) c.note(ta.pts[k].y, info[k].ratio, "#" + std::to_string(k));
        c.save("02_특징점별_YT비율.png");
    }
    {  // 03 조합별 Y RMS / Θ RMS
        std::vector<std::string> cats;
        std::vector<double> yv, tv;
        for (auto& c : combos) { cats.push_back(c.id + " (" + std::to_string(c.cols.size()) + ")"); yv.push_back(c.e.y * 1000); tv.push_back(c.e.t * 1000); }
        cats.push_back("QR dx only (" + std::to_string(qgood.size()) + ")"); yv.push_back(qe.y * 1000); tv.push_back(qe.t * 1000);
        auto& g7 = grid_at("Q7", "C5");
        cats.push_back("QR GLS grid (" + std::to_string(qgood.size()) + ")"); yv.push_back(g7.yr * 1000); tv.push_back(g7.tr * 1000);
        bar_chart("03_조합별_Y오차.png", "Y RMS error by combination (um)", cats, yv, "Y RMS (um)", 3, 50);
        bar_chart("04_조합별_T오차.png", "T RMS error by combination (x1e-3 deg)", cats, tv, "T RMS (1e-3 deg)", 3);
    }
    pred_vs_truth("05_B2_시험141장_오차.png", "B2 held-out 141", ty, py, tt, pt, names);
    auto outlier_fig = [&](size_t k, const std::string& file, const std::string& cap) {
        int r = ev_te[k];
        cv::Mat im = read_color(*rows[r].s);
        std::vector<cv::Mat> tiles;
        for (int j = 0; j < 2; ++j) {
            int p = B2[j];
            cv::Point2d ref = ta.pts[p], now(ref.x + D.at<double>(r, p), ref.y + MA[r][p][1]);
            cv::Rect2d rb(ref.x - 24, ref.y - 24, 48, 48), nb(now.x - 24, now.y - 24, 48, 48);
            cv::Rect roi = cv::Rect(rb | nb) + cv::Size(80, 80) - cv::Point(40, 40);
            tiles.push_back(zoom(im, roi, 420, {{now, {0, 0}, "", kRed}, {ref, {0, 0}, "", kGreen}}, 0, j == 0 ? cap : "",
                                 {{rb, kGreen, "ref"}, {nb, kRed, "#" + std::to_string(p) + " dx " + fmt(D.at<double>(r, p), 2) + " res " + fmts(resid(r, j), 2)}}));
        }
        save_png(file, hstack(tiles));
    };
    const std::string fy = "06_튀는값_Y_" + names[wy] + ".png", ft = "07_튀는값_T_" + names[wt] + ".png";
    outlier_fig(wy, fy, names[wy] + " Y err " + fmts(ey[wy], 3) + " mm");
    outlier_fig(wt, ft, names[wt] + " T err " + fmts(et[wt], 4) + " deg");
    {
        std::vector<double> xs, ys;
        for (auto& h : ghist) { xs.push_back(h[0]); ys.push_back(h[2] * 1000); }
        Chart c("Greedy point addition from best pair", "number of points", "Y RMS (um)");
        c.line(xs, ys, color(0), "Y RMS", true);
        c.save("08_greedy_점추가.png");
    }

    // ---------- 원래 결과와 대조 ----------
    // 출처: 00000.legacies/00_결과요약.md, 00_실험전체목록_요일별.md, analysis_v3/v1_data/combos.json·stable.json, 17_결과표_특징점조합x알고리즘.csv
    Parity par;
    par.add("특징점 수 (center 검출)", 84, N, 0, 0);
    par.add("좋은 특징점 수", 74, (double)good.size(), 0, 0);
    par.add_text("볼트 근처 점 / 최고 쌍", "#47 / #1+#46", "#" + std::to_string(bolt) + " / #" + std::to_string(std::get<3>(pairs[0])) + "+#" + std::to_string(std::get<4>(pairs[0])),
                 bolt == 47 && std::get<3>(pairs[0]) == 1 && std::get<4>(pairs[0]) == 46);
    par.add("남은 점 수 (라벨·QR 데이터·케이블 제외)", 51, (double)ok.size(), 0, 0);
    par.add_text("B2 쌍", "#11+#46", "#" + std::to_string(B2[0]) + "+#" + std::to_string(B2[1]), B2[0] == 11 && B2[1] == 46);
    par.add("조합 A Y RMS (mm)", 0.051, combos[0].e.y, 0.001, 3);
    par.add("조합 A Θ RMS", 0.0086, combos[0].e.t, 0.0002);
    par.add("조합 B Y RMS (mm)", 0.019, combos[1].e.y, 0.001, 3);
    par.add("조합 B Θ RMS", 0.0066, combos[1].e.t, 0.0002);
    par.add("조합 C Y RMS (mm)", 0.034, combos[2].e.y, 0.001, 3);
    par.add("조합 C Θ RMS", 0.0026, combos[2].e.t, 0.0002);
    par.add("조합 B2 Y RMS (mm)", 0.020, combos[3].e.y, 0.001, 3);
    par.add("조합 B2 Θ RMS", 0.0064, combos[3].e.t, 0.0002);
    par.add("안 바뀌는 점 전부 Y RMS (4등분 채점, mm)", 0.030, ok_fin.y, 0.001, 3);
    par.add("QR 내부 안정 특징점 수", 263, (double)qgood.size(), 0, 0);
    par.add("QR 이동량만 Y RMS (mm)", 0.47, qe.y, 0.01, 2);
    par.add("QR 이동량만 Θ RMS", 0.088, qe.t, 0.002, 3);
    par.add_text("QR만 원근 왜곡 1장 Y RMS / Θ RMS", "0.150 / 0.0275", "코드 없음 (Q7×C5 " + fmt(grid_at("Q7", "C5").yr, 3) + " / " + fmt(grid_at("Q7", "C5").tr, 4) + ")", false);
    // 17_결과표: (특징, 알고리즘, T_RMS, Y_RMS)
    const std::vector<std::tuple<std::string, std::string, double, double>> G17{
        {"Q7", "C3", 0.0295, 0.1652}, {"Q7", "C4", 0.0679, 0.3793}, {"Q7", "C5", 0.0244, 0.1371},
        {"O2", "C2", 0.0066, 0.0466}, {"O2", "C3", 0.0066, 0.0465}, {"O2", "C4", 0.0066, 0.0468}, {"O2", "C5", 0.0066, 0.0468},
        {"O3", "C2", 0.0033, 0.0188}, {"O3", "C3", 0.0033, 0.0188}, {"O3", "C4", 0.0033, 0.0188}, {"O3", "C5", 0.0033, 0.0188},
        {"O4", "C2", 0.0018, 0.0141}, {"O4", "C3", 0.0018, 0.0144}, {"O4", "C4", 0.0024, 0.0191}, {"O4", "C5", 0.0018, 0.0142},
        {"O5", "C2", 0.0025, 0.0130}, {"O5", "C3", 0.0021, 0.0134}, {"O5", "C4", 0.0025, 0.0215}, {"O5", "C5", 0.0021, 0.0127}};
    Parity gpar;
    for (auto& [f, a, t, y] : G17) {
        auto& g = grid_at(f, a);
        gpar.add(f + " × " + a + " Y RMS", y, g.yr, 0.0006);
        gpar.add(f + " × " + a + " Θ RMS", t, g.tr, 0.0002);
    }
    std::cout << "parity " << par.ok << "/" << par.total << " grid " << gpar.ok << "/" << gpar.total << "\n";

    // ---------- 결과.md ----------
    Table ct{{"조합", "특징점", "개수", "Θ RMS", "Y RMS (mm)", "cond(J)"}};
    for (auto& c : combos) {
        std::string ids;
        for (size_t j = 0; j < c.cols.size() && j < 6; ++j) ids += (j ? "," : "") + std::string("#") + std::to_string(c.cols[j]);
        if (c.cols.size() > 6) ids += ",...";
        ct.add({c.id + " " + c.label, ids, std::to_string(c.cols.size()), fmt(c.e.t, 4), fmt(c.e.y, 4), fmt(c.e.cond, 1)});
    }
    ct.add({"C 노이즈 가중 (1/σ)", "좋은 점 전부", std::to_string(good.size()), fmt(c_w.t, 4), fmt(c_w.y, 4), "-"});
    ct.add({"B2 (4등분 최종 채점)", "#" + std::to_string(B2[0]) + ",#" + std::to_string(B2[1]), "2", fmt(b2_fin.t, 4), fmt(b2_fin.y, 4), fmt(b2_fin.cond, 1)});
    ct.add({"안 바뀌는 점 (4등분 최종 채점)", "-", std::to_string(ok.size()), fmt(ok_fin.t, 4), fmt(ok_fin.y, 4), fmt(ok_fin.cond, 1)});
    ct.add({"QR 내부 이동량만", "-", std::to_string(qgood.size()), fmt(qe.t, 4), fmt(qe.y, 4), fmt(qe.cond, 1)});
    Table pt10{{"순위", "쌍", "Θ RMS", "Y RMS (mm)", "cond(J)"}};
    for (int r = 0; r < 10; ++r) {
        auto& p = pairs[r];
        pt10.add({std::to_string(r + 1), "#" + std::to_string(std::get<3>(p)) + "(" + fmt(ta.pts[std::get<3>(p)].x, 0) + "," + fmt(ta.pts[std::get<3>(p)].y, 0) + ") + #" +
                                             std::to_string(std::get<4>(p)) + "(" + fmt(ta.pts[std::get<4>(p)].x, 0) + "," + fmt(ta.pts[std::get<4>(p)].y, 0) + ")",
                  fmt(std::get<1>(p), 4), fmt(std::get<0>(p), 4), fmt(std::get<2>(p), 1)});
    }
    Table sp8{{"순위", "쌍 (선택용 1/4로 고름)", "Y RMS 선택용", "Θ RMS 최종", "Y RMS 최종 (mm)"}};
    for (int r = 0; r < 8; ++r) {
        auto& p = spairs[r];
        auto e = evaluate(D, X, {std::get<1>(p), std::get<2>(p)}, fit4, fin4);
        sp8.add({std::to_string(r + 1), "#" + std::to_string(std::get<1>(p)) + " + #" + std::to_string(std::get<2>(p)), fmt(std::get<0>(p), 4), fmt(e.t, 4), fmt(e.y, 4)});
    }
    Table gh{{"점 수", "Θ RMS", "Y RMS (mm)"}};
    for (auto& h : ghist) gh.add({fmt(h[0], 0), fmt(h[1], 4), fmt(h[2], 4)});
    Table gt{{"특징점", "알고리즘", "T 평균", "T RMS", "T 최대", "Y 평균", "Y RMS", "Y 최대", "Y 0.05 안 %", "Y사진 T착각", "T사진 Y착각"}};
    Table gcsv = gt;
    for (auto& g : grid) {
        std::vector<std::string> r{g.feat, g.alg, fmt(g.tm, 4), fmt(g.tr, 4), fmt(g.tx, 4), fmt(g.ym, 4), fmt(g.yr, 4), fmt(g.yx, 4), fmt(g.yin, 2), fmt(g.yT, 4), fmt(g.tY, 4)};
        gt.add(r);
        gcsv.add(r);
    }
    gcsv.csv(out("grid_results.csv"));
    Table pts_t{{"점", "x", "y", "a (px/deg)", "b (px/mm)", "b/a", "잔차 RMS (px)", "최대 잔차", "최소 점수", "좋음", "제외 이유"}};
    for (int k = 0; k < N; ++k) {
        bool g = std::find(good.begin(), good.end(), k) != good.end();
        pts_t.add({std::to_string(k), fmt(ta.pts[k].x, 1), fmt(ta.pts[k].y, 1), fmt(info[k].a, 4), fmt(info[k].b, 4), fmt(info[k].ratio, 4), fmt(info[k].sig, 4), fmt(info[k].maxres, 3),
                   fmt(info[k].minscore, 3), g ? "O" : "X", g ? why(k) : "-"});
    }
    pts_t.csv(out("points_84.csv"));
    std::ostringstream R;
    R << "# 결과: 특징점 조합 선택 (시차 조합 비교)\n\n"
      << "입력: `data/clean/clean_20261014` 281장. 원래 `tracks.npz`와 같이 T 묶음(-1.00..+1.00, 가운데 center)과 Y 묶음(-2.00..+2.00, 가운데 center)을 이어 282행으로 쓴다. 사진은 PIL `convert(\"L\")`과 같은 식으로 회색 변환한다.\n\n"
      << "측정: center에서 `goodFeaturesToTrack(400, 0.01, 90, block 9)` + 파인더 (1846.7, 1010.3) = " << N << "점을 48x48 템플릿, 가로 ±120 / 세로 ±12 px 탐색, NCC + 포물선 보간으로 추적했다. "
      << "QR 내부는 QR 세 개 다각형 안에서 `goodFeaturesToTrack(300, 0.01, 12, block 5)` = " << NQ << "점을 20x20 템플릿, ±110 / ±6 px로 추적했다.\n\n"
      << "좋은 점: 전체 282행에서 dx = a·Θ + b·Y 직선 잔차 최대 < 2 px이고 NCC 최소 > 0.8. 특징점 " << good.size() << "/" << N << ", QR 내부 " << qgood.size() << "/" << NQ
      << " (QR b/a 범위 " << fmt(qr_min, 4) << " ~ " << fmt(qr_max, 4) << ").\n\n"
      << "## 조합별 오차 (짝수 행 학습, 홀수 행 시험, center = 0 고정, 절편 없음)\n\n" << ct.md()
      << "\n## 2점 조합 상위 10 (" << pairs.size() << "쌍)\n\n" << pt10.md()
      << "\n## 바뀌는 점 제외 후 최고 쌍 (행 번호 % 4: 0,1 학습 / 2 선택 / 3 최종 채점)\n\n제외 " << excl.size() << "점 (라벨·QR 데이터·케이블 상자), 남은 " << ok.size() << "점.\n\n" << sp8.md()
      << "\n## greedy 점 추가 (최고 쌍에서 시작, 최적 " << (int)ghist[gbest][0] << "점 Y " << fmt(ghist[gbest][2], 4) << " mm)\n\n" << gh.md()
      << "\n## 통일 기준 결과표 (281장, 짝/홀 2겹 교차검증, 절편 학습)\n\n" << gt.md()
      << "\n## 결과표 대조 (" << gpar.ok << "/" << gpar.total << " 일치)\n\n원래 값: `00000.legacies/17_결과표_특징점조합x알고리즘.csv`.\n\n" << gpar.t.md()
      << "\n## 옮기지 않은 부분\n\n"
      << "- 결과표의 Q1~Q6, O1, O6: 입력인 `14_데이터_QR마커위치.csv`, `15_데이터_파인더_볼트이동량.csv`, `ecc_feats.npy`를 만든 스크립트가 남아 있지 않아 사진에서 다시 잴 수 없다.\n"
      << "- QR만 원근 왜곡 1장(Y 0.150 mm, Θ 0.0275)과 여러 장 평균 시뮬레이션(5·10·20장): 계산 코드가 남아 있지 않다. 실험계획표의 기록(A3 QR 전체 픽셀, B4 ECC, C5 원근 왜곡 회귀)만 있다. QR만 방법은 위 결과표의 Q7(QR 내부 특징점)로 대신 보인다.\n"
      << "- `ecc_tune.py`(ECC 설정 탐색), 영상(07~09, 11, 13): 결과 수치에 쓰이지 않아 옮기지 않았다.\n"
      << "- QR 세 개 다각형은 원래 `cv2.QRCodeDetectorAruco`로 찾았다. OpenCV 3.4에 없어 cv2 5.0으로 같은 center.bmp에서 뽑은 꼭짓점을 코드에 상수로 넣었다.\n\n"
      << "## 원래 결과와 대조 (" << par.ok << "/" << par.total << " 일치)\n\n" << par.t.md()
      << "\n## 파일\n\n- `tracks_84.csv`: 282행 x 84점 dx, dy, NCC\n- `points_84.csv`: 점별 a, b, b/a, 잔차, 좋음 여부, 제외 이유\n- `grid_results.csv`: 통일 기준 결과표\n- `이미지/`: 그림 8장\n";
    write_text(out("결과.md"), R.str());

    // ---------- 보고서.md ----------
    auto cnt = [](const std::vector<double>& v) { return (int)std::count_if(v.begin(), v.end(), [](double x) { return std::abs(x) <= kPass; }); };
    std::ostringstream P;
    auto& g7 = grid_at("Q7", "C5");
    P << "# 보고서: 특징점 조합 선택, 시차 조합 비교 (10-12 15:24)\n\n"
      << "## 문제\n\n사진 1장에서 Y와 Θ를 구분하려면 어떤 특징점 조합과 알고리즘이 필요한지 몰랐다.\n\n"
      << "## 해결 방법\n\n특징점 " << N << "개와 QR 내부 점 " << NQ << "개를 템플릿 매칭으로 추적하고, 점마다 Θ·Y 반응 기울기를 맞춘 뒤 2점 쌍·전체·바뀌지 않는 점 조합과 QR만 조합을 알고리즘 C2~C5로 비교했다.\n\n"
      << "## 결과\n\n바뀌지 않는 부품 두 점 B2(#" << B2[0] << " + #" << B2[1] << ")가 Y RMS " << fmt(combos[3].e.y, 3) << " mm, Θ RMS " << fmt(combos[3].e.t, 4)
      << "로 QR 내부 이동량만 쓴 경우(Y " << fmt(qe.y, 3) << " mm)보다 " << fmt(qe.y / combos[3].e.y, 0) << "배 정확했다.\n\n"
      << "| 지표 | B2 (2점) | QR 내부 이동량만 (" << qgood.size() << "점) | QR 내부 GLS (결과표 Q7×C5) |\n|---|---|---|---|\n"
      << "| 시험 장수 | " << ev_te.size() << " | " << ev_te.size() << " | " << n2 << " |\n"
      << "| Y RMS / 최대 \\|오차\\| (mm) | " << fmt(combos[3].e.y, 4) << " / " << fmt(max_abs(ey), 4) << " | " << fmt(qe.y, 4) << " / - | " << fmt(g7.yr, 4) << " / " << fmt(g7.yx, 4) << " |\n"
      << "| Θ RMS / 최대 \\|오차\\| | " << fmt(combos[3].e.t, 4) << " / " << fmt(max_abs(et), 4) << " | " << fmt(qe.t, 4) << " / - | " << fmt(g7.tr, 4) << " / " << fmt(g7.tx, 4) << " |\n"
      << "| Y 통과 (\\|eY\\|<=0.09) | " << cnt(ey) << " / " << ey.size() << " | - | - |\n"
      << "| T 통과 (\\|eT\\|<=0.09) | " << cnt(et) << " / " << et.size() << " | - | - |\n"
      << "| Y 0.05 mm 안 비율 (결과표) | " << fmt(grid_at("O3", "C2").yin, 1) << " % | - | " << fmt(g7.yin, 1) << " % |\n\n"
      << "조합 A(파인더 + 벽 볼트) Y " << fmt(combos[0].e.y, 3) << " mm, B(라벨 + 고리) " << fmt(combos[1].e.y, 3) << " mm, C(좋은 점 " << good.size() << "개) " << fmt(combos[2].e.y, 3)
      << " mm였다.\n\n" << img_md("01_특징점지도_비율색.png", "특징점 지도") << "\n\n" << img_md("02_특징점별_YT비율.png", "YT 비율") << "\n\n"
      << img_md("03_조합별_Y오차.png", "조합별 Y 오차") << "\n\n" << img_md("04_조합별_T오차.png", "조합별 T 오차") << "\n\n" << img_md("05_B2_시험141장_오차.png", "B2 오차") << "\n\n"
      << "## 튀는 값\n\nB2에서 가장 크게 틀린 사진은 Y는 " << names[wy] << "(정답 Y " << fmt(ty[wy], 2) << ", 추정 " << fmt(py[wy], 3) << " mm), T는 " << names[wt] << "(정답 T "
      << fmt(tt[wt], 2) << ", 추정 " << fmt(pt[wt], 4) << ")이고, 두 점의 이동이 직선 모델에서 " << fmts(resid(ev_te[wy], 0), 3) << " / " << fmts(resid(ev_te[wy], 1), 3)
      << " px 벗어난 만큼이 오차가 됐다.\n\n" << outl.md() << "\n" << img_md(fy, "Y 튀는 값") << "\n\n" << img_md(ft, "T 튀는 값")
      << "\n\n초록 상자·십자는 center에서의 점 위치, 빨간 상자·십자는 이 사진에서 잰 위치다. res는 직선 모델 대비 잔차다.\n\n"
      << "## 한줄평\n\nY와 Θ는 화면에서 거의 같은 수평 이동을 만들고 둘의 비율 b/a만 거리에 따라 달라서(QR 내부 " << fmt(qr_min, 4) << "~" << fmt(qr_max, 4) << ", B2 두 점 "
      << fmt(info[B2[0]].ratio, 4) << " / " << fmt(info[B2[1]].ratio, 4) << "), 비율 차이가 큰 가까운 점·먼 점 쌍(cond " << fmt(combos[3].e.cond, 0) << ")은 Y가 " << fmt(combos[3].e.y, 3)
      << " mm로 풀리고 비율 차이가 작은 QR 내부만으로는 cond " << fmt(qe.cond, 0) << "로 커져 Y가 " << fmt(qe.y, 2) << " mm까지 흔들렸다.\n\n"
      << "## 원래 결과와 대조\n\n원래 값은 `00000.legacies/00_결과요약.md`, `00_실험전체목록_요일별.md`, `analysis_v3/v1_data/combos.json`·`stable.json`에서 가져왔다. 결과표 19조합 대조는 결과.md에 있다(" << gpar.ok << "/" << gpar.total
      << " 일치). " << par.ok << "/" << par.total << " 일치.\n\n" << par.t.md()
      << "\nQR만 원근 왜곡 1장 값은 계산 코드가 남지 않아 재현하지 못했다(결과.md 옮기지 않은 부분).\n";
    write_text(out("보고서.md"), P.str());
    return 0;
}
