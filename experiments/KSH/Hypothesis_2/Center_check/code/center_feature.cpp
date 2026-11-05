// =====================================================================================
// center_feature.cpp — 기준(center) 영상 특징 + 깊이별 랜드마크 시차(parallax)
// =====================================================================================
#include "center_feature.hpp"
#include <chrono>
#include <cmath>
#include <algorithm>

namespace cf {

// ---------------------------------------------------------------------------------
// 랜드마크 기본값 (center.bmp, 3840×2160)
// ---------------------------------------------------------------------------------
std::vector<Landmark> defaultLandmarks()
{
    auto L = [](const char* n, const char* g, int cx, int cy, int hw, int hh, bool use = true) {
        Landmark l; l.name = n; l.group = g; l.cx = cx; l.cy = cy; l.hw = hw; l.hh = hh; l.use = use; return l; };
    return {
        L("QR_L",          "plate",      1650, 1065, 110, 110),
        L("QR_C",          "plate",      1907, 1065, 110, 110),
        L("QR_R",          "plate",      2165, 1065, 110, 110),
        L("label",         "plate",      1900, 1283, 150,  70),
        L("nut_L",         "station",    1547,  707,  60,  60, false),   // 반사·그림자로 잡음 큼 → 기본 미사용
        L("nut_R",         "station",    2240,  707,  60,  60, false),
        L("bracket_scr_L", "station",    1627,  550,  70,  70, false),   // 깊이가 플레이트와 비슷 → 정보 적음
        L("bracket_scr_R", "station",    2167,  550,  70,  70, false),
        L("wall_U1",       "wall_upper", 2483,  367,  60,  60),
        L("wall_U2",       "wall_upper", 2483,  617,  60,  60),
        L("wall_W1",       "wall_lower",  967, 1357,  60,  60),
        L("wall_W2",       "wall_lower", 2867, 1357,  60,  60),
        L("wall_W3",       "wall_lower", 1010, 1677,  60,  60),
        L("wall_W4",       "wall_lower", 2820, 1683,  60,  60),
    };
}

bool saveLandmarks(const std::string& path, const std::vector<Landmark>& lm)
{
    cv::FileStorage fs(path, cv::FileStorage::WRITE);
    if (!fs.isOpened()) return false;
    fs << "note" << "ROI in center.bmp pixels. group = plate / station / wall_upper / wall_lower. use = 1 to include in the model";
    fs << "landmarks" << "[";
    for (const Landmark& l : lm)
        fs << "{" << "name" << l.name << "group" << l.group << "cx" << l.cx << "cy" << l.cy
           << "hw" << l.hw << "hh" << l.hh << "use" << (l.use ? 1 : 0) << "}";
    fs << "]";
    return true;
}

bool loadLandmarks(const std::string& path, std::vector<Landmark>& lm)
{
    try {
        cv::FileStorage fs(path, cv::FileStorage::READ);
        if (!fs.isOpened()) return false;
        cv::FileNode n = fs["landmarks"];
        if (n.type() != cv::FileNode::SEQ) return false;
        std::vector<Landmark> out;
        for (cv::FileNodeIterator it = n.begin(); it != n.end(); ++it) {
            Landmark l;
            (*it)["name"] >> l.name; (*it)["group"] >> l.group;
            l.cx = (int)(*it)["cx"]; l.cy = (int)(*it)["cy"]; l.hw = (int)(*it)["hw"]; l.hh = (int)(*it)["hh"];
            l.use = (int)(*it)["use"] != 0;
            out.push_back(l);
        }
        if (out.empty()) return false;
        lm = out;
        return true;
    } catch (const cv::Exception&) { return false; }
}

// ---------------------------------------------------------------------------------
// 위상 상관 + 행렬곱 DFT 업샘플링 (Guizar-Sicairos 2008)
// ---------------------------------------------------------------------------------
static inline int fftFreq(int k, int n) { return k < (n + 1) / 2 ? k : k - n; }

Shift xcorrShift(const cv::Mat& a64, const cv::Mat& b64, int uf)
{
    Shift r;
    if (a64.size() != b64.size() || a64.empty()) return r;
    const int R = a64.rows, C = a64.cols;
    cv::Mat win;
    cv::createHanningWindow(win, a64.size(), CV_64F);
    cv::Mat a = (a64 - cv::mean(a64)[0]).mul(win), b = (b64 - cv::mean(b64)[0]).mul(win);
    cv::Mat A, B, X;
    cv::dft(a, A, cv::DFT_COMPLEX_OUTPUT);
    cv::dft(b, B, cv::DFT_COMPLEX_OUTPUT);
    cv::mulSpectrums(B, A, X, 0, true);                       // X = B · conj(A)
    {   // 정규화 (위상만)
        double mx = 0;
        for (int y = 0; y < R; ++y) { const cv::Vec2d* p = X.ptr<cv::Vec2d>(y);
            for (int x = 0; x < C; ++x) mx = std::max(mx, std::hypot(p[x][0], p[x][1])); }
        const double e = 1e-9 * mx;
        for (int y = 0; y < R; ++y) { cv::Vec2d* p = X.ptr<cv::Vec2d>(y);
            for (int x = 0; x < C; ++x) { double m = std::hypot(p[x][0], p[x][1]) + e; p[x][0] /= m; p[x][1] /= m; } }
    }
    // 1) 정수 위치
    cv::Mat cc;
    cv::idft(X, cc, cv::DFT_COMPLEX_OUTPUT | cv::DFT_SCALE);
    int py = 0, px = 0; double best = -1;
    for (int y = 0; y < R; ++y) { const cv::Vec2d* p = cc.ptr<cv::Vec2d>(y);
        for (int x = 0; x < C; ++x) { double m = p[x][0] * p[x][0] + p[x][1] * p[x][1]; if (m > best) { best = m; py = y; px = x; } } }
    double sy = py > R / 2 ? py - R : py, sx = px > C / 2 ? px - C : px;

    // 2) 정수 위치 주변 1.5 px 를 1/uf 간격으로 : M = Kr · conj(X) · Kc
    const int U = (int)std::ceil(uf * 1.5);
    const double ds = std::floor(U / 2.0);
    const double offR = ds - sy * uf, offC = ds - sx * uf;
    cv::Mat Kc(C, U, CV_64FC2), Kr(U, R, CV_64FC2), Xc(R, C, CV_64FC2);
    for (int c = 0; c < C; ++c) { cv::Vec2d* p = Kc.ptr<cv::Vec2d>(c);
        for (int u = 0; u < U; ++u) { double ph = -2 * CV_PI / ((double)C * uf) * fftFreq(c, C) * (u - offC);
            p[u] = cv::Vec2d(std::cos(ph), std::sin(ph)); } }
    for (int u = 0; u < U; ++u) { cv::Vec2d* p = Kr.ptr<cv::Vec2d>(u);
        for (int rr = 0; rr < R; ++rr) { double ph = -2 * CV_PI / ((double)R * uf) * (u - offR) * fftFreq(rr, R);
            p[rr] = cv::Vec2d(std::cos(ph), std::sin(ph)); } }
    for (int y = 0; y < R; ++y) { const cv::Vec2d* s = X.ptr<cv::Vec2d>(y); cv::Vec2d* d = Xc.ptr<cv::Vec2d>(y);
        for (int x = 0; x < C; ++x) d[x] = cv::Vec2d(s[x][0], -s[x][1]); }
    cv::Mat T, M;
    cv::gemm(Xc, Kc, 1.0, cv::noArray(), 0.0, T);
    cv::gemm(Kr, T, 1.0, cv::noArray(), 0.0, M);
    int my = 0, mx = 0; best = -1;
    for (int y = 0; y < U; ++y) { const cv::Vec2d* p = M.ptr<cv::Vec2d>(y);
        for (int x = 0; x < U; ++x) { double m = p[x][0] * p[x][0] + p[x][1] * p[x][1]; if (m > best) { best = m; my = y; mx = x; } } }
    r.dy = sy + (my - ds) / uf;
    r.dx = sx + (mx - ds) / uf;
    r.peak = std::sqrt(best) / ((double)R * C);
    r.ok = true;
    return r;
}

// ---------------------------------------------------------------------------------
// 랜드마크 추적
// ---------------------------------------------------------------------------------
static const cv::Rect COARSE_BAND(1190, 950, 1440, 230);   // 플레이트 띠 (거친 이동용, 좌우 여유 포함)

bool LandmarkTracker::setReference(const cv::Mat& gray8, const std::vector<Landmark>& lm)
{
    if (gray8.empty()) return false;
    gray8.convertTo(ref64_, CV_64F);
    lm_ = lm;
    tpl_.clear();
    const cv::Rect img(0, 0, ref64_.cols, ref64_.rows);
    for (const Landmark& l : lm_) {
        cv::Rect r(l.cx - l.hw, l.cy - l.hh, 2 * l.hw, 2 * l.hh);
        if ((r & img) != r) return false;
        tpl_.push_back(ref64_(r).clone());
    }
    return (COARSE_BAND & img) == COARSE_BAND;
}

Measure LandmarkTracker::measure(const cv::Mat& gray8) const
{
    auto t0 = std::chrono::steady_clock::now();
    Measure m;
    if (ref64_.empty() || gray8.size() != ref64_.size()) return m;
    cv::Mat g64;
    gray8.convertTo(g64, CV_64F);
    const cv::Rect img(0, 0, g64.cols, g64.rows);

    Shift cs = xcorrShift(ref64_(COARSE_BAND), g64(COARSE_BAND), 10);
    m.coarseDx = cs.dx; m.coarseDy = cs.dy;
    m.s.resize(lm_.size());
    bool all = true;
    for (size_t k = 0; k < lm_.size(); ++k) {
        const Landmark& l = lm_[k];
        double sx = cs.dx, sy = cs.dy;
        Shift res;
        for (int it = 0; it < 3; ++it) {
            int ix = (int)std::lround(sx), iy = (int)std::lround(sy);
            cv::Rect r(l.cx - l.hw + ix, l.cy - l.hh + iy, 2 * l.hw, 2 * l.hh);
            if ((r & img) != r) { res = Shift(); break; }
            Shift s = xcorrShift(tpl_[k], g64(r));
            sx = ix + s.dx; sy = iy + s.dy;
            res = s; res.dx = sx; res.dy = sy;
            if (std::fabs(s.dx) < 0.5 && std::fabs(s.dy) < 0.5) break;
        }
        m.s[k] = res;
        if (l.use && !res.ok) all = false;
    }
    m.ok = all;
    m.ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return m;
}

// ---------------------------------------------------------------------------------
// 영상 품질
// ---------------------------------------------------------------------------------
Quality quality(const cv::Mat& gray8, const cv::Rect& roi)
{
    Quality q;
    cv::Mat g = gray8(roi & cv::Rect(0, 0, gray8.cols, gray8.rows));
    q.mean = cv::mean(g)[0];
    cv::Mat lap; cv::Laplacian(g, lap, CV_64F);
    cv::Scalar mu, sd; cv::meanStdDev(lap, mu, sd); q.sharp = sd[0] * sd[0];
    // Immerkær : 마스크 [1 -2 1; -2 4 -2; 1 -2 1] 응답의 평균 절댓값 · sqrt(π/2) / 6
    cv::Mat k = (cv::Mat_<double>(3, 3) << 1, -2, 1, -2, 4, -2, 1, -2, 1), r;
    cv::filter2D(g, r, CV_64F, k);
    q.noise = std::sqrt(CV_PI / 2.0) / 6.0 * cv::mean(cv::abs(r))[0];
    q.satFrac = (double)cv::countNonZero(g >= 250) / (double)g.total();
    return q;
}

// ---------------------------------------------------------------------------------
// 시차 모델
// ---------------------------------------------------------------------------------
ParallaxModel fitModel(const std::vector<Sample>& train, const std::vector<int>& idx,
                       const std::vector<Landmark>& lm, const std::string& note)
{
    ParallaxModel m;
    const int K = (int)idx.size(), N = (int)train.size();
    if (K < 2 || N < 6) return m;
    cv::Mat A(N, 4, CV_64F), D(N, K, CV_64F);
    for (int i = 0; i < N; ++i) {
        const Sample& s = train[i];
        A.at<double>(i, 0) = s.y; A.at<double>(i, 1) = s.t; A.at<double>(i, 2) = s.t * s.t; A.at<double>(i, 3) = 1.0;
        for (int k = 0; k < K; ++k) D.at<double>(i, k) = s.dx[idx[k]];
    }
    cv::Mat P;
    cv::solve(A, D, P, cv::DECOMP_SVD);                       // 4 × K
    cv::Mat res = D - A * P;
    cv::Mat cov, mu;
    cv::calcCovarMatrix(res, cov, mu, cv::COVAR_NORMAL | cv::COVAR_ROWS | cv::COVAR_SCALE, CV_64F);
    cov = cov * ((double)N / (N - 1)) + cv::Mat::eye(K, K, CV_64F) * 1e-8;
    m.idx = idx; m.cov = cov; m.nTrain = N; m.note = note;
    for (int k = 0; k < K; ++k) {
        m.names.push_back(lm[idx[k]].name);
        m.a.push_back(P.at<double>(0, k)); m.b.push_back(P.at<double>(1, k));
        m.q.push_back(P.at<double>(2, k)); m.c.push_back(P.at<double>(3, k));
    }
    m.ok = true;
    return m;
}

Estimate estimate(const ParallaxModel& m, const std::vector<double>& dx, const std::vector<double>* refDx)
{
    Estimate e;
    if (!m.ok) return e;
    const int K = (int)m.idx.size();
    cv::Mat W; cv::invert(m.cov, W, cv::DECOMP_SVD);
    cv::Mat d(K, 1, CV_64F);
    for (int k = 0; k < K; ++k) {
        int i = m.idx[k];
        if (i < 0 || i >= (int)dx.size()) return e;
        d.at<double>(k) = refDx ? dx[i] - (*refDx)[i] + m.c[k] : dx[i];
    }
    double y = 0, t = 0;
    cv::Mat H;
    for (int it = 0; it < 10; ++it) {
        cv::Mat J(K, 2, CV_64F), r(K, 1, CV_64F);
        for (int k = 0; k < K; ++k) {
            J.at<double>(k, 0) = m.a[k];
            J.at<double>(k, 1) = m.b[k] + 2 * m.q[k] * t;
            r.at<double>(k) = d.at<double>(k) - (m.a[k] * y + m.b[k] * t + m.q[k] * t * t + m.c[k]);
        }
        H = J.t() * W * J;
        cv::Mat step; cv::solve(H, J.t() * W * r, step, cv::DECOMP_SVD);
        y += step.at<double>(0); t += step.at<double>(1);
        if (std::fabs(step.at<double>(0)) < 1e-9 && std::fabs(step.at<double>(1)) < 1e-9) break;
    }
    cv::Mat Cv = H.inv(cv::DECOMP_SVD);
    e.y = y; e.t = t;
    e.sdY = std::sqrt(std::max(0.0, Cv.at<double>(0, 0)));
    e.sdT = std::sqrt(std::max(0.0, Cv.at<double>(1, 1)));
    e.valid = std::isfinite(y) && std::isfinite(t);
    return e;
}

bool ParallaxModel::save(const std::string& path) const
{
    cv::FileStorage fs(path, cv::FileStorage::WRITE);
    if (!fs.isOpened()) return false;
    fs << "model_type" << "PARALLAX_LANDMARK_v1";
    fs << "note" << note;
    fs << "model_form" << "dx_k = a_k Y + b_k Theta + q_k Theta2 + c_k (dx px, Y mm, Theta deg)";
    fs << "nTrain" << nTrain;
    fs << "names" << "[";
    for (const std::string& n : names) fs << n;
    fs << "]";
    fs << "idx" << idx << "a" << a << "b" << b << "q" << q << "c" << c << "cov" << cov;
    return true;
}

bool ParallaxModel::load(const std::string& path)
{
    try {
        cv::FileStorage fs(path, cv::FileStorage::READ);
        if (!fs.isOpened()) return false;
        std::string type; fs["model_type"] >> type;
        if (type != "PARALLAX_LANDMARK_v1") return false;
        fs["note"] >> note; nTrain = (int)fs["nTrain"];
        names.clear();
        cv::FileNode n = fs["names"];
        for (cv::FileNodeIterator it = n.begin(); it != n.end(); ++it) names.push_back((std::string)*it);
        fs["idx"] >> idx; fs["a"] >> a; fs["b"] >> b; fs["q"] >> q; fs["c"] >> c; fs["cov"] >> cov;
        ok = !idx.empty() && a.size() == idx.size() && cov.rows == (int)idx.size();
        return ok;
    } catch (const cv::Exception&) { return false; }
}

} // namespace cf
