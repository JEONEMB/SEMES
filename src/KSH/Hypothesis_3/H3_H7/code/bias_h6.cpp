// =====================================================================================
// bias_h6.cpp — 가설 H6 : pixel-locking 진단과 보정
// =====================================================================================
#include "bias_h6.hpp"
#include <array>
#include <algorithm>
#include <cmath>
#include <limits>

namespace h3 {

static const double NaN = std::numeric_limits<double>::quiet_NaN();

std::vector<double> plainResidualX(const std::vector<cv::Point2d>& P0, const Track& t, const FieldSpec& s)
{
    const int N = (int)P0.size(), nb = (int)s.bx.size();
    int n = 0; for (uchar o : t.ok) n += o;
    cv::Mat A(n, nb, CV_64F), d(n, 1, CV_64F), Af(N, nb, CV_64F);
    for (int k = 0, r = 0; k < N; ++k) {
        double u = (P0[k].x - s.cx_px) / s.scale_px, v = (P0[k].y - s.cy_px) / s.scale_px;
        for (int j = 0; j < nb; ++j) Af.at<double>(k, j) = basisValue(s.bx[j], u, v);
        if (t.ok[k]) { Af.row(k).copyTo(A.row(r)); d.at<double>(r) = t.q[k].x - P0[k].x; ++r; }
    }
    cv::Mat c; cv::solve(A, d, c, cv::DECOMP_SVD);
    std::vector<double> res(N, NaN);
    for (int k = 0; k < N; ++k) if (t.ok[k]) {
        double pred = 0; for (int j = 0; j < nb; ++j) pred += Af.at<double>(k, j) * c.at<double>(j);
        res[k] = (t.q[k].x - P0[k].x) - pred;
    }
    return res;
}

static void harm(double ph, double h[4])
{
    h[0] = std::sin(2 * CV_PI * ph); h[1] = std::cos(2 * CV_PI * ph);
    h[2] = std::sin(4 * CV_PI * ph); h[3] = std::cos(4 * CV_PI * ph);
}
static double frac1(double x) { return x - std::floor(x); }

// 점별 평균 (NaN 제외)
static std::vector<double> pointMean(const std::vector<std::vector<double>>& R)
{
    const size_t N = R[0].size();
    std::vector<double> m(N, NaN);
    for (size_t k = 0; k < N; ++k) {
        double s = 0; int n = 0;
        for (const auto& row : R) if (!std::isnan(row[k])) { s += row[k]; ++n; }
        if (n) m[k] = s / n;
    }
    return m;
}
static double sdOf(const std::vector<double>& v)
{
    double s = 0, s2 = 0; int n = 0;
    for (double x : v) if (!std::isnan(x)) { s += x; s2 += x * x; ++n; }
    if (n < 2) return 0;
    double mu = s / n; return std::sqrt(std::max(0.0, s2 / n - mu * mu));
}
// 점별 곡선 계수 (rows 의 잔차 R2 와 위상 PH 로)
static cv::Mat fitPointCurves(const std::vector<std::vector<double>>& R2, const std::vector<std::vector<double>>& PH,
                              const std::vector<int>& rows)
{
    const int N = (int)R2[0].size();
    cv::Mat coef = cv::Mat::zeros(N, 4, CV_64F);
    for (int k = 0; k < N; ++k) {
        std::vector<int> use;
        for (int i : rows) if (!std::isnan(R2[i][k])) use.push_back(i);
        if ((int)use.size() <= 20) continue;
        cv::Mat X((int)use.size(), 4, CV_64F), y((int)use.size(), 1, CV_64F);
        for (int r = 0; r < (int)use.size(); ++r) {
            double h[4]; harm(PH[use[r]][k], h);
            for (int j = 0; j < 4; ++j) X.at<double>(r, j) = h[j];
            y.at<double>(r) = R2[use[r]][k];
        }
        cv::Mat c; cv::solve(X, y, c, cv::DECOMP_SVD);
        for (int j = 0; j < 4; ++j) coef.at<double>(k, j) = c.at<double>(j);
    }
    return coef;
}

H6Diag diagnoseH6(const std::vector<cv::Point2d>& P0, const std::vector<Track>& tracks,
                  const std::vector<int>& imgIdx, const FieldSpec& s)
{
    H6Diag d;
    std::vector<std::vector<double>> R, PH;
    for (int i : imgIdx) {
        R.push_back(plainResidualX(P0, tracks[i], s));
        std::vector<double> ph(P0.size());
        for (size_t k = 0; k < P0.size(); ++k) ph[k] = frac1(tracks[i].q[k].x);
        PH.push_back(ph);
    }
    std::vector<double> all;
    for (const auto& r : R) all.insert(all.end(), r.begin(), r.end());
    d.residSd = sdOf(all);
    std::vector<double> pm = pointMean(R);
    d.pointConstSd = sdOf(pm);
    std::vector<std::vector<double>> R2 = R;
    std::vector<double> rem;
    for (auto& r : R2) for (size_t k = 0; k < r.size(); ++k) { r[k] -= pm[k]; rem.push_back(r[k]); }
    d.remainSd = sdOf(rem);
    // 전역 곡선
    {
        std::vector<double> yv; std::vector<std::array<double, 4>> xv;
        for (size_t i = 0; i < R2.size(); ++i) for (size_t k = 0; k < P0.size(); ++k) if (!std::isnan(R2[i][k])) {
            std::array<double, 4> h; harm(PH[i][k], h.data()); xv.push_back(h); yv.push_back(R2[i][k]); }
        cv::Mat X((int)yv.size(), 4, CV_64F), y((int)yv.size(), 1, CV_64F);
        for (int r = 0; r < (int)yv.size(); ++r) { for (int j = 0; j < 4; ++j) X.at<double>(r, j) = xv[r][j]; y.at<double>(r) = yv[r]; }
        cv::Mat c; cv::solve(X, y, c, cv::DECOMP_SVD);
        cv::Mat e = y - X * c; cv::Scalar m1, s1, m2, s2; cv::meanStdDev(y, m1, s1); cv::meanStdDev(e, m2, s2);
        d.globalExplained = 1 - (s2[0] * s2[0]) / (s1[0] * s1[0]);
        d.globalAmp = std::hypot(c.at<double>(0), c.at<double>(1));
    }
    // 점별 곡선 : 짝수 번째 영상으로 맞추고 홀수 번째로 검증
    {
        std::vector<int> tr, te;
        for (int i = 0; i < (int)R2.size(); ++i) (i % 2 == 0 ? tr : te).push_back(i);
        cv::Mat coef = fitPointCurves(R2, PH, tr);
        double sy = 0, sy2 = 0, se = 0, se2 = 0; int n = 0;
        for (int i : te) for (size_t k = 0; k < P0.size(); ++k) if (!std::isnan(R2[i][k])) {
            double h[4]; harm(PH[i][k], h);
            double p = 0; for (int j = 0; j < 4; ++j) p += h[j] * coef.at<double>((int)k, j);
            double y = R2[i][k], e = y - p; sy += y; sy2 += y * y; se += e; se2 += e * e; ++n;
        }
        double vy = sy2 / n - (sy / n) * (sy / n), ve = se2 / n - (se / n) * (se / n);
        d.pointExplainedCV = 1 - ve / vy;
    }
    return d;
}

std::vector<Track> correctH6(const std::vector<cv::Point2d>& P0, const std::vector<Track>& tracks,
                             const std::vector<int>& trainIdx, const FieldSpec& s)
{
    std::vector<std::vector<double>> R, PH;
    for (int i : trainIdx) {
        R.push_back(plainResidualX(P0, tracks[i], s));
        std::vector<double> ph(P0.size());
        for (size_t k = 0; k < P0.size(); ++k) ph[k] = frac1(tracks[i].q[k].x);
        PH.push_back(ph);
    }
    std::vector<double> pm = pointMean(R);
    for (auto& r : R) for (size_t k = 0; k < r.size(); ++k) r[k] -= pm[k];
    std::vector<int> all(R.size()); for (size_t i = 0; i < R.size(); ++i) all[i] = (int)i;
    cv::Mat coef = fitPointCurves(R, PH, all);
    std::vector<Track> out = tracks;
    for (auto& t : out) for (size_t k = 0; k < P0.size(); ++k) {
        double h[4]; harm(frac1(t.q[k].x), h);
        double c = 0; for (int j = 0; j < 4; ++j) c += h[j] * coef.at<double>((int)k, j);
        t.q[k].x -= c;
    }
    return out;
}

} // namespace h3
