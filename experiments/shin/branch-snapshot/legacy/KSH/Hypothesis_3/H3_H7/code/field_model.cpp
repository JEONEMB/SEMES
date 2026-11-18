// =====================================================================================
// field_model.cpp — 가설 H3 (선형화 물리 모델)
// =====================================================================================
#include "field_model.hpp"
#include <algorithm>
#include <cmath>
#include <limits>

namespace h3 {

const char* basisName(Basis b)
{
    static const char* n[] = { "1", "u", "u2", "v", "uv", "v2", "u3" };
    return n[(int)b];
}

std::string FieldSpec::label() const
{
    std::string s = "dx[";
    for (size_t i = 0; i < bx.size(); ++i) s += std::string(i ? "," : "") + basisName(bx[i]);
    s += "] dy[";
    for (size_t i = 0; i < by.size(); ++i) s += std::string(i ? "," : "") + basisName(by[i]);
    return s + "]" + (quad ? " +Θ²" : "");
}

double basisValue(Basis b, double u, double v)
{
    switch (b) {
    case B_ONE: return 1.0;
    case B_U:   return u;
    case B_U2:  return u * u;
    case B_V:   return v;
    case B_UV:  return u * v;
    case B_V2:  return v * v;
    case B_U3:  return u * u * u;
    }
    return 0.0;
}

static double median(std::vector<double> v)
{
    if (v.empty()) return 0.0;
    size_t m = v.size() / 2;
    std::nth_element(v.begin(), v.begin() + m, v.end());
    double hi = v[m];
    if (v.size() % 2) return hi;
    double lo = *std::max_element(v.begin(), v.begin() + m);
    return 0.5 * (lo + hi);
}

std::vector<double> fieldCoeffs(const std::vector<cv::Point2d>& P0, const Track& t, const FieldSpec& s,
                                std::vector<double>* resX)
{
    std::vector<double> out;
    const int N = (int)P0.size();
    for (int comp = 0; comp < 2; ++comp) {
        const std::vector<Basis>& bl = comp == 0 ? s.bx : s.by;
        if (bl.empty()) continue;
        const int nb = (int)bl.size();
        cv::Mat A(N, nb, CV_64F), d(N, 1, CV_64F);
        for (int k = 0; k < N; ++k) {
            double u = (P0[k].x - s.cx_px) / s.scale_px, v = (P0[k].y - s.cy_px) / s.scale_px;
            for (int j = 0; j < nb; ++j) A.at<double>(k, j) = basisValue(bl[j], u, v);
            d.at<double>(k) = comp == 0 ? t.q[k].x - P0[k].x : t.q[k].y - P0[k].y;
        }
        std::vector<uchar> w(t.ok.begin(), t.ok.end());
        cv::Mat c;
        for (int it = 0; it < 3; ++it) {
            int n = 0; for (uchar x : w) n += x;
            cv::Mat Aw(n, nb, CV_64F), dw(n, 1, CV_64F);
            for (int k = 0, r = 0; k < N; ++k) if (w[k]) { A.row(k).copyTo(Aw.row(r)); dw.at<double>(r) = d.at<double>(k); ++r; }
            cv::solve(Aw, dw, c, cv::DECOMP_SVD);
            cv::Mat res = d - A * c;
            std::vector<double> ar;
            for (int k = 0; k < N; ++k) if (t.ok[k]) ar.push_back(std::fabs(res.at<double>(k)));
            double sig = 1.4826 * median(ar) + 1e-9;
            for (int k = 0; k < N; ++k) w[k] = (t.ok[k] && std::fabs(res.at<double>(k)) < 3 * sig) ? 1 : 0;
            if (it == 2 && comp == 0 && resX) {
                resX->assign(N, std::numeric_limits<double>::quiet_NaN());
                for (int k = 0; k < N; ++k) if (t.ok[k]) (*resX)[k] = res.at<double>(k);
            }
        }
        for (int j = 0; j < nb; ++j) out.push_back(c.at<double>(j));
    }
    return out;
}

bool FieldModel::fit(const std::vector<std::vector<double>>& K, const std::vector<double>& y, const std::vector<double>& t, bool quad)
{
    const int N = (int)K.size();
    if (N < 6) return false;
    const int nk = (int)K[0].size(), na = quad ? 4 : 3;
    quad_ = quad;
    cv::Mat A(N, na, CV_64F), D(N, nk, CV_64F);
    for (int i = 0; i < N; ++i) {
        A.at<double>(i, 0) = y[i]; A.at<double>(i, 1) = t[i];
        if (quad) A.at<double>(i, 2) = t[i] * t[i];
        A.at<double>(i, na - 1) = 1.0;
        for (int j = 0; j < nk; ++j) D.at<double>(i, j) = K[i][j];
    }
    cv::solve(A, D, M_, cv::DECOMP_SVD);
    cv::Mat R = D - A * M_, mu;
    cv::calcCovarMatrix(R, C_, mu, cv::COVAR_NORMAL | cv::COVAR_ROWS, CV_64F);
    C_ = C_ / (double)(N - 1) + cv::Mat::eye(nk, nk, CV_64F) * 1e-10;
    cv::invert(C_, W_, cv::DECOMP_SVD);
    return true;
}

Estimate FieldModel::estimate(const std::vector<double>& k) const
{
    Estimate e;
    if (M_.empty() || (int)k.size() != M_.cols) return e;
    const int nk = M_.cols, last = M_.rows - 1;
    cv::Mat kv(nk, 1, CV_64F);
    for (int j = 0; j < nk; ++j) kv.at<double>(j) = k[j];
    double y = 0, t = 0;
    for (int it = 0; it < 10; ++it) {
        cv::Mat J(nk, 2, CV_64F), r(nk, 1, CV_64F);
        for (int j = 0; j < nk; ++j) {
            double a = M_.at<double>(0, j), b = M_.at<double>(1, j), q = quad_ ? M_.at<double>(2, j) : 0.0, c = M_.at<double>(last, j);
            J.at<double>(j, 0) = a;
            J.at<double>(j, 1) = b + 2 * q * t;
            r.at<double>(j) = kv.at<double>(j) - (a * y + b * t + q * t * t + c);
        }
        cv::Mat H = J.t() * W_ * J, step;
        cv::solve(H, J.t() * W_ * r, step, cv::DECOMP_SVD);
        y += step.at<double>(0); t += step.at<double>(1);
    }
    e.y_mm = y; e.t_deg = t; e.valid = std::isfinite(y) && std::isfinite(t);
    return e;
}

Estimate FieldModel::estimateRef(const std::vector<double>& k, const std::vector<double>& kRef) const
{
    std::vector<double> kk(k.size());
    const int last = M_.rows - 1;
    for (size_t j = 0; j < k.size(); ++j) kk[j] = k[j] - kRef[j] + M_.at<double>(last, (int)j);
    return estimate(kk);
}

bool FieldModel::save(const std::string& path, const FieldSpec& s, const std::string& note) const
{
    cv::FileStorage fs(path, cv::FileStorage::WRITE);
    if (!fs.isOpened()) return false;
    fs << "model_type" << "FIELD_BASIS_v1";
    fs << "note" << note;
    fs << "basis" << s.label();
    fs << "center_px" << std::vector<double>{ s.cx_px, s.cy_px } << "scale_px" << s.scale_px << "quad" << (quad_ ? 1 : 0);
    fs << "M_rows" << (quad_ ? "Y, Theta, Theta2, 1" : "Y, Theta, 1");
    fs << "M" << M_ << "cov" << C_;
    return true;
}

} // namespace h3
