// =====================================================================================
// phys_model.cpp — 가설 H3 (완전 물리 모델)
// =====================================================================================
#include "phys_model.hpp"
#include <chrono>
#include <algorithm>
#include <cmath>

namespace h3 {

static cv::Vec3d unitVec(double a, double b, char base)
{
    if (base == 'x') return cv::Vec3d(-std::cos(b) * std::cos(a), std::sin(b), std::cos(b) * std::sin(a));   // 기본 −x
    if (base == 'y') return cv::Vec3d(std::sin(a), -std::cos(a) * std::cos(b), std::cos(a) * std::sin(b));   // 기본 −y
    return cv::Vec3d(std::sin(a) * std::cos(b), std::sin(b), std::cos(a) * std::cos(b));                     // 기본 +z
}

static cv::Matx33d rodrigues(cv::Vec3d axis, double th)
{
    axis = axis / cv::norm(axis);
    cv::Matx33d K(0, -axis[2], axis[1], axis[2], 0, -axis[0], -axis[1], axis[0], 0);
    return cv::Matx33d::eye() + std::sin(th) * K + (1 - std::cos(th)) * (K * K);
}

static inline double sgn(double x) { return (x > 0) - (x < 0); }

std::vector<cv::Vec3d> platePoints(const std::vector<cv::Point2d>& P0, const PhysParams& g)
{
    const double f = g["f"], cx = g["cx"], cy = g["cy"], k1 = g["k1"], d = g["d"];
    const cv::Vec3d n = unitVec(g["na"], g["nb"], 'z');
    std::vector<cv::Vec3d> X(P0.size());
    for (size_t k = 0; k < P0.size(); ++k) {
        double xd = (P0[k].x - cx) / f, yd = (P0[k].y - cy) / f, x = xd, y = yd;
        for (int it = 0; it < 5; ++it) { double s = 1 + k1 * (x * x + y * y); x = xd / s; y = yd / s; }
        cv::Vec3d r(x, y, 1.0);
        X[k] = r * (d / r.dot(n));
    }
    return X;
}

std::vector<cv::Point2d> projectPoints(const std::vector<cv::Vec3d>& X, double y_mm, double t_deg, double nu_px, const PhysParams& g)
{
    const double f = g["f"], cx = g["cx"], cy = g["cy"], k1 = g["k1"];
    const cv::Matx33d R = rodrigues(unitVec(g["w1"], g["w2"], 'y'), t_deg * CV_PI / 180.0);
    const cv::Vec3d e = unitVec(g["e1"], g["e2"], 'x'), A(g["ax"], 0.0, g["az"]);
    const cv::Vec3d c = y_mm * e + (cv::Matx33d::eye() - R) * A;     // Θ 단이 Y 단 위 (TonY)
    const cv::Matx33d Rt = R.t();
    std::vector<cv::Point2d> p(X.size());
    for (size_t k = 0; k < X.size(); ++k) {
        cv::Vec3d xc = Rt * (X[k] - c);
        double x = xc[0] / xc[2], y = xc[1] / xc[2], s = 1 + k1 * (x * x + y * y);
        p[k] = cv::Point2d(f * x * s + cx, f * y * s + cy + nu_px);
    }
    return p;
}

// ------------------------------------------------------------------ 학습 (LM + soft-L1)
namespace {
struct Problem {
    const std::vector<cv::Point2d>& P0;
    const std::vector<Track>& tracks;
    const std::vector<int>& idx;
    const std::vector<double>& y;
    const std::vector<double>& t;
    const std::vector<std::string>& free;
    PhysParams base;
    // 잔차 : 영상 순서, 점 순서, (x, y). 무효 점은 0
    void residual(const std::vector<double>& p, std::vector<double>& r) const {
        PhysParams g = base;
        for (size_t j = 0; j < free.size(); ++j) g.v[free[j]] = p[j];
        const std::vector<cv::Vec3d> X = platePoints(P0, g);
        const size_t N = P0.size();
        r.assign(idx.size() * N * 2, 0.0);
        for (size_t j = 0; j < idx.size(); ++j) {
            const Track& tr = tracks[idx[j]];
            double yl = y[j] + g["Y0"] + g["bY"] * sgn(y[j]);
            double tl = t[j] + g["T0"] + g["bT"] * sgn(t[j]);
            std::vector<cv::Point2d> pr = projectPoints(X, yl, tl, p[free.size() + j], g);
            for (size_t k = 0; k < N; ++k) if (tr.ok[k]) {
                r[(j * N + k) * 2]     = pr[k].x - tr.q[k].x;
                r[(j * N + k) * 2 + 1] = pr[k].y - tr.q[k].y;
            }
        }
    }
};
double softL1Cost(const std::vector<double>& r, double fs, std::vector<double>* w)
{
    double c = 0;
    if (w) w->resize(r.size());
    for (size_t i = 0; i < r.size(); ++i) {
        double z = (r[i] / fs) * (r[i] / fs);
        c += fs * fs * (std::sqrt(1 + z) - 1);             // = ½·fs²·ρ(z),  ρ = 2(√(1+z) − 1)
        if (w) (*w)[i] = 1.0 / std::sqrt(1 + z);
    }
    return c;
}
} // namespace

PhysFitResult fitPhysModel(const std::vector<cv::Point2d>& P0, const std::vector<Track>& tracks,
                           const std::vector<int>& imgIdx, const std::vector<double>& y, const std::vector<double>& t,
                           const std::vector<std::string>& freeNames, const PhysParams& init, int maxIter)
{
    auto t0 = std::chrono::steady_clock::now();
    const double FS = 0.1;                                       // soft-L1 척도 (px)
    Problem pb{ P0, tracks, imgIdx, y, t, freeNames, init };
    const int nG = (int)freeNames.size(), nI = (int)imgIdx.size(), nP = nG + nI, N = (int)P0.size();
    std::vector<double> p(nP, 0.0);
    for (int j = 0; j < nG; ++j) p[j] = init[freeNames[j]];

    std::vector<double> r, w, rh;
    pb.residual(p, r);
    double cost = softL1Cost(r, FS, &w), lambda = 1e-3;
    int it = 0;
    for (; it < maxIter; ++it) {
        // 전역 매개변수 수치 미분 (전진 차분)
        std::vector<std::vector<double>> JG(nG);
        for (int j = 0; j < nG; ++j) {
            double h = 1e-6 * std::max(std::fabs(p[j]), 1.0);
            std::vector<double> pp = p; pp[j] += h;
            pb.residual(pp, rh);
            JG[j].resize(r.size());
            for (size_t i = 0; i < r.size(); ++i) JG[j][i] = (rh[i] - r[i]) / h;
        }
        // 정규 방정식 (영상별 ν 는 해당 영상의 y 잔차에만 1)
        cv::Mat H = cv::Mat::zeros(nP, nP, CV_64F), g = cv::Mat::zeros(nP, 1, CV_64F);
        for (size_t i = 0; i < r.size(); ++i) {
            if (r[i] == 0.0) continue;
            const double wi = w[i];
            const int img = (int)(i / (2 * (size_t)N));
            const bool isY = (i % 2) == 1;
            for (int a = 0; a < nG; ++a) {
                const double ja = JG[a][i];
                g.at<double>(a) += wi * ja * r[i];
                for (int b = a; b < nG; ++b) H.at<double>(a, b) += wi * ja * JG[b][i];
                if (isY) H.at<double>(a, nG + img) += wi * ja;
            }
            if (isY) { H.at<double>(nG + img, nG + img) += wi; g.at<double>(nG + img) += wi * r[i]; }
        }
        for (int a = 0; a < nP; ++a) for (int b = 0; b < a; ++b) H.at<double>(a, b) = H.at<double>(b, a);
        bool accepted = false;
        for (int tries = 0; tries < 12; ++tries) {
            cv::Mat A = H.clone();
            for (int a = 0; a < nP; ++a) A.at<double>(a, a) += lambda * std::max(H.at<double>(a, a), 1e-12);
            cv::Mat step;
            cv::solve(A, -g, step, cv::DECOMP_CHOLESKY);
            std::vector<double> pn = p;
            for (int a = 0; a < nP; ++a) pn[a] += step.at<double>(a);
            std::vector<double> rn, wn;
            pb.residual(pn, rn);
            double cn = softL1Cost(rn, FS, &wn);
            if (cn < cost) {
                double rel = (cost - cn) / cost;
                p = pn; r = rn; w = wn; cost = cn; lambda = std::max(lambda / 3.0, 1e-9); accepted = true;
                if (rel < 1e-9) it = maxIter;
                break;
            }
            lambda *= 4.0;
        }
        if (!accepted) break;
    }
    PhysFitResult out;
    out.g = init;
    for (int j = 0; j < nG; ++j) out.g.v[freeNames[j]] = p[j];
    out.nu.assign(p.begin() + nG, p.end());
    double sx = 0, sy = 0; size_t n = 0;
    for (size_t i = 0; i < r.size(); i += 2) { sx += r[i] * r[i]; sy += r[i + 1] * r[i + 1]; ++n; }
    out.rmsX = std::sqrt(sx / n); out.rmsY = std::sqrt(sy / n);          // 파이썬과 같이 무효 점(0) 포함 평균
    out.cost = cost; out.iters = it;
    out.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return out;
}

std::vector<ResidualRow> residualDiag(const std::vector<cv::Point2d>& P0, const std::vector<Track>& tracks,
                                      const std::vector<int>& imgIdx, const std::vector<double>& y,
                                      const std::vector<double>& t, const PhysParams& g)
{
    std::vector<ResidualRow> out;
    const std::vector<cv::Vec3d> X = platePoints(P0, g);
    for (size_t j = 0; j < imgIdx.size(); ++j) {
        const Track& tr = tracks[imgIdx[j]];
        double yl = y[j] + g["Y0"] + g["bY"] * sgn(y[j]), tl = t[j] + g["T0"] + g["bT"] * sgn(t[j]);
        std::vector<cv::Point2d> pr = projectPoints(X, yl, tl, 0.0, g);
        int n = 0; for (uchar o : tr.ok) n += o;
        cv::Mat A(n, 3, CV_64F), d(n, 1, CV_64F);
        for (size_t k = 0, r = 0; k < P0.size(); ++k) if (tr.ok[k]) {
            double s = (P0[k].x - 1900.0) / 400.0;
            A.at<double>((int)r, 0) = 1; A.at<double>((int)r, 1) = s; A.at<double>((int)r, 2) = s * s;
            d.at<double>((int)r) = pr[k].x - tr.q[k].x; ++r;
        }
        cv::Mat c; cv::solve(A, d, c, cv::DECOMP_SVD);
        cv::Mat res = d - A * c; cv::Scalar mu, sd; cv::meanStdDev(res, mu, sd);
        out.push_back({ y[j], t[j], c.at<double>(0), c.at<double>(1), c.at<double>(2), sd[0] });
    }
    return out;
}

} // namespace h3
