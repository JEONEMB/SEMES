#pragma once
#include <opencv2/opencv.hpp>
#include <cmath>
#include <functional>
#include <vector>
#include <algorithm>

namespace subpixel
{
constexpr double pi = 3.14159265358979323846;
struct Sample
{
    double x, y, v;
};
using Patch = std::vector<Sample>;
struct Fit
{
    cv::Point2d p;
    bool valid = false, boundary = false, converged = false;
    int boundaryMask = 0;
    double rms = 0, condition = 0, sigma = 0;
    int iterations = 0;
};
using Objective = std::function<void(const cv::Mat &, cv::Mat &, cv::Mat &)>;
inline Fit optimize(const cv::Mat &start, const Objective &objective, const cv::Mat &lower,
                    const cv::Mat &upper, int maxit = 60)
{
    cv::Mat p = start.clone(), r, j;
    objective(p, r, j);
    double cost = r.dot(r), lambda = 1e-3;
    Fit out;
    for (int it = 0; it < maxit; ++it)
    {
        out.iterations = it + 1;
        cv::Mat H = j.t() * j, g = j.t() * r;
        // KKT active set: an outward gradient at a bound is stationary
        // in that constrained direction, not evidence of failed convergence.
        for (int k = 0; k < p.rows; ++k)
        {
            double x = p.at<double>(k), gradient = g.at<double>(k);
            bool active = (x - lower.at<double>(k) < 1e-8 && gradient > 0) ||
                          (upper.at<double>(k) - x < 1e-8 && gradient < 0);
            if (active)
            {
                H.row(k).setTo(0);
                H.col(k).setTo(0);
                H.at<double>(k, k) = 1;
                g.at<double>(k) = 0;
            }
        }
        if (cv::norm(g, cv::NORM_INF) < 1e-10)
        {
            out.converged = true;
            break;
        }
        cv::Mat damp = H.clone();
        for (int k = 0; k < p.rows; ++k)
        {
            damp.at<double>(k, k) += lambda * std::max(1e-7, H.at<double>(k, k));
        }
        cv::Mat step;
        if (!cv::solve(damp, -g, step, cv::DECOMP_CHOLESKY))
        {
            lambda *= 10;
            continue;
        }
        cv::Mat trial = p + step;
        for (int k = 0; k < p.rows; ++k)
        {
            trial.at<double>(k) =
                std::clamp(trial.at<double>(k), lower.at<double>(k), upper.at<double>(k));
        }
        cv::Mat tr, tj;
        objective(trial, tr, tj);
        double nc = tr.dot(tr);
        if (std::isfinite(nc) && nc < cost)
        {
            double change = cv::norm(trial - p), improvement = cost - nc;
            p = trial;
            r = tr;
            j = tj;
            cost = nc;
            lambda = std::max(1e-10, lambda / 3);
            if (change < 1e-6 || improvement < 1e-13)
            {
                out.converged = true;
                break;
            }
        }
        else
        {
            lambda *= 5;
            if (lambda > 1e14)
            {
                break;
            }
        }
    }
    cv::Mat norms(1, j.cols, CV_64F);
    cv::Mat scaled = j.clone();
    for (int k = 0; k < j.cols; ++k)
    {
        double n = cv::norm(j.col(k));
        if (n > 1e-12)
        {
            scaled.col(k) /= n;
        }
    }
    cv::SVD svd(scaled, cv::SVD::NO_UV);
    out.condition = svd.w.at<double>(0) / std::max(1e-15, svd.w.at<double>(svd.w.rows - 1));
    out.p = {p.at<double>(0), p.at<double>(1)};
    out.rms = std::sqrt(cost / r.rows);
    for (int k = 0; k < p.rows; ++k)
    {
        if (std::min(p.at<double>(k) - lower.at<double>(k), upper.at<double>(k) - p.at<double>(k)) <
            1e-5)
        {
            out.boundaryMask |= (1 << k);
        }
    }
    out.boundary = out.boundaryMask != 0;
    // A constrained blur/angle optimum remains a valid position measurement.
    // Position bounds themselves indicate that localization escaped the search.
    out.valid = out.converged && !(out.boundaryMask & 3) && std::isfinite(out.condition) &&
                out.condition < 1e5;
    if (p.rows > 2)
    {
        out.sigma = std::exp(p.at<double>(4));
    }
    return out;
}
inline Patch patch(const cv::Mat &gray, cv::Point2d initial, int radius)
{
    Patch v;
    int cx = cvRound(initial.x), cy = cvRound(initial.y);
    if (cx - radius < 0 || cy - radius < 0 || cx + radius >= gray.cols || cy + radius >= gray.rows)
    {
        return v;
    }
    for (int y = cy - radius; y <= cy + radius; ++y)
    {
        for (int x = cx - radius; x <= cx + radius; ++x)
        {
            v.push_back({x - initial.x, y - initial.y, gray.at<uchar>(y, x) / 255.});
        }
    }
    return v;
}
// Exact Gaussian convolution of two oriented sign functions, using the
// bivariate normal correlation integral. NOT the unpublished approximations
// in Krueger & Woehler. The public seven-parameter image model is retained.
inline void corner(double x, double y, double a1, double a2, double logs, double &value,
                   double *derivatives)
{
    double sigma = std::exp(logs), c1 = std::cos(a1), s1 = std::sin(a1), c2 = std::cos(a2),
           s2 = std::sin(a2);
    double a = (c1 * x + s1 * y) / sigma, b = (c2 * x + s2 * y) / sigma, rho = std::cos(a1 - a2);
    double ea = std::erf(a / std::sqrt(2.)), eb = std::erf(b / std::sqrt(2.));
    // 12-point Gauss-Legendre integration from 0 to rho (angles bounded).
    static const double nodes[] = {-.9815606342467192, -.9041172563704749, -.7699026741943047,
                                   -.5873179542866175, -.3678314989981802, -.1252334085114689,
                                   .1252334085114689,  .3678314989981802,  .5873179542866175,
                                   .7699026741943047,  .9041172563704749,  .9815606342467192};
    static const double weights[] = {.0471753363865118, .1069393259953184, .1600783285433462,
                                     .2031674267230659, .2334925365383548, .2491470458134028,
                                     .2491470458134028, .2334925365383548, .2031674267230659,
                                     .1600783285433462, .1069393259953184, .0471753363865118};
    double integral = 0;
    for (int k = 0; k < 12; ++k)
    {
        double t = rho * (nodes[k] + 1) / 2, d = 1 - t * t;
        integral +=
            weights[k] * std::exp(-(a * a - 2 * t * a * b + b * b) / (2 * d)) / std::sqrt(d);
    }
    value = ea * eb + rho * integral / pi;
    double d = 1 - rho * rho;
    double da =
        2 / std::sqrt(2 * pi) * std::exp(-a * a / 2) * std::erf((b - rho * a) / std::sqrt(2 * d));
    double db =
        2 / std::sqrt(2 * pi) * std::exp(-b * b / 2) * std::erf((a - rho * b) / std::sqrt(2 * d));
    double dr = 2 / (pi * std::sqrt(d)) * std::exp(-(a * a - 2 * rho * a * b + b * b) / (2 * d));
    derivatives[0] = -(da * c1 + db * c2) / sigma;
    derivatives[1] = -(da * s1 + db * s2) / sigma;
    derivatives[2] = da * (-s1 * x + c1 * y) / sigma - dr * std::sin(a1 - a2);
    derivatives[3] = db * (-s2 * x + c2 * y) / sigma + dr * std::sin(a1 - a2);
    derivatives[4] = -da * a - db * b;
}
inline Objective modelObjective(const Patch &v)
{
    return [v](const cv::Mat &p, cv::Mat &r, cv::Mat &j)
    {
        r.create(int(v.size()), 1, CV_64F);
        j.create(int(v.size()), 7, CV_64F);
        for (int i = 0; i < int(v.size()); ++i)
        {
            double val, d[5];
            corner(v[i].x - p.at<double>(0), v[i].y - p.at<double>(1), p.at<double>(2),
                   p.at<double>(3), p.at<double>(4), val, d);
            double gain = p.at<double>(5);
            r.at<double>(i) = p.at<double>(6) + gain * val - v[i].v;
            for (int k = 0; k < 5; ++k)
            {
                j.at<double>(i, k) = gain * d[k];
            }
            j.at<double>(i, 5) = val;
            j.at<double>(i, 6) = 1;
        }
    };
}
inline Fit modelFit(const Patch &v, double a1, double a2, double bound)
{
    if (v.empty())
    {
        return {};
    }
    double lo = 1, hi = 0;
    for (auto z : v)
    {
        lo = std::min(lo, z.v);
        hi = std::max(hi, z.v);
    }
    cv::Mat p =
        (cv::Mat_<double>(7, 1) << 0, 0, a1, a2, std::log(.8), (hi - lo) / 2, (hi + lo) / 2);
    double corr = 0;
    for (auto z : v)
    {
        double m, d[5];
        corner(z.x, z.y, a1, a2, p.at<double>(4), m, d);
        corr += m * (z.v - p.at<double>(6));
    }
    p.at<double>(5) *= corr < 0 ? -1 : 1;
    cv::Mat lower =
        (cv::Mat_<double>(7, 1) << -bound, -bound, a1 - .25, a2 - .25, std::log(.18), -1., -.1);
    cv::Mat upper =
        (cv::Mat_<double>(7, 1) << bound, bound, a1 + .25, a2 + .25, std::log(2.5), 1., 1.1);
    return optimize(p, modelObjective(v), lower, upper);
}
// Spitschan & Ostermann Eq. (23), truncated to odd n=1,3,5.
// Fixed original Cartesian samples: no image interpolation.
inline Objective fourierObjective(Patch v, double alpha, bool demean)
{
    if (demean)
    {
        double mean = 0;
        for (auto z : v)
        {
            mean += z.v;
        }
        mean /= v.size();
        for (auto &z : v)
        {
            z.v -= mean;
        }
    }
    return [v, alpha](const cv::Mat &p, cv::Mat &r, cv::Mat &j)
    {
        r = cv::Mat::zeros(6, 1, CV_64F);
        j = cv::Mat::zeros(6, 2, CV_64F);
        double aa = alpha * alpha;
        for (auto z : v)
        {
            double x = z.x - p.at<double>(0), y = z.y - p.at<double>(1), r2 = x * x + y * y;
            double rad = std::sqrt(r2 + 1e-20), phi = std::atan2(y, x),
                   e = std::exp(-r2 / (2 * aa));
            double w = rad * e * z.v / v.size();
            double dwx = e * z.v / v.size() * x * (rad / aa - 1 / rad);
            double dwy = e * z.v / v.size() * y * (rad / aa - 1 / rad);
            double phix = y / (r2 + 1e-20), phiy = -x / (r2 + 1e-20);
            for (int k = 0; k < 3; ++k)
            {
                int n = 2 * k + 1;
                double c = std::cos(n * phi), s = std::sin(n * phi);
                r.at<double>(2 * k) += w * c;
                r.at<double>(2 * k + 1) += w * s;
                j.at<double>(2 * k, 0) += dwx * c - w * s * n * phix;
                j.at<double>(2 * k, 1) += dwy * c - w * s * n * phiy;
                j.at<double>(2 * k + 1, 0) += dwx * s + w * c * n * phix;
                j.at<double>(2 * k + 1, 1) += dwy * s + w * c * n * phiy;
            }
        }
    };
}
inline Fit fourierFit(const Patch &v, double alpha, bool demean, double bound)
{
    if (v.empty())
    {
        return {};
    }
    return optimize(cv::Mat::zeros(2, 1, CV_64F), fourierObjective(v, alpha, demean),
                    cv::Mat(2, 1, CV_64F, cv::Scalar(-bound)),
                    cv::Mat(2, 1, CV_64F, cv::Scalar(bound)));
}
} // namespace subpixel
