#include "theta_pipeline.hpp"
#include <algorithm>
#include <climits>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace theta {
namespace {
constexpr double normalMedian = 0.6744897501960817;
constexpr double radToDeg = 180.0 / CV_PI;
struct Wls {
    double b, m, variance;
    std::vector<double> residuals;
};

Wls leastSquares(const std::vector<cv::Point2d>& p, const std::vector<double>& weights) {
    const int n = static_cast<int>(p.size());
    cv::Mat a(n, 2, CV_64F), y(n, 1, CV_64F), beta;
    for (int i = 0; i < n; ++i) {
        const double w = std::sqrt(weights[i]);
        a.at<double>(i, 0) = w;
        a.at<double>(i, 1) = w * p[i].x;
        y.at<double>(i, 0) = w * p[i].y;
    }
    if (!cv::solve(a, y, beta, cv::DECOMP_SVD))
        throw std::runtime_error("weighted least squares failed");
    Wls out{beta.at<double>(0), beta.at<double>(1), 0, {}};
    if (!std::isfinite(out.b) || !std::isfinite(out.m))
        throw std::runtime_error("non-finite line parameters");
    out.residuals.reserve(n);
    for (int i = 0; i < n; ++i) {
        const double e = p[i].y - (out.b + out.m * p[i].x);
        const double wr = y.at<double>(i) -
            (a.at<double>(i, 0) * out.b + a.at<double>(i, 1) * out.m);
        out.residuals.push_back(e);
        out.variance += wr * wr;
    }
    out.variance /= n - 2;
    return out;
}

double madZero(const std::vector<double>& r) {
    std::vector<double> v;
    v.reserve(r.size());
    for (double e : r) v.push_back(std::abs(e));
    std::sort(v.begin(), v.end());
    const size_t k = v.size() / 2;
    const double median = v.size() % 2 ? v[k] : (v[k - 1] + v[k]) / 2;
    return median / normalMedian;
}

double deviance(const Wls& w, double t) {
    // statsmodels RLM.deviance divides by WLS variance, NOT the MAD scale.
    // Kept explicitly so the previously verified stopping rule is preserved.
    if (!(w.variance > 0) || !std::isfinite(w.variance))
        throw std::runtime_error("undefined deviance (zero/non-finite WLS variance)");
    double sum = 0;
    for (double e : w.residuals) {
        const double u = std::abs(e / w.variance);
        sum += u <= t ? 0.5 * u * u : t * u - 0.5 * t * t;
    }
    if (!std::isfinite(sum)) throw std::runtime_error("non-finite deviance");
    return sum;
}
}

LineResult fitHuberVertical(const std::vector<cv::Point2d>& points, const HuberSettings& s) {
    if (!(s.t > 0) || !std::isfinite(s.t) || s.max_iterations < 2 ||
        !(s.tolerance > 0) || !std::isfinite(s.tolerance))
        throw std::runtime_error("invalid Huber settings");
    // Three is a mathematical requirement for this residual-scale estimator:
    // n - 2 must be positive. This is NOT a selected minimum-quality filter.
    if (points.size() <= 2 || points.size() > static_cast<size_t>(INT_MAX))
        throw std::runtime_error("positive residual degrees of freedom required (n > 2)");
    double xmin = points[0].x, xmax = xmin;
    for (const auto& p : points) {
        if (!std::isfinite(p.x) || !std::isfinite(p.y))
            throw std::runtime_error("non-finite point");
        xmin = std::min(xmin, p.x); xmax = std::max(xmax, p.x);
    }
    if (xmin == xmax) throw std::runtime_error("line design has no x variation");
    std::vector<double> weights(points.size(), 1.0);
    Wls w = leastSquares(points, weights);
    double scale = madZero(w.residuals);
    if (!(scale > 0) || !std::isfinite(scale))
        throw std::runtime_error("undefined Huber scale; no epsilon fallback used");
    double previous = deviance(w, s.t);
    LineResult result;
    result.iterations = 1;
    result.deviance_delta = std::numeric_limits<double>::infinity();
    while (result.iterations < s.max_iterations) {
        for (size_t i = 0; i < points.size(); ++i) {
            const double u = std::abs(w.residuals[i] / scale);
            weights[i] = u <= s.t ? 1.0 : s.t / u;
        }
        w = leastSquares(points, weights);
        scale = madZero(w.residuals);
        if (!(scale > 0) || !std::isfinite(scale))
            throw std::runtime_error("undefined Huber scale; no epsilon fallback used");
        const double current = deviance(w, s.t);
        result.deviance_delta = std::abs(current - previous);
        ++result.iterations;
        previous = current;
        if (result.deviance_delta <= s.tolerance) {
            result.converged = true;
            break;
        }
    }
    result.stop_reason = result.converged ? "tolerance" : "iteration_limit";
    result.intercept_px = w.b;
    result.slope = w.m;
    result.angle_deg = std::atan(w.m) * radToDeg;
    result.mad_scale_px = scale;
    result.tau_px = s.t * scale;
    result.point_count = static_cast<int>(points.size());
    // Diagnostic weights recomputed at the FINAL parameters (not used for filtering).
    for (double e : w.residuals)
        if (std::abs(e / scale) > s.t) ++result.downweighted_points;
    return result;
}
}
