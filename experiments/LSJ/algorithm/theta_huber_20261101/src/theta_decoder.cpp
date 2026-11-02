#include "theta_pipeline.hpp"
#include <cmath>
#include <stdexcept>

namespace theta {
DecodedPose decodeAngles(double upper, double lower, const Calibration& c) {
    const double values[] = {upper, lower, c.upper_intercept, c.upper_per_y_mm,
        c.upper_per_theta_deg, c.lower_intercept, c.lower_per_y_mm,
        c.lower_per_theta_deg, c.target_theta_deg};
    for (double x : values) if (!std::isfinite(x))
        throw std::runtime_error("non-finite angle/calibration");
    const double det = c.upper_per_y_mm * c.lower_per_theta_deg -
                       c.upper_per_theta_deg * c.lower_per_y_mm;
    if (det == 0 || !std::isfinite(det))
        throw std::runtime_error("Y/theta sensitivity matrix is singular");
    const double u = upper - c.upper_intercept;
    const double l = lower - c.lower_intercept;
    DecodedPose out;
    out.nuisance_y_mm = (u * c.lower_per_theta_deg - c.upper_per_theta_deg * l) / det;
    out.theta_deg = (c.upper_per_y_mm * l - c.lower_per_y_mm * u) / det;
    out.theta_correction_deg = c.target_theta_deg - out.theta_deg;
    out.determinant = det;
    if (!std::isfinite(out.nuisance_y_mm) || !std::isfinite(out.theta_deg) ||
        !std::isfinite(out.theta_correction_deg))
        throw std::runtime_error("non-finite decoded pose");
    return out;
}
}
