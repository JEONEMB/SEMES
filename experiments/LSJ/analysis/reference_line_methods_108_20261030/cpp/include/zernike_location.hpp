#pragma once
#include <limits>
#include <string>

namespace lm {
// Values not reached by the computation remain NaN and are serialized as null.
struct ZernikeLocation {
    bool ok = false;
    std::string reason;
    double psi = std::numeric_limits<double>::quiet_NaN();
    double denominator = std::numeric_limits<double>::quiet_NaN();
    double ratio = std::numeric_limits<double>::quiet_NaN();
    double discriminant = std::numeric_limits<double>::quiet_NaN();
    double distance = std::numeric_limits<double>::quiet_NaN();
    double shift = std::numeric_limits<double>::quiet_NaN();
    double u = std::numeric_limits<double>::quiet_NaN(), v = std::numeric_limits<double>::quiet_NaN();
};
// Same existing ramp/step-limit formulas and configured orientation/shift gates.
// Adds arithmetic-domain checks, not an empirical near-zero denominator threshold.
ZernikeLocation zernike_location(double are, double aim, double a20, double wbar,
                                 int np, double x, double y, int polarity,
                                 double max_deviation_deg, double max_shift_px);
}
