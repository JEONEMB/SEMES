#include "zernike_location.hpp"
#include <algorithm>
#include <cmath>

namespace lm {
ZernikeLocation zernike_location(double are, double aim, double a20, double wbar,
                                 int np, double x, double y, int polarity,
                                 double max_dev, double max_shift) {
    ZernikeLocation r;
    auto fail = [&](const char* why) { r.reason=why; return r; };
    if (!std::isfinite(wbar) || wbar<=0 || np<3 || np%2==0 ||
        (polarity!=1 && polarity!=-1) || !std::isfinite(x) || !std::isfinite(y) ||
        !std::isfinite(max_dev) || max_dev<0 || !std::isfinite(max_shift) || max_shift<0)
        return fail("invalid_location_parameters");
    if (!std::isfinite(are) || !std::isfinite(aim) || !std::isfinite(a20))
        return fail("nonfinite_moments");
    if (are==0 && aim==0) return fail("zero_first_moment");
    r.psi=std::atan2(aim,are);
    r.denominator=are*std::cos(r.psi)+aim*std::sin(r.psi);
    if (!std::isfinite(r.psi) || !std::isfinite(r.denominator))
        return fail("nonfinite_moment_denominator");
    if (r.denominator==0) return fail("zero_moment_denominator");
    r.ratio=a20/r.denominator;
    if (!std::isfinite(r.ratio)) return fail("nonfinite_moment_ratio");
    double dev=std::acos(std::max(-1.0,std::min(1.0,polarity*std::sin(r.psi))))*180.0/3.14159265358979323846;
    if (!std::isfinite(dev)) return fail("nonfinite_orientation");
    if (dev>max_dev) return fail("orientation_not_matching_target");
    double w2=wbar*wbar;
    if (!std::isfinite(w2)) return fail("nonfinite_ramp_width_squared");
    r.discriminant=(w2-1)*(w2-1)-2*w2*r.ratio;
    if (!std::isfinite(r.discriminant)) return fail("nonfinite_ramp_discriminant");
    if (r.discriminant<0) return fail("ramp_model_no_real_solution");
    // Preserve the existing small-width branch; its model validity is a separate audit item.
    r.distance=wbar>1e-9 ? (1-w2-std::sqrt(r.discriminant))/w2 : r.ratio;
    if (!std::isfinite(r.distance)) return fail("nonfinite_ramp_distance");
    r.shift=np*r.distance/2.0;
    if (!std::isfinite(r.shift)) return fail("nonfinite_location_shift");
    r.u=x+r.shift*std::cos(r.psi);r.v=y+r.shift*std::sin(r.psi);
    if (!std::isfinite(r.u) || !std::isfinite(r.v)) return fail("nonfinite_location_coordinates");
    if (std::fabs(r.shift)>max_shift) return fail("shift_exceeds_limit");
    r.ok=true;
    return r;
}
}
