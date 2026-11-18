#include "qr_pose.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace qr {
ControlResult run_correction(Stage& stage,const Settings& s,const std::function<double()>& horizontal,
                             const std::function<Pose()>& estimate,const Event& event) {
    if(s.max_moves<1 || s.max_moves>20 || s.cancellation_limit<0 || s.cancellation_limit>3 || !horizontal || !estimate)
        throw std::invalid_argument("Invalid correction settings or observation callbacks");
    auto p=stage.position();const auto limits=stage.limits();
    int moves=0,corrections=0,commanded_y=s.center[0];
    auto done=[&](const char* why){return ControlResult{why,p,moves,corrections};};
    auto within=[&](Position target){return target[0]>=limits[0] && target[0]<=limits[1] &&
                                           target[1]>=limits[2] && target[1]<=limits[3];};
    auto move=[&](Position target){
        if (!within(target)) throw std::runtime_error("Target outside stage limits");
        p=stage.move_absolute(target);++moves;
        if (event) event("move",p,{target[0]/1000.,target[1]/1000.});
    };
    auto bounded=[](double v,int bound){return static_cast<int>(std::nearbyint(std::max(-double(bound),std::min(double(bound),v))));};
    auto observe=[&](){const double v=horizontal();if(!std::isfinite(v)) throw std::runtime_error("Invalid QR observation");return v;};
    const int start_theta=p[1];
    move({commanded_y,start_theta});
    if (std::abs(p[1]-start_theta)>1) return done("theta_changed_during_y_anchor");
    for(;;) {
        if (std::abs(p[0]-commanded_y)>5) return done("y_anchor_lost");
        double dx=observe();
        if (std::abs(dx)<=.30) dx=observe();
        if (std::abs(dx)<=.30) {
            const Pose error=estimate();
            if (event) event("qr_cancellation_check",p,error);
            if (!std::isfinite(error[0]) || !std::isfinite(error[1])) return done("qr_geometry_unavailable");
            if (s.cancellation_limit==0)
                return done(std::abs(error[0])<=.045 && std::abs(error[1])<=.01 ? "qr_center_candidate" : "qr_cancellation_suspected");
            if (std::abs(error[0])<=.020 && std::abs(error[1])<=.005)
                return done("qr_center_candidate");
            if (corrections>=s.cancellation_limit || moves>=s.max_moves) return done("qr_correction_budget");
            const Position delta(bounded(-650.*error[0],150),bounded(-650.*error[1],50));
            const Position target=p+delta;
            if (!within(target) || std::abs(target[0]-s.center[0])>450 || std::abs(target[1]-s.center[1])>1050)
                return done("outside_calibrated_range");
            if (!delta[0] && !delta[1]) return done("move_budget_or_resolution");
            if (event) event("qr_cancellation_command",p,{delta[0]/1000.,delta[1]/1000.});
            commanded_y=target[0];++corrections;move(target);
            continue;
        }
        const int dt=bounded(-1000.*dx/rig_horizontal_slope_px_per_degree,300);
        const Position target(commanded_y,p[1]+dt);
        if (!within(target) || std::abs(target[1]-s.center[1])>1050) return done("outside_calibrated_range");
        if (!dt || moves>=s.max_moves) return done("move_budget_or_resolution");
        move(target);
    }
}
}
