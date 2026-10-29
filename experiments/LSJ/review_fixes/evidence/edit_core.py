from pathlib import Path

payload = Path(__file__).resolve().parent.parent/'payload'
p = payload/'src/theta_alignment.cpp'
s = p.read_text(encoding='utf-8')
s = s.replace('#include "theta_alignment.hpp"', '#include "theta_alignment.hpp"\n#include "validation_utils.hpp"')
s = s.replace('double clamp(double x, double low, double high) { return std::max(low, std::min(x, high)); }',
              'double clamp(double x, double low, double high) { return detail::finiteClamp(x, low, high); }')
s = s.replace('            return delta;', '            for (int j = 0; j < 4; ++j) detail::finite(delta[j], "Non-finite edge solver update");\n            return delta;', 1)
s = s.replace('        p -= delta;\n', '        p -= delta;\n        for (int j = 0; j < 4; ++j) detail::finite(p[j], "Non-finite edge parameters before bounds");\n', 1)
start = s.index('ThetaConfig ThetaConfig::load(')
end = s.index('\nLineMeasurement ThetaEstimator::measureLine', start)
s = s[:start] + r'''ThetaConfig ThetaConfig::load(const std::string& path) {
    try {
        cv::FileStorage fs(readTextUtf8(path), cv::FileStorage::READ | cv::FileStorage::MEMORY | cv::FileStorage::FORMAT_JSON);
        detail::configCheck(fs.isOpened(), "Cannot parse theta config");
        const auto root = fs.root();
        detail::configCheck(detail::integer(root, "schema_version") == 1, "Unsupported theta schema");
        ThetaConfig c;
        c.width = detail::integer(root, "width"); c.height = detail::integer(root, "height");
        const auto roi = detail::integers(root, "locator_roi_xywh", 4);
        c.locatorRoi = cv::Rect(roi[0], roi[1], roi[2], roi[3]);
        c.searchMargin = detail::integer(root, "search_margin_px");
        c.tickDeg = detail::number(root, "tick_deg"); c.toleranceDeg = detail::number(root, "tolerance_deg");
        c.maxStepDeg = detail::number(root, "max_step_deg");
        detail::configCheck(detail::string(root, "target_source") == "first_center_image_measured_by_runtime", "Unsupported theta target source");
        const auto entries = detail::sequence(root, "lines", 2);
        int i = 0;
        for (auto it = entries.begin(); it != entries.end(); ++it, ++i) {
            const auto node = *it; auto& line = c.lines[i];
            line.name = detail::string(node, "name");
            line.x0 = detail::integer(node, "x0"); line.x1 = detail::integer(node, "x1");
            line.expectedY = detail::integer(node, "expected_y"); line.halfHeight = detail::integer(node, "half_height");
            line.polarity = detail::integer(node, "polarity");
            line.positiveSlope = detail::number(node, "group4_slope"); line.negativeSlope = detail::number(node, "group5_slope");
        }
        c.validate(); return c;
    } catch (const cv::Exception& e) { throw AlignmentError(ErrorCode::InvalidConfiguration, e.what()); }
}

void ThetaConfig::validate() const {
    detail::configCheck(tickDeg > 0 && std::isfinite(tickDeg) && toleranceDeg >= 0 && std::isfinite(toleranceDeg) &&
                        maxStepDeg >= tickDeg && std::isfinite(maxStepDeg), "Invalid theta motion settings");
    detail::roiInside(locatorRoi, searchMargin, width, height);
    const std::array<std::string, 2> names{{"rail_UL_edge", "rail_UR_edge"}};
    for (int i = 0; i < 2; ++i) {
        const auto& line = lines[i];
        detail::configCheck(line.name == names[i] && line.x0 >= 0 && line.x1 > line.x0 && line.x1 < width && line.halfHeight >= 2 &&
                            int64_t(line.expectedY)-line.halfHeight-8 >= 0 && int64_t(line.expectedY)+line.halfHeight+9 <= height &&
                            (line.polarity == 1 || line.polarity == -1) && std::isfinite(line.positiveSlope) &&
                            std::isfinite(line.negativeSlope) && line.positiveSlope < 0 && line.negativeSlope < 0,
                            "Invalid upper-line ROI/name/slope");
    }
}
''' + s[end:]
start = s.index('ThetaResult thetaFromAngles(')
end = s.index('\nThetaEstimator::ThetaEstimator', start)
s = s[:start] + r'''namespace {
// Existing branch selection and BOTH-line acceptance predicate, without issuing a command.
ThetaResult thetaState(const std::array<double, 2>& angles, const std::array<double, 2>& targets, const ThetaConfig& config) {
    struct Candidate { double theta, cost; int group; } candidates[2];
    for (int k = 0; k < 2; ++k) {
        double numerator = 0, denominator = 0;
        for (int i = 0; i < 2; ++i) {
            detail::finite(angles[i], "Non-finite angle"); detail::finite(targets[i], "Non-finite target");
            const double a = k == 0 ? config.lines[i].positiveSlope : config.lines[i].negativeSlope;
            numerator += a * (angles[i] - targets[i]); denominator += a * a;
        }
        detail::finite(numerator, "Theta numerator overflow"); detail::finite(denominator, "Theta denominator overflow");
        if (!(denominator > 0)) throw AlignmentError(ErrorCode::NumericalFailure, "Theta slope underflow");
        double theta = numerator / denominator;
        detail::finite(theta, "Theta estimate overflow");
        theta = k == 0 ? std::max(0.0, theta) : std::min(0.0, theta);
        double cost = 0;
        for (int i = 0; i < 2; ++i) {
            const double a = k == 0 ? config.lines[i].positiveSlope : config.lines[i].negativeSlope;
            const double residual = angles[i] - targets[i] - a * theta;
            cost += residual * residual;
        }
        detail::finite(cost, "Theta cost overflow");
        candidates[k] = {theta, cost, k == 0 ? 4 : 5};
    }
    const auto best = candidates[candidates[1].cost < candidates[0].cost ? 1 : 0];
    ThetaResult r; r.angles = angles; r.targets = targets; r.estimatedThetaDeg = best.theta;
    r.group = best.theta == 0 ? 0 : best.group; r.fitResidualImageDeg = std::sqrt(best.cost / 2);
    r.continuousCorrectionDeg = -best.theta; r.outsideCalibration = std::abs(best.theta) > 1.0;
    r.thetaWithinTolerance = true;
    for (int i = 0; i < 2; ++i) {
        const double a = best.group == 4 ? config.lines[i].positiveSlope : config.lines[i].negativeSlope;
        r.perLineTheta[i] = (angles[i] - targets[i]) / a;
        detail::finite(r.perLineTheta[i], "Per-line theta overflow");
        r.thetaWithinTolerance = r.thetaWithinTolerance && std::abs(r.perLineTheta[i]) <= config.toleranceDeg + 1e-12;
    }
    return r;
}
double thetaResponse(const LineConfig& line, double q) {
    return (q >= 0 ? line.positiveSlope : line.negativeSlope) * q;
}
} // namespace

ThetaResult thetaFromAngles(const std::array<double, 2>& angles,
                           const std::array<double, 2>& targets, const ThetaConfig& config) {
    config.validate();
    auto r = thetaState(angles, targets, config);
    if (r.thetaWithinTolerance) { r.status = "theta_within_tolerance"; r.predictedWithinTolerance = true; return r; }
    const double q = r.estimatedThetaDeg;
    std::array<double, 2> residual;
    for (int i = 0; i < 2; ++i) residual[i] = angles[i] - targets[i] - thetaResponse(config.lines[i], q);
    // Carry CURRENT image residuals. This is a prediction for command selection, not a new calibration.
    const auto predict = [&](double command) {
        std::array<double, 2> predicted;
        for (int i = 0; i < 2; ++i)
            predicted[i] = targets[i] + residual[i] + thetaResponse(config.lines[i], q + command);
        return thetaState(predicted, targets, config);
    };

    bool continuousEnvelopeExists = false, found = false;
    double chosen = 0;
    // The envelope covers either acceptance branch. Re-run the EXACT unchanged acceptance
    // predicate on each tick, so a wider envelope never widens the actual tolerance.
    for (int branch = 0; branch < 2; ++branch) {
        double low = branch == 0 ? 0 : -std::numeric_limits<double>::infinity();
        double high = branch == 0 ? std::numeric_limits<double>::infinity() : 0;
        for (int i = 0; i < 2; ++i) {
            const auto& line = config.lines[i];
            const double a = branch == 0 ? line.positiveSlope : line.negativeSlope;
            const double band = std::max(std::abs(line.positiveSlope), std::abs(line.negativeSlope)) * (config.toleranceDeg + 1e-12);
            low = std::max(low, (band - residual[i]) / a);
            high = std::min(high, (-band - residual[i]) / a);
        }
        if (low > high) continue;
        continuousEnvelopeExists = true;
        const double first = std::ceil((low - q) / config.tickDeg - 1e-10);
        const double last = std::floor((high - q) / config.tickDeg + 1e-10);
        detail::finite(first, "Theta tick overflow"); detail::finite(last, "Theta tick overflow");
        // Resource/representation guard for corrupted or extreme configurations, not an acceptance band.
        detail::configCheck(std::abs(first) < 9007199254740992.0 && std::abs(last) < 9007199254740992.0 && last-first < 4096,
                            "Theta candidate search exceeds numeric/resource limits");
        for (double tick = first; tick <= last; ++tick) {
            if (tick == 0) continue;
            const double command = tick * config.tickDeg;
            if (predict(command).thetaWithinTolerance && (!found || std::abs(command) < std::abs(chosen))) {
                found = true; chosen = command;
            }
        }
    }
    if (!found) {
        r.commandAvailable = false;
        r.errorCode = continuousEnvelopeExists ? ErrorCode::ThetaNoReachableTick : ErrorCode::ThetaTargetConflict;
        r.status = errorCodeName(r.errorCode); return r;
    }
    r.plannedTotalCorrectionDeg = chosen;
    const double maxTicks = std::floor(config.maxStepDeg / config.tickDeg + 1e-12);
    detail::finite(maxTicks, "Theta command cap overflow");
    r.commandDeg = clamp(std::round(chosen/config.tickDeg), -maxTicks, maxTicks) * config.tickDeg;
    r.predictedWithinTolerance = predict(r.commandDeg).thetaWithinTolerance;
    // An endpoint beyond the configured step cap is approached in capped, tick-aligned moves.
    r.status = r.predictedWithinTolerance ? (r.outsideCalibration ? "move_extrapolated" : "move") : "move_partial";
    return r;
}
''' + s[end:]
s = s.replace('fs << "status" << r.status << "estimated_theta_deg"', 'fs << "status" << r.status << "error_code" << errorCodeName(r.errorCode)\n       << "planned_total_correction_deg" << r.plannedTotalCorrectionDeg\n       << "predicted_within_tolerance" << static_cast<int>(r.predictedWithinTolerance) << "estimated_theta_deg"')
p.write_text(s, encoding='utf-8')

p = payload/'src/y_alignment.cpp'; s = p.read_text(encoding='utf-8')
s = s.replace('#include "y_alignment.hpp"', '#include "y_alignment.hpp"\n#include "validation_utils.hpp"')
start = s.index('YConfig YConfig::load('); end = s.index('\nYResult yFromPositions', start)
s = s[:start] + r'''YConfig YConfig::load(const std::string& path) {
    try {
        cv::FileStorage fs(readTextUtf8(path), cv::FileStorage::READ | cv::FileStorage::MEMORY | cv::FileStorage::FORMAT_JSON);
        detail::configCheck(fs.isOpened(), "Cannot parse Y configuration");
        const auto root = fs.root();
        detail::configCheck(detail::integer(root, "schema_version") == 1, "Invalid Y schema");
        YConfig c;
        c.width = detail::integer(root, "width"); c.height = detail::integer(root, "height");
        c.searchMargin = detail::integer(root, "search_margin_px"); c.tickMm = detail::number(root, "tick_mm");
        c.targetProfile = detail::string(root, "target_profile");
        detail::configCheck(detail::number(root, "additional_tolerance_px") == 0, "Additional Y tolerance is not supported");
        detail::configCheck(detail::integers(root, "slope_groups", 4) == std::vector<int>{2,3,6,7}, "Expected Y slope groups 2,3,6,7");
        const auto entries = detail::sequence(root, "markers", 3);
        int i = 0;
        for (auto it = entries.begin(); it != entries.end(); ++it, ++i) {
            const auto n = *it; auto& m = c.markers[i];
            m.name = detail::string(n, "name"); const auto roi = detail::integers(n, "roi_xywh", 4);
            m.roi = cv::Rect(roi[0], roi[1], roi[2], roi[3]);
            m.lowPx = detail::number(n, "low_px"); m.highPx = detail::number(n, "high_px");
            const auto slopes = detail::sequence(n, "slopes_px_per_mm", 4);
            int j = 0; for (auto a = slopes.begin(); a != slopes.end(); ++a, ++j)
                m.slopes[j] = detail::number(*a, std::string("slopes_px_per_mm"));
        }
        c.validate(); return c;
    } catch (const cv::Exception& e) { throw AlignmentError(ErrorCode::InvalidConfiguration, e.what()); }
}

void YConfig::validate() const {
    detail::configCheck(tickMm > 0 && std::isfinite(tickMm) && !targetProfile.empty(), "Invalid Y settings");
    const std::array<std::string, 3> names{{"L", "C", "R"}};
    for (int i = 0; i < 3; ++i) {
        const auto& m = markers[i]; detail::roiInside(m.roi, searchMargin, width, height);
        detail::configCheck(m.name == names[i] && std::isfinite(m.lowPx) && std::isfinite(m.highPx) &&
                            m.lowPx >= 0 && m.highPx < width && m.lowPx <= m.highPx, "Invalid Y target range/name");
        for (double a : m.slopes) detail::configCheck(a > 0 && std::isfinite(a), "Expected finite positive Y slope");
    }
}
''' + s[end:]
s = s.replace('r.status = "no_common_y_interval"; return r;', 'r.errorCode = ErrorCode::YTargetConflict; r.status = errorCodeName(r.errorCode); return r;')
s = s.replace('    if (firstTick > lastTick)', '''    r.nearestTickBelowMm = std::floor(r.commonLowMm / config.tickMm) * config.tickMm;
    r.nearestTickAboveMm = std::ceil(r.commonHighMm / config.tickMm) * config.tickMm;
    r.tickDiagnosticsAvailable = true;
    if (firstTick > lastTick)''')
s = s.replace('r.status = "no_reachable_y_tick"; return r;', 'r.errorCode = ErrorCode::YNoReachableTick; r.status = errorCodeName(r.errorCode); return r;')
s = s.replace('fs << "status" << r.status << "command_mm"', '''fs << "status" << r.status << "error_code" << errorCodeName(r.errorCode)
       << "tick_diagnostics_available" << static_cast<int>(r.tickDiagnosticsAvailable)
       << "nearest_tick_below_mm" << r.nearestTickBelowMm << "nearest_tick_above_mm" << r.nearestTickAboveMm << "command_mm"''')
p.write_text(s, encoding='utf-8')
print('Updated strict loading, finite checks, theta prediction/tick feasibility and Y diagnostics.')
