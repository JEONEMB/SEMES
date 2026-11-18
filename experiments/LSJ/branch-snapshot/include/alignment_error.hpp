#pragma once
#include <stdexcept>
#include <string>

namespace semes {
enum class ErrorCode {
    None, InvalidConfiguration, InvalidInput, MeasurementFailed, NumericalFailure,
    IoFailure, OutputAliasesInput, NotConfigured, PendingMove, InvalidTransition,
    MovementUncertain, MoveLimitReached, ThetaOscillation,
    ThetaTargetConflict, ThetaNoReachableTick, YTargetConflict, YNoReachableTick
};
inline const char* errorCodeName(ErrorCode c) {
    switch (c) {
    case ErrorCode::None: return "none";
    case ErrorCode::InvalidConfiguration: return "invalid_configuration";
    case ErrorCode::InvalidInput: return "invalid_input";
    case ErrorCode::MeasurementFailed: return "measurement_failed";
    case ErrorCode::NumericalFailure: return "numerical_failure";
    case ErrorCode::IoFailure: return "io_failure";
    case ErrorCode::OutputAliasesInput: return "output_aliases_input";
    case ErrorCode::NotConfigured: return "not_configured";
    case ErrorCode::PendingMove: return "pending_move";
    case ErrorCode::InvalidTransition: return "invalid_transition";
    case ErrorCode::MovementUncertain: return "movement_uncertain";
    case ErrorCode::MoveLimitReached: return "move_limit_reached";
    case ErrorCode::ThetaOscillation: return "theta_oscillation_detected";
    case ErrorCode::ThetaTargetConflict: return "theta_target_conflict";
    case ErrorCode::ThetaNoReachableTick: return "theta_no_reachable_tick";
    case ErrorCode::YTargetConflict: return "no_common_y_interval";
    case ErrorCode::YNoReachableTick: return "no_reachable_y_tick";
    }
    return "unknown_error";
}
class AlignmentError : public std::runtime_error {
public:
    AlignmentError(ErrorCode code, const std::string& message) : std::runtime_error(message), code_(code) {}
    ErrorCode code() const noexcept { return code_; }
private:
    ErrorCode code_;
};
enum class MoveFailure { NoMotion, PartialOrUnknown };
} // namespace semes
