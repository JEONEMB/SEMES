#pragma once
#include "alignment.hpp"
namespace alignment_detail {
AlignmentDiagnostics& diagnostics_state();
void reset_diagnostics(const std::string& stage);
void set_diagnostics(bool outside,bool precondition,const std::string& message);
}
