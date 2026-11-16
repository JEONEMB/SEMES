#include "detail/alignment_diagnostics.hpp"
namespace alignment_detail {
static thread_local AlignmentDiagnostics last;
AlignmentDiagnostics& diagnostics_state(){return last;}
void reset_diagnostics(const std::string& stage){
    last={};last.stage=stage;
    last.minimum_qr_correlation=last.minimum_line_correlation=std::numeric_limits<double>::quiet_NaN();
    last.message="call has not completed";
}
void set_diagnostics(bool outside,bool precondition,const std::string& message){
    last.outside_training_range=outside;last.control_precondition_met=precondition;last.message=message;
}
}
AlignmentDiagnostics alignment_diagnostics(){return alignment_detail::diagnostics_state();}
