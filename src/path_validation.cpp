#include "path_validation.hpp"
#include "alignment_error.hpp"
#include <filesystem>
#include <system_error>

namespace semes {
void validateOutputPath(const std::string& output, const std::vector<std::string>& inputs) {
    if (output.empty()) return;
    namespace fs = std::filesystem;
    const auto out = fs::u8path(output);
    std::error_code ec;
    const bool exists = fs::exists(out, ec);
    if (ec) throw AlignmentError(ErrorCode::IoFailure, "Cannot inspect output path: " + ec.message());
    if (!exists) return; // Any alias of an existing input also exists.
    for (const auto& input : inputs) {
        if (input.empty()) continue;
        const auto in = fs::u8path(input);
        const bool inputExists = fs::exists(in, ec);
        if (ec) throw AlignmentError(ErrorCode::IoFailure, "Cannot inspect input path: " + ec.message());
        if (!inputExists) continue; // Missing images are reported individually by CLI.
        const bool same = fs::equivalent(out, in, ec);
        if (ec) throw AlignmentError(ErrorCode::IoFailure, "Cannot compare file identity: " + ec.message());
        if (same) throw AlignmentError(ErrorCode::OutputAliasesInput, "Output aliases an input file: " + input);
    }
}
} // namespace semes
