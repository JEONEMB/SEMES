#pragma once

#include <filesystem>

namespace semes_study {
void runResidualT(const std::filesystem::path& cache, const std::filesystem::path& output);
void run(const std::filesystem::path& cache, const std::filesystem::path& output);
}
