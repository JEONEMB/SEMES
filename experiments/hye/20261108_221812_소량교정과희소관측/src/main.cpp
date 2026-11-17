#include "calibration.hpp"

#include <exception>
#include <iostream>
#include <string_view>

int main(int argc, char** argv) {
    try {
        if (argc != 3 && !(argc == 4 && std::string_view(argv[3]) == "--residual-t")) {
            std::cerr << "Usage: sparse_calibration A0_CACHE OUTPUT [--residual-t]\n";
            return 2;
        }
        const auto cache = std::filesystem::u8path(argv[1]);
        const auto output = std::filesystem::u8path(argv[2]);
        if (argc == 4) semes_study::runResidualT(cache, output);
        else semes_study::run(cache, output);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
