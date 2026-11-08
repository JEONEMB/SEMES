#include "calibration.hpp"

#include <exception>
#include <iostream>

int main(int argc, char** argv) {
    try {
        if (argc != 3) {
            std::cerr << "Usage: sparse_calibration A0_CACHE OUTPUT\n";
            return 2;
        }
        semes_study::run(std::filesystem::u8path(argv[1]), std::filesystem::u8path(argv[2]));
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
