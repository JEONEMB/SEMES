#include "alignment.hpp"
#include "joint_reference.hpp"
#include <opencv2/imgcodecs.hpp>
#include <iostream>
#include <iomanip>
#include <stdexcept>
int main(int argc, char** argv) {
    try {
        if (argc != 3) { std::cerr << "usage: measure_one MODEL_DIRECTORY IMAGE\n"; return 2; }
        initialize_joint(argv[1]);
        cv::Mat image = cv::imread(argv[2], cv::IMREAD_GRAYSCALE);
        if (image.empty()) throw std::runtime_error("Cannot read image");
        const auto result = joint(image);
        const auto diagnostic = alignment_diagnostics();
        std::cout << std::setprecision(17) << "y=" << result.y << " t=" << result.t
                  << " valid=" << diagnostic.measurement_valid
                  << " method=" << joint_qr_method() << " model=" << joint_model_id()
                  << "\n" << diagnostic.message << "\n";
        return diagnostic.measurement_valid ? 0 : 1;
    } catch (const std::exception& e) { std::cerr << e.what() << "\n"; return 1; }
}
