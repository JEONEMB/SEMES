#include "qr_pose.hpp"
#include <opencv2/imgcodecs.hpp>

#include <exception>
#include <iomanip>
#include <iostream>

int main(int argc, char **argv)
{
    if (argc != 2) {
        std::cerr << "Usage: main <rectified-bmp>\n";
        return 2;
    }
    try {
        cv::Mat image = cv::imread(argv[1], cv::IMREAD_UNCHANGED);
        const auto pose = qr_pose::estimate(image);
        std::cout << std::setprecision(10) << "y_mm=" << pose.y_mm
                  << " theta_deg=" << pose.theta_deg << '\n';
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
