// Compile the already-ported affine implementation with an isolated OpenCV 4
// runtime, preserving the OpenCV 3 comparison and one shared algorithm source.
#include "io.hpp"
#include <opencv2/opencv.hpp>
namespace semes {
    cv::Mat loadimage(const fs::path& path){
        auto image=cv::imdecode(bytes(path),cv::IMREAD_GRAYSCALE);
        if(image.empty())throw std::runtime_error("Image decode failed");
        return image;
    }
}
#include "../../analysis_v1_20261012_3QR측정유지_회귀C++이식/src/measure_qr.cpp"
