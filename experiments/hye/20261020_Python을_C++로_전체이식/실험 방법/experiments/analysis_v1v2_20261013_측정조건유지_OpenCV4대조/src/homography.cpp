#include "io.hpp"
#include <opencv2/opencv.hpp>
namespace semes {
    cv::Mat loadimage(const fs::path& path){
        auto image=cv::imdecode(bytes(path),cv::IMREAD_GRAYSCALE);
        if(image.empty())throw std::runtime_error("Image decode failed");
        return image;
    }
}
#include "../../analysis_v2_20261013_특징분할유지_통계C++이식/src/extract_homography.cpp"
