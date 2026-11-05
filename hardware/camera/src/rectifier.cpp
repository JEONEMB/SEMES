#include "stage_camera/rectifier.hpp"

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

#include <stdexcept>

namespace stage_camera {

Rectifier::Rectifier(const std::string &calib_path, double zoom) : zoom_(zoom)
{
    if (!(zoom > 0.0))
        throw std::runtime_error("Rectifier: zoom 은 0 보다 커야 함");

    cv::FileStorage fs(calib_path, cv::FileStorage::READ);
    if (!fs.isOpened())
        throw std::runtime_error("Rectifier: 캘리브레이션 파일을 열 수 없음: " + calib_path);

    int w = 0, h = 0;
    fs["image_width"] >> w;
    fs["image_height"] >> h;
    fs["camera_matrix"] >> K_;
    fs["dist_coeffs"] >> D_;
    if (w <= 0 || h <= 0 || K_.size() != cv::Size(3, 3) || D_.empty())
        throw std::runtime_error("Rectifier: image_width/height, camera_matrix, dist_coeffs 가 필요함: " +
                                 calib_path);
    input_size_ = cv::Size(w, h);
    K_.convertTo(K_, CV_64F);
    D_.convertTo(D_, CV_64F);

    // 결과 사진의 카메라 행렬 = SEMES 규격, 렌즈 중심이 사진 가운데로 온다
    const cv::Size out = outputSize();
    K_out_ = (cv::Mat_<double>(3, 3) << SemesSpec::kFx * zoom, 0, (out.width - 1) * 0.5,
              0, SemesSpec::kFy * zoom, (out.height - 1) * 0.5,
              0, 0, 1);

    // 먼저 실수형 변환표로 원본 바깥을 가리키는 픽셀이 있는지 센 뒤, 빠른 고정소수점 형식으로 바꾼다
    cv::Mat mx, my;
    cv::initUndistortRectifyMap(K_, D_, cv::Mat(), K_out_, out, CV_32FC1, mx, my);
    cv::Mat inside = (mx >= 0) & (mx <= w - 1) & (my >= 0) & (my <= h - 1);
    outside_fraction_ = 1.0 - static_cast<double>(cv::countNonZero(inside)) / inside.total();
    cv::convertMaps(mx, my, map1_, map2_, CV_16SC2);
}

cv::Mat Rectifier::apply(const cv::Mat &gray) const
{
    if (gray.type() != CV_8UC1)
        throw std::runtime_error("Rectifier::apply: 8bit 흑백(CV_8UC1)만 받음");
    if (gray.size() != input_size_)
        throw std::runtime_error("Rectifier::apply: 입력 크기가 캘리브레이션 크기와 다름 (" +
                                 std::to_string(gray.cols) + "x" + std::to_string(gray.rows) + " vs " +
                                 std::to_string(input_size_.width) + "x" +
                                 std::to_string(input_size_.height) + ")");
    cv::Mat out;
    cv::remap(gray, out, map1_, map2_, cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0));
    return out;
}

}  // namespace stage_camera
