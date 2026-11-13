// rectifier.hpp — HQ 카메라 원본 사진을 SEMES 규격 사진(3840x2160)으로 바꾼다
//
//  왜곡 펴기 → 렌즈 중심(cx, cy) 기준 크롭 → 3840x2160 확대를
//  변환표(initUndistortRectifyMap) 하나로 한 번에 처리한다. 변환표는 생성자에서 한 번만 만든다.
//
//  결과 사진의 카메라 행렬 = SEMES 3840 사진과 같음
//    fx = 12mm / 2.8um x 3840/4096 ≈ 4017.9,  fy = 12mm / 2.8um ≈ 4285.7,  중심 (1919.5, 1079.5)
//
//  카메라 없이도 쓸 수 있다 (저장된 원본 사진을 변환할 때).
#ifndef STAGE_CAMERA_RECTIFIER_HPP
#define STAGE_CAMERA_RECTIFIER_HPP

#include <opencv2/core.hpp>

#include <string>

namespace stage_camera {

// SEMES 카메라 사진 규격
struct SemesSpec {
    static constexpr double kFx = 12.0 / 2.8e-3 * 3840.0 / 4096.0;  // 가로만 15/16 축소
    static constexpr double kFy = 12.0 / 2.8e-3;
    static constexpr int kWidth = 3840;
    static constexpr int kHeight = 2160;
};

class Rectifier {
public:
    // calib_path : camera_calib.yml (hardware/camera_calib 의 calibrate 결과)
    // zoom       : 1.0 이면 SEMES 와 같은 화각. 장비 거리 차이로 px/mm 가 조금 다를 때
    //              디지털로 맞추는 배율 (예: 14.0 px/mm 를 14.25 로 → 14.25/14.0)
    // 실패 시 std::runtime_error
    explicit Rectifier(const std::string &calib_path, double zoom = 1.0);

    // gray : 캘리브레이션과 같은 크기(예: 4056x3040)의 CV_8UC1
    // 반환 : 3840x2160 CV_8UC1
    cv::Mat apply(const cv::Mat &gray) const;

    cv::Size inputSize() const { return input_size_; }
    static cv::Size outputSize() { return {SemesSpec::kWidth, SemesSpec::kHeight}; }
    const cv::Mat &cameraMatrix() const { return K_; }   // 원본 카메라 행렬
    const cv::Mat &distCoeffs() const { return D_; }
    const cv::Mat &targetMatrix() const { return K_out_; }  // 결과 사진 카메라 행렬
    double zoom() const { return zoom_; }
    // 캘리브레이션 파일의 roll_deg (광축 기준 기울기 보정, 없으면 0). 양수면 결과 사진이 시계 방향으로 돈다
    double rollDeg() const { return roll_deg_; }
    // 결과 사진 중 원본 바깥을 가리켜 검게 채워지는 픽셀 비율 (0 이 정상)
    double outsideFraction() const { return outside_fraction_; }

private:
    cv::Size input_size_;
    cv::Mat K_, D_, K_out_, R_;
    cv::Mat map1_, map2_;
    double zoom_ = 1.0;
    double roll_deg_ = 0.0;
    double outside_fraction_ = 0.0;
};

}  // namespace stage_camera

#endif  // STAGE_CAMERA_RECTIFIER_HPP
