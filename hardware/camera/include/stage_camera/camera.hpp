// camera.hpp — 보정 앱이 부르는 촬영 함수 (Raspberry Pi HQ 카메라 IMX477)
//
//  Camera cam(opt);                         // 카메라 열고 촬영 시작 (노출·게인 고정)
//  Shot ref = cam.captureSettled(4);        // 기준(center) 사진: 흔들림이 멎은 뒤 4장 평균
//  alignment_set_reference(ref.image);
//  ... 이동 명령, DONE 대기 ...
//  Shot s = cam.captureSettled();           // 흔들림이 멎은 뒤 1장
//  if (!s.settle.settled) { /* 시간 초과: 재시도 여부는 호출 측이 결정 */ }
//  Alignment_t r = alignment(s.image);      // s.image = 3840x2160 CV_8UC1
//
//  - 사진은 항상 함수를 부른 뒤에 노출이 시작된 프레임만 쓴다 (이동 전에 찍힌 프레임이 섞이지 않음)
//  - 초점·조리개는 렌즈 링으로 고정한다 (HQ 카메라는 소프트웨어 초점 조절이 없음)
//  - 실패(카메라 없음, 프레임 시간 초과)는 std::runtime_error
#ifndef STAGE_CAMERA_CAMERA_HPP
#define STAGE_CAMERA_CAMERA_HPP

#include <opencv2/core.hpp>

#include <cstdint>
#include <memory>
#include <string>

#include "stage_camera/rectifier.hpp"

namespace stage_camera {

// 안정 대기 설정
//  기준 프레임을 잡고, 이후 프레임이 기준에서 max_shift_px 이내로 stable_frames 장 이어지면 안정으로 본다.
//  벗어나면 그 프레임을 새 기준으로 다시 센다 (천천히 흘러가는 움직임도 잡힘)
struct SettleOptions {
    int min_wait_ms = 200;         // 최소 대기 (진동이 시작되기 전 프레임으로 통과하는 것 방지)
    int timeout_ms = 5000;         // 이 시간 안에 안정되지 않으면 settled = false 로 반환
    double max_shift_px = 0.5;     // 허용 흔들림, 원본(센서) 픽셀 기준 (정지 상태 측정 잡음 약 0.2)
    int stable_frames = 3;         // 연속으로 허용 범위 안이어야 하는 프레임 수
    double analysis_scale = 0.25;  // 비교용 축소 배율 (속도)
};

struct CameraOptions {
    std::string calib_path = "camera_calib.yml";  // hardware/camera_calib 결과
    double zoom = 1.0;                            // Rectifier 참고
    int camera_index = 0;
    int exposure_us = 8333;     // 60Hz 조명 깜빡임 → 8333us 의 배수
    float analogue_gain = 2.0f;
    float colour_gains[2] = {3.3f, 1.4f};  // 빨강, 파랑 (흑백 밝기에 영향 → 고정값 사용)
    double fps = 10.0;                     // 4056x3040 전체 센서 모드 최대 약 10fps
    int frame_timeout_ms = 2000;           // 프레임 하나를 기다리는 최대 시간
    SettleOptions settle;
};

struct FrameInfo {
    int64_t sequence = -1;
    int64_t sensor_timestamp_ns = 0;  // 노출 시작 시각 (libcamera SensorTimestamp)
    int exposure_us = 0;              // 실제 적용된 값 (메타데이터)
    float analogue_gain = 0.0f;
    float lux = 0.0f;
};

struct SettleResult {
    bool settled = false;
    int frames = 0;             // 비교한 프레임 수
    double last_shift_px = -1;  // 마지막 비교의 이동량 (원본 픽셀)
    double max_shift_px = 0;    // 대기 중 가장 컸던 이동량
    int waited_ms = 0;
};

struct Shot {
    cv::Mat image;  // 3840x2160 CV_8UC1 (SEMES 규격) → 알고리즘 입력
    cv::Mat raw;    // 센서 원본 흑백 (평균한 경우 평균 결과). 기록·재캘리브레이션용
    FrameInfo info;           // 마지막으로 쓴 프레임
    int frames_averaged = 1;
    SettleResult settle;      // capture() 에서는 비어 있음
};

class Camera {
public:
    explicit Camera(const CameraOptions &opt);
    ~Camera();
    Camera(const Camera &) = delete;
    Camera &operator=(const Camera &) = delete;

    // 지금 새 프레임을 찍는다 (안정 대기 없음). average 장을 평균
    Shot capture(int average = 1);
    // 흔들림이 멎을 때까지 기다린 뒤 찍는다. 시간 초과여도 마지막 상태로 찍어 반환 (settle.settled 확인)
    Shot captureSettled(int average = 1);
    // 안정 대기만 한다
    SettleResult waitSettled();

    const Rectifier &rectifier() const;
    const CameraOptions &options() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace stage_camera

#endif  // STAGE_CAMERA_CAMERA_HPP
