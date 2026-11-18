// libcamera_source.hpp — libcamera 로 계속 촬영하며 가장 최근 프레임을 보관한다 (내부용)
//
//  - YUV420 의 Y 평면만 복사해 8bit 흑백 cv::Mat 으로 만든다 (sYCC = 풀레인지)
//  - 노출 시간·아날로그 게인·색 게인은 수동 고정 (자동 노출/화이트밸런스 끔)
//  - 프레임은 libcamera 스레드에서 들어오고, next() 가 원하는 번호 이후 프레임을 기다려 돌려준다
#ifndef STAGE_CAMERA_LIBCAMERA_SOURCE_HPP
#define STAGE_CAMERA_LIBCAMERA_SOURCE_HPP

#include <libcamera/libcamera.h>
#include <opencv2/core.hpp>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <vector>

#include "stage_camera/camera.hpp"

namespace stage_camera {

struct RawFrame {
    cv::Mat gray;  // 센서 크기 CV_8UC1 (프레임마다 새로 할당 → 받은 쪽이 그대로 들고 있어도 안전)
    FrameInfo info;
};

class LibcameraSource {
public:
    struct Options {
        cv::Size size{4056, 3040};
        int camera_index = 0;
        int exposure_us = 8333;
        float analogue_gain = 2.0f;
        float colour_gains[2] = {3.3f, 1.4f};  // 빨강, 파랑
        double fps = 10.0;
        int buffer_count = 4;
    };

    explicit LibcameraSource(const Options &opt);  // 실패 시 std::runtime_error
    ~LibcameraSource();
    LibcameraSource(const LibcameraSource &) = delete;
    LibcameraSource &operator=(const LibcameraSource &) = delete;

    void start();
    void stop();

    // 지금까지 받은 마지막 프레임 번호, 아직 없으면 -1
    int64_t lastSequence() const;
    // 번호가 min_seq 이상인 프레임이 올 때까지 기다린다. 시간 초과 시 std::runtime_error
    RawFrame next(int64_t min_seq, std::chrono::milliseconds timeout);

private:
    void onRequestCompleted(libcamera::Request *request);

    Options opt_;
    std::unique_ptr<libcamera::CameraManager> cm_;
    std::shared_ptr<libcamera::Camera> camera_;
    std::unique_ptr<libcamera::CameraConfiguration> config_;
    std::unique_ptr<libcamera::FrameBufferAllocator> allocator_;
    libcamera::Stream *stream_ = nullptr;
    unsigned int stride_ = 0;
    std::vector<std::unique_ptr<libcamera::Request>> requests_;
    struct Mapping {
        void *addr;
        size_t length;
    };
    std::vector<Mapping> mappings_;
    std::map<const libcamera::FrameBuffer *, const uint8_t *> y_plane_;
    bool acquired_ = false;
    std::atomic<bool> running_{false};

    mutable std::mutex mtx_;
    std::condition_variable cv_;
    RawFrame latest_;
    bool have_frame_ = false;
};

}  // namespace stage_camera

#endif  // STAGE_CAMERA_LIBCAMERA_SOURCE_HPP
