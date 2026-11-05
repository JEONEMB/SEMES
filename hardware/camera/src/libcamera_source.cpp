#include "libcamera_source.hpp"

#include <sys/mman.h>

#include <algorithm>
#include <array>
#include <span>
#include <stdexcept>
#include <string>

using namespace libcamera;

namespace stage_camera {

namespace {

[[noreturn]] void fail(const std::string &msg)
{
    throw std::runtime_error("LibcameraSource: " + msg);
}

}  // namespace

LibcameraSource::LibcameraSource(const Options &opt) : opt_(opt)
{
    cm_ = std::make_unique<CameraManager>();
    if (cm_->start())
        fail("CameraManager 시작 실패");

    try {
        auto cams = cm_->cameras();
        if (cams.empty())
            fail("카메라를 찾을 수 없음 (리본 케이블, rpicam-hello --list-cameras 로 확인)");
        if (opt_.camera_index < 0 || opt_.camera_index >= static_cast<int>(cams.size()))
            fail("camera_index 범위 밖: " + std::to_string(opt_.camera_index));
        camera_ = cams[opt_.camera_index];
        if (camera_->acquire())
            fail("카메라를 다른 프로그램이 쓰는 중 (rpicam-*, capture_calib.py 등을 끈다)");
        acquired_ = true;

        config_ = camera_->generateConfiguration({StreamRole::StillCapture});
        if (!config_ || config_->size() != 1)
            fail("설정 생성 실패");
        StreamConfiguration &sc = config_->at(0);
        const Size want(opt_.size.width, opt_.size.height);
        sc.pixelFormat = formats::YUV420;
        sc.size = want;
        sc.bufferCount = opt_.buffer_count;
        sc.colorSpace = ColorSpace::Sycc;  // Y 를 0~255 풀레인지로

        // 캘리브레이션과 같은 센서 모드(전체 화각)를 강제한다. 다른 모드는 화각·중심이 달라진다
        SensorConfiguration sensor;
        sensor.bitDepth = 12;
        sensor.outputSize = want;
        config_->sensorConfig = sensor;

        if (config_->validate() == CameraConfiguration::Invalid)
            fail("설정이 유효하지 않음");
        if (sc.size != want || sc.pixelFormat != formats::YUV420)
            fail("요청한 형식을 쓸 수 없음, 조정된 값: " + sc.toString());
        if (camera_->configure(config_.get()))
            fail("configure 실패");

        stream_ = sc.stream();
        stride_ = sc.stride;

        allocator_ = std::make_unique<FrameBufferAllocator>(camera_);
        if (allocator_->allocate(stream_) < 0)
            fail("버퍼 할당 실패");

        for (const auto &buf : allocator_->buffers(stream_)) {
            const auto &planes = buf->planes();
            const int fd = planes[0].fd.get();
            size_t length = 0;
            for (const auto &p : planes)
                if (p.fd.get() == fd)
                    length = std::max<size_t>(length, p.offset + p.length);
            void *addr = mmap(nullptr, length, PROT_READ, MAP_SHARED, fd, 0);
            if (addr == MAP_FAILED)
                fail("버퍼 mmap 실패");
            mappings_.push_back({addr, length});
            y_plane_[buf.get()] = static_cast<const uint8_t *>(addr) + planes[0].offset;

            std::unique_ptr<Request> req = camera_->createRequest();
            if (!req || req->addBuffer(stream_, buf.get()))
                fail("요청 생성 실패");
            requests_.push_back(std::move(req));
        }

        camera_->requestCompleted.connect(this, &LibcameraSource::onRequestCompleted);
    } catch (...) {
        // 소멸자가 불리지 않으므로 여기서 정리
        for (auto &m : mappings_)
            munmap(m.addr, m.length);
        mappings_.clear();
        requests_.clear();
        allocator_.reset();
        if (acquired_)
            camera_->release();
        camera_.reset();
        cm_->stop();
        throw;
    }
}

LibcameraSource::~LibcameraSource()
{
    stop();
    camera_->requestCompleted.disconnect(this);
    requests_.clear();
    for (auto &m : mappings_)
        munmap(m.addr, m.length);
    allocator_.reset();
    if (acquired_)
        camera_->release();
    camera_.reset();
    cm_->stop();
}

void LibcameraSource::start()
{
    if (running_)
        return;

    ControlList ctrls(camera_->controls());
    const ControlInfoMap &info = camera_->controls();
    auto has = [&](const ControlId &id) { return info.count(id.id()) > 0; };

    // 수동 노출·게인 (libcamera 0.5+ 는 ExposureTimeMode/AnalogueGainMode, 이전은 AeEnable)
    if (has(controls::ExposureTimeMode))
        ctrls.set(controls::ExposureTimeMode, controls::ExposureTimeModeManual);
    if (has(controls::AnalogueGainMode))
        ctrls.set(controls::AnalogueGainMode, controls::AnalogueGainModeManual);
    if (!has(controls::ExposureTimeMode) && has(controls::AeEnable))
        ctrls.set(controls::AeEnable, false);
    ctrls.set(controls::ExposureTime, opt_.exposure_us);
    ctrls.set(controls::AnalogueGain, opt_.analogue_gain);

    if (has(controls::AwbEnable))
        ctrls.set(controls::AwbEnable, false);
    if (has(controls::ColourGains)) {
        const std::array<float, 2> gains{opt_.colour_gains[0], opt_.colour_gains[1]};
        ctrls.set(controls::ColourGains, std::span<const float, 2>(gains));
    }

    if (opt_.fps > 0 && has(controls::FrameDurationLimits)) {
        const int64_t us = static_cast<int64_t>(1e6 / opt_.fps);
        const std::array<int64_t, 2> limits{us, us};
        ctrls.set(controls::FrameDurationLimits, std::span<const int64_t, 2>(limits));
    }

    {
        std::lock_guard<std::mutex> lk(mtx_);
        have_frame_ = false;
    }
    if (camera_->start(&ctrls))
        fail("start 실패");
    running_ = true;
    for (auto &req : requests_) {
        if (camera_->queueRequest(req.get()) < 0) {
            stop();
            fail("queueRequest 실패");
        }
    }
}

void LibcameraSource::stop()
{
    if (!running_.exchange(false))
        return;
    camera_->stop();  // 진행 중인 요청은 취소 상태로 완료된다
}

int64_t LibcameraSource::lastSequence() const
{
    std::lock_guard<std::mutex> lk(mtx_);
    return have_frame_ ? latest_.info.sequence : -1;
}

RawFrame LibcameraSource::next(int64_t min_seq, std::chrono::milliseconds timeout)
{
    std::unique_lock<std::mutex> lk(mtx_);
    const bool ok = cv_.wait_for(lk, timeout, [&] { return have_frame_ && latest_.info.sequence >= min_seq; });
    if (!ok)
        fail("프레임 시간 초과 (" + std::to_string(timeout.count()) + " ms)");
    return latest_;  // cv::Mat 은 참조 카운트 → 복사 비용 없음, 프레임마다 새 메모리라 안전
}

void LibcameraSource::onRequestCompleted(Request *request)
{
    if (request->status() == Request::RequestCancelled)
        return;

    FrameBuffer *buf = request->buffers().begin()->second;
    const FrameMetadata &fmd = buf->metadata();

    if (fmd.status == FrameMetadata::FrameSuccess) {
        RawFrame f;
        const cv::Mat y(opt_.size.height, opt_.size.width, CV_8UC1, const_cast<uint8_t *>(y_plane_.at(buf)),
                        stride_);
        y.copyTo(f.gray);

        const ControlList &md = request->metadata();
        f.info.sequence = fmd.sequence;
        f.info.sensor_timestamp_ns = md.get(controls::SensorTimestamp).value_or(static_cast<int64_t>(fmd.timestamp));
        f.info.exposure_us = md.get(controls::ExposureTime).value_or(0);
        f.info.analogue_gain = md.get(controls::AnalogueGain).value_or(0.0f);
        f.info.lux = md.get(controls::Lux).value_or(0.0f);

        {
            std::lock_guard<std::mutex> lk(mtx_);
            latest_ = std::move(f);
            have_frame_ = true;
        }
        cv_.notify_all();
    }

    if (running_) {
        request->reuse(Request::ReuseBuffers);
        camera_->queueRequest(request);
    }
}

}  // namespace stage_camera
