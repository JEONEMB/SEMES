#include "stage_camera/camera.hpp"

#include <opencv2/imgproc.hpp>

#include <chrono>
#include <cmath>
#include <stdexcept>

#include "libcamera_source.hpp"

namespace stage_camera {

namespace {

using Clock = std::chrono::steady_clock;

int msSince(Clock::time_point t0)
{
    return static_cast<int>(std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - t0).count());
}

LibcameraSource::Options sourceOptions(const CameraOptions &o, cv::Size sensor_size)
{
    LibcameraSource::Options s;
    s.size = sensor_size;
    s.camera_index = o.camera_index;
    s.exposure_us = o.exposure_us;
    s.analogue_gain = o.analogue_gain;
    s.colour_gains[0] = o.colour_gains[0];
    s.colour_gains[1] = o.colour_gains[1];
    s.fps = o.fps;
    return s;
}

// 처음 몇 프레임은 수동 노출이 아직 적용되지 않았을 수 있어 버린다
constexpr int64_t kWarmupFrames = 4;

}  // namespace

struct Camera::Impl {
    CameraOptions opt;
    Rectifier rect;
    LibcameraSource src;
    std::chrono::milliseconds frame_timeout;
    cv::Mat hann;  // 위상 상관용 창 (분석 크기)

    explicit Impl(const CameraOptions &o)
        : opt(o), rect(o.calib_path, o.zoom), src(sourceOptions(o, rect.inputSize())),
          frame_timeout(o.frame_timeout_ms)
    {
    }

    // 이 함수를 부른 뒤에 노출이 시작된 첫 프레임.
    // 마지막으로 받은 프레임 S 다음 프레임(S+1)은 이미 노출 중이었을 수 있으므로 S+2 부터 쓴다
    RawFrame fresh()
    {
        const int64_t last = src.lastSequence();
        return src.next(std::max<int64_t>(last + 2, kWarmupFrames), frame_timeout);
    }

    RawFrame following(const RawFrame &prev) { return src.next(prev.info.sequence + 1, frame_timeout); }

    cv::Mat analysisImage(const cv::Mat &gray)
    {
        cv::Mat small, f;
        cv::resize(gray, small, cv::Size(), opt.settle.analysis_scale, opt.settle.analysis_scale, cv::INTER_AREA);
        small.convertTo(f, CV_32F);
        if (hann.size() != f.size())
            cv::createHanningWindow(hann, f.size(), CV_32F);
        return f;
    }

    // 두 분석 이미지 사이 이동량, 원본 픽셀 단위
    double shiftPx(const cv::Mat &a, const cv::Mat &b)
    {
        const cv::Point2d d = cv::phaseCorrelate(a, b, hann);
        return std::hypot(d.x, d.y) / opt.settle.analysis_scale;
    }

    SettleResult settle(RawFrame *last)
    {
        const SettleOptions &so = opt.settle;
        const auto t0 = Clock::now();
        SettleResult r;

        RawFrame cur = fresh();
        cv::Mat anchor = analysisImage(cur.gray);
        int stable = 0;
        for (;;) {
            r.waited_ms = msSince(t0);
            if (stable >= so.stable_frames && r.waited_ms >= so.min_wait_ms) {
                r.settled = true;
                break;
            }
            if (r.waited_ms >= so.timeout_ms)
                break;

            cur = following(cur);
            const cv::Mat a = analysisImage(cur.gray);
            r.last_shift_px = shiftPx(anchor, a);
            r.max_shift_px = std::max(r.max_shift_px, r.last_shift_px);
            ++r.frames;
            if (r.last_shift_px <= so.max_shift_px) {
                ++stable;
            } else {
                anchor = a;  // 아직 움직이는 중 → 이 프레임을 새 기준으로
                stable = 0;
            }
        }
        if (last)
            *last = std::move(cur);
        return r;
    }

    // first 부터 이어지는 average 장을 평균해 Shot 을 만든다
    Shot makeShot(RawFrame first, int average)
    {
        if (average < 1)
            throw std::invalid_argument("Camera: average 는 1 이상");
        Shot s;
        s.frames_averaged = average;
        if (average == 1) {
            s.raw = first.gray;
            s.info = first.info;
        } else {
            cv::Mat acc;
            first.gray.convertTo(acc, CV_32F);
            RawFrame f = first;
            for (int i = 1; i < average; ++i) {
                f = following(f);
                cv::accumulate(f.gray, acc);
            }
            acc.convertTo(s.raw, CV_8U, 1.0 / average);
            s.info = f.info;
        }
        s.image = rect.apply(s.raw);
        return s;
    }
};

Camera::Camera(const CameraOptions &opt) : impl_(std::make_unique<Impl>(opt))
{
    impl_->src.start();
    // 수동 노출이 적용된 첫 프레임이 올 때까지 기다린다 (스트림 시작 실패를 여기서 알아챔)
    impl_->src.next(kWarmupFrames, std::chrono::milliseconds(opt.frame_timeout_ms + 3000));
}

Camera::~Camera() = default;

Shot Camera::capture(int average)
{
    return impl_->makeShot(impl_->fresh(), average);
}

Shot Camera::captureSettled(int average)
{
    RawFrame last;
    SettleResult r = impl_->settle(&last);
    // 안정 판정에 쓴 마지막 프레임이 곧 안정된 프레임이므로 그대로 첫 장으로 쓴다
    Shot s = impl_->makeShot(std::move(last), average);
    s.settle = r;
    return s;
}

SettleResult Camera::waitSettled()
{
    return impl_->settle(nullptr);
}

const Rectifier &Camera::rectifier() const
{
    return impl_->rect;
}

const CameraOptions &Camera::options() const
{
    return impl_->opt;
}

}  // namespace stage_camera
