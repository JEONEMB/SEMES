// =====================================================================================
// team_alignment_plugin.cpp — 팀 공통 alignment() 를 시뮬레이터 플러그인(C 함수)으로 감싸는 래퍼
//
//  팀원은 이 파일을 고치지 않는다. team/<이름>/ 에 include/, src/ 를 넣으면 CMake 가 이 파일과 함께 DLL 로 만든다
//  · 사진 : 시뮬레이터가 넘긴 8 bit 사진 → (필요하면) BGR 3채널 cv::Mat → alignment(src)
//  · 결과 : y, t 가 유한하면 성공. NaN · 무한대 · 예외는 실패로 바꿔 돌려줌 (C 경계 밖으로 예외를 내보내지 않음)
//  · 제어기형 코드(보정 명령을 돌려줌) : sim_team_output_is_command() = true 이면 부호를 바꿔 '어긋남' 으로 넘김
// =====================================================================================
#include "sim_plugin.h"
#include "sim_team_hooks.hpp"

#include "alignment.hpp"   // 팀원 파일 : Alignment_t alignment(cv::Mat &src);

#include <opencv2/imgproc.hpp>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <exception>
#include <string>

#ifndef SIM_PLUGIN_NAME
#define SIM_PLUGIN_NAME "team alignment()"
#endif

namespace {

void put(sim_result_t* out, const std::string& msg) {
    std::snprintf(out->message, sizeof(out->message), "%s", msg.c_str());
}

// sim_image_t → cv::Mat (복사본 : 팀원 코드가 사진을 바꿔도 시뮬레이터 쪽에 영향 없음)
cv::Mat toMat(const sim_image_t* im, bool wantBgr) {
    if (!im || !im->data || im->width_px <= 0 || im->height_px <= 0) return cv::Mat();
    const int type = im->channels == 3 ? CV_8UC3 : CV_8UC1;
    const cv::Mat view(im->height_px, im->width_px, type, const_cast<unsigned char*>(im->data), size_t(im->stride_bytes));
    cv::Mat out;
    if (wantBgr && im->channels == 1) cv::cvtColor(view, out, cv::COLOR_GRAY2BGR);
    else out = view.clone();
    return out;
}

} // namespace

extern "C" {

SIM_PLUGIN_EXPORT int sim_plugin_api_version(void) { return SIM_PLUGIN_API_VERSION; }

SIM_PLUGIN_EXPORT const char* sim_plugin_name(void) { return SIM_PLUGIN_NAME; }

SIM_PLUGIN_EXPORT int sim_plugin_init(const char* plugin_dir, const char* config_path) {
    try {
        std::string msg;
        return sim_team_init(plugin_dir ? plugin_dir : "", config_path ? config_path : "", msg) ? 1 : 0;
    } catch (...) {
        return 0;
    }
}

SIM_PLUGIN_EXPORT int sim_plugin_set_reference(const sim_image_t* reference) {
    try {
        const cv::Mat ref = toMat(reference, sim_team_wants_bgr());
        if (ref.empty()) return 0;
        return sim_team_set_reference(ref) ? 1 : 0;
    } catch (...) {
        return 0;
    }
}

SIM_PLUGIN_EXPORT int sim_plugin_estimate(const sim_image_t* image, sim_result_t* out) {
    if (!out) return 0;
    std::memset(out, 0, sizeof(*out));
    try {
        cv::Mat src = toMat(image, sim_team_wants_bgr());
        if (src.empty()) { put(out, "빈 사진"); return 1; }
        sim_team_before_estimate();
        const Alignment_t a = alignment(src);
        const double sign = sim_team_output_is_command() ? -1.0 : 1.0;   // 명령 = −어긋남
        out->y_mm = sign * double(a.y) + 0.0;   // + 0.0 : −0 을 0 으로
        out->t_deg = sign * double(a.t) + 0.0;
        out->ok = std::isfinite(out->y_mm) && std::isfinite(out->t_deg) ? 1 : 0;
        if (!out->ok) put(out, "alignment() 가 NaN 을 돌려줌 (측정 실패)");
        return 1;
    } catch (const std::exception& e) {
        put(out, std::string("alignment() 예외 : ") + e.what());
        return 1;
    } catch (...) {
        put(out, "alignment() 에서 알 수 없는 예외");
        return 1;
    }
}

} // extern "C"
