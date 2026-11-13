// =====================================================================================
// sim_estimator.cpp — 내장 추정기(ksh)와 플러그인 추정기(DLL)
// =====================================================================================
#include "sim_estimator.hpp"
#include "sim_paths.hpp"
#include "../sdk/sim_plugin.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <sstream>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace fs = std::filesystem;

namespace sim {

namespace {

#ifdef _WIN32
constexpr const char* kPluginExt = ".dll";
#else
constexpr const char* kPluginExt = ".so";
#endif

// ---------------------------------------------------------------------------------- ksh 내장
class KshEstimator : public Estimator {
public:
    bool init(const std::string& configPath, std::string& message) {
        // 설정의 기준 사진 경로가 없으면 Alignment 가 경고를 찍지만, 시뮬레이터는 기준 사진을 따로 넣으므로 감춤
        std::ostringstream sink;
        std::streambuf* old = std::cerr.rdbuf(sink.rdbuf());
        const bool ok = alignment_init(configPath);
        std::cerr.rdbuf(old);
        if (!ok) message = "Alignment 설정을 읽지 못함 : " + configPath + (sink.str().empty() ? "" : "\n" + sink.str());
        return ok;
    }
    std::string name() const override { return "ksh (내장 Alignment)"; }
    bool supportsReference() const override { return true; }
    bool setReference(const cv::Mat& image) override { return alignment_set_reference(image); }
    EstimateResult estimate(const cv::Mat& image) override {
        cv::Mat src = image;   // alignment() 는 비 const 참조를 받지만 바꾸지 않음
        const Alignment_t a = alignment(src);
        EstimateResult r;
        r.ok = alignment_last_info().ok && std::isfinite(a.y) && std::isfinite(a.t);
        r.y_mm = a.y; r.t_deg = a.t;
        r.message = alignment_last_info().message;
        return r;
    }
    const AlignmentInfo* info() const override { return &alignment_last_info(); }
};

// ---------------------------------------------------------------------------------- 이혜원 모듈 DLL 경계 (include/module_plugin.h)
//  팀 저장소의 joint/control 후보 모듈 DLL 을 다시 빌드하지 않고 그대로 읽기 위한 같은 모양의 선언
//  (함수는 0 = 성공, 실패 이유는 error 에 UTF-8)
extern "C" {
typedef struct SemesEstimate {
    double y, t;
    int measurement_valid, outside_training_range;
    double minimum_qr_correlation, minimum_line_correlation;
    char selected_axis[8];
    char selected_branch[16];
    int tracked_points, valid_points;
    double model_cost, other_axis_cost, normalized_cost_margin, elapsed_ms;
    char message[256];
} SemesEstimate;
typedef int (*semes_module_api_version_fn)(void);
typedef const char* (*semes_module_kind_fn)(void);
typedef int (*semes_module_initialize_fn)(const char* assets_utf8, char* error, int error_size);
typedef int (*semes_module_estimate_fn)(const unsigned char* data, int rows, int cols, int step, int cv_type, SemesEstimate* out, char* error,
                                        int error_size);
}
constexpr int kSemesApiVersion = 1;

// ---------------------------------------------------------------------------------- 플러그인 (DLL)
//  두 가지 경계를 받음 : ① sim_plugin_* (sdk/sim_plugin.h)  ② semes_module_* (팀 저장소 모듈 DLL)
class PluginEstimator : public Estimator {
public:
    ~PluginEstimator() override {
#ifdef _WIN32
        if (lib_) FreeLibrary(HMODULE(lib_));
#else
        if (lib_) dlclose(lib_);
#endif
    }

    bool open(const fs::path& path, const std::string& pluginConfig, std::string& message) {
        std::error_code ec;
        const fs::path abs = fs::absolute(path, ec);
#ifdef _WIN32
        // 절대 경로 + LOAD_WITH_ALTERED_SEARCH_PATH : 플러그인이 쓰는 DLL 을 플러그인 폴더에서도 찾음
        lib_ = LoadLibraryExW(abs.wstring().c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
        if (!lib_) { message = "DLL 을 읽지 못함 (오류 " + std::to_string(GetLastError()) + ") : " + u8s(abs); return false; }
#else
        lib_ = dlopen(abs.string().c_str(), RTLD_NOW | RTLD_LOCAL);
        if (!lib_) { const char* e = dlerror(); message = "라이브러리를 읽지 못함 : " + std::string(e ? e : u8s(abs)); return false; }
#endif
        const auto ver = sym<sim_plugin_api_version_fn>("sim_plugin_api_version");
        const auto nm = sym<sim_plugin_name_fn>("sim_plugin_name");
        const auto ini = sym<sim_plugin_init_fn>("sim_plugin_init");
        setRef_ = sym<sim_plugin_set_reference_fn>("sim_plugin_set_reference");
        est_ = sym<sim_plugin_estimate_fn>("sim_plugin_estimate");
        if (!ver && !est_) return openSemes(path, abs, pluginConfig, message);
        if (!ver || !nm || !ini || !est_) {
            message = "플러그인 함수가 없음 (sim_plugin_api_version / name / init / estimate 필요) : " + u8s(abs);
            return false;
        }
        if (ver() != SIM_PLUGIN_API_VERSION) {
            message = "플러그인 API 버전 " + std::to_string(ver()) + " ≠ 시뮬레이터 " + std::to_string(SIM_PLUGIN_API_VERSION) + " : " + u8s(abs);
            return false;
        }
        const char* n = nm();
        name_ = std::string(n ? n : "plugin") + " (" + u8s(path.filename()) + ")";
        if (!ini(u8s(abs.parent_path()).c_str(), pluginConfig.c_str())) { message = "플러그인 초기화 실패 : " + name_; return false; }
        message = "플러그인을 읽음 : " + name_;
        return true;
    }

    std::string name() const override { return name_; }
    bool supportsReference() const override { return refOk_; }
    bool setReference(const cv::Mat& image) override {
        refOk_ = false;
        if (!setRef_ || image.empty()) return false;
        cv::Mat m;
        const sim_image_t im = wrap(image, m);
        refOk_ = setRef_(&im) == 1;
        return refOk_;
    }
    EstimateResult estimate(const cv::Mat& image) override {
        if (semesEst_) return estimateSemes(image);
        EstimateResult r;
        cv::Mat m;
        const sim_image_t im = wrap(image, m);
        sim_result_t out;
        std::memset(&out, 0, sizeof(out));
        int called = 0;
        try {
            called = est_(&im, &out);
        } catch (...) {   // C 경계 밖으로 예외가 새어 나온 경우 (래퍼가 막아야 하지만 대비)
            r.message = "플러그인에서 예외";
            return r;
        }
        out.message[sizeof(out.message) - 1] = '\0';
        r.message = out.message;
        if (!called) { if (r.message.empty()) r.message = "플러그인 호출 실패"; return r; }
        r.ok = out.ok == 1 && std::isfinite(out.y_mm) && std::isfinite(out.t_deg);
        r.y_mm = out.y_mm; r.t_deg = out.t_deg;
        if (!r.ok && r.message.empty()) r.message = "플러그인 추정 실패";
        return r;
    }

private:
    void* lib_ = nullptr;
    std::string name_;
    sim_plugin_set_reference_fn setRef_ = nullptr;
    sim_plugin_estimate_fn est_ = nullptr;
    semes_module_estimate_fn semesEst_ = nullptr;
    bool refOk_ = false;

    // semes_module_* : 자산 폴더 = plugin_config → 없으면 DLL 옆 assets/ → DLL 폴더. 기준 사진은 자산 폴더의 center.bmp (교체 미지원)
    bool openSemes(const fs::path& path, const fs::path& abs, const std::string& pluginConfig, std::string& message) {
        const auto ver = sym<semes_module_api_version_fn>("semes_module_api_version");
        const auto kind = sym<semes_module_kind_fn>("semes_module_kind");
        const auto ini = sym<semes_module_initialize_fn>("semes_module_initialize");
        semesEst_ = sym<semes_module_estimate_fn>("semes_module_estimate");
        if (!ver || !ini || !semesEst_) {
            message = "알 수 없는 DLL : sim_plugin_* 또는 semes_module_* 함수가 없음 : " + u8s(abs);
            return false;
        }
        if (ver() != kSemesApiVersion) { message = "모듈 API 버전 " + std::to_string(ver()) + " 을 지원하지 않음 : " + u8s(abs); return false; }
        std::error_code ec;
        std::string assets = pluginConfig;
        if (assets.empty()) assets = fs::is_directory(abs.parent_path() / "assets", ec) ? u8s(abs.parent_path() / "assets") : u8s(abs.parent_path());
        char err[512] = {0};
        if (ini(assets.c_str(), err, int(sizeof(err))) != 0) { message = "모듈 초기화 실패 (" + assets + ") : " + std::string(err); return false; }
        const char* k = kind ? kind() : nullptr;
        name_ = std::string("module ") + (k ? k : "?") + " (" + u8s(path.filename()) + ")";
        message = "모듈 DLL 을 읽음 : " + name_ + ", 자산 " + assets;
        return true;
    }
    EstimateResult estimateSemes(const cv::Mat& image) {
        EstimateResult r;
        cv::Mat m;
        const sim_image_t im = wrap(image, m);
        SemesEstimate out;
        std::memset(&out, 0, sizeof(out));
        char err[512] = {0};
        int rc = -1;
        try {
            rc = semesEst_(im.data, im.height_px, im.width_px, im.stride_bytes, im.channels == 3 ? CV_8UC3 : CV_8UC1, &out, err, int(sizeof(err)));
        } catch (...) {
            r.message = "모듈에서 예외";
            return r;
        }
        out.message[sizeof(out.message) - 1] = '\0';
        if (rc != 0) { r.message = err[0] ? err : "모듈 추정 실패"; return r; }
        r.ok = out.measurement_valid != 0 && std::isfinite(out.y) && std::isfinite(out.t);
        r.y_mm = out.y; r.t_deg = out.t;
        r.message = out.message;
        if (out.outside_training_range) r.message += (r.message.empty() ? "" : " · ") + std::string("학습 범위 밖");
        return r;
    }

    template <typename F> F sym(const char* s) const {
#ifdef _WIN32
        return reinterpret_cast<F>(reinterpret_cast<void*>(GetProcAddress(HMODULE(lib_), s)));
#else
        return reinterpret_cast<F>(dlsym(lib_, s));
#endif
    }
    // cv::Mat → sim_image_t (8 bit 1·3 채널만. 그 밖은 8 bit 로 바꿔 hold 에 보관)
    static sim_image_t wrap(const cv::Mat& image, cv::Mat& hold) {
        hold = image;
        if (hold.depth() != CV_8U) hold.convertTo(hold, CV_8U);
        if (hold.channels() == 4) { cv::Mat c; cv::extractChannel(hold, c, 0); hold = c; }   // 드묾 : 첫 채널만
        sim_image_t im;
        im.width_px = hold.cols; im.height_px = hold.rows; im.channels = hold.channels();
        im.stride_bytes = int(hold.step[0]); im.data = hold.data;
        return im;
    }
};

// 확장자가 없으면 .dll / .so 를 붙이고, 폴더가 없으면 plugins/ 에서 찾음
fs::path findPlugin(const std::string& spec) {
    fs::path p = u8p(spec);
    if (!p.has_extension()) p += kPluginExt;
    std::error_code ec;
    if (fs::exists(p, ec)) return p;
    if (!p.has_parent_path()) {
        const fs::path q = fs::path("plugins") / p;
        if (fs::exists(q, ec)) return q;
        const fs::path r = fs::path("plugins") / p.stem() / p;   // plugins/<이름>/<이름>.dll
        if (fs::exists(r, ec)) return r;
    }
    return p;
}

} // namespace

std::unique_ptr<Estimator> makeEstimator(const SimConfig& cfg, std::string& message) {
    if (cfg.estimator_type == "plugin") {
        if (cfg.estimator_plugin.empty()) { message = "estimator.plugin 이 비어 있음"; return nullptr; }
        auto p = std::make_unique<PluginEstimator>();
        if (!p->open(findPlugin(cfg.estimator_plugin), cfg.estimator_plugin_config, message)) return nullptr;
        return p;
    }
    if (cfg.alignment_config.empty()) {
        message = "Alignment 설정 파일이 없음 (paths.alignment_configs 후보를 확인)";
        return nullptr;
    }
    auto k = std::make_unique<KshEstimator>();
    if (!k->init(cfg.alignment_config, message)) return nullptr;
    message = "추정기 : " + k->name() + ", 설정 " + cfg.alignment_config;
    return k;
}

std::vector<std::string> listPlugins(const std::string& pluginDir) {
    std::vector<std::string> v;
    std::error_code ec;
    if (!fs::is_directory(u8p(pluginDir), ec)) return v;
    for (auto it = fs::recursive_directory_iterator(u8p(pluginDir), ec); it != fs::recursive_directory_iterator(); it.increment(ec)) {
        if (ec) break;
        if (it.depth() > 1) { it.disable_recursion_pending(); continue; }
        if (!it->is_regular_file(ec)) continue;
        std::string ext = it->path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return char(std::tolower(c)); });
        if (ext == kPluginExt) { const auto g = it->path().generic_u8string(); v.emplace_back(g.begin(), g.end()); }
    }
    std::sort(v.begin(), v.end());
    return v;
}

} // namespace sim
