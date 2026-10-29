#ifndef __MODULE_PLUGIN_H__
#define __MODULE_PLUGIN_H__

/*
 * C ABI between the service (src/model_loader.cpp) and one candidate module DLL.
 * Each DLL wraps exactly one joint() or control() implementation, so several
 * candidates with the same C++ symbol names can be loaded side by side.
 * Built by tool/module_plugin from src/module_plugin.cpp.
 */

#define SEMES_MODULE_API_VERSION 1

#ifdef _WIN32
#define SEMES_MODULE_EXPORT __declspec(dllexport)
#else
#define SEMES_MODULE_EXPORT
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct SemesEstimate {
    double y;
    double t;
    int measurement_valid;
    int outside_training_range;
    double minimum_qr_correlation;
    double minimum_line_correlation;
    char selected_axis[8];
    char selected_branch[16];
    int tracked_points;
    int valid_points;
    double model_cost;
    double other_axis_cost;
    double normalized_cost_margin;
    double elapsed_ms;
    char message[256];
} SemesEstimate;

/* Every function returns 0 on success. On failure the UTF-8 reason is written to error. */
typedef int (*semes_module_api_version_fn)(void);
typedef const char* (*semes_module_kind_fn)(void);
typedef int (*semes_module_initialize_fn)(const char* assets_utf8, char* error, int error_size);
typedef int (*semes_module_estimate_fn)(const unsigned char* data, int rows, int cols, int step, int cv_type,
                                        SemesEstimate* out, char* error, int error_size);

SEMES_MODULE_EXPORT int semes_module_api_version(void);
SEMES_MODULE_EXPORT const char* semes_module_kind(void); /* "joint" or "control" */
SEMES_MODULE_EXPORT int semes_module_initialize(const char* assets_utf8, char* error, int error_size);
SEMES_MODULE_EXPORT int semes_module_estimate(const unsigned char* data, int rows, int cols, int step, int cv_type,
                                              SemesEstimate* out, char* error, int error_size);

#ifdef __cplusplus
}
#endif

#endif /* __MODULE_PLUGIN_H__ */
