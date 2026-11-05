/* =====================================================================================
 * sim_plugin.h — 시뮬레이터 추정기 플러그인 경계 (C 함수, API 버전 1)
 *
 *  시뮬레이터 exe 가 plugins/<이름>.dll (Linux .so) 을 실행 중에 읽어 아래 함수를 부른다.
 *  경계를 C 함수 + 기본 자료형으로 둔 이유
 *   · exe 와 DLL 의 컴파일러·STL·OpenCV 빌드가 조금 달라도 안전 (std::string, cv::Mat 을 넘기지 않음)
 *   · 팀 공통 함수 이름 alignment() 가 exe 안의 내장 추정기(ksh)와 겹치지 않음
 *
 *  호출 순서 : api_version → name → init → (set_reference) → estimate 반복
 *  사진 : 8 bit, channels = 1 (회색, 시뮬레이터 합성 사진) 또는 3 (BGR, 실제 사진). 함수가 반환된 뒤에는 data 를 쓰지 말 것
 *  단위 : y_mm [mm], t_deg [°] — 기준(티칭) 사진 대비 어긋남, 라벨과 같은 부호 (보정 이동 = −y, −t)
 * ===================================================================================== */
#ifndef SIM_PLUGIN_H
#define SIM_PLUGIN_H

#define SIM_PLUGIN_API_VERSION 1

#ifdef _WIN32
#define SIM_PLUGIN_EXPORT __declspec(dllexport)
#else
#define SIM_PLUGIN_EXPORT __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct sim_image_t {
    int width_px;
    int height_px;
    int channels;                 /* 1 = 회색, 3 = BGR */
    int stride_bytes;             /* 한 줄 바이트 수 */
    const unsigned char *data;
} sim_image_t;

typedef struct sim_result_t {
    int ok;                       /* 1 = 추정 성공, 0 = 측정 실패 (y, t 는 무시) */
    double y_mm;
    double t_deg;
    char message[256];            /* 실패 이유 등 (UTF-8, NUL 끝) */
} sim_result_t;

/* 필수 : SIM_PLUGIN_API_VERSION 을 돌려줌 (다르면 시뮬레이터가 읽지 않음) */
typedef int (*sim_plugin_api_version_fn)(void);
/* 필수 : 화면에 보일 이름 (UTF-8) */
typedef const char *(*sim_plugin_name_fn)(void);
/* 필수 : 초기화. plugin_dir = DLL 이 있는 폴더, config_path = simulator.yml 의 estimator.plugin_config (빈 문자열 가능)
 *        작업 폴더 = 시뮬레이터 루트 (data/img, config 가 있는 곳). 1 = 성공 */
typedef int (*sim_plugin_init_fn)(const char *plugin_dir, const char *config_path);
/* 선택 : 기준(티칭) 사진 교체. 1 = 반영, 0 = 지원 안 함 (그러면 플러그인 자체 기준 사진을 씀) */
typedef int (*sim_plugin_set_reference_fn)(const sim_image_t *reference);
/* 필수 : 한 장 추정. 1 = 호출 성공 (결과는 out->ok 로 판단), 0 = 호출 자체 실패 */
typedef int (*sim_plugin_estimate_fn)(const sim_image_t *image, sim_result_t *out);

#ifdef __cplusplus
}
#endif

#endif /* SIM_PLUGIN_H */
