// team_alignment.cpp — 팀 공통 인터페이스 alignment() 를 통합 정렬기로 구현
//  include/alignment.hpp (shin) 의 선언을 그대로 따르고, 이제 y [mm], t [°] 를 실제로 채운다
//  (shin 의 src/alignment.cpp 는 검출만 하고 NaN 을 돌려주므로 통합 라이브러리에서는 이 파일로 바꿔 넣음)
#include "integrated/integrated.hpp"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <mutex>

namespace {

std::mutex g_mutex;
integrated::Aligner g_aligner;
bool g_tried = false;
integrated::Result g_last;

// 설정 파일 찾기 : 환경 변수 INTEGRATED_CONFIG → config/integrated.yml → src/Integrated/config/integrated.yml
std::string find_config() {
    if (const char *env = std::getenv("INTEGRATED_CONFIG")) return env;
    for (const char *p : {"config/integrated.yml", "src/Integrated/config/integrated.yml", "../config/integrated.yml"}) {
        std::ifstream f(p);
        if (f.good()) return p;
    }
    return "config/integrated.yml";
}

bool ensure_ready() {
    if (g_aligner.ready()) return true;
    if (g_tried) return false;
    g_tried = true;
    std::string error;
    const std::string path = find_config();
    if (!g_aligner.load(path, error)) {
        std::cerr << "[integrated] " << path << " : " << error << "\n";
        return false;
    }
    return true;
}

} // namespace

namespace integrated {
// 진단용 : 마지막 alignment() 결과 (검증 상세 포함)
const Result &last_result() { return g_last; }
// 설정을 명시적으로 다시 읽기
bool configure(const std::string &path, std::string &error) {
    std::lock_guard<std::mutex> lock(g_mutex);
    g_tried = true;
    return g_aligner.load(path, error);
}
} // namespace integrated

Alignment_t alignment(cv::Mat &source) { return alignment(source, AlignmentOptions{}, nullptr); }

Alignment_t alignment(cv::Mat &source, const AlignmentOptions &options, PlateDetectionResult *detection) {
    (void)options;   // QR 정밀화 방식 선택 등은 shin 기본값(Fourier)을 그대로 씀
    std::lock_guard<std::mutex> lock(g_mutex);
    Alignment_t out;
    if (detection) *detection = PlateDetectionResult{};
    if (source.empty() || !ensure_ready()) return out;   // NaN
    g_last = g_aligner.estimate(source);
    if (g_last.ok) { out.y = g_last.y_mm; out.t = g_last.t_deg; }
    return out;
}
