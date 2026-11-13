// =====================================================================================
// alignment.cpp — 외부에 제공하는 함수 : 영상 한 장 → 기준 대비 Y·Θ  (가설 H2 : ECC 전역 원근 정합)
// =====================================================================================
#include "alignment.hpp"
#include "ecc_align.hpp"
#include <fstream>
#include <limits>

namespace {
ea::EccAligner g_aligner;
ea::Model      g_model;
bool           g_ready = false;

cv::Mat readGrayFile(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return cv::Mat();
    std::vector<uchar> buf((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    return buf.empty() ? cv::Mat() : cv::imdecode(buf, cv::IMREAD_GRAYSCALE);
}
}

bool alignment_init(const std::string& refPath, const std::string& modelPath)
{
    g_ready = false;
    if (!g_model.load(modelPath)) return false;
    g_aligner.setParams(g_model.p);                    // 모델과 같은 ECC 매개변수로 측정
    cv::Mat ref = readGrayFile(refPath);
    if (ref.empty() || !g_aligner.setReference(ref)) return false;
    g_ready = true;
    return true;
}

Alignment_t alignment(cv::Mat& src)
{
    const double nan = std::numeric_limits<double>::quiet_NaN();
    Alignment_t result;
    result.y = nan;
    result.t = nan;
    if (!g_ready && !alignment_init()) return result;

    ea::Measure m = g_aligner.measure(src);
    if (!m.ok) return result;
    ea::Estimate e = ea::estimate(g_model, m.f);
    if (!e.valid) return result;
    result.y = e.y;
    result.t = e.t;
    return result;
}
