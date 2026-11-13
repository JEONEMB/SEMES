#include "alignment.hpp"
#include <iostream>
#include <limits>

namespace {
pa::PhaseAligner g_aligner;
bool g_ready = false;
}

pa::PhaseAligner& alignment_engine() { return g_aligner; }

bool alignment_init(const std::string& refPath, const std::string& calibPath)
{
    g_ready = false;

    // 1) 보정 파일 : 선형 모델 + 학습 때 쓴 필터 매개변수
    pa::Calibration cal;
    pa::PhaseParams params;
    if (!cal.load(calibPath, &params)) {
        std::cerr << "[alignment] 보정 파일을 읽을 수 없습니다: " << calibPath
                  << "  (main 의 학습 단계를 먼저 실행하세요)\n";
        return false;
    }
    g_aligner.setParams(params);
    g_aligner.setCalibration(cal);

    // 2) 기준 영상 : 경계 위치와 기준 위상
    Mat ref = imread(refPath, IMREAD_GRAYSCALE);
    if (ref.empty() || !g_aligner.setReference(ref)) {
        std::cerr << "[alignment] 기준 영상을 읽거나 QR 3개를 찾을 수 없습니다: " << refPath << "\n";
        return false;
    }
    g_ready = true;
    return true;
}

Alignment_t alignment(Mat& src)
{
    Alignment_t result;
    result.y = std::numeric_limits<double>::quiet_NaN();
    result.t = std::numeric_limits<double>::quiet_NaN();

    if (!g_ready && !alignment_init()) return result;

    // 거친 단계(위상 상관) → 정밀 단계(QR 3개 위상 변위장) → 특징 [tL,sL,tC,sC,tR,sR] → 일반화 최소제곱
    pa::Features f = g_aligner.measure(src);
    pa::Estimate e = g_aligner.estimate(f);
    if (!e.valid) return result;

    if (f.numValidQR < pa::NUM_QR)
        std::cerr << "[alignment] 경고: 품질 기준을 통과한 QR " << f.numValidQR << "/3 (결과 신뢰도 낮음)\n";

    result.y = e.y;   // 기준 대비 Y 오차 (mm)
    result.t = e.t;   // 기준 대비 Θ 오차 (deg)
    return result;
}
