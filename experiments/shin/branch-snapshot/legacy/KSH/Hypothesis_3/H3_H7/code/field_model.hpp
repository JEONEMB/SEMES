// =====================================================================================
// field_model.hpp — 가설 H3 (선형화 물리 모델) : 변위장을 물리 기저로 분해해 (Y, Θ) 추정
//
//  작은 각에서 순수 카메라 회전은 변위 dx = f·θ·(1 + u²/f²), dy = f·θ·u·v/f² 를 만들고,
//  평행이동은 거의 균일한 이동을 만든다 (Hartley & Zisserman). 그래서 영상마다
//    dx(u,v) = Σ cx_j·φ_j(u,v),  dy(u,v) = Σ cy_j·ψ_j(u,v)
//      φ : 1, u, u², v, uv        ψ : 1, u, v, uv        (u, v = (x − 1912, y − 1080) / 1000)
//  로 분해하고, 계수 벡터 k 를 k = a·Y + b·Θ + q·Θ² + c (잔차 공분산 Σ) 로 학습해 GLS 로 (Y, Θ) 를 구한다.
//  자유 워프(8~9 매개변수) 대신 물리적으로 의미 있는 항만 쓰는 것이 H3 의 핵심이다.
// =====================================================================================
#ifndef __H3_FIELD_MODEL_HPP__
#define __H3_FIELD_MODEL_HPP__

#include "qr_track.hpp"
#include <string>
#include <vector>

namespace h3 {

enum Basis { B_ONE = 0, B_U, B_U2, B_V, B_UV, B_V2, B_U3 };
const char* basisName(Basis b);

struct FieldSpec {
    std::vector<Basis> bx = { B_ONE, B_U, B_U2, B_V, B_UV };
    std::vector<Basis> by = { B_ONE, B_U, B_V, B_UV };
    double cx_px = 1912.0, cy_px = 1080.0, scale_px = 1000.0;
    bool   quad = true;                 // Θ² 항
    std::string label() const;
};

double basisValue(Basis b, double u, double v);

// 영상 한 장의 기저 계수 (로버스트 최소제곱 3회 : 3σ(MAD) 밖 점 제외)
std::vector<double> fieldCoeffs(const std::vector<cv::Point2d>& P0, const Track& t, const FieldSpec& s,
                                std::vector<double>* resX = nullptr);   // resX : dx 잔차 (무효 점은 NaN)

struct Estimate { double y_mm = 0, t_deg = 0; bool valid = false; };

class FieldModel {
public:
    // 학습 : K (영상별 계수), 라벨 (Y, Θ)
    bool fit(const std::vector<std::vector<double>>& K, const std::vector<double>& y, const std::vector<double>& t, bool quad);
    Estimate estimate(const std::vector<double>& k) const;
    // 다른 날 영상 : 같은 날 기준 영상 계수 kRef 를 빼고 학습 원점(c)을 더해 기준을 맞춤
    Estimate estimateRef(const std::vector<double>& k, const std::vector<double>& kRef) const;
    int nCoeff() const { return M_.cols; }
    const cv::Mat& M() const { return M_; }        // (3 또는 4) × nCoeff : 행 = Y, Θ, (Θ²), 1
    const cv::Mat& cov() const { return C_; }
    bool save(const std::string& path, const FieldSpec& s, const std::string& note) const;
private:
    bool quad_ = true;
    cv::Mat M_, C_, W_;
};

} // namespace h3

#endif
