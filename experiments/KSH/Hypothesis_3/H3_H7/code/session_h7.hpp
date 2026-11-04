// =====================================================================================
// session_h7.hpp — 가설 H7 : 세션(촬영 날) 보정
//
//  같은 날 찍은 기지값 영상 몇 장으로 Θ 배율·오프셋과 Y 오프셋·Θ 결합을 다시 맞춘다.
//    Θ_est = g·Θ_true + o           →  Θ_corr = (Θ_est − o) / g
//    Y_est − Y_true = oY + kY·Θ_true →  Y_corr = Y_est − oY − kY·Θ_corr
//  2차 세트 분할 : Θ = 0.01°, 1.00° 자세(24장)로 보정, 나머지 자세(33장)로 평가
// =====================================================================================
#ifndef __H3_SESSION_H7_HPP__
#define __H3_SESSION_H7_HPP__

#include <string>
#include <vector>

namespace h3 {

struct SessionSample { double trueY_mm, trueT_deg, estY_mm, estT_deg; };

struct SessionCal { double gainT = 1, offT = 0, offY = 0, kY = 0; int nCal = 0; bool ok = false; };

struct EvalStat {
    int n = 0, passT = 0, passY = 0, passBoth = 0;
    double rmsT = 0, maxT = 0, meanT = 0, rmsY = 0, meanY = 0;
};

EvalStat evalStat(const std::vector<SessionSample>& s, double tolT_deg = 0.01, double tolY_mm = 0.09);
bool     isCalibrationPose(double trueT_deg);                       // Θ = 0.01° 또는 1.00°
SessionCal fitSession(const std::vector<SessionSample>& s);          // 보정용 표본만 넘김
std::vector<SessionSample> applySession(const SessionCal& c, const std::vector<SessionSample>& s);

struct H7Result { std::string method; SessionCal cal; EvalStat before, after; };
// 표본 전체를 받아 보정/평가로 나눠 실행
H7Result runH7(const std::string& method, const std::vector<SessionSample>& all);

// 다른 프로그램이 저장한 결과 CSV (file, …, trueY_mm, estY_mm, …, trueT_deg, estT_deg 열) 에서 2차 세트 표본 읽기
// 2차 세트 = 이름에 촬영 번호 "(n)" 이 있는 행 (center 제외)
std::vector<SessionSample> readSet2Csv(const std::string& path, bool* found = nullptr);

} // namespace h3

#endif
