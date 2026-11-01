// =====================================================================================
// doe.hpp — 실험계획(DOE) : 최종 방법의 각 선택을 같은 데이터·같은 평가 기준으로 비교
//
//  평가 기준 (모든 실험 공통, 허용 범위 = Y ±45 µm, Θ ±0.005° : 과제의 절반을 기본으로 사용)
//    · 1차 블록 교차검증 : 1차 첫 번째 측정 사진을 연속 구간 14 묶음으로 나눠 번갈아 시험 (학습·선택에 쓰는 유일한 기준)
//        J = √mean((eY / tolY)² + (eΘ / tolΘ)²) , 허용 안 장 수
//    · 2차 시험 : 1차 전체로 학습한 식을 2차 57장에 적용 (선택에는 쓰지 않고 결과만 보고)
//
//  실험 1 — 정렬 순서 : Coarse only / Fine(ECC) only / Coarse → Fine / Fine → Coarse
//        Coarse Alignment = 위상 상관(FFT)으로 수십 px 이동을 한 번에 찾기, Fine Alignment = ECC 반복으로 소수점 이하 다듬기
//  실험 2 — 보간 방법 : 선형, Bicubic(Keys a = −0.5 / OpenCV a = −0.75), Lanczos3, 3차 B-스플라인, 3차 O-MOMS
//        × 가우시안 사전 필터 σ = 0 / 0.5 / 1.0 / 1.5 px  (+ OpenCV findTransformECC 내장 = 선형 + 5×5 가우시안)
//        Fine Alignment(정밀 정렬)은 평행이동 IC-GN (ZNSSD, Pan 2013) 으로 직접 구현 — 보간만 바꿔 끼움
//  실험 3 — s·d 식의 차수 : Θ 에 대한 1·2·3차 조합 (s, d 각각) + Y² 추가
// =====================================================================================
#ifndef YT_DOE_HPP
#define YT_DOE_HPP

#include "../yt/yt_common.hpp"

namespace yt {
namespace doe {

bool run(const std::vector<Sample>& ds, const std::string& outDir, const std::string& which);

} // namespace doe
} // namespace yt

#endif
