// =====================================================================================
// yt_verify.hpp — 방법별 검증 실행기
//   QR_Image.exe verify [all|m1|m2|m4|m3|trackB|trackA|roi|media|doe] [--img 폴더] [--out 폴더] [--force]
//   순서 : M1 → M2 → M4 → M3 → 트랙 B → 트랙 A → H5 최적 ROI → 보고서 영상 프레임 → 실험계획(DOE). 앞 단계 구현 검증이 미통과면 멈춤 (--force 로 계속)
// =====================================================================================
#ifndef YT_VERIFY_HPP
#define YT_VERIFY_HPP

namespace yt {
int verifyMain(int argc, char** argv);
} // namespace yt

#endif
