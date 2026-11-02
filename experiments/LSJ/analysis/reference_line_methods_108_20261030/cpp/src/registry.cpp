// Method registry: 9 external (E01-E09) and 12 QR (Q01-Q12) methods.
#include "common.hpp"
#include "methods.hpp"

namespace lm {

const std::vector<MethodInfo>& registry() {
    static const std::vector<MethodInfo> r = {
        {"E01", "E", u8"밝기 프로파일 모델 피팅(Gaussian 미분 프로파일)", "methods_points.cpp::method_profile_gaussian", method_profile_gaussian},
        {"E02", "E", u8"Canny–Devernay 서브픽셀 경계점", "methods_points.cpp::method_canny_devernay", method_canny_devernay},
        {"E03", "E", u8"Canny → 허프 변환 → 재피팅", "methods_hough.cpp::method_canny_hough", method_canny_hough},
        {"E04", "E", u8"LSD", "methods_segments.cpp::method_lsd", method_lsd},
        {"E05", "E", u8"Zernike 모멘트(ramp edge)", "methods_points.cpp::method_zernike", method_zernike},
        {"E06", "E", u8"PAE(부분 면적 효과)", "methods_points.cpp::method_pae", method_pae},
        {"E07", "E", u8"EDLines", "methods_segments.cpp::method_edlines", method_edlines},
        {"E08", "E", u8"FLD(Fast Line Detector)", "methods_segments.cpp::method_fld", method_fld},
        {"E09", "E", u8"ELSED", "methods_segments.cpp::method_elsed", method_elsed},
        {"Q01", "Q", u8"밝기 프로파일 모델 피팅(Gaussian 미분 프로파일)", "methods_points.cpp::method_profile_gaussian", method_profile_gaussian},
        {"Q02", "Q", u8"Canny–Devernay 서브픽셀 경계점", "methods_points.cpp::method_canny_devernay", method_canny_devernay},
        {"Q03", "Q", u8"Canny → 허프 변환 → 재피팅", "methods_hough.cpp::method_canny_hough", method_canny_hough},
        {"Q04", "Q", u8"LSD", "methods_segments.cpp::method_lsd", method_lsd},
        {"Q05", "Q", u8"Zernike 모멘트(ramp edge)", "methods_points.cpp::method_zernike", method_zernike},
        {"Q06", "Q", u8"PAE(부분 면적 효과)", "methods_points.cpp::method_pae", method_pae},
        {"Q07", "Q", u8"EDLines", "methods_segments.cpp::method_edlines", method_edlines},
        {"Q08", "Q", u8"FLD(Fast Line Detector)", "methods_segments.cpp::method_fld", method_fld},
        {"Q09", "Q", u8"ELSED", "methods_segments.cpp::method_elsed", method_elsed},
        {"Q10", "Q", u8"이진화 → 윤곽선 → 상단 변 윤곽점 피팅", "methods_qr.cpp::method_qr_contour", method_qr_contour},
        {"Q11", "Q", u8"사각 테두리 코너 검출 → cornerSubPix → 상단 두 코너 연결", "methods_qr.cpp::method_qr_corners", method_qr_corners},
        {"Q12", "Q", u8"파인더 검출 → 코너 정밀화·중심 → 위쪽 두 중심 연결", "methods_qr.cpp::method_qr_finders", method_qr_finders},
    };
    return r;
}

}  // namespace lm
