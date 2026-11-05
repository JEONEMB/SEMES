// =====================================================================================
// cv_inc.h — MFC 코드에서 OpenCV·측정 엔진 헤더를 포함할 때 사용
//   Windows 헤더의 min / max 매크로가 OpenCV 의 std::min / std::max 와 충돌하므로
//   포함하는 동안만 매크로를 잠시 지웠다가 되돌린다.
//   (MFC 코드에서 std::min / std::max 를 쓸 때는 (std::min)(a, b) 처럼 괄호로 감쌀 것)
// =====================================================================================
#pragma once

#pragma push_macro("min")
#pragma push_macro("max")
#undef min
#undef max

#include <opencv2/opencv.hpp>
#include "core/phase_align.hpp"

#pragma pop_macro("max")
#pragma pop_macro("min")
