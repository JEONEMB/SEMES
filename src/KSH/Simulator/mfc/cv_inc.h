// =====================================================================================
// cv_inc.h — MFC 코드에서 OpenCV · 모델 · 시뮬레이터 헤더를 포함할 때 사용
//   Windows 헤더의 min / max 매크로가 OpenCV 의 std::min / std::max 와 충돌하므로
//   포함하는 동안만 매크로를 잠시 지웠다가 되돌린다. (MFC 코드에서는 (std::min)(a, b) 처럼 괄호로 감쌀 것)
// =====================================================================================
#pragma once

#pragma push_macro("min")
#pragma push_macro("max")
#undef min
#undef max

#include <opencv2/opencv.hpp>
#include "../../Alignment/alignment.hpp"
#include "../core/sim_runner.hpp"
#include "../core/sim_draw.hpp"

#pragma pop_macro("max")
#pragma pop_macro("min")
