// 숫자 없음(NaN). <cmath> 의 NAN 매크로는 windows.h 와 함께 쓸 때 깨질 수 있어서 따로 둔다.
#pragma once
#include <limits>
namespace hw {
inline constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
}
