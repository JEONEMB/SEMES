#pragma once
// 기준 다시 잡기: 새 센터 사진에서 QR 상자와 나사 중심을 다시 찾아 새 기준을 만든다.
// 카메라가 처져 화면이 많이 밀렸을 때처럼, 원래 기준으로는 QR·나사를 찾지 못할 때 쓴다.
#include "common/reference.hpp"

namespace alignment {

// old_reference 의 위치에서 출발해 new_photo(회색) 안의 QR·나사 위치를 찾는다. 실패하면 false 와 이유.
bool find_new_reference(const Reference& old_reference, const cv::Mat& new_photo, Reference& new_reference, std::string& error);

}   // namespace alignment
