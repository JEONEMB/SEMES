#pragma once
// 기준 재등록: 새 센터 사진의 QR 영역과 나사 중심 검출
#include "common/reference.hpp"

namespace alignment {

// find_new_reference: 새 기준 위치 검출
//   입력: old_reference (기존 기준), new_photo (새 회색 사진)
//   출력: 성공 여부, new_reference (새 기준), error (실패 사유)
bool find_new_reference(const Reference& old_reference, const cv::Mat& new_photo, Reference& new_reference, std::string& error);

}   // alignment
