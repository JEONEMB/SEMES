#pragma once
#include <filesystem>
#include <vector>

namespace tools {
// pick_photos: 사진 또는 폴더 선택 창 표시
//   입력: zero (영점 사진 선택 여부)
//   출력: 선택한 경로들. 취소하면 빈 목록
std::vector<std::filesystem::path> pick_photos(bool zero);

// pick_answers: 파일명 정답 또는 정답표 선택
//   입력: 없음
//   출력: 정답표 경로. 파일명 사용 시 빈 경로, 취소 시 예외
std::filesystem::path pick_answers();
}
