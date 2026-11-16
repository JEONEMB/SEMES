#pragma once
#include <string>
namespace alignment {
// verify_model_manifest: 기준 자료와 조합별 모델의 해시 확인
//   입력: 모델 폴더
//   출력: 성공 여부와 실패 사유
bool verify_model_manifest(const std::string& folder, std::string& error);
}
