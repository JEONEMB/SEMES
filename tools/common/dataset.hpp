#pragma once
// dataset/prepared/ 의 표를 읽어 사진 한 장씩 묶는다 (prepare_dataset 이 만든 것).
//
//   photos.tsv    id, session, path, command_y_mm, command_t_deg
//   features.tsv  id, measured, qr, screw1, screw2, screw3, screw4   (못 찾은 나사는 nan)
//   labels.tsv    id, label_y_mm, label_t_deg                      (학습용 정답 = 명령 − 그 회차의 영점 어긋남)
//   splits.tsv    id, usable, final_check, cv0 … cv9               (usable 0 이면 학습, 평가에 안 씀)
//   sessions.tsv  session, zero_y_mm, zero_t_deg, photos            (회차마다 거꾸로 구한 영점 어긋남)
#include "pose_model.hpp"
#include <filesystem>
#include <string>
#include <vector>

namespace tools {

const int kRepeatCount = 10;   // 섞어 나누기 반복 횟수
const int kFoldCount = 5;      // 묶음 수

struct DatasetPhoto {
    int id = 0;
    std::string session, path;
    double command_y_mm = 0, command_t_deg = 0;
    bool measured = false;
    Values values{};                 // 못 찾은 나사는 nan
    double label_y_mm = 0, label_t_deg = 0;
    bool usable = false;             // 학습 범위 안, 측정 성공, 나사 4개 모두 찾음
    bool final_check = false;        // 마지막 확인용 (방법을 다 정한 뒤 한 번만 시험)
    int fold[kRepeatCount] = {};     // 반복마다 묶음 번호 (마지막 확인 사진은 -1)

    bool single_axis() const { return command_y_mm == 0 || command_t_deg == 0; }
};

struct SessionZero {
    std::string session;
    double zero_y_mm = 0, zero_t_deg = 0;
    int photos = 0;
};

// prepared 폴더를 읽는다. 실패하면 false 와 이유.
bool load_prepared(const std::filesystem::path& folder, std::vector<DatasetPhoto>& photos, std::vector<SessionZero>& sessions, std::string& error);

}   // namespace tools
