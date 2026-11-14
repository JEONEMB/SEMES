#pragma once
// 학습 자료: prepare_dataset의 출력 표
//   photos.tsv: 사진 경로와 명령값
//   features.tsv: QR과 나사 이동량
//   labels.tsv: 학습 정답(명령 - 회차별 영점 차이)
//   splits.tsv: 사용 여부와 검증 묶음
//   sessions.tsv: 회차별 영점 차이
#include "coefficient_fit.hpp"
#include <filesystem>
#include <string>
#include <vector>

namespace tools {

const int kRepeatCount = 10;   // 학습용과 시험용 사진을 바꾸어 나누는 횟수
const int kFoldCount = 5;      // 묶음 수

struct DatasetPhoto {
    int id = 0;
    std::string session, path;
    double command_y_mm = 0, command_t_deg = 0;
    bool measured = false;
    Values values{};                 // 검출하지 못한 나사는 nan
    double label_y_mm = 0, label_t_deg = 0;
    bool usable = false;             // 학습 사용 여부. 측정, 나사 4개, 범위, 명령 일치 조건
    bool final_check = false;        // 최종 검증용 분리 사진 여부
    int fold[kRepeatCount] = {};     // 반복마다 묶음 번호 (마지막 확인 사진은 -1)

    // single_axis: 한 축만 움직인 명령 확인
    //   입력: 없음 (사진의 명령값 사용)
    //   출력: Y 또는 T가 0이면 true
    bool single_axis() const { return command_y_mm == 0 || command_t_deg == 0; }
};

struct SessionZero {
    std::string session;
    double zero_y_mm = 0, zero_t_deg = 0;
    int photos = 0;
};

// load_tables: 학습용 표 읽기
//   입력: folder (tables 폴더)
//   출력: 성공 여부, photos (사진 자료), sessions (회차별 영점), error (실패 사유)
bool load_tables(const std::filesystem::path& folder, std::vector<DatasetPhoto>& photos, std::vector<SessionZero>& sessions, std::string& error);

}   // tools
