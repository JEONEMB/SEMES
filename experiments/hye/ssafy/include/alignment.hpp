#pragma once
// QR 판 정렬 측정: Y 이동량(mm), T 회전량(도)
//
// 영점: 이동량을 0으로 볼 기준. px: 사진의 픽셀 단위
// NaN: 측정값 없음. true와 false: 예와 아니오
// 사용 순서
//   1. initialize("data/model")          시작할 때 한 번
//   2. set_zero_from_photos(영점 사진)   명령 (0, 0)에서 영점 설정
//   3. measure_alignment(사진)          사진마다 측정
#include <opencv2/core.hpp>
#include <string>
#include <vector>
#include <functional>

namespace alignment {

enum class Status {
    Ok,        // 정상
    Warning,   // 값 사용 전 message 확인
    Failed     // 실패. 측정값 사용 불가
};

enum class Action { Correct, Review, Recapture, Stop };
struct Notice { Status status; std::string code; std::string message; Action action; };

struct Result {
    std::vector<Notice> notices;
    bool correction_allowed = false; // 학습 범위 경고만 있으면 true 가능. 다른 차단 조건은 유지
    Action action = Action::Stop;
    double y_mm = 0;     // Y 이동량(mm)
    double t_deg = 0;    // T 회전량(도)
    Status status = Status::Ok;
    std::string code;    // 가장 제한적인 조치의 대표 코드. 알림이 없으면 빈 문자열
    std::string message; // 처리 결과와 주의사항. Ok에도 알림이 있을 수 있음

    // 단계별 확인용
    double qr_shift_px = 0;          // 1단계: 검출한 QR의 평균 가로 이동(px)
    double screw_shift_px[4] = {};   // 2단계: 나사별 가로 이동(px). 미검출은 NaN
    std::string screws_used;         // 3단계: 사용한 나사 번호. 예: "1234", "134", "none"
    double check_value = 0;          // 4단계: 값 일치 점검값. 한계 초과 시 경고 또는 실패
    double check_limit = 0;          // 점검 한계. 점검식이 없으면 두 점검값 모두 NaN
};

// initialize: 측정 준비와 영점 값 초기화
//   입력: data_folder (기준 사진, reference.json, coefficients/, checks/), settings_file (생략 시 기본 설정)
//   출력: Result (성공: Ok, 실패: Failed와 오류 알림)
Result initialize(const std::string& data_folder, const std::string& settings_file = "");

// measure_alignment: 사진 한 장의 Y, T 측정
//   입력: photo (3840x2160, 밝기 0~255의 회색 또는 컬러 BGR, BGRA). initialize와 영점 설정 성공 후 사용
//   출력: Result (현재 영점 기준 Y, T, 점검값, 알림)
Result measure_alignment(const cv::Mat& photo);

// ProgressCallback: 측정 단계의 시작과 완료 알림
//   입력: 단계(1~4), 시작 여부, 단계 시간(ms, 시작 시 0), 추출값과 다음 용도
//   출력: 없음. 측정 중 동기 호출되므로 콜백에서 alignment 함수를 다시 호출하지 않는다
using ProgressCallback = std::function<void(int, bool, double, const std::string&)>;

// measure_alignment: 진행 알림을 받으며 사진 한 장 측정
//   입력: photo (기존 측정 함수와 동일), progress (진행 알림, 예외를 던지지 않아야 함)
//   출력: Result. 실패한 단계 이후의 단계는 실행하지 않음
Result measure_alignment(const cv::Mat& photo, const ProgressCallback& progress);

// set_zero_from_photos: 명령 (0, 0) 사진으로 영점 설정
//   입력: zero_photos (같은 자세의 사진 목록, 여러 장이면 같은 위치의 픽셀 밝기 평균)
//   출력: Result (새 영점의 Y, T, 알림). Failed이면 영점 무효화, 학습 범위 초과만 있으면 Warning으로 설정 성공
Result set_zero_from_photos(const std::vector<cv::Mat>& zero_photos);

// set_zero_value: 저장한 영점 복원
//   입력: y_mm, t_deg (같은 기준의 영점, mm와 도)
//   출력: Result. 유한값만 허용, 실패 시 영점 무효화
Result set_zero_value(double y_mm, double t_deg);

// clear_zero: 영점 값 초기화
//   입력: 없음
//   출력: 없음 (영점만 (0, 0), 기준 사진 유지, 점검 기준 삭제, 영점 무효화)
void clear_zero();

// set_reference_from_photos: 센터 사진으로 기준 재등록
//   입력: center_photos (동일 자세의 센터 사진 목록)
//   출력: Result (새 기준의 Y, T). 성공 시 W_ZERO_RESET. 재등록 시도는 운영용 영점을 무효화하므로 다시 설정 필요
Result set_reference_from_photos(const std::vector<cv::Mat>& center_photos);

// read_photo: 사진 파일 읽기
//   입력: path (UTF-8 경로, 한글 지원)
//   출력: 회색 Mat. 읽기 실패 시 빈 Mat, 영상 해석 중 예외 가능
cv::Mat read_photo(const std::string& path);

// 진단 전용. 영점 없이도 측정하지만 보정은 허용하지 않는다.
Result measure_diagnostic(const cv::Mat& photo, const ProgressCallback& progress = {});

}   // namespace alignment
