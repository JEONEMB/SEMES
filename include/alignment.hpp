#pragma once
// QR 판 정렬 측정: 사진 한 장에서 판이 기준 위치에서 얼마나 벗어났는지 (Y mm, T 도) 를 잰다.
//
// 쓰는 순서
//   1. alignment::initialize("data 폴더")            한 번
//   2. alignment::set_zero_from_photos(센터 사진들)   날마다, 카메라·조명·판을 건드린 뒤 (권장)
//   3. alignment::measure_alignment(사진)            사진마다
//
// 어떤 함수도 예외를 던지지 않는다. 결과의 status 와 message 로 알린다.
#include <opencv2/core.hpp>
#include <string>
#include <vector>

namespace alignment {

enum class Status {
    Ok,        // 정상
    Warning,   // 값은 냈지만 주의가 필요하다 (message 를 본다)
    Failed     // 값을 내지 못했다 (y_mm, t_deg 는 NaN)
};

struct Result {
    double y_mm = 0;     // 기준 위치에서 벗어난 양. 보정 이동 = −y_mm
    double t_deg = 0;    // 기준 각도에서 벗어난 양. 보정 회전 = −t_deg
    Status status = Status::Ok;
    std::string code;    // 가장 중요한 알림의 코드 (예: "W_SCREW_MISSING"). 정상이면 빈 문자열
    std::string message; // 사람이 읽는 설명과 해야 할 일. 정상이면 빈 문자열

    // 아래는 확인용 (옮긴 코드가 맞는지 단계별로 비교할 때 쓴다)
    double qr_shift_px = 0;          // 1단계: QR 3개의 가로 이동 평균
    double screw_shift_px[4] = {};   // 2단계: 나사마다 가로 이동 (못 찾으면 NaN)
    std::string screws_used;         // 3단계: 계산에 쓴 나사 번호 (예: "1234", "134", "none")
    double check_value = 0;          // 4단계: 값 일치 점검값 (기준값보다 크면 경고)
    double check_limit = 0;          //        그 기준값 (점검식이 없는 조합이면 둘 다 NaN)
};

// data_folder: center.bmp, reference.json, coefficients/, checks/ 가 있는 폴더
// settings_file: 비우면 기본 설정. config/settings.json 을 고쳐서 넘기면 그 설정을 쓴다
Result initialize(const std::string& data_folder, const std::string& settings_file = "");

// photo: 3840×2160, 8비트, 회색 또는 컬러(BGR, BGRA)
Result measure_alignment(const cv::Mat& photo);

// 영점 잡기: 이 사진들의 자세를 (0, 0) 으로 삼는다. 여러 장이면 화소 평균을 쓴다(5~10장 권장).
// 돌려주는 y_mm, t_deg = 새 영점이 처음 영점(기준 사진)에서 떨어진 양. 적어 두면 set_zero_value 로 되살릴 수 있다.
Result set_zero_from_photos(const std::vector<cv::Mat>& center_photos);
void set_zero_value(double y_mm, double t_deg);   // 적어 둔 영점 값을 다시 넣는다
void clear_zero();                                // 처음 영점(기준 사진과 같은 화면 = 0, 0)으로 되돌린다

// 기준 사진 다시 잡기: 화면이 많이 밀려 QR·나사를 못 찾을 때만 쓴다.
// 이 사진들에서 QR 과 나사 위치를 다시 찾아 기준으로 삼는다. 영점은 그대로 이어진다.
Result set_reference_from_photos(const std::vector<cv::Mat>& center_photos);

}   // namespace alignment
