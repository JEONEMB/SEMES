// 사진 한 장의 Y, T와 단계별 값 확인
// 사용: measure_one <model 폴더> <사진> [영점 사진]
#include "alignment.hpp"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

namespace {

// status_name: 측정 상태 이름 표시
//   입력: status (측정 상태)
//   출력: 정상, 경고, 실패 중 해당 문자열
const char* status_name(alignment::Status status) {
    return status == alignment::Status::Ok ? "정상" : status == alignment::Status::Warning ? "경고" : "실패";
}

}   // 파일 내부

// main: 사진 한 장의 Y, T와 단계별 값 확인
//   입력: 실행할 때 지정한 경로와 옵션(파일 상단 사용법)
//   출력: 실행 결과와 성공 여부를 나타내는 종료 코드
int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "사용: measure_one <model 폴더> <사진> [영점으로 쓸 센터 사진]\n";
        return 2;
    }
    const alignment::Result ready = alignment::initialize(argv[1]);
    if (ready.status == alignment::Status::Failed) return 1;

    if (argc >= 4) {
        const alignment::Result zero = alignment::set_zero_from_photos({alignment::read_photo(argv[3])});
        std::cout << "영점: Y " << zero.y_mm << " mm, T " << zero.t_deg << " 도 (기준 사진에서 떨어진 양)\n";
    }

    const alignment::Result result = alignment::measure_alignment(alignment::read_photo(argv[2]));
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "1단계  QR 가로 이동      " << result.qr_shift_px << " px\n";
    std::cout << "2단계  나사 가로 이동    ";
    for (double shift : result.screw_shift_px) std::cout << shift << "  ";
    std::cout << "px\n";
    std::cout << "3단계  쓴 나사 " << result.screws_used << "  →  Y " << result.y_mm << " mm,  T " << result.t_deg << " 도\n";
    std::cout << "4단계  값 일치 점검      " << result.check_value << " (기준 " << result.check_limit << ")\n";
    std::cout << "상태   " << status_name(result.status) << (result.code.empty() ? "" : "  " + result.code) << "\n";
    if (!result.message.empty()) std::cout << "알림   " << result.message << "\n";
    return result.status == alignment::Status::Failed ? 1 : 0;
}
