// 사진 한 장의 Y, T와 단계별 값 확인
// 사용: measure_one <model 폴더> <사진> [영점 사진]
// 콘솔에는 요약, 결과 폴더에는 사진별 상세 TXT 저장
#include "alignment.hpp"
#include "common/console.hpp"
#include "common/photo_files.hpp"
#include "common/coefficient_fit.hpp"
#include "common/photo_log.hpp"
#include "common/progress.hpp"
#include <ctime>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>

namespace {

// status_name: 측정 상태 이름 표시
//   입력: status (측정 상태)
//   출력: 정상, 경고, 실패 중 해당 문자열
const char* status_name(alignment::Status status) {
    return status == alignment::Status::Ok ? "정상" : status == alignment::Status::Warning ? "경고" : "실패";
}

// print_axis_error: 축별 정답, 예측값과 허용오차 대비 여유 표시
//   입력: axis, truth, predicted, tolerance, unit (축, 정답, 예측, 허용오차, 단위)
//   출력: 없음 (IN은 남은 여유, OUT은 초과량을 양수로 출력)
void print_axis_error(std::ostream& output, const char* axis, double truth, double predicted, double tolerance, const char* unit) {
    const double error = predicted - truth;
    const double margin = tolerance - std::abs(error);
    output << "  " << axis << ": 원래 " << truth << " / 예측 " << predicted
              << " / 오차 " << std::showpos << error << std::noshowpos << " " << unit << "\n"
              << "     " << (margin >= 0 ? "IN: 여유 " : "OUT: 초과 ") << std::abs(margin) << " " << unit
              << " (허용오차 " << tolerance << " " << unit << ")\n";
}

// print_evaluation: 파일명 명령값과 측정값을 사후 비교
//   입력: path (사진 경로), result (정답을 사용하지 않고 계산한 측정 결과)
//   출력: 없음 (정답, 예측, 오차와 경고 유무의 교차 판정)
void print_evaluation(std::ostream& output, const std::string& path, const alignment::Result& result) {
    double truth_y = 0, truth_t = 0;
    const std::string name = std::filesystem::path(path).filename().string();
    if (!tools::parse_command_from_file_name(name, truth_y, truth_t)) {
        output << "정답 비교: 평가 불가 (파일명에 Y/T 명령값 없음)\n";
        return;
    }
    output << "\n정답 비교 (원래 라벨 = 파일명 명령값, 오차 = 예측 - 원래)\n";
    print_axis_error(output, "Y", truth_y, result.y_mm, tools::kToleranceYmm, "mm");
    print_axis_error(output, "T", truth_t, result.t_deg, tools::kToleranceTdeg, "도");
    const bool inside = tools::passes(result.y_mm - truth_y, result.t_deg - truth_t);
    const bool warned = result.status != alignment::Status::Ok;
    output << "  종합: " << (inside ? "IN" : "OUT") << "\n"
              << "  경고와 실제 오차: " << (warned ? (inside ? "경고 있음 / 허용오차 이내" : "경고 있음 / 허용오차 초과")
                                                         : (inside ? "경고 없음 / 허용오차 이내" : "경고 없음 / 허용오차 초과 (미검출)")) << "\n";
    output << "  경고는 위험 신호이며, 실제 오차 크기를 미리 예측했다는 뜻은 아닙니다.\n";
}

}   // 파일 내부

// main: 사진 한 장의 Y, T와 단계별 값 확인
//   입력: 실행할 때 지정한 경로와 옵션(파일 상단 사용법)
//   출력: 실행 결과와 성공 여부를 나타내는 종료 코드
int run_measurement(int argc, char** argv) {
    tools::use_korean_console();
    int path_count = argc;
    if (argc > 3) {
        const std::string mode = argv[argc - 1];
        if (mode == "simple" || mode == "detail") {
            --path_count;
        }
    }
    if (path_count < 3 || path_count > 4) {
        std::cerr << "사용: measure_one <model 폴더> <사진> [영점 사진]\n"
                  << "  콘솔 요약 + 사진별 상세 TXT 자동 저장\n";
        return 2;
    }
    using Clock = std::chrono::steady_clock;
    tools::PhotoLog log;
    const std::time_t now = std::time(nullptr);
    char stamp[32];
    std::strftime(stamp, sizeof(stamp), "%Y%m%d_%H%M%S", std::localtime(&now));
    const auto base = std::filesystem::path("data/results/measure") / stamp;
    auto folder = base;
    for (int n = 2; std::filesystem::exists(folder); ++n) folder = base.string() + "_" + std::to_string(n);
    std::filesystem::create_directories(folder);
    log.text << "사진: " << argv[2] << "\n모델: " << argv[1] << "\n";
    const auto saved = [&] {
        log.save(folder / "measurement.txt");
        std::cout << "\n[3] 상세 기록: measurement.txt\n  explorer.exe \""
                  << std::filesystem::absolute(folder).u8string() << "\"\n";
    };
    const tools::QuietWarnings quiet;
    const auto init_start = Clock::now();
    const alignment::Result ready = alignment::initialize(argv[1]);
    const double init_ms = std::chrono::duration<double, std::milli>(Clock::now() - init_start).count();
    if (ready.status == alignment::Status::Failed) {
        log.text << "[0] 모델 초기화 실패: " << ready.message << "\n";
        log.finish(ready, 0, 0);
        std::cout << "초기화 실패: " << ready.message << "\n";
        saved(); return 1;
    }
    log.text << "[0] 모델 초기화: " << init_ms << " ms\n";

    std::cout << std::fixed << std::setprecision(4);
    if (path_count == 4) {
        const alignment::Result zero = alignment::set_zero_from_photos({alignment::read_photo(argv[3])});
        log.text << "영점 사진: " << argv[3] << "; 영점 Y=" << zero.y_mm << ", T=" << zero.t_deg << "\n";
        if (zero.status == alignment::Status::Failed) {
            std::cout << "영점 설정 실패: " << zero.message << "\n";
            log.finish(zero, 0, 0); saved(); return 1;
        }
        std::cout << "영점 적용: 완료\n";
    } else {
        log.text << "영점: 미설정\n";
        std::cout << "영점 적용: 미설정 (모델의 기본 기준으로 측정)\n";
    }

    const auto read_start = Clock::now();
    cv::Mat photo;
    try { photo = alignment::read_photo(argv[2]); }
    catch (const std::exception& error) { log.text << "사진 읽기 오류: " << error.what() << "\n"; }
    const double read_ms = std::chrono::duration<double, std::milli>(Clock::now() - read_start).count();
    const auto progress = log.callback();
    const auto measure_start = Clock::now();
    const alignment::Result result = (path_count == 4 ? alignment::measure_alignment(photo, progress) : alignment::measure_diagnostic(photo, progress));
    const double measure_ms = std::chrono::duration<double, std::milli>(Clock::now() - measure_start).count();
    log.finish(result, read_ms, measure_ms);
    std::cout << "\n============================================================\n[1] 측정 결과\n  상태: " << status_name(result.status) << (result.code.empty() ? "" : " (" + result.code + ")") << "\n";
    if (!result.message.empty()) std::cout << "  알림: " << result.message << "\n";
    if (result.status != alignment::Status::Failed) {
        std::cout << "  Y 이동량: " << result.y_mm << " mm\n  T 회전량: " << result.t_deg << " 도\n";
        print_evaluation(std::cout, argv[2], result);
        print_evaluation(log.text, argv[2], result);

    } else {
        std::cout << "  정답 비교: 평가 불가 (측정 실패, 결과 사용 금지)\n";
    }
    std::cout << "\n[2] 시간: 사진 읽기 " << read_ms << " ms / 측정 호출(로그 포함) " << measure_ms << " ms\n";
    saved();
    return result.status == alignment::Status::Failed ? 1 : 0;
}

// main: 실행 오류 안내
//   입력: 명령 인자
//   출력: 측정 성공 0, 처리 실패 1, 사용법 오류 2
int main(int argc, char** argv) {
    try { return run_measurement(argc, argv); }
    catch (const std::exception& error) { std::cerr << "실행 실패: " << error.what() << "\n"; return 1; }
}
