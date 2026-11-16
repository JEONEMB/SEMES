// 사진별 단계 기록과 저장. 재측정 없이 실제 호출의 알림만 기록
#include "photo_log.hpp"
#include <fstream>
#include <iomanip>
#include <stdexcept>

namespace tools {
// callback: 측정 단계 알림 수집
//   입력: 없음
//   출력: 이 객체에 기록하는 진행 함수
alignment::ProgressCallback PhotoLog::callback() {
    return [this](int stage, bool started, double ms, const std::string& info) {
        if (stage < 1 || stage > 4) return;
        if (started) started_[stage] = true;
        else completed_[stage] = true;
        text << "\n[" << stage << "] " << (started ? "시작" : "처리 반환") << "\n" << info << "\n";
        if (!started) text << "  단계 시간: " << std::fixed << std::setprecision(3) << ms << " ms\n";
    };
}
// finish: 실패 지점과 최종 결과 정리
//   입력: result, read_ms, measure_ms
//   출력: 없음 (버퍼에 기록)
void PhotoLog::finish(const alignment::Result& result, double read_ms, double measure_ms) {
    for (int stage = 1; stage <= 4; ++stage) {
        if (!started_[stage]) text << "[" << stage << "] 미실행 (앞선 입력 또는 측정 단계에서 중단)\n";
        else if (!completed_[stage]) text << "[" << stage << "] 처리 중 중단 (아래 실패 사유 확인)\n";
    }
    text << "\n[5] 최종 결과 및 정답 비교\n"
         << "  상태: " << (result.status == alignment::Status::Failed ? "실패" : result.status == alignment::Status::Warning ? "경고" : "정상")
         << "\n  코드: " << result.code << "\n  알림: " << result.message << "\n";
    text << "  보정 허용: " << (result.correction_allowed ? "예" : "아니오") << "; 조치: " << int(result.action) << " (0 보정 가능, 1 확인, 2 재촬영, 3 정지)\n";
    for (const auto& notice : result.notices) text << "  알림 코드: " << notice.code << ": " << notice.message << "\n";
    if (result.status == alignment::Status::Failed) text << "  측정 중단. 예측값 사용 금지, 정답 대비 IN/OUT 평가 불가.\n";
    else text << "  예측 Y: " << std::setprecision(8) << result.y_mm << " mm / 예측 T: " << result.t_deg << " 도\n";
    text << "  사진 읽기: " << read_ms << " ms / 측정 호출: " << measure_ms << " ms\n"
         << "  측정 호출에는 로그 버퍼 작성 시간이 포함됨. TXT 저장 시간과 사진 읽기는 제외.\n"
         << "  단계 시간은 해당 단계의 알림 출력 전까지이며 하위 단계별 시간은 별도로 측정하지 않음.\n";
}
// save: UTF-8 BOM을 붙여 사진별 기록 저장
//   입력: path (출력 파일)
//   출력: 없음. 쓰기 실패 시 예외
void PhotoLog::save(const std::filesystem::path& path) const {
    std::filesystem::create_directories(path.parent_path());
    std::ofstream file(path, std::ios::binary);
    file << "\xEF\xBB\xBF" << text.str();
    file.flush();
    if (!file) throw std::runtime_error("사진 로그를 저장하지 못했습니다: " + path.u8string());
}
}
