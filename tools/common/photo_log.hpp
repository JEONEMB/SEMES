#pragma once
// 사진 한 장의 실제 측정 진행과 결과를 UTF-8 TXT로 저장
#include "alignment.hpp"
#include <filesystem>
#include <sstream>

namespace tools {
class PhotoLog {
public:
    std::ostringstream text;  // 사진 경로, 평가 조건, 정답 비교를 덧붙일 기록
    // callback: 실제 측정 호출에 연결할 진행 기록 함수 생성
    //   입력: 없음
    //   출력: 단계 시작, 완료, 시간을 기록하는 콜백 (이 객체보다 오래 사용하지 않음)
    alignment::ProgressCallback callback();
    // finish: 최종 상태와 실행하지 않은 단계 기록
    //   입력: result (측정 결과), read_ms (읽기 시간), measure_ms (측정 호출 시간)
    //   출력: 없음 (text에 상태, 실패 사유, 누락 단계와 시간 추가)
    void finish(const alignment::Result& result, double read_ms, double measure_ms);
    // save: TXT 기록 저장
    //   입력: path (저장할 경로)
    //   출력: 없음. 저장 실패 시 예외로 호출부에 알림
    void save(const std::filesystem::path& path) const;
private:
    bool started_[5] = {};
    bool completed_[5] = {};
};
}
