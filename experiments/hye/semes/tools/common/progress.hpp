#pragma once
// 처리 건수, 진행률, 남은 시간 표시
// 사용: ProgressBar 생성 -> 항목마다 step() -> 완료 시 finish()
#include <chrono>
#include <iostream>
#include <string>

namespace tools {

class ProgressBar {
public:
    // ProgressBar: 진행 표시 시작
    //   입력: label (작업명), total (전체 건수)
    //   출력: 진행 상황을 표시할 도구
    ProgressBar(const std::string& label, int total);

    // step: 처리 건수 증가
    //   입력: 없음
    //   출력: 없음 (진행 표시 갱신)
    void step();

    // finish: 진행 표시 완료
    //   입력: 없음
    //   출력: 없음 (경과 시간 표시와 줄바꿈)
    void finish();

private:
    // draw: 진행 표시 출력
    //   입력: done (완료 여부)
    //   출력: 없음 (화면 또는 파일 출력)
    void draw(bool done);

    std::string label_;
    int total_ = 0, count_ = 0;
    int last_tenth_ = 0;                                    // 파일 출력용 10% 구간 번호
    bool on_screen_ = false;                                // 화면 출력 여부. 화면이면 같은 줄 갱신
    std::chrono::steady_clock::time_point start_, last_draw_;
};

// 이 객체가 있는 동안 화면의 오류 알림 숨김
// Result의 code와 message는 유지
class QuietWarnings {
public:
    QuietWarnings();
    ~QuietWarnings();

private:
    std::streambuf* saved_ = nullptr;
};

}   // tools
